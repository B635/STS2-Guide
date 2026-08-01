from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from realtime.compatibility import load_compatibility_manifest
from realtime.diagnostics import (
    MAX_LOG_BYTES,
    MAX_LOG_LINES,
    DiagnosticSnapshot,
    export_diagnostics_zip,
)
from realtime.installation import (
    GAME_ASSEMBLY,
    GAME_DIRECTORY_ENV,
    GAME_EXECUTABLE,
    GUIDE_MOD_ARTIFACTS,
    InstallReceipt,
    InstalledArtifact,
    discover_game_directory,
    preflight_installation,
    read_install_receipt,
    sha256_file,
    write_install_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def _make_game(root: Path) -> Path:
    game = root / "Slay the Spire 2"
    (game / GAME_ASSEMBLY).parent.mkdir(parents=True)
    (game / GAME_EXECUTABLE).write_bytes(b"game-exe")
    (game / GAME_ASSEMBLY).write_bytes(b"sts2-assembly")
    return game


def _receipt(game: Path) -> InstallReceipt:
    return InstallReceipt(
        guide_version="0.3.0-alpha.0",
        release_fingerprint="a" * 64,
        guide_executable_sha256="9" * 64,
        game_directory=game.resolve(),
        artifacts=(),
    )


class InstallationDiscoveryTests(unittest.TestCase):
    def test_discovery_priority_is_env_receipt_steam_then_manual(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            env_game = _make_game(root / "env")
            receipt_game = _make_game(root / "receipt")
            steam_root = root / "steam"
            steam_library = root / "library"
            steam_game = _make_game(
                steam_library / "steamapps" / "common"
            )
            manual_game = _make_game(root / "manual")
            (steam_root / "steamapps").mkdir(parents=True)
            escaped_library = str(steam_library).replace("\\", "\\\\")
            (steam_root / "steamapps" / "libraryfolders.vdf").write_text(
                '"libraryfolders"\n{\n'
                f'  "1" {{ "path" "ignored-inline" }}\n'
                f'  "path" "{escaped_library}"\n'
                '}\n',
                encoding="utf-8",
            )
            receipt_path = root / "install-receipt.json"
            write_install_receipt(receipt_path, _receipt(receipt_game))

            result = discover_game_directory(
                environ={GAME_DIRECTORY_ENV: str(env_game)},
                receipt_path=receipt_path,
                registry_steam_roots=[steam_root],
                manual_directory=manual_game,
            )
            self.assertEqual(result.source, "environment")
            self.assertEqual(result.game_directory, env_game.resolve())

            result = discover_game_directory(
                environ={},
                receipt_path=receipt_path,
                registry_steam_roots=[steam_root],
                manual_directory=manual_game,
            )
            self.assertEqual(result.source, "receipt")

            result = discover_game_directory(
                environ={},
                registry_steam_roots=[steam_root],
                manual_directory=manual_game,
            )
            self.assertEqual(result.source, "steam")
            self.assertEqual(result.game_directory, steam_game.resolve())

            result = discover_game_directory(
                environ={},
                registry_steam_roots=[],
                manual_directory=manual_game,
            )
            self.assertEqual(result.source, "manual")

    def test_discovery_failure_uses_stable_reason_without_path(self):
        with tempfile.TemporaryDirectory() as tempdir:
            missing = Path(tempdir) / "Alice" / "secret-game"
            result = discover_game_directory(
                environ={GAME_DIRECTORY_ENV: str(missing)},
                registry_steam_roots=[],
            )
            self.assertFalse(result.found)
            self.assertEqual(result.reason_codes, ("game_directory_invalid",))
            self.assertNotIn(str(missing), repr(result.reason_codes))


class InstallReceiptTests(unittest.TestCase):
    def test_receipt_round_trip_is_strict_and_atomic(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            game = _make_game(root)
            receipt = InstallReceipt(
                guide_version="0.3.0-alpha.0",
                release_fingerprint="b" * 64,
                guide_executable_sha256="8" * 64,
                game_directory=game.resolve(),
                artifacts=(
                    InstalledArtifact("mods/guide.dll", "c" * 64),
                ),
            )
            path = root / "config" / "install-receipt.json"
            write_install_receipt(path, receipt)
            self.assertEqual(read_install_receipt(path), receipt)
            self.assertFalse(path.with_name(f"{path.name}.tmp").exists())

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["unexpected"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(read_install_receipt(path))

    def test_receipt_rejects_absolute_or_traversing_artifacts(self):
        for invalid in ("../other-mod.dll", "C:/private/mod.dll", "/tmp/mod"):
            with self.subTest(path=invalid):
                with self.assertRaises(ValueError):
                    InstalledArtifact(invalid, "d" * 64)


class InstallationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = load_compatibility_manifest(
            ROOT / "packaging" / "compatibility.json"
        )

    def _prepared_installation(self, root: Path):
        game = _make_game(root)
        assembly_hash = sha256_file(game / GAME_ASSEMBLY)
        manifest = replace(
            self.manifest,
            game=replace(
                self.manifest.game,
                version="0.110.1",
                sts2_dll_sha256=assembly_hash,
            ),
        )
        mods = game / "mods"
        mods.mkdir()
        expected: dict[str, str] = {}
        for name in GUIDE_MOD_ARTIFACTS:
            path = mods / name
            path.write_bytes(f"guide:{name}".encode("utf-8"))
            expected[name] = sha256_file(path)
        return game, manifest, expected

    def test_exact_clean_installation_passes_and_reports_only_hashes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            game, manifest, expected = self._prepared_installation(
                Path(tempdir)
            )
            result = preflight_installation(
                game,
                manifest,
                expected,
                version_reader=lambda _: "0.110.1",
            )
            self.assertTrue(result.compatible)
            self.assertEqual(result.reason_codes, ())
            self.assertEqual(result.third_party_mod_count, 0)
            self.assertEqual(
                set(result.artifact_hashes),
                {"game_executable", "game_assembly", *GUIDE_MOD_ARTIFACTS},
            )
            self.assertNotIn(str(game), repr(result.issues))

    def test_mismatches_and_third_party_mods_fail_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            game, manifest, expected = self._prepared_installation(
                Path(tempdir)
            )
            (game / GAME_ASSEMBLY).write_bytes(b"wrong-assembly")
            (game / "mods" / GUIDE_MOD_ARTIFACTS[0]).write_bytes(b"wrong-mod")
            (game / "mods" / "OtherMod.dll").write_bytes(b"third-party")
            result = preflight_installation(
                game,
                manifest,
                expected,
                version_reader=lambda _: "0.999.0",
            )
            self.assertFalse(result.compatible)
            self.assertEqual(
                set(result.reason_codes),
                {
                    "game_version_mismatch",
                    "game_assembly_hash_mismatch",
                    "guide_artifact_hash_mismatch",
                    "third_party_mods_detected",
                },
            )
            self.assertEqual(result.third_party_mod_count, 1)
            self.assertNotIn(str(game), repr(result.issues))

    def test_missing_files_and_invalid_expectations_have_stable_codes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            missing = preflight_installation(
                root / "missing-user-path",
                self.manifest,
                {},
            )
            self.assertEqual(missing.reason_codes, ("game_directory_missing",))
            self.assertNotIn(str(root), repr(missing.issues))

            game = root / "empty-game"
            game.mkdir()
            result = preflight_installation(game, self.manifest, {})
            self.assertIn("game_executable_missing", result.reason_codes)
            self.assertIn("game_assembly_missing", result.reason_codes)
            self.assertIn(
                "guide_artifact_expectation_invalid",
                result.reason_codes,
            )
            self.assertIn("guide_artifact_missing", result.reason_codes)


class DiagnosticArchiveTests(unittest.TestCase):
    def _snapshot(self) -> DiagnosticSnapshot:
        return DiagnosticSnapshot(
            versions={"guide": "0.3.0-alpha.0", "game": "0.110.1"},
            release_fingerprint="e" * 64,
            capabilities={"card_reward": "enabled", "route_choice": "enabled"},
            status="incompatible",
            reason_codes=("third_party_mods_detected",),
            artifact_hashes={"game_assembly": "f" * 64},
        )

    def test_zip_has_strict_whitelist_and_redacts_game_and_secret_data(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "diagnostics.zip"
            username = "InjectedUser"
            secrets = (
                username,
                "SECRET_CARD",
                "SECRET_RELIC",
                "SECRET_POTION",
                "RUN-SECRET",
                "EVENT-SECRET",
                "DECISION-SECRET",
                "sk-super-secret-api-key",
            )
            logs = [
                rf"user={username} path=C:\Users\{username}\save\active-run.json",
                "deck=SECRET_CARD relic=SECRET_RELIC potion=SECRET_POTION",
                "run_id=RUN-SECRET event_id=EVENT-SECRET "
                "decision_id=DECISION-SECRET",
                "API_KEY=sk-super-secret-api-key status=failed",
                "/home/InjectedUser/private/state-event.json",
            ]
            export_diagnostics_zip(
                output,
                self._snapshot(),
                logs,
                sensitive_values=secrets,
            )
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"diagnostic.json", "guide.log"},
                )
                metadata = json.loads(archive.read("diagnostic.json"))
                log_text = archive.read("guide.log").decode("utf-8")
            self.assertEqual(
                set(metadata),
                {
                    "versions",
                    "release_fingerprint",
                    "capabilities",
                    "status",
                    "reason_codes",
                    "artifact_hashes",
                },
            )
            combined = json.dumps(metadata) + log_text
            for secret in secrets:
                self.assertNotIn(secret.lower(), combined.lower())
            self.assertNotIn("C:\\Users", combined)
            self.assertNotIn("/home/", combined)
            for forbidden_entry in (
                "active-run.json",
                "state-event.json",
                "advice-event.json",
                "sts2.db",
                ".env",
            ):
                self.assertNotIn(forbidden_entry, archive.namelist())

    def test_log_export_is_bounded(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "diagnostics.zip"
            export_diagnostics_zip(
                output,
                self._snapshot(),
                ("x" * 5000 for _ in range(MAX_LOG_LINES + 50)),
            )
            with zipfile.ZipFile(output) as archive:
                logs = archive.read("guide.log")
            self.assertLessEqual(len(logs), MAX_LOG_BYTES)
            self.assertLessEqual(len(logs.decode("utf-8").splitlines()), MAX_LOG_LINES)

    def test_unrecognized_state_fields_are_omitted_by_log_allowlist(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "diagnostics.zip"
            leaked = (
                "candidate_id=SECRET_CHOICE option_id=NEOW_TEST "
                "floor=17 hp=9 gold=123"
            )
            export_diagnostics_zip(
                output,
                self._snapshot(),
                [
                    leaked,
                    "2026-08-01 INFO Public Beta controller started.",
                ],
            )
            with zipfile.ZipFile(output) as archive:
                logs = archive.read("guide.log").decode("utf-8")
            self.assertEqual(logs, "event=controller_started\n")
            self.assertNotIn("SECRET_CHOICE", logs)
            self.assertNotIn("floor=17", logs)

    def test_metadata_rejects_paths_or_unstructured_reasons(self):
        with self.assertRaises(ValueError):
            DiagnosticSnapshot(
                versions={"guide": r"C:\Users\Alice\Guide.exe"},
                release_fingerprint="0" * 64,
                capabilities={"card_reward": "enabled"},
                status="failed",
                reason_codes=("bad path C:/Users/Alice",),
                artifact_hashes={"game": "1" * 64},
            )


if __name__ == "__main__":
    unittest.main()
