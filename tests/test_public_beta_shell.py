from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from realtime.compatibility import load_compatibility_manifest
from realtime.controller import CompatibilityResult, ControllerSnapshot, ControllerState
from realtime.host import _runtime_mode, build_bridge
from realtime.installation import (
    DiscoveryResult,
    GAME_ASSEMBLY,
    GAME_EXECUTABLE,
    GUIDE_MOD_ARTIFACTS,
    InstallReceipt,
    InstalledArtifact,
    expected_mod_hashes,
    read_release_info_version,
    sha256_file,
    write_install_receipt,
)
from realtime.public_beta import (
    InstallationCompatibilityChecker,
    StatusTrayBackend,
    build_public_beta_controller,
)
from storage.release_database import build_release_database


ROOT = Path(__file__).resolve().parents[1]


class FakeTray:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.snapshots = []

    def start(self, request_exit):
        self.started = True
        self.request_exit = request_exit

    def update(self, snapshot):
        self.snapshots.append(snapshot)

    def activate(self):
        return

    def stop(self):
        self.stopped = True


class PublicBetaShellTests(unittest.TestCase):
    def _game_and_receipt(self, root: Path):
        manifest = load_compatibility_manifest(
            ROOT / "packaging" / "compatibility.json"
        )
        game = root / "Slay the Spire 2"
        (game / GAME_ASSEMBLY).parent.mkdir(parents=True)
        (game / GAME_EXECUTABLE).write_bytes(b"game")
        (game / GAME_ASSEMBLY).write_bytes(b"assembly")
        (game / "release_info.json").write_text(
            json.dumps(
                {"version": f"v{manifest.game.version}", "commit": "test"}
            ),
            encoding="utf-8",
        )
        mods = game / "mods"
        mods.mkdir()
        artifacts = []
        for name in GUIDE_MOD_ARTIFACTS:
            target = mods / name
            target.write_bytes(name.encode("utf-8"))
            artifacts.append(
                InstalledArtifact(f"mods/{name}", sha256_file(target))
            )
        receipt = InstallReceipt(
            guide_version=manifest.guide.version,
            release_fingerprint=manifest.release_fingerprint,
            guide_executable_sha256="7" * 64,
            game_directory=game.resolve(),
            artifacts=tuple(artifacts),
        )
        receipt_path = root / "install.json"
        write_install_receipt(receipt_path, receipt)
        return manifest, game, receipt_path, receipt

    def test_release_info_is_game_version_source_not_pe_metadata(self):
        with tempfile.TemporaryDirectory() as tempdir:
            game = Path(tempdir)
            (game / "release_info.json").write_text(
                '{"version":"v0.110.1","commit":"db5d3552"}',
                encoding="utf-8",
            )
            self.assertEqual(read_release_info_version(game), "0.110.1")
            (game / "release_info.json").write_text(
                '{"version":"v0.110.1","version":"v9.9.9"}',
                encoding="utf-8",
            )
            self.assertIsNone(read_release_info_version(game))

    def test_receipt_extracts_exact_three_mod_hashes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            _, _, _, receipt = self._game_and_receipt(Path(tempdir))
            hashes = expected_mod_hashes(receipt)
            self.assertEqual(set(hashes or {}), set(GUIDE_MOD_ARTIFACTS))
            extra = InstallReceipt(
                guide_version=receipt.guide_version,
                release_fingerprint=receipt.release_fingerprint,
                guide_executable_sha256=receipt.guide_executable_sha256,
                game_directory=receipt.game_directory,
                artifacts=receipt.artifacts
                + (InstalledArtifact("mods/Other.dll", "0" * 64),),
            )
            self.assertIsNone(expected_mod_hashes(extra))

    def test_installation_checker_passes_exact_release_and_clears_on_drift(self):
        with tempfile.TemporaryDirectory() as tempdir:
            manifest, game, receipt_path, _ = self._game_and_receipt(
                Path(tempdir)
            )
            # The fixture has a synthetic assembly; bind the manifest copy to
            # that exact hash while retaining every other production field.
            from dataclasses import replace

            manifest = replace(
                manifest,
                game=replace(
                    manifest.game,
                    sts2_dll_sha256=sha256_file(game / GAME_ASSEMBLY),
                ),
            )
            cleared = []
            checker = InstallationCompatibilityChecker(
                game,
                manifest,
                receipt_path,
                clear_advice=lambda: cleared.append(True),
            )
            self.assertEqual(checker.check(), CompatibilityResult(True))
            (game / "mods" / GUIDE_MOD_ARTIFACTS[0]).write_bytes(b"drift")
            result = checker.check()
            self.assertFalse(result.compatible)
            self.assertIn("guide_artifact_hash_mismatch", result.reason_codes)
            self.assertEqual(cleared, [True])

    def test_status_file_is_whitelisted_and_removed_on_exit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "runtime-status.json"
            inner = FakeTray()
            tray = StatusTrayBackend(inner, path)
            tray.start(lambda: None)
            snapshot = ControllerSnapshot(
                ControllerState.INCOMPATIBLE,
                True,
                False,
                ("game_version_mismatch",),
                r"C:\Users\Alice\secret",
                3,
            )
            tray.update(snapshot)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "schema_version",
                    "state",
                    "game_running",
                    "worker_running",
                    "compatibility_reasons",
                    "activation_count",
                },
            )
            self.assertNotIn("Alice", json.dumps(payload))
            tray.stop()
            self.assertFalse(path.exists())

    def test_frozen_style_bridge_uses_template_without_source_json(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            template = root / "template.db"
            runtime = root / "runtime.db"
            build_release_database(
                template,
                ROOT / "data" / "knowledge.json",
                ROOT / "data" / "community_scores.json",
            )
            args = SimpleNamespace(
                compatibility_manifest=ROOT / "packaging" / "compatibility.json",
                database=runtime,
                release_database=template,
                catalog=root / "missing-knowledge.json",
                community_scores=root / "missing-community.json",
                local_card_tiers=root / "missing-tiers.json",
                checkpoint=root / "active-run.json",
                input=root / "state-event.json",
                output=root / "advice-event.json",
                events_dir=root / "events",
            )
            bridge = build_bridge(args)
            self.assertTrue(runtime.is_file())
            self.assertTrue(bridge.processor.local_compatibility.compatible)

    def test_frozen_defaults_to_tray_and_development_defaults_to_worker(self):
        base = SimpleNamespace(
            tray=False,
            worker=False,
            once=False,
            startup_check=False,
        )
        with patch("realtime.host.sys.frozen", True, create=True):
            self.assertEqual(_runtime_mode(base), "tray")
        with patch("realtime.host.sys.frozen", False, create=True):
            self.assertEqual(_runtime_mode(base), "worker")

    def test_missing_game_directory_builds_incompatible_tray_controller(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            args = SimpleNamespace(
                compatibility_manifest=ROOT / "packaging" / "compatibility.json",
                install_receipt=root / "missing-install.json",
                game_dir=None,
                output=root / "advice-event.json",
                log_file=root / "controller.log",
                worker_log_file=root / "worker.log",
                runtime_status=root / "runtime-status.json",
                checkpoint=root / "active-run.json",
                poll_interval=0.1,
                drain_seconds=2.0,
                worker_stop_timeout=1.0,
            )
            with (
                patch(
                    "realtime.public_beta.discover_game_directory",
                    return_value=DiscoveryResult(
                        None,
                        None,
                        ("game_directory_not_found",),
                    ),
                ),
                patch("realtime.public_beta.CtypesKernelApi", return_value=object()),
            ):
                controller = build_public_beta_controller(args)
            self.assertEqual(controller.state, ControllerState.INCOMPATIBLE)
            self.assertEqual(
                controller.snapshot().compatibility_reasons,
                ("game_directory_not_found",),
            )


if __name__ == "__main__":
    unittest.main()
