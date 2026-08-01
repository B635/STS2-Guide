from __future__ import annotations

import re
import os
import tempfile
import unittest
from pathlib import Path

from scripts.audit_public_beta import assert_artifact_fresh


ROOT = Path(__file__).resolve().parents[1]


class PublicBetaPackagingTests(unittest.TestCase):
    def test_frozen_payload_uses_release_db_and_original_icon(self):
        spec = (ROOT / "packaging" / "sts2-guide.spec").read_text(
            encoding="utf-8"
        )
        self.assertIn("sts2-guide-template.db", spec)
        self.assertIn("sts2-guide.png", spec)
        self.assertIn("sts2-guide.ico", spec)
        self.assertNotIn('root / "data" / "knowledge.json"', spec)
        self.assertNotIn('root / "data" / "community_scores.json"', spec)
        self.assertIn("upx=False", spec)
        self.assertIn("version_info.txt", spec)
        self.assertIn('"pystray._win32"', spec)

    def test_release_requirements_are_exactly_pinned(self):
        lines = [
            line.strip()
            for line in (ROOT / "requirements-p0.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertTrue(lines)
        self.assertTrue(all("==" in line for line in lines), lines)
        self.assertIn("pystray==0.19.5", lines)
        self.assertIn("pyinstaller==6.21.0", lines)

    def test_installer_is_per_user_exact_and_owns_only_guide_files(self):
        installer = (ROOT / "packaging" / "installer.iss").read_text(
            encoding="utf-8"
        )
        self.assertIn("PrivilegesRequired=lowest", installer)
        self.assertIn("{localappdata}\\Programs\\STS2 Guide", installer)
        self.assertIn("ExpectedGameAssemblySHA256", installer)
        self.assertIn("ExpectedReleaseInfoSHA256", installer)
        self.assertIn("GetSHA256OfFile(AssemblyPath)", installer)
        self.assertIn("GetSHA256OfFile(ReleaseInfoPath)", installer)
        self.assertIn("release_info.json", installer)
        self.assertIn("ExactProcessIsRunning", installer)
        self.assertIn("StopExistingGuide", installer)
        self.assertIn("--shutdown-existing", installer)
        self.assertIn("InspectionSucceeded", installer)
        self.assertIn("{param:GAMEPATH|}", installer)
        self.assertIn("install.json", installer)
        self.assertIn("guide_executable_sha256", installer)
        self.assertNotIn("RouteLiveProbe", installer)
        self.assertNotIn("STS2MCP", installer)
        self.assertNotRegex(installer, r"(?i)Remove-Item|del /|rm -")
        file_sources = re.findall(r'^Source:\s*"([^"]+)"', installer, re.M)
        self.assertEqual(
            len(file_sources),
            6,
            "Installer payload must remain an explicit six-file allowlist",
        )

    def test_release_build_orders_data_mod_exe_installer_and_audit(self):
        script = (ROOT / "scripts" / "build_public_beta.ps1").read_text(
            encoding="utf-8"
        )
        markers = [
            "unittest discover",
            "scripts\\build_icon.py",
            "scripts\\build_release_database.py",
            "dotnet build",
            "PyInstaller",
            "installer.iss",
            "audit_public_beta.py",
        ]
        offsets = [script.index(marker) for marker in markers]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("InstallModOnBuild=false", script)
        self.assertNotIn("pip install", script)

    def test_release_audit_rejects_artifact_older_than_source(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            artifact = root / "guide.exe"
            source = root / "controller.py"
            artifact.write_bytes(b"old")
            source.write_text("new", encoding="utf-8")
            os.utime(artifact, ns=(1_000_000_000, 1_000_000_000))
            os.utime(source, ns=(2_000_000_000, 2_000_000_000))
            with self.assertRaises(SystemExit):
                assert_artifact_fresh(
                    artifact,
                    [source],
                    label="Guide executable",
                )
            os.utime(artifact, ns=(3_000_000_000, 3_000_000_000))
            assert_artifact_fresh(
                artifact,
                [source],
                label="Guide executable",
            )


if __name__ == "__main__":
    unittest.main()
