import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "tools" / "RouteLiveProbe"


class RouteLiveProbeIsolationTests(unittest.TestCase):
    def test_probe_is_separate_from_production_mod(self):
        production_project = (
            ROOT
            / "mod"
            / "STS2Guide.ReadOnlyExporter"
            / "STS2Guide.ReadOnlyExporter.csproj"
        ).read_text(encoding="utf-8")
        production_pack = (
            ROOT / "mod" / "STS2Guide.ReadOnlyExporter" / "pack" / "project.godot"
        ).read_text(encoding="utf-8")
        self.assertNotIn("RouteLiveProbe", production_project)
        self.assertNotIn("RouteLiveProbe", production_pack)

        probe_project = (PROBE / "RouteLiveProbe.csproj").read_text(encoding="utf-8")
        self.assertIn("STS2GuideRouteLiveProbe", probe_project)
        self.assertNotIn("ProjectReference", probe_project)
        manifest = json.loads(
            (PROBE / "STS2GuideRouteLiveProbe.json").read_text(encoding="utf-8")
        )
        self.assertFalse(manifest["affects_gameplay"])

    def test_probe_uses_postfixes_and_never_invokes_selection_actions(self):
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in PROBE.glob("*.cs")
        )
        self.assertIn("HarmonyPostfix", source)
        self.assertNotIn("HarmonyPrefix", source)
        self.assertNotIn(".TravelToMapCoord(", source)
        self.assertNotIn(".EnterMapCoord(", source)
        self.assertNotIn(".OnMapPointSelectedLocally(", source)
        self.assertNotIn(".OnSelected(", source)
        self.assertIn("nameof(NMapScreen.OnMapPointSelectedLocally)", source)
        self.assertIn("DebugOnlyGetState()", source)
        self.assertIn("_logPath = null;", source)

    def test_validator_accepts_minimal_jsonl(self):
        record = {
            "schema_version": 1,
            "session_id": "test-session",
            "sequence": 1,
            "observed_at_utc": "2026-07-14T00:00:00.0000000+00:00",
            "event_name": "probe_session_started",
            "assembly_version": "0.1.0.0",
            "assembly_mvid": "00000000-0000-0000-0000-000000000000",
            "snapshot": None,
            "details": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "probe.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROBE / "validate_probe_log.py"),
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(summary["records"], 1)


if __name__ == "__main__":
    unittest.main()
