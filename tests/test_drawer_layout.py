"""Run the production drawer in a real headless Godot, not source-text assertions."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


class DrawerLayoutTests(unittest.TestCase):
    def test_production_drawer_fits_content_and_scrolls_only_when_needed(self):
        props = ROOT / "mod/STS2Guide.ReadOnlyExporter/local.props"
        godot = os.environ.get("STS2_TEST_GODOT_PATH")
        if not godot and props.exists():
            godot = ET.parse(props).findtext(".//GodotExePath")
        if not godot or not Path(godot).is_file() or not shutil.which("dotnet"):
            if os.environ.get("STS2_REQUIRE_GAME_TESTS") == "1":
                self.fail("Release requires the configured Godot Mono layout runner")
            self.skipTest("Local Godot Mono/.NET environment is not configured")
        env = dict(os.environ, DOTNET_CLI_HOME=str(ROOT / ".dotnet_cli"),
                   DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1",
                   APPDATA=str(ROOT / ".dotnet_cli/AppData"),
                   NUGET_PACKAGES=str(ROOT / ".dotnet_cli/packages"),
                   GODOT_NUGET_SOURCE=str(Path(godot).parent / "GodotSharp/Tools/nupkgs"))
        project = ROOT / "tools/DrawerLayoutTests"
        for command in (["dotnet", "build", str(project), "-v:q"],
                        [godot, "--headless", "--path", str(project)]):
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                    encoding="utf-8", errors="replace", timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DRAWER_LAYOUT_PASS", result.stdout)
