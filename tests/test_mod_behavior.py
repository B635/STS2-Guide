"""Execute shared production C# against the local game, without starting Godot.

Explicitly skipped on machines without the developer assembly environment;
release validation on the configured Windows builder must execute these tests.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from advisor.route import RoutePolicy
from storage.relational import RelationalRepository
from storage.release_database import build_release_database
from test_route_decision import _route_request, _state


ROOT = Path(__file__).resolve().parents[1]


class ModBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        props = ROOT / "mod/STS2Guide.ReadOnlyExporter/local.props"
        game_path = os.environ.get("STS2_TEST_GAME_PATH")
        if not game_path and props.exists():
            game_path = ET.parse(props).findtext(".//STS2GamePath")
        if not game_path or not shutil.which("dotnet"):
            if os.environ.get("STS2_REQUIRE_GAME_TESTS") == "1":
                raise AssertionError("Release requires .NET and its selected game assembly")
            raise unittest.SkipTest("Local .NET/game assembly environment is not configured")
        game_data = Path(game_path) / "data_sts2_windows_x86_64"
        env = dict(os.environ, DOTNET_CLI_HOME=str(ROOT / ".dotnet_cli"),
                   DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1",
                   APPDATA=str(ROOT / ".dotnet_cli/AppData"),
                   NUGET_PACKAGES=str(ROOT / ".dotnet_cli/packages"))
        result = subprocess.run(
            ["dotnet", "run", "--project", str(ROOT / "tools/GuideBehaviorTests"),
             "-p:STS2GamePath=" + str(game_path),
             "--", str(game_data)], cwd=ROOT, env=env, capture_output=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.entries = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('[{"kind"')))
        cls.catalog = json.loads((ROOT / "data/knowledge.json").read_text(encoding="utf-8"))

    def test_production_ids_cover_shipped_catalog(self):
        for kind in ("cards", "potions", "relics", "characters"):
            observed = {entry["id"] for entry in self.entries if entry["kind"] == kind}
            required = {entry["id"] for entry in self.catalog[kind]}
            self.assertFalse(required - observed, (kind, sorted(required - observed)))

    def test_real_exported_potions_resolve_in_shipped_database_and_affect_advice(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "template.db"
            build_release_database(target, ROOT / "data/knowledge.json", ROOT / "data/community_scores.json")
            repository = RelationalRepository(str(target))
            exported = {entry["type"]: entry["id"] for entry in self.entries}
            for name in ("SpeedPotion", "FairyInABottle", "ExpectAFight", "IAmInvincible"):
                kind = "potions" if name in ("SpeedPotion", "FairyInABottle") else "cards"
                self.assertIsNotNone(repository.find_entity(kind, exported[name]), name)
            self.assertIsNone(repository.find_entity("potions", "NOT_A_REAL_POTION"))
            result = RoutePolicy(repository).recommend(_route_request(
                state=_state(potions=[{"potion": exported["SpeedPotion"], "slot": 0}]),
            ))
            self.assertNotIn("unknown_potion:SPEED_POTION", result.data_gaps)
            self.assertTrue(result.presentation["primary_path_node_ids"], result.data_gaps)
            self.assertTrue(any(
                factor["code"] == "potion_buffer_once" and factor["delta"] > 0
                for candidate in result.candidates for factor in candidate.factors
            ))
