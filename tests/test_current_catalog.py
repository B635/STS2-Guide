"""Regression samples for the checked-in 0.111.0 catalog, not policy quality."""
import json
import unittest
import tempfile
from pathlib import Path

from advisor.route import RoutePolicy
from storage.release_database import build_release_database
from storage.relational import RelationalRepository
from storage.effect_tags import derive_effect_tags
from test_route_decision import _route_request


ROOT = Path(__file__).resolve().parents[1]


class CurrentCatalogTests(unittest.TestCase):
    def test_shipped_database_can_produce_a_route_not_only_card_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "template.db"
            build_release_database(target, ROOT / "data/knowledge.json", ROOT / "data/community_scores.json")
            result = RoutePolicy(RelationalRepository(str(target))).recommend(_route_request())
            self.assertTrue(result.presentation["primary_path_node_ids"], result.data_gaps)

    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "data/knowledge.json").read_text(encoding="utf-8"))
        cls.cards = {card["id"]: card for card in cls.catalog["cards"]}

    def test_catalog_channel_version_and_ids(self):
        manifest = json.loads((ROOT / "packaging/compatibility.json").read_text(encoding="utf-8"))
        self.assertEqual(self.catalog["_source"]["entity_channel"], "beta")
        self.assertEqual(self.catalog["_source"]["game_version"], manifest["game"]["version"])
        owned = json.loads((ROOT / "data/guide_mechanics.json").read_text(encoding="utf-8"))
        for key, value in owned.items():
            self.assertEqual(self.catalog["mechanics"][key], value)
        for kind in ("characters", "cards", "relics", "potions", "monsters", "acts", "encounters"):
            ids = [item["id"] for item in self.catalog[kind]]
            self.assertTrue(ids, kind)
            self.assertEqual(len(ids), len(set(ids)), kind)

    def test_reworked_card_is_defense_not_old_energy_source(self):
        card = self.cards["EXPECT_A_FIGHT"]
        self.assertEqual((card["cost"], card["block"]), (3, 15))
        tags = derive_effect_tags("cards", card)
        self.assertEqual(tags["block"][0], 15)
        self.assertNotIn("energy_gain", tags)

    def test_changed_upgrade_and_timing_facts_are_preserved(self):
        self.assertEqual(self.cards["RAMPAGE"]["damage"], 10)
        self.assertEqual(self.cards["RAMPAGE"]["upgrade"]["increase"], "+5")
        self.assertEqual(self.cards["MIRAGE"]["upgrade"], {"remove_exhaust": 1})
        self.assertEqual(self.cards["GUIDING_STAR"]["star_cost"], 1)
        self.assertIn("下个回合", self.cards["GUIDING_STAR"]["description"])
        self.assertEqual(self.cards["HYPERBEAM"]["damage"], 24)
        self.assertIn("本回合", self.cards["HYPERBEAM"]["description"])


if __name__ == "__main__":
    unittest.main()
