import copy
import json
import tempfile
import unittest
from pathlib import Path

from advisor.card_reward import recommend_card_reward
from advisor.scoring_preferences import ScoringPreferences
from storage.relational import RelationalRepository


GAP = "card_mechanic:PERFECTED_STRIKE:strike_count_damage"


class PerfectedStrikeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.repository = RelationalRepository(str(self.path / "catalog.db"))
        self.cards = [
            {
                "id": "PERFECTED_STRIKE", "name": "完美打击", "cost": 2,
                "type_key": "Attack", "color": "ironclad", "damage": 6,
                "tags": ["Strike"],
                "vars": {"CalculationBase": 6, "ExtraDamage": 2},
                "upgrade": {"extradamage": "+1"},
            },
            {
                "id": "TAGGED_CARD", "name": "不按名字分类", "cost": 1,
                "type_key": "Attack", "color": "ironclad", "damage": 6,
                "tags": ["Strike"],
            },
            {
                "id": "FALSE_STRIKE", "name": "名字中的打击不算", "cost": 1,
                "type_key": "Attack", "color": "ironclad", "damage": 6,
                "tags": None,
            },
        ]
        self.sync()
        self.state = {
            "character": "IRONCLAD", "act": 1, "floor": 3,
            "hp": 60, "max_hp": 80, "deck": [{"card": "TAGGED_CARD", "count": 5}],
        }

    def sync(self):
        snapshot = self.path / "catalog.json"
        snapshot.write_text(json.dumps({"cards": self.cards}), encoding="utf-8")
        self.repository.sync_catalog(str(snapshot))

    def recommend(self, option=None, **kwargs):
        return recommend_card_reward(
            self.state, [option or {"card": "PERFECTED_STRIKE"}],
            self.repository, **kwargs,
        )["recommendations"][0]

    def damage_factor(self, row):
        factors = [factor for factor in row["factors"]
                   if factor["code"] == "damage_efficiency"]
        self.assertEqual(len(factors), 1)
        return factors[0]

    def scope(self, row):
        factors = [factor for factor in row["factors"]
                   if factor["code"] == "estimate_scope"]
        self.assertEqual(len(factors), 1)
        self.assertEqual(factors[0]["delta"], 0.0)
        return factors[0]["message"]

    def test_upgraded_offer_changes_only_per_strike_amount_and_is_weighted_once(self):
        base = self.recommend()
        upgraded = self.recommend({"card": "PERFECTED_STRIKE", "upgrades": 1})
        preferred = self.recommend(
            {"card": "PERFECTED_STRIKE", "upgrades": 1},
            preferences=ScoringPreferences(immediate_power=1.25),
        )
        self.assertEqual(self.damage_factor(base)["delta"], 1.65)
        self.assertEqual(self.damage_factor(upgraded)["delta"], 3.3)
        self.assertIn("24 点", self.scope(upgraded))
        self.assertEqual(self.damage_factor(preferred)["delta"], 4.125)
        self.assertNotIn(GAP, upgraded["data_gaps"])
        self.assertFalse(any(f["code"] == "unmodeled_mechanic" for f in upgraded["factors"]))

    def test_tags_not_names_and_permanent_copies_include_existing_perfected_strikes(self):
        self.state["deck"] = [
            {"card": "TAGGED_CARD", "count": 1, "upgrades": 1},
            {"card": "FALSE_STRIKE", "count": 5},
            {"card": "PERFECTED_STRIKE", "count": 2},
        ]
        row = self.recommend()
        factor = self.damage_factor(row)
        self.assertEqual(factor["delta"], 0.55)
        self.assertIn("4 张", self.scope(row))
        self.assertIn("14 点", self.scope(row))
        self.state["deck"] = [{"card": "FALSE_STRIKE"}]
        row = self.recommend()
        factor = self.damage_factor(row)
        self.assertEqual(factor["delta"], -1.1)
        self.assertIn("1 张", self.scope(row))
        self.assertIn("8 点", self.scope(row))

    def test_unknown_or_modified_deck_keeps_original_limited_damage_and_gap(self):
        decks = (
            [], [{"card": "UNRESOLVED"}],
            [{"card": "TAGGED_CARD", "enchantment": "UNKNOWN"}],
            [{"card": "TAGGED_CARD", "affliction": "UNKNOWN"}],
            [{"card": "TAGGED_CARD", "upgrades": 2}],
            [{"card": "TAGGED_CARD", "count": 0}],
        )
        for deck in decks:
            with self.subTest(deck=deck):
                self.state["deck"] = deck
                row = self.recommend()
                self.assertIn(GAP, row["data_gaps"])
                self.assertEqual(self.damage_factor(row)["delta"], -1.65)
                self.assertTrue(any(f["code"] == "unmodeled_mechanic" for f in row["factors"]))

    def test_modified_or_repeatedly_upgraded_offer_is_not_a_known_baseline(self):
        for changes in (
            {"enchantment": "UNKNOWN"}, {"affliction": "UNKNOWN"},
            {"upgrades": 2}, {"upgrades": True},
        ):
            with self.subTest(changes=changes):
                row = self.recommend({"card": "PERFECTED_STRIKE", **changes})
                self.assertIn(GAP, row["data_gaps"])
                self.assertEqual(self.damage_factor(row)["delta"], -1.65)

    def test_missing_or_malformed_static_facts_do_not_count_as_zero_or_guess_damage(self):
        cases = (
            (0, "vars", None), (0, "vars", {}),
            (0, "vars", {"CalculationBase": 6, "ExtraDamage": 3}),
            (0, "tags", ["Strike", 1]),
            (0, "tags", None), (1, "tags", "Strike"),
        )
        original = copy.deepcopy(self.cards)
        for index, field, value in cases:
            with self.subTest(index=index, field=field, value=value):
                self.cards = copy.deepcopy(original)
                self.cards[index][field] = value
                self.sync()
                row = self.recommend()
                self.assertIn(GAP, row["data_gaps"])
                self.assertEqual(self.damage_factor(row)["delta"], -1.65)

    def test_absent_classification_and_unknown_upgrade_delta_stay_unmodeled(self):
        original = copy.deepcopy(self.cards)
        for index, field, option in (
            (1, "tags", {"card": "PERFECTED_STRIKE"}),
            (0, "vars", {"card": "PERFECTED_STRIKE"}),
            (0, "upgrade", {"card": "PERFECTED_STRIKE", "upgrades": 1}),
        ):
            with self.subTest(index=index, field=field):
                self.cards = copy.deepcopy(original)
                del self.cards[index][field]
                self.sync()
                row = self.recommend(option)
                self.assertIn(GAP, row["data_gaps"])
                self.assertEqual(self.damage_factor(row)["delta"], -1.65)
                self.assertFalse(any(f["code"] == "estimate_scope" for f in row["factors"]))


if __name__ == "__main__":
    unittest.main()
