import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from advisor.card_reward import recommend_card_reward
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_PATH = os.path.join(ROOT, "data", "knowledge.json")


class CardQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "card_quality.db")
        )
        cls.repository.sync_catalog(KNOWLEDGE_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def _recommend(self, options, *, deck=None, character="IRONCLAD", energy=3):
        return recommend_card_reward(
            {
                "character": character,
                "act": 1,
                "floor": 6,
                "hp": 70,
                "max_hp": 80,
                "energy": energy,
                "deck": deck if deck is not None else [
                    {"card": "DEFEND_IRONCLAD", "count": 5},
                ],
            },
            options,
            self.repository,
            can_skip=False,
        )

    @staticmethod
    def _factors(row):
        return {factor["code"]: factor["delta"] for factor in row["factors"]}

    def test_fixed_multi_hit_uses_total_damage_without_strength_provider(self):
        result = self._recommend([
            {"card": "STRIKE_IRONCLAD"},
            {"card": "TWIN_STRIKE"},
        ])
        rows = {row["card_id"]: row for row in result["recommendations"]}
        twin = rows["TWIN_STRIKE"]
        self.assertEqual(self._factors(twin)["damage_efficiency"], 2.2)
        self.assertNotIn("damage_efficiency", self._factors(rows["STRIKE_IRONCLAD"]))
        self.assertGreater(twin["state_score"], rows["STRIKE_IRONCLAD"]["state_score"])
        self.assertEqual(result["recommended_option_index"], 1)
        self.assertEqual(twin["option_index"], 1)

    def test_fixed_aoe_hits_keep_existing_aoe_weight(self):
        row = self._recommend(
            [{"card": "DAGGER_SPRAY"}], character="SILENT"
        )["recommendations"][0]
        factors = self._factors(row)
        self.assertEqual(factors["damage_efficiency"], 1.1)
        self.assertEqual(factors["aoe_value"], 2.5)

    def test_conditional_extra_hit_is_not_assumed_active(self):
        card = self.repository.find_card("SPITE")
        self.assertEqual(card["hit_count"], 2)
        row = self._recommend([{"card": "SPITE"}])["recommendations"][0]
        self.assertEqual(self._factors(row)["damage_efficiency"], -0.55)

    def test_fixed_hits_with_random_targets_still_count(self):
        row = self._recommend(
            [{"card": "SWORD_BOOMERANG"}]
        )["recommendations"][0]
        self.assertEqual(self._factors(row)["damage_efficiency"], 1.65)

    def test_unknown_or_invalid_hit_count_never_uses_materialized_multiplier(self):
        original_find = self.repository.find_card
        original = original_find("TWIN_STRIKE")
        for hit_count in (None, 0, -1, True, 1.5, "X"):
            card = copy.deepcopy(original)
            card["hit_count"] = hit_count

            def find(identifier):
                return card if identifier == "TWIN_STRIKE" else original_find(identifier)

            with self.subTest(hit_count=hit_count), patch.object(
                self.repository, "find_card", find
            ):
                row = self._recommend(
                    [{"card": "TWIN_STRIKE"}]
                )["recommendations"][0]
                self.assertEqual(self._factors(row)["damage_efficiency"], -0.55)

    def test_x_cost_does_not_become_free_curve_relief_or_fixed_efficiency(self):
        deck = [{"card": "BLUDGEON", "count": 5}]
        for energy in (1, 9):
            with self.subTest(energy=energy):
                row = self._recommend(
                    [{"card": "WHIRLWIND"}], deck=deck, energy=energy
                )["recommendations"][0]
                factors = self._factors(row)
                self.assertNotIn("curve_relief", factors)
                self.assertNotIn("damage_efficiency", factors)

    def test_x_star_hits_have_no_fixed_efficiency_but_keep_known_energy_cost(self):
        card = self.repository.find_card("STARDUST")
        self.assertEqual(card["cost"], 0)
        self.assertTrue(card["is_x_star_cost"])
        self.assertIsNone(card["hit_count"])
        result = self._recommend(
            [
                {"card": "STARDUST", "upgrades": 0},
                {"card": "STARDUST", "upgrades": 1},
            ],
            deck=[{"card": "SEVEN_STARS", "count": 5}],
            character="REGENT",
        )
        for row in result["recommendations"]:
            with self.subTest(upgrades=row["upgrades"]):
                factors = self._factors(row)
                self.assertNotIn("damage_efficiency", factors)
                self.assertEqual(factors["curve_relief"], 5.0)
        deck_result = self._recommend(
            [{"card": "SEVEN_STARS"}],
            deck=[{"card": "STARDUST", "count": 5}],
            character="REGENT",
        )
        self.assertEqual(deck_result["profile"]["average_cost"], 0.0)
        self.assertEqual(deck_result["profile"]["high_cost_ratio"], 0.0)

    def test_upgraded_cost_changes_candidate_curve_and_preserves_instances(self):
        result = self._recommend(
            [
                {"card": "WELL_LAID_PLANS", "upgrades": 0},
                {"card": "WELL_LAID_PLANS", "upgrades": 1},
            ],
            deck=[{"card": "WRAITH_FORM", "count": 5}],
            character="SILENT",
        )
        rows = {row["option_index"]: row for row in result["recommendations"]}
        self.assertEqual(self._factors(rows[0])["curve_pressure"], -5.0)
        self.assertNotIn("curve_pressure", self._factors(rows[1]))
        self.assertEqual(self._factors(rows[1])["curve_relief"], 5.0)
        self.assertEqual(self._factors(rows[1])["upgraded_offer"], 4.0)
        self.assertEqual([rows[i]["upgrades"] for i in (0, 1)], [0, 1])
        self.assertEqual(result["recommended_option_index"], 1)
        self.assertEqual(self.repository.find_card("WELL_LAID_PLANS")["cost"], 2)

    def test_upgraded_deck_cost_changes_profile_and_candidate_curve(self):
        results = [
            self._recommend(
                [{"card": "DAGGER_THROW"}],
                deck=[
                    {"card": "WELL_LAID_PLANS", "count": 3, "upgrades": upgrades},
                    {"card": "DEFEND_SILENT", "count": 2},
                ],
                character="SILENT",
            )
            for upgrades in (0, 1)
        ]
        self.assertEqual(results[0]["profile"]["average_cost"], 1.6)
        self.assertEqual(results[0]["profile"]["high_cost_ratio"], 0.6)
        self.assertEqual(results[1]["profile"]["average_cost"], 1.0)
        self.assertEqual(results[1]["profile"]["high_cost_ratio"], 0.0)
        self.assertIn("curve_relief", self._factors(results[0]["recommendations"][0]))
        self.assertNotIn("curve_relief", self._factors(results[1]["recommendations"][0]))

    def test_explicit_zero_cost_upgrade_is_not_lost(self):
        result = self._recommend(
            [{"card": "STRIKE_DEFECT"}],
            deck=[{"card": "ZAP", "count": 5, "upgrades": 1}],
            character="DEFECT",
        )
        self.assertEqual(result["profile"]["average_cost"], 0.0)
        self.assertEqual(result["profile"]["high_cost_ratio"], 0.0)

    def test_x_cost_deck_entries_are_excluded_from_known_cost_average(self):
        result = self._recommend(
            [{"card": "TWIN_STRIKE"}],
            deck=[
                {"card": "WHIRLWIND", "count": 5},
                {"card": "BLUDGEON"},
            ],
        )
        self.assertEqual(result["profile"]["average_cost"], 3.0)
        self.assertEqual(result["profile"]["high_cost_ratio"], 1.0)

    def test_unknown_upgrade_cost_is_not_used_as_known_base_cost(self):
        original_find = self.repository.find_card
        original = original_find("WELL_LAID_PLANS")
        for cost in ("X", None, True, -1, 1.5):
            card = copy.deepcopy(original)
            card["upgrade"]["cost"] = cost

            def find(identifier):
                return card if identifier == "WELL_LAID_PLANS" else original_find(identifier)

            with self.subTest(cost=cost), patch.object(self.repository, "find_card", find):
                result = self._recommend(
                    [{"card": "WELL_LAID_PLANS", "upgrades": 1}],
                    deck=[{"card": "WRAITH_FORM", "count": 5}],
                    character="SILENT",
                )
                factors = self._factors(result["recommendations"][0])
                self.assertNotIn("curve_pressure", factors)
                self.assertNotIn("curve_relief", factors)

    def test_signed_upgrade_cost_uses_explicit_delta(self):
        original_find = self.repository.find_card
        card = copy.deepcopy(original_find("WELL_LAID_PLANS"))
        card["upgrade"]["cost"] = "-1"

        def find(identifier):
            return card if identifier == "WELL_LAID_PLANS" else original_find(identifier)

        with patch.object(self.repository, "find_card", find):
            result = self._recommend(
                [{"card": "DAGGER_THROW"}],
                deck=[{"card": "WELL_LAID_PLANS", "upgrades": 1}],
                character="SILENT",
            )
        self.assertEqual(result["profile"]["average_cost"], 1.0)

    def test_unrelated_upgrade_effects_retain_existing_baseline(self):
        result = self._recommend([
            {"card": "POMMEL_STRIKE", "upgrades": 0},
            {"card": "POMMEL_STRIKE", "upgrades": 1},
        ])
        rows = {row["option_index"]: row for row in result["recommendations"]}
        self.assertAlmostEqual(rows[1]["state_score"] - rows[0]["state_score"], 4.0)
        upgraded = self._factors(rows[1])
        upgraded.pop("upgraded_offer")
        self.assertEqual(upgraded, self._factors(rows[0]))


if __name__ == "__main__":
    unittest.main()
