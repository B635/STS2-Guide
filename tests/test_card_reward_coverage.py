import json
import tempfile
import unittest
from pathlib import Path

from advisor.card_reward import recommend_card_reward
from advisor.decision_core import (
    CARD_REWARD,
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from advisor.policies import CardRewardPolicy
from storage.release_database import build_release_database
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]


class CardRewardCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            str(Path(cls.temporary.name) / "catalog.db")
        )
        cls.repository.sync_catalog(str(ROOT / "data" / "knowledge.json"))
        release_path = Path(cls.temporary.name) / "release.db"
        build_release_database(
            release_path,
            ROOT / "data" / "knowledge.json",
            ROOT / "data" / "community_scores.json",
        )
        cls.release_repository = RelationalRepository(str(release_path))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def live_state():
        return {
            "character": "IRONCLAD",
            "act": 1,
            "floor": 3,
            "hp": 66,
            "max_hp": 80,
            "energy": 3,
            "deck": [
                {"card": "BASH", "count": 1},
                {"card": "DARK_EMBRACE", "count": 1},
                {"card": "DEFEND_IRONCLAD", "count": 4},
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "THUNDERCLAP", "count": 1},
            ],
        }

    def test_draw_coverage_describes_low_share_not_missing_draw_or_activation(self):
        result = recommend_card_reward(
            self.live_state(),
            [{"card": "POMMEL_STRIKE"}],
            self.repository,
        )
        row = result["recommendations"][0]
        coverage = next(
            factor for factor in row["factors"]
            if factor["code"] == "draw_coverage"
        )

        self.assertEqual(result["profile"]["effect_tag_counts"]["draw"], 1)
        self.assertEqual(result["profile"]["deck_size"], 12)
        self.assertEqual(coverage["delta"], 4.0)
        self.assertEqual(row["state_score"], 61.65)
        self.assertEqual(
            coverage["message"],
            "当前牌组中带抽牌效果的卡牌占比偏低，该牌可补充抽牌覆盖。",
        )

    def test_perfected_strike_uses_acquired_deck_baseline_in_release(self):
        for repository in (self.repository, self.release_repository):
            with self.subTest(repository=repository.database_path):
                result = recommend_card_reward(
                    self.live_state(), [{"card": "PERFECTED_STRIKE"}], repository,
                )
                row = result["recommendations"][0]
                damage = [factor for factor in row["factors"]
                          if factor["code"] == "damage_efficiency"]
                # Five permanent Strikes plus the acquired card: 18 damage,
                # costing 2. Existing bounded efficiency yields +1.65.
                self.assertEqual(len(damage), 1)
                self.assertEqual(damage[0]["delta"], 1.65)
                scope = next(factor for factor in row["factors"]
                             if factor["code"] == "estimate_scope")
                self.assertEqual(scope["delta"], 0.0)
                self.assertIn("6 张", scope["message"])
                self.assertIn("18 点", scope["message"])
                self.assertIn("战前", scope["message"])
                self.assertNotIn(
                    "card_mechanic:PERFECTED_STRIKE:strike_count_damage",
                    row["data_gaps"],
                )

    def test_neutral_damage_efficiency_still_discloses_baseline_scope(self):
        state = self.live_state()
        state["deck"] = [{"card": "STRIKE_IRONCLAD", "count": 2}]
        row = recommend_card_reward(
            state, [{"card": "PERFECTED_STRIKE"}], self.release_repository,
        )["recommendations"][0]
        baseline = [factor for factor in row["factors"]
                    if factor["code"] == "estimate_scope"]
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["delta"], 0.0)
        self.assertIn("12 点", baseline[0]["message"])
        self.assertIn("不是当回合实伤", baseline[0]["message"])

    def test_baseline_scope_survives_public_policy_for_upgraded_localized_offer(self):
        request = DecisionRequest.create(
            decision_id="strike-baseline", decision_type=CARD_REWARD,
            world=WorldState.create(run_id="baseline-run", sequence=1, state=self.live_state()),
            candidates=[DecisionCandidate.create(
                "actual-option", {"card": "完美打击", "upgrades": 1},
                label="完美打击+", display_index=0,
            )], constraints={"can_skip": False},
        )
        result = CardRewardPolicy(self.release_repository).recommend(request).as_dict()
        row = result["candidates"][0]
        scope = [factor for factor in row["factors"] if factor["code"] == "estimate_scope"]
        self.assertEqual(len(scope), 1)
        self.assertEqual(scope[0]["delta"], 0.0)
        self.assertIn("24 点", scope[0]["message"])
        self.assertIn("不是当回合实伤", scope[0]["message"])
        self.assertNotIn("card_mechanic:PERFECTED_STRIKE:strike_count_damage", result["data_gaps"])
        efficiency = [factor for factor in row["factors"] if factor["code"] == "damage_efficiency"]
        self.assertEqual(len(efficiency), 1)
        self.assertEqual(efficiency[0]["delta"], 3.3)

    def test_release_retains_exact_assembly_verified_strike_membership(self):
        # Every CanonicalTags getter in the 0.111.0 assembly was inspected;
        # these 22 types add CardTag.Strike (enum value 1), no name matching.
        expected = {
            "ADAPTIVE_STRIKE", "ASHEN_STRIKE", "BLIGHT_STRIKE", "FOCUSED_STRIKE",
            "LEADING_STRIKE", "METEOR_STRIKE", "MINION_STRIKE", "MOMENTUM_STRIKE",
            "PERFECTED_STRIKE", "POMMEL_STRIKE", "SCULPTING_STRIKE", "SEEKER_STRIKE",
            "SETUP_STRIKE", "SHINING_STRIKE", "SOLAR_STRIKE", "STRIKE_DEFECT",
            "STRIKE_IRONCLAD", "STRIKE_NECROBINDER", "STRIKE_REGENT", "STRIKE_SILENT",
            "TWIN_STRIKE", "ULTIMATE_STRIKE",
        }
        catalog = json.loads((ROOT / "data" / "knowledge.json").read_text(encoding="utf-8"))
        cards = [self.release_repository.find_card(card["id"]) for card in catalog["cards"]]
        self.assertTrue(all(isinstance(card.get("tags"), list) for card in cards))
        self.assertEqual({card["id"] for card in cards if "Strike" in card["tags"]}, expected)
        self.assertEqual(
            self.release_repository.find_card("PERFECTED_STRIKE")["vars"],
            {"CalculationBase": 6, "ExtraDamage": 2},
        )
        self.assertNotIn("vars", self.release_repository.find_card("POMMEL_STRIKE"))

    def test_unmodeled_strike_damage_survives_release_policy_and_explanation(self):
        state = self.live_state()
        state["capture_warnings"] = ["deck_empty_or_unavailable"]
        world = WorldState.create(
            run_id="coverage-live-run",
            sequence=47,
            state=state,
        )
        request = DecisionRequest.create(
            decision_id="coverage-reward",
            decision_type=CARD_REWARD,
            world=world,
            candidates=[
                DecisionCandidate.create(
                    f"{index}:{card_id}",
                    {"card": card_id, "upgrades": 0},
                    label=card_id,
                    display_index=index,
                )
                for index, card_id in enumerate(
                    ("PERFECTED_STRIKE", "UNRELENTING", "POMMEL_STRIKE")
                )
            ] + [
                DecisionCandidate.create(
                    "skip", {}, label="跳过", display_index=3,
                ),
            ],
            constraints={"can_skip": True},
        )
        gap = "card_mechanic:PERFECTED_STRIKE:strike_count_damage"
        self.assertNotIn(
            "description", self.release_repository.find_card("PERFECTED_STRIKE")
        )
        for repository, expected_score in (
            (self.repository, 51.85),
            (self.release_repository, 46.72),
        ):
            with self.subTest(repository=repository.database_path):
                result = CardRewardPolicy(repository).recommend(request).as_dict()
                row = result["candidates"][0]
                self.assertIn(gap, row["data_gaps"])
                self.assertIn(gap, result["data_gaps"])
                self.assertEqual(result["confidence"], "low")
                self.assertEqual(row["dimensions"]["data_completeness"], 1.0)
                self.assertEqual(row["score"], expected_score)
                self.assertEqual(row["rank"], 3)
                self.assertEqual(result["recommended_candidate_id"], "2:POMMEL_STRIKE")
                warnings = [
                    factor for factor in row["factors"]
                    if factor["code"] == "unmodeled_mechanic"
                ]
                self.assertEqual(len(warnings), 1)
                self.assertEqual(warnings[0]["delta"], 0.0)
                self.assertEqual(
                    warnings[0]["message"],
                    "评分尚未覆盖按打击牌数量增加的伤害；输出效率仅按基础伤害估算。",
                )
                for other in result["candidates"][1:]:
                    self.assertNotIn(gap, other["data_gaps"])
                    self.assertIsNotNone(other["score"])
                    other_warnings = [
                        factor for factor in other["factors"]
                        if factor["code"] == "unmodeled_mechanic"
                    ]
                    if other["candidate_id"] == "1:UNRELENTING":
                        self.assertIn(
                            "card_mechanic:UNRELENTING:next_attack_free",
                            other["data_gaps"],
                        )
                        self.assertEqual(len(other_warnings), 1)
                        self.assertEqual(other_warnings[0]["delta"], 0.0)
                        self.assertIn("下一张攻击牌", other_warnings[0]["message"])
                    else:
                        self.assertEqual(other_warnings, [])

    def test_upgraded_or_localized_lookup_retains_canonical_mechanism_gap(self):
        state = self.live_state()
        state["capture_warnings"] = ["deck_empty_or_unavailable"]
        for identifier in ("PERFECTED_STRIKE", "perfected_strike", "完美打击"):
            for upgrades in (0, 1):
                with self.subTest(identifier=identifier, upgrades=upgrades):
                    result = recommend_card_reward(
                        state,
                        [{"card": identifier, "upgrades": upgrades}],
                        self.release_repository,
                    )
                    row = result["recommendations"][0]
                    self.assertEqual(row["card_id"], "PERFECTED_STRIKE")
                    self.assertIn(
                        "card_mechanic:PERFECTED_STRIKE:strike_count_damage",
                        row["data_gaps"],
                    )
                    self.assertEqual(result["confidence"], "low")
                    warnings = [
                        factor for factor in row["factors"]
                        if factor["code"] == "unmodeled_mechanic"
                    ]
                    self.assertEqual(len(warnings), 1)
                    self.assertEqual(warnings[0]["delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
