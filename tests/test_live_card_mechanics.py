import tempfile
import unittest
from pathlib import Path

from advisor.card_reward import recommend_card_reward
from advisor.decision_core import (
    CARD_REWARD, DecisionCandidate, DecisionRequest, WorldState,
)
from advisor.merchant import MERCHANT_CHOICE, MerchantPolicy
from advisor.policies import CardRewardPolicy
from storage.release_database import build_release_database
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]


class LiveCardMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        release_path = Path(cls.temporary.name) / "release.db"
        build_release_database(
            release_path,
            ROOT / "data" / "knowledge.json",
            ROOT / "data" / "community_scores.json",
        )
        cls.repository = RelationalRepository(str(release_path))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def state(*, with_payoff=True):
        return {
            "character": "IRONCLAD", "act": 1, "floor": 6,
            "hp": 60, "max_hp": 80, "gold": 120, "energy": 3,
            "deck": [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
                *([{"card": "DARK_EMBRACE"}] if with_payoff else []),
            ],
            "relics": [], "potions": [], "max_potion_slots": 3,
        }

    def card_row(self, card, *, upgrades=0, state=None):
        return recommend_card_reward(
            state if state is not None else self.state(),
            [{"card": card, "upgrades": upgrades}], self.repository,
        )["recommendations"][0]

    def test_true_grit_pairs_once_with_dark_embrace_in_release(self):
        for upgrades in (0, 1):
            with self.subTest(upgrades=upgrades):
                row = self.card_row("TRUE_GRIT", upgrades=upgrades)
                synergy = [factor for factor in row["factors"]
                           if factor["code"] == "mechanic_synergy_provider:exhaust"]
                self.assertEqual(len(synergy), 1)
                self.assertEqual(synergy[0]["delta"], 2.1)
                self.assertNotIn("self_exhaust", row["effect_tags"])
                self.assertNotIn("effect_engine_synergy", [
                    factor["code"] for factor in row["factors"]
                ])
                self.assertEqual(row["mechanic_signals"], [{
                    "domain": "exhaust", "role": "provider", "magnitude": 1.0,
                    "source_code": "assembly:0.111.0:TrueGrit.OnPlay.CardCmd.Exhaust",
                    "confidence": "structured",
                }])

    def test_true_grit_without_payoff_does_not_invent_synergy(self):
        for upgrades in (0, 1):
            with self.subTest(upgrades=upgrades):
                row = self.card_row(
                    "TRUE_GRIT", upgrades=upgrades, state=self.state(with_payoff=False),
                )
                self.assertIn("mechanic:exhaust:provider", row["effect_tags"])
                self.assertFalse(any(
                    "synergy" in factor["code"] for factor in row["factors"]
                ))
                self.assertEqual(len([
                    factor for factor in row["factors"]
                    if factor["code"] == "estimate_scope" and factor["delta"] == 0.0
                ]), 1)

    def test_true_grit_in_deck_supplies_payoff_without_mislabeling_other_cards(self):
        state = self.state(with_payoff=False)
        state["deck"].append({"card": "TRUE_GRIT"})
        row = self.card_row("DARK_EMBRACE", state=state)
        synergy = [factor for factor in row["factors"]
                   if factor["code"] == "mechanic_synergy_payoff:exhaust"]
        self.assertEqual(len(synergy), 1)
        self.assertEqual(synergy[0]["delta"], 2.65)
        for card in ("DARK_EMBRACE", "SHRUG_IT_OFF", "DEFEND_IRONCLAD"):
            with self.subTest(card=card):
                other = self.card_row(card)
                self.assertNotIn("mechanic:exhaust:provider", other["effect_tags"])
                self.assertNotIn("self_exhaust", other["effect_tags"])
                self.assertFalse(any(
                    factor["code"] == "estimate_scope" for factor in other["factors"]
                ))

    def test_true_grit_scope_survives_public_policy_for_both_upgrades(self):
        for upgrades, target in ((0, "随机"), (1, "选择")):
            with self.subTest(upgrades=upgrades):
                request = DecisionRequest.create(
                    decision_id="live-card", decision_type=CARD_REWARD,
                    world=WorldState.create(
                        run_id="live-mechanics", sequence=1, state=self.state(),
                    ),
                    candidates=[DecisionCandidate.create(
                        "reward:0", {"card": "坚毅", "upgrades": upgrades},
                    )],
                    constraints={"can_skip": False},
                )
                row = CardRewardPolicy(self.repository).recommend(request).candidates[0]
                scopes = [factor for factor in row.factors
                          if factor["code"] == "estimate_scope"]
                self.assertEqual(len(scopes), 1)
                self.assertEqual(scopes[0]["delta"], 0.0)
                self.assertIn(target, scopes[0]["message"])
                self.assertIn("另一张手牌", scopes[0]["message"])
                self.assertIn("空手", scopes[0]["message"])
                self.assertIn("目标价值", scopes[0]["message"])
                self.assertIn("不代表战斗中已激活", scopes[0]["message"])
                self.assertFalse(row.data_gaps)

    def test_unrelenting_discloses_gap_without_changing_attack_score(self):
        for upgrades, state_score, score in ((0, 50.55, 48.4), (1, 54.55, 52.4)):
            with self.subTest(upgrades=upgrades):
                row = self.card_row("UNRELENTING", upgrades=upgrades)
                self.assertIn(
                    "card_mechanic:UNRELENTING:next_attack_free", row["data_gaps"],
                )
                warnings = [factor for factor in row["factors"]
                            if factor["code"] == "unmodeled_mechanic"]
                self.assertEqual(len(warnings), 1)
                self.assertEqual(warnings[0]["delta"], 0.0)
                self.assertIn("下一张攻击牌", warnings[0]["message"])
                self.assertIn("尚未覆盖", warnings[0]["message"])
                # Captured from the 0.3.10 public recommendation before this fix.
                self.assertEqual(row["state_score"], state_score)
                self.assertEqual(row["score"], score)

    def test_unrelenting_does_not_reward_expensive_skills_or_powers(self):
        state = self.state(with_payoff=False)
        state["deck"] = [
            {"card": "ENTRENCH", "count": 5}, {"card": "BARRICADE", "count": 5},
        ]
        for upgrades in (0, 1):
            with self.subTest(upgrades=upgrades):
                result = recommend_card_reward(
                    state, [{"card": "UNRELENTING", "upgrades": upgrades}],
                    self.repository,
                )
                self.assertEqual(result["profile"]["high_cost_ratio"], 1.0)
                row = result["recommendations"][0]
                self.assertNotIn("cost_reduction", row["effect_tags"])
                self.assertFalse(any(factor["code"] in {
                    "cost_reduction_value", "energy_curve_synergy",
                } for factor in row["factors"]))

    def test_unrelenting_public_policy_keeps_score_and_marks_coverage_low(self):
        request = DecisionRequest.create(
            decision_id="live-card", decision_type=CARD_REWARD,
            world=WorldState.create(
                run_id="live-mechanics", sequence=1, state=self.state(),
            ),
            candidates=[DecisionCandidate.create("reward:0", {"card": "无情猛攻"})],
            constraints={"can_skip": False},
        )
        result = CardRewardPolicy(self.repository).recommend(request)
        gap = "card_mechanic:UNRELENTING:next_attack_free"
        self.assertIn(gap, result.data_gaps)
        self.assertIn(gap, result.candidates[0].data_gaps)
        self.assertIsNotNone(result.candidates[0].score)
        self.assertEqual(result.confidence, "low")
        self.assertEqual(len([
            factor for factor in result.candidates[0].factors
            if factor["code"] == "unmodeled_mechanic" and factor["delta"] == 0.0
        ]), 1)

    def test_merchant_card_adaptation_preserves_scope_and_gap(self):
        for card, code in (("TRUE_GRIT", "estimate_scope"),
                           ("UNRELENTING", "unmodeled_mechanic")):
            for upgrades in (0, 1):
                with self.subTest(card=card, upgrades=upgrades):
                    request = DecisionRequest.create(
                        decision_id="merchant:live-card", decision_type=MERCHANT_CHOICE,
                        world=WorldState.create(
                            run_id="live-mechanics", sequence=1, state=self.state(),
                        ),
                        candidates=[DecisionCandidate.create("offer:0", {
                            "candidate_kind": "merchant_offer", "offer_kind": "card",
                            "entity_id": card, "upgrades": upgrades, "is_stocked": True,
                            "costs": [{"kind": "gold", "amount": 30, "resource_id": None}],
                        })],
                    )
                    result = MerchantPolicy(self.repository).recommend(request)
                    row = result.candidates[0]
                    direct = self.card_row(card, upgrades=upgrades)
                    expected = next(factor for factor in direct["factors"]
                                    if factor["code"] == code)
                    disclosures = [factor for factor in row.factors
                                   if factor["code"] == code]
                    self.assertEqual(len(disclosures), 1)
                    self.assertEqual(disclosures[0]["delta"], 0.0)
                    self.assertEqual(disclosures[0]["message"], expected["message"])
                    if card == "UNRELENTING":
                        # Merchant's existing safety policy blocks unknown
                        # effect value, while retaining the scored factors.
                        self.assertIsNone(row.score)
                        self.assertEqual(result.status, "uncertain")
                    else:
                        self.assertIsNotNone(row.score)
                    for gap in direct["data_gaps"]:
                        self.assertIn(gap, row.data_gaps)
                        self.assertIn(gap, result.data_gaps)

    def test_unchanged_catalog_rebuilds_v7_true_grit_tags(self):
        # Prepare a real old-tag catalog fixture; the public sync must notice
        # the tag version even though the original catalog hash is unchanged.
        repository = RelationalRepository(str(Path(self.temporary.name) / "old-tags.db"))
        catalog = str(ROOT / "data" / "knowledge.json")
        repository.sync_catalog(catalog)
        with repository.connect() as connection:
            connection.execute(
                "UPDATE schema_metadata SET value = '7' WHERE key = 'effect_tag_version'"
            )
            connection.execute(
                "DELETE FROM entity_effect_tags "
                "WHERE entity_key = 'cards:TRUE_GRIT' AND tag = 'mechanic:exhaust:provider'"
            )
        before = recommend_card_reward(
            self.state(), [{"card": "TRUE_GRIT"}], repository,
        )["recommendations"][0]
        self.assertNotIn("mechanic:exhaust:provider", before["effect_tags"])

        self.assertFalse(repository.sync_catalog(catalog))

        after = recommend_card_reward(
            self.state(), [{"card": "TRUE_GRIT"}], repository,
        )["recommendations"][0]
        synergy = [factor for factor in after["factors"]
                   if factor["code"] == "mechanic_synergy_provider:exhaust"]
        self.assertEqual(len(synergy), 1)
        self.assertEqual(synergy[0]["delta"], 2.1)
        self.assertNotIn("self_exhaust", after["effect_tags"])


if __name__ == "__main__":
    unittest.main()
