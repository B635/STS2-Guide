import math
import os
import tempfile
import unittest
from unittest.mock import patch

from advisor.decision_core import (
    DecisionCandidate,
    DecisionRequest,
    PolicyRegistry,
    WorldState,
)
from advisor.merchant import MERCHANT_CHOICE, MerchantPolicy
from advisor.scoring_preferences import ScoringPreferences
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _world(
    *,
    gold=120,
    deck=None,
    potions=(),
    max_potion_slots=3,
    hp=60,
    max_hp=80,
):
    return WorldState.create(
        run_id="merchant-run",
        sequence=3,
        state={
            "character": "IRONCLAD",
            "act": 1,
            "floor": 6,
            "hp": hp,
            "max_hp": max_hp,
            "gold": gold,
            "energy": 3,
            "deck": deck or [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
            ],
            "relics": [],
            "potions": list(potions),
            "max_potion_slots": max_potion_slots,
        },
    )


def _candidate(candidate_id, kind, *, entity_id=None, price=None, **extra):
    legacy_to_v9 = {
        "card_offer": "card",
        "relic_offer": "relic",
        "potion_offer": "potion",
        "card_remove_service": "card_removal",
    }
    if kind == "leave":
        payload = {
            "candidate_kind": "leave",
            "costs": [],
            **extra,
        }
    else:
        payload = {
            "candidate_kind": "merchant_offer",
            "offer_kind": legacy_to_v9.get(kind, kind),
            "is_stocked": True,
            **({"upgrades": 0} if legacy_to_v9.get(kind, kind) == "card" else {}),
            "costs": (
                [{"kind": "gold", "amount": price, "resource_id": None}]
                if price is not None
                else []
            ),
            **extra,
        }
    if entity_id is not None:
        payload["entity_id"] = entity_id
    return DecisionCandidate.create(
        candidate_id,
        payload,
        label=candidate_id,
    )


def _request(candidates, *, world=None):
    return DecisionRequest.create(
        decision_id="merchant:visit:1",
        decision_type=MERCHANT_CHOICE,
        world=world or _world(),
        candidates=candidates,
    )


class MerchantPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "merchant.db")
        )
        cls.repository.sync_catalog(
            os.path.join(ROOT, "data", "knowledge.json")
        )
        cls.policy = MerchantPolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_registry_dispatch_and_same_card_cheaper_is_better(self):
        request = _request([
            _candidate(
                "slot:0",
                "card_offer",
                entity_id="SHRUG_IT_OFF",
                price=30,
            ),
            _candidate(
                "slot:1",
                "card_offer",
                entity_id="SHRUG_IT_OFF",
                price=90,
            ),
            _candidate("leave", "leave"),
        ])
        result = PolicyRegistry([self.policy]).recommend(request)
        by_id = {row.candidate_id: row for row in result.candidates}
        self.assertGreater(by_id["slot:0"].score, by_id["slot:1"].score)
        self.assertEqual(result.recommended_candidate_id, "slot:0")

    def test_unknown_eligible_offer_forces_uncertain(self):
        result = self.policy.recommend(_request([
            _candidate(
                "unknown",
                "relic_offer",
                entity_id="NOT_A_RELIC",
                price=10,
            ),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIn("unknown_relic:NOT_A_RELIC", result.data_gaps)

    def test_unknown_ineligible_offer_does_not_block(self):
        unknown = DecisionCandidate.create(
            "sold",
            {
                "candidate_kind": "merchant_offer",
                "offer_kind": "relic",
                "is_stocked": False,
                "entity_id": "NOT_A_RELIC",
                "costs": [{
                    "kind": "gold",
                    "amount": 10,
                    "resource_id": None,
                }],
            },
            label="sold",
            eligible=False,
        )
        result = self.policy.recommend(_request([
            unknown,
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "recommend")
        self.assertEqual(result.recommended_candidate_id, "leave")

    def test_missing_actual_price_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate(
                "card",
                "card_offer",
                entity_id="SHRUG_IT_OFF",
            ),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("merchant:costs_invalid")
            for gap in result.data_gaps
        ))

    def test_non_gold_costs_are_not_silently_ignored(self):
        for cost in (
            {"kind": "hp", "amount": 100, "resource_id": None},
            {"kind": "max_hp", "amount": 10, "resource_id": None},
            {"kind": "resource", "amount": 1, "resource_id": "fixture"},
        ):
            with self.subTest(cost=cost):
                result = self.policy.recommend(_request([
                    _candidate(
                        "relic",
                        "relic_offer",
                        entity_id="SHURIKEN",
                        costs=[
                            {"kind": "gold", "amount": 0, "resource_id": None},
                            cost,
                        ],
                    ),
                    _candidate("leave", "leave"),
                ]))
                self.assertEqual(result.status, "uncertain")
                self.assertIsNone(result.recommended_candidate_id)
                self.assertIsNone(result.candidates[0].score)
                self.assertTrue(any(
                    gap.startswith("merchant:costs_invalid:")
                    for gap in result.data_gaps
                ))

    def test_known_free_offer_with_only_gold_cost_remains_scoreable(self):
        result = self.policy.recommend(_request([
            _candidate(
                "relic",
                "relic_offer",
                entity_id="SHURIKEN",
                price=0,
            ),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.recommended_candidate_id, "relic")
        self.assertIsNotNone(result.candidates[0].score)
        self.assertEqual(result.data_gaps, ())

    def test_affordability_inconsistency_fails_closed(self):
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "card",
                    "card_offer",
                    entity_id="SHRUG_IT_OFF",
                    price=200,
                ),
                _candidate("leave", "leave"),
            ],
            world=_world(gold=50),
        ))
        self.assertEqual(result.status, "uncertain")
        self.assertIn(
            "resource:eligibility_gold_inconsistent",
            result.data_gaps,
        )

    def test_full_potion_slots_require_explicit_replacement_support(self):
        full = _world(
            potions=(
                {"potion": "STRENGTH_POTION", "slot": 0},
                {"potion": "REGEN_POTION", "slot": 1},
                {"potion": "GLOWWATER_POTION", "slot": 2},
            )
        )
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "potion",
                    "potion_offer",
                    entity_id="STRENGTH_POTION",
                    price=20,
                ),
                _candidate("leave", "leave"),
            ],
            world=full,
        ))
        self.assertEqual(result.status, "uncertain")
        self.assertIn(
            "resource:potion_replacement_unresolved",
            result.data_gaps,
        )

    def test_remove_status_card_can_beat_leaving(self):
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "remove",
                    "card_remove_service",
                    price=30,
                    deck_edit_targets={
                        "operation": "remove", "source_code": "fixture:merchant_removal",
                        "targets": [{"target_id": "remove:0", "card": "WOUND", "upgrades": 0}],
                    },
                ),
                _candidate("leave", "leave"),
            ],
            world=_world(deck=[
                {"card": "WOUND", "count": 1},
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
            ]),
        ))
        self.assertEqual(result.recommended_candidate_id, "remove")

    def test_unknown_offer_kind_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate("mystery", "mystery_offer", price=0),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("merchant:unknown_offer_kind")
            for gap in result.data_gaps
        ))

    def test_scores_are_finite_and_bounded(self):
        result = self.policy.recommend(_request([
            _candidate(
                "relic",
                "relic_offer",
                entity_id="SHURIKEN",
                price=50,
            ),
            _candidate("leave", "leave"),
        ]))
        for row in result.candidates:
            if row.score is not None:
                self.assertTrue(math.isfinite(row.score))
                self.assertGreaterEqual(row.score, 0)
                self.assertLessEqual(row.score, 100)

    def test_actual_upgrade_state_is_used_without_defaulting_missing(self):
        candidates = [
            _candidate("base", "card_offer", entity_id="SHRUG_IT_OFF", price=20, upgrades=0),
            _candidate("upgraded", "card_offer", entity_id="SHRUG_IT_OFF", price=20, upgrades=1),
        ]
        result = self.policy.recommend(_request(candidates))
        self.assertGreater(result.candidates[1].score, result.candidates[0].score)
        for value in (None, True, "0", -1, 10):
            with self.subTest(value=value):
                candidate = _candidate("unknown", "card_offer", entity_id="SHRUG_IT_OFF", price=20, upgrades=value)
                result = self.policy.recommend(_request([candidate]))
                self.assertIsNone(result.candidates[0].score)
                self.assertIn("merchant:card_upgrades_unknown", result.data_gaps)

    def test_card_estimate_scope_survives_merchant_without_adding_score(self):
        result = self.policy.recommend(_request([
            _candidate("card", "card_offer", entity_id="PERFECTED_STRIKE", price=30),
            _candidate("leave", "leave"),
        ]))
        card = result.candidates[0]
        factors = {factor["code"]: factor for factor in card.factors}
        # Five existing Strikes plus this offer give the 18-damage baseline.
        # The existing inner state_score=51.65 contributes 0.9075 to merchant.
        self.assertEqual(card.score, 54.9075)
        self.assertEqual(factors["merchant_card_context_fit"]["delta"], 0.9075)
        self.assertEqual(set(factors), {
            "merchant_card_access", "merchant_card_context_fit",
            "gold_liquidity_cost", "estimate_scope",
        })
        self.assertEqual(factors["estimate_scope"]["delta"], 0.0)
        self.assertIn("战前基准为 18 点伤害", factors["estimate_scope"]["message"])
        self.assertIn("不是当回合实伤", factors["estimate_scope"]["message"])
        self.assertEqual(result.data_gaps, ())
        self.assertEqual(result.recommended_candidate_id, "card")

    def test_defensive_relic_uses_actual_tag_hp_and_deck_need(self):
        offer = _candidate("anchor", "relic_offer", entity_id="ANCHOR", price=20)
        healthy = self.policy.recommend(_request([offer], world=_world(hp=80)))
        injured = self.policy.recommend(_request([offer], world=_world(hp=20)))
        self.assertGreater(injured.candidates[0].score, healthy.candidates[0].score)
        self.assertIn("merchant_relic_survival", {f["code"] for f in injured.candidates[0].factors})
        lacking = self.policy.recommend(_request([offer], world=_world(deck=[{"card": "STRIKE_IRONCLAD", "count": 8}])))
        defended = self.policy.recommend(_request([offer], world=_world(deck=[{"card": "DEFEND_IRONCLAD", "count": 8}])))
        self.assertGreater(lacking.candidates[0].score, defended.candidates[0].score)

    def test_unmodeled_potion_is_not_a_precise_purchase_recommendation(self):
        result = self.policy.recommend(_request([
            _candidate("potion", "potion_offer", entity_id="STRENGTH_POTION", price=0),
            _candidate("leave", "leave"),
        ]))
        self.assertIsNone(result.candidates[0].score)
        self.assertIn("merchant:potion_mechanism_unmodeled:STRENGTH_POTION", result.data_gaps)
        self.assertIsNone(result.recommended_candidate_id)

    def test_identity_without_relic_effects_is_not_full_mechanism_coverage(self):
        offer = _candidate("relic", "relic_offer", entity_id="SHURIKEN", price=0)
        with patch.object(self.repository, "find_relic", return_value={"id": "SHURIKEN", "_effect_tags": {}}):
            result = self.policy.recommend(_request([offer]))
        self.assertIsNone(result.candidates[0].score)
        self.assertIn("merchant:relic_mechanism_unmodeled:SHURIKEN", result.data_gaps)

    def test_preferences_apply_once_and_leave_price_and_internal_card_fit_unchanged(self):
        request = _request([
            _candidate("card", "card_offer", entity_id="SHRUG_IT_OFF", price=30, upgrades=1),
            _candidate("leave", "leave"),
        ])
        baseline = self.policy.recommend(request)
        configured = MerchantPolicy(self.repository, ScoringPreferences(survival=1.25, long_term_growth=1.25)).recommend(request)
        before = {f["code"]: f["delta"] for f in baseline.candidates[0].factors}
        after = {f["code"]: f["delta"] for f in configured.candidates[0].factors}
        self.assertEqual(after["merchant_card_access"], before["merchant_card_access"] * 1.25)
        self.assertEqual(after["merchant_card_context_fit"], before["merchant_card_context_fit"])
        self.assertEqual(after["gold_liquidity_cost"], before["gold_liquidity_cost"])
        self.assertEqual(configured.candidates[1].score, baseline.candidates[1].score)


if __name__ == "__main__":
    unittest.main()
