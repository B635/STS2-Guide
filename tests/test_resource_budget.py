import math
import unittest

from advisor.choice_effects import (
    AssessmentDraft,
    ChoiceEffect,
    EffectBundle,
    finalize_recommendation,
)
from advisor.decision_core import (
    RECOMMENDATION_DIMENSIONS,
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from advisor.resource_budget import (
    build_resource_budget,
    score_price,
    score_resource_effects,
)


def _world(
    *,
    hp=60,
    max_hp=80,
    gold=100,
    potions=(),
    max_potion_slots=3,
    route_mode="balanced",
    map_context=None,
):
    return WorldState.create(
        run_id="resource-run",
        sequence=1,
        route_mode=route_mode,
        state={
            "character": "IRONCLAD",
            "act": 1,
            "floor": 4,
            "hp": hp,
            "max_hp": max_hp,
            "gold": gold,
            "energy": 3,
            "deck": [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
            ],
            "relics": [],
            "potions": list(potions),
            "max_potion_slots": max_potion_slots,
        },
        map_context=map_context,
    )


def _route():
    return {
        "nodes": [
            {
                "node_id": "1:0",
                "kind": "ELITE",
                "row": 1,
                "col": 0,
                "edges": ["2:0"],
            },
            {
                "node_id": "2:0",
                "kind": "MONSTER",
                "row": 2,
                "col": 0,
                "edges": [],
            },
        ],
        "available_next_node_ids": ["1:0"],
    }


class ResourceBudgetTests(unittest.TestCase):
    def test_route_pressure_and_survival_mode_raise_hp_floor(self):
        plain = build_resource_budget(_world())
        dangerous = build_resource_budget(_world(map_context=_route()))
        survival = build_resource_budget(_world(
            map_context=_route(),
            route_mode="survival",
        ))
        self.assertGreater(
            dangerous.health.safety_floor,
            plain.health.safety_floor,
        )
        self.assertGreater(
            survival.health.safety_floor,
            dangerous.health.safety_floor,
        )

    def test_growth_mode_never_lowers_base_hp_floor(self):
        balanced = build_resource_budget(_world())
        growth = build_resource_budget(_world(route_mode="growth"))
        self.assertEqual(
            growth.health.safety_floor,
            balanced.health.safety_floor,
        )

    def test_potion_capacity_uses_real_slots(self):
        budget = build_resource_budget(_world(
            potions=({"potion": "STRENGTH_POTION", "slot": 1},),
            max_potion_slots=3,
        ))
        self.assertTrue(budget.potions.known)
        self.assertEqual(budget.potions.free_slots, 2)

    def test_duplicate_potion_slots_are_unknown(self):
        budget = build_resource_budget(_world(
            potions=(
                {"potion": "STRENGTH_POTION", "slot": 0},
                {"potion": "REGEN_POTION", "slot": 0},
            ),
        ))
        self.assertFalse(budget.potions.known)
        self.assertIsNone(budget.potions.free_slots)

    def test_price_uses_actual_gold_and_missing_gold_blocks(self):
        candidate = DecisionCandidate.create("buy", label="购买")
        known = AssessmentDraft(candidate)
        score_price(known, build_resource_budget(_world()), 40)
        self.assertIsNotNone(known.score)
        self.assertIn(
            "gold_liquidity_cost",
            {item["code"] for item in known.factors},
        )

        missing = AssessmentDraft(candidate)
        score_price(
            missing,
            build_resource_budget(_world(gold=None)),
            40,
        )
        self.assertTrue(missing.blocking_unknown)
        self.assertIn("resource:gold_missing", missing.data_gaps)

    def test_hp_cost_below_floor_receives_hard_penalty(self):
        candidate = DecisionCandidate.create("hurt", label="受伤")
        draft = AssessmentDraft(candidate)
        bundle = EffectBundle.from_payload({
            "effects": [{
                "kind": "hp_delta",
                "amount": -20,
                "source_code": "fixture",
            }],
        })
        score_resource_effects(
            draft,
            build_resource_budget(_world(hp=30)),
            bundle,
        )
        self.assertTrue(draft.safety_violation)
        self.assertIn(
            "hp_safety_floor_violation",
            {item["code"] for item in draft.factors},
        )
        self.assertEqual(draft.score, 0.0)

    def test_unknown_eligible_candidate_forces_uncertain(self):
        world = _world()
        request = DecisionRequest.create(
            decision_id="resource-decision",
            decision_type="resource_test",
            world=world,
            candidates=[
                DecisionCandidate.create("known", label="已知"),
                DecisionCandidate.create("unknown", label="未知"),
            ],
        )
        known = AssessmentDraft(request.candidates[0])
        unknown = AssessmentDraft(request.candidates[1])
        unknown.gap("fixture:unknown")
        result = finalize_recommendation(
            request,
            [known, unknown],
            policy_version="fixture_v1",
        )
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIn("fixture:unknown", result.data_gaps)

    def test_safety_floor_candidates_are_never_recommended(self):
        world = _world(hp=20)
        request = DecisionRequest.create(
            decision_id="resource-safety",
            decision_type="resource_test",
            world=world,
            candidates=[
                DecisionCandidate.create("unsafe", label="危险"),
                DecisionCandidate.create("safe", label="安全"),
            ],
        )
        unsafe = AssessmentDraft(request.candidates[0])
        unsafe.safety_violation = True
        unsafe.add("tempting", 90, "高收益但致命。", "long_term_growth")
        safe = AssessmentDraft(request.candidates[1])
        safe.add("safe", 1, "安全选择。", "survival")

        result = finalize_recommendation(
            request,
            [unsafe, safe],
            policy_version="fixture_v1",
        )
        self.assertEqual(result.recommended_candidate_id, "safe")
        self.assertIsNone(result.candidates[0].rank)

        only_unsafe_request = DecisionRequest.create(
            decision_id="resource-only-unsafe",
            decision_type="resource_test",
            world=world,
            candidates=[DecisionCandidate.create("unsafe", label="危险")],
        )
        only_unsafe = AssessmentDraft(only_unsafe_request.candidates[0])
        only_unsafe.safety_violation = True
        only_unsafe.add("tempting", 90, "高收益但致命。", "long_term_growth")
        uncertain = finalize_recommendation(
            only_unsafe_request,
            [only_unsafe],
            policy_version="fixture_v1",
        )
        self.assertEqual(uncertain.status, "uncertain")
        self.assertIsNone(uncertain.recommended_candidate_id)
        self.assertIsNone(uncertain.candidates[0].rank)

    def test_effect_contract_rejects_wrong_entity_type_and_fractional_hp(self):
        with self.assertRaisesRegex(ValueError, "entity_type=cards"):
            ChoiceEffect.create({
                "kind": "add_card",
                "entity_type": "relics",
                "entity_id": "SHRUG_IT_OFF",
                "target_mode": "specific",
                "source_code": "fixture",
            })
        with self.assertRaisesRegex(ValueError, "integer amounts"):
            ChoiceEffect.create({
                "kind": "hp_delta",
                "amount": 1.5,
                "source_code": "fixture",
            })

    def test_all_eight_dimensions_and_scores_are_finite(self):
        world = _world()
        request = DecisionRequest.create(
            decision_id="resource-dimensions",
            decision_type="resource_test",
            world=world,
            candidates=[DecisionCandidate.create("known", label="已知")],
        )
        draft = AssessmentDraft(request.candidates[0])
        draft.add("test", 500.0, "测试上界。", "immediate_power")
        result = finalize_recommendation(
            request,
            [draft],
            policy_version="fixture_v1",
        )
        row = result.candidates[0]
        self.assertEqual(set(row.dimensions), set(RECOMMENDATION_DIMENSIONS))
        self.assertTrue(math.isfinite(row.score))
        self.assertEqual(row.score, 100.0)


if __name__ == "__main__":
    unittest.main()
