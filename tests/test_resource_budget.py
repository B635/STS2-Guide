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
    def _score_bundle(self, effects, *, world=None, replacement=False):
        draft = AssessmentDraft(DecisionCandidate.create("bundle", label="资源变化"))
        budget = build_resource_budget(world or _world())
        score_resource_effects(
            draft,
            budget,
            EffectBundle.from_payload({"effects": [
                {"source_code": "fixture:compound", **effect}
                for effect in effects
            ]}),
            replacement_supported=replacement,
        )
        return draft

    def test_split_hp_costs_cannot_reuse_original_health(self):
        effects = [{"kind": "hp_delta", "amount": -15}] * 2
        split = self._score_bundle(effects, world=_world(hp=50))
        combined = self._score_bundle(
            [{"kind": "hp_delta", "amount": -30}], world=_world(hp=50)
        )
        self.assertTrue(split.safety_violation)
        self.assertEqual(split.score, combined.score)
        self.assertEqual(split.factors, combined.factors)

    def test_split_healing_cannot_reuse_missing_health(self):
        split = self._score_bundle(
            [{"kind": "hp_delta", "amount": 10}] * 2, world=_world(hp=75)
        )
        combined = self._score_bundle(
            [{"kind": "hp_delta", "amount": 20}], world=_world(hp=75)
        )
        self.assertEqual(split.score, combined.score)
        self.assertEqual(len(split.factors), 1)

    def test_split_gold_costs_cannot_reuse_balance(self):
        draft = self._score_bundle(
            [{"kind": "gold_delta", "amount": -60}] * 2
        )
        self.assertIsNone(draft.score)
        self.assertIn("resource:gold_effect_infeasible", draft.data_gaps)

    def test_unordered_payment_and_reward_are_not_netted(self):
        for kind in ("hp_delta", "gold_delta", "max_hp_delta"):
            with self.subTest(kind=kind):
                effects = [
                    {"kind": kind, "amount": -100},
                    {"kind": kind, "amount": 120},
                ]
                first = self._score_bundle(effects)
                second = self._score_bundle(list(reversed(effects)))
                self.assertIsNone(first.score)
                self.assertIn(f"resource:{kind}:order_unmodeled", first.data_gaps)
                self.assertEqual(first.data_gaps, second.data_gaps)

    def test_coupled_health_effects_do_not_guess_execution_order(self):
        draft = self._score_bundle([
            {"kind": "max_hp_delta", "amount": 10},
            {"kind": "hp_delta", "amount": 10},
        ])
        self.assertIsNone(draft.score)
        self.assertIn("resource:coupled_health_effects_unmodeled", draft.data_gaps)

    def test_max_hp_loss_checks_feasibility_and_safety(self):
        impossible = self._score_bundle([{"kind": "max_hp_delta", "amount": -80}])
        self.assertIsNone(impossible.score)
        self.assertIn("resource:max_hp_effect_infeasible", impossible.data_gaps)
        unsafe = self._score_bundle([{"kind": "max_hp_delta", "amount": -70}])
        self.assertTrue(unsafe.safety_violation)

    def test_potion_slot_outside_capacity_is_unknown(self):
        for slot, capacity in ((3, 3), (0, 0)):
            budget = build_resource_budget(_world(
                potions=({"potion": "STRENGTH_POTION", "slot": slot},),
                max_potion_slots=capacity,
            ))
            self.assertFalse(budget.potions.known)
            self.assertIsNone(budget.potions.free_slots)

    def test_multiple_potions_cannot_share_one_free_slot(self):
        effect = {"kind": "add_potion", "entity_type": "potions",
                  "entity_id": "STRENGTH_POTION", "target_mode": "specific"}
        draft = self._score_bundle([effect, effect], world=_world(max_potion_slots=1))
        self.assertIsNone(draft.score)
        self.assertIn("resource:potion_capacity_exceeded", draft.data_gaps)

    def test_replacement_cannot_claim_more_potions_than_total_capacity(self):
        effect = {"kind": "add_potion", "entity_type": "potions",
                  "entity_id": "STRENGTH_POTION", "target_mode": "specific"}
        draft = self._score_bundle(
            [effect, effect], world=_world(max_potion_slots=1), replacement=True
        )
        self.assertIsNone(draft.score)

    def test_multiple_potions_can_use_distinct_free_slots(self):
        effect = {"kind": "add_potion", "entity_type": "potions",
                  "entity_id": "STRENGTH_POTION", "target_mode": "specific"}
        draft = self._score_bundle([effect, effect], world=_world(max_potion_slots=2))
        self.assertIsNotNone(draft.score)
        self.assertFalse(draft.data_gaps)

    def test_potion_removal_uses_each_inventory_item_once(self):
        effect = {"kind": "remove_potion", "entity_type": "potions",
                  "entity_id": "STRENGTH_POTION", "target_mode": "specific"}
        world = _world(potions=({"potion": "STRENGTH_POTION", "slot": 0},))
        single = self._score_bundle([effect], world=world)
        duplicate = self._score_bundle([effect, effect], world=world)
        self.assertIsNotNone(single.score)
        self.assertFalse(single.data_gaps)
        self.assertIsNone(duplicate.score)
        self.assertIn("resource:potion_remove_target_missing", duplicate.data_gaps)

    def test_add_and_remove_potion_does_not_guess_replacement_order(self):
        effects = [
            {"kind": kind, "entity_type": "potions",
             "entity_id": "STRENGTH_POTION", "target_mode": "specific"}
            for kind in ("add_potion", "remove_potion")
        ]
        world = _world(potions=({"potion": "STRENGTH_POTION", "slot": 0},))
        first = self._score_bundle(effects, world=world)
        reverse = self._score_bundle(list(reversed(effects)), world=world)
        self.assertIsNone(first.score)
        self.assertIn("resource:potion_order_unmodeled", first.data_gaps)
        self.assertEqual(first.data_gaps, reverse.data_gaps)

    def test_unknown_potion_removal_does_not_claim_known_inventory(self):
        effect = {"kind": "remove_potion", "entity_type": "potions",
                  "entity_id": "STRENGTH_POTION", "target_mode": "specific"}
        draft = self._score_bundle([effect])
        self.assertIsNone(draft.score)
        self.assertIn("resource:potion_remove_target_missing", draft.data_gaps)

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
