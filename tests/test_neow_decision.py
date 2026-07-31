import os
import tempfile
import unittest

from advisor.decision_core import (
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from advisor.neow import NEOW_BLESSING, NeowPolicy
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTERS = {
    "IRONCLAD": ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD", 80),
    "SILENT": ("STRIKE_SILENT", "DEFEND_SILENT", 70),
    "DEFECT": ("STRIKE_DEFECT", "DEFEND_DEFECT", 75),
    "REGENT": ("STRIKE_REGENT", "DEFEND_REGENT", 75),
    "NECROBINDER": (
        "STRIKE_NECROBINDER",
        "DEFEND_NECROBINDER",
        66,
    ),
}


def _world(
    character="IRONCLAD",
    *,
    hp=None,
    potions=(),
    max_potion_slots=3,
):
    strike, defend, default_hp = STARTERS[character]
    hp = default_hp if hp is None else hp
    return WorldState.create(
        run_id=f"neow-{character.lower()}",
        sequence=1,
        state={
            "character": character,
            "act": 1,
            "floor": 0,
            "hp": hp,
            "max_hp": default_hp,
            "gold": 99,
            "energy": 3,
            "deck": [
                {"card": strike, "count": 5},
                {"card": defend, "count": 4},
            ],
            "relics": [],
            "potions": list(potions),
            "max_potion_slots": max_potion_slots,
        },
    )


def _candidate(candidate_id, effects, **extra):
    return DecisionCandidate.create(
        candidate_id,
        {
            "candidate_kind": "neow_blessing",
            "blessing_id": extra.pop("blessing_id", candidate_id),
            "stage_id": extra.pop("stage_id", "initial"),
            "costs": [],
            "effects": effects,
            **extra,
        },
        label=candidate_id,
    )


def _request(candidates, *, world=None):
    return DecisionRequest.create(
        decision_id="neow:run:initial",
        decision_type=NEOW_BLESSING,
        world=world or _world(),
        candidates=candidates,
    )


def _effect(kind, **values):
    return {
        "kind": kind,
        "source_code": "catalog:neow",
        **values,
    }


class NeowPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "neow.db")
        )
        cls.repository.sync_catalog(
            os.path.join(ROOT, "data", "knowledge.json")
        )
        cls.policy = NeowPolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_exact_gold_blessing_beats_no_op(self):
        result = self.policy.recommend(_request([
            _candidate("gold", [_effect("gold_delta", amount=100)]),
            _candidate("none", [_effect("no_op")]),
        ]))
        self.assertEqual(result.recommended_candidate_id, "gold")

    def test_hp_cost_below_safety_floor_does_not_beat_no_op(self):
        result = self.policy.recommend(_request(
            [
                _candidate("hurt", [_effect("hp_delta", amount=-30)]),
                _candidate("none", [_effect("no_op")]),
            ],
            world=_world(hp=35),
        ))
        self.assertEqual(result.recommended_candidate_id, "none")

    def test_unknown_card_reward_forces_uncertain(self):
        result = self.policy.recommend(_request([
            _candidate("card", [_effect(
                "add_card",
                entity_type="cards",
                entity_id="NOT_A_CARD",
                target_mode="specific",
            )]),
            _candidate("none", [_effect("no_op")]),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIn("unknown_card:NOT_A_CARD", result.data_gaps)

    def test_bounded_random_effect_is_not_given_fake_expectation(self):
        result = self.policy.recommend(_request([
            _candidate("random", [_effect(
                "gold_delta",
                certainty="bounded",
                min_amount=50,
                max_amount=200,
            )]),
            _candidate("none", [_effect("no_op")]),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("effect:gold_delta:bounded")
            for gap in result.data_gaps
        ))

    def test_transform_without_exact_outcome_is_unknown(self):
        result = self.policy.recommend(_request([
            _candidate("transform", [_effect(
                "transform_card",
                entity_type="cards",
                target_mode="choose",
            )]),
            _candidate("none", [_effect("no_op")]),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIn(
            "deck_edit:transform_outcome_unknown",
            result.data_gaps,
        )

    def test_full_potion_slots_need_followup_resolution(self):
        world = _world(
            potions=(
                {"potion": "STRENGTH_POTION", "slot": 0},
                {"potion": "REGEN_POTION", "slot": 1},
                {"potion": "GLOWWATER_POTION", "slot": 2},
            )
        )
        result = self.policy.recommend(_request(
            [
                _candidate("potion", [_effect(
                    "add_potion",
                    entity_type="potions",
                    entity_id="STRENGTH_POTION",
                    target_mode="specific",
                )]),
                _candidate("none", [_effect("no_op")]),
            ],
            world=world,
        ))
        self.assertEqual(result.status, "uncertain")
        self.assertIn(
            "resource:potion_replacement_unresolved",
            result.data_gaps,
        )

    def test_missing_stage_identity_fails_closed(self):
        candidate = DecisionCandidate.create(
            "broken",
            {
                "candidate_kind": "neow_blessing",
                "blessing_id": "BLESSING",
                "costs": [],
                "effects": [_effect("no_op")],
            },
            label="broken",
        )
        result = self.policy.recommend(_request([candidate]))
        self.assertEqual(result.status, "uncertain")
        self.assertIn("neow:stage_id_missing", result.data_gaps)

    def test_shared_resource_rule_supports_all_five_characters(self):
        for character in STARTERS:
            with self.subTest(character=character):
                result = self.policy.recommend(_request(
                    [
                        _candidate(
                            "gold",
                            [_effect("gold_delta", amount=100)],
                        ),
                        _candidate("none", [_effect("no_op")]),
                    ],
                    world=_world(character),
                ))
                self.assertEqual(result.recommended_candidate_id, "gold")

    def test_same_input_is_deterministic(self):
        request = _request([
            _candidate("relic", [_effect(
                "add_relic",
                entity_type="relics",
                entity_id="SHURIKEN",
                target_mode="specific",
            )]),
            _candidate("none", [_effect("no_op")]),
        ])
        self.assertEqual(
            self.policy.recommend(request).as_dict(),
            self.policy.recommend(request).as_dict(),
        )


if __name__ == "__main__":
    unittest.main()
