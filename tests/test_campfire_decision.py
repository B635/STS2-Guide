import os
import tempfile
import unittest
from unittest.mock import patch

from advisor.campfire import CAMPFIRE_ACTION, CampfirePolicy
from advisor.decision_core import (
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _world(*, hp=60, deck=None, route_mode="balanced", route=False):
    map_context = None
    if route:
        map_context = {
            "nodes": [{
                "node_id": "1:0",
                "kind": "ELITE",
                "row": 1,
                "col": 0,
                "edges": [],
            }],
            "available_next_node_ids": ["1:0"],
        }
    return WorldState.create(
        run_id="campfire-run",
        sequence=4,
        route_mode=route_mode,
        state={
            "character": "IRONCLAD",
            "act": 1,
            "floor": 7,
            "hp": hp,
            "max_hp": 80,
            "gold": 99,
            "energy": 3,
            "deck": deck or [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
            ],
            "relics": [],
            "potions": [],
            "max_potion_slots": 3,
        },
        map_context=map_context,
    )


def _candidate(candidate_id, action_kind, effects=None, **extra):
    if action_kind == "leave":
        payload = {
            "candidate_kind": "leave",
            "costs": [],
            **extra,
        }
    else:
        payload = {
            "candidate_kind": "rest_action",
            "action_id": action_kind,
            "costs": [],
            **extra,
        }
    if action_kind == "smith" and effects is None:
        effects = [{
            "kind": "upgrade_card",
            "entity_type": "cards",
            "target_mode": "choose",
            "source_code": "game_api:campfire_smith",
        }]
    if effects is not None:
        payload["effects"] = effects
    return DecisionCandidate.create(
        candidate_id,
        payload,
        label=candidate_id,
    )


def _request(candidates, *, world=None):
    world = world or _world()
    # These positive fixtures model a normal single Smith screen explicitly.
    # Missing/unknown eligibility is exercised independently in target tests.
    verified = []
    for candidate in candidates:
        payload = dict(candidate.payload)
        if payload.get("action_id") == "smith" and "deck_edit_targets" not in payload:
            targets = []
            for entry in world.deck:
                if entry.upgrades == 0:
                    for _ in range(entry.count):
                        targets.append({"target_id": f"fixture:{len(targets)}",
                                        "card": entry.card, "upgrades": entry.upgrades})
            payload["deck_edit_targets"] = {
                "operation": "upgrade", "source_code": "fixture:single_smith", "targets": targets,
            }
        verified.append(DecisionCandidate.create(candidate.candidate_id, payload,
                                                label=candidate.label, eligible=candidate.eligible))
    return DecisionRequest.create(
        decision_id="campfire:visit:1",
        decision_type=CAMPFIRE_ACTION,
        world=world,
        candidates=verified,
    )


def _heal(amount):
    return [{
        "kind": "hp_delta",
        "amount": amount,
        "source_code": "game_api:campfire_heal",
    }]


class CampfirePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "campfire.db")
        )
        cls.repository.sync_catalog(
            os.path.join(ROOT, "data", "knowledge.json")
        )
        cls.policy = CampfirePolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_low_hp_dangerous_route_prefers_rest(self):
        result = self.policy.recommend(_request(
            [
                _candidate("rest", "rest", _heal(24)),
                _candidate("smith", "smith"),
            ],
            world=_world(hp=20, route=True),
        ))
        self.assertEqual(result.recommended_candidate_id, "rest")

    def test_full_hp_prefers_known_upgrade_over_wasted_heal(self):
        result = self.policy.recommend(_request(
            [
                _candidate("rest", "rest", _heal(24)),
                _candidate("smith", "smith"),
            ],
            world=_world(hp=80),
        ))
        self.assertEqual(result.recommended_candidate_id, "smith")

    def test_healing_is_capped_by_missing_hp(self):
        small = self.policy.recommend(_request(
            [_candidate("rest", "rest", _heal(5))],
            world=_world(hp=75),
        ))
        large = self.policy.recommend(_request(
            [_candidate("rest", "rest", _heal(50))],
            world=_world(hp=75),
        ))
        self.assertEqual(
            small.candidates[0].score,
            large.candidates[0].score,
        )

    def test_smith_without_structured_upgrade_target_is_unknown(self):
        result = self.policy.recommend(_request(
            [_candidate("smith", "smith")],
            world=_world(deck=[{
                "card": "STRIKE_IRONCLAD",
                "upgrades": 1,
            }]),
        ))
        self.assertEqual(result.status, "uncertain")
        self.assertIn("deck_edit:no_upgrade_target", result.data_gaps)

    def test_smith_does_not_discard_unmodeled_upgrade_targets(self):
        for deck in (
            [{"card": "STRIKE_IRONCLAD"}, {"card": "INFLAME"}],
            [{"card": "INFLAME"}, {"card": "STRIKE_IRONCLAD"}],
            [{"card": "STRIKE_IRONCLAD"}, {"card": "BASH"}],
        ):
            with self.subTest(deck=deck):
                result = self.policy.recommend(_request(
                    [
                        _candidate("rest", "rest", _heal(24)),
                        _candidate("smith", "smith"),
                    ],
                    world=_world(hp=80, deck=deck),
                ))
                self.assertEqual(result.status, "uncertain")
                self.assertIsNone(result.recommended_candidate_id)
                smith = result.candidates[1]
                self.assertIsNone(smith.score)
                self.assertIn("deck_edit:upgrade_delta_unknown", smith.data_gaps)
                self.assertFalse(any(
                    factor["code"] == "deck_upgrade_opportunity"
                    for factor in smith.factors
                ))

    def test_already_upgraded_unmodeled_card_does_not_block_smith(self):
        result = self.policy.recommend(_request(
            [
                _candidate("rest", "rest", _heal(24)),
                _candidate("smith", "smith"),
            ],
            world=_world(hp=80, deck=[
                {"card": "STRIKE_IRONCLAD"},
                {"card": "INFLAME", "upgrades": 1},
            ]),
        ))
        self.assertEqual(result.recommended_candidate_id, "smith")
        self.assertEqual(result.data_gaps, ())

    def test_known_cost_upgrade_remains_scoreable(self):
        result = self.policy.recommend(_request(
            [
                _candidate("rest", "rest", _heal(24)),
                _candidate("smith", "smith"),
            ],
            world=_world(hp=80, deck=[{"card": "BODY_SLAM"}]),
        ))
        self.assertEqual(result.recommended_candidate_id, "smith")
        self.assertEqual(result.data_gaps, ())

    def test_smith_respects_signed_and_absolute_upgrade_costs(self):
        original = self.repository.find_card("BASH")
        for cost, expected in (
            ("+1", "rest"),
            ("-1", "smith"),
            (1, "smith"),
            (0, "smith"),
            (3, "rest"),
        ):
            card = {**original, "cost": 2, "upgrade": {"cost": cost}}
            with self.subTest(cost=cost), patch.object(
                self.repository, "find_card", return_value=card
            ):
                result = self.policy.recommend(_request(
                    [
                        _candidate("rest", "rest", _heal(1)),
                        _candidate("smith", "smith"),
                    ],
                    world=_world(hp=79, deck=[{"card": "BASH"}]),
                ))
                self.assertEqual(result.recommended_candidate_id, expected)
                self.assertEqual(result.data_gaps, ())

    def test_unknown_special_action_without_effects_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate(
                "special",
                "special",
                description="获得神秘奖励。",
            ),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("campfire:effects_invalid")
            for gap in result.data_gaps
        ))

    def test_structured_special_action_is_scoreable(self):
        result = self.policy.recommend(_request([
            _candidate("special", "special", [{
                "kind": "gold_delta",
                "amount": 120,
                "source_code": "catalog:campfire_action",
            }]),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "recommend")
        self.assertEqual(result.recommended_candidate_id, "special")

    def test_hp_safety_floor_prevents_harmful_special(self):
        result = self.policy.recommend(_request(
            [
                _candidate("harm", "special", [{
                    "kind": "hp_delta",
                    "amount": -30,
                    "source_code": "catalog:campfire_action",
                }]),
                _candidate("leave", "leave"),
            ],
            world=_world(hp=35, route=True),
        ))
        self.assertEqual(result.recommended_candidate_id, "leave")
        harmful = next(
            row for row in result.candidates if row.candidate_id == "harm"
        )
        self.assertIsNone(harmful.score)
        self.assertIsNone(harmful.rank)

    def test_unknown_action_id_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate("dig", "unverified_action"),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("campfire:effects_invalid")
            for gap in result.data_gaps
        ))

    def test_same_input_is_deterministic(self):
        request = _request([
            _candidate("rest", "rest", _heal(20)),
            _candidate("smith", "smith"),
        ])
        first = self.policy.recommend(request).as_dict()
        second = self.policy.recommend(request).as_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
