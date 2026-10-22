import json
import os
import tempfile
import unittest
from unittest.mock import patch

from advisor.campfire import (
    CAMPFIRE_ACTION,
    REST_RULE_KEY,
    CampfirePolicy,
    rest_rule_parameter_gaps,
)
from advisor.decision_core import (
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from storage.release_database import (
    DEFAULT_GUIDE_PARAMETERS_PATH,
    GUIDE_PARAMETER_KEYS,
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
        # Guide-owned parameters reach the release database the same way.
        cls.repository.sync_guide_parameters(
            str(DEFAULT_GUIDE_PARAMETERS_PATH),
            GUIDE_PARAMETER_KEYS,
            "test",
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

    def _rest_or_smith(self, hp, *, heal=24, route=False):
        return self.policy.recommend(_request(
            [
                _candidate("rest", "rest", _heal(heal)),
                _candidate("smith", "smith"),
            ],
            world=_world(hp=hp, route=route),
        ))

    @staticmethod
    def _row(result, candidate_id):
        return next(
            row for row in result.candidates
            if row.candidate_id == candidate_id
        )

    def _rule_message(self, result, candidate_id):
        return next(
            item["message"]
            for item in self._row(result, candidate_id).factors
            if item["code"] == "campfire_rest_rule"
        )

    def test_rest_rule_decides_at_the_sixty_percent_edge(self):
        # Max HP 80: 47 is 58.75%, 48 is exactly 60%. A 2 HP heal scores
        # below the upgrade and a 24 HP heal above it; the rule alone decides.
        margin = self.repository.mechanic_constant(REST_RULE_KEY)["rule_margin"]
        for hp, expected, other, message in (
            (47, "rest", "smith", "生命 58%，低于 60%：休息更稳。"),
            (48, "smith", "rest", "生命 60%，不低于 60%：锻造更划算。"),
        ):
            for heal in (2, 24):
                with self.subTest(hp=hp, heal=heal):
                    result = self._rest_or_smith(hp, heal=heal)
                    self.assertEqual(result.status, "recommend")
                    self.assertEqual(result.recommended_candidate_id, expected)
                    self.assertEqual(result.data_gaps, ())
                    self.assertEqual(self._rule_message(result, expected), message)
                    self.assertGreaterEqual(
                        self._row(result, expected).score
                        - self._row(result, other).score,
                        margin,
                    )

    def test_rest_rule_does_not_read_the_route(self):
        # An elite next raises the heal value, but 62% HP still smiths.
        result = self._rest_or_smith(50, route=True)
        self.assertEqual(result.recommended_candidate_id, "smith")
        self.assertEqual(
            self._rule_message(result, "rest"),
            "生命 62%，不低于 60%：休息收益有限。",
        )

    def test_unknown_hp_is_a_disclosed_gap(self):
        result = self._rest_or_smith(None)
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIn("campfire_rule:hp_missing", result.data_gaps)
        for row in result.candidates:
            self.assertIsNone(row.score)
            self.assertIn("campfire_rule:hp_missing", row.data_gaps)

    def test_rest_rule_parameters_missing_or_invalid_fail_closed(self):
        valid = self.repository.mechanic_constant(REST_RULE_KEY)
        original = self.repository.mechanic_constant
        cases = {
            "missing": (None, "campfire_rule:parameters_missing"),
            "other version": (
                {**valid, "version": "campfire-rest-rule-v0"},
                "campfire_rule:parameters_unsupported_version",
            ),
            "no threshold": (
                {k: v for k, v in valid.items() if k != "heal_below_hp_ratio"},
                "campfire_rule:parameter_missing:heal_below_hp_ratio",
            ),
            "threshold as percent": (
                {**valid, "heal_below_hp_ratio": 60},
                "campfire_rule:parameter_invalid:heal_below_hp_ratio",
            ),
            "threshold not a number": (
                {**valid, "heal_below_hp_ratio": True},
                "campfire_rule:parameter_invalid:heal_below_hp_ratio",
            ),
            "no margin": (
                {k: v for k, v in valid.items() if k != "rule_margin"},
                "campfire_rule:parameter_missing:rule_margin",
            ),
            "zero margin": (
                {**valid, "rule_margin": 0},
                "campfire_rule:parameter_invalid:rule_margin",
            ),
        }
        for name, (params, gap) in cases.items():
            self.assertEqual(rest_rule_parameter_gaps(params), (gap,), name)
            with self.subTest(name), patch.object(
                self.repository,
                "mechanic_constant",
                side_effect=lambda key, p=params: (
                    p if key == REST_RULE_KEY else original(key)
                ),
            ):
                result = self._rest_or_smith(30)
                self.assertEqual(result.status, "uncertain")
                self.assertIsNone(result.recommended_candidate_id)
                self.assertIn(gap, result.data_gaps)
                self.assertTrue(all(row.score is None for row in result.candidates))

    def test_shipped_rest_rule_parameters_are_valid(self):
        with open(DEFAULT_GUIDE_PARAMETERS_PATH, encoding="utf-8") as stream:
            shipped = json.load(stream)[REST_RULE_KEY]
        self.assertEqual(rest_rule_parameter_gaps(shipped), ())

    def test_rule_does_not_push_an_upgrade_without_gain(self):
        original = self.repository.find_card("BASH")
        card = {**original, "cost": 2, "upgrade": {"cost": "+1"}}
        with patch.object(self.repository, "find_card", return_value=card):
            result = self.policy.recommend(_request(
                [
                    _candidate("rest", "rest", _heal(1)),
                    _candidate("smith", "smith"),
                ],
                world=_world(hp=79, deck=[{"card": "BASH"}]),
            ))
        self.assertEqual(result.recommended_candidate_id, "rest")
        self.assertIn(
            "campfire_rest_rule_not_applied",
            {item["code"] for item in self._row(result, "smith").factors},
        )


if __name__ == "__main__":
    unittest.main()
