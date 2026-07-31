import os
import tempfile
import unittest

from advisor.decision_core import (
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from advisor.event_policy import EVENT_OPTION, EventPolicy
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_ID = "ABYSSAL_BATHS"


def _world(*, hp=60):
    return WorldState.create(
        run_id="event-run",
        sequence=8,
        state={
            "character": "IRONCLAD",
            "act": 1,
            "floor": 8,
            "hp": hp,
            "max_hp": 80,
            "gold": 100,
            "energy": 3,
            "deck": [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
            ],
            "relics": [],
            "potions": [],
            "max_potion_slots": 3,
        },
    )


def _effect(kind, **values):
    return {
        "kind": kind,
        "source_code": "catalog:event_effect_fixture",
        **values,
    }


def _candidate(
    candidate_id,
    effects=None,
    *,
    page_id="INITIAL",
    option_id=None,
    **extra,
):
    payload = {
        "candidate_kind": "event_option",
        "event_id": EVENT_ID,
        "page_id": page_id,
        "option_id": option_id or candidate_id,
        "costs": [],
        **extra,
    }
    if effects is not None:
        payload["effects"] = effects
    return DecisionCandidate.create(
        candidate_id,
        payload,
        label=candidate_id,
    )


def _request(candidates, *, world=None):
    return DecisionRequest.create(
        decision_id="event:visit:page",
        decision_type=EVENT_OPTION,
        world=world or _world(),
        candidates=candidates,
    )


class EventPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "event.db")
        )
        cls.repository.sync_catalog(
            os.path.join(ROOT, "data", "knowledge.json")
        )
        cls.policy = EventPolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_low_hp_prefers_structured_heal(self):
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "IMMERSE",
                    [_effect("hp_delta", amount=20)],
                ),
                _candidate("ABSTAIN", [_effect("no_op")]),
            ],
            world=_world(hp=25),
        ))
        self.assertEqual(result.recommended_candidate_id, "IMMERSE")

    def test_description_only_is_not_parsed(self):
        result = self.policy.recommend(_request([
            _candidate(
                "IMMERSE",
                None,
                description="获得999金币并回复全部生命。",
            ),
            _candidate("ABSTAIN", [_effect("no_op")]),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("event:effects_invalid")
            for gap in result.data_gaps
        ))

    def test_description_cannot_change_structured_score(self):
        first = self.policy.recommend(_request([
            _candidate(
                "IMMERSE",
                [_effect("gold_delta", amount=100)],
                description="失去全部金币。",
            ),
            _candidate("ABSTAIN", [_effect("no_op")]),
        ]))
        second = self.policy.recommend(_request([
            _candidate(
                "IMMERSE",
                [_effect("gold_delta", amount=100)],
                description="获得大量金币。",
            ),
            _candidate("ABSTAIN", [_effect("no_op")]),
        ]))
        self.assertEqual(
            [row.score for row in first.candidates],
            [row.score for row in second.candidates],
        )

    def test_page_scope_mismatch_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate("IMMERSE", [_effect("no_op")]),
            _candidate(
                "LINGER",
                [_effect("no_op")],
                page_id="ALL",
            ),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIn("event:candidate_scope_mismatch", result.data_gaps)

    def test_unknown_catalog_option_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate("NOT_AN_OPTION", [_effect("no_op")]),
            _candidate("ABSTAIN", [_effect("no_op")]),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("event:unknown_option")
            for gap in result.data_gaps
        ))

    def test_hp_safety_floor_prefers_safe_option(self):
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "IMMERSE",
                    [_effect("hp_delta", amount=-30)],
                ),
                _candidate("ABSTAIN", [_effect("no_op")]),
            ],
            world=_world(hp=35),
        ))
        self.assertEqual(result.recommended_candidate_id, "ABSTAIN")

    def test_forced_combat_penalty_grows_at_low_hp(self):
        high = self.policy.recommend(_request(
            [
                _candidate("IMMERSE", [_effect("start_combat")]),
                _candidate("ABSTAIN", [_effect("no_op")]),
            ],
            world=_world(hp=70),
        ))
        low = self.policy.recommend(_request(
            [
                _candidate("IMMERSE", [_effect("start_combat")]),
                _candidate("ABSTAIN", [_effect("no_op")]),
            ],
            world=_world(hp=20),
        ))
        high_combat = next(
            row for row in high.candidates if row.candidate_id == "IMMERSE"
        )
        low_combat = next(
            row for row in low.candidates if row.candidate_id == "IMMERSE"
        )
        self.assertGreater(high_combat.score, low_combat.score)

    def test_random_or_followup_effects_do_not_get_fake_scores(self):
        for effect in (
            _effect(
                "gold_delta",
                certainty="bounded",
                min_amount=10,
                max_amount=100,
            ),
            _effect("followup_choice"),
        ):
            with self.subTest(effect=effect["kind"]):
                result = self.policy.recommend(_request([
                    _candidate("IMMERSE", [effect]),
                    _candidate("ABSTAIN", [_effect("no_op")]),
                ]))
                self.assertEqual(result.status, "uncertain")

    def test_known_gold_effect_is_deterministic(self):
        request = _request([
            _candidate(
                "IMMERSE",
                [_effect("gold_delta", amount=120)],
            ),
            _candidate("ABSTAIN", [_effect("no_op")]),
        ])
        first = self.policy.recommend(request).as_dict()
        second = self.policy.recommend(request).as_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["recommended_candidate_id"], "IMMERSE")


if __name__ == "__main__":
    unittest.main()
