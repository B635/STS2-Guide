import os
import tempfile
import unittest
from unittest.mock import patch

from advisor.deck_edit import DECK_EDIT, DeckEditPolicy
from advisor.decision_core import (
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _world():
    return WorldState.create(
        run_id="deck-edit-run",
        sequence=9,
        state={
            "character": "IRONCLAD",
            "act": 1,
            "floor": 7,
            "hp": 80,
            "max_hp": 80,
            "gold": 100,
            "energy": 3,
            "deck": [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
                {"card": "BASH", "count": 1},
            ],
            "relics": [],
            "potions": [],
            "max_potion_slots": 3,
        },
    )


def _candidate(candidate_id, card, operation="upgrade", eligible=True):
    return DecisionCandidate.create(
        candidate_id,
        {
            "candidate_kind": "deck_edit",
            "entity_id": card,
            "operation": operation,
            "upgrades": 0,
            "costs": [],
        },
        label=card,
        eligible=eligible,
    )


class DeckEditPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "deck-edit.db")
        )
        cls.repository.sync_catalog(
            os.path.join(ROOT, "data", "knowledge.json")
        )
        cls.policy = DeckEditPolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def _request(self, candidates):
        return DecisionRequest.create(
            decision_id="rest:smith:targets",
            decision_type=DECK_EDIT,
            world=_world(),
            candidates=candidates,
            constraints={
                "decision_parent": {
                    "decision_id": "rest:choice",
                    "candidate_id": "smith",
                    "source_type": "rest_site",
                    "source_id": "REST_SITE",
                },
            },
        )

    def test_upgrade_targets_are_scored_without_mutating_deck(self):
        request = self._request([
            _candidate("deck:0", "STRIKE_IRONCLAD"),
            _candidate("deck:5", "DEFEND_IRONCLAD"),
        ])
        first = self.policy.recommend(request)
        second = self.policy.recommend(request)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(
            {row.candidate_id for row in first.candidates},
            {"deck:0", "deck:5"},
        )
        self.assertTrue(all(
            row.score is not None for row in first.candidates
        ))

    def test_partially_modeled_upgrade_does_not_claim_complete_value(self):
        result = self.policy.recommend(self._request([
            _candidate("deck:0", "STRIKE_IRONCLAD"),
            _candidate("deck:9", "BASH"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIsNotNone(result.candidates[0].score)
        bash = result.candidates[1]
        self.assertIsNone(bash.score)
        self.assertIn("deck_edit:upgrade_delta_unknown", bash.data_gaps)
        self.assertFalse(any(
            factor["code"] == "deck_upgrade_opportunity"
            for factor in bash.factors
        ))

    def test_upgrade_targets_respect_signed_and_absolute_costs(self):
        original_find = self.repository.find_card
        original = original_find("BASH")
        for cost, expected in (
            ("+1", "deck:5"),
            ("-1", "deck:9"),
            (1, "deck:9"),
            (0, "deck:9"),
            (3, "deck:5"),
        ):
            card = {**original, "cost": 2, "upgrade": {"cost": cost}}

            def find(identifier):
                return card if identifier == "BASH" else original_find(identifier)

            with self.subTest(cost=cost), patch.object(
                self.repository, "find_card", side_effect=find
            ):
                result = self.policy.recommend(self._request([
                    _candidate("deck:5", "DEFEND_IRONCLAD"),
                    _candidate("deck:9", "BASH"),
                ]))
                self.assertEqual(result.recommended_candidate_id, expected)
                self.assertEqual(result.data_gaps, ())

    def test_unknown_upgrade_cost_is_not_given_a_growth_value(self):
        original = self.repository.find_card("BASH")
        for cost in (None, "X", True, -1, 1.5, "-3"):
            card = {**original, "cost": 2, "upgrade": {"cost": cost}}
            with self.subTest(cost=cost), patch.object(
                self.repository, "find_card", return_value=card
            ):
                result = self.policy.recommend(self._request([
                    _candidate("deck:9", "BASH"),
                ]))
                self.assertEqual(result.status, "uncertain")
                self.assertIsNone(result.candidates[0].score)
                self.assertIn("deck_edit:upgrade_delta_unknown", result.data_gaps)

    def test_unknown_target_fails_closed(self):
        result = self.policy.recommend(self._request([
            _candidate("deck:bad", "NOT_A_CARD"),
            _candidate("deck:0", "STRIKE_IRONCLAD"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIn("unknown_card:NOT_A_CARD", result.data_gaps)

    def test_transform_outcome_is_not_invented(self):
        result = self.policy.recommend(self._request([
            _candidate("deck:0", "STRIKE_IRONCLAD", "transform"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIn(
            "deck_edit:transform_outcome_unknown",
            result.data_gaps,
        )


if __name__ == "__main__":
    unittest.main()
