import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from advisor.card_reward_switch import FALLBACK_GAP, SwitchedCardRewardPolicy
from advisor.decision_core import CARD_REWARD, PolicyRegistry
from advisor.deck_fitness import DeckFitnessPolicy
from advisor.golden_set import card_reward_request, load_golden_set
from advisor.policies import CardRewardPolicy
from advisor.versions import (
    ACTIVE_CARD_REWARD_POLICY_VERSION, CARD_REWARD_POLICY_VERSION, DECK_FITNESS_POLICY_VERSION,
)
from realtime.processor import RealtimeEventProcessor
from storage.relational import RelationalRepository
from storage.release_database import build_release_database

ROOT = Path(__file__).resolve().parents[1]


class Fixed:
    decision_type = CARD_REWARD

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def recommend(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class SwitchedCardRewardPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        database = Path(cls.temporary.name) / "release.db"
        build_release_database(
            database, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json"
        )
        cls.repository = RelationalRepository(str(database))
        cls.request = card_reward_request(load_golden_set()[0]["scoring_view"])
        cls.new = DeckFitnessPolicy(cls.repository).recommend(cls.request)
        cls.old = CardRewardPolicy(cls.repository).recommend(cls.request)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_the_shipped_host_answers_card_rewards_with_deck_fitness(self):
        self.assertEqual(ACTIVE_CARD_REWARD_POLICY_VERSION, DECK_FITNESS_POLICY_VERSION)
        result = RealtimeEventProcessor(self.repository).policies.recommend(self.request)
        self.assertEqual(result.policy_version, DECK_FITNESS_POLICY_VERSION)
        self.assertNotIn(FALLBACK_GAP, result.data_gaps)

    def test_a_scored_decision_never_reaches_the_fallback(self):
        fallback = Fixed(self.old)
        result = SwitchedCardRewardPolicy(Fixed(self.new), fallback).recommend(self.request)
        self.assertIs(result, self.new)
        self.assertEqual(fallback.calls, 0)

    def test_an_uncertain_decision_is_answered_by_the_previous_version(self):
        uncertain = replace(
            self.new, status="uncertain", recommended_candidate_id=None,
            data_gaps=("situation_needs_unavailable",),
        )
        result = SwitchedCardRewardPolicy(Fixed(uncertain), Fixed(self.old)).recommend(self.request)
        self.assertEqual(result.policy_version, CARD_REWARD_POLICY_VERSION)
        self.assertEqual(result.recommended_candidate_id, self.old.recommended_candidate_id)
        # The reader is told that, and why, the new kernel did not answer.
        self.assertIn(FALLBACK_GAP, result.data_gaps)
        self.assertIn("situation_needs_unavailable", result.data_gaps)

    def test_a_failure_of_the_new_kernel_still_gives_advice(self):
        policy = SwitchedCardRewardPolicy(Fixed(error=RuntimeError("boom")), Fixed(self.old))
        with self.assertLogs("advisor.card_reward_switch", level="ERROR"):
            result = PolicyRegistry([policy]).recommend(self.request)
        self.assertEqual(result.policy_version, CARD_REWARD_POLICY_VERSION)
        self.assertIn("deck_fitness_error", result.data_gaps)

    def test_when_both_versions_fail_the_decision_is_uncertain_not_an_error(self):
        policy = SwitchedCardRewardPolicy(Fixed(error=RuntimeError("new")), Fixed(error=ValueError("old")))
        with self.assertLogs("advisor.card_reward_switch", level="ERROR"):
            result = PolicyRegistry([policy]).recommend(self.request)
        self.assertEqual((result.status, result.recommended_candidate_id), ("uncertain", None))
        self.assertIn("card_reward_fallback_error", result.data_gaps)
        self.assertIn("deck_fitness_error", result.data_gaps)

    def test_warming_up_changes_no_answer(self):
        cold = RealtimeEventProcessor(self.repository).policies.recommend(self.request)
        warm_processor = RealtimeEventProcessor(self.repository)
        warm_processor.policies.warm_up()
        warm = warm_processor.policies.recommend(self.request)
        self.assertEqual(
            [(row.candidate_id, row.score, row.rank) for row in warm.candidates],
            [(row.candidate_id, row.score, row.rank) for row in cold.candidates],
        )
        primary = warm_processor.policies._policies[CARD_REWARD].primary
        self.assertEqual(len(primary._pools), 5)   # every character's reward pool is ready


if __name__ == "__main__":
    unittest.main()
