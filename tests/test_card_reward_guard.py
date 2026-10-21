import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from advisor.card_reward_guard import ERROR_GAP, GuardedCardRewardPolicy
from advisor.decision_core import CARD_REWARD, PolicyRegistry
from advisor.deck_fitness import DeckFitnessPolicy
from advisor.golden_set import card_reward_request, load_golden_set
from advisor.versions import ACTIVE_CARD_REWARD_POLICY_VERSION, DECK_FITNESS_POLICY_VERSION
from realtime.processor import RealtimeEventProcessor
from storage.relational import RelationalRepository
from storage.release_database import build_release_database

ROOT = Path(__file__).resolve().parents[1]


class Fixed:
    decision_type = CARD_REWARD

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def recommend(self, request):
        if self.error:
            raise self.error
        return self.result


class GuardedCardRewardPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        database = Path(cls.temporary.name) / "release.db"
        build_release_database(
            database, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json"
        )
        cls.repository = RelationalRepository(str(database))
        cls.request = card_reward_request(load_golden_set()[0]["scoring_view"])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_the_shipped_host_answers_card_rewards_with_deck_fitness_only(self):
        self.assertEqual(ACTIVE_CARD_REWARD_POLICY_VERSION, DECK_FITNESS_POLICY_VERSION)
        processor = RealtimeEventProcessor(self.repository)
        policy = processor.policies._policies[CARD_REWARD]
        self.assertIsInstance(policy, GuardedCardRewardPolicy)
        self.assertIsInstance(policy.primary, DeckFitnessPolicy)
        self.assertFalse(hasattr(policy, "fallback"))
        result = processor.policies.recommend(self.request)
        self.assertEqual(result.policy_version, DECK_FITNESS_POLICY_VERSION)

    def test_an_uncertain_decision_stays_uncertain(self):
        new = DeckFitnessPolicy(self.repository).recommend(self.request)
        uncertain = replace(new, status="uncertain", recommended_candidate_id=None,
                            data_gaps=("situation_needs_unavailable",))
        result = GuardedCardRewardPolicy(Fixed(uncertain)).recommend(self.request)
        self.assertIs(result, uncertain)

    def test_a_failure_is_shown_as_uncertain_not_as_another_score(self):
        policy = GuardedCardRewardPolicy(Fixed(error=RuntimeError("boom")))
        with self.assertLogs("advisor.card_reward_guard", level="ERROR"):
            result = PolicyRegistry([policy]).recommend(self.request)
        self.assertEqual((result.status, result.recommended_candidate_id), ("uncertain", None))
        self.assertEqual(result.policy_version, DECK_FITNESS_POLICY_VERSION)
        self.assertIn(ERROR_GAP, result.data_gaps)
        self.assertTrue(all(row.score is None for row in result.candidates))

    def test_warming_up_changes_no_answer(self):
        cold = RealtimeEventProcessor(self.repository).policies.recommend(self.request)
        warm_processor = RealtimeEventProcessor(self.repository)
        warm_processor.policies.warm_up()
        warm = warm_processor.policies.recommend(self.request)
        self.assertEqual(
            [(row.candidate_id, row.score, row.rank) for row in warm.candidates],
            [(row.candidate_id, row.score, row.rank) for row in cold.candidates],
        )
        self.assertEqual(len(warm_processor.policies._policies[CARD_REWARD].primary._pools), 5)


if __name__ == "__main__":
    unittest.main()
