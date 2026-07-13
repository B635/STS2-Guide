import unittest

from advisor.decision_core import (
    CARD_REWARD,
    DecisionCandidate,
    DecisionRequest,
    CandidateAssessment,
    PolicyRegistry,
    Recommendation,
    WorldState,
)


class _EchoPolicy:
    decision_type = CARD_REWARD

    def recommend(self, request):
        return Recommendation(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            payload={"candidate_ids": [
                candidate.candidate_id
                for candidate in request.candidates
            ]},
            candidates=tuple(
                CandidateAssessment(candidate.candidate_id, 50.0, index + 1)
                for index, candidate in enumerate(request.candidates)
            ),
            recommended_candidate_id=request.candidates[0].candidate_id,
        )


class DecisionCoreTests(unittest.TestCase):
    def setUp(self):
        self.world = WorldState.create(
            run_id="run-1",
            sequence=3,
            state={"character": "IRONCLAD", "hp": 70},
            map_context={"boss_encounter_ids": ["VANTOM_BOSS"]},
        )

    def test_world_state_builds_policy_scoring_view(self):
        scoring = self.world.scoring_state()
        self.assertEqual(scoring["hp"], 70)
        self.assertEqual(
            scoring["map_context"]["boss_encounter_ids"],
            ["VANTOM_BOSS"],
        )

    def test_request_rejects_duplicate_candidate_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            DecisionRequest.create(
                decision_id="decision-1",
                decision_type=CARD_REWARD,
                world=self.world,
                candidates=[
                    DecisionCandidate.create("BASH", {"card": "BASH"}),
                    DecisionCandidate.create("BASH", {"card": "BASH"}),
                ],
            )

    def test_registry_dispatches_by_decision_type(self):
        registry = PolicyRegistry([_EchoPolicy()])
        request = DecisionRequest.create(
            decision_id="decision-1",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[
                DecisionCandidate.create("BASH", {"card": "BASH"}),
                DecisionCandidate.create("ANGER", {"card": "ANGER"}),
            ],
        )
        result = registry.recommend(request)
        self.assertEqual(result.payload["candidate_ids"], ["BASH", "ANGER"])

    def test_registry_fails_closed_for_unsupported_decision(self):
        request = DecisionRequest.create(
            decision_id="decision-1",
            decision_type="merchant",
            world=self.world,
            candidates=[DecisionCandidate.create("item-1")],
        )
        with self.assertRaisesRegex(LookupError, "no policy"):
            PolicyRegistry().recommend(request)


if __name__ == "__main__":
    unittest.main()
