import unittest
from dataclasses import replace

from advisor.decision_core import (
    CARD_REWARD,
    DecisionLifecycleManager,
    DecisionPhase,
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
                CandidateAssessment(
                    candidate_id=candidate.candidate_id,
                    label=candidate.label,
                    display_index=int(candidate.display_index),
                    eligible=candidate.eligible,
                    score=50.0,
                    rank=(index + 1 if candidate.eligible else None),
                )
                for index, candidate in enumerate(request.candidates)
            ),
            recommended_candidate_id=request.candidates[0].candidate_id,
            world_sequence=request.world.sequence,
            policy_version="echo:1",
        )


class _TransformPolicy(_EchoPolicy):
    def __init__(self, transform):
        self._transform = transform

    def recommend(self, request):
        return self._transform(request, super().recommend(request))


class DecisionCoreTests(unittest.TestCase):
    def test_candidate_assessment_score_rank_invariants(self):
        base = {
            "candidate_id": "candidate",
            "label": "候选",
            "display_index": 0,
            "eligible": True,
            "score": None,
            "rank": None,
        }
        CandidateAssessment(**base)
        for changes in (
            {"eligible": False, "score": 10.0},
            {"eligible": False, "rank": 1},
            {"score": None, "rank": 1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    CandidateAssessment(**{**base, **changes})
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

    def test_world_state_uses_typed_run_collections(self):
        world = WorldState.create(
            run_id="run-typed",
            sequence=8,
            game_version="0.108.0",
            state={
                "character": "REGENT",
                "act": 2,
                "floor": 19,
                "deck": [{"card": "GUIDING_STAR", "count": 2}],
                "relic_states": [{"relic": "DIVINE_RIGHT"}],
                "potions": [{"potion": "STRENGTH", "slot": 0}],
            },
            map_context={
                "nodes": [{
                    "node_id": "1:2",
                    "kind": "SHOP",
                    "row": 1,
                    "col": 2,
                    "edges": ["2:2"],
                }],
                "available_next_node_ids": ["1:2"],
            },
        )
        self.assertEqual(world.deck[0].card, "GUIDING_STAR")
        self.assertEqual(world.relics[0].relic, "DIVINE_RIGHT")
        self.assertEqual(world.potions[0].potion, "STRENGTH")
        self.assertEqual(world.map.nodes[0].kind, "SHOP")
        scoring = world.scoring_state()
        self.assertEqual(scoring["deck"][0]["count"], 2)
        self.assertEqual(scoring["map_context"]["node_count"], 1)
        self.assertEqual(scoring["game_version"], "0.108.0")

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
        self.assertEqual(result.payload["candidate_ids"], ("BASH", "ANGER"))
        self.assertEqual(
            result.payload_dict()["candidate_ids"],
            ["BASH", "ANGER"],
        )

    def test_registry_fails_closed_for_unsupported_decision(self):
        request = DecisionRequest.create(
            decision_id="decision-1",
            decision_type="merchant",
            world=self.world,
            candidates=[DecisionCandidate.create("item-1")],
        )
        with self.assertRaisesRegex(LookupError, "no policy"):
            PolicyRegistry().recommend(request)

    def test_registry_rejects_invalid_policy_output(self):
        request = DecisionRequest.create(
            decision_id="decision-invalid",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[
                DecisionCandidate.create("0:BASH"),
                DecisionCandidate.create("1:ANGER"),
            ],
        )

        def extra_candidate(_request, result):
            ghost = CandidateAssessment(
                candidate_id="ghost",
                label="ghost",
                display_index=99,
                eligible=True,
                score=1,
                rank=3,
            )
            return replace(result, candidates=result.candidates + (ghost,))

        def duplicate_rank(_request, result):
            duplicate = replace(
                result.candidates[1],
                rank=result.candidates[0].rank,
            )
            return replace(
                result,
                candidates=(result.candidates[0], duplicate),
            )

        cases = (
            (
                "sequence",
                lambda _request, result: replace(
                    result,
                    world_sequence=result.world_sequence + 1,
                ),
            ),
            (
                "missing",
                lambda _request, result: replace(
                    result,
                    candidates=result.candidates[:-1],
                ),
            ),
            ("invented", extra_candidate),
            (
                "reordered",
                lambda _request, result: replace(
                    result,
                    candidates=tuple(reversed(result.candidates)),
                ),
            ),
            ("duplicate rank", duplicate_rank),
        )
        for label, transform in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    PolicyRegistry(
                        [_TransformPolicy(transform)]
                    ).recommend(request)

    def test_registry_rejects_recommendation_of_ineligible_candidate(self):
        request = DecisionRequest.create(
            decision_id="decision-ineligible",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[
                DecisionCandidate.create("leave", eligible=False),
                DecisionCandidate.create("buy", eligible=True),
            ],
        )
        with self.assertRaisesRegex(ValueError, "ineligible"):
            PolicyRegistry([_EchoPolicy()]).recommend(request)

    def test_recommendation_payload_is_deeply_immutable_and_materializable(self):
        source = {"nested": {"values": [1, 2]}}
        result = Recommendation(
            decision_id="decision-frozen",
            decision_type=CARD_REWARD,
            payload=source,
            world_sequence=3,
            policy_version="test:1",
            candidates=(
                CandidateAssessment(
                    candidate_id="card",
                    label="Card",
                    display_index=0,
                    eligible=True,
                    score=50.0,
                    rank=1,
                ),
            ),
            recommended_candidate_id="card",
        )
        source["nested"]["values"].append(3)
        self.assertEqual(
            result.payload_dict(),
            {"nested": {"values": [1, 2]}},
        )
        with self.assertRaises(TypeError):
            result.payload["nested"]["other"] = True

    def test_recommendation_rejects_semantic_status_and_metadata_mismatch(self):
        assessment = CandidateAssessment(
            candidate_id="card",
            label="Card",
            display_index=0,
            eligible=True,
            score=50.0,
            rank=1,
        )
        common = {
            "decision_id": "decision-status",
            "decision_type": CARD_REWARD,
            "payload": {},
            "world_sequence": 3,
            "policy_version": "test:1",
            "candidates": (assessment,),
        }
        for status, recommended in (
            ("recommend", None),
            ("recommend", "skip"),
            ("uncertain", "card"),
            ("skip", "card"),
        ):
            with self.subTest(status=status, recommended=recommended):
                with self.assertRaises(ValueError):
                    Recommendation(
                        **common,
                        status=status,
                        recommended_candidate_id=recommended,
                    )

        invalid_assessments = (
            {"factors": ("not-an-object",)},
            {"dimensions": {"survival": float("nan")}},
            {"data_gaps": ("missing", "missing")},
            {"data_gaps": ("",)},
        )
        for changes in invalid_assessments:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    CandidateAssessment(
                        candidate_id="card",
                        label="Card",
                        display_index=0,
                        eligible=True,
                        score=50.0,
                        rank=1,
                        **changes,
                    )

    def test_recommendation_serializes_policy_neutral_contract(self):
        request = DecisionRequest.create(
            decision_id="decision-contract",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        result = PolicyRegistry([_EchoPolicy()]).recommend(request)
        payload = result.as_dict()
        self.assertEqual(payload["contract_version"], 1)
        self.assertEqual(payload["decision_type"], CARD_REWARD)
        self.assertEqual(payload["candidates"][0]["candidate_id"], "0:BASH")

    def test_lifecycle_opens_and_closes_one_run_scoped_decision(self):
        manager = DecisionLifecycleManager()
        request = DecisionRequest.create(
            decision_id="decision-life",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        opened = manager.open(request)
        self.assertEqual(opened.phase, DecisionPhase.OPENED)
        self.assertEqual(manager.active("run-1").decision_id, "decision-life")
        closed = manager.close(
            run_id="run-1",
            decision_id="decision-life",
            decision_type=CARD_REWARD,
            sequence=4,
            outcome="selected",
        )
        self.assertEqual(closed.phase, DecisionPhase.CLOSED)
        self.assertIsNone(manager.active("run-1"))

    def test_lifecycle_allows_checkpoint_recovered_close(self):
        manager = DecisionLifecycleManager()
        closed = manager.close(
            run_id="run-recovered",
            decision_id="decision-recovered",
            decision_type=CARD_REWARD,
            sequence=9,
            outcome="skipped",
            allow_recovered=True,
        )
        self.assertEqual(closed.outcome, "skipped")

    def test_lifecycle_rejects_close_for_another_decision(self):
        manager = DecisionLifecycleManager()
        request = DecisionRequest.create(
            decision_id="decision-a",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        manager.open(request)
        with self.assertRaisesRegex(ValueError, "another decision"):
            manager.close(
                run_id="run-1",
                decision_id="decision-b",
                decision_type=CARD_REWARD,
                sequence=4,
                outcome="selected",
            )

    def test_lifecycle_updates_same_stable_decision(self):
        manager = DecisionLifecycleManager()
        first = DecisionRequest.create(
            decision_id="decision-stable",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        updated_world = WorldState.create(
            run_id="run-1",
            sequence=4,
            state={"character": "IRONCLAD", "hp": 69},
        )
        second = DecisionRequest.create(
            decision_id="decision-stable",
            decision_type=CARD_REWARD,
            world=updated_world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        self.assertEqual(manager.open(first).phase, DecisionPhase.OPENED)
        self.assertEqual(manager.open(second).phase, DecisionPhase.UPDATED)

    def test_lifecycle_rejects_type_change_for_stable_decision_id(self):
        manager = DecisionLifecycleManager()
        first = DecisionRequest.create(
            decision_id="decision-stable",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        changed_world = WorldState.create(
            run_id="run-1",
            sequence=4,
            state={"character": "IRONCLAD", "hp": 69},
        )
        changed = DecisionRequest.create(
            decision_id="decision-stable",
            decision_type="merchant",
            world=changed_world,
            candidates=[DecisionCandidate.create("relic:anchor")],
        )

        manager.open(first)
        with self.assertRaisesRegex(ValueError, "decision type"):
            manager.open(changed)

        self.assertEqual(manager.active("run-1").decision_type, CARD_REWARD)

    def test_lifecycle_duplicate_close_is_idempotent_but_conflict_fails(self):
        manager = DecisionLifecycleManager()
        request = DecisionRequest.create(
            decision_id="decision-final",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        manager.open(request)
        first = manager.close(
            run_id="run-1",
            decision_id="decision-final",
            decision_type=CARD_REWARD,
            sequence=4,
            outcome="skip",
        )
        duplicate = manager.close(
            run_id="run-1",
            decision_id="decision-final",
            decision_type=CARD_REWARD,
            sequence=5,
            outcome="skip",
        )
        self.assertEqual(duplicate, first)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            manager.close(
                run_id="run-1",
                decision_id="decision-final",
                decision_type=CARD_REWARD,
                sequence=6,
                outcome="0:BASH",
            )

    def test_lifecycle_closed_stable_id_cannot_reopen(self):
        manager = DecisionLifecycleManager()
        request = DecisionRequest.create(
            decision_id="decision-final",
            decision_type=CARD_REWARD,
            world=self.world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )
        manager.open(request)
        manager.close(
            run_id="run-1",
            decision_id="decision-final",
            decision_type=CARD_REWARD,
            sequence=4,
            outcome="skip",
        )
        later_world = WorldState.create(
            run_id="run-1",
            sequence=5,
            state={"character": "IRONCLAD", "hp": 69},
        )
        repeated = DecisionRequest.create(
            decision_id="decision-final",
            decision_type=CARD_REWARD,
            world=later_world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )

        with self.assertRaisesRegex(ValueError, "cannot be reopened"):
            manager.open(repeated)

        self.assertEqual(manager.closed("run-1").decision_id, "decision-final")
        self.assertIsNone(manager.active("run-1"))

    def test_lifecycle_new_decision_id_replaces_closed_tombstone(self):
        manager = DecisionLifecycleManager()
        manager.restore_closed(
            run_id="run-1",
            decision_id="decision-old",
            decision_type=CARD_REWARD,
            sequence=4,
            outcome="skip",
            observation_event_id="run-1:4",
        )
        later_world = WorldState.create(
            run_id="run-1",
            sequence=5,
            state={"character": "IRONCLAD", "hp": 69},
        )
        replacement = DecisionRequest.create(
            decision_id="decision-new",
            decision_type=CARD_REWARD,
            world=later_world,
            candidates=[DecisionCandidate.create("0:BASH")],
        )

        transition = manager.open(replacement)

        self.assertEqual(transition.phase, DecisionPhase.OPENED)
        self.assertEqual(manager.active("run-1").decision_id, "decision-new")
        self.assertIsNone(manager.closed("run-1"))


if __name__ == "__main__":
    unittest.main()
