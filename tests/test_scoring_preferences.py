import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from advisor.card_reward import recommend_card_reward
from advisor.campfire import CampfirePolicy
from advisor.choice_effects import AssessmentDraft, finalize_recommendation
from advisor.decision_core import DecisionCandidate, DecisionRequest, WorldState
from advisor.scoring_preferences import (
    DEFAULT_SCORING_PREFERENCES, ScoringPreferences, load_scoring_preferences,
)
from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.host import _application_scoring_preferences
from realtime.processor import RealtimeEventProcessor
from realtime.protocol import GameStateEvent
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]


class ScoringPreferencesTests(unittest.TestCase):
    def test_defaults_bounds_and_strict_shape(self):
        self.assertEqual(ScoringPreferences.from_dict(DEFAULT_SCORING_PREFERENCES.as_dict()),
                         DEFAULT_SCORING_PREFERENCES)
        for value in (0.75, 1, 1.25):
            self.assertEqual(ScoringPreferences(survival=value).survival, value)
        for value in (True, "1.0", None, 0.749, 1.251, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ScoringPreferences(survival=value)
        for change in ({"schema_version": True}, {"schema_version": 2}, {"unknown": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ScoringPreferences.from_dict({**DEFAULT_SCORING_PREFERENCES.as_dict(), **change})
        with self.assertRaises(ValueError):
            ScoringPreferences.from_dict({"schema_version": 1})

    def test_missing_default_invalid_whole_file_fallback_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefs.json"
            self.assertEqual(load_scoring_preferences(path), DEFAULT_SCORING_PREFERENCES)
            path.write_text(json.dumps(ScoringPreferences(survival=1.25).as_dict()), encoding="utf-8")
            self.assertEqual(load_scoring_preferences(path).survival, 1.25)
            for text in (
                '{"schema_version":1,"survival":1.25,"immediate_power":false,"long_term_growth":1}',
                '{"schema_version":1,"survival":1.25,"immediate_power":NaN,"long_term_growth":1}',
                '{"schema_version":1,"survival":1.25,"survival":1,"immediate_power":1,"long_term_growth":1}',
                "{broken", "[]", "[" * 2000 + "]" * 2000,
            ):
                path.write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertLogs("sts2-guide", level="WARNING"):
                    self.assertEqual(load_scoring_preferences(path), DEFAULT_SCORING_PREFERENCES)
            path.unlink()
            self.assertEqual(load_scoring_preferences(path), DEFAULT_SCORING_PREFERENCES)

    def test_application_freezes_preferences_until_a_new_application(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefs.json"
            path.write_text(json.dumps(ScoringPreferences(survival=1.25).as_dict()), encoding="utf-8")
            args = SimpleNamespace(scoring_preferences=path)
            first = _application_scoring_preferences(args)
            path.write_text(json.dumps(ScoringPreferences(survival=0.75).as_dict()), encoding="utf-8")
            self.assertIs(_application_scoring_preferences(args), first)
            self.assertEqual(first.survival, 1.25)
            self.assertEqual(_application_scoring_preferences(SimpleNamespace(scoring_preferences=path)).survival, 0.75)

    def test_only_explicit_card_factors_are_adjusted_and_safety_penalties_do_not_shrink(self):
        preferences = ScoringPreferences(survival=0.75, immediate_power=1.25, long_term_growth=1.25)
        factors = [{"code": code, "delta": delta, "message": code} for code, delta in (
            ("damage_efficiency", 4), ("defense_coverage", 4), ("scaling_value", 4),
            ("self_harm_cost", -4), ("critical_hp_self_harm", -4), ("strong_candidate", -10),
            ("community_prior", 4), ("route_mode_growth_safety_floor", 0),
            ("unknown_damage_word", 4), ("boss_prepare_defense", 4),
            ("boss_prepare_damage", 4), ("boss_prepare_scaling", 4),
        )]
        adjusted = preferences.card_factors(factors)
        self.assertEqual([item["delta"] for item in adjusted], [5, 3, 5, -4, -4, -10, 4, 0, 4, 3, 5, 5])
        self.assertEqual(factors[0]["delta"], 4)
        self.assertEqual(ScoringPreferences(long_term_growth=0.75).adjust(
            -7, "long_term_growth", code="max_hp_change"), -7)

    def test_tray_captures_configuration_before_any_worker_and_reuses_it(self):
        from realtime.installation import DiscoveryResult
        from realtime.public_beta import build_public_beta_controller
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "prefs.json"
            original = ScoringPreferences(survival=1.25)
            path.write_text(json.dumps(original.as_dict()), encoding="utf-8")
            args = SimpleNamespace(
                scoring_preferences=path, compatibility_manifest=ROOT / "packaging/compatibility.json",
                install_receipt=root / "missing.json", game_dir=None, output=root / "advice.json",
                log_file=root / "controller.log", worker_log_file=root / "worker.log",
                runtime_status=root / "status.json", checkpoint=root / "active-run.json",
                poll_interval=0.1, drain_seconds=2.0, worker_stop_timeout=1.0,
            )
            observed = []
            def fake_bridge(worker_args):
                observed.append(_application_scoring_preferences(worker_args))
                return SimpleNamespace(run_once=lambda: None)
            with (
                patch("realtime.public_beta.discover_game_directory", return_value=DiscoveryResult(
                    None, None, ("game_directory_not_found",))),
                patch("realtime.public_beta.CtypesKernelApi", return_value=object()),
                patch("realtime.host._acquire_instance_lock"),
                patch("realtime.host.build_bridge", side_effect=fake_bridge),
            ):
                controller = build_public_beta_controller(args)
                path.write_text(json.dumps(ScoringPreferences(survival=0.75).as_dict()), encoding="utf-8")
                controller.worker_factory()  # construct only; no thread/service starts
                controller.worker_factory()
            self.assertEqual(observed, [original, original])
            self.assertIs(observed[0], observed[1])

    def test_draft_preferences_do_not_override_blocking_unknown_safety_or_eligibility(self):
        world = WorldState.create(run_id="run", sequence=1, state={"character": "IRONCLAD"})
        candidates = [DecisionCandidate.create(str(index), {}, eligible=index != 2) for index in range(4)]
        request = DecisionRequest.create(decision_id="decision", decision_type="event_option",
                                         world=world, candidates=candidates)
        preferences = ScoringPreferences(survival=0.75, long_term_growth=1.25)
        drafts = [AssessmentDraft(candidate, preferences=preferences) for candidate in candidates]
        for draft in drafts:
            draft.add("hp_cost", -8, "cost", "survival")
            draft.add("future", 8, "benefit", "long_term_growth")
        drafts[0].gap("unknown_effect")
        drafts[1].safety_violation = True
        result = finalize_recommendation(request, drafts, policy_version="test")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertEqual([row.score for row in result.candidates], [None, None, None, 52])
        self.assertEqual(drafts[3].factors[0]["delta"], -8)


class PreferencePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(str(Path(cls.directory.name) / "catalog.db"))
        cls.repository.sync_catalog(ROOT / "data/knowledge.json")
        cls.base = json.loads((ROOT / "protocol/state-event.example.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _event(self):
        payload = copy.deepcopy(self.base)
        payload["map_context"] = None
        return GameStateEvent.model_validate(payload)

    def test_default_scores_and_metadata_are_equivalent_to_implicit_defaults(self):
        event = self._event()
        state = event.state.model_dump(mode="json")
        options = [{"card": "TWIN_STRIKE"}, {"card": "SHRUG_IT_OFF"}, {"card": "INFLAME"}]
        implicit = recommend_card_reward(state, options, self.repository)
        explicit = recommend_card_reward(state, options, self.repository, preferences=ScoringPreferences())
        self.assertEqual(implicit, explicit)

    def test_card_preferences_apply_before_dynamic_skip_comparison(self):
        state = {"character": "IRONCLAD", "act": 1, "floor": 6, "hp": 70, "max_hp": 80,
                 "deck": [{"card": "STRIKE_IRONCLAD", "count": 20}]}
        factors = [{"code": "damage_efficiency", "delta": 6.4, "message": "fixture"}]
        with patch("advisor.card_reward.score_state_factors", return_value=factors):
            normal = recommend_card_reward(state, [{"card": "TWIN_STRIKE"}], self.repository)
            weaker = recommend_card_reward(state, [{"card": "TWIN_STRIKE"}], self.repository,
                                           preferences=ScoringPreferences(immediate_power=0.75))
        self.assertEqual(normal["recommendations"][0]["state_score"], 56.4)
        self.assertEqual(weaker["recommendations"][0]["state_score"], 54.8)
        normal_skip = {item["code"]: item["delta"] for item in normal["skip_candidate"]["factors"]}
        weaker_skip = {item["code"]: item["delta"] for item in weaker["skip_candidate"]["factors"]}
        self.assertEqual(normal_skip["useful_candidate"], -6)
        self.assertNotIn("useful_candidate", weaker_skip)
        self.assertGreater(weaker["skip_score"], normal["skip_score"])

    def test_weighted_campfire_keeps_dangerous_unknown_and_ineligible_actions_unscored(self):
        world = WorldState.create(run_id="run", sequence=1, state={
            "character": "IRONCLAD", "hp": 10, "max_hp": 80, "gold": 0,
        })
        candidates = [DecisionCandidate.create("cost", {
            "candidate_kind": "rest_action", "action_id": "cost", "effects": [
                {"kind": "hp_delta", "amount": -20, "source_code": "fixture:cost"},
            ],
        }), DecisionCandidate.create("unknown", {
            "candidate_kind": "rest_action", "action_id": "unknown", "effects": [
                {"kind": "hp_delta", "certainty": "unknown", "source_code": "fixture:unknown"},
            ],
        }), DecisionCandidate.create("disabled", {"candidate_kind": "leave"}, eligible=False)]
        request = DecisionRequest.create(decision_id="rest", decision_type="campfire_action",
                                         world=world, candidates=candidates)
        result = CampfirePolicy(self.repository, ScoringPreferences(
            survival=0.75, long_term_growth=1.25)).recommend(request)
        self.assertIsNone(result.recommended_candidate_id)
        self.assertTrue(all(row.score is None for row in result.candidates))
        self.assertIn("hp_safety_floor_violation", {item["code"] for item in result.candidates[0].factors})

    def test_restart_recomputes_current_snapshot_and_then_reuses_new_result(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
            event = self._event()
            first = RealtimeEventProcessor(self.repository, checkpoint=checkpoint).process(event)
            preferences = ScoringPreferences(immediate_power=1.25, survival=1.25, long_term_growth=1.25)
            restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint, preferences=preferences)
            second = restored.process(event)
            self.assertEqual(second["sequence"], first["sequence"])
            self.assertEqual(second["event_id"], first["event_id"])
            self.assertEqual(second["scoring_preferences"], preferences.as_dict())
            self.assertNotEqual(second["recommendation"]["candidates"], first["recommendation"]["candidates"])
            self.assertEqual(restored.lifecycle.active(event.run_id).sequence, event.sequence)
            with patch.object(restored.policies, "recommend", side_effect=AssertionError("must replay")):
                third = restored.process(event)
            self.assertEqual(third["recommendation"], second["recommendation"])
            self.assertEqual(checkpoint.load()["current_decision"]["result"]["scoring_preferences"], preferences.as_dict())
            reset = RealtimeEventProcessor(self.repository, checkpoint=checkpoint).process(event)
            self.assertEqual(reset["recommendation"], first["recommendation"])

    def test_preference_restart_does_not_reopen_closed_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
            event = self._event()
            processor = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
            processor.process(event)
            payload = event.model_dump(mode="json")
            payload.update(event_id="example-run:close:2", event_type="decision_closed", sequence=2,
                           state_revision=2, candidates=[], decision=None, parent_event_id=event.event_id,
                           outcome={"kind": "selected", "selected_candidate_id": event.candidates[0].candidate_id,
                                    "selected_card": None, "selected_option_index": None})
            close = GameStateEvent.model_validate(payload)
            processor.process(close)
            restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint,
                                              preferences=ScoringPreferences(survival=1.25))
            replay = restored.process(close)
            self.assertEqual(replay["advice_disposition"]["action"], "clear")
            self.assertIsNone(restored.lifecycle.active(event.run_id))
            self.assertIsNone(checkpoint.load()["current_decision"])

    def test_child_rescore_retains_verified_parent_provenance(self):
        from tests.test_deck_target_pipeline import DeckTargetPipelineTests
        helper = DeckTargetPipelineTests()
        helper.repository, helper.base = self.repository, self.base
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
            processor = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
            parent = helper._event()
            processor.process(parent)
            helper._close_parent(processor, parent)
            child = helper._child(parent, [("STRIKE_IRONCLAD", 0)])
            first = processor.process(child)
            before = checkpoint.load()["current_decision"]["parent_close"]
            preferences = ScoringPreferences(long_term_growth=1.25)
            restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint, preferences=preferences)
            second = restored.process(child)
            self.assertGreater(second["recommendation"]["candidates"][0]["score"],
                               first["recommendation"]["candidates"][0]["score"])
            self.assertEqual(checkpoint.load()["current_decision"]["parent_close"], before)
            again = RealtimeEventProcessor(self.repository, checkpoint=checkpoint, preferences=preferences)
            self.assertEqual(again.process(child)["recommendation"], second["recommendation"])
            payload = child.model_dump(mode="json")
            payload.update(event_id="target-run:child:close", event_type="decision_closed", sequence=4,
                           state_revision=4, candidates=[], decision=None, decision_parent=None,
                           parent_event_id=child.event_id,
                           outcome={"kind": "selected", "selected_candidate_id": child.candidates[0].candidate_id,
                                    "selected_card": None, "selected_option_index": None})
            closed = again.process(GameStateEvent.model_validate(payload))
            self.assertEqual(closed["advice_disposition"]["action"], "clear")

    def test_reweighting_cannot_bypass_event_identity_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
            event = self._event()
            RealtimeEventProcessor(self.repository, checkpoint=checkpoint).process(event)
            payload = event.model_dump(mode="json")
            payload["state"]["hp"] -= 1
            forged = GameStateEvent.model_validate(payload)
            restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint,
                                              preferences=ScoringPreferences(survival=1.25))
            with self.assertRaisesRegex(ValueError, "identity collision"):
                restored.process(forged)


if __name__ == "__main__":
    unittest.main()
