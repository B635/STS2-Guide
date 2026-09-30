"""Verified event facts cross schema, canonical policy, and checkpoint boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator

from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.processor import RealtimeEventProcessor
from realtime.protocol import GameStateEvent
import test_v9_decision_pipeline as fixtures


ROOT = Path(__file__).resolve().parents[1]
PAGE = "PAGE_0123456789ABCDEF"


class EventEffectPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.V9DecisionPipelineTests.setUpClass()
        cls.repository = fixtures.V9DecisionPipelineTests.repository
        cls.base = fixtures.V9DecisionPipelineTests.base
        cls.validator = Draft202012Validator(json.loads(
            (ROOT / "protocol/state-event.schema.json").read_text(encoding="utf-8"),
        ))

    @classmethod
    def tearDownClass(cls):
        fixtures.V9DecisionPipelineTests.tearDownClass()

    def _event(self, amount=10, *, sequence=1, run_id="verified-event", neow=False):
        key = ("NEOW.pages.INITIAL.options.GOLDEN_PEARL" if neow else
               "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN")
        effect = fixtures._effect("gold_delta" if neow else "hp_delta", amount=amount)
        effect["source_code"] = ("game_api:golden_pearl_gold" if neow else
                                 "game_api:abyssal_baths_abstain_heal")
        if amount is None:
            effect = fixtures._effect("followup_choice", certainty="unknown")
            effect["source_code"] = "game_api:event_effect_read_failed"
        body = ({"blessing_id": key, "stage_id": PAGE, "option_id": key} if neow else
                {"event_id": "ABYSSAL_BATHS", "page_id": PAGE, "option_id": key})
        body["effects"] = [effect]
        event = fixtures.V9DecisionPipelineTests._event(self,
            "neow_choice" if neow else "event_choice", [{
                "candidate_id": "event:unchanged-id", "entity_id": None,
                "kind": "neow_blessing" if neow else "event_option", "label": "真实选项",
                "eligible": True, "unavailable_reason": None, "costs": [], "payload": body,
            }], sequence=sequence, run_id=run_id,
        )
        raw = event.model_dump(mode="json")
        raw["state"]["hp"] = 20
        raw["state"]["max_hp"] = 80
        self.validator.validate(raw)
        return GameStateEvent.model_validate(raw)

    def test_same_screen_actual_heal_change_recomputes_score(self):
        processor = RealtimeEventProcessor(self.repository)
        first = processor.process(self._event(10))
        second = processor.process(self._event(15, sequence=2))
        self.assertEqual(second["decision_phase"], "updated")
        self.assertEqual(first["decision_id"], second["decision_id"])
        self.assertGreater(second["recommendation"]["candidates"][0]["score"],
                           first["recommendation"]["candidates"][0]["score"])
        self.assertEqual(second["recommendation"]["recommended_candidate_id"], "event:unchanged-id")

    def test_neow_golden_pearl_uses_observed_amount(self):
        processor = RealtimeEventProcessor(self.repository)
        first = processor.process(self._event(100, neow=True))
        second = processor.process(self._event(150, sequence=2, neow=True))
        self.assertEqual(second["recommendation"]["status"], "recommend")
        self.assertGreater(second["recommendation"]["candidates"][0]["score"],
                           first["recommendation"]["candidates"][0]["score"])

    def test_failed_effect_after_checkpoint_cannot_reuse_known_result(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
            RealtimeEventProcessor(self.repository, checkpoint=checkpoint).process(self._event())
            restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
            result = restored.process(self._event(None, sequence=2))
            self.assertEqual(result["decision_phase"], "updated")
            self.assertEqual(result["recommendation"]["status"], "uncertain")
            self.assertIsNone(result["recommendation"]["candidates"][0]["score"])
            self.assertIsNone(result["recommendation"]["recommended_candidate_id"])

    def test_new_run_never_inherits_old_known_effect(self):
        processor = RealtimeEventProcessor(self.repository)
        processor.process(self._event())
        result = processor.process(self._event(None, run_id="new-event-run"))
        self.assertEqual(result["decision_phase"], "opened")
        self.assertEqual(result["recommendation"]["status"], "uncertain")
        self.assertIsNone(result["recommendation"]["recommended_candidate_id"])


if __name__ == "__main__":
    unittest.main()
