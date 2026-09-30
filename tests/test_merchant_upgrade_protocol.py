"""Shop card upgrade truth stays typed and isolated from other products."""
import copy
import json
from pathlib import Path
import unittest
import tempfile

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from realtime.protocol import GameStateEvent
from realtime.processor import RealtimeEventProcessor
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]


class MerchantUpgradeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = json.loads((ROOT / "protocol/deck-targets-event.example.json").read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(json.loads((ROOT / "protocol/state-event.schema.json").read_text(encoding="utf-8")))

    def _offer(self, kind="card", upgrades=0):
        event = copy.deepcopy(self.example)
        row = event["candidates"][0]
        row["candidate_id"] = "merchant:card:0"
        row["entity_id"] = None if kind == "card_removal" else "SHRUG_IT_OFF"
        row["payload"] = {
            "slot_id": "merchant:card:0", "offer_kind": kind,
            "is_stocked": True, "upgrades": upgrades,
        }
        return event

    def test_card_upgrade_state_roundtrips_including_explicit_unknown(self):
        for upgrades in (None, 0, 1, 9):
            event = self._offer(upgrades=upgrades)
            with self.subTest(upgrades=upgrades):
                self.validator.validate(event)
                actual = GameStateEvent.model_validate(event)
                self.assertEqual(actual.candidates[0].payload.upgrades, upgrades)
        event = self._offer()
        del event["candidates"][0]["payload"]["upgrades"]
        self.validator.validate(event)
        self.assertIsNone(GameStateEvent.model_validate(event).candidates[0].payload.upgrades)

    def test_non_card_upgrades_rejected_by_both_contracts(self):
        for kind in ("relic", "potion", "card_removal"):
            with self.subTest(kind=kind):
                event = self._offer(kind=kind)
                self.assertTrue(list(self.validator.iter_errors(event)))
                with self.assertRaises(ValidationError):
                    GameStateEvent.model_validate(event)
                event["candidates"][0]["payload"]["upgrades"] = None
                self.validator.validate(event)
                GameStateEvent.model_validate(event)

    def test_upgrade_scalar_validation_is_strict(self):
        for value in (True, "0", 0.5, -1, 10):
            with self.subTest(value=value):
                event = self._offer(upgrades=value)
                self.assertTrue(list(self.validator.iter_errors(event)))
                with self.assertRaises(ValidationError):
                    GameStateEvent.model_validate(event)

    def test_same_inventory_slot_upgrade_revision_recomputes_and_missing_clears(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = RelationalRepository(str(Path(directory) / "catalog.db"))
            repository.sync_catalog(ROOT / "data/knowledge.json")
            processor = RealtimeEventProcessor(repository)
            payload = self._offer(upgrades=0)
            payload["candidates"] = [payload["candidates"][0], payload["candidates"][-1]]
            first = processor.process(GameStateEvent.model_validate(payload))
            payload.update(sequence=2, state_revision=2, event_id="example-target-run:2")
            payload["candidates"][0]["payload"]["upgrades"] = 1
            second = processor.process(GameStateEvent.model_validate(payload))
            self.assertEqual(second["decision_phase"], "updated")
            self.assertGreater(second["recommendation"]["candidates"][0]["score"], first["recommendation"]["candidates"][0]["score"])
            payload.update(sequence=3, state_revision=3, event_id="example-target-run:3")
            del payload["candidates"][0]["payload"]["upgrades"]
            third = processor.process(GameStateEvent.model_validate(payload))
            self.assertIsNone(third["recommendation"]["candidates"][0]["score"])
            self.assertIn("merchant:card_upgrades_unknown", third["recommendation"]["data_gaps"])
