"""Executable boundaries for snapshot-scoped deck-edit target facts."""
import copy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from realtime.protocol import GameStateEvent


ROOT = Path(__file__).resolve().parents[1]


class DeckTargetProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = json.loads(
            (ROOT / "protocol/deck-targets-event.example.json").read_text(
                encoding="utf-8"
            )
        )
        cls.schema = json.loads(
            (ROOT / "protocol/state-event.schema.json").read_text(encoding="utf-8")
        )
        cls.validator = Draft202012Validator(cls.schema)

    def test_complete_v9_example_preserves_distinct_same_card_targets(self):
        self.validator.validate(self.example)
        event = GameStateEvent.model_validate(self.example)
        self.assertEqual(
            event.model_dump(mode="json")["candidates"][0]["payload"][
                "deck_edit_targets"
            ],
            self.example["candidates"][0]["payload"]["deck_edit_targets"],
        )

    def _assert_rejected(self, payload):
        self.assertTrue(list(self.validator.iter_errors(payload)))
        with self.assertRaises(ValidationError):
            GameStateEvent.model_validate(payload)

    def test_unknown_null_and_known_empty_remain_distinct(self):
        for mode in ("missing", "null", "empty"):
            payload = copy.deepcopy(self.example)
            body = payload["candidates"][0]["payload"]
            if mode == "missing":
                del body["deck_edit_targets"]
            elif mode == "null":
                body["deck_edit_targets"] = None
            else:
                body["deck_edit_targets"]["targets"] = []
            with self.subTest(mode=mode):
                self.validator.validate(payload)
                facts = GameStateEvent.model_validate(payload).candidates[0].payload.deck_edit_targets
                self.assertEqual(facts is None, mode != "empty")
                if facts is not None:
                    self.assertEqual(facts.targets, [])

    def test_target_scalar_types_and_unknown_fields_are_strict(self):
        for key, value in (("upgrades", True), ("upgrades", "0"), ("upgrades", -1),
                           ("upgrades", 10), ("upgrades", 0.5), ("card", " "),
                           ("target_id", ""), ("extra", 1)):
            with self.subTest(key=key, value=value):
                payload = copy.deepcopy(self.example)
                payload["candidates"][0]["payload"]["deck_edit_targets"]["targets"][0][key] = value
                self._assert_rejected(payload)

    def test_collection_limits_and_provenance_are_strict(self):
        for key, value in (("source_code", " "), ("operation", "transform"),
                           ("targets", None), ("unknown", True)):
            payload = copy.deepcopy(self.example)
            payload["candidates"][0]["payload"]["deck_edit_targets"][key] = value
            with self.subTest(key=key):
                self._assert_rejected(payload)
        payload = copy.deepcopy(self.example)
        payload["candidates"][0]["payload"]["deck_edit_targets"]["targets"] = [
            {"target_id": str(i), "card": "STRIKE_IRONCLAD", "upgrades": 0}
            for i in range(201)
        ]
        self._assert_rejected(payload)

    def test_duplicate_target_ids_with_different_facts_are_semantically_rejected(self):
        payload = copy.deepcopy(self.example)
        targets = payload["candidates"][0]["payload"]["deck_edit_targets"]["targets"]
        targets[1]["target_id"] = targets[0]["target_id"]
        targets[1]["upgrades"] = 1
        # JSON Schema uniqueItems compares whole rows; keyed uniqueness is semantic.
        with self.assertRaises(ValidationError):
            GameStateEvent.model_validate(payload)

    def test_wrong_kind_offer_or_operation_rejects_target_facts(self):
        payload = copy.deepcopy(self.example)
        payload["candidates"][0]["payload"]["deck_edit_targets"]["operation"] = "upgrade"
        self._assert_rejected(payload)
        payload = copy.deepcopy(self.example)
        payload["candidates"][0]["entity_id"] = "STRIKE_IRONCLAD"
        payload["candidates"][0]["payload"]["offer_kind"] = "card"
        self._assert_rejected(payload)
        payload = copy.deepcopy(self.example)
        payload["candidates"][1]["payload"]["deck_edit_targets"] = payload["candidates"][0]["payload"]["deck_edit_targets"]
        self._assert_rejected(payload)

    def test_smith_requires_exact_matching_choose_effect(self):
        payload = copy.deepcopy(self.example)
        payload["candidates"] = [payload["candidates"][0], payload["candidates"][-1]]
        payload.update(event_type="rest_site")
        candidate = payload["candidates"][0]
        candidate.update(kind="rest_action", costs=[])
        facts = candidate["payload"]["deck_edit_targets"]
        facts["operation"] = "upgrade"
        effect = candidate["payload"]["effects"][0]
        effect["kind"] = "upgrade_card"
        candidate["payload"] = {"action_id": "SMITH", "deck_edit_targets": facts, "effects": [effect]}
        self.validator.validate(payload)
        GameStateEvent.model_validate(payload)
        for change in ({"kind": "remove_card"}, {"certainty": "unknown"}, {"target_mode": "random"}):
            changed = copy.deepcopy(payload)
            changed["candidates"][0]["payload"]["effects"][0].update(change)
            self._assert_rejected(changed)
        multiple = copy.deepcopy(payload)
        multiple["candidates"][0]["payload"]["effects"] *= 2
        self._assert_rejected(multiple)
        multiple = copy.deepcopy(payload)
        multiple["candidates"][0]["payload"]["effects"][0]["amount"] = 2
        self._assert_rejected(multiple)


if __name__ == "__main__":
    unittest.main()
