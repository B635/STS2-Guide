"""Verified parent targets survive the real event/checkpoint policy pipeline."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.processor import RealtimeEventProcessor, RealtimeEventValidationError
from realtime.protocol import GameStateEvent
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]


class DeckTargetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(str(Path(cls.directory.name) / "catalog.db"))
        cls.repository.sync_catalog(ROOT / "data/knowledge.json")
        cls.base = json.loads((ROOT / "protocol/state-event.example.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _event(self, card="STRIKE_IRONCLAD", *, sequence=1, run_id="target-run"):
        payload = copy.deepcopy(self.base)
        payload.update(
            event_id=f"{run_id}:rest:{sequence}", event_type="rest_site",
            run_id=run_id, decision_id=f"{run_id}:rest", sequence=sequence,
            state_revision=sequence, options=[], map_context=None, decision_parent=None,
            decision={"can_skip": False, "can_reroll": False, "reward_source": "REST_SITE"},
        )
        payload["state"]["hp"] = payload["state"]["max_hp"]
        body = {
            "action_id": "SMITH",
            "effects": [{
                "kind": "upgrade_card", "entity_type": "cards", "target_mode": "choose",
                "certainty": "exact", "source_code": "game_api:rest_smith_option",
                "amount": None, "min_amount": None, "max_amount": None,
                "entity_id": None, "child_decision_type": None,
            }],
        }
        if card is not None:
            body["deck_edit_targets"] = {
                "operation": "upgrade", "source_code": "fixture:verified_smith_targets",
                "targets": [{"target_id": "snapshot-target-0", "card": card, "upgrades": 0}],
            }
        payload["candidates"] = [{
            "candidate_id": "rest:SMITH", "kind": "rest_action", "entity_id": None,
            "label": "锻造", "eligible": True, "unavailable_reason": None,
            "costs": [], "payload": body,
        }]
        return GameStateEvent.model_validate(payload)

    def test_only_verified_targets_reach_parent_recommendation(self):
        # The full world includes BASH, whose upgrade is not fully modeled.
        # It is deliberately absent from this observed legal target set.
        result = RealtimeEventProcessor(self.repository).process(self._event())
        recommendation = result["recommendation"]
        self.assertEqual(recommendation["status"], "recommend")
        self.assertEqual(recommendation["recommended_candidate_id"], "rest:SMITH")
        self.assertFalse(recommendation["data_gaps"])

    def test_target_change_updates_same_decision_not_cached_score(self):
        processor = RealtimeEventProcessor(self.repository)
        first = processor.process(self._event())
        second = processor.process(self._event("DEFEND_IRONCLAD", sequence=2))
        self.assertEqual(first["decision_phase"], "opened")
        self.assertEqual(second["decision_phase"], "updated")
        self.assertEqual(first["decision_id"], second["decision_id"])
        first_row = first["recommendation"]["candidates"][0]
        second_row = second["recommendation"]["candidates"][0]
        self.assertNotEqual(first_row["score"], second_row["score"])
        self.assertIn("DEFEND_IRONCLAD", json.dumps(second_row["factors"]))
        self.assertNotIn("STRIKE_IRONCLAD", json.dumps(second_row["factors"]))

    def test_checkpoint_preserves_targets_but_new_snapshot_cannot_reuse_them(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
            processor = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
            event = self._event()
            processor.process(event)
            record = checkpoint.load()["current_decision"]
            saved_targets = record["payload"]["candidates"][0]["payload"]["deck_edit_targets"]
            self.assertEqual(saved_targets, event.candidates[0].payload.deck_edit_targets.model_dump())

            restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
            update = restored.process(self._event("DEFEND_IRONCLAD", sequence=2))
            self.assertEqual(update["decision_phase"], "updated")
            self.assertEqual(update["recommendation"]["status"], "recommend")
            missing = restored.process(self._event(None, sequence=3))
            self.assertEqual(missing["decision_phase"], "updated")
            self.assertEqual(missing["recommendation"]["status"], "uncertain")
            self.assertIsNone(missing["recommendation"]["recommended_candidate_id"])
            self.assertIsNone(missing["recommendation"]["candidates"][0]["score"])
            latest = checkpoint.load()["current_decision"]["payload"]["candidates"][0]["payload"]
            self.assertIsNone(latest.get("deck_edit_targets"))

    def test_new_run_does_not_inherit_old_target_eligibility(self):
        processor = RealtimeEventProcessor(self.repository)
        processor.process(self._event())
        missing = processor.process(self._event(None, run_id="another-run"))
        self.assertEqual(missing["decision_phase"], "opened")
        self.assertEqual(missing["recommendation"]["status"], "uncertain")
        self.assertIsNone(missing["recommendation"]["recommended_candidate_id"])

    def _close_parent(self, processor, parent):
        payload = parent.model_dump(mode="json")
        payload.update(
            event_id=f"{parent.run_id}:closed:2", event_type="decision_closed",
            sequence=2, state_revision=2, candidates=[], decision=None,
            decision_parent=None, parent_event_id=parent.event_id,
            outcome={"kind": "selected", "selected_candidate_id": parent.candidates[0].candidate_id,
                     "selected_card": None, "selected_option_index": None},
        )
        processor.process(GameStateEvent.model_validate(payload))

    def _child(self, parent, cards, *, sequence=3):
        payload = parent.model_dump(mode="json")
        candidate_id = parent.candidates[0].candidate_id
        operation = "remove" if parent.event_type.value == "merchant" else "upgrade"
        payload.update(
            event_id=f"{parent.run_id}:child:{sequence}", event_type="deck_edit",
            decision_id=f"{parent.run_id}:child", sequence=sequence,
            state_revision=sequence, decision_parent={
                "decision_id": parent.decision_id, "candidate_id": candidate_id,
                "source_type": parent.event_type.value, "source_id": "SOURCE_" +
                hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:24].upper(),
            },
            candidates=[{
                "candidate_id": f"child:{index}", "kind": "deck_edit",
                "entity_id": card, "label": card, "eligible": True,
                "unavailable_reason": None, "costs": [],
                "payload": {"operation": operation, "upgrades": upgrades},
            } for index, (card, upgrades) in enumerate(cards)],
        )
        return GameStateEvent.model_validate(payload)

    def _assert_parent_mismatch(self, result):
        recommendation = result["recommendation"]
        self.assertEqual(recommendation["status"], "uncertain")
        self.assertIsNone(recommendation["recommended_candidate_id"])
        self.assertIn("deck_edit:parent_targets_mismatch", recommendation["data_gaps"])
        for row in recommendation["candidates"]:
            self.assertIsNone(row["score"])
            self.assertIsNone(row["rank"])
            self.assertFalse(row["factors"])

    def test_child_disagreement_with_complete_parent_targets_is_unknown(self):
        processor = RealtimeEventProcessor(self.repository)
        parent = self._event()
        processor.process(parent)
        self._close_parent(processor, parent)
        self._assert_parent_mismatch(processor.process(
            self._child(parent, [("DEFEND_IRONCLAD", 0)])
        ))
        recovered = processor.process(self._child(parent, [("STRIKE_IRONCLAD", 0)], sequence=4))
        self.assertEqual(recovered["recommendation"]["status"], "recommend")

    def test_child_multiset_counts_and_upgrade_states_must_match_parent(self):
        for targets, children in (
            ([("STRIKE_IRONCLAD", 0)], [("STRIKE_IRONCLAD", 0)] * 2),
            ([("STRIKE_IRONCLAD", 0)] * 2, [("STRIKE_IRONCLAD", 0)]),
            ([("STRIKE_IRONCLAD", 0)], [("STRIKE_IRONCLAD", 1)]),
            ([], [("STRIKE_IRONCLAD", 0)]),
        ):
            with self.subTest(targets=targets, children=children):
                payload = self._event().model_dump(mode="json")
                upgraded = copy.deepcopy(payload["state"]["deck"][0])
                upgraded.update(card="STRIKE_IRONCLAD", upgrades=1, count=1)
                payload["state"]["deck"].append(upgraded)
                payload["candidates"][0]["payload"]["deck_edit_targets"]["targets"] = [
                    {"target_id": f"parent:{i}", "card": card, "upgrades": upgrades}
                    for i, (card, upgrades) in enumerate(targets)
                ]
                parent = GameStateEvent.model_validate(payload)
                processor = RealtimeEventProcessor(self.repository)
                processor.process(parent)
                self._close_parent(processor, parent)
                self._assert_parent_mismatch(processor.process(self._child(parent, children)))

    def test_matching_instances_ignore_snapshot_ids_and_order(self):
        payload = self._event().model_dump(mode="json")
        cards = [("STRIKE_IRONCLAD", 0), ("DEFEND_IRONCLAD", 0), ("STRIKE_IRONCLAD", 0)]
        payload["candidates"][0]["payload"]["deck_edit_targets"]["targets"] = [
            {"target_id": f"parent:{i}", "card": card, "upgrades": upgrades}
            for i, (card, upgrades) in enumerate(cards)
        ]
        parent = GameStateEvent.model_validate(payload)
        processor = RealtimeEventProcessor(self.repository)
        processor.process(parent)
        self._close_parent(processor, parent)
        result = processor.process(self._child(parent, sorted(cards)))
        self.assertEqual(result["recommendation"]["status"], "recommend")
        self.assertFalse(result["recommendation"]["data_gaps"])

    def test_ineligible_child_entries_are_not_legal_targets(self):
        parent = self._event()
        processor = RealtimeEventProcessor(self.repository)
        processor.process(parent)
        self._close_parent(processor, parent)
        child = self._child(parent, [("STRIKE_IRONCLAD", 0), ("DEFEND_IRONCLAD", 0)]).model_dump(mode="json")
        child["candidates"][1].update(eligible=False, unavailable_reason="unavailable")
        result = processor.process(GameStateEvent.model_validate(child))
        self.assertEqual(result["recommendation"]["status"], "recommend")
        self.assertFalse(result["recommendation"]["data_gaps"])

    def test_legacy_unknown_parent_targets_do_not_disable_actual_child(self):
        parent = self._event(None)
        processor = RealtimeEventProcessor(self.repository)
        processor.process(parent)
        self._close_parent(processor, parent)
        result = processor.process(self._child(parent, [("DEFEND_IRONCLAD", 0)]))
        self.assertEqual(result["recommendation"]["status"], "recommend")
        self.assertFalse(result["recommendation"]["data_gaps"])

    def test_merchant_removal_child_uses_same_parent_consistency_rule(self):
        payload = self._event().model_dump(mode="json")
        payload["event_type"] = "merchant"
        payload["decision"]["reward_source"] = "MERCHANT"
        candidate = payload["candidates"][0]
        candidate.update(candidate_id="merchant:remove", kind="merchant_offer",
                         costs=[{"kind": "gold", "amount": 30, "resource_id": None}])
        body = candidate["payload"]
        body.pop("action_id")
        body.update(slot_id="removal", offer_kind="card_removal", is_stocked=True)
        body["effects"][0]["kind"] = "remove_card"
        body["deck_edit_targets"]["operation"] = "remove"
        parent = GameStateEvent.model_validate(payload)
        processor = RealtimeEventProcessor(self.repository)
        processor.process(parent)
        self._close_parent(processor, parent)
        self._assert_parent_mismatch(processor.process(self._child(parent, [("DEFEND_IRONCLAD", 0)])))
        matching = processor.process(self._child(parent, [("STRIKE_IRONCLAD", 0)], sequence=4))
        self.assertEqual(matching["recommendation"]["status"], "recommend")

    def test_parent_target_consistency_survives_closed_and_active_checkpoint_restore(self):
        for restore_after_child in (False, True):
            with self.subTest(restore_after_child=restore_after_child), tempfile.TemporaryDirectory() as directory:
                checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
                processor = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
                parent = self._event()
                processor.process(parent)
                self._close_parent(processor, parent)
                if restore_after_child:
                    processor.process(self._child(parent, [("STRIKE_IRONCLAD", 0)]))
                restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
                self._assert_parent_mismatch(restored.process(self._child(
                    parent, [("DEFEND_IRONCLAD", 0)], sequence=4 if restore_after_child else 3
                )))

    def test_old_lineage_cache_rederives_targets_from_retained_parent(self):
        for active in (False, True):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as directory:
                checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
                processor = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
                parent = self._event()
                processor.process(parent)
                self._close_parent(processor, parent)
                if active:
                    processor.process(self._child(parent, [("STRIKE_IRONCLAD", 0)]))
                legacy = checkpoint.load()
                closed = legacy["closed_decision"]
                if active:
                    current = legacy["current_decision"]
                    current["result"]["active_child_binding"].pop("deck_edit_targets")
                    closed = current["parent_close"]
                closed["result"]["child_expectation"].pop("deck_edit_targets")
                with patch.object(checkpoint, "load", return_value=legacy):
                    restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
                self._assert_parent_mismatch(restored.process(self._child(
                    parent, [("DEFEND_IRONCLAD", 0)], sequence=4 if active else 3
                )))

    def test_checkpoint_cached_targets_cannot_override_parent_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = ActiveRunCheckpointStore(Path(directory) / "active-run.json")
            processor = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
            parent = self._event()
            processor.process(parent)
            self._close_parent(processor, parent)
            forged = checkpoint.load()
            targets = forged["closed_decision"]["result"]["child_expectation"]["deck_edit_targets"]
            targets["targets"][0]["card"] = "DEFEND_IRONCLAD"
            with patch.object(checkpoint, "load", return_value=forged):
                restored = RealtimeEventProcessor(self.repository, checkpoint=checkpoint)
            with self.assertRaises(RealtimeEventValidationError):
                restored.process(self._child(parent, [("DEFEND_IRONCLAD", 0)]))


if __name__ == "__main__":
    unittest.main()
