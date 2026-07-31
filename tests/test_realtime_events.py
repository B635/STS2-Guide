import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from api import app
from advisor.card_reward import recommend_card_reward
from advisor.decision_core import (
    CandidateAssessment,
    DecisionCandidate,
    DecisionRequest,
    PolicyRegistry,
    Recommendation,
)
from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.file_bridge import GameStateFileBridge
from realtime.processor import (
    RealtimeEventProcessor,
    RealtimeEventValidationError,
)
from realtime.protocol import GameStateEvent
from realtime.session import TransientSessionStore
from storage.relational import RelationalRepository


def _knowledge_payload():
    return {
        "characters": [
            {
                "id": "TEST_HERO",
                "name": "测试角色",
                "description": "",
                "embed_text": "角色测试角色",
            }
        ],
        "cards": [
            {
                "id": "HEAVY_ATTACK",
                "name": "重击测试牌",
                "description": "造成较高伤害。",
                "cost": 3,
                "type_key": "Attack",
                "rarity_key": "Common",
                "color": "test",
                "damage": 18,
                "block": None,
                "keywords_key": [],
                "embed_text": "卡牌重击测试牌：造成较高伤害。",
            },
            {
                "id": "CHEAP_BLOCK",
                "name": "轻防测试牌",
                "description": "获得格挡。",
                "cost": 1,
                "type_key": "Skill",
                "rarity_key": "Common",
                "color": "test",
                "damage": None,
                "block": 8,
                "keywords_key": [],
                "embed_text": "卡牌轻防测试牌：获得8点格挡。",
            },
        ],
        "relics": [],
        "potions": [],
        "monsters": [],
    }


def _event_payload(
    sequence=1,
    event_type="card_reward",
    outcome=None,
):
    payload = {
        "schema_version": 2,
        "event_id": f"test-run:{sequence}",
        "event_type": event_type,
        "emitted_at": "2026-07-02T08:00:00+00:00",
        "source": "test-readonly-mod",
        "game_version": "test",
        "run_id": "test-run",
        "sequence": sequence,
        "state": {
            "character": "TEST_HERO",
            "ascension": 0,
            "act": 1,
            "floor": 3,
            "hp": 60,
            "max_hp": 70,
            "energy": 3,
            "deck": [
                {
                    "card": "HEAVY_ATTACK",
                    "count": 3,
                    "upgrades": 0,
                    "enchantment": "TEST_ENCHANTMENT",
                    "enchantment_amount": 2,
                    "affliction": None,
                    "affliction_amount": None,
                }
            ],
            "relics": [],
            "relic_states": [
                {
                    "relic": "TEST_RELIC",
                    "display_amount": 2,
                    "stack_count": 1,
                    "status": "ACTIVE",
                }
            ],
            "potions": [{"potion": "TEST_POTION", "slot": 1}],
            "max_potion_slots": 3,
            "modifiers": ["TEST_MODIFIER"],
            "capture_warnings": [],
        },
        "options": (
            [
                {"card": "HEAVY_ATTACK", "upgrades": 0},
                {
                    "card": "CHEAP_BLOCK",
                    "upgrades": 0,
                    "enchantment": "TEST_REWARD_ENCHANTMENT",
                    "enchantment_amount": 1,
                },
            ]
            if event_type == "card_reward"
            else []
        ),
    }
    if event_type == "card_reward":
        payload["decision"] = {
            "can_skip": True,
            "can_reroll": False,
            "reward_source": "MONSTER",
        }
    elif event_type == "decision_closed":
        payload["parent_event_id"] = "test-run:1"
        payload["outcome"] = outcome or {"kind": "closed_unknown"}
    return payload


def _v5_decision_payload(
    *,
    sequence=1,
    event_type="card_reward",
    decision_id="test-run:card-reward:stable",
    outcome=None,
):
    payload = _event_payload(sequence, event_type, outcome)
    payload["schema_version"] = 5
    payload["decision_id"] = decision_id
    if event_type == "card_reward":
        for index, option in enumerate(payload["options"]):
            option["candidate_id"] = f"{index}:{option['card']}"
    return payload


class RealtimeEventTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.knowledge_path = os.path.join(
            self.tempdir.name,
            "knowledge.json",
        )
        self.database_path = os.path.join(self.tempdir.name, "sts2.db")
        self.checkpoint_path = (
            Path(self.tempdir.name) / "active-run.json"
        )
        with open(self.knowledge_path, "w", encoding="utf-8") as file:
            json.dump(_knowledge_payload(), file, ensure_ascii=False)
        self.repository = RelationalRepository(self.database_path)
        self.repository.sync_catalog(self.knowledge_path)
        self.sessions = TransientSessionStore()
        self.checkpoint = ActiveRunCheckpointStore(self.checkpoint_path)
        self.processor = RealtimeEventProcessor(
            self.repository,
            sessions=self.sessions,
            checkpoint=self.checkpoint,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _open_generic_merchant_decision(
        self,
        *,
        include_ineligible: bool = False,
    ) -> RealtimeEventProcessor:
        class MerchantPolicy:
            decision_type = "merchant"

            @staticmethod
            def recommend(request):
                assessments = []
                rank = 1
                for candidate in request.candidates:
                    assessments.append(CandidateAssessment(
                        candidate_id=candidate.candidate_id,
                        label=candidate.label,
                        display_index=candidate.display_index,
                        eligible=candidate.eligible,
                        score=60 if candidate.eligible else None,
                        rank=rank if candidate.eligible else None,
                    ))
                    if candidate.eligible:
                        rank += 1
                return Recommendation(
                    decision_id=request.decision_id,
                    decision_type=request.decision_type,
                    payload={"merchant_note": "structured-only"},
                    world_sequence=request.world.sequence,
                    policy_version="merchant:test:1",
                    candidates=tuple(assessments),
                    recommended_candidate_id="relic:anchor",
                )

        processor = RealtimeEventProcessor(
            self.repository,
            sessions=self.sessions,
            checkpoint=self.checkpoint,
            policies=PolicyRegistry([MerchantPolicy()]),
        )
        payload = _event_payload(1, "merchant")
        payload["schema_version"] = 5
        payload["event_id"] = "test-run:merchant:1"
        event = GameStateEvent.model_validate(payload)
        request_candidates = [DecisionCandidate.create(
            "relic:anchor",
            label="锚",
        )]
        if include_ineligible:
            request_candidates.append(DecisionCandidate.create(
                "relic:forbidden",
                label="不可购买遗物",
                eligible=False,
            ))
        request = DecisionRequest.create(
            decision_id="test-run:merchant:stable",
            decision_type="merchant",
            world=processor._build_world(event),
            candidates=request_candidates,
        )
        event_payload = event.model_dump(mode="json")
        claimed, _ = self.sessions.claim(event_payload)
        self.assertTrue(claimed)
        result = processor._process_decision(event, request)
        result.update({
            "run_id": event.run_id,
            "sequence": event.sequence,
            "emitted_at": event.emitted_at.isoformat(),
            "processed_at": "2026-07-14T00:00:00+00:00",
        })
        self.sessions.complete(
            event.event_id,
            status=result["status"],
            decision_id=request.decision_id,
            result=result,
        )
        self.checkpoint.update(event_payload, result)
        return processor

    def test_card_reward_event_is_processed_once(self):
        event = GameStateEvent.model_validate(_event_payload())
        first = self.processor.process(event)
        second = self.processor.process(event)

        legacy_state = event.state.model_dump()
        legacy_state["game_version"] = event.game_version
        legacy_advice = recommend_card_reward(
            legacy_state,
            [option.model_dump() for option in event.options],
            self.repository,
        )

        self.assertEqual(first["status"], "processed")
        self.assertEqual(first["advice"]["recommended_option"], "轻防测试牌")
        self.assertEqual(first["advice"], legacy_advice)
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["decision_id"], second["decision_id"])
        self.assertEqual(self.sessions.count(), 1)
        self.assertEqual(first["decision_phase"], "opened")
        canonical = first["recommendation"]
        # This unit processor intentionally has no production compatibility
        # manifest.  Its recommendation remains useful for lifecycle tests,
        # but it must not satisfy the publishable advice envelope: only the
        # fail-closed Host path may add the compatibility attestation.
        advice_schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "protocol"
                / "advice-event.schema.json"
            ).read_text(encoding="utf-8")
        )
        errors = list(
            Draft202012Validator(advice_schema).iter_errors(first)
        )
        self.assertTrue(
            any(error.validator == "required"
                and "compatibility" in error.message
                for error in errors)
        )
        self.assertEqual(canonical["contract_version"], 1)
        self.assertEqual(canonical["decision_id"], event.event_id)
        self.assertEqual(canonical["decision_type"], "card_reward")
        self.assertEqual(canonical["world_sequence"], event.sequence)
        self.assertEqual(
            canonical["recommended_candidate_id"],
            "1:CHEAP_BLOCK",
        )
        self.assertEqual(
            len(canonical["candidates"]),
            len(event.options) + 1,
        )
        self.assertEqual(
            canonical["candidates"][-1]["candidate_id"],
            "skip",
        )

        latest = self.sessions.latest()
        self.assertEqual(latest["event_id"], event.event_id)
        self.assertEqual(latest["decision_id"], first["decision_id"])
        self.assertIsNone(first["state_id"])
        self.assertEqual(first["decision_id"], event.event_id)
        self.assertTrue(latest["payload"]["decision"]["can_skip"])
        self.assertFalse(latest["payload"]["decision"]["can_reroll"])
        self.assertEqual(
            latest["payload"]["decision"]["reward_source"],
            "MONSTER",
        )
        self.assertEqual(
            first["advice"]["recommendations"][0]["enchantment"],
            "TEST_REWARD_ENCHANTMENT",
        )
        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["run_id"], "test-run")
        self.assertEqual(
            checkpoint["current_decision"]["event_id"],
            event.event_id,
        )

    def test_event_identity_collision_is_rejected(self):
        first = GameStateEvent.model_validate(_event_payload())
        self.processor.process(first)
        changed_payload = _event_payload()
        changed_payload["options"] = [{"card": "CHEAP_BLOCK"}]
        changed = GameStateEvent.model_validate(changed_payload)
        with self.assertRaisesRegex(ValueError, "identity collision"):
            self.processor.process(changed)

    def test_same_stable_decision_id_produces_updated_phase(self):
        first_payload = _v5_decision_payload(sequence=1)
        first = self.processor.process(
            GameStateEvent.model_validate(first_payload)
        )
        update_payload = _v5_decision_payload(sequence=2)
        update_payload["event_id"] = "test-run:2"
        update_payload["state"]["hp"] = 59
        updated = self.processor.process(
            GameStateEvent.model_validate(update_payload)
        )
        self.assertEqual(first["decision_phase"], "opened")
        self.assertEqual(updated["decision_phase"], "updated")
        self.assertEqual(first["decision_id"], updated["decision_id"])
        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["current_decision"]["event_id"], "test-run:2")

    def test_restarted_host_updates_checkpointed_stable_decision(self):
        first_payload = _v5_decision_payload(sequence=1)
        self.processor.process(GameStateEvent.model_validate(first_payload))
        restarted = RealtimeEventProcessor(
            self.repository,
            sessions=TransientSessionStore(),
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )
        update_payload = _v5_decision_payload(sequence=2)
        update_payload["event_id"] = "test-run:2"
        update_payload["state"]["hp"] = 59

        updated = restarted.process(
            GameStateEvent.model_validate(update_payload)
        )

        self.assertEqual(updated["decision_phase"], "updated")
        self.assertEqual(
            self.checkpoint.load()["current_decision"]["event_id"],
            "test-run:2",
        )

    def test_updated_decision_rejects_close_for_stale_observation(self):
        first_payload = _v5_decision_payload(sequence=1)
        self.processor.process(GameStateEvent.model_validate(first_payload))
        update_payload = _v5_decision_payload(sequence=2)
        update_payload["event_id"] = "test-run:2"
        update_payload["options"] = update_payload["options"][:1]
        self.processor.process(GameStateEvent.model_validate(update_payload))

        stale_close = _v5_decision_payload(
            sequence=3,
            event_type="decision_closed",
            outcome={
                "kind": "selected",
                "selected_candidate_id": "1:CHEAP_BLOCK",
                "selected_card": "CHEAP_BLOCK",
                "selected_option_index": 1,
            },
        )
        stale_close["event_id"] = "test-run:3"
        stale_close["parent_event_id"] = "test-run:1"
        with self.assertRaisesRegex(ValueError, "latest observation"):
            self.processor.process(GameStateEvent.model_validate(stale_close))

        self.assertEqual(
            self.checkpoint.load()["current_decision"]["event_id"],
            "test-run:2",
        )

    def test_cross_run_close_is_rejected_without_mutating_parent(self):
        opened_payload = _v5_decision_payload(sequence=1)
        opened_payload["run_id"] = "run-alpha"
        opened_payload["event_id"] = "shared-parent"
        opened_payload["decision_id"] = "run-alpha:decision"
        self.processor.process(GameStateEvent.model_validate(opened_payload))
        checkpoint_before = self.checkpoint.load()

        close_payload = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            decision_id="run-alpha:decision",
            outcome={"kind": "skipped"},
        )
        close_payload["run_id"] = "run-beta"
        close_payload["event_id"] = "run-beta:2"
        close_payload["parent_event_id"] = "shared-parent"
        with self.assertRaisesRegex(
            RealtimeEventValidationError,
            "another run",
        ):
            self.processor.process(GameStateEvent.model_validate(close_payload))

        self.assertEqual(self.checkpoint.load(), checkpoint_before)
        parent = self.sessions.load("shared-parent")
        self.assertIsNone(parent["result"].get("outcome"))

    def test_delayed_previous_run_close_cannot_reclaim_checkpoint(self):
        run_a = _v5_decision_payload(
            sequence=1,
            decision_id="run-alpha:decision",
        )
        run_a["run_id"] = "run-alpha"
        run_a["event_id"] = "run-alpha:1"
        self.processor.process(GameStateEvent.model_validate(run_a))

        run_b = _v5_decision_payload(
            sequence=1,
            decision_id="run-beta:decision",
        )
        run_b["run_id"] = "run-beta"
        run_b["event_id"] = "run-beta:1"
        run_b["emitted_at"] = "2026-07-02T08:01:00+00:00"
        self.processor.process(GameStateEvent.model_validate(run_b))

        delayed = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            decision_id="run-alpha:decision",
            outcome={"kind": "skipped"},
        )
        delayed["run_id"] = "run-alpha"
        delayed["event_id"] = "run-alpha:2"
        delayed["parent_event_id"] = "run-alpha:1"
        restarted = RealtimeEventProcessor(
            self.repository,
            sessions=TransientSessionStore(),
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )
        with self.assertRaisesRegex(ValueError, "another run"):
            restarted.process(GameStateEvent.model_validate(delayed))

        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["run_id"], "run-beta")
        self.assertEqual(
            checkpoint["current_decision"]["decision_id"],
            "run-beta:decision",
        )
        self.assertIsNone(self.sessions.load("run-alpha:1"))

    def test_delayed_old_run_sequence_one_cannot_reclaim_checkpoint(self):
        current = _v5_decision_payload(
            sequence=1,
            decision_id="run-beta:decision",
        )
        current.update({
            "run_id": "run-beta",
            "event_id": "run-beta:1",
            "emitted_at": "2026-07-02T08:02:00+00:00",
        })
        self.processor.process(GameStateEvent.model_validate(current))

        delayed = _v5_decision_payload(
            sequence=1,
            decision_id="run-alpha:decision",
        )
        delayed.update({
            "run_id": "run-alpha",
            "event_id": "run-alpha:1",
            "emitted_at": "2026-07-02T08:01:00+00:00",
        })
        restarted = RealtimeEventProcessor(
            self.repository,
            sessions=TransientSessionStore(),
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )

        with self.assertRaisesRegex(ValueError, "non-newer"):
            restarted.process(GameStateEvent.model_validate(delayed))

        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["run_id"], "run-beta")
        self.assertEqual(
            checkpoint["current_decision"]["decision_id"],
            "run-beta:decision",
        )

    def test_duplicate_close_is_idempotent_and_conflict_preserves_first(self):
        opened_payload = _v5_decision_payload(sequence=1)
        opened = self.processor.process(
            GameStateEvent.model_validate(opened_payload)
        )
        first_close = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={"kind": "skipped"},
        )
        first_close["parent_event_id"] = opened["event_id"]
        self.processor.process(GameStateEvent.model_validate(first_close))

        duplicate = _v5_decision_payload(
            sequence=3,
            event_type="decision_closed",
            outcome={"kind": "skipped"},
        )
        duplicate["event_id"] = "test-run:3"
        duplicate["parent_event_id"] = opened["event_id"]
        duplicate_result = self.processor.process(
            GameStateEvent.model_validate(duplicate)
        )
        self.assertEqual(duplicate_result["status"], "outcome_recorded")
        checkpoint_after_duplicate = self.checkpoint.load()

        conflict = _v5_decision_payload(
            sequence=4,
            event_type="decision_closed",
            outcome={
                "kind": "selected",
                "selected_candidate_id": "0:HEAVY_ATTACK",
                "selected_card": "HEAVY_ATTACK",
                "selected_option_index": 0,
            },
        )
        conflict["event_id"] = "test-run:4"
        conflict["parent_event_id"] = opened["event_id"]
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.processor.process(GameStateEvent.model_validate(conflict))
        self.assertEqual(self.checkpoint.load(), checkpoint_after_duplicate)
        parent = self.sessions.load(opened["event_id"])
        self.assertEqual(parent["result"]["outcome"]["chosen_option"], "skip")

    def test_closed_stable_decision_cannot_reopen_in_same_process(self):
        opened = _v5_decision_payload(sequence=1)
        self.processor.process(GameStateEvent.model_validate(opened))
        closed = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={"kind": "skipped"},
        )
        closed["event_id"] = "test-run:2"
        closed["parent_event_id"] = "test-run:1"
        self.processor.process(GameStateEvent.model_validate(closed))
        repeated = _v5_decision_payload(sequence=3)
        repeated["event_id"] = "test-run:3"

        with patch.object(
            self.processor.policies,
            "recommend",
            side_effect=AssertionError("closed decision reached policy"),
        ):
            with self.assertRaisesRegex(
                RealtimeEventValidationError,
                "cannot be reopened",
            ):
                self.processor.process(
                    GameStateEvent.model_validate(repeated)
                )

        checkpoint = self.checkpoint.load()
        self.assertIsNone(checkpoint["current_decision"])
        self.assertEqual(
            checkpoint["closed_decision"]["decision_id"],
            opened["decision_id"],
        )

    def test_closed_tombstone_survives_host_restart(self):
        opened = _v5_decision_payload(sequence=1)
        self.processor.process(GameStateEvent.model_validate(opened))
        closed = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={"kind": "skipped"},
        )
        closed["event_id"] = "test-run:2"
        closed["parent_event_id"] = "test-run:1"
        self.processor.process(GameStateEvent.model_validate(closed))
        restarted = RealtimeEventProcessor(
            self.repository,
            sessions=TransientSessionStore(),
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )
        repeated = _v5_decision_payload(sequence=3)
        repeated["event_id"] = "test-run:3"

        self.assertEqual(
            restarted.lifecycle.closed("test-run").decision_id,
            opened["decision_id"],
        )
        with self.assertRaisesRegex(
            RealtimeEventValidationError,
            "cannot be reopened",
        ):
            restarted.process(GameStateEvent.model_validate(repeated))

        self.assertEqual(
            self.checkpoint.load()["closed_decision"]["decision_id"],
            opened["decision_id"],
        )

    def test_new_decision_id_replaces_closed_checkpoint_tombstone(self):
        opened = _v5_decision_payload(sequence=1)
        self.processor.process(GameStateEvent.model_validate(opened))
        closed = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={"kind": "skipped"},
        )
        closed["event_id"] = "test-run:2"
        closed["parent_event_id"] = "test-run:1"
        self.processor.process(GameStateEvent.model_validate(closed))
        replacement = _v5_decision_payload(
            sequence=3,
            decision_id="test-run:card-reward:new",
        )
        replacement["event_id"] = "test-run:3"

        result = self.processor.process(
            GameStateEvent.model_validate(replacement)
        )

        self.assertEqual(result["decision_phase"], "opened")
        checkpoint = self.checkpoint.load()
        self.assertEqual(
            checkpoint["current_decision"]["decision_id"],
            "test-run:card-reward:new",
        )
        self.assertIsNone(checkpoint["closed_decision"])

    def test_restarted_host_rejects_same_kind_different_selected_candidate(self):
        opened_payload = _v5_decision_payload(sequence=1)
        opened = self.processor.process(
            GameStateEvent.model_validate(opened_payload)
        )
        selected = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={
                "kind": "selected",
                "selected_candidate_id": "0:HEAVY_ATTACK",
                "selected_card": "HEAVY_ATTACK",
                "selected_option_index": 0,
            },
        )
        selected["parent_event_id"] = opened["event_id"]
        self.processor.process(GameStateEvent.model_validate(selected))
        checkpoint_after_first = self.checkpoint.load()

        restarted = RealtimeEventProcessor(
            self.repository,
            sessions=TransientSessionStore(),
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )
        conflict = _v5_decision_payload(
            sequence=3,
            event_type="decision_closed",
            outcome={
                "kind": "selected",
                "selected_candidate_id": "1:CHEAP_BLOCK",
                "selected_card": "CHEAP_BLOCK",
                "selected_option_index": 1,
            },
        )
        conflict["event_id"] = "test-run:3"
        conflict["parent_event_id"] = opened["event_id"]
        with self.assertRaisesRegex(
            RealtimeEventValidationError,
            "conflicts",
        ):
            restarted.process(GameStateEvent.model_validate(conflict))
        self.assertEqual(self.checkpoint.load(), checkpoint_after_first)

    def test_closed_event_is_observed_without_decision(self):
        opened = self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        closed = GameStateEvent.model_validate(
            _event_payload(2, "decision_closed")
        )
        result = self.processor.process(closed)
        self.assertEqual(result["status"], "observed_unresolved")
        self.assertIsNone(result["advice"])
        self.assertEqual(result["decision_id"], opened["decision_id"])
        self.assertEqual(result["decision_phase"], "closed")

    def test_selected_card_outcome_is_linked_to_parent_decision(self):
        opened = self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        closed = GameStateEvent.model_validate(
            _event_payload(
                2,
                "decision_closed",
                {
                    "kind": "selected",
                    "selected_card": "CHEAP_BLOCK",
                    "selected_option_index": 1,
                },
            )
        )
        result = self.processor.process(closed)

        self.assertEqual(result["status"], "outcome_recorded")
        self.assertEqual(result["decision_id"], opened["decision_id"])
        parent = self.sessions.load("test-run:1")
        self.assertEqual(
            parent["result"]["outcome"]["chosen_option"],
            "1:CHEAP_BLOCK",
        )

    def test_legacy_selected_card_without_index_resolves_unique_option(self):
        opened = self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        closed_payload = _event_payload(
            2,
            "decision_closed",
            {"kind": "selected", "selected_card": "CHEAP_BLOCK"},
        )
        result = self.processor.process(
            GameStateEvent.model_validate(closed_payload)
        )

        self.assertEqual(result["status"], "outcome_recorded")
        parent = self.sessions.load(opened["event_id"])
        self.assertEqual(
            parent["result"]["outcome"]["chosen_option"],
            "1:CHEAP_BLOCK",
        )

    def test_legacy_selected_card_without_index_rejects_ambiguity(self):
        opened_payload = _event_payload()
        opened_payload["options"] = [
            {"card": "HEAVY_ATTACK"},
            {"card": "HEAVY_ATTACK"},
        ]
        self.processor.process(GameStateEvent.model_validate(opened_payload))
        closed_payload = _event_payload(
            2,
            "decision_closed",
            {"kind": "selected", "selected_card": "HEAVY_ATTACK"},
        )

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.processor.process(GameStateEvent.model_validate(closed_payload))

        self.assertIsNotNone(self.checkpoint.load()["current_decision"])

    def test_skip_outcome_is_not_confused_with_unknown_close(self):
        opened = self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        closed = GameStateEvent.model_validate(
            _event_payload(
                2,
                "decision_closed",
                {"kind": "skipped"},
            )
        )
        result = self.processor.process(closed)

        self.assertEqual(result["status"], "outcome_recorded")
        parent = self.sessions.load("test-run:1")
        self.assertEqual(
            parent["result"]["outcome"]["chosen_option"],
            "skip",
        )

    def test_duplicate_claim_reports_processing_during_concurrent_consume(self):
        event = GameStateEvent.model_validate(_event_payload())
        claimed, _ = self.sessions.claim(
            event.model_dump(mode="json")
        )
        self.assertTrue(claimed)
        result = self.processor.process(event)
        self.assertEqual(result["status"], "processing")
        self.assertTrue(result["duplicate"])

    def test_naive_timestamp_does_not_drive_advice(self):
        payload = _event_payload()
        payload["emitted_at"] = "2026-07-02T08:00:00"
        event = GameStateEvent.model_validate(payload)
        with self.assertRaises(RealtimeEventValidationError):
            self.processor.process(event)
        self.assertEqual(self.sessions.count(), 0)

    def test_file_bridge_writes_advice_and_ignores_unchanged_file(self):
        input_path = Path(self.tempdir.name) / "state-event.json"
        output_path = Path(self.tempdir.name) / "advice-event.json"
        input_path.write_text(
            json.dumps(_event_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        bridge = GameStateFileBridge(
            self.processor,
            input_path=input_path,
            output_path=output_path,
        )
        result = bridge.run_once()
        self.assertEqual(result["status"], "processed")
        self.assertEqual(
            result["advice_disposition"]["action"],
            "publish",
        )
        self.assertTrue(output_path.exists())
        written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written["event_id"], "test-run:1")
        self.assertIsNone(bridge.run_once())

    def test_publish_failure_keeps_queue_for_checkpoint_replay(self):
        exchange_dir = Path(self.tempdir.name) / "exchange-publish-retry"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        event_path = events_dir / "001-card.json"
        event_path.write_text(
            json.dumps(_v5_decision_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        output_path = exchange_dir / "advice-event.json"
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=output_path,
        )

        original_replace = os.replace

        def fail_advice_replace(source, destination):
            if Path(destination) == output_path:
                raise PermissionError("advice locked")
            return original_replace(source, destination)

        with patch(
            "realtime.file_bridge.os.replace",
            side_effect=fail_advice_replace,
        ):
            with self.assertRaises(PermissionError):
                bridge.run_once()

        self.assertTrue(event_path.exists())
        self.assertFalse(output_path.exists())
        self.assertEqual(
            self.checkpoint.load()["current_decision"]["decision_id"],
            "test-run:card-reward:stable",
        )

        replayed = bridge.run_once()

        self.assertTrue(replayed["duplicate"])
        self.assertFalse(event_path.exists())
        visible = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            visible["decision_id"],
            "test-run:card-reward:stable",
        )

    def test_clear_failure_keeps_queue_then_replays_same_owner(self):
        exchange_dir = Path(self.tempdir.name) / "exchange-clear-retry"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        output_path = exchange_dir / "advice-event.json"
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=output_path,
        )
        reward_path = events_dir / "001-card.json"
        reward_path.write_text(
            json.dumps(_v5_decision_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        bridge.run_once()
        close = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={"kind": "skipped"},
        )
        close["event_id"] = "test-run:2"
        close["parent_event_id"] = "test-run:1"
        close_path = events_dir / "002-close.json"
        close_path.write_text(
            json.dumps(close, ensure_ascii=False),
            encoding="utf-8",
        )
        original_unlink = Path.unlink
        failed = False

        def fail_visible_advice_once(path, *args, **kwargs):
            nonlocal failed
            if path == output_path and not failed:
                failed = True
                raise PermissionError("advice locked")
            return original_unlink(path, *args, **kwargs)

        with patch.object(
            Path,
            "unlink",
            autospec=True,
            side_effect=fail_visible_advice_once,
        ):
            with self.assertRaises(PermissionError):
                bridge.run_once()

        self.assertTrue(close_path.exists())
        self.assertTrue(output_path.exists())

        replayed = bridge.run_once()

        self.assertTrue(replayed["duplicate"])
        self.assertFalse(close_path.exists())
        self.assertFalse(output_path.exists())

    def test_retried_close_does_not_clear_replacement_owner(self):
        exchange_dir = Path(self.tempdir.name) / "exchange-owner-retry"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        output_path = exchange_dir / "advice-event.json"
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=output_path,
        )
        (events_dir / "001-card.json").write_text(
            json.dumps(_v5_decision_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        bridge.run_once()
        close = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={"kind": "skipped"},
        )
        close["event_id"] = "test-run:2"
        close["parent_event_id"] = "test-run:1"
        close_path = events_dir / "002-close.json"
        close_path.write_text(
            json.dumps(close, ensure_ascii=False),
            encoding="utf-8",
        )
        original_unlink = Path.unlink

        def fail_visible_advice(path, *args, **kwargs):
            if path == output_path:
                raise PermissionError("advice locked")
            return original_unlink(path, *args, **kwargs)

        with patch.object(
            Path,
            "unlink",
            autospec=True,
            side_effect=fail_visible_advice,
        ):
            with self.assertRaises(PermissionError):
                bridge.run_once()

        replacement = {
            "run_id": "new-run",
            "decision_id": "new-run:new-decision",
            "recommendation": {
                "decision_id": "new-run:new-decision",
            },
        }
        output_path.write_text(
            json.dumps(replacement, ensure_ascii=False),
            encoding="utf-8",
        )

        replayed = bridge.run_once()

        self.assertTrue(replayed["duplicate"])
        self.assertFalse(close_path.exists())
        self.assertEqual(
            json.loads(output_path.read_text(encoding="utf-8")),
            replacement,
        )

    def test_file_bridge_consumes_rapid_events_from_ordered_spool(self):
        exchange_dir = Path(self.tempdir.name) / "exchange"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        (events_dir / "test-run-000000000001-card_reward.json").write_text(
            json.dumps(_event_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        (events_dir / "test-run-000000000002-decision_closed.json").write_text(
            json.dumps(
                _event_payload(
                    2,
                    "decision_closed",
                    {"kind": "skipped"},
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=exchange_dir / "advice-event.json",
        )

        opened = bridge.run_once()
        self.assertTrue((exchange_dir / "advice-event.json").exists())
        closed = bridge.run_once()

        self.assertEqual(opened["status"], "processed")
        self.assertEqual(closed["status"], "outcome_recorded")
        self.assertFalse((exchange_dir / "advice-event.json").exists())
        self.assertFalse(any(events_dir.glob("*.json")))
        self.assertFalse((events_dir / "processed").exists())

    def test_unrelated_map_observation_does_not_erase_open_advice(self):
        exchange_dir = Path(self.tempdir.name) / "exchange-preserve"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        reward = _event_payload(1, "card_reward")
        map_event = _event_payload(2, "map_choice")
        map_event["map_context"] = {
            "map_name": "Act_1",
            "player_row": 0,
            "nodes": [{
                "node_id": "0:0",
                "kind": "MONSTER",
                "row": 0,
                "col": 0,
                "edges": [],
            }],
            "node_count": 1,
            "current_node_id": "0:0",
            "available_next_node_ids": [],
            "boss_node_ids": [],
            "boss_encounter_ids": ["VANTOM_BOSS"],
        }
        (events_dir / "001-card.json").write_text(
            json.dumps(reward, ensure_ascii=False),
            encoding="utf-8",
        )
        (events_dir / "002-map.json").write_text(
            json.dumps(map_event, ensure_ascii=False),
            encoding="utf-8",
        )
        output_path = exchange_dir / "advice-event.json"
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=output_path,
        )
        bridge.run_once()
        original = output_path.read_text(encoding="utf-8")
        map_result = bridge.run_once()
        self.assertEqual(map_result["status"], "map_captured")
        self.assertEqual(output_path.read_text(encoding="utf-8"), original)

    def test_unmatched_close_does_not_erase_owned_open_advice(self):
        exchange_dir = Path(self.tempdir.name) / "exchange-unmatched"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        reward = _v5_decision_payload(sequence=1)
        unmatched = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            decision_id="other-decision",
            outcome={"kind": "skipped"},
        )
        unmatched["event_id"] = "test-run:2"
        unmatched["parent_event_id"] = "missing-parent"
        (events_dir / "001-card.json").write_text(
            json.dumps(reward, ensure_ascii=False),
            encoding="utf-8",
        )
        (events_dir / "002-close.json").write_text(
            json.dumps(unmatched, ensure_ascii=False),
            encoding="utf-8",
        )
        output_path = exchange_dir / "advice-event.json"
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=output_path,
        )

        bridge.run_once()
        original = output_path.read_text(encoding="utf-8")
        close_result = bridge.run_once()

        self.assertEqual(close_result["status"], "outcome_unmatched")
        self.assertEqual(
            close_result["advice_disposition"]["action"],
            "preserve",
        )
        self.assertEqual(output_path.read_text(encoding="utf-8"), original)
        self.assertEqual(
            self.checkpoint.load()["current_decision"]["decision_id"],
            reward["decision_id"],
        )

    def test_invalid_matching_close_preserves_owned_open_advice(self):
        exchange_dir = Path(self.tempdir.name) / "exchange-invalid-close"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        reward = _v5_decision_payload(sequence=1)
        reward_path = events_dir / "001-card.json"
        reward_path.write_text(
            json.dumps(reward, ensure_ascii=False),
            encoding="utf-8",
        )
        output_path = exchange_dir / "advice-event.json"
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=output_path,
        )
        bridge.run_once()
        original = output_path.read_text(encoding="utf-8")

        invalid_close = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            outcome={
                "kind": "selected",
                "selected_candidate_id": "not-a-real-candidate",
            },
        )
        invalid_close["event_id"] = "test-run:2"
        invalid_close["parent_event_id"] = "test-run:1"
        (events_dir / "002-invalid-close.json").write_text(
            json.dumps(invalid_close, ensure_ascii=False),
            encoding="utf-8",
        )

        result = bridge.run_once()

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(
            result["advice_disposition"]["action"],
            "preserve",
        )
        self.assertEqual(output_path.read_text(encoding="utf-8"), original)
        self.assertEqual(
            self.checkpoint.load()["current_decision"]["decision_id"],
            reward["decision_id"],
        )

    def test_run_end_clears_memory_and_exchange_files(self):
        exchange_dir = Path(self.tempdir.name) / "exchange"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        latest_path = exchange_dir / "state-event.json"
        output_path = exchange_dir / "advice-event.json"
        (events_dir / "test-run-000000000001-card_reward.json").write_text(
            json.dumps(_event_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        run_end = _event_payload(2, "run_ended")
        (events_dir / "test-run-000000000002-run_ended.json").write_text(
            json.dumps(run_end, ensure_ascii=False),
            encoding="utf-8",
        )
        latest_path.write_text(
            json.dumps(run_end, ensure_ascii=False),
            encoding="utf-8",
        )
        bridge = GameStateFileBridge(
            self.processor,
            input_path=latest_path,
            output_path=output_path,
        )

        self.assertEqual(bridge.run_once()["status"], "processed")
        self.assertTrue(self.checkpoint_path.exists())
        ended = bridge.run_once()

        self.assertEqual(ended["status"], "session_cleared")
        self.assertEqual(self.sessions.count(), 0)
        self.assertFalse(latest_path.exists())
        self.assertFalse(output_path.exists())
        self.assertFalse(events_dir.exists())
        self.assertFalse(self.checkpoint_path.exists())

    def test_cross_run_spool_uses_emitted_time_and_preserves_next_run(self):
        exchange_dir = Path(self.tempdir.name) / "exchange-cross-run"
        events_dir = exchange_dir / "events"
        events_dir.mkdir(parents=True)
        output_path = exchange_dir / "advice-event.json"
        bridge = GameStateFileBridge(
            self.processor,
            input_path=exchange_dir / "state-event.json",
            output_path=output_path,
        )

        opened_a = _event_payload(1, "card_reward")
        opened_a.update({
            "run_id": "zz-run-a",
            "event_id": "zz-run-a:1",
            "emitted_at": "2026-07-02T08:00:00+00:00",
        })
        (events_dir / "zz-run-a-open.json").write_text(
            json.dumps(opened_a, ensure_ascii=False),
            encoding="utf-8",
        )
        self.assertEqual(bridge.run_once()["run_id"], "zz-run-a")
        self.assertTrue(output_path.exists())

        ended_a = _event_payload(2, "run_ended")
        ended_a.update({
            "run_id": "zz-run-a",
            "event_id": "zz-run-a:2",
            "emitted_at": "2026-07-02T08:01:00+00:00",
        })
        opened_b = _event_payload(1, "card_reward")
        opened_b.update({
            "run_id": "aa-run-b",
            "event_id": "aa-run-b:1",
            "emitted_at": "2026-07-02T08:02:00+00:00",
        })
        (events_dir / "zz-run-a-ended.json").write_text(
            json.dumps(ended_a, ensure_ascii=False),
            encoding="utf-8",
        )
        (events_dir / "aa-run-b-open.json").write_text(
            json.dumps(opened_b, ensure_ascii=False),
            encoding="utf-8",
        )

        ended_result = bridge.run_once()
        self.assertEqual(ended_result["event_id"], "zz-run-a:2")
        self.assertEqual(ended_result["status"], "session_cleared")
        self.assertTrue((events_dir / "aa-run-b-open.json").exists())
        self.assertFalse(output_path.exists())

        opened_result = bridge.run_once()
        self.assertEqual(opened_result["event_id"], "aa-run-b:1")
        self.assertEqual(opened_result["status"], "processed")
        visible = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(visible["run_id"], "aa-run-b")
        self.assertFalse(any(events_dir.glob("*.json")))

    def test_map_event_is_checkpointed_without_route_advice(self):
        payload = _event_payload(2, "map_choice")
        payload["map_context"] = {
            "map_name": "Act_1",
            "player_row": 0,
            "current_node_id": "0:0",
            "available_next_node_ids": ["1:0"],
            "boss_node_ids": [],
            "boss_encounter_ids": ["VANTOM_BOSS"],
            "node_count": 2,
            "nodes": [
                {
                    "node_id": "0:0",
                    "kind": "MONSTER",
                    "row": 0,
                    "col": 0,
                    "edges": ["1:0"],
                },
                {
                    "node_id": "1:0",
                    "kind": "ELITE",
                    "row": 1,
                    "col": 0,
                    "edges": [],
                },
            ],
        }
        result = self.processor.process(
            GameStateEvent.model_validate(payload)
        )
        self.assertEqual(result["status"], "map_captured")
        self.assertIsNone(result["advice"])
        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["map_context"]["node_count"], 2)
        self.assertEqual(
            checkpoint["map_context"]["nodes"][0]["edges"],
            ["1:0"],
        )
        self.assertEqual(
            checkpoint["map_context"]["available_next_node_ids"],
            ["1:0"],
        )
        self.assertEqual(
            checkpoint["map_context"]["boss_encounter_ids"],
            ["VANTOM_BOSS"],
        )

    def test_card_reward_never_reuses_checkpointed_route_context(self):
        map_payload = _event_payload(1, "map_choice")
        map_payload["map_context"] = {
            "map_name": "Act_1",
            "player_row": 0,
            "current_node_id": "0:0",
            "available_next_node_ids": ["1:0"],
            "boss_node_ids": [],
            "boss_encounter_ids": ["VANTOM_BOSS"],
            "node_count": 2,
            "nodes": [
                {
                    "node_id": "0:0",
                    "kind": "MONSTER",
                    "row": 0,
                    "col": 0,
                    "edges": ["1:0"],
                },
                {
                    "node_id": "1:0",
                    "kind": "ELITE",
                    "row": 1,
                    "col": 0,
                    "edges": [],
                },
            ],
        }
        self.processor.process(
            GameStateEvent.model_validate(map_payload)
        )

        reward = self.processor.process(
            GameStateEvent.model_validate(_event_payload(2))
        )
        cheap_block = next(
            row
            for row in reward["advice"]["recommendations"]
            if row["card_id"] == "CHEAP_BLOCK"
        )
        codes = {factor["code"] for factor in cheap_block["factors"]}
        self.assertNotIn("route_elite_fit", codes)
        self.assertFalse(reward["advice"]["profile"]["route"]["known"])
        self.assertEqual(
            reward["advice"]["profile"]["route"]["elite_path_ratio"],
            0.0,
        )
        self.assertIsNone(self.checkpoint.load()["map_context"])

    def test_card_reward_respects_cannot_skip_context(self):
        payload = _event_payload()
        payload["decision"]["can_skip"] = False
        result = self.processor.process(
            GameStateEvent.model_validate(payload)
        )
        self.assertFalse(result["advice"]["skip_candidate"]["eligible"])
        self.assertFalse(result["advice"]["skip_recommended"])

    def test_schema_v3_run_end_saves_only_final_summary(self):
        self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        payload = _event_payload(2, "run_ended")
        payload["schema_version"] = 3
        payload["run_result"] = {
            "outcome": "loss",
            "final_score": 321,
            "started_at": "2026-07-02T07:00:00+00:00",
            "ended_at": "2026-07-02T08:30:00+00:00",
        }
        result = self.processor.process(
            GameStateEvent.model_validate(payload)
        )
        self.assertEqual(result["status"], "run_finalized")
        self.assertTrue(result["summary_saved"])
        self.assertFalse(self.checkpoint_path.exists())

        summary = self.repository.load_run_summary("test-run")
        self.assertEqual(summary["outcome"], "loss")
        self.assertEqual(summary["character"], "TEST_HERO")
        self.assertEqual(summary["final_floor"], 3)
        self.assertEqual(summary["final_score"], 321)
        self.assertEqual(
            summary["final_deck"][0]["card"],
            "HEAVY_ATTACK",
        )
        with self.repository.connect() as connection:
            runtime_counts = {
                table: connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table}"
                ).fetchone()["count"]
                for table in (
                    "run_states",
                    "decision_events",
                    "decision_outcomes",
                    "game_state_events",
                )
            }
        self.assertEqual(runtime_counts, {
            "run_states": 0,
            "decision_events": 0,
            "decision_outcomes": 0,
            "game_state_events": 0,
        })

    def test_realtime_pipeline_uses_checkpoint_not_sqlite_history(self):
        result = self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        with self.repository.connect() as connection:
            run_states = connection.execute(
                "SELECT COUNT(*) AS count FROM run_states"
            ).fetchone()["count"]
            run_deck = connection.execute(
                "SELECT COUNT(*) AS count FROM run_deck_cards"
            ).fetchone()["count"]
            run_relics = connection.execute(
                "SELECT COUNT(*) AS count FROM run_relics"
            ).fetchone()["count"]
            run_potions = connection.execute(
                "SELECT COUNT(*) AS count FROM run_potions"
            ).fetchone()["count"]
            decisions = connection.execute(
                "SELECT COUNT(*) AS count FROM decision_events"
            ).fetchone()["count"]
            candidates = connection.execute(
                "SELECT COUNT(*) AS count FROM decision_candidates"
            ).fetchone()["count"]

        self.assertIsNone(result["state_id"])
        self.assertEqual(result["decision_id"], "test-run:1")
        self.assertEqual(run_states, 0)
        self.assertEqual(run_deck, 0)
        self.assertEqual(run_relics, 0)
        self.assertEqual(run_potions, 0)
        self.assertEqual(decisions, 0)
        self.assertEqual(candidates, 0)
        live = self.sessions.latest()
        self.assertEqual(live["decision_id"], result["decision_id"])
        self.assertEqual(live["payload"]["state"]["max_potion_slots"], 3)
        self.assertEqual(
            live["payload"]["state"]["potions"][0]["potion"],
            "TEST_POTION",
        )
        self.assertEqual(
            live["payload"]["state"]["relic_states"][0]["display_amount"],
            2,
        )
        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["state"]["max_potion_slots"], 3)
        self.assertEqual(
            checkpoint["state"]["potions"][0]["potion"],
            "TEST_POTION",
        )

    def test_api_accepts_mod_event_and_exposes_latest(self):
        with patch(
            "api.get_relational_repository",
            return_value=self.repository,
        ), patch(
            "api.get_realtime_sessions",
            return_value=self.sessions,
        ), patch(
            "api.get_realtime_checkpoint",
            return_value=self.checkpoint,
        ), patch(
            "api.KNOWLEDGE_FILE",
            self.knowledge_path,
        ), patch(
            "api.COMMUNITY_SCORES_FILE",
            os.path.join(self.tempdir.name, "missing.json"),
        ):
            client = TestClient(app)
            response = client.post(
                "/events/game-state",
                json=_event_payload(),
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "processed")

            latest = client.get("/events/game-state/latest")
            self.assertEqual(latest.status_code, 200)
            self.assertEqual(
                latest.json()["event"]["event_id"],
                "test-run:1",
            )

            persisted = client.post(
                "/recommend/card-reward",
                json={
                    "state": {
                        "character": "TEST_HERO",
                        "deck": [],
                    },
                    "options": ["HEAVY_ATTACK"],
                    "persist": True,
                },
            )
            self.assertEqual(persisted.status_code, 400)

            routes = client.post(
                "/recommend/paths",
                json={"nodes": []},
            )
            self.assertEqual(routes.status_code, 410)

    def test_protocol_example_and_read_only_mod_invariant(self):
        root = Path(__file__).resolve().parents[1]
        example = json.loads(
            (root / "protocol" / "state-event.example.json").read_text(
                encoding="utf-8"
            )
        )
        schema = json.loads(
            (root / "protocol" / "state-event.schema.json").read_text(
                encoding="utf-8"
            )
        )
        advice_schema = json.loads(
            (root / "protocol" / "advice-event.schema.json").read_text(
                encoding="utf-8"
            )
        )
        advice_example = json.loads(
            (root / "protocol" / "advice-event.example.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema).validate(example)
        Draft202012Validator(advice_schema).validate(advice_example)
        state_candidate_ids = [
            candidate["candidate_id"]
            for candidate in example["candidates"]
            if candidate["kind"] != "skip"
        ]
        advice_candidate_ids = [
            candidate["candidate_id"]
            for candidate in advice_example["recommendation"]["candidates"]
            if candidate["candidate_id"] != "skip"
        ]
        self.assertEqual(advice_example["event_id"], example["event_id"])
        self.assertEqual(advice_example["run_id"], example["run_id"])
        self.assertEqual(advice_example["sequence"], example["sequence"])
        self.assertEqual(
            advice_example["decision_id"],
            example["decision_id"],
        )
        self.assertEqual(
            advice_example["recommendation"]["decision_id"],
            example["decision_id"],
        )
        self.assertEqual(state_candidate_ids, advice_candidate_ids)
        self.assertEqual(
            advice_example["advice_disposition"],
            {
                "action": "publish",
                "run_id": example["run_id"],
                "decision_id": example["decision_id"],
            },
        )
        invalid_v9 = dict(example)
        invalid_v9.pop("decision_id")
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(invalid_v9)
        missing_candidate = json.loads(json.dumps(example))
        missing_candidate["candidates"][0].pop("candidate_id")
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(missing_candidate)
        missing_selected_id = json.loads(json.dumps(example))
        missing_selected_id["event_type"] = "decision_closed"
        missing_selected_id["candidates"] = []
        missing_selected_id["parent_event_id"] = example["event_id"]
        missing_selected_id["outcome"] = {
            "kind": "selected",
            "selected_card": example["candidates"][0]["payload"]["card"],
            "selected_option_index": 0,
        }
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(missing_selected_id)
        event = GameStateEvent.model_validate(example)
        self.assertEqual(
            event.schema_version,
            max(schema["properties"]["schema_version"]["enum"]),
        )
        recommendation_items = advice_schema["$defs"]["recommendation"][
            "properties"
        ]["candidates"]
        self.assertEqual(recommendation_items["minItems"], 1)
        self.assertEqual(recommendation_items["maxItems"], 200)
        self.assertIn("recommendation", advice_schema["required"])
        self.assertEqual(
            advice_schema["$defs"]["recommendation"]["properties"]
            ["contract_version"]["enum"],
            [1, 2],
        )

        mod_root = root / "mod" / "STS2Guide.ReadOnlyExporter"
        manifest = json.loads(
            (
                mod_root / "STS2GuideReadOnlyExporter.json"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in mod_root.glob("*.cs")
        )
        self.assertFalse(manifest["affects_gameplay"])
        self.assertNotIn("HarmonyPrefix", source)
        self.assertIn(
            "AfterShowScreen(\n        NCardRewardSelectionScreen __result",
            source,
        )
        self.assertGreaterEqual(source.count("[HarmonyPostfix]"), 8)

        observer_source = (
            mod_root / "CardRewardObserver.cs"
        ).read_text(encoding="utf-8")
        writer_source = (
            mod_root / "StateEventWriter.cs"
        ).read_text(encoding="utf-8")
        drawer_source = (
            mod_root / "ContextDrawer.cs"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "AfterSkipped(CardReward __instance)",
            observer_source,
        )
        self.assertIn(
            "EmitCardSkipped(__instance)",
            observer_source,
        )
        self.assertIn(
            'typeof(NCardRewardSelectionScreen),\n'
            '    "OnAlternateRewardSelected"',
            observer_source,
        )
        self.assertIn(
            "int index",
            observer_source,
        )
        self.assertIn('"_extraOptions"', observer_source)
        self.assertIn("as IReadOnlyList<CardRewardAlternative>", observer_source)
        self.assertIn('alternative.OptionId,\n                "Skip"', observer_source)
        self.assertIn(
            "EmitCardSkippedFromScreen(__instance)",
            observer_source,
        )
        self.assertNotIn(
            "NRewardsScreen.AfterOverlayClosed",
            observer_source,
        )
        panel_source = (
            mod_root / "CardRewardAdvicePanel.cs"
        ).read_text(encoding="utf-8")
        for needle in (
            '"advice_disposition"',
            'action != "publish"',
            "dispositionRunId != pending.RunId",
            "dispositionDecisionId != pending.DecisionId",
            "!double.IsFinite(numericScore)",
            "numericScore < 0",
            "numericScore > 100",
            "!expected.Contains(recommendedId)",
            "recommendedMatched",
        ):
            self.assertIn(needle, panel_source)
        self.assertNotIn(
            "TryGetOwnedDecisionId",
            observer_source + panel_source,
        )
        self.assertIn(
            "EmitCardSelected(card, __instance)",
            observer_source,
        )
        self.assertIn("BindCardRewardScreen", writer_source)
        self.assertIn(
            "UnbindCardRewardScreen(__instance)",
            observer_source,
        )
        self.assertIn(
            "ReferenceEquals(pending.ScreenOwner, owner)",
            writer_source,
        )
        self.assertIn(
            "Ignoring skip from a stale card",
            writer_source,
        )
        self.assertIn("ScreenOwner = screenOwner", writer_source)
        self.assertLess(
            panel_source.index("BindCardRewardScreen"),
            panel_source.index("ContextDrawer.Show"),
        )
        self.assertIn(
            "pending.DecisionSource,\n"
            "                expectedDecisionSource",
            writer_source,
        )
        self.assertIn(
            "SelectedCandidateId = pending.Options[optionIndex]",
            writer_source,
        )
        self.assertNotIn(
            "SelectedCandidateId = optionIndex >= 0",
            writer_source,
        )
        self.assertIn("ReadResumableDecisionId", writer_source)
        self.assertIn(
            "PayloadStateMatches(payload, state)",
            writer_source,
        )
        self.assertIn("HasLaterInvalidatingEvent", writer_source)
        self.assertIn("new ScrollContainer", drawer_source)
        self.assertIn("AvailableHeight(viewport)", drawer_source)
        self.assertIn(
            "Show is transactional",
            drawer_source,
        )

    def test_state_schema_enforces_processor_decision_requirements(self):
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "protocol" / "state-event.schema.json").read_text(
                encoding="utf-8"
            )
        )
        validator = Draft202012Validator(schema)

        no_options = _v5_decision_payload()
        no_options["options"] = []
        missing_options = _v5_decision_payload()
        missing_options.pop("options", None)
        no_decision = _v5_decision_payload()
        no_decision.pop("decision", None)
        incomplete_close = _v5_decision_payload(
            event_type="decision_closed",
        )
        incomplete_close.pop("parent_event_id", None)
        incomplete_close.pop("outcome", None)
        incomplete_run_end = _event_payload(2, "run_ended")
        incomplete_run_end["schema_version"] = 5

        for payload in (
            no_options,
            missing_options,
            no_decision,
            incomplete_close,
            incomplete_run_end,
        ):
            with self.assertRaises(JsonSchemaValidationError):
                validator.validate(payload)

    def test_mod_run_identity_guard_survives_stale_retry(self):
        root = Path(__file__).resolve().parents[1]
        mod_root = root / "mod" / "STS2Guide.ReadOnlyExporter"
        writer_source = (
            mod_root / "StateEventWriter.cs"
        ).read_text(encoding="utf-8")
        guard_source = (
            mod_root / "RunIdentityGuard.cs"
        ).read_text(encoding="utf-8")
        compact_guard = "".join(guard_source.split())

        self.assertNotIn("_previousRunEnded", writer_source)
        self.assertNotIn("_previousRunId", writer_source)
        self.assertIn("_lastEndedStableRunId", writer_source)
        # BeginRun and the later current-player retry must use the same
        # persistent guard. Abandon/end paths must both arm it.
        self.assertEqual(
            writer_source.count("TryAcceptStableIdentity("),
            2,
        )
        self.assertEqual(
            writer_source.count("RememberEndedStableIdentity("),
            2,
        )
        # A temporary candidate never clears the guard. The exact ended ID
        # is rejected without mutation; only a different stable ID clears it.
        self.assertIn(
            "if(!candidateIsStable){returnfalse;}",
            compact_guard,
        )
        self.assertIn(
            "string.Equals(lastEndedStableRunId,candidateRunId,"
            "StringComparison.Ordinal)",
            compact_guard,
        )
        self.assertIn(
            "lastEndedStableRunId=null;returntrue;",
            compact_guard,
        )
        # A save/continue Launch can temporarily have no stable identity.
        # That is not evidence of a new run and must retain the current ID
        # until the first real state resolves it.
        self.assertIn(
            "hasUnendedState&&currentIsStable&&candidateIsStable",
            compact_guard,
        )
        self.assertIn(
            "hasUnendedState&&currentIsStable&&!candidateIsStable",
            compact_guard,
        )
        self.assertIn("ShouldCloseActiveRun(", writer_source)
        self.assertIn("ShouldRetainCurrentIdentity(", writer_source)
        self.assertIn("_runIdentityLockedByEmission", writer_source)
        self.assertNotIn("if (_sequence > 0)", writer_source)

    def test_mod_resume_guard_matches_state_and_run_end_is_transactional(self):
        root = Path(__file__).resolve().parents[1]
        writer_source = (
            root
            / "mod"
            / "STS2Guide.ReadOnlyExporter"
            / "StateEventWriter.cs"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "TryRecoverDecisionId(options, decision, state)",
            writer_source,
        )
        state_match_start = writer_source.index(
            "private static bool PayloadStateMatches("
        )
        invalidation_start = writer_source.index(
            "private static bool HasLaterInvalidatingEvent("
        )
        state_match = writer_source[
            state_match_start:invalidation_start
        ]
        for property_name in (
            '"character"',
            '"ascension"',
            '"act"',
            '"floor"',
            '"hp"',
            '"max_hp"',
            '"gold"',
            '"energy"',
            '"deck"',
            '"relics"',
            '"relic_states"',
            '"potions"',
            '"max_potion_slots"',
            '"modifiers"',
        ):
            self.assertIn(property_name, state_match)

        invalidation_end = writer_source.index(
            "private static bool TryReadRequiredString("
        )
        invalidation = writer_source[
            invalidation_start:invalidation_end
        ]
        for event_type in (
            '"run_ended"',
            '"decision_closed"',
            '"merchant"',
            '"rest_site"',
            '"card_reward"',
        ):
            self.assertIn(event_type, invalidation)
        self.assertNotIn('"map_choice"', invalidation)

        end_start = writer_source.index(
            "internal static void EmitRunEnded("
        )
        end_stop = writer_source.index(
            "internal static void EmitCardReward("
        )
        end_section = writer_source[end_start:end_stop]
        self.assertLess(
            end_section.index("if (state is null)"),
            end_section.index("var eventId = Write("),
        )
        self.assertLess(
            end_section.index("if (eventId is null)"),
            end_section.index("_ended = true"),
        )
        self.assertLess(
            end_section.index("_ended = true"),
            end_section.index("_lastState = null"),
        )

    def test_unsupported_merchant_recommendation_is_not_publishable_advice(self):
        root = Path(__file__).resolve().parents[1]
        advice_schema = json.loads(
            (root / "protocol" / "advice-event.schema.json").read_text(
                encoding="utf-8"
            )
        )
        class MerchantPolicy:
            decision_type = "merchant"

            @staticmethod
            def recommend(request):
                candidate = request.candidates[0]
                return Recommendation(
                    decision_id=request.decision_id,
                    decision_type=request.decision_type,
                    payload={"merchant_note": "structured-only"},
                    world_sequence=request.world.sequence,
                    policy_version="merchant:test:1",
                    candidates=(CandidateAssessment(
                        candidate_id=candidate.candidate_id,
                        label=candidate.label,
                        display_index=candidate.display_index,
                        eligible=candidate.eligible,
                        score=60,
                        rank=1,
                    ),),
                    recommended_candidate_id=candidate.candidate_id,
                )

        payload = _event_payload(7, "merchant")
        payload["event_id"] = "merchant:event:7"
        event = GameStateEvent.model_validate(payload)
        processor = RealtimeEventProcessor(
            self.repository,
            policies=PolicyRegistry([MerchantPolicy()]),
        )
        request = DecisionRequest.create(
            decision_id="merchant:decision:1",
            decision_type="merchant",
            world=processor._build_world(event),
            candidates=[DecisionCandidate.create(
                "relic:anchor",
                label="锚",
            )],
        )
        envelope = processor._process_decision(event, request)
        envelope.update({
            "run_id": event.run_id,
            "sequence": event.sequence,
            "processed_at": "2026-07-14T00:00:00+00:00",
        })

        self.assertNotIn("advice", envelope)
        with self.assertRaises(JsonSchemaValidationError):
            Draft202012Validator(advice_schema).validate(envelope)

    def test_v5_generic_close_uses_only_canonical_candidate_identity(self):
        processor = self._open_generic_merchant_decision()
        close = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            decision_id="test-run:merchant:stable",
            outcome={
                "kind": "selected",
                "selected_candidate_id": "relic:anchor",
                "selected_card": "INTENTIONALLY_WRONG_LEGACY_CARD",
                "selected_option_index": 9,
            },
        )
        close["event_id"] = "test-run:merchant:2"
        close["parent_event_id"] = "test-run:merchant:1"

        result = processor.process(GameStateEvent.model_validate(close))

        self.assertEqual(result["status"], "outcome_recorded")
        self.assertEqual(result["decision_id"], "test-run:merchant:stable")
        self.assertEqual(
            result["advice_disposition"],
            {
                "action": "clear",
                "run_id": "test-run",
                "decision_id": "test-run:merchant:stable",
            },
        )
        self.assertNotIn("选牌", result["message"])

    def test_v5_generic_close_rejects_ineligible_canonical_candidate(self):
        processor = self._open_generic_merchant_decision(
            include_ineligible=True,
        )
        close = _v5_decision_payload(
            sequence=2,
            event_type="decision_closed",
            decision_id="test-run:merchant:stable",
            outcome={
                "kind": "selected",
                "selected_candidate_id": "relic:forbidden",
            },
        )
        close["event_id"] = "test-run:merchant:2"
        close["parent_event_id"] = "test-run:merchant:1"

        with self.assertRaisesRegex(
            RealtimeEventValidationError,
            "ineligible",
        ):
            processor.process(GameStateEvent.model_validate(close))

        self.assertEqual(
            self.checkpoint.load()["current_decision"]["decision_id"],
            "test-run:merchant:stable",
        )

    # ── Regression: multi-floor, restart, and cross-run isolation ──────

    def test_multi_floor_card_rewards_replace_one_checkpoint(self):
        floors = [3, 7, 12]
        for index, floor in enumerate(floors):
            payload = _event_payload(sequence=floor, event_type="card_reward")
            payload["state"]["floor"] = floor
            payload["event_id"] = f"test-run:{floor}"
            event = GameStateEvent.model_validate(payload)
            result = self.processor.process(event)
            self.assertEqual(result["status"], "processed")
            self.assertIsNone(result["state_id"])
            if index < len(floors) - 1:
                closed = _event_payload(floor + 1, "decision_closed")
                closed["event_id"] = f"test-run:{floor + 1}"
                closed["parent_event_id"] = event.event_id
                closed["outcome"] = {"kind": "skipped"}
                self.processor.process(
                    GameStateEvent.model_validate(closed)
                )

        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["state"]["floor"], 12)
        self.assertEqual(checkpoint["last_sequence"], 12)
        with self.repository.connect() as connection:
            count = connection.execute(
                "SELECT COUNT(*) AS count FROM run_states"
            ).fetchone()["count"]
        self.assertEqual(count, 0)

    def test_game_restart_does_not_leak_previous_run_state(self):
        # Run 1
        payload_a = _event_payload(1, "card_reward")
        payload_a["run_id"] = "run-alpha"
        event_a = GameStateEvent.model_validate(payload_a)
        self.processor.process(event_a)
        self.assertEqual(self.checkpoint.load()["run_id"], "run-alpha")

        # End run 1
        end_a = _event_payload(2, "run_ended")
        end_a["run_id"] = "run-alpha"
        self.processor.process(GameStateEvent.model_validate(end_a))

        # Run 2 (new game)
        payload_b = _event_payload(1, "card_reward")
        payload_b["run_id"] = "run-beta"
        payload_b["event_id"] = "run-beta:1"
        event_b = GameStateEvent.model_validate(payload_b)
        self.processor.process(event_b)

        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["run_id"], "run-beta")
        self.assertEqual(checkpoint["state"]["floor"], 3)
        with self.repository.connect() as connection:
            states = connection.execute(
                "SELECT COUNT(*) AS count FROM run_states"
            ).fetchone()["count"]
        self.assertEqual(states, 0)

    def test_open_decision_survives_local_service_restart(self):
        opened = self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        restarted_sessions = TransientSessionStore()
        restarted_processor = RealtimeEventProcessor(
            self.repository,
            sessions=restarted_sessions,
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )

        closed = GameStateEvent.model_validate(
            _event_payload(
                2,
                "decision_closed",
                {
                    "kind": "selected",
                    "selected_card": "CHEAP_BLOCK",
                    "selected_option_index": 1,
                },
            )
        )
        result = restarted_processor.process(closed)
        self.assertEqual(result["status"], "outcome_recorded")
        self.assertEqual(result["decision_id"], opened["decision_id"])
        checkpoint = self.checkpoint.load()
        self.assertIsNone(checkpoint["current_decision"])
        with self.repository.connect() as connection:
            outcomes = connection.execute(
                "SELECT COUNT(*) AS count FROM decision_outcomes"
            ).fetchone()["count"]
        self.assertEqual(outcomes, 0)

    def test_corrupt_checkpoint_is_discarded_on_restart(self):
        self.checkpoint_path.write_text("{broken", encoding="utf-8")

        with self.assertLogs("sts2-guide", level="WARNING") as captured:
            restarted = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
            )

        self.assertIsNotNone(restarted)
        self.assertFalse(self.checkpoint_path.exists())
        self.assertTrue(any(
            "unreadable" in message
            and "defaults to balanced" in message
            for message in captured.output
        ))

    def test_unsupported_checkpoint_version_is_discarded(self):
        self.checkpoint_path.write_text(
            json.dumps({
                "checkpoint_version": 999,
                "guide_preferences": {"route_mode": "growth"},
            }),
            encoding="utf-8",
        )

        with self.assertLogs("sts2-guide", level="WARNING") as captured:
            loaded = self.checkpoint.load()

        self.assertIsNone(loaded)
        self.assertFalse(self.checkpoint_path.exists())
        self.assertTrue(any(
            "unsupported_version" in message
            for message in captured.output
        ))

    def test_v1_checkpoint_recovers_legacy_skippable_decision(self):
        self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        checkpoint = self.checkpoint.load()
        checkpoint["checkpoint_version"] = 1
        checkpoint["open_decision"] = checkpoint.pop("current_decision")
        checkpoint.pop("closed_decision", None)
        checkpoint["open_decision"]["result"].pop("recommendation", None)
        self.checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False),
            encoding="utf-8",
        )
        restarted = RealtimeEventProcessor(
            self.repository,
            sessions=TransientSessionStore(),
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )
        closed = _event_payload(
            2,
            "decision_closed",
            {"kind": "skipped"},
        )

        result = restarted.process(GameStateEvent.model_validate(closed))

        self.assertEqual(result["status"], "outcome_recorded")
        self.assertEqual(result["decision_id"], "test-run:1")
        self.assertIsNone(self.checkpoint.load()["current_decision"])

    def test_duplicate_reward_survives_local_service_restart(self):
        opened = self.processor.process(
            GameStateEvent.model_validate(_event_payload())
        )
        restarted = RealtimeEventProcessor(
            self.repository,
            sessions=TransientSessionStore(),
            checkpoint=ActiveRunCheckpointStore(self.checkpoint_path),
        )
        duplicate = restarted.process(
            GameStateEvent.model_validate(_event_payload())
        )
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(
            duplicate["decision_id"],
            opened["decision_id"],
        )

    def test_consecutive_runs_have_isolated_decision_ids(self):
        # Run 1: card reward + select
        r1 = _event_payload(1, "card_reward")
        r1["run_id"] = "run-one"
        r1["event_id"] = "run-one:1"
        d1 = self.processor.process(
            GameStateEvent.model_validate(r1)
        )
        c1 = _event_payload(2, "decision_closed")
        c1["run_id"] = "run-one"
        c1["event_id"] = "run-one:2"
        c1["parent_event_id"] = "run-one:1"
        c1["outcome"] = {"kind": "skipped"}
        self.processor.process(GameStateEvent.model_validate(c1))
        end1 = _event_payload(3, "run_ended")
        end1["run_id"] = "run-one"
        end1["event_id"] = "run-one:3"
        self.processor.process(GameStateEvent.model_validate(end1))

        # Run 2: card reward + select
        r2 = _event_payload(1, "card_reward")
        r2["run_id"] = "run-two"
        r2["event_id"] = "run-two:1"
        d2 = self.processor.process(
            GameStateEvent.model_validate(r2)
        )
        c2 = _event_payload(2, "decision_closed")
        c2["run_id"] = "run-two"
        c2["event_id"] = "run-two:2"
        c2["parent_event_id"] = "run-two:1"
        c2["outcome"] = {
            "kind": "selected",
            "selected_card": "CHEAP_BLOCK",
            "selected_option_index": 1,
        }
        self.processor.process(GameStateEvent.model_validate(c2))

        self.assertNotEqual(d1["decision_id"], d2["decision_id"])
        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["run_id"], "run-two")
        self.assertIsNone(checkpoint["current_decision"])
        with self.repository.connect() as connection:
            decisions = connection.execute(
                "SELECT COUNT(*) AS count FROM decision_events"
            ).fetchone()["count"]
        self.assertEqual(decisions, 0)

    # -- Single-instance lock & atomic write -----------------------------------

    def test_instance_lock_prevents_duplicate_host(self):
        from realtime.host import _acquire_instance_lock
        with tempfile.TemporaryDirectory() as td:
            exchange = Path(td) / "exchange"
            # First acquisition succeeds (atomic O_CREAT|O_EXCL).
            _acquire_instance_lock(exchange)
            lock_file = exchange / ".host.lock"
            self.assertTrue(lock_file.exists())
            # Second call from same process fails because lock exists
            # and PID matches ours (would be a re-entry guard, but
            # the atomic create fails first).
            # To test real blocking: use a subprocess.
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10)"]
            )
            try:
                # Write lock with child's PID (simulates another host).
                lock_file.unlink()
                lock_file.write_text(str(child.pid), encoding="utf-8")
                with self.assertRaises(RuntimeError) as ctx:
                    _acquire_instance_lock(exchange)
                self.assertIn("already", str(ctx.exception))
            finally:
                child.kill()
                child.wait()

    def test_instance_lock_clears_after_stale_pid(self):
        from realtime.host import _acquire_instance_lock
        with tempfile.TemporaryDirectory() as td:
            exchange = Path(td) / "exchange"
            exchange.mkdir(parents=True)
            (exchange / ".host.lock").write_text("99999999", encoding="utf-8")
            # Stale PID should not block — the lock is acquired.
            _acquire_instance_lock(exchange)

    def test_instance_lock_is_released_by_owner(self):
        from realtime.host import (
            _acquire_instance_lock,
            _release_instance_lock,
        )
        with tempfile.TemporaryDirectory() as td:
            exchange = Path(td) / "exchange"
            _acquire_instance_lock(exchange)
            self.assertTrue((exchange / ".host.lock").exists())
            _release_instance_lock(exchange)
            self.assertFalse((exchange / ".host.lock").exists())

    def test_windowed_host_supplies_argparse_stdio_sinks(self):
        from realtime.host import _ensure_windowed_stdio

        with patch.object(sys, "stdout", None), patch.object(sys, "stderr", None):
            _ensure_windowed_stdio()
            self.assertIsNotNone(sys.stdout)
            self.assertIsNotNone(sys.stderr)
            self.assertTrue(sys.stdout.writable())
            self.assertTrue(sys.stderr.writable())
            sys.stdout.close()
            sys.stderr.close()

    def test_pyinstaller_spec_bundles_conda_runtime_dlls(self):
        spec = (
            Path(__file__).resolve().parents[1]
            / "packaging"
            / "sts2-guide.spec"
        ).read_text(encoding="utf-8")
        self.assertIn('python_runtime_bin / "ffi.dll"', spec)
        self.assertIn('python_runtime_bin / "libexpat.dll"', spec)
        self.assertIn('binaries=[(str(path), ".")', spec)
        self.assertIn('"advice-event.schema.json"', spec)
        build_script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_p0_exe.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("--startup-check", build_script)
        self.assertIn(".host.lock", build_script)

    def test_atomic_write_retries_on_permission_error(self):
        from realtime.file_bridge import _atomic_write_json
        call_count = 0
        original_replace = os.replace

        def _flaky_replace(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise PermissionError("transient")
            original_replace(src, dst)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.json"
            with patch("os.replace", side_effect=_flaky_replace):
                _atomic_write_json(dest, {"key": "value"})
            self.assertEqual(call_count, 3)
            self.assertTrue(dest.exists())

    def test_atomic_write_fails_after_all_retries(self):
        from realtime.file_bridge import _atomic_write_json
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "test.json"
            with patch("os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    _atomic_write_json(dest, {"key": "value"})

    def test_checkpoint_atomic_write_retries_on_permission_error(self):
        from realtime.checkpoint import _atomic_write_json
        call_count = 0
        original_replace = os.replace

        def _flaky(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise PermissionError("transient")
            original_replace(src, dst)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "checkpoint.json"
            with patch("os.replace", side_effect=_flaky):
                _atomic_write_json(dest, {"k": "v"})
            self.assertEqual(call_count, 3)
            self.assertTrue(dest.exists())

    def test_checkpoint_atomic_write_fails_after_all_retries(self):
        from realtime.checkpoint import _atomic_write_json
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "checkpoint.json"
            with patch("os.replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    _atomic_write_json(dest, {"k": "v"})


if __name__ == "__main__":
    unittest.main()
