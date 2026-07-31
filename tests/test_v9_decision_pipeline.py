"""End-to-end v9 event -> policy -> Recommendation coverage."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
import unittest
from statistics import quantiles
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.processor import (
    RealtimeEventProcessor,
    RealtimeEventValidationError,
)
from realtime.protocol import GameStateEvent
from realtime.session import TransientSessionStore
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]


def _effect(
    kind: str,
    *,
    amount: int | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    target_mode: str = "none",
    certainty: str = "exact",
    child_decision_type: str | None = None,
) -> dict:
    return {
        "kind": kind,
        "amount": amount,
        "min_amount": None,
        "max_amount": None,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "target_mode": target_mode,
        "certainty": certainty,
        "source_code": "test:v9_pipeline",
        "child_decision_type": child_decision_type,
    }


def _stable_source_id(candidate_id: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    return "SOURCE_" + digest[:24].upper()


class V9DecisionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "pipeline.db")
        )
        cls.repository.sync_catalog(ROOT / "data" / "knowledge.json")
        cls.base = json.loads(
            (ROOT / "protocol" / "state-event.example.json").read_text(
                encoding="utf-8"
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def _event(
        self,
        event_type: str,
        candidates: list[dict],
        *,
        parent: dict | None = None,
        source: str | None = None,
        run_id: str | None = None,
        sequence: int | None = None,
        decision_id: str | None = None,
    ) -> GameStateEvent:
        actual_run_id = run_id or f"pipeline-{event_type}"
        actual_sequence = sequence or (3 if parent is not None else 1)
        payload = copy.deepcopy(self.base)
        payload.update({
            "event_id": f"{actual_run_id}:{event_type}:{actual_sequence}",
            "event_type": event_type,
            "run_id": actual_run_id,
            "decision_id": decision_id or f"pipeline:{event_type}:decision",
            "sequence": actual_sequence,
            "state_revision": actual_sequence,
            "options": [],
            "candidates": candidates,
            "decision_parent": parent,
            "map_context": None,
        })
        payload["decision"] = {
            "can_skip": False,
            "can_reroll": False,
            "reward_source": source or event_type.upper(),
        }
        return GameStateEvent.model_validate(payload)

    def _prime_parent(
        self,
        processor: RealtimeEventProcessor,
        child: GameStateEvent,
    ) -> None:
        parent = child.decision_parent
        self.assertIsNotNone(parent)
        parent_candidate_id = str(parent.candidate_id)
        parent_event = self._event(
            "rest_site",
            [{
                "candidate_id": parent_candidate_id,
                "kind": "rest_action",
                "entity_id": None,
                "label": "锻造",
                "eligible": True,
                "unavailable_reason": None,
                "costs": [],
                "payload": {
                    "action_id": "SMITH",
                    "effects": [_effect(
                        "upgrade_card",
                        entity_type="cards",
                        target_mode="choose",
                    )],
                },
            }],
            run_id=child.run_id,
            sequence=1,
            decision_id=parent.decision_id,
        )
        processor.process(parent_event)
        self._close_parent(processor, parent_event, parent_candidate_id)

    def _close_parent(
        self,
        processor: RealtimeEventProcessor,
        parent_event: GameStateEvent,
        selected_candidate_id: str,
    ) -> dict:
        closed_payload = parent_event.model_dump(mode="json")
        close_sequence = parent_event.sequence + 1
        closed_payload.update({
            "event_id": (
                f"{parent_event.run_id}:decision_closed:{close_sequence}"
            ),
            "event_type": "decision_closed",
            "sequence": close_sequence,
            "state_revision": close_sequence,
            "decision": None,
            "candidates": [],
            "decision_parent": None,
            "parent_event_id": parent_event.event_id,
            "outcome": {
                "kind": "selected",
                "selected_candidate_id": selected_candidate_id,
                "selected_card": None,
                "selected_option_index": None,
            },
        })
        return processor.process(GameStateEvent.model_validate(closed_payload))

    def _process(self, event: GameStateEvent) -> tuple[dict, float]:
        processor = RealtimeEventProcessor(self.repository)
        if event.decision_parent is not None:
            self._prime_parent(processor, event)
        started = time.perf_counter()
        result = processor.process(event)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return result, elapsed_ms

    def _cases(self) -> list[tuple[str, str, list[dict], dict | None]]:
        return [
            (
                "merchant",
                "merchant_choice",
                [
                    {
                        "candidate_id": "merchant:card:0:SHRUG_IT_OFF",
                        "kind": "merchant_offer",
                        "entity_id": "SHRUG_IT_OFF",
                        "label": "耸肩无视",
                        "eligible": True,
                        "unavailable_reason": None,
                        "costs": [{
                            "kind": "gold",
                            "amount": 30,
                            "resource_id": None,
                        }],
                        "payload": {
                            "slot_id": "merchant:card:0",
                            "offer_kind": "card",
                            "is_stocked": True,
                            "effects": [],
                        },
                    },
                    {
                        "candidate_id": "leave",
                        "kind": "leave",
                        "entity_id": None,
                        "label": "离开",
                        "eligible": True,
                        "unavailable_reason": None,
                        "costs": [],
                        "payload": {},
                    },
                ],
                None,
            ),
            (
                "rest_site",
                "campfire_action",
                [{
                    "candidate_id": "rest:heal",
                    "kind": "rest_action",
                    "entity_id": None,
                    "label": "休息",
                    "eligible": True,
                    "unavailable_reason": None,
                    "costs": [],
                    "payload": {
                        "action_id": "HEAL",
                        "effects": [_effect("hp_delta", amount=20)],
                    },
                }],
                None,
            ),
            (
                "neow_choice",
                "neow_blessing",
                [{
                    "candidate_id": "neow:stage-1:gold",
                    "kind": "neow_blessing",
                    "entity_id": None,
                    "label": "获得金币",
                    "eligible": True,
                    "unavailable_reason": None,
                    "costs": [],
                    "payload": {
                        "blessing_id": "GOLD",
                        "stage_id": "STAGE_1",
                        "option_id": "GOLD",
                        "effects": [_effect("gold_delta", amount=100)],
                    },
                }],
                None,
            ),
            (
                "event_choice",
                "event_option",
                [{
                    "candidate_id": "ABYSSAL_BATHS:INITIAL:IMMERSE",
                    "kind": "event_option",
                    "entity_id": None,
                    "label": "进入浴池",
                    "eligible": True,
                    "unavailable_reason": None,
                    "costs": [],
                    "payload": {
                        "event_id": "ABYSSAL_BATHS",
                        "page_id": "INITIAL",
                        "option_id": "IMMERSE",
                        "effects": [_effect("no_op")],
                    },
                }],
                None,
            ),
            (
                "deck_edit",
                "deck_edit",
                [{
                    "candidate_id": "upgrade:STRIKE_IRONCLAD",
                    "kind": "deck_edit",
                    "entity_id": "STRIKE_IRONCLAD",
                    "label": "打击",
                    "eligible": True,
                    "unavailable_reason": None,
                    "costs": [],
                    "payload": {
                        "operation": "upgrade",
                        "target_candidate_ids": [],
                    },
                }],
                {
                    "decision_id": "pipeline:rest:parent",
                    "candidate_id": "rest:smith",
                    "source_type": "rest_site",
                    "source_id": _stable_source_id("rest:smith"),
                },
            ),
        ]

    def test_every_new_decision_dispatches_to_one_canonical_policy(self):
        elapsed: list[float] = []
        for event_type, decision_type, candidates, parent in self._cases():
            with self.subTest(event_type=event_type):
                result, elapsed_ms = self._process(
                    self._event(
                        event_type,
                        candidates,
                        parent=parent,
                    )
                )
                elapsed.append(elapsed_ms)
                self.assertEqual(result["status"], "processed")
                self.assertEqual(
                    result["recommendation"]["decision_type"],
                    decision_type,
                )
                self.assertEqual(
                    [
                        row["candidate_id"]
                        for row in result["recommendation"]["candidates"]
                    ],
                    [candidate["candidate_id"] for candidate in candidates],
                )
                self.assertEqual(
                    result["advice_disposition"]["action"],
                    "publish",
                )
        self.assertLess(
            max(elapsed),
            300.0,
            f"slowest v9 decision took {max(elapsed):.1f} ms",
        )

    def test_new_decision_suite_p95_remains_within_realtime_budget(self):
        elapsed: list[float] = []
        for _ in range(20):
            for event_type, _, candidates, parent in self._cases():
                _, elapsed_ms = self._process(
                    self._event(
                        event_type,
                        copy.deepcopy(candidates),
                        parent=copy.deepcopy(parent),
                    )
                )
                elapsed.append(elapsed_ms)
        p95 = quantiles(elapsed, n=100, method="inclusive")[94]
        self.assertLess(
            p95,
            300.0,
            f"v9 decision suite P95 was {p95:.1f} ms",
        )

    def test_merchant_purchase_recomputes_same_decision_as_updated(self):
        event_type, _, candidates, _ = self._cases()[0]
        processor = RealtimeEventProcessor(self.repository)
        opened = self._event(event_type, copy.deepcopy(candidates))
        first = processor.process(opened)
        updated_payload = opened.model_dump(mode="json")
        updated_payload["event_id"] = "pipeline:merchant:2"
        updated_payload["sequence"] = 2
        updated_payload["state_revision"] = 2
        offer = updated_payload["candidates"][0]
        offer["eligible"] = False
        offer["unavailable_reason"] = "sold_out"
        offer["payload"]["is_stocked"] = False
        updated = processor.process(
            GameStateEvent.model_validate(updated_payload)
        )
        self.assertEqual(first["decision_phase"], "opened")
        self.assertEqual(updated["decision_phase"], "updated")
        self.assertEqual(
            updated["decision_id"],
            first["decision_id"],
        )
        self.assertIsNone(
            updated["recommendation"]["candidates"][0]["score"]
        )
        advice_schema = json.loads(
            (ROOT / "protocol" / "advice-event.schema.json").read_text(
                encoding="utf-8"
            )
        )
        advice_example = json.loads(
            (ROOT / "protocol" / "advice-event.example.json").read_text(
                encoding="utf-8"
            )
        )
        updated.update({
            "run_id": opened.run_id,
            "sequence": 2,
            "processed_at": "2026-08-01T00:00:00+00:00",
            "compatibility": advice_example["compatibility"],
        })
        Draft202012Validator(advice_schema).validate(updated)

    def test_nested_decision_rejects_unverified_or_wrong_parent(self):
        _, _, candidates, valid_parent = self._cases()[-1]
        child = self._event(
            "deck_edit",
            copy.deepcopy(candidates),
            parent=copy.deepcopy(valid_parent),
        )
        with self.assertRaisesRegex(
            RealtimeEventValidationError,
            "no verified recently closed parent",
        ):
            RealtimeEventProcessor(self.repository).process(child)

        mutations = (
            ("candidate_id", "rest:wrong"),
            ("source_id", "SOURCE_WRONG"),
            ("decision_id", "pipeline:rest:wrong"),
        )
        for field, value in mutations:
            with self.subTest(parent_field=field):
                mutated = child.model_dump(mode="json")
                mutated["decision_parent"][field] = value
                processor = RealtimeEventProcessor(self.repository)
                self._prime_parent(processor, child)
                with self.assertRaisesRegex(
                    RealtimeEventValidationError,
                    "does not match its latest closed parent",
                ):
                    processor.process(GameStateEvent.model_validate(mutated))

        wrong_operation = child.model_dump(mode="json")
        wrong_operation["candidates"][0]["payload"]["operation"] = "remove"
        processor = RealtimeEventProcessor(self.repository)
        self._prime_parent(processor, child)
        with self.assertRaisesRegex(
            RealtimeEventValidationError,
            "operation does not match parent expectation",
        ):
            processor.process(GameStateEvent.model_validate(wrong_operation))

    def test_nested_parent_survives_checkpoint_restart(self):
        _, _, candidates, valid_parent = self._cases()[-1]
        child = self._event(
            "deck_edit",
            copy.deepcopy(candidates),
            parent=copy.deepcopy(valid_parent),
        )
        with tempfile.TemporaryDirectory() as tempdir:
            checkpoint = ActiveRunCheckpointStore(
                Path(tempdir) / "active-run.json"
            )
            first = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=checkpoint,
            )
            self._prime_parent(first, child)
            closed = checkpoint.load()["closed_decision"]
            expectation = closed["result"]["child_expectation"]
            self.assertEqual(expectation["child_decision_type"], "deck_edit")
            self.assertEqual(expectation["operation"], "upgrade")

            restarted = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=checkpoint,
            )
            result = restarted.process(child)
            self.assertEqual(result["status"], "processed")
            self.assertEqual(
                result["recommendation"]["decision_type"],
                "deck_edit",
            )

            current = checkpoint.load()["current_decision"]
            self.assertIn("parent_close", current)
            self.assertEqual(
                current["result"]["active_child_binding"][
                    "child_decision_id"
                ],
                child.decision_id,
            )
            update_payload = child.model_dump(mode="json")
            update_payload.update({
                "event_id": f"{child.run_id}:deck_edit:4",
                "sequence": 4,
                "state_revision": 4,
            })
            update = GameStateEvent.model_validate(update_payload)
            after_child_restart = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=checkpoint,
            )
            updated = after_child_restart.process(update)
            self.assertEqual(updated["decision_phase"], "updated")

            omitted_parent = update.model_dump(mode="json")
            omitted_parent.update({
                "event_id": f"{child.run_id}:deck_edit:5",
                "sequence": 5,
                "state_revision": 5,
                "decision_parent": None,
            })
            with self.assertRaises(ValidationError):
                GameStateEvent.model_validate(omitted_parent)

            closed_child = self._close_parent(
                after_child_restart,
                update,
                update.candidates[0].candidate_id,
            )
            self.assertEqual(closed_child["decision_phase"], "closed")
            self.assertNotIn(
                child.run_id,
                after_child_restart._active_child_bindings,
            )
            self.assertIsNone(checkpoint.load()["current_decision"])

    def test_checkpoint_rederives_parent_and_rejects_tampering(self):
        _, _, candidates, valid_parent = self._cases()[-1]
        child = self._event(
            "deck_edit",
            copy.deepcopy(candidates),
            parent=copy.deepcopy(valid_parent),
        )
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "active-run.json"
            checkpoint = ActiveRunCheckpointStore(path)
            processor = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=checkpoint,
            )
            self._prime_parent(processor, child)
            saved = checkpoint.load()
            self.assertIn(
                "parent_observation",
                saved["closed_decision"],
            )
            saved["closed_decision"]["result"]["child_expectation"][
                "operation"
            ] = "remove"
            path.write_text(json.dumps(saved), encoding="utf-8")
            restarted = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=checkpoint,
            )
            with self.assertRaisesRegex(
                RealtimeEventValidationError,
                "no verified recently closed parent",
            ):
                restarted.process(child)

    def test_active_child_binding_rejects_tamper_and_clears_on_close(self):
        _, _, candidates, valid_parent = self._cases()[-1]
        child = self._event(
            "deck_edit",
            copy.deepcopy(candidates),
            parent=copy.deepcopy(valid_parent),
        )
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "active-run.json"
            checkpoint = ActiveRunCheckpointStore(path)
            processor = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=checkpoint,
            )
            self._prime_parent(processor, child)
            processor.process(child)
            saved = checkpoint.load()
            saved["current_decision"]["result"][
                "active_child_binding"
            ]["operation"] = "remove"
            path.write_text(json.dumps(saved), encoding="utf-8")
            update_payload = child.model_dump(mode="json")
            update_payload.update({
                "event_id": f"{child.run_id}:deck_edit:4",
                "sequence": 4,
                "state_revision": 4,
            })
            restarted = RealtimeEventProcessor(
                self.repository,
                sessions=TransientSessionStore(),
                checkpoint=checkpoint,
            )
            with self.assertRaisesRegex(
                RealtimeEventValidationError,
                "no verified recently closed parent",
            ):
                restarted.process(
                    GameStateEvent.model_validate(update_payload)
                )

        processor = RealtimeEventProcessor(self.repository)
        self._prime_parent(processor, child)
        processor.process(child)
        self._close_parent(
            processor,
            child,
            child.candidates[0].candidate_id,
        )
        self.assertNotIn(child.run_id, processor._active_child_bindings)

    def test_deck_edit_validates_every_candidate_operation(self):
        _, _, candidates, valid_parent = self._cases()[-1]
        candidates = copy.deepcopy(candidates)
        second = copy.deepcopy(candidates[0])
        second.update({
            "candidate_id": "remove:DEFEND_IRONCLAD",
            "entity_id": "DEFEND_IRONCLAD",
            "eligible": False,
            "unavailable_reason": "locked",
        })
        second["payload"]["operation"] = "remove"
        candidates.append(second)
        child = self._event(
            "deck_edit",
            candidates,
            parent=copy.deepcopy(valid_parent),
        )
        processor = RealtimeEventProcessor(self.repository)
        self._prime_parent(processor, child)
        with self.assertRaisesRegex(
            RealtimeEventValidationError,
            "operation does not match parent expectation",
        ):
            processor.process(child)

    def test_followup_choice_requires_explicit_exact_child_type(self):
        parent_candidate_id = "neow:stage-1:cards"
        child_parent = {
            "decision_id": "pipeline:neow:parent",
            "candidate_id": parent_candidate_id,
            "source_type": "neow_choice",
            "source_id": _stable_source_id(parent_candidate_id),
        }
        card_candidates = copy.deepcopy(self.base["candidates"][:3])
        child = self._event(
            "card_reward",
            card_candidates,
            parent=child_parent,
            run_id="pipeline-nested-card",
            source="NEOW",
        )
        for certainty, child_type, should_accept in (
            ("unknown", None, False),
            ("exact", "card_reward", True),
        ):
            with self.subTest(certainty=certainty):
                parent_event = self._event(
                    "neow_choice",
                    [{
                        "candidate_id": parent_candidate_id,
                        "kind": "neow_blessing",
                        "entity_id": None,
                        "label": "选择一张牌",
                        "eligible": True,
                        "unavailable_reason": None,
                        "costs": [],
                        "payload": {
                            "blessing_id": "CHOOSE_CARD",
                            "stage_id": "STAGE_1",
                            "option_id": "CHOOSE_CARD",
                            "effects": [_effect(
                                "followup_choice",
                                certainty=certainty,
                                child_decision_type=child_type,
                            )],
                        },
                    }],
                    run_id=child.run_id,
                    sequence=1,
                    decision_id=child_parent["decision_id"],
                    source="NEOW",
                )
                processor = RealtimeEventProcessor(self.repository)
                processor.process(parent_event)
                close_result = self._close_parent(
                    processor,
                    parent_event,
                    parent_candidate_id,
                )
                if should_accept:
                    self.assertEqual(
                        close_result["child_expectation"][
                            "child_decision_type"
                        ],
                        "card_reward",
                    )
                    self.assertEqual(
                        processor.process(child)["status"],
                        "processed",
                    )
                else:
                    self.assertIsNone(close_result["child_expectation"])
                    with self.assertRaisesRegex(
                        RealtimeEventValidationError,
                        "no verified recently closed parent",
                    ):
                        processor.process(child)

    def test_card_reward_parent_requires_exact_neow_or_event_source(self):
        card_candidates = copy.deepcopy(self.base["candidates"][:3])
        cases = (
            (
                "neow_choice",
                "neow_blessing",
                "NEOW",
                "neow:stage-1:cards",
                {
                    "blessing_id": "CHOOSE_CARD",
                    "stage_id": "STAGE_1",
                    "option_id": "CHOOSE_CARD",
                },
            ),
            (
                "event_choice",
                "event_option",
                "EVENT",
                "TEST_EVENT:INITIAL:CARDS",
                {
                    "event_id": "TEST_EVENT",
                    "page_id": "INITIAL",
                    "option_id": "CARDS",
                },
            ),
        )
        for event_type, kind, reward_source, candidate_id, identity in cases:
            with self.subTest(parent=event_type):
                run_id = f"pipeline-{event_type}-nested-card"
                parent_decision_id = f"{run_id}:parent"
                parent_context = {
                    "decision_id": parent_decision_id,
                    "candidate_id": candidate_id,
                    "source_type": event_type,
                    "source_id": _stable_source_id(candidate_id),
                }
                parent_event = self._event(
                    event_type,
                    [{
                        "candidate_id": candidate_id,
                        "kind": kind,
                        "entity_id": None,
                        "label": "选择一张牌",
                        "eligible": True,
                        "unavailable_reason": None,
                        "costs": [],
                        "payload": {
                            **identity,
                            "effects": [_effect(
                                "followup_choice",
                                child_decision_type="card_reward",
                            )],
                        },
                    }],
                    run_id=run_id,
                    sequence=1,
                    decision_id=parent_decision_id,
                    source=reward_source,
                )
                child = self._event(
                    "card_reward",
                    copy.deepcopy(card_candidates),
                    parent=parent_context,
                    run_id=run_id,
                    source=reward_source,
                )
                processor = RealtimeEventProcessor(self.repository)
                processor.process(parent_event)
                close = self._close_parent(
                    processor,
                    parent_event,
                    candidate_id,
                )
                self.assertEqual(
                    close["child_expectation"]["child_decision_type"],
                    "card_reward",
                )
                self.assertEqual(processor.process(child)["status"], "processed")

                wrong_child = child.model_dump(mode="json")
                wrong_child["decision"]["reward_source"] = "MONSTER"
                processor = RealtimeEventProcessor(self.repository)
                processor.process(parent_event)
                self._close_parent(processor, parent_event, candidate_id)
                with self.assertRaises(ValidationError):
                    GameStateEvent.model_validate(wrong_child)

                wrong_parent = parent_event.model_dump(mode="json")
                wrong_parent["decision"]["reward_source"] = "MONSTER"
                processor = RealtimeEventProcessor(self.repository)
                wrong_parent_event = GameStateEvent.model_validate(wrong_parent)
                processor.process(wrong_parent_event)
                close = self._close_parent(
                    processor,
                    wrong_parent_event,
                    candidate_id,
                )
                self.assertIsNone(close["child_expectation"])

        rest_candidate_id = "rest:special-card"
        rest = self._event(
            "rest_site",
            [{
                "candidate_id": rest_candidate_id,
                "kind": "rest_action",
                "entity_id": None,
                "label": "特殊选牌",
                "eligible": True,
                "unavailable_reason": None,
                "costs": [],
                "payload": {
                    "action_id": "SPECIAL_CARD",
                    "effects": [_effect(
                        "followup_choice",
                        child_decision_type="card_reward",
                    )],
                },
            }],
            run_id="pipeline-rest-followup-card",
            source="REST_SITE",
        )
        processor = RealtimeEventProcessor(self.repository)
        processor.process(rest)
        close = self._close_parent(processor, rest, rest_candidate_id)
        self.assertIsNone(close["child_expectation"])


if __name__ == "__main__":
    unittest.main()
