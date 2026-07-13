import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app
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

    def test_card_reward_event_is_processed_once(self):
        event = GameStateEvent.model_validate(_event_payload())
        first = self.processor.process(event)
        second = self.processor.process(event)

        self.assertEqual(first["status"], "processed")
        self.assertEqual(first["advice"]["recommended_option"], "轻防测试牌")
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["decision_id"], second["decision_id"])
        self.assertEqual(self.sessions.count(), 1)

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
            checkpoint["open_decision"]["event_id"],
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
            "CHEAP_BLOCK",
        )

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
        self.assertTrue(output_path.exists())
        written = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written["event_id"], "test-run:1")
        self.assertIsNone(bridge.run_once())

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
        closed = bridge.run_once()

        self.assertEqual(opened["status"], "processed")
        self.assertEqual(closed["status"], "outcome_recorded")
        self.assertFalse(any(events_dir.glob("*.json")))
        self.assertFalse((events_dir / "processed").exists())

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

    def test_card_reward_reuses_checkpointed_route_context(self):
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
        self.assertIn("route_elite_fit", codes)
        self.assertEqual(
            reward["advice"]["profile"]["route"]["elite_path_ratio"],
            1.0,
        )

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
        event = GameStateEvent.model_validate(example)
        self.assertEqual(
            event.schema_version,
            max(schema["properties"]["schema_version"]["enum"]),
        )
        recommendation_items = advice_schema["properties"]["advice"][
            "properties"
        ]["recommendations"]
        self.assertEqual(recommendation_items["minItems"], 1)
        self.assertEqual(recommendation_items["maxItems"], 10)

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

    # ── Regression: multi-floor, restart, and cross-run isolation ──────

    def test_multi_floor_card_rewards_replace_one_checkpoint(self):
        floors = [3, 7, 12]
        for floor in floors:
            payload = _event_payload(sequence=floor, event_type="card_reward")
            payload["state"]["floor"] = floor
            payload["event_id"] = f"test-run:{floor}"
            event = GameStateEvent.model_validate(payload)
            result = self.processor.process(event)
            self.assertEqual(result["status"], "processed")
            self.assertIsNone(result["state_id"])

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
        self.assertIsNone(checkpoint["open_decision"])
        with self.repository.connect() as connection:
            outcomes = connection.execute(
                "SELECT COUNT(*) AS count FROM decision_outcomes"
            ).fetchone()["count"]
        self.assertEqual(outcomes, 0)

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
        self.assertIsNone(checkpoint["open_decision"])
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
