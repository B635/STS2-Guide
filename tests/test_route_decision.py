"""Contract, lifecycle, safety, and performance coverage for TASK-007."""
from __future__ import annotations

import copy
import json
import math
import os
import socket
import statistics
import tempfile
import time
from datetime import datetime, timedelta, timezone
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator

from advisor.decision_core import (
    CARD_REWARD,
    ROUTE_CHOICE,
    CandidateAssessment,
    DecisionCandidate,
    DecisionRequest,
    Recommendation,
    WorldState,
)
from advisor.route import RoutePolicy, validate_route_graph
from realtime.checkpoint import ActiveRunCheckpointStore
import realtime.checkpoint as checkpoint_module
from realtime.file_bridge import GameStateFileBridge
from realtime.processor import RealtimeEventProcessor, RealtimeEventValidationError
from realtime.protocol import GameStateEvent
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "protocol"


def _csharp_method(source: str, signature: str) -> str:
    """Return one balanced C# method body for source-level safety contracts.

    The gameplay assembly is intentionally not launched by this test suite.
    These checks therefore assert the ordering and scope of the fail-closed
    guards rather than merely checking that unrelated keywords occur in the
    same source file.
    """
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unbalanced C# method: {signature}")


def _risk_profile(**overrides):
    profile = {
        "version": "route-risk-v1",
        "kind_weights": {
            "MONSTER": 0.0, "ELITE": 5.5, "CAMPFIRE": -1.8,
            "SHOP": -1.0, "EVENT": 1.0, "TREASURE": -2.0,
            "BOSS": 5.0, "UNKNOWN": 3.0,
        },
        "hp_low_threshold": 0.45,
        "hp_critical_threshold": 0.25,
        "elite_hp_penalty": 3.0,
        "boss_path_bonus": 2.0,
        "rest_low_hp_bonus": 5.0,
        "shop_gold_threshold": 150,
        "encounter_expected_risk": {
            "MONSTER": 2.2, "ELITE": 6.6, "BOSS": 9.5,
        },
        "capability_weights": {
            "immediate_power": 0.08,
            "survival": 0.07,
            "aoe": 0.7,
            "growth": 0.5,
            "resource": 0.4,
        },
    }
    profile.update(overrides)
    return profile


def _catalog():
    return {
        "characters": [{
            "id": "TEST_HERO", "name": "测试角色", "description": "",
            "embed_text": "角色",
        }],
        "cards": [
            {
                "id": "FRONT", "name": "前置输出", "description": "造成15点伤害。",
                "cost": 1, "type_key": "Attack", "rarity_key": "Common",
                "color": "test", "damage": 15, "block": None,
                "keywords_key": [], "embed_text": "前置输出",
            },
            {
                "id": "BLOCK", "name": "防御", "description": "获得12点格挡。",
                "cost": 1, "type_key": "Skill", "rarity_key": "Common",
                "color": "test", "damage": None, "block": 12,
                "keywords_key": [], "embed_text": "防御",
            },
            {
                "id": "AOE", "name": "横扫", "description": "对所有敌人造成6点伤害。",
                "target": "AllEnemies", "cost": 1, "type_key": "Attack",
                "rarity_key": "Common", "color": "test", "damage": 6,
                "block": None, "keywords_key": [], "embed_text": "横扫",
            },
            {
                "id": "GROWTH", "name": "成长", "description": "每当使用攻击牌时，获得1点力量。",
                "cost": 1, "type_key": "Power", "rarity_key": "Uncommon",
                "color": "test", "damage": None, "block": None,
                "keywords_key": [], "embed_text": "成长",
            },
            {
                "id": "RESOURCE", "name": "资源", "description": "抽2张牌，获得1点能量。",
                "cost": 1, "type_key": "Skill", "rarity_key": "Common",
                "color": "test", "damage": None, "block": None,
                "cards_draw": 2, "energy_gain": 1, "keywords_key": [],
                "embed_text": "资源",
            },
            {
                "id": "MULTI", "name": "多段攻击", "description": "造成4点伤害3次。",
                "cost": 1, "type_key": "Attack", "rarity_key": "Common",
                "color": "test", "damage": 4, "block": None, "hit_count": 3,
                "upgrade": {"damage": "+2", "hit_count": "+1"},
                "keywords_key": [], "embed_text": "多段攻击",
            },
        ],
        "relics": [{
            "id": "SUPPORT_RELIC", "name": "支援遗物",
            "description": "攻击牌和技能牌提供格挡并获得能量。",
            "pool": "test", "rarity_key": "Common", "embed_text": "支援遗物",
        }],
        "potions": [{
            "id": "TEST_POTION", "name": "测试药水", "description": "获得格挡。",
            "embed_text": "测试药水",
        }],
        "monsters": [
            {
                "id": "NORMAL_MONSTER", "name": "普通敌人", "type": "Monster",
                "min_hp": 30, "max_hp": 40, "min_hp_ascension": 36, "max_hp_ascension": 48, "moves": [{
                    "id": "HIT", "name": "攻击", "intent": "Attack",
                    "damage": {"normal": 8, "ascension": 10, "hit_count": 1},
                    "block": None, "heal": None, "powers": [],
                }], "attack_pattern": {}, "embed_text": "普通敌人",
            },
            {
                "id": "ELITE_MONSTER", "name": "精英敌人", "type": "Elite",
                "min_hp": 70, "max_hp": 80, "min_hp_ascension": 84, "max_hp_ascension": 96, "moves": [{
                    "id": "SLAM", "name": "猛击", "intent": "Attack",
                    "damage": {"normal": 18, "ascension": 20, "hit_count": 1},
                    "block": None, "heal": None, "powers": [],
                }], "attack_pattern": {}, "embed_text": "精英敌人",
            },
            {
                "id": "TEST_BOSS", "name": "测试首领", "type": "Boss",
                "min_hp": 120, "max_hp": 140, "min_hp_ascension": 150, "max_hp_ascension": 170, "moves": [{
                    "id": "CRUSH", "name": "重击", "intent": "Attack",
                    "damage": {"normal": 24, "ascension": 28, "hit_count": 1},
                    "block": None, "heal": None, "powers": [],
                }], "attack_pattern": {}, "embed_text": "测试首领",
            },
            {
                "id": "HIGH_BOSS", "name": "高压首领", "type": "Boss",
                "min_hp": 280, "max_hp": 320, "min_hp_ascension": 350, "max_hp_ascension": 390, "moves": [{
                    "id": "SMASH", "name": "重压", "intent": "Attack",
                    "damage": {"normal": 48, "ascension": 60, "hit_count": 2},
                    "block": None, "heal": None, "powers": [],
                }], "attack_pattern": {}, "embed_text": "高压首领",
            },
        ],
        "encounters": [
            {"id": "NORMAL_POOL", "name": "普通池", "room_type": "Monster", "act": "Test", "is_weak": False,
             "monsters": [{"id": "NORMAL_MONSTER", "name": "普通敌人"}], "embed_text": "普通池"},
            {"id": "ELITE_POOL", "name": "精英池", "room_type": "Elite", "act": "Test", "is_weak": False,
             "monsters": [{"id": "ELITE_MONSTER", "name": "精英敌人"}], "embed_text": "精英池"},
            {"id": "TEST_BOSS_ENCOUNTER", "name": "首领池", "room_type": "Boss", "act": "Test", "is_weak": False,
             "monsters": [{"id": "TEST_BOSS", "name": "测试首领"}], "embed_text": "首领池"},
            {"id": "HIGH_BOSS_ENCOUNTER", "name": "高压首领池", "room_type": "Boss", "act": "Test", "is_weak": False,
             "monsters": [{"id": "HIGH_BOSS", "name": "高压首领"}], "embed_text": "高压首领池"},
        ],
        "acts": [{
            "id": "TEST_ACT", "name": "测试章节", "num_rooms": 15,
            "bosses": ["TEST_BOSS_ENCOUNTER", "HIGH_BOSS_ENCOUNTER"], "ancients": [], "events": [],
            "encounters": ["NORMAL_POOL", "ELITE_POOL", "TEST_BOSS_ENCOUNTER", "HIGH_BOSS_ENCOUNTER"],
            "embed_text": "测试章节",
        }],
        "mechanics": {"route_risk_profile_v1": _risk_profile()},
    }


def _map_context(*, candidates=("1:2", "1:4"), nodes=None, boss_ids=("3:3",)):
    if nodes is None:
        nodes = [
            {"node_id": "0:3", "kind": "MONSTER", "row": 0, "col": 3, "edges": list(candidates)},
            {"node_id": "1:2", "kind": "ELITE", "row": 1, "col": 2, "edges": ["2:2"]},
            {"node_id": "1:4", "kind": "CAMPFIRE", "row": 1, "col": 4, "edges": ["2:4"]},
            {"node_id": "2:2", "kind": "MONSTER", "row": 2, "col": 2, "edges": ["3:3"]},
            {"node_id": "2:4", "kind": "SHOP", "row": 2, "col": 4, "edges": ["3:3"]},
            {"node_id": "3:3", "kind": "BOSS", "row": 3, "col": 3, "edges": []},
        ]
    return {
        "map_name": "Test Act", "player_row": 0, "current_node_id": "0:3",
        "origin_node_id": "0:3", "available_next_node_ids": list(candidates),
        "boss_node_ids": list(boss_ids),
        "boss_encounter_ids": ["TEST_BOSS_ENCOUNTER"],
        "node_count": len(nodes), "nodes": nodes,
    }


def _state(**overrides):
    state = {
        "character": "TEST_HERO", "ascension": 0, "act": 1, "floor": 3,
        "hp": 70, "max_hp": 80, "gold": 180, "energy": 3,
        "deck": [{"card": "FRONT", "count": 1}],
        "relics": ["SUPPORT_RELIC"], "relic_states": [],
        "potions": [{"potion": "TEST_POTION", "slot": 0}],
        "max_potion_slots": 3, "modifiers": [], "capture_warnings": [],
    }
    state.update(overrides)
    return state


def _route_event(*, sequence=3, decision_id="route-decision", run_id="route-run", context=None, state=None):
    context = context or _map_context()
    return {
        "schema_version": 6,
        "event_id": f"{run_id}:{sequence}",
        "event_type": "route_choice",
        "emitted_at": "2026-07-16T08:00:00+00:00",
        "run_id": run_id,
        "sequence": sequence,
        "decision_id": decision_id,
        "state": state or _state(),
        "options": [],
        "decision": {"can_skip": False, "can_reroll": False, "reward_source": "MAP"},
        "map_context": context,
    }


def _route_request(
    context=None,
    *,
    state=None,
    decision_id="route-decision",
    route_mode="balanced",
):
    context = context or _map_context()
    world = WorldState.create(
        run_id="route-run",
        sequence=3,
        state=state or _state(),
        map_context=context,
        route_mode=route_mode,
    )
    nodes = {node["node_id"]: node for node in context["nodes"]}
    return DecisionRequest.create(
        decision_id=decision_id,
        decision_type=ROUTE_CHOICE,
        world=world,
        candidates=[
            DecisionCandidate.create(node_id, nodes.get(node_id, {}), label=node_id)
            for node_id in context["available_next_node_ids"]
        ],
    )


class RouteDecisionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repository = RelationalRepository(str(Path(self.tempdir.name) / "route.db"))
        self.repository.ensure_schema()
        self.knowledge_path = Path(self.tempdir.name) / "knowledge.json"
        self.knowledge_path.write_text(json.dumps(_catalog(), ensure_ascii=False), encoding="utf-8")
        self.repository.sync_catalog(str(self.knowledge_path))
        self.checkpoint = ActiveRunCheckpointStore(Path(self.tempdir.name) / "active-run.json")

    def tearDown(self):
        self.tempdir.cleanup()

    def _processor(self, *, checkpoint=True):
        return RealtimeEventProcessor(
            self.repository,
            checkpoint=self.checkpoint if checkpoint else None,
        )

    def test_legacy_v1_to_v5_fixtures_validate_and_parse(self):
        schema = json.loads((ROOT / "protocol" / "state-event.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        for version in range(1, 6):
            payload = json.loads((FIXTURES / f"state-v{version}.json").read_text(encoding="utf-8"))
            validator.validate(payload)
            event = GameStateEvent.model_validate(payload)
            self.assertEqual(event.schema_version, version)

    def test_v6_route_contract_rejects_pre_v6_and_incomplete_context(self):
        schema = json.loads((ROOT / "protocol" / "state-event.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        valid = _route_event()
        validator.validate(valid)
        GameStateEvent.model_validate(valid)
        invalids = []
        for version in range(1, 6):
            payload = copy.deepcopy(valid)
            payload["schema_version"] = version
            invalids.append(payload)
        for key, value in (
            ("decision_id", None), ("options", [{"card": "BAD"}]),
        ):
            payload = copy.deepcopy(valid)
            payload[key] = value
            invalids.append(payload)
        for key, value in (
            ("origin_node_id", None), ("nodes", []),
            ("available_next_node_ids", []), ("boss_node_ids", []),
            ("available_next_node_ids", [""]),
            ("boss_node_ids", ["1:boss", "1:boss"]),
            ("boss_encounter_ids", ["", "BOSS_A"]),
            ("boss_encounter_ids", ["BOSS_A", "BOSS_A"]),
        ):
            payload = copy.deepcopy(valid)
            payload["map_context"][key] = value
            invalids.append(payload)
        # Defaults in the Pydantic model must not let a producer omit fields
        # that schema v6 makes explicit route facts.
        missing_options = copy.deepcopy(valid)
        missing_options.pop("options")
        invalids.append(missing_options)
        missing_reroll = copy.deepcopy(valid)
        missing_reroll["decision"].pop("can_reroll")
        invalids.append(missing_reroll)
        missing_boss_encounters = copy.deepcopy(valid)
        missing_boss_encounters["map_context"].pop("boss_encounter_ids")
        invalids.append(missing_boss_encounters)
        for payload in invalids:
            with self.assertRaises(Exception):
                validator.validate(payload)
            with self.assertRaises(Exception):
                GameStateEvent.model_validate(payload)

    def test_route_advice_and_card_v1_v2_contracts_are_schema_valid(self):
        advice_schema = json.loads((ROOT / "protocol" / "advice-event.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(advice_schema)
        route = json.loads((ROOT / "protocol" / "route-advice-event.example.json").read_text(encoding="utf-8"))
        validator.validate(route)
        card = json.loads((ROOT / "protocol" / "advice-event.example.json").read_text(encoding="utf-8"))
        validator.validate(card)
        card_v2 = copy.deepcopy(card)
        card_v2["recommendation"]["contract_version"] = 2
        validator.validate(card_v2)
        invalid_card_presentation = copy.deepcopy(card_v2)
        invalid_card_presentation["recommendation"]["presentation"] = {
            "kind": "route_paths",
        }
        with self.assertRaises(Exception):
            validator.validate(invalid_card_presentation)
        invalid = copy.deepcopy(route)
        presentation = invalid["recommendation"]["presentation"]
        presentation["primary_path"] = presentation.pop("primary_path_node_ids")
        with self.assertRaises(Exception):
            validator.validate(invalid)
        mismatch = copy.deepcopy(route)
        mismatch["event_type"] = "card_reward"
        with self.assertRaises(Exception):
            validator.validate(mismatch)
        uncertain_with_paths = copy.deepcopy(route)
        uncertain_with_paths["recommendation"]["recommended_candidate_id"] = None
        with self.assertRaises(Exception):
            validator.validate(uncertain_with_paths)
        invalid_primary = copy.deepcopy(route)
        invalid_primary["recommendation"]["presentation"]["primary_path_node_ids"] = []
        with self.assertRaises(Exception):
            validator.validate(invalid_primary)
        invalid_empty_recommended = copy.deepcopy(route)
        invalid_empty_recommended["recommendation"]["recommended_candidate_id"] = ""
        with self.assertRaises(Exception):
            validator.validate(invalid_empty_recommended)
        invalid_null_origin = copy.deepcopy(route)
        invalid_null_origin["recommendation"]["presentation"]["origin_node_id"] = None
        with self.assertRaises(Exception):
            validator.validate(invalid_null_origin)
        for target, field, value in (
            ("presentation", "unexpected", True),
            ("path", "unexpected", True),
            ("path", "score", "68"),
        ):
            invalid = copy.deepcopy(route)
            destination = (
                invalid["recommendation"]["presentation"]
                if target == "presentation"
                else invalid["recommendation"]["presentation"]["paths"][0]
            )
            destination[field] = value
            with self.assertRaises(Exception):
                validator.validate(invalid)
        invalid_event_type = copy.deepcopy(route)
        invalid_event_type["event_type"] = "merchant"
        with self.assertRaises(Exception):
            validator.validate(invalid_event_type)
        for path, value in (
            (("status",), "processing"),
            (("recommendation", "status"), "maybe"),
            (("recommendation", "confidence"), "certain-ish"),
        ):
            invalid = copy.deepcopy(route)
            destination = invalid
            for part in path[:-1]:
                destination = destination[part]
            destination[path[-1]] = value
            with self.assertRaises(Exception):
                validator.validate(invalid)
        duplicate_gap = copy.deepcopy(route)
        duplicate_gap["recommendation"]["data_gaps"] = [
            "map_incomplete",
            "map_incomplete",
        ]
        with self.assertRaises(Exception):
            validator.validate(duplicate_gap)
        for status, recommended in (
            ("recommend", None),
            ("recommend", "skip"),
            ("uncertain", "1:4"),
            ("skip", "1:4"),
        ):
            invalid = copy.deepcopy(route)
            invalid["recommendation"]["status"] = status
            invalid["recommendation"]["recommended_candidate_id"] = recommended
            with self.assertRaises(Exception):
                validator.validate(invalid)
        ineligible = copy.deepcopy(route)
        ineligible["recommendation"]["candidates"][0]["eligible"] = False
        ineligible["recommendation"]["candidates"][0]["rank"] = None
        with self.assertRaises(Exception):
            validator.validate(ineligible)
        for field in ("rank", "factors", "dimensions", "data_gaps"):
            incomplete = copy.deepcopy(route)
            incomplete["recommendation"]["candidates"][0].pop(field)
            with self.assertRaises(Exception):
                validator.validate(incomplete)

    def test_recommendation_contract_allows_card_v2_but_guards_route_paths(self):
        candidate = CandidateAssessment("card", "卡", 0, True, 50.0, 1)
        card_v2 = Recommendation(
            decision_id="card-decision", decision_type=CARD_REWARD, payload={},
            world_sequence=1, policy_version="test", candidates=(candidate,),
            recommended_candidate_id="card", contract_version=2,
        )
        self.assertNotIn("presentation", card_v2.as_dict())
        with self.assertRaises(ValueError):
            Recommendation(
                decision_id="route", decision_type=ROUTE_CHOICE, payload={},
                world_sequence=1, policy_version="test", candidates=(candidate,),
                recommended_candidate_id="card", contract_version=2,
                presentation={"kind": "route_paths", "origin_node_id": "0:0", "primary_path_node_ids": [], "backup_path_node_ids": [], "paths": []},
            )
        valid_route = Recommendation(
            decision_id="route", decision_type=ROUTE_CHOICE, payload={},
            world_sequence=1, policy_version="test", candidates=(candidate,),
            recommended_candidate_id="card", contract_version=2,
            presentation={
                "kind": "route_paths", "origin_node_id": "0:0",
                "primary_path_node_ids": ["card", "boss"],
                "backup_path_node_ids": [],
                "paths": [{
                    "candidate_id": "card", "node_ids": ["card", "boss"],
                    "score": 50.0,
                }],
            },
        )
        self.assertEqual(valid_route.presentation["paths"][0]["candidate_id"], "card")
        for field, value in (
            ("score", float("nan")),
            ("score", True),
            ("display_index", True),
            ("rank", True),
        ):
            values = {
                "candidate_id": "card",
                "label": "卡",
                "display_index": 0,
                "eligible": True,
                "score": 50.0,
                "rank": 1,
            }
            values[field] = value
            with self.assertRaises(ValueError):
                CandidateAssessment(**values)
        with self.assertRaises(ValueError):
            Recommendation(
                decision_id="card-decision",
                decision_type=CARD_REWARD,
                payload={},
                world_sequence=1,
                policy_version="test",
                candidates=(
                    CandidateAssessment(
                        "card",
                        "卡",
                        0,
                        True,
                        None,
                        1,
                    ),
                ),
                recommended_candidate_id="card",
            )
        with self.assertRaises(ValueError):
            Recommendation(
                decision_id="route", decision_type=ROUTE_CHOICE, payload={},
                world_sequence=1, policy_version="test", candidates=(candidate,),
                recommended_candidate_id="card", contract_version=2,
                presentation={
                    "kind": "route_paths", "origin_node_id": "0:0",
                    "primary_path_node_ids": ["card"],
                    "backup_path_node_ids": [],
                    "paths": [{
                        "candidate_id": "card", "node_ids": ["wrong"], "score": 50.0,
                    }],
                },
            )
        with self.assertRaises(ValueError):
            Recommendation(
                decision_id="route", decision_type=ROUTE_CHOICE, payload={},
                world_sequence=1, policy_version="test", candidates=(candidate,),
                recommended_candidate_id="card", contract_version=2,
                presentation={
                    "kind": "route_paths", "origin_node_id": "0:0",
                    "primary_path_node_ids": ["card"],
                    "backup_path_node_ids": [],
                    "paths": [{
                        "candidate_id": "card", "node_ids": ["card"],
                        "score": "50", "unexpected": True,
                    }],
                    "unexpected": True,
                },
            )

    def test_route_open_update_close_and_latest_parent(self):
        processor = self._processor()
        opened_event = GameStateEvent.model_validate(_route_event(sequence=3))
        opened = processor.process(opened_event)
        self.assertEqual(opened["decision_phase"], "opened")
        updated_payload = _route_event(sequence=4, state=_state(hp=42))
        updated = processor.process(GameStateEvent.model_validate(updated_payload))
        self.assertEqual(updated["decision_phase"], "updated")
        checkpoint = self.checkpoint.load()
        self.assertEqual(checkpoint["current_decision"]["event_id"], "route-run:4")
        close = {
            "schema_version": 6, "event_id": "route-run:5", "event_type": "decision_closed",
            "emitted_at": "2026-07-16T08:00:02+00:00", "run_id": "route-run", "sequence": 5,
            "decision_id": "route-decision", "state": _state(hp=42), "options": [],
            "parent_event_id": "route-run:4",
            "outcome": {"kind": "selected", "selected_candidate_id": "1:4"},
        }
        closed = processor.process(GameStateEvent.model_validate(close))
        self.assertEqual(closed["decision_phase"], "closed")
        self.assertIsNone(self.checkpoint.load()["current_decision"])

    def test_route_checkpoint_restart_restores_and_close_rejects_stale_or_illegal(self):
        opened = _route_event(sequence=3)
        self._processor().process(GameStateEvent.model_validate(opened))
        restarted = self._processor()
        stale_close = {
            "schema_version": 6, "event_id": "route-run:4", "event_type": "decision_closed",
            "emitted_at": "2026-07-16T08:00:03+00:00", "run_id": "route-run", "sequence": 4,
            "decision_id": "route-decision", "state": _state(), "options": [],
            "parent_event_id": "not-the-parent",
            "outcome": {"kind": "selected", "selected_candidate_id": "1:2"},
        }
        stale = restarted.process(GameStateEvent.model_validate(stale_close))
        self.assertEqual(stale["status"], "outcome_unmatched")
        illegal = copy.deepcopy(stale_close)
        illegal["event_id"] = "route-run:5"
        illegal["sequence"] = 5
        illegal["parent_event_id"] = "route-run:3"
        illegal["outcome"]["selected_candidate_id"] = "not-a-node"
        with self.assertRaises(RealtimeEventValidationError):
            restarted.process(GameStateEvent.model_validate(illegal))
        valid = copy.deepcopy(stale_close)
        valid["event_id"] = "route-run:6"
        valid["sequence"] = 6
        valid["parent_event_id"] = "route-run:3"
        closed = restarted.process(GameStateEvent.model_validate(valid))
        self.assertEqual(closed["decision_phase"], "closed")

    def test_route_cross_run_late_close_and_new_origin_do_not_share_identity(self):
        processor = self._processor()
        processor.process(GameStateEvent.model_validate(_route_event(sequence=3)))
        cross = _route_event(sequence=1, run_id="other-run", decision_id="other-route")
        cross["emitted_at"] = "2026-07-16T09:00:00+00:00"
        cross["state"] = _state()
        fresh = processor.process(GameStateEvent.model_validate(cross))
        self.assertEqual(fresh["decision_phase"], "opened")
        self.assertEqual(self.checkpoint.load()["run_id"], "other-run")
        # A delayed close from the replaced run cannot delete the current
        # run's pending decision, even if its sequence is locally newer.
        old_run_close = {
            "schema_version": 6, "event_id": "route-run:4", "event_type": "decision_closed",
            "emitted_at": "2026-07-16T09:00:00+00:00", "run_id": "route-run", "sequence": 4,
            "decision_id": "route-decision", "state": _state(), "options": [],
            "parent_event_id": "route-run:3",
            "outcome": {"kind": "selected", "selected_candidate_id": "1:2"},
        }
        with self.assertRaises(ValueError):
            processor.process(GameStateEvent.model_validate(old_run_close))
        self.assertEqual(self.checkpoint.load()["run_id"], "other-run")
        self.assertEqual(
            self.checkpoint.load()["current_decision"]["decision_id"],
            "other-route",
        )
        close = {
            "schema_version": 6, "event_id": "other-run:2", "event_type": "decision_closed",
            "emitted_at": "2026-07-16T09:00:01+00:00", "run_id": "other-run", "sequence": 2,
            "decision_id": "other-route", "state": _state(), "options": [],
            "parent_event_id": "other-run:1",
            "outcome": {"kind": "selected", "selected_candidate_id": "1:2"},
        }
        self.assertEqual(
            processor.process(GameStateEvent.model_validate(close))["decision_phase"],
            "closed",
        )
        # A different verified origin is a new opportunity even in the same run.
        context = _map_context()
        context["origin_node_id"] = "1:2"
        context["current_node_id"] = "1:2"
        context["available_next_node_ids"] = ["2:2"]
        event = _route_event(sequence=3, run_id="other-run", decision_id="new-origin", context=context)
        event["emitted_at"] = "2026-07-16T09:00:02+00:00"
        newer = processor.process(GameStateEvent.model_validate(event))
        self.assertEqual(newer["decision_phase"], "opened")
        self.assertEqual(newer["decision_id"], "new-origin")

    def test_map_preview_does_not_create_route_or_erase_card_advice(self):
        processor = self._processor()
        card_event = {
            "schema_version": 5, "event_id": "route-run:1", "event_type": "card_reward",
            "emitted_at": "2026-07-16T08:00:00+00:00", "run_id": "route-run", "sequence": 1,
            "decision_id": "card-decision", "state": _state(),
            "options": [{"candidate_id": "0:FRONT", "card": "FRONT"}],
            "decision": {"can_skip": True, "can_reroll": False, "reward_source": "COMBAT"},
        }
        processor.process(GameStateEvent.model_validate(card_event))
        preview = copy.deepcopy(card_event)
        preview.update({"event_id": "route-run:2", "event_type": "map_choice", "sequence": 2, "decision_id": None, "options": []})
        preview.pop("decision")
        preview["map_context"] = _map_context()
        result = processor.process(GameStateEvent.model_validate(preview))
        self.assertEqual(result["status"], "map_captured")
        self.assertEqual(result["advice_disposition"]["action"], "preserve")
        self.assertEqual(self.checkpoint.load()["current_decision"]["decision_id"], "card-decision")

    def test_route_event_atomically_updates_map_and_current_decision(self):
        original_write = checkpoint_module._atomic_write_json
        writes = []

        def record_write(path, payload):
            writes.append(copy.deepcopy(payload))
            return original_write(path, payload)

        with patch.object(
            checkpoint_module,
            "_atomic_write_json",
            side_effect=record_write,
        ):
            result = self._processor().process(
                GameStateEvent.model_validate(_route_event())
            )
        saved = self.checkpoint.load()
        self.assertEqual(result["decision_id"], "route-decision")
        self.assertEqual(saved["map_context"]["origin_node_id"], "0:3")
        self.assertEqual(saved["current_decision"]["decision_id"], "route-decision")
        self.assertEqual(saved["last_sequence"], 3)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]["map_context"]["origin_node_id"], "0:3")
        self.assertEqual(writes[0]["current_decision"]["decision_id"], "route-decision")

    def test_v8_route_mode_update_is_atomic_and_restart_recoverable(self):
        payload = json.loads(
            (ROOT / "protocol" / "route-event.example.json").read_text(
                encoding="utf-8"
            )
        )
        payload["run_id"] = "route-mode-run"
        payload["event_id"] = "route-mode-run:4"
        payload["decision_id"] = "route-mode-decision"
        payload["state"]["character"] = "TEST_HERO"
        payload["state"]["relics"] = []
        payload["state"]["relic_states"] = []
        first = self._processor().process(
            GameStateEvent.model_validate(payload)
        )
        self.assertEqual(first["decision_phase"], "opened")

        updated = copy.deepcopy(payload)
        updated["event_id"] = "route-mode-run:5"
        updated["sequence"] = 5
        updated["state_revision"] = 5
        updated["guide_preferences"]["route_mode"] = "survival"
        second = self._processor().process(
            GameStateEvent.model_validate(updated)
        )
        self.assertEqual(second["decision_phase"], "updated")
        checkpoint = self.checkpoint.load()
        self.assertEqual(
            checkpoint["guide_preferences"],
            {"route_mode": "survival"},
        )
        self.assertEqual(
            checkpoint["current_decision"]["payload"][
                "guide_preferences"
            ],
            {"route_mode": "survival"},
        )
        self.assertEqual(
            checkpoint["current_decision"]["result"][
                "guide_preferences"
            ],
            {"route_mode": "survival"},
        )

        restarted = self._processor()
        restored = restarted.lifecycle.active("route-mode-run")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.sequence, 5)
        self.assertEqual(restored.decision_id, "route-mode-decision")

        damaged = self.checkpoint.load()
        damaged["guide_preferences"] = {"route_mode": "invalid"}
        checkpoint_module._atomic_write_json(
            self.checkpoint.path,
            damaged,
        )
        fail_closed = self._processor()
        self.assertIsNone(
            fail_closed.lifecycle.active("route-mode-run")
        )

    def test_graph_rejects_duplicate_unknown_edge_cycle_and_nonchild_candidate(self):
        base = _map_context()
        cases = []
        duplicate = copy.deepcopy(base)
        duplicate["nodes"].append(copy.deepcopy(duplicate["nodes"][1]))
        duplicate["node_count"] += 1
        cases.append(duplicate)
        unknown = copy.deepcopy(base)
        unknown["nodes"][1]["edges"] = ["missing"]
        cases.append(unknown)
        cycle = copy.deepcopy(base)
        cycle["nodes"][1]["edges"] = ["0:3"]
        cases.append(cycle)
        nonchild = copy.deepcopy(base)
        nonchild["available_next_node_ids"] = ["2:2"]
        cases.append(nonchild)
        for context in cases:
            result = RoutePolicy(self.repository).recommend(_route_request(context))
            self.assertIsNone(result.recommended_candidate_id)
            self.assertTrue(all(row.score is None and row.rank is None for row in result.candidates))

    def test_any_unreachable_candidate_fails_the_whole_recommendation(self):
        context = _map_context()
        context["nodes"][2]["edges"] = []
        result = RoutePolicy(self.repository).recommend(_route_request(context))
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIn("boss_unreachable_from_candidate:1:4", result.data_gaps)
        self.assertTrue(all(row.score is None for row in result.candidates))

    def test_paths_follow_real_edges_and_public_output_has_one_route(self):
        nodes = [
            {"node_id": "0:3", "kind": "MONSTER", "row": 0, "col": 3, "edges": ["1:1", "1:3", "1:5"]},
            {"node_id": "1:1", "kind": "ELITE", "row": 1, "col": 1, "edges": ["2:3"]},
            {"node_id": "1:3", "kind": "CAMPFIRE", "row": 1, "col": 3, "edges": ["2:3"]},
            {"node_id": "1:5", "kind": "SHOP", "row": 1, "col": 5, "edges": ["2:3"]},
            {"node_id": "2:3", "kind": "BOSS", "row": 2, "col": 3, "edges": []},
        ]
        context = _map_context(candidates=("1:1", "1:3", "1:5"), nodes=nodes, boss_ids=("2:3",))
        result = RoutePolicy(self.repository).recommend(_route_request(context))
        ranked = sorted(result.candidates, key=lambda row: row.rank or 99)
        presentation = result.presentation
        self.assertEqual(presentation["primary_path_node_ids"][0], ranked[0].candidate_id)
        self.assertEqual(tuple(presentation["backup_path_node_ids"]), ())
        by_id = {node["node_id"]: node for node in nodes}
        paths = {
            row["candidate_id"]: row["node_ids"]
            for row in presentation["paths"]
        }
        self.assertEqual(set(paths), {row.candidate_id for row in ranked})
        for path in paths.values():
            self.assertEqual(path[-1], "2:3")
            self.assertTrue(all(second in by_id[first]["edges"] for first, second in zip(path, path[1:])))

    def test_one_candidate_has_no_forged_backup(self):
        nodes = [
            {"node_id": "0:3", "kind": "MONSTER", "row": 0, "col": 3, "edges": ["1:2"]},
            {"node_id": "1:2", "kind": "MONSTER", "row": 1, "col": 2, "edges": ["2:3"]},
            {"node_id": "2:3", "kind": "BOSS", "row": 2, "col": 3, "edges": []},
        ]
        context = _map_context(candidates=("1:2",), nodes=nodes, boss_ids=("2:3",))
        result = RoutePolicy(self.repository).recommend(_route_request(context))
        self.assertEqual(tuple(result.presentation["backup_path_node_ids"]), ())

    def test_missing_or_malformed_risk_profile_fails_closed(self):
        request = _route_request()
        empty_repository = RelationalRepository(str(Path(self.tempdir.name) / "empty.db"))
        empty_repository.ensure_schema()
        policies = [
            RoutePolicy(empty_repository),
            RoutePolicy(self.repository, risk_profile={}),
            RoutePolicy(self.repository, risk_profile=_risk_profile(version="wrong")),
            RoutePolicy(self.repository, risk_profile=_risk_profile(capability_weights={})),
            RoutePolicy(self.repository, risk_profile=_risk_profile(hp_low_threshold=0.1, hp_critical_threshold=0.2)),
            RoutePolicy(self.repository, risk_profile=_risk_profile(encounter_expected_risk={"MONSTER": -1, "ELITE": 1, "BOSS": 1})),
            RoutePolicy(self.repository, risk_profile=_risk_profile(capability_weights={"immediate_power": -1, "survival": 1, "aoe": 1, "growth": 1, "resource": 1})),
        ]
        for policy in policies:
            result = policy.recommend(request)
            self.assertIsNone(result.recommended_candidate_id)
            self.assertTrue(all(row.score is None for row in result.candidates))
            self.assertTrue(result.data_gaps)

    def test_multidimensional_state_is_deterministic_and_changes_scoring(self):
        naked = _state(relics=[], relic_states=[], potions=[])
        baseline = RoutePolicy(self.repository).recommend(_route_request(state=naked))
        repeated = RoutePolicy(self.repository).recommend(_route_request(state=naked))
        self.assertEqual(baseline.as_dict(), repeated.as_dict())
        baseline_scores = {row.candidate_id: row.score for row in baseline.candidates}

        high_aoe = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "AOE", "count": 12}], relics=[], relic_states=[], potions=[])
        ))
        self.assertGreater(
            {row.candidate_id: row.score for row in high_aoe.candidates}["1:2"],
            baseline_scores["1:2"],
        )
        high_growth = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "GROWTH", "count": 12}], relics=[], relic_states=[], potions=[])
        ))
        self.assertGreater(
            {row.candidate_id: row.dimensions["long_term_growth"] for row in high_growth.candidates}["1:2"],
            0.0,
        )
        self.assertTrue(any(
            factor.get("code") == "elite_growth_relief"
            and float(factor.get("delta") or 0.0) > 0
            for row in high_growth.candidates
            if row.candidate_id == "1:2"
            for factor in row.factors
        ))
        high_resource = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "RESOURCE", "count": 12}], relics=[], relic_states=[], potions=[])
        ))
        one_resource = RoutePolicy(self.repository)._capabilities(_route_request(
            state=_state(deck=[{"card": "RESOURCE", "count": 1}], relics=[], relic_states=[], potions=[])
        ))
        self.assertEqual(one_resource.resources, 3.0)
        self.assertGreater(
            {row.candidate_id: row.dimensions["resource_efficiency"] for row in high_resource.candidates}["1:2"],
            0.0,
        )
        self.assertGreater(
            {row.candidate_id: row.score for row in high_resource.candidates}["1:2"],
            baseline_scores["1:2"],
        )

    def test_route_modes_are_deterministic_explainable_and_safety_bounded(self):
        results = {}
        for mode in ("balanced", "survival", "growth"):
            request = _route_request(route_mode=mode)
            first = RoutePolicy(self.repository).recommend(request)
            repeated = RoutePolicy(self.repository).recommend(request)
            self.assertEqual(first.as_dict(), repeated.as_dict())
            results[mode] = first
            factor_codes = {
                str(factor.get("code") or "")
                for candidate in first.candidates
                for factor in candidate.factors
            }
            self.assertTrue(
                any(code.startswith(f"route_mode_{mode}") for code in factor_codes)
            )

        def score(mode, candidate_id):
            return next(
                candidate.score
                for candidate in results[mode].candidates
                if candidate.candidate_id == candidate_id
            )

        self.assertGreater(
            score("survival", "1:4"),
            score("balanced", "1:4"),
        )
        growth_elite_factors = [
            factor
            for candidate in results["growth"].candidates
            if candidate.candidate_id == "1:2"
            for factor in candidate.factors
            if str(factor.get("code") or "").startswith(
                "route_mode_growth_elite"
            )
        ]
        self.assertTrue(growth_elite_factors)
        self.assertGreater(growth_elite_factors[0]["delta"], 0)

        critical = RoutePolicy(self.repository).recommend(
            _route_request(
                state=_state(hp=10),
                route_mode="growth",
            )
        )
        self.assertEqual(critical.recommended_candidate_id, "1:4")
        critical_codes = {
            str(factor.get("code") or "")
            for candidate in critical.candidates
            for factor in candidate.factors
        }
        self.assertIn(
            "route_mode_growth_safety_floor_elite",
            critical_codes,
        )
        relic = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "FRONT", "count": 1}], potions=[])
        ))
        self.assertGreater(
            {row.candidate_id: row.dimensions["synergy"] for row in relic.candidates}["1:2"],
            0.0,
        )

    def test_explicit_upgrade_multihit_and_one_potion_buffer(self):
        unupgraded = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "MULTI", "count": 1, "upgrades": 0}], relics=[], relic_states=[], potions=[])
        ))
        upgraded = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "MULTI", "count": 1, "upgrades": 1}], relics=[], relic_states=[], potions=[])
        ))
        base_power = {row.candidate_id: row.dimensions["immediate_power"] for row in unupgraded.candidates}
        upgraded_power = {row.candidate_id: row.dimensions["immediate_power"] for row in upgraded.candidates}
        self.assertGreater(upgraded_power["1:2"], base_power["1:2"])

        potion = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "FRONT", "count": 1}], relics=[], relic_states=[])
        ))
        naked = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(deck=[{"card": "FRONT", "count": 1}], relics=[], relic_states=[], potions=[])
        ))
        self.assertGreater(
            {row.candidate_id: row.score for row in potion.candidates}["1:4"],
            {row.candidate_id: row.score for row in naked.candidates}["1:4"],
        )
        for row in potion.candidates:
            self.assertEqual(
                sum(factor["code"] == "potion_buffer_once" for factor in row.factors),
                1,
            )

    def test_low_hp_known_boss_and_encounter_pool_change_route_factors(self):
        baseline = RoutePolicy(self.repository).recommend(_route_request())
        baseline_scores = {row.candidate_id: row.score for row in baseline.candidates}
        low_hp = RoutePolicy(self.repository).recommend(_route_request(state=_state(hp=10)))
        self.assertGreater(
            {row.candidate_id: row.score for row in low_hp.candidates}["1:4"],
            baseline_scores["1:4"],
        )
        self.assertIn(
            "known_boss_risk",
            {factor["code"] for row in baseline.candidates for factor in row.factors},
        )
        self.assertNotIn("encounter_pool_missing:MONSTER", baseline.data_gaps)

        boss_choice_nodes = [
            {"node_id": "0:3", "kind": "MONSTER", "row": 0, "col": 3, "edges": ["1:2", "1:4"]},
            {"node_id": "1:2", "kind": "CAMPFIRE", "row": 1, "col": 2, "edges": ["2:3"]},
            {"node_id": "1:4", "kind": "SHOP", "row": 1, "col": 4, "edges": ["2:3"]},
            {"node_id": "2:3", "kind": "BOSS", "row": 2, "col": 3, "edges": []},
        ]
        weak_context = _map_context(
            candidates=("1:2", "1:4"),
            nodes=boss_choice_nodes,
            boss_ids=("2:3",),
        )
        weak_boss = RoutePolicy(self.repository).recommend(_route_request(weak_context))
        high_context = copy.deepcopy(weak_context)
        high_context["boss_encounter_ids"] = ["HIGH_BOSS_ENCOUNTER"]
        high_boss = RoutePolicy(self.repository).recommend(_route_request(high_context))
        weak_scores = {row.candidate_id: row.score for row in weak_boss.candidates}
        base_gap = weak_scores["1:2"] - weak_scores["1:4"]
        high_scores = {row.candidate_id: row.score for row in high_boss.candidates}
        high_gap = high_scores["1:2"] - high_scores["1:4"]
        self.assertGreater(high_gap, base_gap)
        self.assertIn(
            "boss_preparation_opportunity",
            {factor["code"] for row in high_boss.candidates if row.candidate_id == "1:2" for factor in row.factors},
        )

    def test_missing_hp_fails_closed_while_missing_gold_degrades_honestly(self):
        missing_hp = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(hp=None, max_hp=None)
        ))
        self.assertIsNone(missing_hp.recommended_candidate_id)
        self.assertTrue({"hp_missing", "max_hp_missing"}.issubset(missing_hp.data_gaps))
        self.assertTrue(all(row.score is None for row in missing_hp.candidates))

        missing_gold = RoutePolicy(self.repository).recommend(_route_request(
            state=_state(gold=None)
        ))
        self.assertIsNotNone(missing_gold.recommended_candidate_id)
        self.assertIn("gold_missing", missing_gold.data_gaps)
        shop = next(row for row in missing_gold.candidates if row.candidate_id == "1:4")
        self.assertEqual(shop.dimensions["data_completeness"], 0.0)
        self.assertIn("shop_gold_unknown", {factor["code"] for factor in shop.factors})

    def test_ascension_pool_thresholds_match_tough_and_deadly_enemies(self):
        a0 = self.repository.encounter_pool_expectations(["TEST_BOSS_ENCOUNTER"], ascension=0)
        a7 = self.repository.encounter_pool_expectations(["TEST_BOSS_ENCOUNTER"], ascension=7)
        a8 = self.repository.encounter_pool_expectations(["TEST_BOSS_ENCOUNTER"], ascension=8)
        a9 = self.repository.encounter_pool_expectations(["TEST_BOSS_ENCOUNTER"], ascension=9)
        self.assertEqual(a0["MONSTER"]["average_hp"], a7["MONSTER"]["average_hp"])
        self.assertEqual(a0["MONSTER"]["average_attack"], a8["MONSTER"]["average_attack"])
        self.assertGreater(a8["MONSTER"]["average_hp"], a0["MONSTER"]["average_hp"])
        self.assertGreater(a9["MONSTER"]["average_attack"], a8["MONSTER"]["average_attack"])

    def test_unknown_card_relic_potion_are_explicit_degradable_gaps(self):
        state = _state(
            deck=[{"card": "MISSING_CARD", "count": 1}],
            relics=["MISSING_RELIC"],
            potions=[{"potion": "MISSING_POTION", "slot": 0}],
        )
        result = RoutePolicy(self.repository).recommend(_route_request(state=state))
        self.assertIsNotNone(result.recommended_candidate_id)
        self.assertEqual(result.confidence, "low")
        self.assertTrue({"unknown_card:MISSING_CARD", "unknown_relic:MISSING_RELIC", "unknown_potion:MISSING_POTION"}.issubset(result.data_gaps))

    def test_bounded_search_and_realistic_p95(self):
        nodes = [{"node_id": "0:3", "kind": "MONSTER", "row": 0, "col": 3, "edges": ["1:0", "1:1"]}]
        # 64-node bounded synthetic stress DAG with map-like width (2–4
        # nodes per row) and depth. It is intentionally not presented as a
        # recorded real map topology; real graph facts remain covered by the
        # protocol/replay fixtures and later true-machine validation.
        kinds = ("MONSTER", "ELITE", "CAMPFIRE", "SHOP")
        for row in range(1, 16):
            for col in range(4):
                edges = (
                    ["16:0", "16:1"]
                    if row == 15
                    else [f"{row + 1}:{col}", f"{row + 1}:{(col + 1) % 4}"]
                )
                nodes.append({"node_id": f"{row}:{col}", "kind": kinds[col], "row": row, "col": col, "edges": edges})
        nodes.extend([
            {"node_id": "16:0", "kind": "MONSTER", "row": 16, "col": 0, "edges": ["17:3"]},
            {"node_id": "16:1", "kind": "ELITE", "row": 16, "col": 1, "edges": ["17:3"]},
            {"node_id": "17:3", "kind": "BOSS", "row": 17, "col": 3, "edges": []},
        ])
        self.assertEqual(len(nodes), 64)
        context = _map_context(candidates=("1:0", "1:1"), nodes=nodes, boss_ids=("17:3",))
        request = _route_request(context)
        policy = RoutePolicy(self.repository, max_search_expansions=128)
        policy.recommend(request)  # warm caches and sqlite pages
        elapsed = []
        for _ in range(100):
            started = time.perf_counter()
            result = policy.recommend(request)
            elapsed.append((time.perf_counter() - started) * 1000)
            self.assertIsNotNone(result.recommended_candidate_id)
        p95 = sorted(elapsed)[math.ceil(0.95 * len(elapsed)) - 1]
        self.assertLessEqual(p95, 300.0, f"P95 route policy latency {p95:.2f} ms")

        limited = RoutePolicy(self.repository, max_search_expansions=1).recommend(request)
        self.assertIsNone(limited.recommended_candidate_id)
        self.assertIn("route_search_limit_exceeded", limited.data_gaps)

    def test_route_event_to_advice_atomic_write_p95(self):
        exchange = Path(self.tempdir.name) / "route-bridge"
        events = exchange / "events"
        events.mkdir(parents=True)
        output = exchange / "advice-event.json"
        processor = RealtimeEventProcessor(
            self.repository,
            checkpoint=ActiveRunCheckpointStore(exchange / "active-run.json"),
        )
        bridge = GameStateFileBridge(
            processor,
            input_path=exchange / "state-event.json",
            output_path=output,
            events_dir=events,
        )
        # Prewrite the same decision as 100 ordered route observations. The
        # timed section begins at discovery, so it includes JSON parse,
        # Pydantic, policy, one checkpoint replace, advice replace and queue
        # acknowledgement but not producer-side file creation.
        for sequence in range(1, 101):
            payload = _route_event(
                sequence=sequence,
                run_id="bridge-run",
                decision_id="bridge-route",
            )
            payload["event_id"] = f"bridge-run:{sequence}"
            payload["emitted_at"] = (
                datetime(2026, 7, 16, tzinfo=timezone.utc)
                + timedelta(seconds=sequence)
            ).isoformat()
            (events / f"{sequence:04}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        elapsed = []
        for sequence in range(1, 101):
            started = time.perf_counter()
            result = bridge.run_once()
            elapsed.append((time.perf_counter() - started) * 1000)
            self.assertEqual(result["status"], "processed")
            visible = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(visible["event_id"], f"bridge-run:{sequence}")
        p95 = sorted(elapsed)[math.ceil(0.95 * len(elapsed)) - 1]
        self.assertLessEqual(p95, 300.0, f"P95 route bridge latency {p95:.2f} ms")

    def test_route_pipeline_uses_no_network_and_no_history_tables(self):
        with patch.object(socket, "create_connection", side_effect=AssertionError("network")):
            self._processor().process(GameStateEvent.model_validate(_route_event()))
        with self.repository.connect() as connection:
            for table in ("run_states", "decision_events", "decision_outcomes", "game_state_events"):
                self.assertEqual(connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"], 0)

    def test_mod_source_contracts_cover_actionability_identity_and_fail_closed_overlay(self):
        mod = ROOT / "mod" / "STS2Guide.ReadOnlyExporter"
        observer = (mod / "RouteChoiceObserver.cs").read_text(encoding="utf-8")
        writer = (mod / "StateEventWriter.cs").read_text(encoding="utf-8")
        reader = (mod / "MapNodeReader.cs").read_text(encoding="utf-8")
        controller = (mod / "RouteAdviceController.cs").read_text(encoding="utf-8")
        overlay = (mod / "RouteMapOverlay.cs").read_text(encoding="utf-8")
        contract_reader = (
            mod / "AdviceContractReader.cs"
        ).read_text(encoding="utf-8")
        all_route = "\n".join((
            observer,
            writer,
            reader,
            controller,
            overlay,
            contract_reader,
        ))
        for needle in (
            "_openedFromTopBar", "MapPointState.Travelable", "TryReadVisualTravelableIds",
            "MapFingerprint", "CreateRouteObservationFingerprint", "Guid.NewGuid().ToString(\"N\")",
            "primary_path_node_ids", "backup_path_node_ids", "world_sequence",
            "ValidateTopology", "duplicate mapping", "public override void _Process(double delta)",
            "MouseFilter = MouseFilterEnum.Ignore", "HarmonyPostfix",
        ):
            self.assertIn(needle, all_route)
        self.assertNotIn("HarmonyPrefix", all_route)
        self.assertNotIn("TravelToMapCoord", all_route)
        self.assertNotIn("EnterMapCoord", all_route)
        self.assertNotIn("startMapPoints\n                    .Where", reader)
        self.assertIn("if (_pendingDecision is not null && !sameOpportunity)", writer)
        self.assertIn("RouteMapOverlay.Hide(_screen, _pending.DecisionId)", controller)
        self.assertNotIn("BackupPathNodeIds", overlay)
        self.assertEqual(overlay.count("DrawPolyline("), 1)
        self.assertIn(
            "nodes[context.OriginNodeId].Edges.Contains(",
            overlay,
        )
        self.assertIn(
            "new[] { _originNodeId }\n"
            "                    .Concat(_presentation.PrimaryPathNodeIds)",
            overlay,
        )
        card_panel = (mod / "CardRewardAdvicePanel.cs").read_text(encoding="utf-8")
        self.assertIn("(version != 1 && version != 2)", card_panel)
        for consumer in (card_panel, controller):
            self.assertIn(
                "AdviceContractReader.HasRecommendationMetadata(",
                consumer,
            )
            self.assertIn(
                "AdviceContractReader.StatusMatchesRecommendation(",
                consumer,
            )
            self.assertIn(
                "AdviceContractReader.HasCandidateMetadata(candidate)",
                consumer,
            )

        # The observer must reject a delayed old owner before it can close or
        # clear the new owner's decision.  This checks the actual callback
        # body and guard order, not just a file-wide string occurrence.
        selected = _csharp_method(
            observer,
            "internal static void OnSelected(NMapScreen screen, NMapPoint point)",
        )
        old_owner_guard = selected.index("if (!ReferenceEquals(_screen, screen))")
        self.assertLess(old_owner_guard, selected.index("return;", old_owner_guard))
        self.assertLess(old_owner_guard, selected.index("StateEventWriter.EmitRouteSelected"))
        self.assertLess(
            selected.index("StateEventWriter.EmitRouteSelected"),
            selected.index("ClearStagedObservation"),
        )
        # A native click can happen while the close event is temporarily
        # unwritable.  The pending writer state remains for recovery, but the
        # already-invalid map UI must fail closed instead of lingering during
        # travel.
        self.assertIn("else\n                {\n                    // Keep the writer", selected)
        self.assertIn("FailClosed(screen);", selected)

        # Both the write commit and the drawer binding precede the committed
        # fingerprint.  A failed write/bind falls out without changing it,
        # leaving the next _Process free to retry.
        observe = _csharp_method(observer, "private static void TryObserveSafe")
        emit = observe.index("StateEventWriter.EmitRouteChoice")
        bind = observe.index("StateEventWriter.BindDecisionScreen")
        committed = observe.index("_lastCommittedObservationFingerprint = actionableFingerprint")
        self.assertLess(emit, bind)
        self.assertLess(bind, committed)
        self.assertIn("if (pending is null", observe)
        failed_commit = observe[observe.index("if (pending is null"):committed]
        self.assertIn("RouteAdviceController.Hide(screen);", failed_commit)
        self.assertIn("return;", failed_commit)
        # Losing the actionable predicate (preview, travel, owner/state or
        # model/visual mismatch) must also remove a previous route UI.  It is
        # not enough merely to clear the debounce fingerprint.
        before_debounce = observe[:observe.index("var now = DateTime.UtcNow.Ticks")]
        self.assertIn("FailClosed(screen);", before_debounce)
        self.assertIn("private static void FailClosed", observer)
        self.assertIn(
            "if (!ReferenceEquals(_screen, screen))\n            {\n                return;",
            observe,
        )
        fail_closed = _csharp_method(observer, "private static void FailClosed")
        self.assertLess(
            fail_closed.index("if (ReferenceEquals(_screen, screen))"),
            fail_closed.index("ClearStagedObservation();"),
        )
        self.assertIn("TryLogRouteFailure", observer)

        # A candidate/decision handoff first destroys the old scoped UI.
        # Reuse is allowed only for the same decision and identical immutable
        # row shape, so ContextDrawer cannot retain wrong captions/row count.
        show = _csharp_method(controller, "internal static void Show")
        self.assertIn("CanReuseDrawer(_pending, pending)", show)
        self.assertLess(show.index("HideInternal();"), show.rindex("_pending = pending;"))
        reuse = _csharp_method(controller, "private static bool CanReuseDrawer")
        for needle in ("current.DecisionId == next.DecisionId", "CandidateId", "DisplayIndex", "Label"):
            self.assertIn(needle, reuse)
        self.assertIn("TryReadNullableString", controller)
        self.assertNotIn("NullableString(JsonElement", controller)

        # Recovery and UPDATE identity cover the complete logical map plus
        # every exported policy input, not only current HP/deck fields.
        checkpoint_recovery = _csharp_method(
            writer,
            "private static void TryAddCheckpointRouteDecision(",
        )
        self.assertIn("payloadSequence", checkpoint_recovery)
        self.assertIn(
            "payloadSequenceValue != currentSequenceValue",
            checkpoint_recovery,
        )
        self.assertIn("State = state", writer)
        self.assertIn("MapContext = context", writer)
        for needle in ("node.Kind", "node.Row", "node.Col", "out var kind", "Kind = kind.GetString()!"):
            self.assertIn(needle, reader)

        # Every public observer entry point is exception-isolated before a
        # Harmony Postfix can reach game code.
        for signature in (
            "internal static void OnOpened",
            "internal static void TryObserve",
            "internal static void OnSelected",
            "internal static void OnClosed",
            "internal static void OnOwnerGone",
        ):
            self.assertIn("=> Safely", _csharp_method(observer, signature))

        # An identity/candidate mismatch invalidates both Drawer rows and
        # overlay.  The controller's rejection branch must not render partial
        # results, and the overlay redraw is genuinely frame-driven.
        poll = _csharp_method(controller, "private static void Poll()")
        mismatch = poll.index("if (matched is null)")
        self.assertIn("Invalidate();", poll[mismatch:poll.index("ContextDrawer.Render", mismatch)])
        self.assertIn("return;", poll[mismatch:poll.index("ContextDrawer.Render", mismatch)])
        invalidate = _csharp_method(controller, "private static void Invalidate()")
        self.assertIn("RouteMapOverlay.Hide(_screen, _pending.DecisionId)", invalidate)
        process = _csharp_method(overlay, "public override void _Process(double delta)")
        self.assertIn("QueueRedraw();", process)
        self.assertNotIn("Godot.Timer", overlay)


if __name__ == "__main__":
    unittest.main()
