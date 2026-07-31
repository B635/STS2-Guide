"""Differential coverage for the state-event Schema/Pydantic boundary."""
from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from realtime.protocol import GameStateEvent


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol"
FIXTURES = ROOT / "tests" / "fixtures" / "protocol"


class ProtocolStrictParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(
            (PROTOCOL / "state-event.schema.json").read_text(encoding="utf-8")
        )
        cls.validator = Draft202012Validator(schema)
        cls.card_event = json.loads(
            (PROTOCOL / "state-event.example.json").read_text(encoding="utf-8")
        )
        cls.route_event = json.loads(
            (PROTOCOL / "route-event.example.json").read_text(encoding="utf-8")
        )

    def assert_schema_and_runtime_reject(self, payload: dict) -> None:
        self.assertFalse(
            self.validator.is_valid(payload),
            "JSON Schema unexpectedly accepted the differential case",
        )
        with self.assertRaises(ValidationError):
            GameStateEvent.model_validate(payload)

    def test_valid_v9_production_and_v6_v7_replay_examples_have_parity(self):
        v7 = json.loads(
            (FIXTURES / "state-v7.json").read_text(encoding="utf-8")
        )
        v6_route = json.loads(
            (FIXTURES / "state-v6-route.json").read_text(
                encoding="utf-8"
            )
        )
        for payload, expected_version in (
            (self.card_event, 9),
            (self.route_event, 9),
            (v7, 7),
            (v6_route, 6),
        ):
            with self.subTest(event_type=payload["event_type"]):
                self.validator.validate(payload)
                event = GameStateEvent.model_validate(payload)
                self.assertEqual(event.schema_version, expected_version)

    def test_live_v6_rejects_coercive_scalar_types(self):
        cases: list[tuple[str, dict]] = []

        payload = copy.deepcopy(self.card_event)
        payload["schema_version"] = "8"
        cases.append(("schema_version_string", payload))

        payload = copy.deepcopy(self.card_event)
        payload["sequence"] = "1"
        cases.append(("sequence_string", payload))

        payload = copy.deepcopy(self.card_event)
        payload["decision"]["can_skip"] = 0
        cases.append(("zero_boolean", payload))

        payload = copy.deepcopy(self.card_event)
        payload["decision"]["can_reroll"] = 1
        cases.append(("one_boolean", payload))

        payload = copy.deepcopy(self.card_event)
        payload["state"]["hp"] = "70"
        cases.append(("hp_string", payload))

        payload = copy.deepcopy(self.route_event)
        payload["map_context"]["node_count"] = "64"
        cases.append(("node_count_string", payload))

        payload = copy.deepcopy(self.card_event)
        payload["state"]["hp"] = math.nan
        cases.append(("nan_integer", payload))

        payload = copy.deepcopy(self.card_event)
        payload["state"]["hp"] = math.inf
        cases.append(("infinite_integer", payload))

        for name, invalid in cases:
            with self.subTest(name=name):
                self.assert_schema_and_runtime_reject(invalid)

    def test_live_v6_rejects_blank_or_duplicate_critical_ids(self):
        cases: list[tuple[str, dict]] = []

        for field in ("event_id", "run_id", "decision_id"):
            payload = copy.deepcopy(self.card_event)
            payload[field] = "   "
            cases.append((f"blank_{field}", payload))

        payload = copy.deepcopy(self.card_event)
        payload["candidates"][0]["candidate_id"] = " "
        cases.append(("blank_candidate_id", payload))

        payload = copy.deepcopy(self.route_event)
        payload["map_context"]["origin_node_id"] = " "
        cases.append(("blank_origin_id", payload))

        payload = copy.deepcopy(self.route_event)
        next_id = payload["map_context"]["available_next_node_ids"][0]
        payload["map_context"]["available_next_node_ids"].append(next_id)
        cases.append(("duplicate_next_id", payload))

        payload = copy.deepcopy(self.route_event)
        boss_id = payload["map_context"]["boss_encounter_ids"][0]
        payload["map_context"]["boss_encounter_ids"].append(boss_id)
        cases.append(("duplicate_boss_encounter_id", payload))

        payload = copy.deepcopy(self.route_event)
        edge_id = payload["map_context"]["nodes"][0]["edges"][0]
        payload["map_context"]["nodes"][0]["edges"].append(edge_id)
        cases.append(("duplicate_edge_id", payload))

        payload = copy.deepcopy(self.card_event)
        payload["candidates"].append(copy.deepcopy(payload["candidates"][0]))
        cases.append(("duplicate_candidate", payload))

        payload = copy.deepcopy(self.card_event)
        payload.update({
            "event_type": "deck_edit",
            "decision_id": "strict:deck-edit:1",
            "options": [],
            "decision": {
                "can_skip": False,
                "can_reroll": False,
                "reward_source": "REST_SITE",
            },
            "decision_parent": {
                "decision_id": "strict:rest:1",
                "candidate_id": "rest:smith",
                "source_type": "rest_site",
                "source_id": "SOURCE_TEST",
            },
            "candidates": [{
                "candidate_id": "upgrade:STRIKE_IRONCLAD",
                "kind": "deck_edit",
                "entity_id": "STRIKE_IRONCLAD",
                "label": "打击",
                "eligible": True,
                "unavailable_reason": None,
                "costs": [],
                "payload": {
                    "operation": "upgrade",
                    "target_candidate_ids": [
                        "upgrade:STRIKE_IRONCLAD",
                        "upgrade:STRIKE_IRONCLAD",
                    ],
                },
            }],
        })
        cases.append(("duplicate_target_candidate_id", payload))

        for name, invalid_id in (
            ("blank_target_candidate_id", "   "),
            ("oversized_target_candidate_id", "X" * 241),
        ):
            invalid_target = copy.deepcopy(payload)
            invalid_target["candidates"][0]["payload"][
                "target_candidate_ids"
            ] = [invalid_id]
            cases.append((name, invalid_target))

        payload = copy.deepcopy(self.route_event)
        payload["map_context"]["nodes"].append(
            copy.deepcopy(payload["map_context"]["nodes"][0])
        )
        payload["map_context"]["node_count"] += 1
        cases.append(("duplicate_node", payload))

        for name, invalid in cases:
            with self.subTest(name=name):
                self.assert_schema_and_runtime_reject(invalid)

    def test_live_v6_requires_explicit_complete_state_and_decision(self):
        state_fields = {
            "character",
            "ascension",
            "act",
            "floor",
            "hp",
            "max_hp",
            "gold",
            "energy",
            "deck",
            "relics",
            "relic_states",
            "potions",
            "max_potion_slots",
            "modifiers",
            "capture_warnings",
        }
        for field in state_fields:
            with self.subTest(missing_state_field=field):
                payload = copy.deepcopy(self.card_event)
                payload["state"].pop(field)
                self.assert_schema_and_runtime_reject(payload)

    def test_v7_v8_require_complete_handshake_and_committed_revision(self):
        envelope_fields = (
            "source",
            "game_version",
            "snapshot_kind",
            "state_revision",
            "producer_id",
            "producer_version",
            "game_assembly_sha256",
            "release_fingerprint",
            "decision_id",
            "options",
            "decision",
            "parent_event_id",
            "outcome",
            "map_context",
            "run_result",
        )
        for field in envelope_fields:
            with self.subTest(missing_envelope_field=field):
                payload = copy.deepcopy(self.card_event)
                payload.pop(field)
                self.assert_schema_and_runtime_reject(payload)
        v7 = json.loads(
            (FIXTURES / "state-v7.json").read_text(encoding="utf-8")
        )
        for field in envelope_fields:
            with self.subTest(
                schema_version=7,
                missing_envelope_field=field,
            ):
                payload = copy.deepcopy(v7)
                payload.pop(field)
                self.assert_schema_and_runtime_reject(payload)

        for field in (
            "count",
            "upgrades",
            "enchantment",
            "enchantment_amount",
            "affliction",
            "affliction_amount",
        ):
            with self.subTest(missing_nested_card_field=field):
                payload = copy.deepcopy(self.card_event)
                payload["state"]["deck"][0].pop(field)
                self.assert_schema_and_runtime_reject(payload)

        payload = copy.deepcopy(self.card_event)
        payload["state_revision"] = payload["sequence"] + 1
        # Equality is an executable runtime invariant. JSON Schema 2020-12
        # has no portable sibling-field equality keyword.
        self.validator.validate(payload)
        with self.assertRaises(ValidationError):
            GameStateEvent.model_validate(payload)

        for field, value in (
            ("snapshot_kind", "delta"),
            ("game_assembly_sha256", "0" * 63),
            ("release_fingerprint", "0" * 63),
            ("producer_id", ""),
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.card_event)
                payload[field] = value
                self.assert_schema_and_runtime_reject(payload)

        for field in ("can_skip", "can_reroll", "reward_source"):
            with self.subTest(missing_decision_field=field):
                payload = copy.deepcopy(self.card_event)
                payload["decision"].pop(field)
                self.assert_schema_and_runtime_reject(payload)

    def test_v8_requires_strict_guide_preferences(self):
        for mutation in ("missing", "null", "invalid", "extra", "wrong_type"):
            with self.subTest(mutation=mutation):
                payload = copy.deepcopy(self.card_event)
                if mutation == "missing":
                    payload.pop("guide_preferences")
                elif mutation == "null":
                    payload["guide_preferences"] = None
                elif mutation == "invalid":
                    payload["guide_preferences"]["route_mode"] = "fast"
                elif mutation == "extra":
                    payload["guide_preferences"]["other"] = True
                else:
                    payload["guide_preferences"]["route_mode"] = 1
                self.assert_schema_and_runtime_reject(payload)

    def test_v9_generic_candidates_reject_cross_kind_and_effect_drift(self):
        cases: list[tuple[str, dict]] = []

        payload = copy.deepcopy(self.card_event)
        payload["candidates"][0]["payload"]["node_id"] = "1:2"
        cases.append(("card_with_route_payload", payload))

        payload = copy.deepcopy(self.card_event)
        payload["candidates"][-1]["costs"] = [
            {"kind": "gold", "amount": 1, "resource_id": None}
        ]
        cases.append(("skip_with_cost", payload))

        payload = self._event_choice_payload()
        payload["candidates"][0]["payload"]["effects"] = []
        cases.append(("event_without_explicit_effect", payload))

        payload = self._event_choice_payload()
        effect = payload["candidates"][0]["payload"]["effects"][0]
        effect.update({
            "kind": "add_card",
            "entity_type": "relics",
            "entity_id": "TEST_CARD",
            "target_mode": "specific",
            "certainty": "exact",
        })
        cases.append(("wrong_effect_entity_type", payload))

        payload = self._event_choice_payload()
        effect = payload["candidates"][0]["payload"]["effects"][0]
        effect.update({
            "kind": "hp_delta",
            "amount": 2.5,
            "target_mode": "none",
            "certainty": "exact",
        })
        cases.append(("fractional_hp_effect", payload))

        payload = self._event_choice_payload()
        payload["candidates"][0]["payload"]["effects"][0].pop(
            "child_decision_type"
        )
        cases.append(("effect_without_child_decision_type_field", payload))

        payload = self._event_choice_payload()
        payload["candidates"][0]["payload"]["effects"][0][
            "child_decision_type"
        ] = "card_reward"
        cases.append(("non_followup_with_child_decision_type", payload))

        payload = self._event_choice_payload()
        effect = payload["candidates"][0]["payload"]["effects"][0]
        effect.update({
            "kind": "followup_choice",
            "certainty": "unknown",
            "child_decision_type": "card_reward",
        })
        cases.append(("unknown_followup_with_child_decision_type", payload))

        payload = self._event_choice_payload()
        payload["candidates"][0]["eligible"] = False
        payload["candidates"][0]["unavailable_reason"] = "locked"
        cases.append(("decision_without_eligible_candidate", payload))

        payload = self._merchant_payload()
        payload["candidates"][0]["payload"]["is_stocked"] = False
        payload["candidates"][0]["eligible"] = True
        cases.append(("stocked_eligibility_conflict", payload))

        payload = self._merchant_payload()
        payload["candidates"][0]["entity_id"] = None
        cases.append(("entity_offer_without_entity_id", payload))

        for name, invalid in cases:
            with self.subTest(name=name):
                self.assert_schema_and_runtime_reject(invalid)

    def test_v9_exact_followup_declares_its_child_decision_type(self):
        payload = self._event_choice_payload()
        effect = payload["candidates"][0]["payload"]["effects"][0]
        effect.update({
            "kind": "followup_choice",
            "certainty": "exact",
            "child_decision_type": "card_reward",
        })
        self.validator.validate(payload)
        parsed = GameStateEvent.model_validate(payload)
        self.assertEqual(
            parsed.candidates[0].payload.effects[0].child_decision_type,
            "card_reward",
        )

    def test_v9_parentless_card_reward_requires_card_source(self):
        valid = copy.deepcopy(self.card_event)
        valid["decision_parent"] = None
        valid["decision"]["reward_source"] = "CARD"
        self.validator.validate(valid)
        parsed = GameStateEvent.model_validate(valid)
        self.assertEqual(parsed.decision.reward_source, "CARD")

        for invalid_source in ("COMBAT", "NEOW", None):
            with self.subTest(reward_source=invalid_source):
                invalid = copy.deepcopy(valid)
                invalid["decision"]["reward_source"] = invalid_source
                self.assert_schema_and_runtime_reject(invalid)

    def test_v9_runtime_enforces_identity_equalities_schema_cannot_express(self):
        payload = copy.deepcopy(self.card_event)
        payload["candidates"][0]["entity_id"] = "ANOTHER_CARD"
        self.validator.validate(payload)
        with self.assertRaises(ValidationError):
            GameStateEvent.model_validate(payload)

        payload = copy.deepcopy(self.route_event)
        payload["candidates"][0]["entity_id"] = "99:99"
        self.validator.validate(payload)
        with self.assertRaises(ValidationError):
            GameStateEvent.model_validate(payload)

    def _event_choice_payload(self) -> dict:
        payload = copy.deepcopy(self.card_event)
        payload.update({
            "event_type": "event_choice",
            "decision_id": "test-run:event:one",
            "decision_parent": None,
            "map_context": None,
        })
        payload["decision"] = {
            "can_skip": False,
            "can_reroll": False,
            "reward_source": "EVENT",
        }
        payload["candidates"] = [{
            "candidate_id": "TEST_EVENT:INITIAL:WAIT",
            "kind": "event_option",
            "entity_id": None,
            "label": "等待",
            "eligible": True,
            "unavailable_reason": None,
            "costs": [],
            "payload": {
                "event_id": "TEST_EVENT",
                "page_id": "INITIAL",
                "option_id": "WAIT",
                "effects": [{
                    "kind": "no_op",
                    "amount": None,
                    "min_amount": None,
                    "max_amount": None,
                    "entity_type": None,
                    "entity_id": None,
                    "target_mode": "none",
                    "certainty": "exact",
                    "source_code": "catalog:test",
                    "child_decision_type": None,
                }],
            },
        }]
        return payload

    def _merchant_payload(self) -> dict:
        payload = copy.deepcopy(self.card_event)
        payload.update({
            "event_type": "merchant",
            "decision_id": "test-run:merchant:one",
            "decision_parent": None,
            "map_context": None,
        })
        payload["decision"] = {
            "can_skip": False,
            "can_reroll": False,
            "reward_source": "MERCHANT",
        }
        payload["candidates"] = [{
            "candidate_id": "merchant:card:0:TEST_CARD",
            "kind": "merchant_offer",
            "entity_id": "TEST_CARD",
            "label": "测试牌",
            "eligible": True,
            "unavailable_reason": None,
            "costs": [
                {"kind": "gold", "amount": 50, "resource_id": None}
            ],
            "payload": {
                "slot_id": "merchant:card:0",
                "offer_kind": "card",
                "is_stocked": True,
                "effects": [],
            },
        }]
        return payload

    def test_legacy_v1_to_v7_replays_remain_compatible(self):
        fixtures = [
            (
                version,
                json.loads(
                    (FIXTURES / f"state-v{version}.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            for version in range(1, 6)
        ]
        fixtures.extend([
            (
                6,
                json.loads(
                    (FIXTURES / "state-v6-route.json").read_text(
                        encoding="utf-8"
                    )
                ),
            ),
            (
                7,
                json.loads(
                    (FIXTURES / "state-v7.json").read_text(
                        encoding="utf-8"
                    )
                ),
            ),
        ])
        for version, payload in fixtures:
            with self.subTest(schema_version=version):
                self.validator.validate(payload)
                event = GameStateEvent.model_validate(payload)
                self.assertEqual(event.schema_version, version)


if __name__ == "__main__":
    unittest.main()
