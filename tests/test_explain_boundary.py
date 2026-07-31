from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import unittest
from unittest import mock
from uuid import uuid4

from pydantic import ValidationError

from advisor.decision_core import CandidateAssessment, Recommendation
from explain.contracts import (
    ExplainIdentity,
    ExplainRequest,
    canonical_recommendation_digest,
)
from explain.deterministic import build_deterministic_response
from explain.worker import ExplainWorker


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 7, 31, 8, 0, tzinfo=timezone.utc)
FINGERPRINT = "a" * 64


def _identity(**overrides) -> ExplainIdentity:
    values = {
        "run_id": "run-one",
        "event_id": "run-one:8",
        "decision_id": "run-one:card-reward:3",
        "state_revision": 8,
        "world_sequence": 8,
        "recommendation_contract_version": 1,
        "policy_version": "card_reward:test-v1",
    }
    values.update(overrides)
    return ExplainIdentity(**values)


def _recommendation() -> dict:
    return {
        "contract_version": 1,
        "decision_id": "run-one:card-reward:3",
        "decision_type": "card_reward",
        "world_sequence": 8,
        "policy_version": "card_reward:test-v1",
        "status": "recommend",
        "confidence": "medium",
        "recommended_candidate_id": "0:ANGER",
        "candidates": [
            {
                "candidate_id": "0:ANGER",
                "label": "愤怒",
                "display_index": 0,
                "eligible": True,
                "score": 64.0,
                "rank": 1,
                "factors": [
                    {
                        "code": "attack_coverage",
                        "delta": 5.0,
                        "message": "当前牌组缺少直接输出。",
                    },
                    {
                        "code": "route_fit",
                        "delta": 2.5,
                        "message": "当前路线需要前期战斗能力。",
                    },
                ],
                "dimensions": {
                    "immediate_power": 5.0,
                    "route_fit": 2.5,
                    "data_completeness": 1.0,
                },
                "data_gaps": [],
            },
            {
                "candidate_id": "1:SHRUG",
                "label": "耸肩无视",
                "display_index": 1,
                "eligible": True,
                "score": 58.0,
                "rank": 2,
                "factors": [
                    {
                        "code": "defense_coverage",
                        "delta": 3,
                        "message": "该候选补充了防御覆盖。",
                    }
                ],
                "dimensions": {
                    "survival": 3.0,
                    "data_completeness": 1.0,
                },
                "data_gaps": ["relic_effect:UNKNOWN_RELIC"],
            },
        ],
        "data_gaps": ["relic_effect:UNKNOWN_RELIC"],
    }


def _route_recommendation() -> dict:
    return {
        "contract_version": 2,
        "decision_id": "run-one:route:4",
        "decision_type": "route_choice",
        "world_sequence": 9,
        "policy_version": "route:test-v1",
        "status": "recommend",
        "confidence": "medium",
        "recommended_candidate_id": "1:4",
        "candidates": [
            {
                "candidate_id": "1:4",
                "label": "RestSite",
                "display_index": 0,
                "eligible": True,
                "score": 67.0,
                "rank": 1,
                "factors": [
                    {
                        "code": "route_campfire_relief",
                        "delta": 6.0,
                        "message": "当前生命压力提高了休息节点价值。",
                    }
                ],
                "dimensions": {
                    "survival": 6.0,
                    "route_fit": 6.0,
                    "data_completeness": 1.0,
                },
                "data_gaps": [],
            }
        ],
        "data_gaps": [],
        "presentation": {
            "kind": "route_paths",
            "origin_node_id": "0:2",
            "primary_path_node_ids": ["1:4"],
            "backup_path_node_ids": [],
            "paths": [
                {
                    "candidate_id": "1:4",
                    "node_ids": ["1:4"],
                    "score": 67.0,
                }
            ],
        },
    }


def _request(
    recommendation: dict,
    *,
    identity: ExplainIdentity | None = None,
    intent: str = "why_recommended",
    candidate_ids: tuple[str, ...] = ("0:ANGER",),
    created_at: datetime = NOW,
    expires_at: datetime | None = None,
    allow_remote_model: bool = False,
) -> ExplainRequest:
    return ExplainRequest(
        request_id=str(uuid4()),
        created_at=created_at,
        expires_at=expires_at or (created_at + timedelta(seconds=10)),
        release_fingerprint=FINGERPRINT,
        identity=identity or _identity(),
        recommendation_digest=canonical_recommendation_digest(
            recommendation
        ),
        intent=intent,
        candidate_ids=candidate_ids,
        allow_remote_model=allow_remote_model,
    )


class ExplainContractTests(unittest.TestCase):
    def test_request_is_strict_and_remote_model_defaults_off(self):
        request = _request(_recommendation())
        self.assertFalse(request.allow_remote_model)
        self.assertEqual(
            ExplainRequest.model_validate_json(
                request.model_dump_json()
            ),
            request,
        )
        payload = request.model_dump(mode="json")
        payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            ExplainRequest.model_validate(payload)
        payload.pop("unexpected")
        payload["allow_remote_model"] = "false"
        with self.assertRaises(ValidationError):
            ExplainRequest.model_validate(payload)

    def test_intents_enforce_candidate_shape_and_expiry_order(self):
        recommendation = _recommendation()
        with self.assertRaises(ValidationError):
            _request(
                recommendation,
                intent="compare_candidates",
                candidate_ids=("0:ANGER",),
            )
        with self.assertRaises(ValidationError):
            _request(
                recommendation,
                intent="why_recommended",
                candidate_ids=("0:ANGER", "1:SHRUG"),
            )
        with self.assertRaises(ValidationError):
            _request(
                recommendation,
                created_at=NOW,
                expires_at=NOW,
            )

    def test_canonical_digest_is_order_stable_and_content_sensitive(self):
        recommendation = _recommendation()
        reversed_root = {
            key: recommendation[key]
            for key in reversed(list(recommendation))
        }
        self.assertEqual(
            canonical_recommendation_digest(recommendation),
            canonical_recommendation_digest(reversed_root),
        )
        changed = deepcopy(recommendation)
        changed["candidates"][0]["factors"][0]["delta"] = 5.25
        self.assertNotEqual(
            canonical_recommendation_digest(recommendation),
            canonical_recommendation_digest(changed),
        )
        invalid = deepcopy(recommendation)
        invalid["candidates"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            canonical_recommendation_digest(invalid)

    def test_digest_accepts_the_formal_recommendation_object(self):
        candidate = CandidateAssessment(
            candidate_id="0:ANGER",
            label="愤怒",
            display_index=0,
            eligible=True,
            score=64.0,
            rank=1,
            factors=(
                {
                    "code": "attack_coverage",
                    "delta": 5.0,
                    "message": "当前牌组缺少直接输出。",
                },
            ),
            dimensions={"immediate_power": 5.0},
        )
        recommendation = Recommendation(
            decision_id="run-one:card-reward:3",
            decision_type="card_reward",
            payload={},
            world_sequence=8,
            policy_version="card_reward:test-v1",
            candidates=(candidate,),
            recommended_candidate_id="0:ANGER",
        )
        self.assertEqual(
            canonical_recommendation_digest(recommendation),
            canonical_recommendation_digest(recommendation.as_dict()),
        )


class ExplainWorkerTests(unittest.TestCase):
    def test_no_model_returns_offline_ready_without_mutating_recommendation(self):
        recommendation = _recommendation()
        original = deepcopy(recommendation)
        request = _request(recommendation, allow_remote_model=True)
        worker = ExplainWorker(clock=lambda: NOW)

        response = worker.submit(
            request,
            current_release_fingerprint=FINGERPRINT,
            current_identity=_identity(),
            recommendation=recommendation,
        )

        self.assertEqual(response.status, "offline_ready")
        self.assertFalse(response.remote_model_used)
        self.assertEqual(
            response.reason_blocks[0].candidate_id,
            "0:ANGER",
        )
        self.assertEqual(
            [factor.code for factor in response.reason_blocks[0].factors],
            ["attack_coverage", "route_fit"],
        )
        self.assertEqual(recommendation, original)
        self.assertIs(worker.latest_response(), response)

    def test_compare_uses_only_requested_canonical_candidate_fields(self):
        recommendation = _recommendation()
        request = _request(
            recommendation,
            intent="compare_candidates",
            candidate_ids=("0:ANGER", "1:SHRUG"),
        )
        response = ExplainWorker(clock=lambda: NOW).submit(
            request,
            current_release_fingerprint=FINGERPRINT,
            current_identity=_identity(),
            recommendation=recommendation,
        )
        serialized = response.model_dump(mode="json")

        self.assertEqual(response.status, "offline_ready")
        self.assertEqual(
            [block["candidate_id"] for block in serialized["reason_blocks"]],
            ["0:ANGER", "1:SHRUG"],
        )
        self.assertIn(
            "relic_effect:UNKNOWN_RELIC",
            serialized["data_gaps"],
        )
        self.assertNotIn("score", json.dumps(serialized, ensure_ascii=False))

    def test_route_risk_intent_uses_route_recommendation_only(self):
        recommendation = _route_recommendation()
        identity = _identity(
            event_id="run-one:9",
            decision_id="run-one:route:4",
            state_revision=9,
            world_sequence=9,
            recommendation_contract_version=2,
            policy_version="route:test-v1",
        )
        request = _request(
            recommendation,
            identity=identity,
            intent="explain_route_risk",
            candidate_ids=("1:4",),
        )
        response = ExplainWorker(clock=lambda: NOW).submit(
            request,
            current_release_fingerprint=FINGERPRINT,
            current_identity=identity,
            recommendation=recommendation,
        )
        self.assertEqual(response.status, "offline_ready")
        self.assertEqual(
            response.reason_blocks[0].factors[0].code,
            "route_campfire_relief",
        )

    def test_expired_stale_digest_and_unknown_candidate_fail_closed(self):
        recommendation = _recommendation()
        worker = ExplainWorker(clock=lambda: NOW)

        expired = _request(
            recommendation,
            created_at=NOW - timedelta(seconds=20),
            expires_at=NOW - timedelta(seconds=10),
        )
        self.assertEqual(
            worker.submit(
                expired,
                current_release_fingerprint=FINGERPRINT,
                current_identity=_identity(),
                recommendation=recommendation,
            ).status,
            "expired",
        )

        current = _request(recommendation)
        self.assertEqual(
            worker.submit(
                current,
                current_release_fingerprint=FINGERPRINT,
                current_identity=_identity(state_revision=9),
                recommendation=recommendation,
            ).status,
            "stale",
        )
        self.assertEqual(
            worker.submit(
                current,
                current_release_fingerprint="b" * 64,
                current_identity=_identity(),
                recommendation=recommendation,
            ).reason_code,
            "release_fingerprint_mismatch",
        )

        changed = deepcopy(recommendation)
        changed["candidates"][0]["factors"][0]["delta"] = 6.0
        self.assertEqual(
            worker.submit(
                current,
                current_release_fingerprint=FINGERPRINT,
                current_identity=_identity(),
                recommendation=changed,
            ).status,
            "digest_mismatch",
        )

        unknown = _request(
            recommendation,
            candidate_ids=("missing",),
        )
        self.assertEqual(
            worker.submit(
                unknown,
                current_release_fingerprint=FINGERPRINT,
                current_identity=_identity(),
                recommendation=recommendation,
            ).status,
            "invalid_candidate",
        )

    def test_new_request_supersedes_inflight_request(self):
        recommendation = _recommendation()
        first_started = threading.Event()
        release_first = threading.Event()

        def blocking_renderer(request, payload, *, created_at):
            if request.candidate_ids == ("0:ANGER",):
                first_started.set()
                self.assertTrue(release_first.wait(timeout=5))
            return build_deterministic_response(
                request,
                payload,
                created_at=created_at,
            )

        worker = ExplainWorker(clock=lambda: NOW)
        first = _request(recommendation)
        second = _request(
            recommendation,
            intent="compare_candidates",
            candidate_ids=("0:ANGER", "1:SHRUG"),
        )
        first_result: list = []

        with mock.patch(
            "explain.worker.build_deterministic_response",
            side_effect=blocking_renderer,
        ):
            thread = threading.Thread(
                target=lambda: first_result.append(worker.submit(
                    first,
                    current_release_fingerprint=FINGERPRINT,
                    current_identity=_identity(),
                    recommendation=recommendation,
                )),
            )
            thread.start()
            self.assertTrue(first_started.wait(timeout=5))
            second_response = worker.submit(
                second,
                current_release_fingerprint=FINGERPRINT,
                current_identity=_identity(),
                recommendation=recommendation,
            )
            release_first.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(first_result[0].status, "cancelled")
        self.assertEqual(second_response.status, "offline_ready")
        self.assertEqual(
            worker.latest_response().request_id,
            second.request_id,
        )

    def test_close_decision_cancels_matching_memory_only(self):
        recommendation = _recommendation()
        worker = ExplainWorker(clock=lambda: NOW)
        request = _request(recommendation)
        worker.submit(
            request,
            current_release_fingerprint=FINGERPRINT,
            current_identity=_identity(),
            recommendation=recommendation,
        )

        self.assertFalse(
            worker.close_decision(
                run_id="other-run",
                decision_id=request.identity.decision_id,
            )
        )
        self.assertIsNotNone(worker.latest_response())
        self.assertTrue(
            worker.close_decision(
                run_id=request.identity.run_id,
                decision_id=request.identity.decision_id,
            )
        )
        self.assertIsNone(worker.latest_response())
        self.assertIsNone(worker.active_request_id)

    def test_close_decision_cancels_inflight_publish(self):
        recommendation = _recommendation()
        started = threading.Event()
        release = threading.Event()

        def blocking_renderer(request, payload, *, created_at):
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return build_deterministic_response(
                request,
                payload,
                created_at=created_at,
            )

        worker = ExplainWorker(clock=lambda: NOW)
        request = _request(recommendation)
        results: list = []
        with mock.patch(
            "explain.worker.build_deterministic_response",
            side_effect=blocking_renderer,
        ):
            thread = threading.Thread(
                target=lambda: results.append(worker.submit(
                    request,
                    current_release_fingerprint=FINGERPRINT,
                    current_identity=_identity(),
                    recommendation=recommendation,
                )),
            )
            thread.start()
            self.assertTrue(started.wait(timeout=5))
            self.assertTrue(worker.close_decision(
                run_id=request.identity.run_id,
                decision_id=request.identity.decision_id,
            ))
            release.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0].status, "cancelled")
        self.assertIsNone(worker.latest_response())


class ExplainBoundaryTests(unittest.TestCase):
    def test_package_has_no_realtime_legacy_rag_model_or_network_imports(self):
        forbidden = {
            "realtime",
            "rag",
            "openai",
            "langgraph",
            "faiss",
            "torch",
            "requests",
            "httpx",
            "socket",
            "urllib",
        }
        observed = set()
        for path in (ROOT / "explain").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    observed.update(
                        alias.name.split(".", 1)[0]
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    observed.add(node.module.split(".", 1)[0])
        self.assertFalse(observed & forbidden, observed & forbidden)


if __name__ == "__main__":
    unittest.main()
