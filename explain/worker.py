"""Latest-only in-memory worker for deterministic explanations."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
from typing import Any, Callable, Mapping

from explain.contracts import (
    ExplainIdentity,
    ExplainRequest,
    ExplainResponse,
    _canonical_recommendation_bytes,
    canonicalize_recommendation,
)
from explain.deterministic import build_deterministic_response

import hashlib

class ExplainWorker:
    """Own at most one active request and one latest response in memory.

    ``submit`` is synchronous by design; a future bridge may execute it on a
    separate low-priority process.  The generation token ensures that an older
    concurrent call can never publish after a newer request or decision close.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
    ):
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._generation = 0
        self._active: tuple[int, ExplainRequest] | None = None
        self._latest_response: ExplainResponse | None = None

    @property
    def active_request_id(self) -> str | None:
        with self._lock:
            return (
                self._active[1].request_id
                if self._active is not None
                else None
            )

    def latest_response(self) -> ExplainResponse | None:
        with self._lock:
            return self._latest_response

    def cancel_active(self) -> bool:
        """Cancel the active generation and forget its unpublished response."""

        with self._lock:
            if self._active is None and self._latest_response is None:
                return False
            self._generation += 1
            self._active = None
            self._latest_response = None
            return True

    def close_decision(self, *, run_id: str, decision_id: str) -> bool:
        """Drop only state that belongs to the decision being closed."""

        with self._lock:
            active_matches = (
                self._active is not None
                and self._active[1].identity.run_id == run_id
                and self._active[1].identity.decision_id == decision_id
            )
            response_matches = (
                self._latest_response is not None
                and self._latest_response.identity.run_id == run_id
                and self._latest_response.identity.decision_id == decision_id
            )
            if not active_matches and not response_matches:
                return False
            self._generation += 1
            self._active = None
            self._latest_response = None
            return True

    def submit(
        self,
        request: ExplainRequest,
        *,
        current_release_fingerprint: str,
        current_identity: ExplainIdentity,
        recommendation: Any,
    ) -> ExplainResponse:
        """Validate and explain one current canonical Recommendation.

        The caller supplies the current authoritative identity.  This worker
        never opens a checkpoint, reads game state, writes a file, or invokes a
        model.  A newer ``submit`` or ``close_decision`` invalidates this call's
        generation token before it can publish.
        """

        if not isinstance(request, ExplainRequest):
            request = ExplainRequest.model_validate(request)
        if not isinstance(current_identity, ExplainIdentity):
            current_identity = ExplainIdentity.model_validate(
                current_identity
            )
        token = self._begin(request)
        now = self._aware_now()

        if request.expires_at <= now:
            response = self._rejection(
                request,
                now,
                status="expired",
                reason_code="request_expired",
                summary="解释请求已过期，未读取或复用旧建议。",
            )
            return self._finish(token, request, response)

        if request.release_fingerprint != current_release_fingerprint:
            response = self._rejection(
                request,
                now,
                status="stale",
                reason_code="release_fingerprint_mismatch",
                summary="Guide release 已经变化，旧解释请求已拒绝。",
            )
            return self._finish(token, request, response)

        if request.identity != current_identity:
            response = self._rejection(
                request,
                now,
                status="stale",
                reason_code="identity_mismatch",
                summary="当前局面或建议身份已经变化，旧解释请求已拒绝。",
            )
            return self._finish(token, request, response)

        try:
            payload = canonicalize_recommendation(recommendation)
        except (TypeError, ValueError):
            response = self._rejection(
                request,
                now,
                status="invalid_recommendation",
                reason_code="noncanonical_recommendation",
                summary="Recommendation 不符合正式规范，未生成解释。",
            )
            return self._finish(token, request, response)

        actual_digest = hashlib.sha256(
            _canonical_recommendation_bytes(payload)
        ).hexdigest()
        if actual_digest != request.recommendation_digest:
            response = self._rejection(
                request,
                now,
                status="digest_mismatch",
                reason_code="recommendation_digest_mismatch",
                summary="Recommendation 内容已经变化，旧解释请求已拒绝。",
            )
            return self._finish(token, request, response)

        if not self._recommendation_matches_identity(payload, request.identity):
            response = self._rejection(
                request,
                now,
                status="stale",
                reason_code="recommendation_identity_mismatch",
                summary="Recommendation 与当前决策身份不一致，未生成解释。",
            )
            return self._finish(token, request, response)

        invalid_reason = self._validate_candidates(request, payload)
        if invalid_reason is not None:
            response = self._rejection(
                request,
                now,
                status="invalid_candidate",
                reason_code=invalid_reason,
                summary="解释请求引用了无效或不适用的候选项。",
            )
            return self._finish(token, request, response)

        response = build_deterministic_response(
            request,
            payload,
            created_at=now,
        )
        return self._finish(token, request, response)

    def _begin(self, request: ExplainRequest) -> int:
        with self._lock:
            self._generation += 1
            token = self._generation
            self._active = (token, request)
            self._latest_response = None
            return token

    def _finish(
        self,
        token: int,
        request: ExplainRequest,
        response: ExplainResponse,
    ) -> ExplainResponse:
        with self._lock:
            if (
                self._active is None
                or self._generation != token
                or self._active[0] != token
                or self._active[1].request_id != request.request_id
            ):
                return self._rejection(
                    request,
                    self._aware_now(),
                    status="cancelled",
                    reason_code="superseded_or_closed",
                    summary="解释请求已被更新的请求或决策关闭取消。",
                )
            self._active = None
            self._latest_response = response
            return response

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ExplainWorker clock must return an aware datetime")
        return value

    @staticmethod
    def _recommendation_matches_identity(
        recommendation: Mapping[str, Any],
        identity: ExplainIdentity,
    ) -> bool:
        return (
            recommendation["decision_id"] == identity.decision_id
            and recommendation["world_sequence"] == identity.world_sequence
            and recommendation["contract_version"]
            == identity.recommendation_contract_version
            and recommendation["policy_version"] == identity.policy_version
        )

    @staticmethod
    def _validate_candidates(
        request: ExplainRequest,
        recommendation: Mapping[str, Any],
    ) -> str | None:
        candidates = {
            candidate["candidate_id"]: candidate
            for candidate in recommendation["candidates"]
        }
        if any(
            candidate_id not in candidates
            for candidate_id in request.candidate_ids
        ):
            return "unknown_candidate_id"
        if request.intent == "why_recommended" and (
            recommendation["recommended_candidate_id"]
            != request.candidate_ids[0]
        ):
            return "candidate_is_not_recommended"
        if request.intent == "explain_route_risk" and (
            recommendation["decision_type"] != "route_choice"
        ):
            return "route_intent_requires_route_choice"
        return None

    @staticmethod
    def _rejection(
        request: ExplainRequest,
        created_at: datetime,
        *,
        status: str,
        reason_code: str,
        summary: str,
    ) -> ExplainResponse:
        return ExplainResponse(
            request_id=request.request_id,
            created_at=created_at,
            release_fingerprint=request.release_fingerprint,
            identity=request.identity,
            recommendation_digest=request.recommendation_digest,
            status=status,
            summary=summary,
            reason_code=reason_code,
        )
