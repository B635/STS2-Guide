"""Strict v1 contracts and canonical Recommendation hashing.

The digest covers the complete public ``Recommendation.as_dict()`` payload.
Explanation rendering may consume only candidate factors, dimensions, and data
gaps, but identity validation must notice any change to the canonical
Recommendation envelope.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Dict, Literal, Mapping, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


EXPLAIN_CONTRACT_VERSION = 1
ExplainIntent = Literal[
    "why_recommended",
    "compare_candidates",
    "explain_route_risk",
]
ExplainStatus = Literal[
    "offline_ready",
    "expired",
    "stale",
    "digest_mismatch",
    "invalid_candidate",
    "invalid_recommendation",
    "cancelled",
]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECOMMENDATION_REQUIRED_FIELDS = {
    "contract_version",
    "decision_id",
    "decision_type",
    "world_sequence",
    "policy_version",
    "status",
    "confidence",
    "recommended_candidate_id",
    "candidates",
    "data_gaps",
}
_RECOMMENDATION_OPTIONAL_FIELDS = {"presentation"}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "label",
    "display_index",
    "eligible",
    "score",
    "rank",
    "factors",
    "dimensions",
    "data_gaps",
}
_FACTOR_REQUIRED_FIELDS = {"code", "delta", "message"}
_RECOMMENDATION_DIMENSIONS = {
    "immediate_power",
    "survival",
    "long_term_growth",
    "resource_efficiency",
    "deck_burden",
    "synergy",
    "route_fit",
    "data_completeness",
}


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )


class ExplainIdentity(_StrictFrozenModel):
    """Exact live identity that an asynchronous explanation must match."""

    run_id: str = Field(min_length=1, max_length=256)
    event_id: str = Field(min_length=1, max_length=512)
    decision_id: str = Field(min_length=1, max_length=512)
    state_revision: int = Field(ge=1)
    world_sequence: int = Field(ge=1)
    recommendation_contract_version: int = Field(ge=1)
    policy_version: str = Field(min_length=1, max_length=256)

    @field_validator(
        "run_id",
        "event_id",
        "decision_id",
        "policy_version",
    )
    @classmethod
    def _no_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("identity strings must not contain outer whitespace")
        return value


class ExplainRequest(_StrictFrozenModel):
    """A user-triggered request bound to one current Recommendation."""

    contract_version: Literal[EXPLAIN_CONTRACT_VERSION] = (
        EXPLAIN_CONTRACT_VERSION
    )
    request_id: str = Field(min_length=36, max_length=36)
    created_at: datetime
    expires_at: datetime
    release_fingerprint: str = Field(min_length=64, max_length=64)
    identity: ExplainIdentity
    recommendation_digest: str = Field(min_length=64, max_length=64)
    intent: ExplainIntent
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    locale: str = Field(default="zh-CN", min_length=2, max_length=35)
    allow_remote_model: bool = False

    @field_validator("request_id")
    @classmethod
    def _canonical_uuid(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError) as exc:
            raise ValueError("request_id must be a UUID") from exc
        if str(parsed) != value.lower():
            raise ValueError("request_id must use canonical UUID syntax")
        return value.lower()

    @field_validator("release_fingerprint", "recommendation_digest")
    @classmethod
    def _lowercase_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("digest fields must be lowercase SHA-256 hex")
        return value

    @field_validator("candidate_ids")
    @classmethod
    def _candidate_ids_are_stable(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not candidate_id.strip() for candidate_id in value):
            raise ValueError("candidate IDs must not be blank")
        if any(candidate_id != candidate_id.strip() for candidate_id in value):
            raise ValueError("candidate IDs must not contain outer whitespace")
        if len(value) != len(set(value)):
            raise ValueError("candidate IDs must be unique")
        return value

    @field_validator("created_at", "expires_at")
    @classmethod
    def _timestamps_are_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("explanation timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_intent_shape(self) -> "ExplainRequest":
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        expected_count = 2 if self.intent == "compare_candidates" else 1
        if len(self.candidate_ids) != expected_count:
            raise ValueError(
                f"{self.intent} requires exactly {expected_count} candidate ID(s)"
            )
        return self


class ExplainFactor(_StrictFrozenModel):
    code: str = Field(min_length=1, max_length=256)
    delta: float
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("delta")
    @classmethod
    def _finite_delta(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("factor delta must be finite")
        return value


class ExplainReasonBlock(_StrictFrozenModel):
    candidate_id: str = Field(min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=512)
    factors: tuple[ExplainFactor, ...] = ()
    dimensions: Dict[str, Optional[float]] = Field(default_factory=dict)
    data_gaps: tuple[str, ...] = ()

    @field_validator("dimensions")
    @classmethod
    def _finite_dimensions(
        cls,
        value: Dict[str, Optional[float]],
    ) -> Dict[str, Optional[float]]:
        for name, dimension in value.items():
            if not name.strip():
                raise ValueError("dimension names must not be blank")
            if name not in _RECOMMENDATION_DIMENSIONS:
                raise ValueError("unknown Recommendation dimension")
            if dimension is not None and not math.isfinite(dimension):
                raise ValueError("dimension values must be finite or null")
        return value

    @field_validator("data_gaps")
    @classmethod
    def _unique_data_gaps(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not gap.strip() for gap in value):
            raise ValueError("data gaps must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("data gaps must be unique")
        return value


class ExplainResponse(_StrictFrozenModel):
    """A bounded response that cannot replace or mutate a Recommendation."""

    contract_version: Literal[EXPLAIN_CONTRACT_VERSION] = (
        EXPLAIN_CONTRACT_VERSION
    )
    request_id: str = Field(min_length=36, max_length=36)
    created_at: datetime
    release_fingerprint: str = Field(min_length=64, max_length=64)
    identity: ExplainIdentity
    recommendation_digest: str = Field(min_length=64, max_length=64)
    status: ExplainStatus
    mode: Literal["deterministic"] = "deterministic"
    remote_model_used: Literal[False] = False
    summary: str = Field(max_length=4000)
    reason_code: Optional[str] = Field(default=None, max_length=256)
    reason_blocks: tuple[ExplainReasonBlock, ...] = ()
    data_gaps: tuple[str, ...] = ()

    @field_validator("request_id")
    @classmethod
    def _response_uuid(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError) as exc:
            raise ValueError("request_id must be a UUID") from exc
        if str(parsed) != value.lower():
            raise ValueError("request_id must use canonical UUID syntax")
        return value.lower()

    @field_validator("release_fingerprint", "recommendation_digest")
    @classmethod
    def _response_digest(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("response digests must be lowercase SHA-256")
        return value

    @field_validator("created_at")
    @classmethod
    def _response_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("response timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _status_shape(self) -> "ExplainResponse":
        if self.status == "offline_ready":
            if self.reason_code is not None or not self.reason_blocks:
                raise ValueError(
                    "offline_ready requires reason blocks and no rejection code"
                )
        elif self.reason_code is None or self.reason_blocks:
            raise ValueError(
                "rejected responses require a reason code and no reason blocks"
            )
        return self


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain outer whitespace")
    return value


def _validate_data_gaps(value: Any, field: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    normalized = [
        _require_non_empty_string(gap, f"{field}[]")
        for gap in value
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must contain unique values")


def _recommendation_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        value = value.as_dict()
    if not isinstance(value, Mapping):
        raise ValueError("Recommendation must be a mapping or expose as_dict()")
    return value


def canonicalize_recommendation(value: Any) -> dict[str, Any]:
    """Validate and deep-clone the public canonical Recommendation payload."""

    mapping = _recommendation_mapping(value)
    fields = set(mapping)
    if not _RECOMMENDATION_REQUIRED_FIELDS.issubset(fields):
        missing = sorted(_RECOMMENDATION_REQUIRED_FIELDS - fields)
        raise ValueError(f"Recommendation is missing fields: {missing}")
    unknown = fields - (
        _RECOMMENDATION_REQUIRED_FIELDS | _RECOMMENDATION_OPTIONAL_FIELDS
    )
    if unknown:
        raise ValueError(f"Recommendation has unknown fields: {sorted(unknown)}")

    if not _is_int(mapping["contract_version"]) or mapping[
        "contract_version"
    ] < 1:
        raise ValueError("Recommendation contract_version must be positive")
    _require_non_empty_string(mapping["decision_id"], "decision_id")
    _require_non_empty_string(mapping["decision_type"], "decision_type")
    if not _is_int(mapping["world_sequence"]) or mapping["world_sequence"] < 1:
        raise ValueError("Recommendation world_sequence must be positive")
    _require_non_empty_string(mapping["policy_version"], "policy_version")
    _require_non_empty_string(mapping["status"], "status")
    _require_non_empty_string(mapping["confidence"], "confidence")
    recommended = mapping["recommended_candidate_id"]
    if recommended is not None:
        _require_non_empty_string(recommended, "recommended_candidate_id")
    _validate_data_gaps(mapping["data_gaps"], "data_gaps")

    candidates = mapping["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Recommendation candidates must be a non-empty array")
    candidate_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidates[{index}] must be an object")
        if set(candidate) != _CANDIDATE_FIELDS:
            raise ValueError(
                f"candidates[{index}] does not match the canonical contract"
            )
        candidate_id = _require_non_empty_string(
            candidate["candidate_id"],
            f"candidates[{index}].candidate_id",
        )
        candidate_ids.append(candidate_id)
        _require_non_empty_string(
            candidate["label"],
            f"candidates[{index}].label",
        )
        if (
            not _is_int(candidate["display_index"])
            or candidate["display_index"] < 0
        ):
            raise ValueError("candidate display_index must not be negative")
        if not isinstance(candidate["eligible"], bool):
            raise ValueError("candidate eligible must be a boolean")
        if candidate["score"] is not None and not _is_number(
            candidate["score"]
        ):
            raise ValueError("candidate score must be finite or null")
        if candidate["rank"] is not None and (
            not _is_int(candidate["rank"]) or candidate["rank"] < 1
        ):
            raise ValueError("candidate rank must be positive or null")

        factors = candidate["factors"]
        if not isinstance(factors, list):
            raise ValueError("candidate factors must be an array")
        for factor_index, factor in enumerate(factors):
            if not isinstance(factor, Mapping):
                raise ValueError("candidate factors must contain objects")
            if not _FACTOR_REQUIRED_FIELDS.issubset(factor):
                raise ValueError(
                    f"factor {factor_index} is missing canonical fields"
                )
            _require_non_empty_string(factor["code"], "factor.code")
            _require_non_empty_string(factor["message"], "factor.message")
            if not _is_number(factor["delta"]):
                raise ValueError("factor delta must be finite")

        dimensions = candidate["dimensions"]
        if not isinstance(dimensions, Mapping):
            raise ValueError("candidate dimensions must be an object")
        for name, dimension in dimensions.items():
            _require_non_empty_string(name, "dimension name")
            if name not in _RECOMMENDATION_DIMENSIONS:
                raise ValueError("unknown Recommendation dimension")
            if dimension is not None and not _is_number(dimension):
                raise ValueError("dimension values must be finite or null")
        _validate_data_gaps(
            candidate["data_gaps"],
            f"candidates[{index}].data_gaps",
        )

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Recommendation candidate IDs must be unique")
    if recommended is not None and recommended not in candidate_ids:
        raise ValueError("recommended candidate is absent from candidates")

    try:
        serialized = json.dumps(
            mapping,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Recommendation is not canonical JSON") from exc
    cloned = json.loads(serialized)
    if not isinstance(cloned, dict):
        raise ValueError("Recommendation root must be an object")
    return cloned


def _canonical_recommendation_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_recommendation_digest(value: Any) -> str:
    """Return a stable SHA-256 over the complete canonical Recommendation."""

    payload = canonicalize_recommendation(value)
    return hashlib.sha256(_canonical_recommendation_bytes(payload)).hexdigest()
