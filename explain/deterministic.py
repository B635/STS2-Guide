"""Deterministic explanation rendering from a canonical Recommendation only."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from explain.contracts import (
    ExplainFactor,
    ExplainReasonBlock,
    ExplainRequest,
    ExplainResponse,
)


_MAX_FACTORS_PER_CANDIDATE = 3


def _selected_candidates(
    request: ExplainRequest,
    recommendation: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in recommendation["candidates"]
    }
    return [by_id[candidate_id] for candidate_id in request.candidate_ids]


def _factor_rows(candidate: Mapping[str, Any]) -> tuple[ExplainFactor, ...]:
    factors = sorted(
        candidate["factors"],
        key=lambda factor: (
            -abs(float(factor["delta"])),
            str(factor["code"]),
            str(factor["message"]),
        ),
    )
    return tuple(
        ExplainFactor(
            code=str(factor["code"]),
            delta=float(factor["delta"]),
            message=str(factor["message"]),
        )
        for factor in factors[:_MAX_FACTORS_PER_CANDIDATE]
    )


def _reason_block(candidate: Mapping[str, Any]) -> ExplainReasonBlock:
    return ExplainReasonBlock(
        candidate_id=str(candidate["candidate_id"]),
        label=str(candidate["label"]),
        factors=_factor_rows(candidate),
        dimensions={
            str(name): (
                float(value) if value is not None else None
            )
            for name, value in candidate["dimensions"].items()
        },
        data_gaps=tuple(str(gap) for gap in candidate["data_gaps"]),
    )


def _summary(
    request: ExplainRequest,
    candidates: list[Mapping[str, Any]],
) -> str:
    labels = "、".join(str(candidate["label"]) for candidate in candidates)
    if request.intent == "why_recommended":
        return (
            f"{labels} 是本次 Recommendation 的当前建议。"
            "以下内容只复述该建议已记录的因子、维度和数据缺口。"
        )
    if request.intent == "compare_candidates":
        return (
            f"以下只比较本次 Recommendation 中的 {labels}，"
            "不重新评分，也不引入外部结论。"
        )
    return (
        f"以下只复述本次 Recommendation 对 {labels} 记录的路线风险"
        "因子、维度和数据缺口。"
    )


def build_deterministic_response(
    request: ExplainRequest,
    recommendation: Mapping[str, Any],
    *,
    created_at: datetime,
) -> ExplainResponse:
    """Render an offline explanation without reading or mutating other state."""

    candidates = _selected_candidates(request, recommendation)
    blocks = tuple(_reason_block(candidate) for candidate in candidates)
    gaps = tuple(dict.fromkeys(
        (
            *(str(gap) for gap in recommendation["data_gaps"]),
            *(
                gap
                for block in blocks
                for gap in block.data_gaps
            ),
        )
    ))
    return ExplainResponse(
        request_id=request.request_id,
        created_at=created_at,
        release_fingerprint=request.release_fingerprint,
        identity=request.identity,
        recommendation_digest=request.recommendation_digest,
        status="offline_ready",
        summary=_summary(request, candidates),
        reason_blocks=blocks,
        data_gaps=gaps,
    )
