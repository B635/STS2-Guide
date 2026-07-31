"""Strict, policy-neutral effects and recommendation assembly helpers.

These contracts deliberately contain no text parser.  Runtime policies may
only score effects that an already-verified adapter or versioned catalog has
expressed as structured data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from advisor.decision_core import (
    RECOMMENDATION_DIMENSIONS,
    CandidateAssessment,
    DecisionCandidate,
    DecisionRequest,
    Recommendation,
)


EFFECT_KINDS = frozenset({
    "hp_delta",
    "max_hp_delta",
    "gold_delta",
    "add_card",
    "remove_card",
    "upgrade_card",
    "transform_card",
    "add_relic",
    "remove_relic",
    "add_potion",
    "remove_potion",
    "start_combat",
    "followup_choice",
    "no_op",
})
EFFECT_CERTAINTIES = frozenset({"exact", "bounded", "unknown"})
TARGET_MODES = frozenset({"none", "specific", "choose", "random"})
_NUMERIC_EFFECTS = frozenset({"hp_delta", "max_hp_delta", "gold_delta"})
_ENTITY_EFFECTS = frozenset({
    "add_card",
    "remove_card",
    "upgrade_card",
    "transform_card",
    "add_relic",
    "remove_relic",
    "add_potion",
    "remove_potion",
})
_ENTITY_TYPES = {
    "add_card": "cards",
    "remove_card": "cards",
    "upgrade_card": "cards",
    "transform_card": "cards",
    "add_relic": "relics",
    "remove_relic": "relics",
    "add_potion": "potions",
    "remove_potion": "potions",
}


def _finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ChoiceEffect:
    kind: str
    amount: float | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    target_mode: str = "none"
    certainty: str = "exact"
    source_code: str = ""
    child_decision_type: str | None = None

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "ChoiceEffect":
        if not isinstance(value, Mapping):
            raise ValueError("choice effect must be an object")
        allowed = {
            "kind",
            "amount",
            "min_amount",
            "max_amount",
            "entity_type",
            "entity_id",
            "target_mode",
            "certainty",
            "source_code",
            "child_decision_type",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown choice effect fields: {sorted(unknown)}")
        return cls(
            kind=str(value.get("kind") or "").strip(),
            amount=value.get("amount"),
            min_amount=value.get("min_amount"),
            max_amount=value.get("max_amount"),
            entity_type=(
                str(value["entity_type"]).strip()
                if value.get("entity_type") is not None
                else None
            ),
            entity_id=(
                str(value["entity_id"]).strip()
                if value.get("entity_id") is not None
                else None
            ),
            target_mode=str(value.get("target_mode") or "none").strip(),
            certainty=str(value.get("certainty") or "exact").strip(),
            source_code=str(value.get("source_code") or "").strip(),
            child_decision_type=(
                str(value["child_decision_type"]).strip()
                if value.get("child_decision_type") is not None
                else None
            ),
        )

    def __post_init__(self) -> None:
        if self.kind not in EFFECT_KINDS:
            raise ValueError(f"unsupported choice effect: {self.kind}")
        if self.certainty not in EFFECT_CERTAINTIES:
            raise ValueError("unsupported effect certainty")
        if self.target_mode not in TARGET_MODES:
            raise ValueError("unsupported effect target mode")
        if not self.source_code:
            raise ValueError("choice effect requires a source_code")
        if self.child_decision_type is not None:
            if self.kind != "followup_choice":
                raise ValueError(
                    "child_decision_type is only valid for followup_choice"
                )
            if self.certainty != "exact":
                raise ValueError(
                    "child_decision_type requires exact effect certainty"
                )
            if self.child_decision_type != "card_reward":
                raise ValueError("unsupported child decision type")

        if self.amount is not None:
            object.__setattr__(
                self,
                "amount",
                _finite_number(self.amount, "effect amount"),
            )
        if self.min_amount is not None:
            object.__setattr__(
                self,
                "min_amount",
                _finite_number(self.min_amount, "effect min_amount"),
            )
        if self.max_amount is not None:
            object.__setattr__(
                self,
                "max_amount",
                _finite_number(self.max_amount, "effect max_amount"),
            )
        if (
            self.min_amount is not None
            and self.max_amount is not None
            and self.min_amount > self.max_amount
        ):
            raise ValueError("effect amount bounds are inverted")

        if self.kind in _NUMERIC_EFFECTS:
            if self.certainty == "exact" and self.amount is None:
                raise ValueError("exact numeric effect requires amount")
            if self.certainty == "bounded" and (
                self.min_amount is None or self.max_amount is None
            ):
                raise ValueError("bounded numeric effect requires both bounds")
            for numeric in (
                self.amount,
                self.min_amount,
                self.max_amount,
            ):
                if numeric is not None and not float(numeric).is_integer():
                    raise ValueError(
                        "HP and gold effects require integer amounts"
                    )
        if self.kind in _ENTITY_EFFECTS:
            expected_type = _ENTITY_TYPES[self.kind]
            if self.entity_type != expected_type:
                raise ValueError(
                    f"{self.kind} requires entity_type={expected_type}"
                )
            if self.target_mode == "none":
                raise ValueError("entity effect requires an explicit target mode")
            if self.target_mode == "specific" and not self.entity_id:
                raise ValueError("specific entity effect requires entity_id")


@dataclass(frozen=True)
class EffectBundle:
    effects: tuple[ChoiceEffect, ...]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        required: bool = True,
    ) -> "EffectBundle":
        raw = payload.get("effects")
        if raw is None:
            if required:
                raise ValueError("candidate requires structured effects")
            return cls(())
        if (
            isinstance(raw, (str, bytes, Mapping))
            or not isinstance(raw, (list, tuple))
        ):
            raise ValueError("candidate effects must be an array")
        effects = tuple(ChoiceEffect.create(item) for item in raw)
        if required and not effects:
            raise ValueError("candidate requires at least one effect")
        return cls(effects)

    @property
    def is_fully_exact(self) -> bool:
        return all(effect.certainty == "exact" for effect in self.effects)


def bundle_with_costs(
    payload: Mapping[str, Any],
    *,
    effects_required: bool = True,
) -> EffectBundle:
    """Merge explicit envelope costs into the structured effect projection."""
    bundle = EffectBundle.from_payload(
        payload,
        required=effects_required,
    )
    raw_costs = payload.get("costs") or ()
    if (
        isinstance(raw_costs, (str, bytes, Mapping))
        or not isinstance(raw_costs, (list, tuple))
    ):
        raise ValueError("candidate costs must be an array")
    effects = list(bundle.effects)
    effect_kind = {
        "gold": "gold_delta",
        "hp": "hp_delta",
        "max_hp": "max_hp_delta",
    }
    for cost in raw_costs:
        if not isinstance(cost, Mapping):
            raise ValueError("candidate cost must be an object")
        if set(cost) - {"kind", "amount", "resource_id"}:
            raise ValueError("candidate cost has unknown fields")
        kind = str(cost.get("kind") or "").strip()
        amount = cost.get("amount")
        if (
            isinstance(amount, bool)
            or not isinstance(amount, int)
            or amount < 0
        ):
            raise ValueError("candidate cost amount must be a non-negative integer")
        if kind not in effect_kind:
            raise ValueError(f"unsupported candidate cost: {kind or 'missing'}")
        effects.append(ChoiceEffect(
            kind=effect_kind[kind],
            amount=-float(amount),
            target_mode="none",
            certainty="exact",
            source_code=f"candidate_cost:{kind}",
        ))
    return EffectBundle(tuple(effects))


def actual_gold_cost(payload: Mapping[str, Any]) -> int:
    raw_costs = payload.get("costs")
    if not isinstance(raw_costs, (list, tuple)):
        raise ValueError("merchant candidate requires costs")
    gold_costs = [
        cost
        for cost in raw_costs
        if isinstance(cost, Mapping)
        and str(cost.get("kind") or "").strip() == "gold"
    ]
    if len(gold_costs) != 1:
        raise ValueError("merchant candidate requires one gold cost")
    amount = gold_costs[0].get("amount")
    if (
        isinstance(amount, bool)
        or not isinstance(amount, int)
        or amount < 0
    ):
        raise ValueError("merchant gold cost must be a non-negative integer")
    return amount


def factor(
    code: str,
    delta: float,
    message: str,
    dimension: str,
) -> dict[str, Any]:
    if dimension not in RECOMMENDATION_DIMENSIONS:
        raise ValueError(f"unknown recommendation dimension: {dimension}")
    numeric = _finite_number(delta, "factor delta")
    return {
        "code": str(code),
        "delta": round(numeric, 4),
        "message": str(message),
        "dimension": dimension,
    }


@dataclass
class AssessmentDraft:
    candidate: DecisionCandidate
    base_score: float = 50.0
    factors: list[dict[str, Any]] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    blocking_unknown: bool = False
    safety_violation: bool = False

    def add(
        self,
        code: str,
        delta: float,
        message: str,
        dimension: str,
    ) -> None:
        self.factors.append(factor(code, delta, message, dimension))

    def gap(self, code: str, *, blocking: bool = True) -> None:
        code = str(code).strip()
        if not code:
            raise ValueError("data gap must not be empty")
        if code not in self.data_gaps:
            self.data_gaps.append(code)
        if blocking:
            self.blocking_unknown = True

    @property
    def score(self) -> float | None:
        if not self.candidate.eligible or self.blocking_unknown:
            return None
        total = self.base_score + sum(
            float(item["delta"]) for item in self.factors
        )
        if not math.isfinite(total):
            raise ValueError("candidate score is not finite")
        return round(max(0.0, min(100.0, total)), 4)

    def dimensions(self) -> dict[str, float]:
        values = {name: 0.0 for name in RECOMMENDATION_DIMENSIONS}
        for item in self.factors:
            dimension = str(item["dimension"])
            values[dimension] += float(item["delta"])
        values["data_completeness"] = (
            0.0 if self.blocking_unknown else 1.0
        )
        return {
            name: round(float(value), 4)
            for name, value in values.items()
        }


def finalize_recommendation(
    request: DecisionRequest,
    drafts: Iterable[AssessmentDraft],
    *,
    policy_version: str,
    payload: Mapping[str, Any] | None = None,
) -> Recommendation:
    draft_list = list(drafts)
    if len(draft_list) != len(request.candidates):
        raise ValueError("policy must draft every request candidate")
    for expected, draft in zip(request.candidates, draft_list):
        if draft.candidate.candidate_id != expected.candidate_id:
            raise ValueError("draft candidate order does not match request")

    ranked = sorted(
        (
            draft
            for draft in draft_list
            if (
                draft.candidate.eligible
                and draft.score is not None
                and not draft.safety_violation
            )
        ),
        key=lambda draft: (
            -float(draft.score),
            int(draft.candidate.display_index or 0),
        ),
    )
    rank_by_id = {
        draft.candidate.candidate_id: index
        for index, draft in enumerate(ranked, start=1)
    }
    assessments = tuple(
        CandidateAssessment(
            candidate_id=draft.candidate.candidate_id,
            label=draft.candidate.label,
            display_index=int(draft.candidate.display_index or 0),
            eligible=draft.candidate.eligible,
            score=(None if draft.safety_violation else draft.score),
            rank=rank_by_id.get(draft.candidate.candidate_id),
            factors=tuple(draft.factors),
            dimensions=draft.dimensions(),
            data_gaps=tuple(draft.data_gaps),
        )
        for draft in draft_list
    )

    has_unknown_eligible = any(
        draft.candidate.eligible and draft.blocking_unknown
        for draft in draft_list
    )
    recommended = (
        None
        if has_unknown_eligible or not ranked
        else ranked[0].candidate.candidate_id
    )
    status = "uncertain" if recommended is None else "recommend"
    gaps = tuple(dict.fromkeys(
        (
            *(
                gap
                for draft in draft_list
                for gap in draft.data_gaps
            ),
            *(
                f"capture_warning:{warning}"
                for warning in request.world.capture_warnings
            ),
        )
    ))
    result_payload = {
        "method": policy_version,
        "decision_type": request.decision_type,
        **dict(payload or {}),
    }
    return Recommendation(
        decision_id=request.decision_id,
        decision_type=request.decision_type,
        payload=result_payload,
        world_sequence=request.world.sequence,
        policy_version=policy_version,
        candidates=assessments,
        recommended_candidate_id=recommended,
        status=status,
        confidence=("low" if gaps or status == "uncertain" else "medium"),
        data_gaps=gaps,
    )
