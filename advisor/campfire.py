"""Deterministic campfire action policy using structured effects only."""
from __future__ import annotations

import math
from typing import Any, Mapping

from advisor.choice_effects import (
    AssessmentDraft,
    EffectBundle,
    bundle_with_costs,
    finalize_recommendation,
)
from advisor.decision_core import DecisionRequest, Recommendation
from advisor.resource_budget import (
    ResourceBudget,
    build_resource_budget,
    score_effect_bundle,
)
from storage.relational import RelationalRepository
from advisor.versions import CAMPFIRE_POLICY_VERSION
from advisor.scoring_preferences import DEFAULT_SCORING_PREFERENCES, ScoringPreferences


CAMPFIRE_ACTION = "campfire_action"
# Guide-owned rest-or-smith rule (data/guide_mechanics.json). Set by hand from
# the player's run history; see scripts/analyze_rest_choices.py.
REST_RULE_KEY = "campfire_rest_rule_v1"
_REST_RULE_VERSION = "campfire-rest-rule-v1"
_REST_RULE_FIELDS = ("heal_below_hp_ratio", "rule_margin")


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def rest_rule_parameter_gaps(params: object) -> tuple[str, ...]:
    """Fail closed: the rule only runs on complete, valid hand-set values."""
    if not isinstance(params, Mapping):
        return ("campfire_rule:parameters_missing",)
    if params.get("version") != _REST_RULE_VERSION:
        return ("campfire_rule:parameters_unsupported_version",)
    missing = [key for key in _REST_RULE_FIELDS if key not in params]
    if missing:
        return tuple(f"campfire_rule:parameter_missing:{key}" for key in missing)
    invalid = []
    ratio = params["heal_below_hp_ratio"]
    if not _finite_number(ratio) or not 0 < float(ratio) < 1:
        invalid.append("heal_below_hp_ratio")
    margin = params["rule_margin"]
    if not _finite_number(margin) or not 0 < float(margin) <= 20:
        invalid.append("rule_margin")
    return tuple(f"campfire_rule:parameter_invalid:{key}" for key in invalid)


def rest_rule_prefers_heal(hp: int, max_hp: int, params: Mapping[str, Any]) -> bool:
    """Rest below the HP ratio threshold, smith from it upwards."""
    return hp / max_hp < float(params["heal_below_hp_ratio"])


def _rest_shape(bundle: EffectBundle) -> str | None:
    """Recognise the plain rest and smith actions by their structured effect."""
    if len(bundle.effects) != 1:
        return None
    effect = bundle.effects[0]
    if effect.kind == "hp_delta" and (
        (effect.amount or 0) > 0 or (effect.min_amount or 0) > 0
    ):
        return "heal"
    if effect.kind == "upgrade_card":
        return "smith"
    return None


class CampfirePolicy:
    decision_type = CAMPFIRE_ACTION

    def __init__(self, repository: RelationalRepository,
                 preferences: ScoringPreferences = DEFAULT_SCORING_PREFERENCES):
        self.repository = repository
        self.preferences = preferences

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("campfire policy received another decision type")
        budget = build_resource_budget(request.world)
        drafts = []
        shapes: dict[str, list[AssessmentDraft]] = {"heal": [], "smith": []}
        for candidate in request.candidates:
            draft = AssessmentDraft(candidate, preferences=self.preferences)
            drafts.append(draft)
            if not candidate.eligible:
                continue
            kind = str(
                candidate.payload.get("candidate_kind") or ""
            ).strip()
            if kind == "leave":
                draft.add(
                    "campfire_leave",
                    -2.0,
                    "离开会放弃本次篝火机会。",
                    "long_term_growth",
                )
                continue
            if kind != "rest_action":
                draft.gap(
                    f"campfire:unknown_candidate_kind:{kind or 'missing'}"
                )
                continue
            if not str(candidate.payload.get("action_id") or "").strip():
                draft.gap("campfire:action_id_missing")
                continue
            shape = self._score_effect_action(
                draft,
                request,
                budget,
            )
            if shape is not None:
                shapes[shape].append(draft)
        if shapes["heal"] and shapes["smith"]:
            self._apply_rest_rule(shapes["heal"], shapes["smith"], budget)
        return finalize_recommendation(
            request,
            drafts,
            policy_version=CAMPFIRE_POLICY_VERSION,
        )

    def _score_effect_action(
        self,
        draft: AssessmentDraft,
        request: DecisionRequest,
        budget,
    ) -> str | None:
        try:
            bundle = bundle_with_costs(draft.candidate.payload)
        except ValueError as exc:
            draft.gap(f"campfire:effects_invalid:{_gap_token(exc)}")
            return None
        score_effect_bundle(
            draft,
            budget,
            bundle,
            request.world,
            self.repository,
            replacement_supported=(
                draft.candidate.payload.get("replacement_supported") is True
            ),
        )
        return _rest_shape(bundle)

    def _apply_rest_rule(
        self,
        heals: list[AssessmentDraft],
        smiths: list[AssessmentDraft],
        budget: ResourceBudget,
    ) -> None:
        """Order rest against smith by the HP rule; other candidates keep
        their effect scores."""
        pair = (*heals, *smiths)
        params = self.repository.mechanic_constant(REST_RULE_KEY)
        gaps = list(rest_rule_parameter_gaps(params))
        if not budget.health.known:
            gaps.append("campfire_rule:hp_missing")
        if len(heals) > 1 or len(smiths) > 1:
            gaps.append("campfire_rule:multiple_rest_or_smith_actions")
        for draft in pair:
            for gap in gaps:
                draft.gap(gap)
        if gaps or any(
            draft.blocking_unknown or draft.safety_violation for draft in pair
        ):
            return
        heal, smith = heals[0], smiths[0]
        hp = int(budget.health.hp or 0)
        max_hp = int(budget.health.max_hp or 1)
        percent = hp * 100 // max_hp
        threshold = round(float(params["heal_below_hp_ratio"]) * 100)
        if rest_rule_prefers_heal(hp, max_hp, params):
            favored, other = heal, smith
            favored_text = f"生命 {percent}%，低于 {threshold}%：休息更稳。"
            other_text = f"生命 {percent}%，低于 {threshold}%：先休息，锻造留到生命充足时。"
        else:
            favored, other = smith, heal
            if not any(
                item["code"] == "deck_upgrade_opportunity" and item["delta"] > 0
                for item in smith.factors
            ):
                smith.add(
                    "campfire_rest_rule_not_applied",
                    0.0,
                    f"生命 {percent}%，不低于 {threshold}%，但没有能变强的升级：按效果比较。",
                    "resource_efficiency",
                )
                return
            favored_text = f"生命 {percent}%，不低于 {threshold}%：锻造更划算。"
            other_text = f"生命 {percent}%，不低于 {threshold}%：休息收益有限。"
        # The rule decides the order: lift the favoured action just enough to
        # lead by the margin (an unweighted dimension, so preferences that
        # already weighted the effect factors cannot undo it).
        lead = max(
            0.0,
            float(other.score) - float(favored.score) + float(params["rule_margin"]),
        )
        favored.add("campfire_rest_rule", lead, favored_text, "resource_efficiency")
        other.add("campfire_rest_rule", 0.0, other_text, "resource_efficiency")


def _gap_token(error: Exception) -> str:
    return "_".join(str(error).lower().split())[:80]
