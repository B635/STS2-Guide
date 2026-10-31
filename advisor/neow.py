"""Relic choices offered by Neow and the other ancients (TASK-027).

Every option of an ancient (``AncientEventModel.RelicOption``, 0.111.0) is a
relic; the Mod sends its ID as the candidate's ``entity_id``. Each option is
scored by ``RelicChoiceValuer`` (the deck after its pickup changes, with its
combat mechanics, on the card-reward fitness scale) plus its gold / HP / max
HP changes converted by ``resource_exchange``. The score is 50 plus those
points; the reasons list each part. An option without a relic ID, or whose
relic has no reviewed fact, cannot be scored and the decision is uncertain.
"""
from __future__ import annotations

from dataclasses import replace

from advisor.choice_effects import AssessmentDraft, factor, finalize_recommendation
from advisor.deck_fitness import PARAMETERS_KEY as FITNESS_PARAMETERS_KEY
from advisor.deck_fitness import DeckFitnessPolicy
from advisor.decision_core import DecisionRequest, Recommendation
from advisor.relic_choice import RelicChoiceValuer
from advisor.resource_exchange import CAMPFIRE_KEY
from advisor.resource_exchange import PARAMETERS_KEY as EXCHANGE_KEY
from advisor.resource_exchange import parameter_gaps, resource_points
from advisor.scoring_preferences import DEFAULT_SCORING_PREFERENCES, ScoringPreferences
from advisor.versions import NEOW_POLICY_VERSION
from storage.relational import RelationalRepository


NEOW_BLESSING = "neow_blessing"
CHARACTERS = frozenset({"IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"})


class NeowPolicy:
    decision_type = NEOW_BLESSING

    def __init__(self, repository: RelationalRepository,
                 preferences: ScoringPreferences = DEFAULT_SCORING_PREFERENCES,
                 fitness_policy: DeckFitnessPolicy | None = None):
        self.repository = repository
        self.preferences = preferences
        self.fitness_policy = fitness_policy or DeckFitnessPolicy(repository, preferences)
        self.valuer = RelicChoiceValuer(self.fitness_policy)

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("Neow policy received another decision type")
        drafts = [AssessmentDraft(candidate, preferences=self.preferences) for candidate in request.candidates]
        options = []
        relic_by_candidate = {}
        for draft in drafts:
            candidate = draft.candidate
            if not candidate.eligible:
                continue
            relic = str(candidate.payload.get("entity_id") or "").strip().upper()
            if candidate.payload.get("candidate_kind") != "neow_blessing" or not relic:
                draft.gap("neow:relic_id_missing")
                continue
            relic_by_candidate[candidate.candidate_id] = relic
            options.append((relic, _option_params(relic, candidate.payload)))
        baseline, values, context_gaps = (
            self.valuer.evaluate(request, options) if options else (None, {}, ())
        )
        exchange = self.repository.mechanic_constant(EXCHANGE_KEY)
        campfire = self.repository.mechanic_constant(CAMPFIRE_KEY)
        exchange_gaps = parameter_gaps(exchange, campfire)
        partly = []
        for draft in drafts:
            relic = relic_by_candidate.get(draft.candidate.candidate_id)
            if relic is None:
                continue
            value = values.get(relic)
            if baseline is None or value is None or value.raw is None:
                for gap in (*(value.gaps if value else ()), *context_gaps) or ("neow:situation_unscored",):
                    draft.gap(gap)
                continue
            name = self.fitness_policy._relic_name(relic)
            # The deck part is on the card-reward scale already: not re-weighted by preferences.
            draft.factors.append(factor(
                "relic_choice:deck",
                value.raw - baseline,
                f"拿 {name} 后的牌组适合度（计入本幕后续选牌）"
                + ("：" + "，".join(value.notes) if value.notes else ""),
                "long_term_growth",
            ))
            if value.resources:
                if exchange_gaps:
                    for gap in exchange_gaps:
                        draft.gap(gap)
                    continue
                points, reasons, gaps = resource_points(value.resources, request.world, exchange, campfire)
                for reason in reasons:
                    draft.factors.append(factor(**reason))
                for gap in gaps:
                    draft.gap(gap)
            if value.gaps:
                partly.append(draft.candidate.candidate_id)
                for gap in value.gaps:
                    draft.gap(gap, blocking=False)    # a lower bound, disclosed
                draft.factors.append(factor(
                    "unmodeled_mechanic", 0.0,
                    f"{name}的部分效果尚未计入评分（见数据缺口），分数只反映已计入的部分。",
                    "data_completeness",
                ))
        for gap in context_gaps:
            for draft in drafts:
                if draft.candidate.candidate_id in relic_by_candidate and draft.score is not None:
                    draft.gap(gap, blocking=False)
        recommendation = finalize_recommendation(
            request, drafts, policy_version=NEOW_POLICY_VERSION,
        )
        return replace(recommendation, confidence=self._confidence(recommendation, partly))

    def _confidence(self, recommendation: Recommendation, partly: list) -> str:
        """"low" when the call could change: a near tie, or an option within
        reach that is only partly scored. Unlike a card, a relic's unscored
        part may cost (a between-fight drawback), so the recommended option
        being partly scored counts too."""

        if recommendation.recommended_candidate_id is None:
            return "low"
        params = self.repository.mechanic_constant(FITNESS_PARAMETERS_KEY) or {}
        allowance = params.get("confidence_gap_allowance")
        near_tie = params.get("near_tie_points")
        if not isinstance(allowance, (int, float)) or not isinstance(near_tie, (int, float)):
            return "low"
        scored = sorted(
            (c for c in recommendation.candidates if c.score is not None),
            key=lambda c: -c.score,
        )
        best = scored[0]
        if len(scored) > 1 and best.score - scored[1].score < near_tie:
            return "low"
        if any(c.candidate_id in partly and best.score - c.score <= allowance for c in scored):
            return "low"
        return "medium"


def _option_params(relic: str, payload) -> dict:
    """What the option fixed: Orobas's Sea Glass names a character in its option key."""

    if relic == "SEA_GLASS":
        key = str(payload.get("option_id") or "").rsplit(".", 1)[-1].upper()
        if key in CHARACTERS:
            return {"character": key}
    return {}
