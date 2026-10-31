"""Merchant policy v2 (TASK-027 ticket 06): every offer on the deck-fitness scale.

Each stocked, affordable offer is scored as 50 plus what it does to the deck
(the same engine, needs and lookahead over the act's coming rewards as a card
reward) minus the gold it costs, priced with ``resource_exchange``:

* a card: the deck with it (relic modifiers on added cards applied);
* a relic: ``RelicChoiceValuer`` (pickup changes, combat mechanics, gold / HP);
* card removal: the best of the cards the game lets the player remove
  (``deck_edit_targets``, verified by the Mod), removed;
* a potion: no reviewed potion facts yet, so it stays unscored (shown, never
  ranked) without blocking the other offers.

Leaving keeps the gold (50). Buying one offer changes the shop's state and the
Mod sends it again, so the next purchase is scored on the new deck and gold.
No community prior is used.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from advisor.choice_effects import AssessmentDraft, actual_gold_cost, factor, finalize_recommendation
from advisor.deck_edit_value import verified_targets
from advisor.deck_fitness import PARAMETERS_KEY as FITNESS_PARAMETERS_KEY
from advisor.deck_fitness import DeckFitnessPolicy
from advisor.deck_mechanics import CardProfile, adjust_added_card
from advisor.decision_core import DecisionRequest, Recommendation
from advisor.relic_choice import RelicChoiceValuer
from advisor.resource_exchange import CAMPFIRE_KEY
from advisor.resource_exchange import PARAMETERS_KEY as EXCHANGE_KEY
from advisor.resource_exchange import gold_points, parameter_gaps, resource_points
from advisor.scoring_preferences import DEFAULT_SCORING_PREFERENCES, ScoringPreferences
from advisor.versions import MERCHANT_POLICY_VERSION
from storage.relational import RelationalRepository


MERCHANT_CHOICE = "merchant_choice"
_OFFER_KINDS = frozenset({"card", "relic", "potion", "card_removal"})


class MerchantPolicy:
    decision_type = MERCHANT_CHOICE

    def __init__(
        self, repository: RelationalRepository,
        preferences: ScoringPreferences = DEFAULT_SCORING_PREFERENCES,
        fitness_policy: DeckFitnessPolicy | None = None,
    ):
        self.repository = repository
        self.preferences = preferences
        self.fitness_policy = fitness_policy or DeckFitnessPolicy(repository, preferences)
        self.valuer = RelicChoiceValuer(self.fitness_policy)

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("merchant policy received another decision type")
        drafts = [AssessmentDraft(candidate, preferences=self.preferences) for candidate in request.candidates]
        # The candidate's own shape first: a malformed offer reports its own gap.
        offers: List[tuple] = []
        for draft in drafts:
            candidate = draft.candidate
            if not candidate.eligible:
                continue
            kind = _kind(draft)
            if kind == "leave":
                draft.factors.append(factor("merchant_leave", 0.0, "离开商店，保留金币。", "resource_efficiency"))
                continue
            if kind != "merchant_offer":
                draft.gap(f"merchant:unknown_candidate_kind:{kind or 'missing'}")
                continue
            offer = str(candidate.payload.get("offer_kind") or "").strip()
            if offer not in _OFFER_KINDS:
                draft.gap(f"merchant:unknown_offer_kind:{offer or 'missing'}")
                continue
            if candidate.payload.get("is_stocked") is not True:
                draft.gap("merchant:eligible_offer_not_stocked")
                continue
            if offer != "card_removal" and not _entity(draft):
                draft.gap("merchant:entity_id_missing")
                continue
            try:
                price = actual_gold_cost(candidate.payload)
            except ValueError as exc:
                draft.gap(f"merchant:costs_invalid:{_gap_token(exc)}")
                continue
            if offer == "card_removal":
                # The game's own removable set, checked before anything is scored.
                targets, target_gaps = verified_targets(request.world, "remove", candidate.payload.get("deck_edit_targets"))
                if target_gaps:
                    for gap in target_gaps:
                        draft.gap(gap)
                    continue
                offers.append((draft, offer, price, targets))
                continue
            if offer == "potion":
                draft.unscored = True
                draft.gap(f"merchant:potion_unmodeled:{_entity(draft)}", blocking=False)
                draft.factors.append(factor(
                    "insufficient_data", 0.0, "药水还没有经程序集核对的效果事实，暂不评分。", "data_completeness",
                ))
                continue
            offers.append((draft, offer, price, None))
        context, context_gaps = self.fitness_policy.deck_context(request) if offers else (None, ())
        exchange = self.repository.mechanic_constant(EXCHANGE_KEY)
        campfire = self.repository.mechanic_constant(CAMPFIRE_KEY)
        setup_gaps = (
            *(() if context is not None else (tuple(context_gaps) or ("merchant:situation_unscored",))),
            *parameter_gaps(exchange, campfire),
        )
        if offers and setup_gaps:
            for draft, _, _, _ in offers:
                for gap in setup_gaps:
                    draft.gap(gap)
            offers = []
        if offers:
            scoring = _Scoring(self.fitness_policy, request, context)
            relic_ids = [(_entity(d), {}) for d, offer, _, _ in offers if offer == "relic"]
            relic_values = self.valuer.evaluate(request, relic_ids)[1] if relic_ids else {}
            held = float(request.world.gold) if request.world.gold is not None else None
        partly: List[str] = []
        for draft, offer, price, targets in offers:
            candidate = draft.candidate
            if offer == "card":
                value, note, gaps = scoring.card(_entity(draft), candidate.payload.get("upgrades"))
            elif offer == "relic":
                value, note, gaps = scoring.relic(relic_values.get(_entity(draft)), draft, exchange, campfire)
            else:
                value, note, gaps = scoring.removal(targets)
            if value is None:
                for gap in gaps:
                    draft.gap(gap)
                continue
            draft.factors.append(factor("merchant:deck", value, note, "long_term_growth"))
            if held is None:
                draft.gap("resource:gold_missing")
                continue
            draft.factors.append(factor(
                "merchant:price", gold_points(held, -price, request.world, exchange),
                f"花 {price} 金币（现有 {held:g}）：这些金币在之后的商店能买到的价值；金币越多每枚越不值钱",
                "resource_efficiency",
            ))
            if gaps:
                partly.append(candidate.candidate_id)
                for gap in gaps:
                    draft.gap(gap, blocking=False)
                draft.factors.append(factor(
                    "unmodeled_mechanic", 0.0, "部分效果尚未计入评分（见数据缺口），分数只反映已计入的部分。",
                    "data_completeness",
                ))
        for draft, _, _, _ in offers:
            if draft.score is not None:
                for gap in context_gaps:
                    draft.gap(gap, blocking=False)
        recommendation = finalize_recommendation(request, drafts, policy_version=MERCHANT_POLICY_VERSION)
        return replace(recommendation, confidence=self._confidence(recommendation, partly))

    def _confidence(self, recommendation: Recommendation, partly: Sequence[str]) -> str:
        """As for relic choices: "low" on a near tie or when an option within
        reach (the recommended one included) is only partly scored."""

        if recommendation.recommended_candidate_id is None:
            return "low"
        params = self.repository.mechanic_constant(FITNESS_PARAMETERS_KEY) or {}
        allowance, near_tie = params.get("confidence_gap_allowance"), params.get("near_tie_points")
        if not isinstance(allowance, (int, float)) or not isinstance(near_tie, (int, float)):
            return "low"
        scored = sorted((c for c in recommendation.candidates if c.score is not None), key=lambda c: -c.score)
        best = scored[0]
        if len(scored) > 1 and best.score - scored[1].score < near_tie:
            return "low"
        if any(c.candidate_id in partly and best.score - c.score <= allowance for c in scored):
            return "low"
        return "medium"


class _Scoring:
    """Deck values for one shop screen, all against the same lookahead base."""

    def __init__(self, policy: DeckFitnessPolicy, request: DecisionRequest, context):
        self.policy = policy
        self.request = request
        self.context = context
        self.cache: Dict[tuple, float] = {}
        self.base = self.ahead(context.deck)

    def score(self, cards: Sequence[Optional[CardProfile]]) -> float:
        key = tuple(sorted(map(id, cards)))
        if key not in self.cache:
            self.cache[key] = self.context.score(cards)[0].raw
        return self.cache[key]

    def ahead(self, cards: Sequence[Optional[CardProfile]]) -> float:
        values, _ = self.policy._lookahead(
            self.request, {"deck": list(cards)}, self.score, self.context.needs, self.context.params,
        )
        return values.get("deck", self.score(cards))

    def card(self, card_id: str, upgrades) -> tuple:
        if type(upgrades) is not int or not 0 <= upgrades <= 9:
            return None, "", ("merchant:card_upgrades_unknown",)
        profile, profile_gaps = self.policy._card(card_id.upper(), upgrades)
        if profile is None:
            return None, "", (f"unknown_card:{card_id}",)
        profile = adjust_added_card(profile, self.context.relics, self.policy.repository)
        value = self.ahead([*self.context.deck, profile]) - self.base
        return value, f"加入 {self.policy._card_name(card_id)} 后的牌组适合度（计入本幕后续选牌）", tuple(profile_gaps)

    def relic(self, value, draft: AssessmentDraft, exchange, campfire) -> tuple:
        if value is None or value.raw is None:
            return None, "", tuple(value.gaps if value else ("merchant:relic_unscored",))
        name = self.policy._relic_name(value.relic_id)
        note = f"拿 {name} 后的牌组适合度（计入本幕后续选牌）" + ("：" + "，".join(value.notes) if value.notes else "")
        total = value.raw - self.base
        gaps = list(value.gaps)
        if value.resources:
            points, reasons, resource_gaps = resource_points(value.resources, self.request.world, exchange, campfire)
            if resource_gaps:
                return None, "", resource_gaps
            for reason in reasons:
                draft.factors.append(factor(**reason))
        return total, note, tuple(gaps)

    def removal(self, targets) -> tuple:
        removable = {(t["card"].upper(), int(t["upgrades"])) for t in targets}
        best_value, best_cards, best_name = None, None, ""
        seen = set()
        for i, card in enumerate(self.context.deck):
            if card is None or id(card) in seen or (card.card_id, 1 if card.upgraded else 0) not in removable:
                continue
            seen.add(id(card))
            cards = [*self.context.deck[:i], *self.context.deck[i + 1:]]
            value = self.score(cards)
            if best_value is None or value > best_value:
                best_value, best_cards, best_name = value, cards, self.policy._card_name(card.card_id)
        if best_cards is None:
            return None, "", ("merchant:no_scorable_removal_target",)
        return self.ahead(best_cards) - self.base, f"移除 {best_name}（可移除的牌中估值最好的一张）", ()


def _kind(draft: AssessmentDraft) -> str:
    return str(draft.candidate.payload.get("candidate_kind") or "").strip()


def _entity(draft: AssessmentDraft) -> str:
    return str(draft.candidate.payload.get("entity_id") or "").strip().upper()


def _gap_token(error: Exception) -> str:
    return "_".join(str(error).lower().split())[:80]
