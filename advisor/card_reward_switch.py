"""The shipped card-reward policy: deck fitness (v2), with the previous
version kept as the fallback.

The fallback answers when the new kernel cannot score a decision (its
recommendation is ``uncertain``) or fails. Its output keeps its own policy
version, so a reader can always tell which version produced an advice.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from advisor.decision_core import CARD_REWARD, DecisionPolicy, DecisionRequest, Recommendation
from advisor.deck_fitness import DeckFitnessPolicy

LOGGER = logging.getLogger(__name__)
FALLBACK_GAP = "deck_fitness_unavailable"


class SwitchedCardRewardPolicy:
    decision_type = CARD_REWARD

    def __init__(self, primary: DecisionPolicy, fallback: DecisionPolicy):
        self.primary = primary
        self.fallback = fallback

    def warm_up(self) -> None:
        warm_up = getattr(self.primary, "warm_up", None)
        if callable(warm_up):
            warm_up()

    def recommend(self, request: DecisionRequest) -> Recommendation:
        reasons: tuple = ()
        try:
            result = self.primary.recommend(request)
        except Exception:  # the fallback must answer whatever went wrong
            LOGGER.exception("Deck-fitness policy failed; using the fallback policy.")
            reasons = ("deck_fitness_error",)
        else:
            if result.status != "uncertain":
                return result
            reasons = tuple(result.data_gaps)
        try:
            legacy = self.fallback.recommend(request)
        except Exception:
            # Neither version can answer: say so instead of failing the event.
            LOGGER.exception("The fallback card-reward policy failed as well.")
            return DeckFitnessPolicy._uncertain(
                request, tuple(dict.fromkeys((*reasons, FALLBACK_GAP, "card_reward_fallback_error"))),
            )
        return replace(
            legacy,
            data_gaps=tuple(dict.fromkeys((*legacy.data_gaps, FALLBACK_GAP, *reasons))),
        )
