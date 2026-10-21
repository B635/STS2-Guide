"""The shipped card-reward policy: deck fitness (v2) alone.

There is no fallback to another scoring (user decision 2026-10-22: no
community-based scores). When the kernel cannot score a decision it answers
``uncertain`` itself; when it fails, this guard logs the error and answers
``uncertain`` with ``deck_fitness_error`` instead of failing the event.
"""

from __future__ import annotations

import logging

from advisor.decision_core import CARD_REWARD, DecisionPolicy, DecisionRequest, Recommendation
from advisor.deck_fitness import DeckFitnessPolicy

LOGGER = logging.getLogger(__name__)
ERROR_GAP = "deck_fitness_error"


class GuardedCardRewardPolicy:
    decision_type = CARD_REWARD

    def __init__(self, primary: DecisionPolicy):
        self.primary = primary

    def warm_up(self) -> None:
        warm_up = getattr(self.primary, "warm_up", None)
        if callable(warm_up):
            warm_up()

    def recommend(self, request: DecisionRequest) -> Recommendation:
        try:
            return self.primary.recommend(request)
        except Exception:  # a failure must not take the whole event down
            LOGGER.exception("Deck-fitness policy failed; the decision is shown as uncertain.")
            return DeckFitnessPolicy._uncertain(request, (ERROR_GAP,))
