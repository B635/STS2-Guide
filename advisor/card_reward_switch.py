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

LOGGER = logging.getLogger(__name__)
FALLBACK_GAP = "deck_fitness_unavailable"


class SwitchedCardRewardPolicy:
    decision_type = CARD_REWARD

    def __init__(self, primary: DecisionPolicy, fallback: DecisionPolicy):
        self.primary = primary
        self.fallback = fallback

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
        legacy = self.fallback.recommend(request)
        return replace(
            legacy,
            data_gaps=tuple(dict.fromkeys((*legacy.data_gaps, FALLBACK_GAP, *reasons))),
        )
