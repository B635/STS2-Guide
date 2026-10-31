"""MCP server (stdio) exposing the review engine to the coach agent.

    python -m review.mcp_server

``review_run`` and ``list_decisions`` are for the coach's deterministic
orchestrator; the other tools are the ones the agent may call while it looks
into one flagged decision. Every tool answers with the engine's own numbers
and the game's own (Chinese) names.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Literal

from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from review import tools  # noqa: E402
from review.engine import Engine  # noqa: E402
from review.run_review import ReviewError, RunReview, review_file  # noqa: E402

mcp = FastMCP(
    "sts2-review",
    instructions="Slay the Spire 2 post-run review: the deterministic deck-fitness engine's scores for one run.",
)
_engine: Engine | None = None
_reviews: Dict[str, RunReview] = {}


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def _review(run_id: str) -> RunReview:
    review = _reviews.get(run_id)
    if review is None:
        raise ToolError(f"run {run_id} is not loaded; call review_run first")
    return review


def _call(function, *args):
    try:
        return function(*args)
    except ReviewError as exc:
        raise ToolError(str(exc)) from None


@mcp.tool()
def review_run(path: str) -> dict:
    """Load one finished single-player .run file and score it. Returns the run summary,
    including the strict flags (tier1, largest gap first) and the other disagreements (tier2)."""
    review = _call(review_file, _get_engine(), Path(path))
    _reviews[review.run_key] = review
    return review.summary()


@mcp.tool()
def list_decisions(run_id: str) -> dict:
    """Every scored decision of the run in floor order, plus the ones that could not be scored and why."""
    review = _review(run_id)
    rows = []
    for d in review.decisions.values():
        r = d.record
        names = {o["id"]: o["name"] for o in r["options"]}
        rows.append({
            "id": r["id"], "kind": r["kind"], "floor": r["floor"], "act": r["act"],
            "chosen": r["chosen"], "chosen_name": names.get(r["chosen"]),
            "engine_top": r["engine_top"], "engine_top_name": names.get(r["engine_top"]),
            "confidence": r["confidence"], "tier": r["tier"], "gap": r["gap"],
            "not_strict_because": r["not_strict_because"],
        })
    return {"decisions": rows, "skipped": review.skipped}


@mcp.tool()
def get_decision(run_id: str, decision_id: str) -> dict:
    """One decision in full: the player's situation (HP, gold), every option with its score, deck_score
    (score without policy leanings), rank, whether it is fully scored, and the engine's factors with messages."""
    return _call(tools.decision_detail, _review(run_id), decision_id)


@mcp.tool()
def compare_options(run_id: str, decision_id: str, option_a: str, option_b: str) -> dict:
    """Why option A scores differently from option B: each factor's delta for both, largest difference first."""
    return _call(tools.compare_options, _review(run_id), decision_id, option_a, option_b)


@mcp.tool()
def get_deck(run_id: str, decision_id: str) -> dict:
    """The deck, relics, potions, HP and gold at the moment of this decision."""
    return _call(tools.deck_at, _review(run_id), decision_id)


@mcp.tool()
def get_card(run_id: str, card_id: str) -> dict:
    """A card's facts from the game: name, cost, type, rarity, description, upgrade."""
    return _call(tools.card_facts, _review(run_id), card_id)


@mcp.tool()
def get_relic(run_id: str, relic_id: str) -> dict:
    """A relic's facts from the game: name, rarity, description."""
    return _call(tools.relic_facts, _review(run_id), relic_id)


@mcp.tool()
def what_if_choices(run_id: str, decision_id: str) -> dict:
    """Which deck changes what_if accepts at this card reward: cards to remove or upgrade (in the deck then),
    and cards to add (offered earlier in this run)."""
    return _call(tools.what_if_choices, _review(run_id), decision_id)


class DeckChange(BaseModel):
    operation: Literal["remove", "add", "upgrade"]
    card: str = Field(description="card ID, e.g. STRIKE_NECROBINDER")


@mcp.tool()
def what_if(run_id: str, decision_id: str, changes: List[DeckChange]) -> dict:
    """Re-score this card reward with 1-3 deck changes and report whether the engine's pick moves and whether it
    would then agree with the player. Only changes this run really allowed are accepted (see what_if_choices)."""
    return _call(tools.what_if, _review(run_id), decision_id, [c.model_dump() for c in changes])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
