"""What the review agent may ask about one reviewed run.

Every answer is built from the engine's own output, so each claim the agent
writes can be checked against a tool result. The what-if tool only edits the
deck with things this run really had: cards in the deck at that moment, cards
offered earlier in the run, upgrades of cards in the deck.
"""

from __future__ import annotations

import copy
from typing import List, Mapping, Optional

from advisor.deck_fitness import DIMENSION_LABELS, DIMENSIONS
from advisor.golden_set import card_reward_request
from review.run_review import POLICY_LEANINGS, ReviewError, RunReview, _candidate_kinds, _option_rows, _tier

WHAT_IF_OPERATIONS = ("remove", "add", "upgrade")
MAX_WHAT_IF_CHANGES = 3


def decision_detail(review: RunReview, decision_id: str) -> dict:
    return review.decision(decision_id).record


def compare_options(review: RunReview, decision_id: str, option_a: str, option_b: str) -> dict:
    """Factor by factor, how option A's score differs from option B's."""

    record = review.decision(decision_id).record
    by_id = {o["id"]: o for o in record["options"]}
    for option in (option_a, option_b):
        if option not in by_id:
            raise ReviewError(f"{decision_id} has no option {option}; options: {', '.join(by_id)}")
    a, b = by_id[option_a], by_id[option_b]
    deltas_a = {f["code"]: f for f in a["factors"]}
    deltas_b = {f["code"]: f for f in b["factors"]}
    rows = []
    for code in sorted((set(deltas_a) | set(deltas_b)) - POLICY_LEANINGS):
        da = float((deltas_a.get(code) or {}).get("delta") or 0.0)
        db = float((deltas_b.get(code) or {}).get("delta") or 0.0)
        rows.append({
            "factor": code, "a": round(da, 2), "b": round(db, 2), "difference": round(da - db, 2),
            "a_message": (deltas_a.get(code) or {}).get("message"),
            "b_message": (deltas_b.get(code) or {}).get("message"),
        })
    rows.sort(key=lambda r: -abs(r["difference"]))
    return {
        "decision_id": decision_id,
        "a": {"id": a["id"], "name": a["name"], "deck_score": a["deck_score"], "fully_scored": a["fully_scored"]},
        "b": {"id": b["id"], "name": b["name"], "deck_score": b["deck_score"], "fully_scored": b["fully_scored"]},
        "deck_score_difference": None if a["deck_score"] is None or b["deck_score"] is None
        else round(a["deck_score"] - b["deck_score"], 2),
        "factors": rows,
        "note": "factor deltas are each option's change to the deck against taking nothing; "
                "policy leanings (pick margin, campfire rule) are not in deck_score",
    }


def deck_at(review: RunReview, decision_id: str) -> dict:
    decision = review.decision(decision_id)
    state = decision.view["state"]
    catalog = review.engine.catalog
    cards = [
        {"card": e["card"], "name": catalog.name("card", e["card"]), "upgrades": e["upgrades"], "count": e["count"]}
        for e in state["deck"]
    ]
    return {
        "decision_id": decision_id, "floor": decision.floor, "act": decision.act,
        "hp": state["hp"], "max_hp": state["max_hp"], "gold": state["gold"],
        "deck_size": sum(c["count"] for c in cards),
        "deck": cards,
        "relics": [{"relic": r, "name": catalog.name("relic", r)} for r in state["relics"]],
        "potions": [{"potion": p["potion"], "name": catalog.name("potion", p["potion"])} for p in state.get("potions") or ()],
    }


def card_facts(review: RunReview, card_id: str) -> dict:
    facts = review.engine.catalog.card(card_id)
    if facts is None:
        raise ReviewError(f"unknown card {card_id}")
    return facts


def relic_facts(review: RunReview, relic_id: str) -> dict:
    facts = review.engine.catalog.relic(relic_id)
    if facts is None:
        raise ReviewError(f"unknown relic {relic_id}")
    return facts


def _allowed_additions(review: RunReview, floor: int) -> List[str]:
    return sorted({card for seen_floor, card in review.offered if seen_floor < floor})


def what_if(review: RunReview, decision_id: str, changes: List[Mapping]) -> dict:
    """Re-score one card reward with the deck changed, and say whether the engine's pick moves."""

    decision = review.decision(decision_id)
    if decision.kind != "card_reward":
        raise ReviewError("what-if is available for card rewards only")
    if not changes or len(changes) > MAX_WHAT_IF_CHANGES:
        raise ReviewError(f"give 1 to {MAX_WHAT_IF_CHANGES} changes")
    view = copy.deepcopy(dict(decision.view))
    deck: List[dict] = view["state"]["deck"]
    allowed = _allowed_additions(review, decision.floor)
    applied = []
    for change in changes:
        operation, card = change.get("operation"), str(change.get("card") or "").upper()
        if operation not in WHAT_IF_OPERATIONS:
            raise ReviewError(f"operation must be one of {WHAT_IF_OPERATIONS}")
        if operation == "add":
            if card not in allowed:
                raise ReviewError(f"{card} was not offered before floor {decision.floor}; "
                                  f"cards that were: {', '.join(allowed) or 'none'}")
            entry = next((e for e in deck if e["card"] == card and e["upgrades"] == 0), None)
            if entry is None:
                template = copy.deepcopy(deck[0])
                template.update(card=card, count=0, upgrades=0, enchantment=None, enchantment_amount=None,
                                affliction=None, affliction_amount=None)
                deck.append(template)
                entry = template
            entry["count"] += 1
        else:
            if operation == "upgrade" and not review.engine.upgradable.get(card):
                raise ReviewError(f"{card} has no upgrade")
            entry = next((e for e in deck if e["card"] == card and (operation == "remove" or e["upgrades"] == 0)), None)
            if entry is None:
                raise ReviewError(f"no {'' if operation == 'remove' else 'unupgraded '}{card} in the deck at "
                                  f"{decision_id}")
            entry["count"] -= 1
            if operation == "upgrade":
                upgraded = next((e for e in deck if e["card"] == card and e["upgrades"] == 1), None)
                if upgraded is None:
                    upgraded = copy.deepcopy(entry)
                    upgraded.update(count=0, upgrades=1)
                    deck.append(upgraded)
                upgraded["count"] += 1
        applied.append({"operation": operation, "card": card, "name": review.engine.catalog.name("card", card)})
    view["state"]["deck"] = [e for e in deck if e["count"] > 0]
    request = card_reward_request(view)
    rec = review.engine.card_rewards.recommend(request)
    options = _option_rows(rec, review.engine, _candidate_kinds(request))
    before = decision.record
    tier, gap, _ = _tier(options, rec.recommended_candidate_id, before["chosen"], rec.confidence,
                         before["pick_margin"])
    return {
        "decision_id": decision_id,
        "changes": applied,
        "engine_top_before": before["engine_top"],
        "engine_top_after": rec.recommended_candidate_id,
        "pick_changed": rec.recommended_candidate_id != before["engine_top"],
        "agrees_with_player_after": rec.recommended_candidate_id == before["chosen"],
        "confidence_after": rec.confidence,
        "gap_after": gap,
        "options_after": [
            {k: o[k] for k in ("id", "name", "deck_score", "rank", "fully_scored")} for o in options
        ],
    }


def what_if_choices(review: RunReview, decision_id: str) -> dict:
    """What the what-if tool accepts at this decision."""

    decision = review.decision(decision_id)
    deck = decision.view["state"]["deck"]
    catalog = review.engine.catalog
    return {
        "remove": sorted({e["card"] for e in deck}),
        "upgrade": sorted({e["card"] for e in deck if e["upgrades"] == 0 and review.engine.upgradable.get(e["card"])}),
        "add": [{"card": c, "name": catalog.name("card", c)} for c in _allowed_additions(review, decision.floor)],
    }


def deck_trend(review: RunReview, dimension: Optional[str] = None) -> dict:
    """The deck's per-turn capability against what the coming fights need, at every card reward.

    The same numbers the factor messages show as "每回合 6.8 → ...（需求 11.8）" (the value before
    the pick), so a pattern over the run can be read without opening each decision."""

    dimensions = DIMENSIONS if dimension is None else (dimension,)
    if any(d not in DIMENSIONS for d in dimensions):
        raise ReviewError(f"dimension must be one of {', '.join(DIMENSIONS)}")
    policy = review.engine.deck_fitness
    rows = []
    for decision in review.decisions.values():
        if decision.kind != "card_reward":
            continue
        context, _ = policy.deck_context(card_reward_request(decision.view))
        if context is None:
            continue
        fit, _ = context.score(context.deck)
        values = {}
        for d in dimensions:
            have, need = fit.capabilities[d], context.needs.values[d]
            values[d] = {"have": round(have, 2 if d == "cycle" else 1), "need": round(need, 1),
                         "below_need": have < need}
        rows.append({"decision_id": decision.id, "floor": decision.floor, "act": decision.act,
                     "deck_size": len(context.deck), "values": values})
    return {
        "labels": {d: DIMENSION_LABELS[d] for d in dimensions},
        "decisions": rows,
        "note": "have: what the deck does per turn before that pick (cycle: extra cards drawn per turn); "
                "need: what the coming fights of that act need per turn",
    }
