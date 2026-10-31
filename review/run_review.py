"""Re-score one finished run and find the decisions worth a second look.

Pipeline (deterministic; no LLM): load the ``.run`` file, replay it into the
history views the importer builds (``advisor/run_history.py``), score every
card reward, rest site and shop visit with the shipped policies, then mark
each decision:

- tier 1 ("值得想一想"): the engine's pick differs from the player's, both are
  fully scored, the engine's confidence is not low, and the gap is at least
  ``STRICT_GAP`` points or more than the act's pick margin;
- tier 2: any other disagreement (appendix only);
- tier 0: the engine agrees.

Gaps compare the decks' own scores: the hand-set lifts that are policy
leanings rather than deck value (skip's pick margin, the campfire HP rule)
are taken out first. Shop visits are compared as buy or leave only.
"""

from __future__ import annotations

import collections
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from advisor.deck_fitness import _pick_margin
from advisor.golden_set import card_reward_request
from advisor.history_requests import deck_counts, rest_request, shop_request, starting_decks, unscorable_reason
from advisor.run_history import RunHistoryError, bare_id, load_run, replay_run
from review.engine import KNOWLEDGE_PATH, Engine

STRICT_GAP = 2.0
POLICY_LEANINGS = frozenset({"deck_fitness:pick_margin", "campfire_rest_rule"})
SHOP_CARD_SLOTS = 7
EXTRA_REST_OPTION_RELICS = frozenset({"MEAT_CLEAVER", "PUMPKIN_CANDLE", "SHOVEL", "MINIATURE_TENT"})


class ReviewError(ValueError):
    """The run cannot be reviewed (multiplayer, abandoned, unreadable)."""


@dataclass
class Decision:
    id: str
    kind: str
    floor: int
    act: int
    view: Mapping = field(repr=False)
    record: Dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict:
        return self.record


def _option_rows(rec, engine: Engine, kinds: Mapping[str, str]) -> List[dict]:
    rows = []
    for c in sorted(rec.candidates, key=lambda c: (c.rank is None, c.rank or 0, c.display_index)):
        kind, entity = kinds.get(c.candidate_id, ("other", None))
        leaning = sum(float(f.get("delta") or 0.0) for f in c.factors if f.get("code") in POLICY_LEANINGS)
        rows.append({
            "id": c.candidate_id,
            "kind": kind,
            "entity_id": entity,
            "name": engine.catalog.name(kind, entity) if kind in ("card", "relic", "potion") else _action_name(c.candidate_id),
            "score": c.score,
            "deck_score": None if c.score is None else round(c.score - leaning, 2),
            "rank": c.rank,
            "eligible": c.eligible,
            "fully_scored": not c.data_gaps,
            "gaps": list(c.data_gaps),
            "factors": [
                {"code": f.get("code"), "delta": f.get("delta"), "message": f.get("message")} for f in c.factors
            ],
            "dimensions": dict(c.dimensions or {}),
        })
    return rows


def _action_name(candidate_id: str) -> str:
    return {"skip": "跳过", "leave": "离开商店", "removal": "删牌", "rest:HEAL": "休息", "rest:SMITH": "锻造"}.get(
        candidate_id, candidate_id
    )


def _candidate_kinds(request) -> Dict[str, tuple]:
    kinds = {}
    for c in request.candidates:
        payload = c.payload or {}
        kind = payload.get("offer_kind") or payload.get("candidate_kind") or "other"
        entity = payload.get("entity_id") or payload.get("card")
        if c.candidate_id == "skip":
            kind, entity = "skip", None
        elif kind in ("card", "relic", "potion"):
            pass
        elif ":" in c.candidate_id and c.candidate_id.split(":")[0].isdigit():
            kind, entity = "card", c.candidate_id.split(":", 1)[1]
        kinds[c.candidate_id] = (kind, entity)
    return kinds


def _tier(options: List[dict], top: Optional[str], chosen: Optional[str], confidence: str,
          margin: float) -> tuple:
    """(tier, gap, reasons not tier 1)."""

    if top is None or chosen is None:
        return None, None, ["engine_no_pick" if top is None else "choice_unknown"]
    if top == chosen:
        return 0, 0.0, []
    by_id = {o["id"]: o for o in options}
    a, b = by_id.get(top), by_id.get(chosen)
    if a is None or b is None or a["deck_score"] is None or b["deck_score"] is None:
        return 2, None, ["choice_not_scored"]
    gap = round(a["deck_score"] - b["deck_score"], 2)
    reasons = []
    if not (a["fully_scored"] and b["fully_scored"]):
        reasons.append("partly_scored")
    if confidence == "low":
        reasons.append("low_confidence")
    if not (gap >= STRICT_GAP or gap > margin):
        reasons.append("small_gap")
    return (2 if reasons else 1), gap, reasons


class RunReview:
    """One run, scored. Holds the views so the tools can look deeper."""

    def __init__(self, engine: Engine, run: Mapping, run_key: str):
        self.engine = engine
        self.run = run
        self.run_key = run_key
        self.decisions: Dict[str, Decision] = {}
        self.skipped: List[dict] = []
        self.offered: List[tuple] = []          # (floor, card_id) seen in card rewards and shops
        self._build()

    # ---------------------------------------------------------------- building
    def _build(self) -> None:
        players = self.run.get("players") or ()
        if len(players) != 1:
            raise ReviewError("only single-player runs can be reviewed")
        if self.run.get("was_abandoned"):
            raise ReviewError("abandoned runs are not reviewed")
        character = bare_id(players[0]["character"])
        decks = starting_decks(KNOWLEDGE_PATH)
        if character not in decks:
            raise ReviewError(f"unknown character {character}")
        try:
            replay = replay_run(self.run, decks[character], self.run_key)
        except (AttributeError, KeyError, TypeError, ValueError, RunHistoryError) as exc:
            raise ReviewError(f"the run cannot be replayed ({type(exc).__name__}: {exc})") from exc
        if replay["problems"]:
            raise ReviewError(f"the run cannot be replayed exactly: {replay['problems'][0]}")
        self.character = character
        self.warnings = list(replay["warnings"])

        decks_at: Dict[int, list] = {}
        for row in replay["decisions"]:
            decks_at[row["floor"]] = row["scoring_view"]["state"]["deck"]
        for s in replay["situations"]:
            decks_at[s["source"]["floor"]] = s["scoring_view"]["state"]["deck"]
        for f in replay["fights"]:
            decks_at[f["floor"]] = f["deck"]

        rows = [("card_reward", s["source"]["floor"], s) for s in replay["situations"]]
        rows += [(r["kind"], r["floor"], r) for r in replay["decisions"] if r["kind"] in ("rest", "shop")]
        rows.sort(key=lambda r: (r[1], r[0]))
        for kind, floor, row in rows:
            view = row["scoring_view"]
            ident = self._new_id(floor, kind)
            if kind == "card_reward":
                self.offered += [(floor, c["entity_id"]) for c in view["candidates"] if c.get("entity_id")]
            elif kind == "shop":
                self.offered += [(floor, e["card"]) for e in row["offers"]["cards"]]
            reason = unscorable_reason(view, self.engine.known_cards, self.engine.known_encounters)
            if reason is not None:
                self.skipped.append({"id": ident, "kind": kind, "floor": floor, "reason": reason})
                continue
            act = int(view["state"]["act"])
            decision = Decision(ident, kind, floor, act, view)
            if kind == "card_reward":
                self._score_card_reward(decision, row)
            elif kind == "rest":
                if not self._score_rest(decision, row):
                    continue
            else:
                self._score_shop(decision, row, decks_at.get(floor + 1))
            self.decisions[ident] = decision

    def _new_id(self, floor: int, kind: str) -> str:
        short = {"card_reward": "card", "rest": "rest", "shop": "shop"}[kind]
        ident = f"F{floor:02d}-{short}"
        n = 2
        while ident in self.decisions or any(s["id"] == ident for s in self.skipped):
            ident = f"F{floor:02d}-{short}{n}"
            n += 1
        return ident

    def _base(self, decision: Decision, rec, request) -> dict:
        state = decision.view["state"]
        return {
            "id": decision.id, "kind": decision.kind, "floor": decision.floor, "act": decision.act,
            "hp": state["hp"], "max_hp": state["max_hp"], "gold": state["gold"],
            "engine_top": rec.recommended_candidate_id, "confidence": rec.confidence, "policy_status": rec.status,
            "pick_margin": _pick_margin(self.engine.params, decision.act),
            "options": _option_rows(rec, self.engine, _candidate_kinds(request)),
        }

    def _finish(self, record: dict, chosen: Optional[str]) -> None:
        record["chosen"] = chosen
        tier, gap, reasons = _tier(record["options"], record["engine_top"], chosen, record["confidence"],
                                   record["pick_margin"])
        record.update(tier=tier, gap=gap, not_strict_because=reasons)

    def _score_card_reward(self, decision: Decision, situation: Mapping) -> None:
        request = card_reward_request(decision.view)
        rec = self.engine.card_rewards.recommend(request)
        record = self._base(decision, rec, request)
        decision.record = record
        self._finish(record, situation["best"][0] if situation.get("best") else None)

    def _score_rest(self, decision: Decision, row: Mapping) -> bool:
        chosen = "rest:" + str(row.get("chosen"))
        if chosen not in ("rest:HEAL", "rest:SMITH"):
            self.skipped.append({"id": decision.id, "kind": "rest", "floor": decision.floor,
                                 "reason": f"rest_option_not_modeled:{row.get('chosen')}"})
            return False
        request = rest_request(dict(row), self.engine.upgradable)
        rec = self.engine.campfire.recommend(request)
        record = self._base(decision, rec, request)
        extra = sorted(EXTRA_REST_OPTION_RELICS & set(decision.view["state"]["relics"]))
        if extra:
            record["note"] = "extra_rest_options:" + ",".join(extra)
        if row.get("upgraded"):
            record["upgraded"] = {"card": row["upgraded"], "name": self.engine.catalog.name("card", row["upgraded"])}
        decision.record = record
        self._finish(record, chosen)
        return True

    def _score_shop(self, decision: Decision, row: Mapping, deck_after: Optional[list]) -> None:
        # The record lists only the cards left on sale, so the cards bought are
        # read from the deck at the next recorded point and put back as offers.
        before = deck_counts(row["scoring_view"]["state"]["deck"])
        removed = collections.Counter((e["card"], int(e["upgrades"] or 0)) for e in row["bought"]["removed"])
        unknown = 0
        if deck_after is not None:
            bought_cards = sorted((deck_counts(deck_after) - (before - removed)).elements())
        else:
            bought_cards = [(c, 0) for c in row["bought"]["colorless"]]
            unknown = max(0, SHOP_CARD_SLOTS - len(row["offers"]["cards"]) - len(bought_cards))
        fixed = copy.deepcopy(dict(row))
        fixed["scoring_view"]["state"]["gold"] = row["gold_before"]
        first = len(fixed["offers"]["cards"])
        for card, upgrades in bought_cards:
            fixed["offers"]["cards"].append({"card": card, "upgrades": upgrades, "enchantment": None,
                                             "enchantment_amount": None})
        request = shop_request(fixed, self.engine.rarity)
        rec = self.engine.merchant.recommend(request)
        record = self._base(decision, rec, request)
        record["gold"] = row["gold_before"]
        ids = {c.candidate_id for c in request.candidates}
        bought = [f"card:{first + i}:{card}" for i, (card, _) in enumerate(bought_cards)]
        unmatched = [b for b in bought if b not in ids]
        bought = [b for b in bought if b in ids]
        names = set(row["bought"]["relics"]) | set(row["bought"]["potions"])
        bought += [c.candidate_id for c in request.candidates
                   if c.candidate_id.startswith(("relic:", "potion:")) and c.label in names]
        if row["bought"]["removed"]:
            bought.append("removal")
        bought_any = bool(bought or unmatched or unknown or names or row["bought"]["colorless"])
        record["bought"] = sorted(set(bought))
        record["bought_unscored"] = len(unmatched) + unknown
        decision.record = record
        # Buy or leave: compare the engine's pick with the player's best-scored purchase, or with leaving.
        top = record["engine_top"]
        if not bought_any:
            chosen = "leave"
        elif top == "leave" or top is None:
            scored = [o for o in record["options"] if o["id"] in record["bought"]]
            chosen = max(scored, key=lambda o: o["deck_score"] or float("-inf"))["id"] if scored else None
            if record["bought_unscored"] and chosen is not None:
                chosen = None      # a purchase the engine could not price: buy or leave unknown
        else:
            chosen = top           # both bought something: buy-or-leave agrees
        self._finish(record, chosen)

    # ---------------------------------------------------------------- views
    def summary(self) -> dict:
        records = [d.record for d in self.decisions.values()]
        judged = [r for r in records if r["tier"] is not None]
        agree = sum(r["tier"] == 0 for r in judged)
        flags = sorted((r for r in records if r["tier"] == 1), key=lambda r: (-r["gap"], r["floor"]))
        players = self.run["players"][0]
        last_floor = max((d.floor for d in self.decisions.values()), default=0)
        return {
            "run_id": self.run_key,
            "character": self.character,
            "ascension": self.run.get("ascension"),
            "build_id": self.run.get("build_id"),
            "win": bool(self.run.get("win")),
            "killed_by": self.engine.catalog.name("encounter", bare_id(self.run["killed_by_encounter"]))
            if self.run.get("killed_by_encounter") and self.run.get("killed_by_encounter") != "NONE.NONE" else None,
            "acts_reached": len(self.run.get("map_point_history") or ()),
            "last_scored_floor": last_floor,
            "final_deck_size": len(players.get("deck") or ()),
            "decisions_scored": collections.Counter(r["kind"] for r in records),
            "decisions_skipped": collections.Counter(s["reason"].split(":")[0] for s in self.skipped),
            "agreement": {"agree": agree, "judged": len(judged),
                          "rate": round(agree / len(judged), 3) if judged else None},
            "tier1": [r["id"] for r in flags],
            "tier2": [r["id"] for r in records if r["tier"] == 2],
            "warnings": self.warnings,
        }

    def decision(self, decision_id: str) -> Decision:
        try:
            return self.decisions[decision_id]
        except KeyError:
            raise ReviewError(f"no scored decision {decision_id}; scored: {', '.join(self.decisions)}") from None


def review_file(engine: Engine, path: Path, run_key: Optional[str] = None) -> RunReview:
    try:
        run = load_run(Path(path))
    except RunHistoryError as exc:
        raise ReviewError(str(exc)) from exc
    return RunReview(engine, run, run_key or Path(path).stem)
