"""Anchors for the gold and HP exchange parameters (TASK-027 ticket 03).

Measured with the deck-fitness engine on the run history, never on what the
player chose:

* upgrade: at each rest site, the fitness gain of the best upgrade of the deck
  of that moment. The campfire rule (rest below 60% HP, validated on 260 rest
  decisions) says a rest heal of 30% of max HP is worth about that much at 60%
  HP, which prices HP there;
* shop: at each shop visit, the best fitness gain per gold among the cards
  offered and a card removal, at their base prices.

Gains are on the scale the policies score with: fitness looking ahead over
the act's coming rewards (``DeckFitnessPolicy._lookahead``), so a price and
the value it buys compare like with like.

    python scripts/analyze_resource_exchange.py [--decisions build/history-all/decisions.json]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.deck_fitness import DeckFitnessPolicy  # noqa: E402
from advisor.golden_set import world_request  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402
from storage.relational import RelationalRepository  # noqa: E402


def quantiles(values):
    if not values:
        return "none"
    values = sorted(values)
    q = statistics.quantiles(values, n=4) if len(values) > 3 else [values[0]] * 3
    return f"n={len(values)} median {statistics.median(values):.2f} (quartiles {q[0]:.2f} / {q[2]:.2f})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decisions", type=Path, default=ROOT_DIR / "build" / "history-all" / "decisions.json")
    args = parser.parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))["decisions"]
    with tempfile.TemporaryDirectory() as temp:
        database = Path(temp) / "release.db"
        build_release_database(database, ROOT_DIR / "data" / "knowledge.json", ROOT_DIR / "data" / "community_scores.json")
        repository = RelationalRepository(str(database))
        policy = DeckFitnessPolicy(repository)

        def ahead(request, context, cards):
            values, _ = policy._lookahead(
                request, {"deck": list(cards)}, lambda c: context.score(c)[0].raw, context.needs, context.params,
            )
            return values.get("deck", context.score(cards)[0].raw)

        upgrade_by_act = {1: [], 2: [], 3: []}
        heal_points = []
        for row in decisions:
            if row["kind"] != "rest":
                continue
            request = world_request(row["scoring_view"])
            context, _ = policy.deck_context(request)
            if context is None:
                continue
            base, immediate = ahead(request, context, context.deck), context.score(context.deck)[0].raw
            best_cards, best_immediate = None, 0.0
            seen = set()
            for i, card in enumerate(context.deck):
                if card is None or card.upgraded_profile is None or id(card) in seen:
                    continue
                seen.add(id(card))
                cards = [*context.deck[:i], card.upgraded_profile, *context.deck[i + 1:]]
                gain = context.score(cards)[0].raw - immediate
                if gain > best_immediate:
                    best_cards, best_immediate = cards, gain
            best = ahead(request, context, best_cards) - base if best_cards else 0.0
            upgrade_by_act.setdefault(row["act"], []).append(best)
            max_hp = row["scoring_view"]["state"].get("max_hp")
            if max_hp:
                # Points per HP at the 60% threshold: a 30% heal is worth the best upgrade there.
                heal_points.append(best / (0.3 * max_hp))
        # Shop: base prices read from the assembly (MerchantCardEntry.GetCost,
        # MerchantCardRemovalEntry.BaseCost / PriceIncrease; A6+ removal 100 + 50 each).
        rarity = {c["id"]: str(c.get("rarity_key") or "").lower() for c in json.loads(
            (ROOT_DIR / "data" / "knowledge.json").read_text(encoding="utf-8"))["cards"]}
        card_price = {"common": 50, "uncommon": 75, "rare": 150}
        best_by_act = {1: [], 2: [], 3: []}
        typical_by_act = {1: [], 2: [], 3: []}
        for row in decisions:
            if row["kind"] != "shop":
                continue
            request = world_request(row["scoring_view"])
            context, _ = policy.deck_context(request)
            if context is None:
                continue
            base = ahead(request, context, context.deck)
            gold = row["gold_before"]
            high = int(row["scoring_view"]["state"].get("ascension") or 0) >= 6
            deals = []
            for offer in row["offers"]["cards"]:
                price = card_price.get(rarity.get(offer["card"], ""))
                profile, _ = policy._card(offer["card"], offer["upgrades"])
                if price is None or profile is None or price > gold:
                    continue
                deals.append((ahead(request, context, [*context.deck, profile]) - base) / price)
            removal_price = (100 + 50 * row["removals_before"]) if high else (75 + 25 * row["removals_before"])
            if removal_price <= gold:
                # The removal the shop policy makes: the best by immediate fitness, then looked ahead.
                seen, best_cards, best_value = set(), None, None
                for i, card in enumerate(context.deck):
                    if card is None or id(card) in seen:
                        continue
                    seen.add(id(card))
                    cards = [*context.deck[:i], *context.deck[i + 1:]]
                    value = context.score(cards)[0].raw
                    if best_value is None or value > best_value:
                        best_cards, best_value = cards, value
                if best_cards is not None:
                    deals.append((ahead(request, context, best_cards) - base) / removal_price)
            positive = [d for d in deals if d > 0]
            if positive:
                best_by_act.setdefault(row["act"], []).append(100 * max(positive))
                typical_by_act.setdefault(row["act"], []).append(100 * statistics.median(positive))
        for act in sorted(best_by_act):
            print(f"shop, act {act}: best deal points per 100 gold {quantiles(best_by_act[act])}; "
                  f"median positive deal {quantiles(typical_by_act[act])}")
        for act, values in sorted(upgrade_by_act.items()):
            print(f"best upgrade gain, act {act}: {quantiles(values)}")
        print(f"points per HP at 60% HP (best upgrade / 30% max HP): {quantiles(heal_points)}")
        del policy, repository
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
