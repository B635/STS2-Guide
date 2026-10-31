"""Marginal value of gold by how much is held (TASK-027 ticket 03).

At each shop visit of the run history, the deck-fitness gain of every
affordable offer (cards at their base price, the best removal at its
removal price) is divided by its price, on the policies' scale (fitness
looking ahead over the act's coming rewards). A player spending G gold buys
the best deals first, so the G-th gold buys the deal whose cumulative price
covers G. The median of that rate over the visits, by band of gold spent, is
the marginal value of gold at that amount. Measured on the engine and the
shops' structure, never on what the player chose.

    python scripts/analyze_gold_curve.py [--decisions build/history-all/decisions.json] [--band 50]
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

CARD_PRICE = {"common": 50, "uncommon": 75, "rare": 150}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decisions", type=Path, default=ROOT_DIR / "build" / "history-all" / "decisions.json")
    parser.add_argument("--band", type=int, default=50)
    parser.add_argument("--max-gold", type=int, default=600)
    args = parser.parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    rarity = {c["id"]: str(c.get("rarity_key") or "").lower() for c in json.loads(
        (ROOT_DIR / "data" / "knowledge.json").read_text(encoding="utf-8"))["cards"]}
    rows = [d for d in json.loads(args.decisions.read_text(encoding="utf-8"))["decisions"] if d["kind"] == "shop"]
    bands = list(range(0, args.max_gold, args.band))
    curves = {1: {b: [] for b in bands}, 2: {b: [] for b in bands}, 3: {b: [] for b in bands}}
    with tempfile.TemporaryDirectory() as temp:
        database = Path(temp) / "release.db"
        build_release_database(database, ROOT_DIR / "data" / "knowledge.json", ROOT_DIR / "data" / "community_scores.json")
        repository = RelationalRepository(str(database))
        policy = DeckFitnessPolicy(repository)
        for row in rows:
            request = world_request(row["scoring_view"])
            context, _ = policy.deck_context(request)
            if context is None:
                continue

            def ahead(cards):
                values, _ = policy._lookahead(
                    request, {"deck": list(cards)}, lambda c: context.score(c)[0].raw, context.needs, context.params,
                )
                return values.get("deck", context.score(cards)[0].raw)

            base = ahead(context.deck)
            deals = []
            for offer in row["offers"]["cards"]:
                price = CARD_PRICE.get(rarity.get(offer["card"], ""))
                profile, _ = policy._card(offer["card"], offer["upgrades"])
                if price is not None and profile is not None:
                    deals.append((ahead([*context.deck, profile]) - base, price))
            high = int(row["scoring_view"]["state"].get("ascension") or 0) >= 6
            removal = (100 + 50 * row["removals_before"]) if high else (75 + 25 * row["removals_before"])
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
                deals.append((ahead(best_cards) - base, removal))
            # Best deals first; the gold spent so far decides which deal the next gold buys.
            deals = sorted(((gain / price, price) for gain, price in deals if gain > 0), reverse=True)
            for band in bands:
                spent, rate = 0, 0.0
                for ratio, price in deals:
                    if spent + price > band:
                        rate = ratio
                        break
                    spent += price
                curves.setdefault(row["act"], {b: [] for b in bands})[band].append(100 * rate)
        del policy, repository
    for act, curve in sorted(curves.items()):
        cells = [f"{band}-{band + args.band}: {statistics.median(values):.2f}" for band, values in curve.items() if values]
        print(f"act {act} (n={len(curve[0])}), points per 100 gold at gold spent " + ", ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
