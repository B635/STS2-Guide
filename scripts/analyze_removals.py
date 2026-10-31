"""Which card the deck engine would remove, against what the player removed (TASK-027 follow-up).

For each shop removal of the run history (``kind: remove`` in decisions.json)
the engine removes, in turn, one copy of every distinct card of the deck of
that moment and ranks the results by fitness (immediate, and looking ahead
over the act's coming rewards as the policies score). As the shipped policies
do, a card valued only in part is chosen only when no fully valued card can
be (its removal gain is overstated). Reports agreement and a table by card
kind (Strike / Defend / starter attack / curse or status / other), to see
where the engine and the player part ways.

    python scripts/analyze_removals.py [--decisions build/history-all/decisions.json] [-v]
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.deck_fitness import DeckFitnessPolicy  # noqa: E402
from advisor.golden_set import world_request  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402
from storage.relational import RelationalRepository  # noqa: E402


def kind(card_id: str, card_type: str, rarity: str) -> str:
    if card_type in ("Curse", "Status"):
        return "curse/status"
    if card_id.startswith("STRIKE_"):
        return "strike"
    if card_id.startswith("DEFEND_"):
        return "defend"
    if rarity == "basic":
        return "other starter"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decisions", type=Path, default=ROOT_DIR / "build" / "history-all" / "decisions.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    knowledge = json.loads((ROOT_DIR / "data" / "knowledge.json").read_text(encoding="utf-8"))
    rarity = {c["id"]: str(c.get("rarity_key") or "").lower() for c in knowledge["cards"]}
    rows = [d for d in json.loads(args.decisions.read_text(encoding="utf-8"))["decisions"] if d["kind"] == "remove"]
    counts = {"immediate": collections.Counter(), "lookahead": collections.Counter()}
    table = {"immediate": collections.Counter(), "lookahead": collections.Counter()}
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
                values, _ = policy._lookahead(request, {"d": list(cards)}, lambda c: context.score(c)[0].raw,
                                              context.needs, context.params)
                return values.get("d", context.score(cards)[0].raw)

            options = {}
            for i, card in enumerate(context.deck):
                if card is None or card.card_id in options:
                    continue
                cards = context.deck[:i] + context.deck[i + 1:]
                partial = bool(policy._card(card.card_id, 1 if card.upgraded else 0)[1])
                options[card.card_id] = (context.score(cards)[0].raw, ahead(cards), card.card_type, partial)
            chosen = row["chosen"]
            if chosen not in options:
                continue
            player_kind = kind(chosen, options[chosen][2], rarity.get(chosen, ""))
            whole = [c for c in options if not options[c][3]] or list(options)
            for scale, index in (("immediate", 0), ("lookahead", 1)):
                best = max(whole, key=lambda c: options[c][index])
                counts[scale]["n"] += 1
                counts[scale]["hit"] += best == chosen
                table[scale][(player_kind, kind(best, options[best][2], rarity.get(best, "")))] += 1
            if args.verbose:
                ranked = sorted(whole, key=lambda c: -options[c][1])
                print(f"{row['character']:11} act {row['act']} removed {chosen:18} engine order: {', '.join(ranked[:4])}")
        del policy, repository
    for scale in ("immediate", "lookahead"):
        n = max(1, counts[scale]["n"])
        print(f"{scale}: engine's removal = player's {counts[scale]['hit']}/{counts[scale]['n']} ({counts[scale]['hit'] / n:.0%})")
        print("   player removed -> engine would remove: " + "; ".join(
            f"{p} -> {e}: {c}" for (p, e), c in sorted(table[scale].items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
