"""Score the shop visits of the run history with the shipped merchant policy (TASK-027 ticket 06).

Reads the ``shop`` decisions written by ``scripts/import_run_history.py``: what
was on sale, what the player bought and the gold before the visit. The record
has no prices, so each offer gets its base price from the assembly (cards
50 / 75 / 150 by rarity, relics 175 / 225 / 275 / 200 by rarity, potions
50 / 75 / 100, removal 75 + 25 per earlier removal, A6+ 100 + 50); an offer
dearer than the gold is not eligible, as in the game. Colorless cards are not
in the record unless bought and are left out. Removal may target any card.

Reports how often the first recommendation is something the player bought
(or leaving when nothing was bought), and the buy-or-leave agreement.

    python scripts/eval_shop_choices.py [--decisions build/history-all/decisions.json] [-v]
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

from advisor.history_requests import bought_ids, rarities, shop_request  # noqa: E402
from advisor.merchant import MerchantPolicy  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402
from storage.relational import RelationalRepository  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decisions", type=Path, default=ROOT_DIR / "build" / "history-all" / "decisions.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    knowledge = json.loads((ROOT_DIR / "data" / "knowledge.json").read_text(encoding="utf-8"))
    rarity = rarities(knowledge)
    rows = [d for d in json.loads(args.decisions.read_text(encoding="utf-8"))["decisions"] if d["kind"] == "shop"]
    with tempfile.TemporaryDirectory() as temp:
        database = Path(temp) / "release.db"
        build_release_database(database, ROOT_DIR / "data" / "knowledge.json", ROOT_DIR / "data" / "community_scores.json")
        repository = RelationalRepository(str(database))
        policy = MerchantPolicy(repository)
        counts = collections.Counter()
        gaps = collections.Counter()
        for row in rows:
            request = shop_request(row, rarity)
            result = policy.recommend(request)
            counts["visits"] += 1
            if result.recommended_candidate_id is None:
                counts["uncertain"] += 1
                gaps.update(":".join(g.split(":")[:2]) for g in result.data_gaps)
                continue
            counts["recommended"] += 1
            bought = bought_ids(row, request)
            hit = result.recommended_candidate_id in bought
            counts["hits"] += hit
            counts["buy_agrees"] += (result.recommended_candidate_id == "leave") == (bought == {"leave"})
            counts["player_left"] += bought == {"leave"}
            counts["engine_left"] += result.recommended_candidate_id == "leave"
            gaps.update(":".join(g.split(":")[:2]) for c in result.candidates for g in c.data_gaps)
            if args.verbose:
                top = sorted((c for c in result.candidates if c.score is not None), key=lambda c: -c.score)[:4]
                print(f"{'HIT ' if hit else 'miss'} {row['character']:11} act {row['act']} gold {row['gold_before']:4} "
                      f"bought {sorted(bought)}: " + " | ".join(f"{c.candidate_id} {c.score:.1f}" for c in top))
        del policy, repository
    n = max(1, counts["recommended"])
    print(f"shop visits {counts['visits']}: recommended {counts['recommended']}, uncertain {counts['uncertain']}; "
          f"first recommendation bought by the player {counts['hits']} ({counts['hits'] / n:.0%}); "
          f"buy-or-leave agrees {counts['buy_agrees']} ({counts['buy_agrees'] / n:.0%}); "
          f"player left {counts['player_left']}, engine says leave {counts['engine_left']}")
    print("left out:", gaps.most_common(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
