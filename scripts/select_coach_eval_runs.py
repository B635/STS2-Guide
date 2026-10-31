"""Pick the review coach's evaluation runs once (TASK-028): a dev set and a held-out test set.

Reads this machine's run history (the developer's own runs), keeps the runs
the review can score (single-player, not abandoned, replayed exactly), and
splits them by character with a fixed seed, before any prompt work and
without looking at the flags:

- test (15): every v0.111.0 run first, then filled round-robin by character;
- dev (10): round-robin by character from the rest;
- the remaining runs are scored for the automatic metrics only.

Copies the dev and test files to ``coach/eval/runs/{dev,test}/`` and writes
``coach/eval/split.json``. Run it once; re-running with the same history and
seed gives the same split.

    python scripts/select_coach_eval_runs.py [--history-root ...] [--seed 28]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from review.engine import Engine  # noqa: E402
from review.run_review import ReviewError, review_file  # noqa: E402

DEFAULT_HISTORY_ROOT = Path(os.environ.get("APPDATA", "")) / "SlayTheSpire2" / "steam"
OUTPUT = ROOT_DIR / "coach" / "eval"
TEST_SIZE, DEV_SIZE = 15, 10
CURRENT_BUILD = "v0.111.0"


def round_robin(by_character: dict, size: int, rng: random.Random, taken: list) -> list:
    pools = {c: [r for r in runs if r not in taken] for c, runs in sorted(by_character.items())}
    for runs in pools.values():
        rng.shuffle(runs)
    picked = []
    while len(picked) < size and any(pools.values()):
        for character in sorted(pools):
            if pools[character] and len(picked) < size:
                picked.append(pools[character].pop())
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    engine = Engine()
    runs, rejected, seen = [], collections.Counter(), set()
    for path in sorted(args.history_root.glob("**/history/*.run")):
        try:
            review = review_file(engine, path)
        except ReviewError as exc:
            rejected[str(exc).split(" (")[0].split(":")[0]] += 1
            continue
        identity = (review.run.get("seed"), review.run.get("start_time"))
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        summary = review.summary()
        runs.append({"run": path.stem, "path": path, "character": summary["character"],
                     "build_id": summary["build_id"], "win": summary["win"], "scored": sum(summary["decisions_scored"].values())})
    reviewable = [r for r in runs if r["scored"]]
    by_character = collections.defaultdict(list)
    for r in reviewable:
        by_character[r["character"]].append(r["run"])

    rng = random.Random(args.seed)
    test = sorted(r["run"] for r in reviewable if r["build_id"] == CURRENT_BUILD)
    test += round_robin(by_character, TEST_SIZE - len(test), rng, test)
    dev = round_robin(by_character, DEV_SIZE, rng, test)
    rows = {r["run"]: r for r in reviewable}
    split = {"seed": args.seed, "test": sorted(test), "dev": sorted(dev),
             "other": sorted(set(rows) - set(test) - set(dev))}

    for name in ("dev", "test"):
        folder = OUTPUT / "runs" / name
        folder.mkdir(parents=True, exist_ok=True)
        for run in split[name]:
            shutil.copy2(rows[run]["path"], folder / f"{run}.run")
    (OUTPUT / "split.json").write_text(json.dumps({
        "note": "Chosen once by scripts/select_coach_eval_runs.py before any prompt work; never tune on test.",
        **split,
        "runs": {run: {k: rows[run][k] for k in ("character", "build_id", "win")} for run in sorted(rows)},
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"reviewable {len(reviewable)} (rejected {dict(rejected)}, nothing scorable "
          f"{len(runs) - len(reviewable)}): test {len(test)}, dev {len(dev)}, other {len(split['other'])}")
    for name in ("test", "dev"):
        print(name, dict(collections.Counter(rows[r]["character"] for r in split[name])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
