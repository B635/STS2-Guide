"""Card-reward decision time on the golden set, after warm-up.

Builds the release database from the source data, warms the shipped policy up
like the host does at start, then times every golden situation ``--rounds``
times. Reports P50/P95/max and how many decisions went over the real-time
budget (spec: P95 <= 300 ms for the whole loop). CPU time by default, so other
programs on the machine move it less; ``--wall`` for wall-clock time.

    python scripts/benchmark_card_reward_latency.py [--rounds 3] [--wall]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.deck_fitness import DeckFitnessPolicy  # noqa: E402
from advisor.golden_set import card_reward_request, load_golden_set  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402
from storage.relational import RelationalRepository  # noqa: E402

BUDGET_MS = 300.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--wall", action="store_true", help="wall-clock time instead of CPU time")
    args = parser.parse_args()
    clock = time.perf_counter if args.wall else time.process_time
    situations = load_golden_set(str(ROOT_DIR / "data" / "golden_set" / "card_reward.json"))
    with tempfile.TemporaryDirectory() as temp:
        database = Path(temp) / "release.db"
        build_release_database(database, ROOT_DIR / "data" / "knowledge.json", ROOT_DIR / "data" / "community_scores.json")
        repository = RelationalRepository(str(database))
        policy = DeckFitnessPolicy(repository)
        start = clock()
        policy.warm_up()
        warm_up_ms = (clock() - start) * 1000
        requests = [card_reward_request(s["scoring_view"]) for s in situations]
        times = []
        for _ in range(max(1, args.rounds)):
            for request in requests:
                start = clock()
                policy.recommend(request)
                times.append((clock() - start) * 1000)
        del policy, repository
    times.sort()
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    print(f"{'wall' if args.wall else 'cpu'} time, {len(requests)} situations x {args.rounds}: warm-up {warm_up_ms:.0f} ms, "
          f"P50 {times[len(times) // 2]:.0f} ms, P95 {p95:.0f} ms, max {times[-1]:.0f} ms, "
          f"over {BUDGET_MS:.0f} ms {sum(t > BUDGET_MS for t in times)}/{len(times)}")
    return 0 if p95 <= BUDGET_MS else 1


if __name__ == "__main__":
    raise SystemExit(main())
