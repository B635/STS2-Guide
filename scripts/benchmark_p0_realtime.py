"""P0 realtime recommendation latency benchmark.

Uses a fixed scenario with a temporary SQLite database, no network,
no history persistence, and no live run state.  Outputs sample count,
P50, P95, and max latency.
"""
from __future__ import annotations

import copy
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from advisor.card_reward import recommend_card_reward
from storage.relational import RelationalRepository

BENCHMARK_STATE = {
    "character": "IRONCLAD",
    "ascension": 0,
    "act": 1,
    "floor": 6,
    "energy": 3,
    "hp": 70,
    "max_hp": 80,
    "deck": [
        {"card": "STRIKE_IRONCLAD", "count": 4},
        {"card": "DEFEND_IRONCLAD", "count": 4},
        {"card": "BASH", "count": 1},
    ],
    "relics": ["BURNING_BLOOD", "SHURIKEN"],
    "map_context": {
        "nodes": [
            {"node_id": "1:0:ELITE", "kind": "ELITE", "row": 1, "col": 0, "edges": ["2:0:CAMPFIRE"]},
            {"node_id": "1:1:MONSTER", "kind": "MONSTER", "row": 1, "col": 1, "edges": ["2:1:MONSTER"]},
            {"node_id": "2:0:CAMPFIRE", "kind": "CAMPFIRE", "row": 2, "col": 0, "edges": []},
            {"node_id": "2:1:MONSTER", "kind": "MONSTER", "row": 2, "col": 1, "edges": []},
        ],
        "available_next_node_ids": ["1:0:ELITE", "1:1:MONSTER"],
    },
}

BENCHMARK_OPTIONS = [
    {"card": "POMMEL_STRIKE", "upgrades": 0},
    {"card": "SHRUG_IT_OFF", "upgrades": 0},
    {"card": "DEFEND_IRONCLAD", "upgrades": 0},
]

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
P95_TARGET_MS = 300


def main() -> None:
    knowledge_path = os.path.join(ROOT_DIR, "data", "knowledge.json")
    if not os.path.exists(knowledge_path):
        print(f"Knowledge file not found: {knowledge_path}", file=sys.stderr)
        sys.exit(2)

    temp_dir = tempfile.mkdtemp(prefix="p0_bench_")
    db_path = os.path.join(temp_dir, "bench.db")

    try:
        repository = RelationalRepository(db_path)
        repository.sync_catalog(knowledge_path)

        # Warmup
        for _ in range(WARMUP_RUNS):
            state = copy.deepcopy(BENCHMARK_STATE)
            options = copy.deepcopy(BENCHMARK_OPTIONS)
            recommend_card_reward(
                state=state, options=options,
                repository=repository, can_skip=True,
            )

        # Benchmark
        latencies: list[float] = []
        for _ in range(BENCHMARK_RUNS):
            state = copy.deepcopy(BENCHMARK_STATE)
            options = copy.deepcopy(BENCHMARK_OPTIONS)
            t0 = time.perf_counter()
            recommend_card_reward(
                state=state, options=options,
                repository=repository, can_skip=True,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed)

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    maximum = latencies[-1]

    print(f"P0 推荐延迟基准")
    print(f"  样本数: {len(latencies)}")
    print(f"  P50:    {p50:.2f} ms")
    print(f"  P95:    {p95:.2f} ms")
    print(f"  最大:   {maximum:.2f} ms")
    print(f"  P95 目标: ≤ {P95_TARGET_MS} ms")

    if p95 <= P95_TARGET_MS:
        print(f"  [PASS] P95 满足目标")
    else:
        print(f"  [FAIL] P95 超过 {P95_TARGET_MS} ms 目标")
        sys.exit(1)


if __name__ == "__main__":
    main()
