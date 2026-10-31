"""Score the ancient relic choices of the run history (TASK-027 ticket 05).

Reads ``decisions.json`` written by ``scripts/import_run_history.py`` (kind
``ancient_relic``: the options offered, the one taken and the state before
the choice), asks the shipped Neow / ancient policy for each, and reports how
often its first choice is the relic the player took, how often the call was
firm ("medium"), and what was left out of the scores.

    python scripts/eval_relic_choices.py [--decisions build/history-all/decisions.json] [-v]
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.decision_core import DecisionCandidate, DecisionRequest  # noqa: E402
from advisor.golden_set import world_request  # noqa: E402
from advisor.neow import CHARACTERS, NEOW_BLESSING, NeowPolicy  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402
from storage.relational import RelationalRepository  # noqa: E402


def relic_request(row: dict) -> DecisionRequest:
    """The choice as the Mod sends it: one candidate per option, the relic as
    its entity (Orobas's Sea Glass options are keyed by the character)."""

    world = world_request(row["scoring_view"]).world
    candidates = []
    for index, key in enumerate(row["options"]):
        relic = "SEA_GLASS" if key in CHARACTERS else key
        candidates.append(DecisionCandidate.create(
            f"{index}:{key}",
            {"candidate_kind": "neow_blessing", "entity_id": relic, "blessing_id": key,
             "stage_id": "INITIAL", "option_id": key, "effects": []},
            label=key, display_index=index,
        ))
    return DecisionRequest.create(
        decision_id=f"{row['run']}:ancient:{row['floor']}",
        decision_type=NEOW_BLESSING, world=world, candidates=candidates,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--decisions", type=Path, default=ROOT_DIR / "build" / "history-all" / "decisions.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    rows = [d for d in json.loads(args.decisions.read_text(encoding="utf-8"))["decisions"] if d["kind"] == "ancient_relic"]
    with tempfile.TemporaryDirectory() as temp:
        database = Path(temp) / "release.db"
        build_release_database(database, ROOT_DIR / "data" / "knowledge.json", ROOT_DIR / "data" / "community_scores.json")
        repository = RelationalRepository(str(database))
        policy = NeowPolicy(repository)
        counts = collections.Counter()
        gaps = collections.Counter()
        times = []
        for row in rows:
            request = relic_request(row)
            start = time.perf_counter()
            result = policy.recommend(request)
            times.append(time.perf_counter() - start)
            counts["choices"] += 1
            if result.recommended_candidate_id is None:
                counts["uncertain"] += 1
                gaps.update(":".join(g.split(":")[:2]) for g in result.data_gaps)
                continue
            counts["recommended"] += 1
            chosen = next(c.candidate_id for c in request.candidates if c.label == row["chosen"])
            hit = result.recommended_candidate_id == chosen
            counts["hits"] += hit
            counts["firm"] += result.confidence == "medium"
            counts["firm_hits"] += hit and result.confidence == "medium"
            gaps.update(":".join(g.split(":")[:2]) for c in result.candidates for g in c.data_gaps)
            if args.verbose:
                rows_text = " | ".join(
                    f"{c.label} {c.score:.1f}{'*' if c.data_gaps else ''}"
                    for c in sorted(result.candidates, key=lambda c: -(c.score or 0)) if c.score is not None
                )
                print(f"{'HIT ' if hit else 'miss'} {row['character']:11} f{row['floor']:<2} {result.confidence:6} "
                      f"took {row['chosen']}: {rows_text}")
        del policy, repository
    times.sort()
    recommended = max(1, counts["recommended"])
    print(f"ancient choices {counts['choices']}: recommended {counts['recommended']}, uncertain {counts['uncertain']}; "
          f"first choice = player's pick {counts['hits']} ({counts['hits'] / recommended:.0%}); "
          f"firm calls {counts['firm']}, of which the player's pick {counts['firm_hits']}")
    if times:
        print(f"time per choice: median {times[len(times) // 2] * 1000:.0f} ms, max {times[-1] * 1000:.0f} ms")
    print("left out:", gaps.most_common(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
