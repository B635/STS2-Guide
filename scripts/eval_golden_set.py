"""Evaluate card-reward policy versions on the golden set (标准答案集).

Builds a temporary release database from the committed inputs, runs every
situation through each requested policy version and prints hit rate and
obvious-error counts. ``--save`` archives the full report for later comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.golden_set import (  # noqa: E402
    CARD_REWARD_VERSIONS,
    DEFAULT_GOLDEN_SET_PATH,
    evaluate_golden_set,
    load_golden_set,
)
from storage.relational import RelationalRepository  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402

REPORT_DIR = ROOT_DIR / "data" / "golden_set" / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument(
        "--versions", nargs="+", default=sorted(CARD_REWARD_VERSIONS),
        choices=sorted(CARD_REWARD_VERSIONS),
    )
    parser.add_argument("--save", action="store_true", help="archive each report")
    args = parser.parse_args()

    situations = load_golden_set(args.golden_set)
    with tempfile.TemporaryDirectory() as temporary:
        database = Path(temporary) / "release.db"
        build_release_database(
            database,
            ROOT_DIR / "data" / "knowledge.json",
            ROOT_DIR / "data" / "community_scores.json",
        )
        repository = RelationalRepository(str(database))
        for version in args.versions:
            report = evaluate_golden_set(situations, CARD_REWARD_VERSIONS[version](repository))
            for scope in ("confirmed", "all"):
                part = report[scope]
                print(
                    f"{version} [{scope} labels]: {part['hits']}/{part['situations']} "
                    f"best-choice hits (hit rate {part['hit_rate']}), "
                    f"obvious errors ranked first {part['obvious_errors_first']}"
                )
            for row in report["rows"]:
                mark = "HIT " if row["hit"] else ("ERR " if row["obvious_error_first"] else "miss")
                notes = ",".join([row["label_status"], *row["warnings"]])
                print(f"  {mark} {row['id']} -> {row['first']} ({notes})")
            if args.save:
                REPORT_DIR.mkdir(parents=True, exist_ok=True)
                path = REPORT_DIR / f"{version}.json"
                with path.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
                print(f"  saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
