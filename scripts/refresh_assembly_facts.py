"""Refresh the committed assembly facts from the committed extract.

Run after ``tools/Sts2FactExtractor`` regenerates the extract for a new game
build. Reviewed facts whose class code changed become ``stale``; nothing is
re-used silently. Prints the coverage report.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from storage.assembly_facts import (  # noqa: E402
    DEFAULT_EXTRACT_PATH,
    DEFAULT_FACTS_PATH,
    catalog_keys,
    coverage_summary,
    evaluate_facts,
    load_extract,
    load_facts,
    refresh_facts,
)

KNOWLEDGE_PATH = ROOT_DIR / "data" / "knowledge.json"
REPORT_PATH = ROOT_DIR / "build" / "assembly-coverage.json"


def write_facts(path: Path, facts: dict) -> None:
    text = json.dumps(facts, ensure_ascii=False, indent=1, sort_keys=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS_PATH)
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    extract = load_extract(args.extract)
    previous = load_facts(args.facts) if args.facts.exists() else None
    facts = refresh_facts(extract, catalog_keys(args.knowledge), previous)
    write_facts(args.facts, facts)

    evaluated = evaluate_facts(facts, extract)
    states = {key: entry["state"] for key, entry in sorted(evaluated.items())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(states, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"assembly {facts['assembly_sha256']}")
    for entity_type, counts in sorted(coverage_summary(states.items()).items()):
        print(entity_type, json.dumps(counts, sort_keys=True))
    stale = [key for key, state in states.items() if state == "stale"]
    print("stale", json.dumps(stale))
    print(f"per-entry states: {args.report}")
    triage = collections.Counter(
        entry["triage"]["category"]
        for entry in facts["entities"]
        if "triage" in entry
    )
    print("relic triage", json.dumps(dict(sorted(triage.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
