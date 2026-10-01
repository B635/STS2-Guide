"""Add one recorded card-reward state event to the golden set as a draft.

Only the scoring view is kept. Labels start as ``draft`` and must be
confirmed by the user before they count as confirmed results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.golden_set import (  # noqa: E402
    DEFAULT_GOLDEN_SET_PATH,
    load_golden_set,
    scoring_view,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="recorded state-event JSON")
    parser.add_argument("--id", required=True)
    parser.add_argument("--best", nargs="+", required=True, help="candidate ids")
    parser.add_argument("--errors", nargs="*", default=[], help="obvious-error candidate ids")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    args = parser.parse_args()

    event = json.loads(args.snapshot.read_text(encoding="utf-8"))
    payload = json.loads(args.golden_set.read_text(encoding="utf-8"))
    payload["situations"].append({
        "id": args.id,
        "character": event["state"]["character"],
        "source": {
            "kind": "live_snapshot",
            "run_id": event["run_id"],
            "sequence": event["sequence"],
            "producer_version": event.get("producer_version"),
        },
        "best": args.best,
        "obvious_errors": args.errors,
        "label_status": "draft",
        "rationale": args.rationale,
        "scoring_view": scoring_view(event),
    })
    temporary = args.golden_set.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    load_golden_set(temporary)  # refuse to write an invalid set
    temporary.replace(args.golden_set)
    print(f"added {args.id} ({len(payload['situations'])} situations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
