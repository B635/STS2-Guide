"""Build the offline SQLite template shipped with a Guide release."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.release_database import build_release_database


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean, release-bound STS2 Guide SQLite template."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "data" / "knowledge.json",
    )
    parser.add_argument(
        "--community-scores",
        type=Path,
        default=ROOT / "data" / "community_scores.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "release" / "sts2-guide-template.db",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_release_database(
        args.output,
        args.catalog,
        args.community_scores,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
