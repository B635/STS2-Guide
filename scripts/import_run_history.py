"""Turn the game's own run history into card-reward situations and fight outcomes.

Reads ``<game data>/steam/<user>/[modded/]profile*/saves/history/*.run`` and
never writes there. Output goes to ``build/history/`` (not committed: it is
the player's own play record):

- ``card_rewards.json``: situations in the golden-set format, usable with
  ``scripts/eval_golden_set.py --golden-set``;
- ``fights.json``: every fight with its encounter, turns and damage taken.

Runs that cannot be replayed exactly (the replayed deck must equal the deck
the file ends with) are listed and left out.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.golden_set import GOLDEN_SET_SCHEMA_VERSION  # noqa: E402
from advisor.run_history import RunHistoryError, bare_id, load_run, replay_run  # noqa: E402
from realtime.run_records import unpack_collected  # noqa: E402

DEFAULT_HISTORY_ROOT = Path(os.environ.get("APPDATA", "")) / "SlayTheSpire2" / "steam"
DEFAULT_OUTPUT = ROOT_DIR / "build" / "history"


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def starting_decks(knowledge_path: Path) -> dict:
    knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))
    return {
        str(character["id"]).upper(): [_snake(card) for card in character.get("starting_deck") or ()]
        for character in knowledge["characters"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--all-runs", action="store_true", help="include lost runs (default: wins only)")
    parser.add_argument("--knowledge", type=Path, default=ROOT_DIR / "data" / "knowledge.json")
    parser.add_argument(
        "--collected", type=Path, action="append", default=[],
        help="a folder of run-record zips exported by other players (tray menu: 导出对局记录); repeatable",
    )
    parser.add_argument("--no-local", action="store_true", help="only the collected runs, not this machine's")
    args = parser.parse_args()

    decks = starting_decks(args.knowledge)
    known_cards = {
        str(card["id"]).upper()
        for card in json.loads(args.knowledge.read_text(encoding="utf-8"))["cards"]
    }
    known_encounters = {
        str(encounter["id"]).upper()
        for encounter in json.loads(args.knowledge.read_text(encoding="utf-8"))["encounters"]
    }
    # (file, run key): this machine's runs, then each collected player's,
    # keyed by their collector ID so file names cannot clash.
    files = [] if args.no_local else [(path, path.stem) for path in sorted(args.history_root.glob("**/history/*.run"))]
    unpacked = ROOT_DIR / "build" / "collected-runs"
    for folder in args.collected:
        for path in sorted(unpack_collected(sorted(folder.glob("*.zip")), unpacked)):
            files.append((path, f"{path.relative_to(unpacked).parts[0][:8]}-{path.stem}"))
    counts = collections.Counter()
    situations, fights, decisions, rejected = [], [], [], []
    seen_runs = set()
    for path, run_key in files:
        counts["files"] += 1
        try:
            run = load_run(path)
        except RunHistoryError as exc:
            rejected.append(f"{path.name}: {exc}")
            continue
        identity = (run.get("seed"), run.get("start_time"))
        if None in identity:
            identity = (str(path), None)   # no way to tell: kept
        if identity in seen_runs:
            # The same run sent twice, or by two players who shared it.
            counts["duplicates"] += 1
            continue
        seen_runs.add(identity)
        if len(run.get("players") or ()) != 1:
            counts["multiplayer"] += 1
            continue
        if run.get("was_abandoned"):
            counts["abandoned"] += 1
            continue
        if not run.get("win") and not args.all_runs:
            counts["lost"] += 1
            continue
        character = bare_id(run["players"][0]["character"])
        if character not in decks:
            rejected.append(f"{path.name}: unknown character {character}")
            continue
        try:
            result = replay_run(run, decks[character], run_key)
        except (AttributeError, KeyError, TypeError, ValueError, RunHistoryError) as exc:
            # One malformed file must not stop the import.
            rejected.append(f"{path.name}: unreadable run ({type(exc).__name__}: {exc})")
            continue
        if result["problems"]:
            rejected.append(f"{path.name} ({run.get('build_id')}): {result['problems'][0]}")
            continue
        counts["runs"] += 1
        counts["runs_with_warnings"] += bool(result["warnings"])
        for situation in result["situations"]:
            cards = {
                row["card"] for row in situation["scoring_view"]["state"]["deck"]
            } | {
                c["entity_id"] for c in situation["scoring_view"]["candidates"] if c["entity_id"]
            }
            unknown = sorted(cards - known_cards)
            if unknown:
                # A card that no longer exists in the current catalog.
                counts["situations_with_unknown_cards"] += 1
                continue
            bosses = situation["scoring_view"]["map_context"]["boss_encounter_ids"]
            if not bosses or any(boss not in known_encounters for boss in bosses):
                # A boss the current catalog does not have: no needs to score against.
                counts["situations_with_unknown_boss"] += 1
                continue
            situations.append(situation)
        fights.extend(result["fights"])
        decisions.extend(result["decisions"])

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "card_rewards.json").write_text(json.dumps({
        "schema_version": GOLDEN_SET_SCHEMA_VERSION,
        "note": "Generated from the game's run history by scripts/import_run_history.py; not committed.",
        "situations": situations,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.output / "decisions.json").write_text(
        json.dumps({"decisions": decisions}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (args.output / "fights.json").write_text(
        json.dumps({"fights": fights}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(dict(counts))
    print(f"situations {len(situations)} (skip answers {sum(s['best'] == ['skip'] for s in situations)}), "
          f"fights {len(fights)} -> {args.output}")
    print("other decisions", dict(collections.Counter(d["kind"] for d in decisions)))
    by_character = collections.Counter(s["character"] for s in situations)
    print("by character", dict(by_character))
    if rejected:
        print(f"rejected {len(rejected)}:")
        for line in rejected:
            print("  " + line[:230])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
