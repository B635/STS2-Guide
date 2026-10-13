"""List what the deck engine cannot convert yet, fact by fact.

Every reviewed card, relic, player-side power and enchantment goes through
the same profile builders the v2 policy uses. A fact with no gap is fully
scored; the others are listed with the reason the engine gave. Writes
``build/engine-coverage.json`` and prints a summary grouped by reason.
Analysis only: it changes nothing.
"""

from __future__ import annotations

import collections
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from advisor.deck_mechanics import card_profile, enchant, power_use, relic_profile  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402
from storage.relational import RelationalRepository  # noqa: E402

REPORT_PATH = ROOT_DIR / "build" / "engine-coverage.json"


def reason(gap: str) -> str:
    """``card_effect_unsupported:ID:detail`` -> ``detail`` (the mechanic)."""

    parts = gap.split(":")
    return ":".join(parts[2:]) if len(parts) > 2 else parts[0]


def main() -> int:
    knowledge = json.loads((ROOT_DIR / "data" / "knowledge.json").read_text(encoding="utf-8"))
    facts = json.loads((ROOT_DIR / "data" / "assembly" / "facts.json").read_text(encoding="utf-8"))
    cards = {str(c["id"]).upper(): c for c in knowledge["cards"]}
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        database = Path(folder) / "release.db"
        build_release_database(
            database, ROOT_DIR / "data" / "knowledge.json", ROOT_DIR / "data" / "community_scores.json"
        )
        repository = RelationalRepository(str(database))
        for entry in facts["entities"]:
            kind, _, entity_id = str(entry.get("entity_key") or "").partition(":")
            if entry.get("status") != "converted":
                continue
            gaps: tuple = ()
            group = None
            if kind == "cards":
                card = cards.get(entity_id, {})
                group = str(card.get("color") or "other").upper()
                seen = []
                for upgrades in (0, 1):
                    _, found = card_profile(repository, entity_id, upgrades)
                    seen.extend(found)
                gaps = tuple(dict.fromkeys(seen))
            elif kind == "relics":
                group = "RELIC"
                _, gaps = relic_profile(repository, entity_id)
            elif kind == "powers":
                group = "POWER"
                _, gaps = power_use(repository, str(entry.get("class") or "").removesuffix("Power"), 1)
            elif kind == "enchantments":
                group = "ENCHANTMENT"
                strike, _ = card_profile(repository, "STRIKE_IRONCLAD", 0)
                _, gaps = enchant(strike, entry, 1.0)
            else:
                continue
            rows.append({
                "entity_key": entry["entity_key"], "kind": kind, "group": group,
                "gaps": [reason(gap) for gap in gaps], "raw": list(gaps),
            })
        del repository

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    by_group = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        by_group[row["group"]][0] += 1
        by_group[row["group"]][1] += not row["gaps"]
    print("fully scored / reviewed, by group")
    for group, (total, full) in sorted(by_group.items()):
        print(f"  {group:14} {full:4}/{total:4} = {full / total:.0%}")
    total = sum(v[0] for v in by_group.values())
    full = sum(v[1] for v in by_group.values())
    print(f"  {'ALL':14} {full:4}/{total:4} = {full / total:.0%}")

    reasons = collections.defaultdict(list)
    for row in rows:
        for gap in row["gaps"]:
            reasons[gap].append(row["entity_key"])
    print(f"\n{len(reasons)} distinct reasons, most common first")
    for gap, keys in sorted(reasons.items(), key=lambda item: -len(item[1])):
        names = ", ".join(key.split(":", 1)[1] for key in keys[:6])
        print(f"  {len(keys):3}  {gap[:70]:70}  {names}{' ...' if len(keys) > 6 else ''}")
    print(f"\nfull list: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
