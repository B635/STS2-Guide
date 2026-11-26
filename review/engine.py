"""The engine the review scores with: the release database and the shipped policies.

The database is rebuilt from tracked files only (``data/``), into
``build/coach/release.db``, whenever an input is newer than it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Optional

from advisor.campfire import CampfirePolicy
from advisor.decision_core import PolicyRegistry
from advisor.golden_set import CARD_REWARD_VERSIONS
from advisor.history_requests import rarities
from advisor.merchant import MerchantPolicy
from advisor.deck_fitness import PARAMETERS_KEY
from storage.assembly_facts import DEFAULT_EXTRACT_PATH, DEFAULT_FACTS_PATH
from storage.release_database import DEFAULT_GUIDE_PARAMETERS_PATH, build_release_database
from storage.relational import RelationalRepository

ROOT_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = ROOT_DIR / "data" / "knowledge.json"
COMMUNITY_PATH = ROOT_DIR / "data" / "community_scores.json"
DEFAULT_DATABASE = ROOT_DIR / "build" / "coach" / "release.db"


def ensure_database(path: Path = DEFAULT_DATABASE) -> Path:
    inputs = (KNOWLEDGE_PATH, COMMUNITY_PATH, Path(DEFAULT_EXTRACT_PATH), Path(DEFAULT_FACTS_PATH),
              Path(DEFAULT_GUIDE_PARAMETERS_PATH))
    newest = max(p.stat().st_mtime for p in inputs)
    if not path.is_file() or path.stat().st_mtime < newest:
        build_release_database(path, KNOWLEDGE_PATH, COMMUNITY_PATH)
    return path


def _apply_upgrade_delta(base: Optional[int], upgrade: Optional[Mapping], key: str) -> Optional[int]:
    if base is None or not upgrade or key not in upgrade:
        return base
    raw = upgrade[key]
    if isinstance(raw, bool):
        return base
    if isinstance(raw, (int, float)):
        return int(base + raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("+", "-")):
            try:
                return int(base + int(text))
            except ValueError:
                return base
    return base


def restore_icons(
    text: Optional[str],
    vars_map: Optional[Mapping],
    upgrade: Optional[Mapping] = None,
    *,
    upgraded: bool = False,
) -> str:
    """Restore inline star/energy icon amounts stripped from catalog Chinese descriptions."""
    cleaned = re.sub(r"\s*\n\s*", " ", text or "")
    if not cleaned:
        return cleaned
    v = vars_map or {}
    stars = v.get("Stars")
    energy = v.get("Energy")
    if upgraded:
        stars = _apply_upgrade_delta(stars, upgrade, "stars")
        energy = _apply_upgrade_delta(energy, upgrade, "energy")
    if stars is not None and energy is not None:
        cleaned = re.sub(
            r"获得与(?=[。，；]|$)",
            f"获得 {energy} 点能量与 {stars} 颗星",
            cleaned,
        )
        gains = [f"{stars} 颗星", f"{energy} 点能量"]
        matches = list(re.finditer(r"获得(?=[。，；并]|$)", cleaned))
        if len(matches) >= 2:
            idx = 0

            def _repl_pair(_match: re.Match) -> str:
                nonlocal idx
                piece = gains[idx] if idx < len(gains) else f"{energy} 点能量与 {stars} 颗星"
                idx += 1
                return f"获得 {piece}"

            cleaned = re.sub(r"获得(?=[。，；并]|$)", _repl_pair, cleaned)
        elif len(matches) == 1:
            cleaned = re.sub(r"获得(?=[。，；并]|$)", f"获得 {energy} 点能量与 {stars} 颗星", cleaned)
    elif stars is not None:
        cleaned = re.sub(r"获得(?=[。，；并]|$)", f"获得 {stars} 颗星", cleaned)
        cleaned = re.sub(rf"获得{re.escape(str(stars))}(?=[。，；并]|$)", f"获得 {stars} 颗星", cleaned)
    elif energy is not None:
        cleaned = re.sub(r"获得(?=[。，；并]|$)", f"获得 {energy} 点能量", cleaned)
        cleaned = re.sub(rf"获得{re.escape(str(energy))}(?=[。，；并]|$)", f"获得 {energy} 点能量", cleaned)
    return cleaned


class Catalog:
    """Display names and descriptions (the game's own Chinese text) by ID."""

    def __init__(self, knowledge: Mapping):
        self.cards = {c["id"]: c for c in knowledge["cards"]}
        self.relics = {r["id"]: r for r in knowledge["relics"]}
        self.potions = {p["id"]: p for p in knowledge["potions"]}
        self.encounters = {e["id"]: e for e in knowledge["encounters"]}

    def name(self, kind: str, entity_id: Optional[str]) -> Optional[str]:
        if entity_id is None:
            return None
        table = {"card": self.cards, "relic": self.relics, "potion": self.potions,
                 "encounter": self.encounters}[kind]
        row = table.get(entity_id)
        return row["name"] if row else entity_id

    def card(self, card_id: str) -> Optional[dict]:
        row = self.cards.get(card_id)
        if row is None:
            return None
        keys = ("id", "name", "description", "cost", "star_cost", "type", "rarity", "color", "target", "damage",
                "block", "hit_count", "cards_draw", "energy_gain", "keywords", "upgrade_description")
        out = {k: row.get(k) for k in keys if row.get(k) is not None}
        if "description" in out:
            out["description"] = restore_icons(
                out["description"], row.get("vars"), row.get("upgrade"), upgraded=False
            )
        if "upgrade_description" in out:
            out["upgrade_description"] = restore_icons(
                out["upgrade_description"], row.get("vars"), row.get("upgrade"), upgraded=True
            )
        return out

    def relic(self, relic_id: str) -> Optional[dict]:
        row = self.relics.get(relic_id)
        if row is None:
            return None
        return {k: row.get(k) for k in ("id", "name", "description", "rarity", "pool") if row.get(k) is not None}


class Engine:
    def __init__(self, database: Optional[Path] = None):
        self.database = ensure_database(database or DEFAULT_DATABASE)
        self.repository = RelationalRepository(str(self.database))
        self.params = self.repository.mechanic_constant(PARAMETERS_KEY)
        # Kept by itself too: deck_trend asks it for the deck's capabilities and needs.
        self.deck_fitness = CARD_REWARD_VERSIONS["v2"](self.repository)
        self.card_rewards = PolicyRegistry([self.deck_fitness])
        self.campfire = CampfirePolicy(self.repository)
        self.merchant = MerchantPolicy(self.repository)
        knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        self.catalog = Catalog(knowledge)
        self.rarity = rarities(knowledge)
        self.upgradable = {c["id"]: bool(c.get("upgrade")) for c in knowledge["cards"]}
        self.known_cards = {str(c["id"]).upper() for c in knowledge["cards"]}
        self.known_encounters = {str(e["id"]).upper() for e in knowledge["encounters"]}
