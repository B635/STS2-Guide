"""SQLite-backed repository for structured STS2 data and final run summaries.

The JSON files under ``data/`` are ingestion snapshots. Runtime structured
queries use this repository; unstructured guide text remains in the vector
index.  Legacy decision-trace tables remain migration-compatible, but the P0
realtime pipeline does not write them.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from storage.effect_tags import EFFECT_TAG_VERSION, derive_effect_tags


CATALOG_TYPES = (
    "characters",
    "cards",
    "relics",
    "potions",
    "monsters",
    "encounters",
    "events",
    "acts",
    "powers",
    "intents",
    "keywords",
    "enchantments",
    "afflictions",
    "orbs",
    "modifiers",
)
SCHEMA_VERSION = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class RelationalRepository:
    """Owns relational catalog, statistics, and final run summaries."""

    def __init__(self, database_path: str):
        self.database_path = str(Path(database_path))

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ensure_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS data_sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    terms_url TEXT,
                    usage_status TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS source_snapshots (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL
                        REFERENCES data_sources(id),
                    entity_type TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    game_version TEXT,
                    fetched_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    methodology TEXT NOT NULL DEFAULT '',
                    UNIQUE(source_id, entity_type, data_type, content_hash)
                );

                CREATE TABLE IF NOT EXISTS entity_statistics (
                    snapshot_id TEXT NOT NULL
                        REFERENCES source_snapshots(id) ON DELETE CASCADE,
                    entity_key TEXT NOT NULL
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    score REAL,
                    elo REAL,
                    picks INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    win_rate REAL,
                    PRIMARY KEY(snapshot_id, entity_key)
                );
                CREATE INDEX IF NOT EXISTS idx_entity_statistics_entity
                    ON entity_statistics(entity_key, snapshot_id);

                CREATE TABLE IF NOT EXISTS catalog_entities (
                    entity_key TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    embed_text TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    UNIQUE(entity_type, external_id)
                );
                CREATE INDEX IF NOT EXISTS idx_catalog_entities_type_name
                    ON catalog_entities(entity_type, name);

                CREATE TABLE IF NOT EXISTS cards (
                    entity_key TEXT PRIMARY KEY
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    color TEXT,
                    type_key TEXT,
                    rarity_key TEXT,
                    cost INTEGER,
                    damage INTEGER,
                    block INTEGER,
                    hit_count INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_cards_color_type
                    ON cards(color, type_key);

                CREATE TABLE IF NOT EXISTS entity_effect_tags (
                    entity_key TEXT NOT NULL
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    tag TEXT NOT NULL,
                    magnitude REAL NOT NULL DEFAULT 1,
                    source_field TEXT NOT NULL,
                    PRIMARY KEY(entity_key, tag)
                );
                CREATE INDEX IF NOT EXISTS idx_entity_effect_tags_tag
                    ON entity_effect_tags(tag, entity_key);

                CREATE TABLE IF NOT EXISTS relics (
                    entity_key TEXT PRIMARY KEY
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    pool TEXT,
                    rarity_key TEXT
                );

                CREATE TABLE IF NOT EXISTS monsters (
                    entity_key TEXT PRIMARY KEY
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    monster_type TEXT,
                    min_hp INTEGER,
                    max_hp INTEGER,
                    min_hp_ascension INTEGER,
                    max_hp_ascension INTEGER,
                    attack_pattern_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS monster_moves (
                    monster_key TEXT NOT NULL
                        REFERENCES monsters(entity_key) ON DELETE CASCADE,
                    move_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    intent TEXT,
                    damage_normal INTEGER,
                    damage_ascension INTEGER,
                    hit_count INTEGER,
                    block INTEGER,
                    heal INTEGER,
                    powers_json TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY(monster_key, move_id)
                );

                CREATE TABLE IF NOT EXISTS encounters (
                    entity_key TEXT PRIMARY KEY
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    room_type TEXT,
                    act TEXT,
                    is_weak INTEGER NOT NULL DEFAULT 0,
                    tags_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS encounter_monsters (
                    encounter_key TEXT NOT NULL
                        REFERENCES encounters(entity_key) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    monster_external_id TEXT NOT NULL,
                    monster_name TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(encounter_key, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_encounter_monsters_id
                    ON encounter_monsters(monster_external_id, encounter_key);

                CREATE TABLE IF NOT EXISTS events (
                    entity_key TEXT PRIMARY KEY
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    event_type TEXT,
                    act TEXT,
                    preconditions_json TEXT NOT NULL DEFAULT 'null'
                );

                CREATE TABLE IF NOT EXISTS event_pages (
                    event_key TEXT NOT NULL
                        REFERENCES events(entity_key) ON DELETE CASCADE,
                    page_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(event_key, page_id)
                );

                CREATE TABLE IF NOT EXISTS event_options (
                    event_key TEXT NOT NULL
                        REFERENCES events(entity_key) ON DELETE CASCADE,
                    page_id TEXT NOT NULL,
                    option_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(event_key, page_id, option_id)
                );

                CREATE TABLE IF NOT EXISTS acts (
                    entity_key TEXT PRIMARY KEY
                        REFERENCES catalog_entities(entity_key) ON DELETE CASCADE,
                    num_rooms INTEGER
                );

                CREATE TABLE IF NOT EXISTS act_entity_memberships (
                    act_key TEXT NOT NULL
                        REFERENCES acts(entity_key) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    target_external_id TEXT NOT NULL,
                    PRIMARY KEY(act_key, relation_type, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_act_membership_target
                    ON act_entity_memberships(relation_type, target_external_id);

                CREATE TABLE IF NOT EXISTS mechanic_constants (
                    constant_key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_summaries (
                    run_id TEXT PRIMARY KEY,
                    outcome TEXT NOT NULL
                        CHECK(outcome IN ('win', 'loss', 'abandon')),
                    character TEXT NOT NULL,
                    ascension INTEGER NOT NULL DEFAULT 0,
                    final_floor INTEGER NOT NULL DEFAULT 0,
                    final_score INTEGER,
                    started_at TEXT,
                    ended_at TEXT NOT NULL,
                    game_version TEXT,
                    final_deck_json TEXT NOT NULL,
                    final_relics_json TEXT NOT NULL,
                    final_potions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_run_summaries_ended
                    ON run_summaries(ended_at DESC);

                -- Legacy offline-evaluation tables.  The realtime P0 path
                -- intentionally does not write these tables.
                CREATE TABLE IF NOT EXISTS run_states (
                    id TEXT PRIMARY KEY,
                    character TEXT NOT NULL,
                    ascension INTEGER NOT NULL DEFAULT 0,
                    act INTEGER NOT NULL,
                    floor INTEGER NOT NULL,
                    hp INTEGER,
                    max_hp INTEGER,
                    gold INTEGER,
                    energy INTEGER NOT NULL DEFAULT 3,
                    max_potion_slots INTEGER,
                    game_version TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_deck_cards (
                    state_id TEXT NOT NULL
                        REFERENCES run_states(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    card_entity_key TEXT
                        REFERENCES catalog_entities(entity_key) ON DELETE SET NULL,
                    card_name TEXT NOT NULL,
                    upgrades INTEGER NOT NULL DEFAULT 0,
                    quantity INTEGER NOT NULL DEFAULT 1,
                    enchantment_name TEXT,
                    enchantment_amount INTEGER,
                    affliction_name TEXT,
                    affliction_amount INTEGER,
                    PRIMARY KEY(state_id, position)
                );

                CREATE TABLE IF NOT EXISTS run_relics (
                    state_id TEXT NOT NULL
                        REFERENCES run_states(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    relic_entity_key TEXT
                        REFERENCES catalog_entities(entity_key) ON DELETE SET NULL,
                    relic_name TEXT NOT NULL,
                    display_amount INTEGER,
                    stack_count INTEGER NOT NULL DEFAULT 1,
                    status TEXT,
                    PRIMARY KEY(state_id, position)
                );

                CREATE TABLE IF NOT EXISTS run_potions (
                    state_id TEXT NOT NULL
                        REFERENCES run_states(id) ON DELETE CASCADE,
                    slot INTEGER NOT NULL,
                    potion_entity_key TEXT
                        REFERENCES catalog_entities(entity_key) ON DELETE SET NULL,
                    potion_name TEXT NOT NULL,
                    PRIMARY KEY(state_id, slot)
                );

                CREATE TABLE IF NOT EXISTS run_modifiers (
                    state_id TEXT NOT NULL
                        REFERENCES run_states(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    modifier_name TEXT NOT NULL,
                    PRIMARY KEY(state_id, position)
                );

                CREATE TABLE IF NOT EXISTS decision_events (
                    id TEXT PRIMARY KEY,
                    state_id TEXT NOT NULL
                        REFERENCES run_states(id) ON DELETE CASCADE,
                    decision_type TEXT NOT NULL,
                    method TEXT NOT NULL,
                    recommended_option TEXT,
                    decision_status TEXT NOT NULL DEFAULT 'recommend',
                    skip_score REAL NOT NULL DEFAULT 50,
                    skip_eligible INTEGER NOT NULL DEFAULT 0,
                    can_skip INTEGER,
                    can_reroll INTEGER,
                    reward_source TEXT,
                    confidence TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_decision_events_state
                    ON decision_events(state_id, created_at);

                CREATE TABLE IF NOT EXISTS decision_candidates (
                    decision_id TEXT NOT NULL
                        REFERENCES decision_events(id) ON DELETE CASCADE,
                    option_index INTEGER NOT NULL,
                    card_entity_key TEXT
                        REFERENCES catalog_entities(entity_key) ON DELETE SET NULL,
                    card_name TEXT NOT NULL,
                    upgrades INTEGER NOT NULL DEFAULT 0,
                    enchantment_name TEXT,
                    enchantment_amount INTEGER,
                    affliction_name TEXT,
                    affliction_amount INTEGER,
                    score REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    factors_json TEXT NOT NULL,
                    PRIMARY KEY(decision_id, option_index)
                );

                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    decision_id TEXT PRIMARY KEY
                        REFERENCES decision_events(id) ON DELETE CASCADE,
                    chosen_option TEXT,
                    run_won INTEGER,
                    final_floor INTEGER,
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS game_state_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    game_version TEXT,
                    emitted_at TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    processed_at TEXT,
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_id TEXT
                        REFERENCES run_states(id) ON DELETE SET NULL,
                    decision_id TEXT
                        REFERENCES decision_events(id) ON DELETE SET NULL,
                    result_json TEXT,
                    error TEXT,
                    UNIQUE(run_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_game_state_events_latest
                    ON game_state_events(received_at DESC);
                """
            )
            self._ensure_column(
                connection,
                "run_states",
                "max_potion_slots",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "run_deck_cards",
                "enchantment_name",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "run_deck_cards",
                "enchantment_amount",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "run_deck_cards",
                "affliction_name",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "run_deck_cards",
                "affliction_amount",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "run_relics",
                "display_amount",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "run_relics",
                "stack_count",
                "INTEGER NOT NULL DEFAULT 1",
            )
            self._ensure_column(
                connection,
                "run_relics",
                "status",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "decision_events",
                "decision_status",
                "TEXT NOT NULL DEFAULT 'recommend'",
            )
            self._ensure_column(
                connection,
                "decision_events",
                "skip_score",
                "REAL NOT NULL DEFAULT 50",
            )
            self._ensure_column(
                connection,
                "decision_events",
                "skip_eligible",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                connection,
                "decision_events",
                "can_skip",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "decision_events",
                "can_reroll",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "decision_events",
                "reward_source",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "decision_candidates",
                "enchantment_name",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "decision_candidates",
                "enchantment_amount",
                "INTEGER",
            )
            self._ensure_column(
                connection,
                "decision_candidates",
                "affliction_name",
                "TEXT",
            )
            self._ensure_column(
                connection,
                "decision_candidates",
                "affliction_amount",
                "INTEGER",
            )
            connection.execute(
                """
                INSERT INTO schema_metadata(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )
            connection.executemany(
                """
                INSERT INTO data_sources(
                    id, name, base_url, terms_url, usage_status, notes
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    base_url = excluded.base_url,
                    terms_url = excluded.terms_url,
                    usage_status = excluded.usage_status,
                    notes = excluded.notes
                """,
                [
                    (
                        "spire_codex_api",
                        "Spire Codex API",
                        "https://spire-codex.com/api",
                        "https://github.com/ptrlrd/spire-codex/blob/main/API_TERMS.md",
                        "community_api_allowed",
                        "Use within published rate limits; keep source attribution.",
                    ),
                    (
                        "slaythespire_2_reference",
                        "slaythespire-2.com",
                        "https://slaythespire-2.com",
                        "https://slaythespire-2.com/terms-of-service",
                        "reference_only",
                        "Terms prohibit automated scraping without permission.",
                    ),
                ],
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def sync_catalog(self, knowledge_path: str) -> bool:
        """Import a knowledge snapshot when its content hash changes."""
        path = Path(knowledge_path)
        raw_bytes = path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        payload = json.loads(raw_bytes.decode("utf-8"))
        imported_at = _utc_now()

        self.ensure_schema()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'catalog_sha256'"
            ).fetchone()
            catalog_changed = not row or row["value"] != content_hash
            if catalog_changed:
                connection.execute("DELETE FROM catalog_entities")
                for entity_type in CATALOG_TYPES:
                    for ordinal, item in enumerate(
                        payload.get(entity_type, [])
                    ):
                        external_id = str(item.get("id") or "").strip()
                        name = str(
                            item.get("name")
                            or item.get("title")
                            or external_id
                        ).strip()
                        if not external_id or not name:
                            continue
                        entity_key = f"{entity_type}:{external_id}"
                        connection.execute(
                            """
                            INSERT INTO catalog_entities(
                                entity_key, entity_type, external_id, ordinal,
                                name, description, embed_text, payload_json,
                                imported_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                entity_key,
                                entity_type,
                                external_id,
                                ordinal,
                                name,
                                str(item.get("description") or ""),
                                str(item.get("embed_text") or ""),
                                _json_dumps(item),
                                imported_at,
                            ),
                        )
                        if entity_type == "cards":
                            connection.execute(
                                """
                                INSERT INTO cards(
                                    entity_key, color, type_key, rarity_key,
                                    cost, damage, block, hit_count
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    entity_key,
                                    item.get("color"),
                                    item.get("type_key"),
                                    item.get("rarity_key"),
                                    item.get("cost"),
                                    item.get("damage"),
                                    item.get("block"),
                                    item.get("hit_count"),
                                ),
                            )
                        elif entity_type == "relics":
                            connection.execute(
                                """
                                INSERT INTO relics(
                                    entity_key, pool, rarity_key
                                ) VALUES (?, ?, ?)
                                """,
                                (
                                    entity_key,
                                    item.get("pool"),
                                    item.get("rarity_key"),
                                ),
                            )
                        elif entity_type == "monsters":
                            connection.execute(
                                """
                                INSERT INTO monsters(
                                    entity_key, monster_type, min_hp, max_hp,
                                    min_hp_ascension, max_hp_ascension,
                                    attack_pattern_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    entity_key,
                                    item.get("type"),
                                    item.get("min_hp"),
                                    item.get("max_hp"),
                                    item.get("min_hp_ascension"),
                                    item.get("max_hp_ascension"),
                                    _json_dumps(item.get("attack_pattern") or {}),
                                ),
                            )
                            for move_ordinal, move in enumerate(
                                item.get("moves") or []
                            ):
                                damage = move.get("damage") or {}
                                connection.execute(
                                    """
                                    INSERT INTO monster_moves(
                                        monster_key, move_id, ordinal, name,
                                        intent, damage_normal,
                                        damage_ascension, hit_count, block,
                                        heal, powers_json
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        entity_key,
                                        str(move.get("id") or move_ordinal),
                                        move_ordinal,
                                        str(move.get("name") or ""),
                                        move.get("intent"),
                                        damage.get("normal"),
                                        damage.get("ascension"),
                                        damage.get("hit_count"),
                                        move.get("block"),
                                        move.get("heal"),
                                        _json_dumps(move.get("powers") or []),
                                    ),
                                )
                        elif entity_type == "encounters":
                            connection.execute(
                                """
                                INSERT INTO encounters(
                                    entity_key, room_type, act, is_weak,
                                    tags_json
                                ) VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    entity_key,
                                    item.get("room_type"),
                                    item.get("act"),
                                    int(bool(item.get("is_weak"))),
                                    _json_dumps(item.get("tags") or []),
                                ),
                            )
                            for member_ordinal, monster in enumerate(
                                item.get("monsters") or []
                            ):
                                connection.execute(
                                    """
                                    INSERT INTO encounter_monsters(
                                        encounter_key, ordinal,
                                        monster_external_id, monster_name
                                    ) VALUES (?, ?, ?, ?)
                                    """,
                                    (
                                        entity_key,
                                        member_ordinal,
                                        str(monster.get("id") or ""),
                                        str(monster.get("name") or ""),
                                    ),
                                )
                        elif entity_type == "events":
                            connection.execute(
                                """
                                INSERT INTO events(
                                    entity_key, event_type, act,
                                    preconditions_json
                                ) VALUES (?, ?, ?, ?)
                                """,
                                (
                                    entity_key,
                                    item.get("type"),
                                    item.get("act"),
                                    _json_dumps(item.get("preconditions")),
                                ),
                            )
                            self._insert_event_page(
                                connection,
                                entity_key,
                                "__ROOT__",
                                -1,
                                str(item.get("description") or ""),
                                item.get("options") or [],
                            )
                            for page_ordinal, page in enumerate(
                                item.get("pages") or []
                            ):
                                self._insert_event_page(
                                    connection,
                                    entity_key,
                                    str(page.get("id") or page_ordinal),
                                    page_ordinal,
                                    str(page.get("description") or ""),
                                    page.get("options") or [],
                                )
                        elif entity_type == "acts":
                            connection.execute(
                                "INSERT INTO acts(entity_key, num_rooms) VALUES (?, ?)",
                                (entity_key, item.get("num_rooms")),
                            )
                            for relation_type in (
                                "bosses", "ancients", "events", "encounters"
                            ):
                                for member_ordinal, target_id in enumerate(
                                    item.get(relation_type) or []
                                ):
                                    connection.execute(
                                        """
                                        INSERT INTO act_entity_memberships(
                                            act_key, relation_type, ordinal,
                                            target_external_id
                                        ) VALUES (?, ?, ?, ?)
                                        """,
                                        (
                                            entity_key,
                                            relation_type,
                                            member_ordinal,
                                            str(target_id),
                                        ),
                                    )

                connection.execute("DELETE FROM mechanic_constants")
                for constant_key, value in (
                    payload.get("mechanics") or {}
                ).items():
                    connection.execute(
                        """
                        INSERT INTO mechanic_constants(
                            constant_key, value_json, imported_at
                        ) VALUES (?, ?, ?)
                        """,
                        (constant_key, _json_dumps(value), imported_at),
                    )

                connection.execute(
                    """
                    INSERT INTO schema_metadata(key, value)
                    VALUES('catalog_sha256', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (content_hash,),
                )
                connection.execute(
                    """
                    INSERT INTO schema_metadata(key, value)
                    VALUES('catalog_imported_at', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (imported_at,),
                )

            tag_version = connection.execute(
                """
                SELECT value FROM schema_metadata
                WHERE key = 'effect_tag_version'
                """
            ).fetchone()
            if (
                catalog_changed
                or tag_version is None
                or tag_version["value"] != EFFECT_TAG_VERSION
            ):
                self._rebuild_effect_tags(connection)
        return catalog_changed

    @staticmethod
    def _insert_event_page(
        connection: sqlite3.Connection,
        event_key: str,
        page_id: str,
        ordinal: int,
        description: str,
        options: Iterable[Dict],
    ) -> None:
        connection.execute(
            """
            INSERT INTO event_pages(
                event_key, page_id, ordinal, description
            ) VALUES (?, ?, ?, ?)
            """,
            (event_key, page_id, ordinal, description),
        )
        for option_ordinal, option in enumerate(options):
            connection.execute(
                """
                INSERT INTO event_options(
                    event_key, page_id, option_id, ordinal,
                    title, description
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_key,
                    page_id,
                    str(option.get("id") or option_ordinal),
                    option_ordinal,
                    str(option.get("title") or ""),
                    str(option.get("description") or ""),
                ),
            )

    @staticmethod
    def _rebuild_effect_tags(
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute("DELETE FROM entity_effect_tags")
        rows = connection.execute(
            """
            SELECT entity_key, entity_type, payload_json
            FROM catalog_entities
            """
        ).fetchall()
        for row in rows:
            item = json.loads(row["payload_json"])
            for tag, (magnitude, source) in derive_effect_tags(
                row["entity_type"],
                item,
            ).items():
                connection.execute(
                    """
                    INSERT INTO entity_effect_tags(
                        entity_key, tag, magnitude, source_field
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (row["entity_key"], tag, magnitude, source),
                )
        connection.execute(
            """
            INSERT INTO schema_metadata(key, value)
            VALUES('effect_tag_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (EFFECT_TAG_VERSION,),
        )

    def load_catalog_index(self) -> Dict[str, List[Dict]]:
        index = {entity_type: [] for entity_type in CATALOG_TYPES}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT entity_type, payload_json
                FROM catalog_entities
                ORDER BY entity_type, ordinal
                """
            ).fetchall()
        for row in rows:
            item = json.loads(row["payload_json"])
            item["_type"] = row["entity_type"]
            index[row["entity_type"]].append(item)
        return index

    def sync_entity_statistics(self, snapshot_path: str) -> Dict[str, int]:
        """Import filtered, attributable community aggregates."""
        path = Path(snapshot_path)
        raw_bytes = path.read_bytes()
        snapshot_hash = hashlib.sha256(raw_bytes).hexdigest()
        payload = json.loads(raw_bytes.decode("utf-8"))
        source = payload.get("source") or {}
        source_id = str(source.get("id") or "").strip()
        if source_id != "spire_codex_api":
            raise ValueError("Unsupported or unapproved statistics source")

        imported: Dict[str, int] = {}
        self.ensure_schema()
        with self.connect() as connection:
            source_row = connection.execute(
                "SELECT usage_status FROM data_sources WHERE id = ?",
                (source_id,),
            ).fetchone()
            if (
                source_row is None
                or source_row["usage_status"] != "community_api_allowed"
            ):
                raise ValueError("Statistics source is not approved for import")

            for entity_type, items in (payload.get("entities") or {}).items():
                if entity_type not in CATALOG_TYPES or not isinstance(items, dict):
                    continue
                stable_items = _json_dumps(items)
                content_hash = hashlib.sha256(
                    stable_items.encode("utf-8")
                ).hexdigest()
                snapshot_id = (
                    f"{source_id}:{entity_type}:{content_hash[:16]}"
                )
                existing_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM entity_statistics
                    WHERE snapshot_id = ?
                    """,
                    (snapshot_id,),
                ).fetchone()["count"]
                if existing_count:
                    imported[entity_type] = int(existing_count)
                    continue
                connection.execute(
                    """
                    INSERT INTO source_snapshots(
                        id, source_id, entity_type, data_type, source_url,
                        game_version, fetched_at, content_hash, methodology
                    ) VALUES (?, ?, ?, 'aggregate_scores', ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        snapshot_id,
                        source_id,
                        entity_type,
                        str(source.get("endpoints", {}).get(entity_type) or ""),
                        payload.get("game_version"),
                        payload["fetched_at"],
                        content_hash,
                        str(payload.get("methodology") or ""),
                    ),
                )
                connection.execute(
                    "DELETE FROM entity_statistics WHERE snapshot_id = ?",
                    (snapshot_id,),
                )

                count = 0
                for external_id, values in items.items():
                    entity_key = f"{entity_type}:{external_id}"
                    exists = connection.execute(
                        """
                        SELECT 1 FROM catalog_entities
                        WHERE entity_key = ?
                        """,
                        (entity_key,),
                    ).fetchone()
                    if exists is None or not isinstance(values, dict):
                        continue
                    connection.execute(
                        """
                        INSERT INTO entity_statistics(
                            snapshot_id, entity_key, score, elo,
                            picks, wins, win_rate
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            entity_key,
                            values.get("score"),
                            values.get("elo"),
                            int(values.get("picks") or 0),
                            int(values.get("wins") or 0),
                            values.get("win_rate"),
                        ),
                    )
                    count += 1
                imported[entity_type] = count
            connection.execute(
                """
                INSERT INTO schema_metadata(key, value)
                VALUES('community_scores_sha256', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (snapshot_hash,),
            )
        return imported

    def catalog_counts(self) -> Dict[str, int]:
        counts = {entity_type: 0 for entity_type in CATALOG_TYPES}
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT entity_type, COUNT(*) AS count
                FROM catalog_entities
                GROUP BY entity_type
                """
            ).fetchall()
        for row in rows:
            counts[row["entity_type"]] = int(row["count"])
        return counts

    def find_entity(self, entity_type: str, identifier: str) -> Optional[Dict]:
        identifier = str(identifier or "").strip()
        if entity_type not in CATALOG_TYPES or not identifier:
            return None
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT entity_key, payload_json
                FROM catalog_entities
                WHERE entity_type = ?
                  AND (
                    lower(external_id) = lower(?)
                    OR lower(name) = lower(?)
                  )
                LIMIT 1
                """,
                (entity_type, identifier, identifier),
            ).fetchone()
            effect_rows = (
                connection.execute(
                    """
                    SELECT tag, magnitude, source_field
                    FROM entity_effect_tags
                    WHERE entity_key = ?
                    ORDER BY tag
                    """,
                    (row["entity_key"],),
                ).fetchall()
                if row is not None
                else []
            )
        if row is None:
            return None
        item = json.loads(row["payload_json"])
        item["_type"] = entity_type
        item["_entity_key"] = row["entity_key"]
        item["_effect_tags"] = {
            effect["tag"]: float(effect["magnitude"])
            for effect in effect_rows
        }
        item["_effect_tag_sources"] = {
            effect["tag"]: str(effect["source_field"])
            for effect in effect_rows
        }
        return item

    def find_card(self, identifier: str) -> Optional[Dict]:
        return self.find_entity("cards", identifier)

    def find_relic(self, identifier: str) -> Optional[Dict]:
        return self.find_entity("relics", identifier)

    def mechanic_constant(self, identifier: str) -> Optional[object]:
        """Read one imported structured mechanic/risk profile.

        Runtime route scoring may use this small relational profile, while the
        active run graph itself remains in the replaceable checkpoint.
        """
        key = str(identifier or "").strip()
        if not key:
            return None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM mechanic_constants WHERE constant_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value_json"])
        except (TypeError, json.JSONDecodeError):
            return None

    def encounter_profile(self, identifier: str) -> Optional[Dict]:
        encounter = self.find_entity("encounters", identifier)
        if encounter is None:
            return None
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT em.ordinal, em.monster_external_id, em.monster_name,
                       m.monster_type, m.min_hp, m.max_hp,
                       m.min_hp_ascension, m.max_hp_ascension,
                       m.attack_pattern_json
                FROM encounter_monsters AS em
                LEFT JOIN catalog_entities AS ce
                  ON ce.entity_type = 'monsters'
                 AND ce.external_id = em.monster_external_id
                LEFT JOIN monsters AS m ON m.entity_key = ce.entity_key
                WHERE em.encounter_key = ?
                ORDER BY em.ordinal
                """,
                (encounter["_entity_key"],),
            ).fetchall()
            move_rows = connection.execute(
                """
                SELECT em.monster_external_id,
                       mm.ordinal,
                       mm.move_id,
                       mm.intent,
                       mm.damage_normal,
                       mm.damage_ascension,
                       mm.hit_count
                FROM encounter_monsters AS em
                JOIN catalog_entities AS ce
                  ON ce.entity_type = 'monsters'
                 AND ce.external_id = em.monster_external_id
                JOIN monsters AS m ON m.entity_key = ce.entity_key
                JOIN monster_moves AS mm ON mm.monster_key = m.entity_key
                WHERE em.encounter_key = ?
                ORDER BY em.monster_external_id, mm.ordinal, mm.move_id
                """,
                (encounter["_entity_key"],),
            ).fetchall()
        moves_by_monster: Dict[str, List[Dict[str, object]]] = {}
        for move in move_rows:
            monster_id = str(move["monster_external_id"] or "")
            if not monster_id:
                continue
            moves_by_monster.setdefault(monster_id, []).append(
                {
                    "id": move["move_id"],
                    "intent": move["intent"],
                    "damage_normal": move["damage_normal"],
                    "damage_ascension": move["damage_ascension"],
                    "hit_count": move["hit_count"],
                }
            )
        encounter["_monsters"] = [
            {
                "id": row["monster_external_id"],
                "name": row["monster_name"],
                "type": row["monster_type"],
                "min_hp": row["min_hp"],
                "max_hp": row["max_hp"],
                "min_hp_ascension": row["min_hp_ascension"],
                "max_hp_ascension": row["max_hp_ascension"],
                "attack_pattern": json.loads(
                    row["attack_pattern_json"] or "{}"
                ),
                # Attack-pattern JSON is a state-machine description, not a
                # numeric combat estimate.  Keep it for callers that need
                # the raw fact, but expose normalized static moves for route
                # pressure calculations.
                "moves": moves_by_monster.get(
                    str(row["monster_external_id"] or ""),
                    [],
                ),
            }
            for row in rows
        ]
        return encounter

    def encounter_pool_expectations(
        self,
        boss_identifiers: Iterable[str],
        *,
        ascension: int = 0,
    ) -> Dict[str, Dict[str, float]]:
        """Return static act encounter expectations keyed by room class.

        The active run never tells us which normal monster or elite will be
        faced.  This query follows the catalog's Act -> encounter membership
        instead, aggregates static monster HP/attack records, and returns only
        an expected pressure profile.  It does not read or write any run
        history table.
        """
        identifiers = tuple(
            dict.fromkeys(
                str(identifier).strip()
                for identifier in boss_identifiers
                if str(identifier).strip()
            )
        )
        if not identifiers:
            return {}
        placeholders = ", ".join("?" for _ in identifiers)
        # The catalog imports both normal and ascension values.  Select the
        # appropriate column explicitly rather than always preferring the
        # ascension field, so an A0 route does not silently inherit harder
        # combat expectations just because that optional field exists.
        # The imported static rules name Tough Enemies at A8 and Deadly
        # Enemies at A9.  HP and damage therefore have separate gates; A1
        # must never silently use either advanced value.
        use_ascension_hp = int(ascension) >= 8
        use_ascension_damage = int(ascension) >= 9
        if use_ascension_hp:
            hp_expression = """
                COALESCE(
                    monsters.max_hp_ascension,
                    monsters.max_hp,
                    monsters.min_hp_ascension,
                    monsters.min_hp,
                    0
                )
            """
        else:
            hp_expression = """
                COALESCE(
                    monsters.max_hp,
                    monsters.min_hp,
                    monsters.max_hp_ascension,
                    monsters.min_hp_ascension,
                    0
                )
            """
        if use_ascension_damage:
            damage_expression = """
                COALESCE(
                    monster_moves.damage_ascension,
                    monster_moves.damage_normal,
                    0
                )
            """
        else:
            damage_expression = """
                COALESCE(
                    monster_moves.damage_normal,
                    monster_moves.damage_ascension,
                    0
                )
            """
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT encounter_entity.external_id AS encounter_id,
                       encounters.room_type AS room_type,
                       encounter_monsters.monster_external_id AS monster_id,
                       {hp_expression} AS max_hp,
                       MAX({damage_expression} * COALESCE(
                           monster_moves.hit_count,
                           1
                       )) AS peak_attack
                FROM act_entity_memberships AS boss_membership
                JOIN act_entity_memberships AS encounter_membership
                  ON encounter_membership.act_key = boss_membership.act_key
                 AND encounter_membership.relation_type = 'encounters'
                JOIN catalog_entities AS encounter_entity
                  ON encounter_entity.entity_type = 'encounters'
                 AND encounter_entity.external_id = encounter_membership.target_external_id
                JOIN encounters ON encounters.entity_key = encounter_entity.entity_key
                LEFT JOIN encounter_monsters
                  ON encounter_monsters.encounter_key = encounter_entity.entity_key
                LEFT JOIN catalog_entities AS monster_entity
                  ON monster_entity.entity_type = 'monsters'
                 AND monster_entity.external_id = encounter_monsters.monster_external_id
                LEFT JOIN monsters ON monsters.entity_key = monster_entity.entity_key
                LEFT JOIN monster_moves ON monster_moves.monster_key = monsters.entity_key
                WHERE boss_membership.relation_type = 'bosses'
                  AND boss_membership.target_external_id IN ({placeholders})
                GROUP BY encounter_entity.external_id,
                         encounters.room_type,
                         encounter_monsters.monster_external_id
                """,
                identifiers,
            ).fetchall()
        encounters: Dict[str, Dict[str, object]] = {}
        identifier_set = set(identifiers)
        for row in rows:
            encounter_id = str(row["encounter_id"] or "")
            if not encounter_id:
                continue
            room_type = str(row["room_type"] or "").upper()
            if encounter_id in identifier_set or "BOSS" in room_type:
                kind = "BOSS"
            elif "ELITE" in room_type or "ELITE" in encounter_id.upper():
                kind = "ELITE"
            else:
                kind = "MONSTER"
            bucket = encounters.setdefault(encounter_id, {
                "kind": kind,
                "hp": 0.0,
                "attack": 0.0,
                "monsters": set(),
            })
            monster_id = str(row["monster_id"] or "")
            if monster_id and monster_id not in bucket["monsters"]:
                bucket["monsters"].add(monster_id)
                bucket["hp"] += float(row["max_hp"] or 0)
                bucket["attack"] += float(row["peak_attack"] or 0)

        grouped: Dict[str, List[Dict[str, object]]] = {}
        for value in encounters.values():
            grouped.setdefault(str(value["kind"]), []).append(value)
        result: Dict[str, Dict[str, float]] = {}
        for kind, values in grouped.items():
            count = len(values)
            if count == 0:
                continue
            result[kind] = {
                "sample_count": float(count),
                "average_hp": round(sum(float(value["hp"]) for value in values) / count, 4),
                "average_attack": round(sum(float(value["attack"]) for value in values) / count, 4),
                "average_enemy_count": round(
                    sum(len(value["monsters"]) for value in values) / count,
                    4,
                ),
            }
        return result

    def event_tree(self, identifier: str) -> Optional[Dict]:
        event = self.find_entity("events", identifier)
        if event is None:
            return None
        with self.connect() as connection:
            pages = connection.execute(
                """
                SELECT page_id, ordinal, description
                FROM event_pages
                WHERE event_key = ?
                ORDER BY ordinal, page_id
                """,
                (event["_entity_key"],),
            ).fetchall()
            options = connection.execute(
                """
                SELECT page_id, option_id, ordinal, title, description
                FROM event_options
                WHERE event_key = ?
                ORDER BY page_id, ordinal
                """,
                (event["_entity_key"],),
            ).fetchall()
        options_by_page: Dict[str, List[Dict]] = {}
        for row in options:
            options_by_page.setdefault(row["page_id"], []).append(
                {
                    "id": row["option_id"],
                    "title": row["title"],
                    "description": row["description"],
                }
            )
        event["_pages"] = [
            {
                "id": row["page_id"],
                "description": row["description"],
                "options": options_by_page.get(row["page_id"], []),
            }
            for row in pages
        ]
        return event

    def act_members(
        self,
        identifier: str,
        relation_type: Optional[str] = None,
    ) -> Optional[Dict[str, List[str]]]:
        act = self.find_entity("acts", identifier)
        if act is None:
            return None
        query = """
            SELECT relation_type, target_external_id
            FROM act_entity_memberships
            WHERE act_key = ?
        """
        parameters: List[object] = [act["_entity_key"]]
        if relation_type is not None:
            query += " AND relation_type = ?"
            parameters.append(relation_type)
        query += " ORDER BY relation_type, ordinal"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result: Dict[str, List[str]] = {}
        for row in rows:
            result.setdefault(row["relation_type"], []).append(
                row["target_external_id"]
            )
        return result

    def find_latest_entity_stat(
        self,
        entity_type: str,
        identifier: str,
    ) -> Optional[Dict]:
        entity = self.find_entity(entity_type, identifier)
        if entity is None:
            return None
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    es.score, es.elo, es.picks, es.wins, es.win_rate,
                    ss.id AS snapshot_id, ss.source_url, ss.fetched_at,
                    ss.game_version, ss.methodology,
                    ds.name AS source_name
                FROM entity_statistics AS es
                JOIN source_snapshots AS ss ON ss.id = es.snapshot_id
                JOIN data_sources AS ds ON ds.id = ss.source_id
                WHERE es.entity_key = ?
                ORDER BY ss.fetched_at DESC
                LIMIT 1
                """,
                (entity["_entity_key"],),
            ).fetchone()
        return dict(row) if row is not None else None

    def statistics_status(self) -> Dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT ss.entity_type, COUNT(DISTINCT es.entity_key) AS count
                FROM entity_statistics AS es
                JOIN source_snapshots AS ss ON ss.id = es.snapshot_id
                GROUP BY ss.entity_type
                """
            ).fetchall()
        return {row["entity_type"]: int(row["count"]) for row in rows}

    def save_run_summary(self, summary: Dict) -> str:
        """Upsert the single long-lived record for a completed run."""
        outcome = str(summary.get("outcome") or "").strip().lower()
        if outcome not in {"win", "loss", "abandon"}:
            raise ValueError("run summary outcome must be win/loss/abandon")
        run_id = str(summary.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("run summary requires run_id")
        ended_at = str(summary.get("ended_at") or "").strip()
        if not ended_at:
            raise ValueError("run summary requires ended_at")
        character = str(summary.get("character") or "").strip()
        if not character:
            raise ValueError("run summary requires character")

        now = _utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO run_summaries(
                    run_id, outcome, character, ascension, final_floor,
                    final_score, started_at, ended_at, game_version,
                    final_deck_json, final_relics_json,
                    final_potions_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    outcome = excluded.outcome,
                    character = excluded.character,
                    ascension = excluded.ascension,
                    final_floor = excluded.final_floor,
                    final_score = excluded.final_score,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    game_version = excluded.game_version,
                    final_deck_json = excluded.final_deck_json,
                    final_relics_json = excluded.final_relics_json,
                    final_potions_json = excluded.final_potions_json,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    outcome,
                    character,
                    int(summary.get("ascension", 0)),
                    int(summary.get("final_floor", 0)),
                    summary.get("final_score"),
                    summary.get("started_at"),
                    ended_at,
                    summary.get("game_version"),
                    _json_dumps(summary.get("final_deck", [])),
                    _json_dumps(summary.get("final_relics", [])),
                    _json_dumps(summary.get("final_potions", [])),
                    now,
                    now,
                ),
            )
        return run_id

    def load_run_summary(self, run_id: str) -> Optional[Dict]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM run_summaries
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        summary = dict(row)
        summary["final_deck"] = json.loads(
            summary.pop("final_deck_json")
        )
        summary["final_relics"] = json.loads(
            summary.pop("final_relics_json")
        )
        summary["final_potions"] = json.loads(
            summary.pop("final_potions_json")
        )
        return summary

    def run_summary_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM run_summaries"
            ).fetchone()
        return int(row["count"])

    def save_run_state(self, state: Dict) -> str:
        state_id = str(state.get("id") or uuid.uuid4().hex)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO run_states(
                    id, character, ascension, act, floor, hp, max_hp, gold,
                    energy, max_potion_slots, game_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state_id,
                    state["character"],
                    int(state.get("ascension", 0)),
                    int(state["act"]),
                    int(state["floor"]),
                    state.get("hp"),
                    state.get("max_hp"),
                    state.get("gold"),
                    int(state.get("energy", 3)),
                    state.get("max_potion_slots"),
                    state.get("game_version"),
                    _utc_now(),
                ),
            )
            for position, deck_card in enumerate(state.get("deck", [])):
                card = self.find_card(deck_card["card"])
                connection.execute(
                    """
                    INSERT INTO run_deck_cards(
                        state_id, position, card_entity_key, card_name,
                        upgrades, quantity, enchantment_name,
                        enchantment_amount, affliction_name,
                        affliction_amount
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state_id,
                        position,
                        card.get("_entity_key") if card else None,
                        deck_card["card"],
                        int(deck_card.get("upgrades", 0)),
                        int(deck_card.get("count", 1)),
                        deck_card.get("enchantment"),
                        deck_card.get("enchantment_amount"),
                        deck_card.get("affliction"),
                        deck_card.get("affliction_amount"),
                    ),
                )
            relic_states = state.get("relic_states") or [
                {"relic": relic_name}
                for relic_name in state.get("relics", [])
            ]
            for position, relic_state in enumerate(relic_states):
                relic_name = relic_state["relic"]
                relic = self.find_relic(relic_name)
                connection.execute(
                    """
                    INSERT INTO run_relics(
                        state_id, position, relic_entity_key, relic_name,
                        display_amount, stack_count, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state_id,
                        position,
                        relic.get("_entity_key") if relic else None,
                        relic_name,
                        relic_state.get("display_amount"),
                        int(relic_state.get("stack_count", 1)),
                        relic_state.get("status"),
                    ),
                )
            for potion_state in state.get("potions", []):
                potion_name = potion_state["potion"]
                potion = self.find_entity("potions", potion_name)
                connection.execute(
                    """
                    INSERT INTO run_potions(
                        state_id, slot, potion_entity_key, potion_name
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        state_id,
                        int(potion_state["slot"]),
                        potion.get("_entity_key") if potion else None,
                        potion_name,
                    ),
                )
            for position, modifier_name in enumerate(
                state.get("modifiers", [])
            ):
                connection.execute(
                    """
                    INSERT INTO run_modifiers(
                        state_id, position, modifier_name
                    ) VALUES (?, ?, ?)
                    """,
                    (state_id, position, modifier_name),
                )
        return state_id

    def save_card_reward_decision(
        self,
        state_id: str,
        recommendations: Iterable[Dict],
        recommended_option: Optional[str],
        confidence: str,
        method: str,
        decision_status: str = "recommend",
        skip_score: float = 50.0,
        skip_eligible: bool = False,
        can_skip: Optional[bool] = None,
        can_reroll: Optional[bool] = None,
        reward_source: Optional[str] = None,
    ) -> str:
        decision_id = uuid.uuid4().hex
        recommendations = list(recommendations)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO decision_events(
                    id, state_id, decision_type, method, recommended_option,
                    decision_status, skip_score, skip_eligible, can_skip,
                    can_reroll, reward_source, confidence, created_at
                ) VALUES (
                    ?, ?, 'card_reward', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    decision_id,
                    state_id,
                    method,
                    recommended_option,
                    decision_status,
                    float(skip_score),
                    int(skip_eligible),
                    int(can_skip) if can_skip is not None else None,
                    int(can_reroll) if can_reroll is not None else None,
                    reward_source,
                    confidence,
                    _utc_now(),
                ),
            )
            for fallback_index, recommendation in enumerate(recommendations):
                option_index = int(
                    recommendation.get("option_index", fallback_index)
                )
                card = self.find_card(recommendation["card"])
                connection.execute(
                    """
                    INSERT INTO decision_candidates(
                        decision_id, option_index, card_entity_key, card_name,
                        upgrades, enchantment_name, enchantment_amount,
                        affliction_name, affliction_amount, score, rank,
                        factors_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        option_index,
                        card.get("_entity_key") if card else None,
                        recommendation["card"],
                        int(recommendation.get("upgrades", 0)),
                        recommendation.get("enchantment"),
                        recommendation.get("enchantment_amount"),
                        recommendation.get("affliction"),
                        recommendation.get("affliction_amount"),
                        float(recommendation["score"]),
                        int(recommendation["rank"]),
                        _json_dumps(recommendation.get("factors", [])),
                    ),
                )
        return decision_id

    def load_decision(self, decision_id: str) -> Optional[Dict]:
        with self.connect() as connection:
            event = connection.execute(
                """
                SELECT id, state_id, decision_type, method,
                       recommended_option, decision_status, skip_score,
                       skip_eligible, can_skip, can_reroll, reward_source,
                       confidence, created_at
                FROM decision_events
                WHERE id = ?
                """,
                (decision_id,),
            ).fetchone()
            if event is None:
                return None
            candidates = connection.execute(
                """
                SELECT card_name, upgrades, enchantment_name,
                       enchantment_amount, affliction_name,
                       affliction_amount, score, rank, factors_json
                FROM decision_candidates
                WHERE decision_id = ?
                ORDER BY rank, option_index
                """,
                (decision_id,),
            ).fetchall()
            outcome = connection.execute(
                """
                SELECT chosen_option, run_won, final_floor, recorded_at
                FROM decision_outcomes
                WHERE decision_id = ?
                """,
                (decision_id,),
            ).fetchone()
        result = dict(event)
        result["skip_eligible"] = bool(result["skip_eligible"])
        result["can_skip"] = (
            bool(result["can_skip"])
            if result["can_skip"] is not None
            else None
        )
        result["can_reroll"] = (
            bool(result["can_reroll"])
            if result["can_reroll"] is not None
            else None
        )
        result["recommendations"] = [
            {
                "card": row["card_name"],
                "upgrades": int(row["upgrades"]),
                "enchantment": row["enchantment_name"],
                "enchantment_amount": row["enchantment_amount"],
                "affliction": row["affliction_name"],
                "affliction_amount": row["affliction_amount"],
                "score": float(row["score"]),
                "rank": int(row["rank"]),
                "factors": json.loads(row["factors_json"]),
            }
            for row in candidates
        ]
        result["outcome"] = None
        if outcome is not None:
            result["outcome"] = {
                "chosen_option": outcome["chosen_option"],
                "run_won": (
                    bool(outcome["run_won"])
                    if outcome["run_won"] is not None
                    else None
                ),
                "final_floor": outcome["final_floor"],
                "recorded_at": outcome["recorded_at"],
            }
        return result

    def save_decision_outcome(
        self,
        decision_id: str,
        chosen_option: Optional[str] = None,
        run_won: Optional[bool] = None,
        final_floor: Optional[int] = None,
    ) -> None:
        with self.connect() as connection:
            if chosen_option:
                normalized_choice = chosen_option.strip().lower()
                allowed = {"skip", "跳过"}
                rows = connection.execute(
                    """
                    SELECT dc.card_name, ce.external_id
                    FROM decision_candidates AS dc
                    LEFT JOIN catalog_entities AS ce
                      ON ce.entity_key = dc.card_entity_key
                    WHERE dc.decision_id = ?
                    """,
                    (decision_id,),
                ).fetchall()
                for row in rows:
                    allowed.add(str(row["card_name"]).strip().lower())
                    if row["external_id"]:
                        allowed.add(str(row["external_id"]).strip().lower())
                if normalized_choice not in allowed:
                    raise ValueError(
                        "chosen_option must be one of the recorded candidates or skip"
                    )
            connection.execute(
                """
                INSERT INTO decision_outcomes(
                    decision_id, chosen_option, run_won, final_floor, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    chosen_option = COALESCE(
                        excluded.chosen_option,
                        decision_outcomes.chosen_option
                    ),
                    run_won = COALESCE(
                        excluded.run_won,
                        decision_outcomes.run_won
                    ),
                    final_floor = COALESCE(
                        excluded.final_floor,
                        decision_outcomes.final_floor
                    ),
                    recorded_at = excluded.recorded_at
                """,
                (
                    decision_id,
                    chosen_option,
                    int(run_won) if run_won is not None else None,
                    final_floor,
                    _utc_now(),
                ),
            )

    @staticmethod
    def _decode_game_state_event(row: sqlite3.Row) -> Dict:
        result = {
            "event_id": row["event_id"],
            "run_id": row["run_id"],
            "sequence": int(row["sequence"]),
            "event_type": row["event_type"],
            "schema_version": int(row["schema_version"]),
            "source": row["source"],
            "game_version": row["game_version"],
            "emitted_at": row["emitted_at"],
            "received_at": row["received_at"],
            "processed_at": row["processed_at"],
            "status": row["status"],
            "state_id": row["state_id"],
            "decision_id": row["decision_id"],
            "error": row["error"],
            "payload": json.loads(row["payload_json"]),
            "result": None,
        }
        if row["result_json"]:
            result["result"] = json.loads(row["result_json"])
        return result

    def claim_game_state_event(self, event: Dict) -> tuple[bool, Dict]:
        """Persist an immutable observation and reject identity collisions."""
        payload_json = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        content_hash = hashlib.sha256(
            payload_json.encode("utf-8")
        ).hexdigest()
        received_at = _utc_now()
        with self.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO game_state_events(
                        event_id, run_id, sequence, event_type,
                        schema_version, source, game_version, emitted_at,
                        received_at, content_hash, payload_json, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'received')
                    """,
                    (
                        event["event_id"],
                        event["run_id"],
                        int(event["sequence"]),
                        event["event_type"],
                        int(event["schema_version"]),
                        event["source"],
                        event.get("game_version"),
                        event["emitted_at"],
                        received_at,
                        content_hash,
                        payload_json,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT *
                    FROM game_state_events
                    WHERE event_id = ?
                    """,
                    (event["event_id"],),
                ).fetchone()
                return True, self._decode_game_state_event(row)
            except sqlite3.IntegrityError:
                row = connection.execute(
                    """
                    SELECT *
                    FROM game_state_events
                    WHERE event_id = ?
                       OR (run_id = ? AND sequence = ?)
                    LIMIT 1
                    """,
                    (
                        event["event_id"],
                        event["run_id"],
                        int(event["sequence"]),
                    ),
                ).fetchone()
                if row is None:
                    raise
                if row["content_hash"] != content_hash:
                    raise ValueError(
                        "Event identity collision: the same event ID or "
                        "run sequence was reused with a different payload"
                    )
                return False, self._decode_game_state_event(row)

    def complete_game_state_event(
        self,
        event_id: str,
        status: str,
        result: Dict,
        state_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE game_state_events
                SET status = ?,
                    processed_at = ?,
                    state_id = ?,
                    decision_id = ?,
                    result_json = ?,
                    error = ?
                WHERE event_id = ?
                """,
                (
                    status,
                    _utc_now(),
                    state_id,
                    decision_id,
                    _json_dumps(result),
                    error,
                    event_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Game-state event was not claimed")

    def load_game_state_event(self, event_id: str) -> Optional[Dict]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM game_state_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
        return self._decode_game_state_event(row) if row else None

    def load_latest_game_state_event(self) -> Optional[Dict]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM game_state_events
                ORDER BY received_at DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
        return self._decode_game_state_event(row) if row else None

    def game_state_event_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM game_state_events"
            ).fetchone()
        return int(row["count"])

    def load_labeled_card_reward_decisions(self) -> List[Dict]:
        """Return completed card-reward traces for offline policy evaluation."""
        with self.connect() as connection:
            events = connection.execute(
                """
                SELECT
                    de.id, de.method, de.created_at,
                    outcome.chosen_option, outcome.run_won,
                    outcome.final_floor
                FROM decision_events AS de
                JOIN decision_outcomes AS outcome
                  ON outcome.decision_id = de.id
                WHERE de.decision_type = 'card_reward'
                  AND outcome.chosen_option IS NOT NULL
                ORDER BY de.created_at
                """
            ).fetchall()
            output = []
            for event in events:
                candidates = connection.execute(
                    """
                    SELECT
                        dc.card_name, dc.score, dc.rank, dc.factors_json,
                        ce.external_id
                    FROM decision_candidates AS dc
                    LEFT JOIN catalog_entities AS ce
                      ON ce.entity_key = dc.card_entity_key
                    WHERE dc.decision_id = ?
                    ORDER BY dc.option_index
                    """,
                    (event["id"],),
                ).fetchall()
                output.append(
                    {
                        **dict(event),
                        "run_won": (
                            bool(event["run_won"])
                            if event["run_won"] is not None
                            else None
                        ),
                        "candidates": [
                            {
                                "card": row["card_name"],
                                "card_id": row["external_id"],
                                "score": float(row["score"]),
                                "rank": int(row["rank"]),
                                "factors": json.loads(row["factors_json"]),
                            }
                            for row in candidates
                        ],
                    }
                )
        return output
