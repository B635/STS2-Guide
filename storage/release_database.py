"""Build and install the release-bound SQLite database template.

The release template is produced only from checked-in structured snapshots.
At runtime it is copied as a whole.  When the template identity changes, the
old database contributes only final ``run_summaries``; no transient or legacy
decision history crosses the atomic replacement boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from storage.effect_tags import EFFECT_TAG_VERSION
from storage.relational import RelationalRepository, SCHEMA_VERSION


RELEASE_IDENTITY_KEYS = (
    "schema_version",
    "effect_tag_version",
    "catalog_sha256",
    "community_scores_sha256",
    "release_catalog_profile",
)
RELEASE_CATALOG_PROFILE = "public-beta-card-route-v1"
PUBLIC_CATALOG_TYPES = frozenset(
    {
        "characters",
        "cards",
        "relics",
        "potions",
        "monsters",
        "encounters",
        "acts",
    }
)
_PUBLIC_PAYLOAD_FIELDS = {
    "characters": ("id", "name"),
    "cards": (
        "id",
        "name",
        "cost",
        "is_x_cost",
        "is_x_star_cost",
        "star_cost",
        "type_key",
        "rarity_key",
        "color",
        "target",
        "damage",
        "block",
        "hit_count",
        "cards_draw",
        "energy_gain",
        "hp_loss",
        "upgrade",
    ),
    "relics": ("id", "name", "pool", "rarity_key"),
    "potions": ("id", "name", "pool", "rarity_key"),
    "monsters": (
        "id",
        "name",
        "type",
        "min_hp",
        "max_hp",
        "min_hp_ascension",
        "max_hp_ascension",
    ),
    "encounters": (
        "id",
        "name",
        "room_type",
        "act",
        "is_weak",
        "tags",
    ),
    "acts": ("id", "name", "num_rooms"),
}
PROHIBITED_HISTORY_TABLES = (
    "run_states",
    "run_deck_cards",
    "run_relics",
    "run_potions",
    "run_modifiers",
    "decision_events",
    "decision_candidates",
    "decision_outcomes",
    "game_state_events",
)
RUN_SUMMARY_COLUMNS = (
    "run_id",
    "outcome",
    "character",
    "ascension",
    "final_floor",
    "final_score",
    "started_at",
    "ended_at",
    "game_version",
    "final_deck_json",
    "final_relics_json",
    "final_potions_json",
    "created_at",
    "updated_at",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseDatabaseError(RuntimeError):
    """The release database cannot be safely built or installed."""


@dataclass(frozen=True)
class ReleaseDatabaseResult:
    action: str
    path: Path
    database_sha256: str
    metadata: Mapping[str, str]
    migrated_run_summaries: int = 0

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "path": str(self.path),
            "database_sha256": self.database_sha256,
            "metadata": dict(self.metadata),
            "migrated_run_summaries": self.migrated_run_summaries,
        }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _temporary_peer(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )


def _sidecar_paths(path: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{path}{suffix}") for suffix in ("-wal", "-shm", "-journal"))


def _remove_temporary_database(path: Path) -> None:
    path.unlink(missing_ok=True)
    for sidecar in _sidecar_paths(path):
        sidecar.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for fsync; ``r+b`` preserves the
    # already-validated SQLite bytes while making the durability call valid.
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = connection.execute(
            "SELECT key, value FROM schema_metadata"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ReleaseDatabaseError(
            "database does not contain readable release metadata"
        ) from exc
    return {str(row["key"]): str(row["value"]) for row in rows}


def _validate_hash_metadata(metadata: Mapping[str, str]) -> None:
    for key in ("catalog_sha256", "community_scores_sha256"):
        value = metadata.get(key, "")
        if not _SHA256_RE.fullmatch(value):
            raise ReleaseDatabaseError(
                f"release database has invalid or missing {key}"
            )


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    try:
        row = connection.execute(
            f'SELECT COUNT(*) AS count FROM "{table}"'
        ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise ReleaseDatabaseError(
            f"release database is missing required table {table}"
        ) from exc
    return int(row["count"])


def _validate_integrity(connection: sqlite3.Connection) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise ReleaseDatabaseError("SQLite integrity_check failed")
    foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_rows:
        raise ReleaseDatabaseError("SQLite foreign_key_check failed")


def _validate_template(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ReleaseDatabaseError(f"release database template is missing: {path}")
    try:
        with _readonly_connection(path) as connection:
            _validate_integrity(connection)
            metadata = _read_metadata(connection)
            _validate_hash_metadata(metadata)
            if metadata.get("schema_version") != str(SCHEMA_VERSION):
                raise ReleaseDatabaseError(
                    "release database schema version does not match this build"
                )
            if metadata.get("effect_tag_version") != EFFECT_TAG_VERSION:
                raise ReleaseDatabaseError(
                    "release database effect-tag version does not match this build"
                )
            if metadata.get("release_catalog_profile") != RELEASE_CATALOG_PROFILE:
                raise ReleaseDatabaseError(
                    "release database does not use the public minimal catalog profile"
                )
            for table in ("run_summaries", *PROHIBITED_HISTORY_TABLES):
                if _table_count(connection, table):
                    raise ReleaseDatabaseError(
                        f"release template contains forbidden history in {table}"
                    )
    except sqlite3.DatabaseError as exc:
        raise ReleaseDatabaseError("release database template is unreadable") from exc
    return metadata


def _database_identity(metadata: Mapping[str, str]) -> tuple[str | None, ...]:
    return tuple(metadata.get(key) for key in RELEASE_IDENTITY_KEYS)


def _read_runtime_state(
    runtime_path: Path,
) -> tuple[dict[str, str], list[tuple], bool]:
    try:
        with _readonly_connection(runtime_path) as connection:
            _validate_integrity(connection)
            metadata = _read_metadata(connection)
            # A pre-Public-Beta runtime DB may not yet have release hashes or
            # the minimal-catalog profile.  It is never trusted as static
            # data: only an exact, validated run_summaries table is read and
            # copied into the new immutable template.
            columns = tuple(
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(run_summaries)"
                ).fetchall()
            )
            if columns != RUN_SUMMARY_COLUMNS:
                raise ReleaseDatabaseError(
                    "runtime run_summaries schema cannot be migrated safely"
                )
            select_columns = ", ".join(
                f'"{column}"' for column in RUN_SUMMARY_COLUMNS
            )
            summaries = [
                tuple(row[column] for column in RUN_SUMMARY_COLUMNS)
                for row in connection.execute(
                    f"SELECT {select_columns} FROM run_summaries "
                    "ORDER BY run_id"
                ).fetchall()
            ]
            has_forbidden_history = any(
                _table_count(connection, table)
                for table in PROHIBITED_HISTORY_TABLES
            )
    except sqlite3.DatabaseError as exc:
        raise ReleaseDatabaseError("runtime database is unreadable") from exc
    return metadata, summaries, has_forbidden_history


def _assert_no_sidecars(path: Path) -> None:
    present = [str(sidecar) for sidecar in _sidecar_paths(path) if sidecar.exists()]
    if present:
        raise ReleaseDatabaseError(
            "runtime database has active or stale SQLite sidecars: "
            + ", ".join(present)
        )


def _copy_template(template_path: Path, destination: Path) -> None:
    shutil.copyfile(template_path, destination)
    _fsync_file(destination)


def _normalize_release_database(
    path: Path,
    *,
    imported_at: str,
) -> None:
    """Remove wall-clock build variance and compact to stable page order."""

    with closing(sqlite3.connect(str(path))) as connection:
        connection.execute(
            "UPDATE catalog_entities SET imported_at = ?",
            (imported_at,),
        )
        connection.execute(
            "UPDATE mechanic_constants SET imported_at = ?",
            (imported_at,),
        )
        connection.execute(
            """
            INSERT INTO schema_metadata(key, value)
            VALUES('catalog_imported_at', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (imported_at,),
        )
        connection.commit()
        connection.execute("VACUUM")


def _minimal_payload(entity_type: str, item: Mapping[str, object]) -> dict:
    payload = {
        field: item[field]
        for field in _PUBLIC_PAYLOAD_FIELDS[entity_type]
        if field in item and item[field] is not None
    }
    if entity_type == "cards":
        # The runtime cannot observe Osty's HP.  Materialize this one data
        # dependency as a boolean before prose is removed so the Necrobinder
        # adapter still fails transparently instead of silently overrating it.
        description = " ".join(
            (
                str(item.get("description") or ""),
                str(item.get("upgrade_description") or ""),
            )
        )
        if str(item.get("id") or "").upper() == "UNLEASH" or re.search(
            r"奥斯提.{0,18}当前生命值|"
            r"osty.{0,18}current.{0,8}(hp|health)",
            description,
            flags=re.IGNORECASE,
        ):
            payload["requires_osty_current_hp"] = True
    return payload


def _minimize_release_catalog(connection: sqlite3.Connection) -> None:
    """Keep only structured facts used by enabled Card + Route policies."""

    connection.execute("PRAGMA foreign_keys = ON")
    placeholders = ", ".join("?" for _ in PUBLIC_CATALOG_TYPES)
    connection.execute(
        f"DELETE FROM catalog_entities WHERE entity_type NOT IN ({placeholders})",
        tuple(sorted(PUBLIC_CATALOG_TYPES)),
    )
    connection.execute(
        "DELETE FROM act_entity_memberships "
        "WHERE relation_type NOT IN ('bosses', 'encounters')"
    )
    connection.execute(
        "DELETE FROM mechanic_constants WHERE constant_key <> ?",
        ("route_risk_profile_v1",),
    )
    connection.execute("DELETE FROM source_snapshots WHERE entity_type <> 'cards'")
    connection.execute(
        "DELETE FROM data_sources WHERE id NOT IN "
        "(SELECT DISTINCT source_id FROM source_snapshots)"
    )
    connection.execute("UPDATE monster_moves SET name = '', powers_json = '[]'")
    connection.execute("UPDATE monsters SET attack_pattern_json = '{}'")
    connection.execute("UPDATE encounter_monsters SET monster_name = ''")

    rows = connection.execute(
        "SELECT entity_key, entity_type, payload_json FROM catalog_entities"
    ).fetchall()
    for row in rows:
        entity_type = str(row["entity_type"])
        try:
            item = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReleaseDatabaseError(
                f"catalog payload is invalid for {row['entity_key']}"
            ) from exc
        payload = _minimal_payload(entity_type, item)
        connection.execute(
            """
            UPDATE catalog_entities
            SET description = '', embed_text = '', payload_json = ?
            WHERE entity_key = ?
            """,
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                row["entity_key"],
            ),
        )
    connection.execute(
        """
        INSERT INTO schema_metadata(key, value)
        VALUES('release_catalog_profile', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (RELEASE_CATALOG_PROFILE,),
    )


def build_release_database(
    template_path: str | Path,
    catalog_path: str | Path,
    community_scores_path: str | Path,
) -> ReleaseDatabaseResult:
    """Create a clean template and publish it with one atomic replacement."""
    template = Path(template_path)
    catalog = Path(catalog_path)
    community = Path(community_scores_path)
    if not catalog.is_file():
        raise ReleaseDatabaseError(f"catalog snapshot is missing: {catalog}")
    if not community.is_file():
        raise ReleaseDatabaseError(
            f"community score snapshot is missing: {community}"
        )

    try:
        community_payload = json.loads(community.read_text(encoding="utf-8"))
        imported_at = str(community_payload["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReleaseDatabaseError(
            "community score snapshot has no deterministic fetched_at"
        ) from exc

    template.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_peer(template)
    try:
        repository = RelationalRepository(str(temporary))
        repository.ensure_schema()
        repository.sync_catalog(str(catalog))
        repository.sync_entity_statistics(str(community))
        with repository.connect() as connection:
            _minimize_release_catalog(connection)
        _normalize_release_database(temporary, imported_at=imported_at)

        metadata = _validate_template(temporary)
        if metadata["catalog_sha256"] != _sha256_path(catalog):
            raise ReleaseDatabaseError("catalog hash metadata does not match input")
        if metadata["community_scores_sha256"] != _sha256_path(community):
            raise ReleaseDatabaseError(
                "community score hash metadata does not match input"
            )
        _fsync_file(temporary)
        os.replace(temporary, template)
    except Exception:
        _remove_temporary_database(temporary)
        raise

    return ReleaseDatabaseResult(
        action="built",
        path=template,
        database_sha256=_sha256_path(template),
        metadata=metadata,
    )


def ensure_runtime_database(
    template_path: str | Path,
    runtime_path: str | Path,
) -> ReleaseDatabaseResult:
    """Install or upgrade a runtime DB without exposing a partial database."""
    template = Path(template_path)
    runtime = Path(runtime_path)
    if template.resolve() == runtime.resolve():
        raise ReleaseDatabaseError(
            "release template and runtime database must be different files"
        )
    template_metadata = _validate_template(template)
    runtime.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_sidecars(runtime)

    if not runtime.exists():
        temporary = _temporary_peer(runtime)
        try:
            _copy_template(template, temporary)
            installed_metadata = _validate_template(temporary)
            os.replace(temporary, runtime)
        except Exception:
            _remove_temporary_database(temporary)
            raise
        return ReleaseDatabaseResult(
            action="created",
            path=runtime,
            database_sha256=_sha256_path(runtime),
            metadata=installed_metadata,
        )

    runtime_metadata, summaries, has_forbidden_history = _read_runtime_state(
        runtime
    )
    # Metadata lives inside the mutable runtime database and is therefore not
    # an integrity boundary.  Only a byte-identical copy of the immutable
    # bundled template may bypass reconstruction.  Once a summary is written,
    # or any static byte is changed, rebuild from the template and migrate the
    # exact run_summaries rows only.
    runtime_hash = _sha256_path(runtime)
    template_hash = _sha256_path(template)
    if (
        runtime_hash == template_hash
        and not has_forbidden_history
    ):
        return ReleaseDatabaseResult(
            action="unchanged",
            path=runtime,
            database_sha256=_sha256_path(runtime),
            metadata=runtime_metadata,
            migrated_run_summaries=len(summaries),
        )

    temporary = _temporary_peer(runtime)
    try:
        _copy_template(template, temporary)
        placeholders = ", ".join("?" for _ in RUN_SUMMARY_COLUMNS)
        insert_columns = ", ".join(
            f'"{column}"' for column in RUN_SUMMARY_COLUMNS
        )
        with closing(sqlite3.connect(str(temporary))) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executemany(
                f"INSERT INTO run_summaries({insert_columns}) "
                f"VALUES({placeholders})",
                summaries,
            )
            connection.commit()
            _validate_integrity(connection)

        upgraded_metadata = _validate_runtime_upgrade(
            temporary,
            template_metadata,
            len(summaries),
        )
        _fsync_file(temporary)
        os.replace(temporary, runtime)
    except Exception:
        _remove_temporary_database(temporary)
        raise

    return ReleaseDatabaseResult(
        action=(
            "repaired"
            if has_forbidden_history or not summaries
            else "upgraded"
        ),
        path=runtime,
        database_sha256=_sha256_path(runtime),
        metadata=upgraded_metadata,
        migrated_run_summaries=len(summaries),
    )


def _validate_runtime_upgrade(
    path: Path,
    template_metadata: Mapping[str, str],
    expected_summary_count: int,
) -> dict[str, str]:
    try:
        with _readonly_connection(path) as connection:
            _validate_integrity(connection)
            metadata = _read_metadata(connection)
            if _database_identity(metadata) != _database_identity(
                template_metadata
            ):
                raise ReleaseDatabaseError(
                    "upgraded runtime metadata does not match the template"
                )
            if _table_count(connection, "run_summaries") != expected_summary_count:
                raise ReleaseDatabaseError(
                    "upgraded runtime did not preserve every run summary"
                )
            for table in PROHIBITED_HISTORY_TABLES:
                if _table_count(connection, table):
                    raise ReleaseDatabaseError(
                        f"upgraded runtime retained forbidden history in {table}"
                    )
    except sqlite3.DatabaseError as exc:
        raise ReleaseDatabaseError("upgraded runtime is unreadable") from exc
    return metadata
