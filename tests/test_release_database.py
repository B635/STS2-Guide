import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage.release_database import (
    PROHIBITED_HISTORY_TABLES,
    PUBLIC_CATALOG_TYPES,
    RELEASE_CATALOG_PROFILE,
    ReleaseDatabaseError,
    build_release_database,
    ensure_runtime_database,
)
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]


class ReleaseDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "knowledge.json"
        self.community = self.root / "community-scores.json"
        self.template = self.root / "release" / "template.db"
        self.runtime = self.root / "runtime" / "sts2-guide.db"
        self._write_sources(version=1)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_sources(self, *, version: int) -> None:
        self.catalog.write_text(
            json.dumps(
                {
                    "characters": [
                        {
                            "id": "TEST_HERO",
                            "name": "测试角色",
                            "description": "",
                            "embed_text": "测试角色",
                        }
                    ],
                    "cards": [
                        {
                            "id": "TEST_CARD",
                            "name": f"测试卡牌 v{version}",
                            "description": (
                                f"DO_NOT_REDISTRIBUTE_NARRATIVE_{version}. "
                                f"Deal {version + 4} damage. Gain Strength."
                            ),
                            "upgrade_description": "SECRET_UPGRADE_PROSE",
                            "embed_text": f"SECRET_SEMANTIC_TEXT_{version}",
                            "type_key": "Attack",
                            "color": "test",
                            "rarity_key": "Common",
                            "cost": 1,
                            "damage": version + 4,
                            "upgrade": {"damage": "+2"},
                        }
                    ],
                    "relics": [],
                    "potions": [],
                    "monsters": [],
                    "encounters": [],
                    "events": [
                        {
                            "id": "SECRET_EVENT",
                            "name": "Secret event",
                            "description": "SECRET_EVENT_DIALOGUE",
                            "embed_text": "SECRET_EVENT_VECTOR_TEXT",
                            "pages": [],
                        }
                    ],
                    "acts": [],
                    "powers": [
                        {
                            "id": "SECRET_POWER",
                            "name": "Secret power",
                            "description": "SECRET_POWER_PROSE",
                            "embed_text": "SECRET_POWER_VECTOR_TEXT",
                        }
                    ],
                    "intents": [],
                    "keywords": [],
                    "enchantments": [],
                    "afflictions": [],
                    "orbs": [],
                    "modifiers": [],
                    "mechanics": {},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.community.write_text(
            json.dumps(
                {
                    "source": {
                        "id": "spire_codex_api",
                        "name": "Spire Codex API",
                        "endpoints": {
                            "cards": "https://spire-codex.com/api/cards"
                        },
                    },
                    "fetched_at": f"2026-08-0{version}T00:00:00+00:00",
                    "game_version": "0.110.1",
                    "methodology": "test aggregate",
                    "entities": {
                        "cards": {
                            "TEST_CARD": {
                                "score": 50 + version,
                                "elo": 1000 + version,
                                "picks": 100,
                                "wins": 50,
                                "win_rate": 50.0,
                            }
                        }
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _metadata(path: Path) -> dict[str, str]:
        with sqlite3.connect(str(path)) as connection:
            return dict(connection.execute("SELECT key, value FROM schema_metadata"))

    @staticmethod
    def _count(path: Path, table: str) -> int:
        with sqlite3.connect(str(path)) as connection:
            return int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            )

    @staticmethod
    def _summary() -> dict:
        return {
            "run_id": "completed-run",
            "outcome": "win",
            "character": "TEST_HERO",
            "ascension": 3,
            "final_floor": 50,
            "final_score": 1234,
            "started_at": "2026-08-01T00:00:00+00:00",
            "ended_at": "2026-08-01T01:00:00+00:00",
            "game_version": "0.110.1",
            "final_deck": [{"card": "TEST_CARD"}],
            "final_relics": ["TEST_RELIC"],
            "final_potions": [],
        }

    def _build_template(self):
        return build_release_database(
            self.template,
            self.catalog,
            self.community,
        )

    def test_build_creates_clean_template_with_exact_source_hashes(self):
        result = self._build_template()

        self.assertEqual(result.action, "built")
        self.assertEqual(result.database_sha256, self._sha256(self.template))
        metadata = self._metadata(self.template)
        self.assertEqual(metadata["catalog_sha256"], self._sha256(self.catalog))
        self.assertEqual(
            metadata["community_scores_sha256"],
            self._sha256(self.community),
        )
        self.assertEqual(
            metadata["release_catalog_profile"],
            RELEASE_CATALOG_PROFILE,
        )
        for table in ("run_summaries", *PROHIBITED_HISTORY_TABLES):
            self.assertEqual(self._count(self.template, table), 0, table)

        repository = RelationalRepository(str(self.template))
        card = repository.find_card("TEST_CARD")
        self.assertEqual(card["name"], "测试卡牌 v1")
        self.assertEqual(card["upgrade"], {"damage": "+2"})
        self.assertNotIn("description", card)
        self.assertNotIn("embed_text", card)
        self.assertIn("strength", card["_effect_tags"])
        self.assertEqual(
            card["_effect_tag_sources"]["strength"],
            "description_rule",
        )
        self.assertEqual(
            repository.find_latest_entity_stat("cards", "TEST_CARD")["score"],
            51,
        )

        with sqlite3.connect(str(self.template)) as connection:
            entity_types = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT entity_type FROM catalog_entities"
                )
            }
            self.assertTrue(entity_types <= PUBLIC_CATALOG_TYPES)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM catalog_entities "
                    "WHERE entity_type IN ('events', 'powers')"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM catalog_entities "
                    "WHERE description <> '' OR embed_text <> ''"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM mechanic_constants "
                    "WHERE constant_key <> 'route_risk_profile_v1'"
                ).fetchone()[0],
                0,
            )
        database_text = self.template.read_bytes().decode(
            "utf-8", errors="ignore"
        )
        for forbidden in (
            "DO_NOT_REDISTRIBUTE_NARRATIVE",
            "SECRET_UPGRADE_PROSE",
            "SECRET_SEMANTIC_TEXT",
            "SECRET_EVENT_DIALOGUE",
            "SECRET_EVENT_VECTOR_TEXT",
            "SECRET_POWER_PROSE",
            "SECRET_POWER_VECTOR_TEXT",
        ):
            self.assertNotIn(forbidden, database_text)

    def test_same_sources_produce_byte_identical_template(self):
        first = self.root / "release" / "first.db"
        second = self.root / "release" / "second.db"
        build_release_database(first, self.catalog, self.community)
        build_release_database(second, self.catalog, self.community)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_missing_runtime_is_installed_by_atomic_copy(self):
        self._build_template()
        result = ensure_runtime_database(self.template, self.runtime)

        self.assertEqual(result.action, "created")
        self.assertEqual(self._sha256(self.runtime), self._sha256(self.template))
        self.assertFalse(list(self.runtime.parent.glob(".*.tmp")))

    def test_same_template_preserves_existing_run_summaries(self):
        self._build_template()
        ensure_runtime_database(self.template, self.runtime)
        repository = RelationalRepository(str(self.runtime))
        repository.save_run_summary(self._summary())

        result = ensure_runtime_database(self.template, self.runtime)

        self.assertEqual(result.action, "upgraded")
        self.assertEqual(result.migrated_run_summaries, 1)
        self.assertEqual(repository.load_run_summary("completed-run")["final_score"], 1234)

    def test_static_runtime_tampering_is_replaced_even_when_metadata_matches(self):
        self._build_template()
        ensure_runtime_database(self.template, self.runtime)
        connection = sqlite3.connect(str(self.runtime))
        try:
            connection.execute(
                "UPDATE cards SET damage = 999999",
            )
            connection.commit()
        finally:
            connection.close()

        result = ensure_runtime_database(self.template, self.runtime)

        self.assertEqual(result.action, "repaired")
        repository = RelationalRepository(str(self.runtime))
        self.assertEqual(repository.find_card("TEST_CARD")["damage"], 5)
        self.assertEqual(self._sha256(self.runtime), self._sha256(self.template))

    def test_template_upgrade_migrates_only_run_summaries(self):
        self._build_template()
        ensure_runtime_database(self.template, self.runtime)
        old_repository = RelationalRepository(str(self.runtime))
        old_repository.save_run_summary(self._summary())
        with old_repository.connect() as connection:
            connection.execute(
                """
                INSERT INTO run_states(
                    id, character, act, floor, created_at
                ) VALUES('legacy-state', 'TEST_HERO', 1, 4, 'now')
                """
            )

        self._write_sources(version=2)
        self._build_template()
        result = ensure_runtime_database(self.template, self.runtime)

        self.assertEqual(result.action, "repaired")
        self.assertEqual(result.migrated_run_summaries, 1)
        upgraded = RelationalRepository(str(self.runtime))
        self.assertEqual(upgraded.find_card("TEST_CARD")["name"], "测试卡牌 v2")
        self.assertEqual(
            upgraded.find_latest_entity_stat("cards", "TEST_CARD")["score"],
            52,
        )
        self.assertEqual(
            upgraded.load_run_summary("completed-run")["final_score"],
            1234,
        )
        for table in PROHIBITED_HISTORY_TABLES:
            self.assertEqual(self._count(self.runtime, table), 0, table)

    def test_pre_release_runtime_without_hashes_is_safely_upgraded(self):
        self._build_template()
        ensure_runtime_database(self.template, self.runtime)
        repository = RelationalRepository(str(self.runtime))
        repository.save_run_summary(self._summary())
        with repository.connect() as connection:
            connection.execute(
                "DELETE FROM schema_metadata WHERE key IN "
                "('community_scores_sha256', 'release_catalog_profile')"
            )

        result = ensure_runtime_database(self.template, self.runtime)

        self.assertEqual(result.action, "upgraded")
        self.assertEqual(result.migrated_run_summaries, 1)
        upgraded = RelationalRepository(str(self.runtime))
        self.assertEqual(
            upgraded.load_run_summary("completed-run")["final_score"],
            1234,
        )
        metadata = self._metadata(self.runtime)
        self.assertEqual(
            metadata["release_catalog_profile"],
            RELEASE_CATALOG_PROFILE,
        )

    def test_upgrade_replace_failure_leaves_runtime_and_summary_intact(self):
        self._build_template()
        ensure_runtime_database(self.template, self.runtime)
        repository = RelationalRepository(str(self.runtime))
        repository.save_run_summary(self._summary())
        original_hash = self._sha256(self.runtime)

        self._write_sources(version=2)
        self._build_template()
        with patch(
            "storage.release_database.os.replace",
            side_effect=PermissionError("runtime is locked"),
        ):
            with self.assertRaises(PermissionError):
                ensure_runtime_database(self.template, self.runtime)

        self.assertEqual(self._sha256(self.runtime), original_hash)
        self.assertEqual(repository.run_summary_count(), 1)
        self.assertFalse(list(self.runtime.parent.glob(".*.tmp")))

    def test_failed_template_build_preserves_previous_template(self):
        self._build_template()
        original_hash = self._sha256(self.template)
        self.community.write_text("not-json", encoding="utf-8")

        with self.assertRaises(ReleaseDatabaseError):
            self._build_template()

        self.assertEqual(self._sha256(self.template), original_hash)
        self.assertFalse(list(self.template.parent.glob(".*.tmp")))

    def test_template_with_run_history_is_rejected_without_runtime_write(self):
        self._build_template()
        repository = RelationalRepository(str(self.template))
        repository.save_run_summary(self._summary())

        with self.assertRaisesRegex(
            ReleaseDatabaseError,
            "run_summaries",
        ):
            ensure_runtime_database(self.template, self.runtime)

        self.assertFalse(self.runtime.exists())

    def test_build_script_uses_explicit_offline_inputs(self):
        output = self.root / "script-output" / "template.db"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_release_database.py"),
                "--catalog",
                str(self.catalog),
                "--community-scores",
                str(self.community),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["action"], "built")
        self.assertEqual(Path(report["path"]), output)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
