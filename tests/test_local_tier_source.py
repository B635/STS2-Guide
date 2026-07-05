import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from advisor.card_reward import (
    COMMUNITY_PRIOR_METHOD,
    recommend_card_reward,
)
from advisor.data_sources import load_local_card_tiers


class _Repository:
    def find_card(self, identifier):
        if str(identifier).upper() != "TEST_CARD":
            return None
        return {
            "id": "TEST_CARD",
            "name": "测试牌",
            "cost": 1,
            "type_key": "Skill",
            "block": 0,
            "keywords_key": [],
        }

    def find_latest_entity_stat(self, entity_type, entity_id):
        return None


class LocalTierSourceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "tiers.json"
        self.now = datetime(2026, 7, 3, tzinfo=timezone.utc)

    def tearDown(self):
        self.tempdir.cleanup()

    def _write(self, *, captured_at=None, entries=None):
        self.path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": "mobalytics-manual-local",
                    "source_url": (
                        "https://mobalytics.gg/slay-the-spire-2/"
                        "tier-lists/cards"
                    ),
                    "captured_at": (
                        captured_at or self.now.isoformat()
                    ),
                    "entries": entries or [],
                }
            ),
            encoding="utf-8",
        )

    def test_missing_invalid_and_stale_files_degrade_safely(self):
        missing = load_local_card_tiers(self.path, now=self.now)
        self.assertEqual(missing.status, "missing")
        self.assertFalse(missing.available)

        self.path.write_text("{broken", encoding="utf-8")
        invalid = load_local_card_tiers(self.path, now=self.now)
        self.assertEqual(invalid.status, "invalid")
        self.assertFalse(invalid.available)

        self._write(
            captured_at=(
                self.now - timedelta(days=46)
            ).isoformat(),
            entries=[
                {
                    "card_id": "TEST_CARD",
                    "character_id": "TEST_HERO",
                    "tier": "S",
                }
            ],
        )
        stale = load_local_card_tiers(self.path, now=self.now)
        self.assertEqual(stale.status, "stale")
        self.assertFalse(stale.available)
        self.assertEqual(stale.entries, {})

    def test_stable_card_and_character_ids_are_exact(self):
        self._write(
            entries=[
                {
                    "card_id": "TEST_CARD",
                    "character_id": "TEST_HERO",
                    "tier": "A",
                }
            ]
        )
        source = load_local_card_tiers(self.path, now=self.now)
        self.assertTrue(source.available)
        self.assertEqual(
            source.find("test_card", "test_hero").tier,
            "A",
        )
        self.assertIsNone(source.find("测试牌", "TEST_HERO"))
        self.assertIsNone(source.find("TEST_CARD", "OTHER_HERO"))

    def test_duplicate_stable_ids_reject_whole_snapshot(self):
        entry = {
            "card_id": "TEST_CARD",
            "character_id": "TEST_HERO",
            "tier": "S",
        }
        self._write(entries=[entry, dict(entry)])
        source = load_local_card_tiers(self.path, now=self.now)
        self.assertEqual(source.status, "invalid")
        self.assertEqual(source.entries, {})

    def test_local_tier_is_bounded_prior_not_fuzzy_lookup(self):
        self._write(
            entries=[
                {
                    "card_id": "TEST_CARD",
                    "character_id": "TEST_HERO",
                    "tier": "S",
                }
            ]
        )
        source = load_local_card_tiers(self.path, now=self.now)
        result = recommend_card_reward(
            {
                "character": "TEST_HERO",
                "deck": [],
            },
            [{"card": "TEST_CARD"}],
            _Repository(),
            local_tiers=source,
        )
        recommendation = result["recommendations"][0]
        factor_codes = {
            factor["code"] for factor in recommendation["factors"]
        }
        self.assertEqual(result["method"], COMMUNITY_PRIOR_METHOD)
        self.assertIn("local_tier_prior", factor_codes)
        self.assertEqual(recommendation["state_score"], 50.0)
        self.assertEqual(recommendation["score"], 60.0)
        self.assertEqual(result["profile"]["local_tier_coverage"], 1)


if __name__ == "__main__":
    unittest.main()
