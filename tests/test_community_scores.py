import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.fetch_community_scores import fetch_scores


class CommunityScoreFetchTests(unittest.TestCase):
    @patch("scripts.fetch_community_scores._get_json")
    def test_fetch_keeps_only_official_catalog_entities(self, get_json):
        def response_for(_session, url):
            if url.endswith("/runs/versions"):
                return {"versions": ["v0.107.1"]}
            if url.endswith("/cards"):
                return {
                    "OFFICIAL_CARD": {
                        "score": 80,
                        "picks": 1000,
                        "wins": 600,
                        "win_rate": 60.0,
                        "elo": 2000,
                    },
                    "MOD-CARD": {
                        "score": 100,
                        "picks": 1000,
                        "wins": 900,
                        "win_rate": 90.0,
                    },
                }
            return {}

        get_json.side_effect = response_for
        with tempfile.TemporaryDirectory() as directory:
            knowledge_path = os.path.join(directory, "knowledge.json")
            output_path = os.path.join(directory, "scores.json")
            with open(knowledge_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "cards": [{"id": "OFFICIAL_CARD"}],
                        "relics": [],
                        "potions": [],
                    },
                    file,
                )

            counts = fetch_scores(
                knowledge_path=knowledge_path,
                output_path=output_path,
                base_url="https://example.test/api",
            )
            payload = json.loads(
                Path(output_path).read_text(encoding="utf-8")
            )

        self.assertEqual(counts["cards"], 1)
        self.assertIn(
            "OFFICIAL_CARD",
            payload["entities"]["cards"],
        )
        self.assertNotIn("MOD-CARD", payload["entities"]["cards"])
        self.assertEqual(payload["observed_versions"], ["v0.107.1"])


if __name__ == "__main__":
    unittest.main()
