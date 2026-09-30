import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import fetch_knowledge as fetcher


class FetchKnowledgeChannelTests(unittest.TestCase):
    def fake_api(self, versions):
        versions = iter(versions)

        def respond(endpoint, params=None):
            if endpoint == "beta/version":
                return {"beta_version": next(versions)}
            if endpoint in {"mechanics/constants", "merchant/config"}:
                return {}
            return []

        return respond

    def test_beta_query_and_provenance_are_explicit(self):
        with patch.object(fetcher, "fetch_with_delay", side_effect=self.fake_api(["v0.111.0"] * 2)) as api:
            with contextlib.redirect_stdout(io.StringIO()):
                result = fetcher.fetch_all("beta", "0.111.0")
        source = result["_source"]
        self.assertEqual(source["game_version"], "0.111.0")
        self.assertEqual(source["entity_channel"], "beta")
        for call in api.call_args_list:
            endpoint = call.args[0]
            if endpoint not in {"beta/version", "mechanics/constants", "merchant/config"}:
                self.assertEqual(call.kwargs["params"], {"channel": "beta"})
        self.assertEqual(source["unversioned_endpoints"], ["mechanics/constants", "merchant/config"])
        self.assertEqual(result["mechanics"]["route_risk_profile_v1"]["version"], "route-risk-v1")
        self.assertEqual(result["_guide_source"]["id"], "sts2_guide_policy")

    def test_wrong_beta_version_stops_before_fetching_entities(self):
        with patch.object(fetcher, "fetch_with_delay", side_effect=self.fake_api(["v0.112.0"])) as api:
            with self.assertRaisesRegex(RuntimeError, "Expected game"):
                fetcher.fetch_all("beta", "0.111.0")
        self.assertEqual(api.call_count, 1)

    def test_version_change_does_not_overwrite_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "knowledge.json"
            target.write_text('{"existing": true}', encoding="utf-8")
            with patch.object(fetcher, "fetch_with_delay", side_effect=self.fake_api(["v0.111.0", "v0.112.0"])):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "changed during"):
                    fetcher.main(["--channel", "beta", "--expected-game-version", "0.111.0", "--output", str(target)])
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"existing": True})
            self.assertFalse(Path(str(target) + ".tmp").exists())

    def test_main_does_not_claim_a_verified_game_version(self):
        with patch.object(fetcher, "fetch_with_delay", side_effect=self.fake_api([])):
            with contextlib.redirect_stdout(io.StringIO()):
                result = fetcher.fetch_all()
            self.assertIsNone(result["_source"]["game_version"])
            with self.assertRaises(ValueError):
                fetcher.fetch_all("main", "0.111.0")


if __name__ == "__main__":
    unittest.main()
