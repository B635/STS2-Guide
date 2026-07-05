import unittest

from fastapi.testclient import TestClient

from api import _serialize_sources, app


class ApiSourceTests(unittest.TestCase):
    def test_health_endpoint_does_not_require_model_loading(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_source_metadata_and_safe_urls_are_serialized(self):
        sources = _serialize_sources(
            [
                {
                    "text": "Guide chunk",
                    "score": 0.5,
                    "source_type": "guides",
                    "source_id": "guide:test:0",
                    "title": "Test Guide",
                    "author": "Author",
                    "url": "https://example.com/guide",
                    "original_url": "javascript:alert(1)",
                    "section": "Opening",
                    "language": "eng",
                }
            ]
        )
        payload = sources[0].model_dump()
        self.assertEqual(payload["title"], "Test Guide")
        self.assertEqual(payload["url"], "https://example.com/guide")
        self.assertIsNone(payload["original_url"])


if __name__ == "__main__":
    unittest.main()
