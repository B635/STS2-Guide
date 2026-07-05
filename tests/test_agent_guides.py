import unittest
from unittest.mock import patch

import numpy as np

from rag.agent import AgentConfig, merge_results, run_agent
from rag.bm25 import build_bm25_index
from rag.knowledge import ENTITY_TYPES


class FakeModel:
    def encode(self, _texts):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeStore:
    size = 1

    def search(self, _query_vec, n):
        count = min(n, self.size)
        return (
            np.array([0] * count, dtype=np.int64),
            np.array([0.9] * count, dtype=np.float32),
        )


class AgentGuideIntegrationTests(unittest.TestCase):
    def test_merge_results_uses_rank_not_incomparable_component_scores(self):
        retrieved = [{"index": 1, "text": "guide", "score": 0.02}]
        heuristic = [{"index": 2, "text": "fact", "score": 4.0}]
        merged = merge_results(retrieved, heuristic)
        self.assertEqual(merged[0]["index"], 1)
        self.assertAlmostEqual(merged[0]["score"], merged[1]["score"])

    @patch("rag.agent.rag_chat", return_value="该攻略建议先保持牌组精简。[1]")
    def test_agent_returns_guide_source_metadata(self, _mock_chat):
        docs = ["杀戮尖塔2 攻略 新手入门\nKeep the deck focused."]
        items = [
            {
                "id": "guide:getting-started:0",
                "_type": "guides",
                "embed_text": docs[0],
                "source_title": "Getting Started",
                "source_author": "Author",
                "source_url": "https://spire-codex.com/guides/getting-started",
                "source_language": "eng",
                "section": "Deckbuilding",
            }
        ]
        index = {entity_type: [] for entity_type in ENTITY_TYPES}

        result = run_agent(
            "新手入门基础",
            history=[],
            docs=docs,
            items=items,
            index=index,
            store=FakeStore(),
            model=FakeModel(),
            client=None,
            bm25_index=build_bm25_index(docs),
            config=AgentConfig(top_n=1, candidate_n=1),
        )

        self.assertEqual(result.selected_tool, "hybrid_search")
        self.assertEqual(result.results[0]["source_type"], "guides")
        self.assertEqual(result.results[0]["title"], "Getting Started")
        self.assertEqual(result.results[0]["author"], "Author")
        self.assertTrue(result.verification.passed)


if __name__ == "__main__":
    unittest.main()
