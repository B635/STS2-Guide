import json
import os
import tempfile
import unittest
from unittest.mock import patch

from rag.guide_knowledge import (
    build_guide_items,
    chunk_markdown,
    split_markdown_sections,
)
from rag.bm25 import tokenize
from rag.embedder import load_model, validate_embedding_documents
from rag.knowledge import load_knowledge
from rag.retriever import attach_result_metadata


class GuideChunkingTests(unittest.TestCase):
    def test_bm25_tokenizer_drops_punctuation_only_tokens(self):
        self.assertNotIn("？", tokenize("怎么玩？"))

    def test_markdown_sections_preserve_heading_path(self):
        sections = split_markdown_sections(
            "# Character\nIntro\n## Build\nTake cards.\n### Boss\nPrepare damage."
        )
        self.assertEqual(
            [section for section, _ in sections],
            ["Character", "Character > Build", "Character > Build > Boss"],
        )

    def test_chunks_are_bounded_and_stable(self):
        markdown = "# Plan\n" + ("word " * 160)
        chunks = chunk_markdown(markdown, max_chars=120, overlap_chars=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["content"]) <= 120 for chunk in chunks))
        self.assertEqual(
            [chunk["ordinal"] for chunk in chunks],
            list(range(len(chunks))),
        )

    def test_official_chinese_entity_names_are_added(self):
        facts = [
            {"id": "NECROBINDER", "name": "亡灵契约师"},
            {"id": "GRAVEBLAST", "name": "坟冢爆射"},
        ]
        payload = {
            "guides": [
                {
                    "id": "test-guide",
                    "slug": "test-guide",
                    "title": "Test Guide",
                    "author": "Author",
                    "category": "character",
                    "character": "necrobinder",
                    "tags": ["beginner", "graveblast"],
                    "content": "# Loop\nUse [[Graveblast]] twice.",
                }
            ]
        }
        items = build_guide_items(payload, facts, max_chars=200, overlap_chars=20)
        self.assertEqual(len(items), 1)
        self.assertIn("亡灵契约师", items[0]["embed_text"])
        self.assertIn("坟冢爆射", items[0]["embed_text"])
        self.assertIn("新手", items[0]["embed_text"])

    def test_embedding_length_validation_rejects_silent_truncation(self):
        class FakeTokenizer:
            def __call__(self, docs, **_kwargs):
                return {"input_ids": [[1] * len(doc) for doc in docs]}

        class FakeModel:
            tokenizer = FakeTokenizer()
            max_seq_length = 4

        with self.assertRaisesRegex(ValueError, "exceed embedding"):
            validate_embedding_documents(["short", "no"], FakeModel())

    def test_embedding_model_prefers_complete_local_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = os.path.join(
                directory,
                "models--org--model",
            )
            snapshot = os.path.join(repository, "snapshots", "revision")
            os.makedirs(snapshot)
            os.makedirs(os.path.join(repository, "refs"))
            with open(
                os.path.join(repository, "refs", "main"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write("revision")
            with open(
                os.path.join(snapshot, "modules.json"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write("{}")

            with patch(
                "rag.embedder.EMBEDDING_CACHE_DIR",
                directory,
            ), patch(
                "rag.embedder.EMBEDDING_MODEL",
                "org/model",
            ), patch(
                "rag.embedder.SentenceTransformer"
            ) as transformer:
                load_model()

        transformer.assert_called_once_with(
            snapshot,
            local_files_only=True,
        )


class KnowledgeIntegrationTests(unittest.TestCase):
    def test_guides_append_without_entering_structured_index(self):
        knowledge = {
            "characters": [{"id": "HERO", "name": "英雄", "embed_text": "角色英雄"}],
            "cards": [],
            "relics": [],
            "potions": [],
            "monsters": [],
        }
        guides = {
            "guides": [
                {
                    "id": "guide",
                    "slug": "guide",
                    "title": "Guide",
                    "content": "# Start\nA useful strategy.",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            knowledge_path = os.path.join(directory, "knowledge.json")
            guides_path = os.path.join(directory, "guides.json")
            with open(knowledge_path, "w", encoding="utf-8") as file:
                json.dump(knowledge, file, ensure_ascii=False)
            with open(guides_path, "w", encoding="utf-8") as file:
                json.dump(guides, file, ensure_ascii=False)

            docs, items, index = load_knowledge(
                knowledge_path,
                guides_path=guides_path,
            )

        self.assertEqual(len(docs), 2)
        self.assertEqual(items[-1]["_type"], "guides")
        self.assertNotIn("guides", index)

    def test_result_metadata_uses_aligned_item(self):
        results = [{"text": "chunk", "score": 0.8, "index": 0}]
        items = [
            {
                "id": "guide:test:0",
                "_type": "guides",
                "source_title": "Test Guide",
                "source_author": "Author",
                "source_url": "https://example.com/guide",
            }
        ]
        enriched = attach_result_metadata(results, items)
        self.assertEqual(enriched[0]["source_type"], "guides")
        self.assertEqual(enriched[0]["title"], "Test Guide")
        self.assertEqual(enriched[0]["author"], "Author")
        self.assertEqual(enriched[0]["url"], "https://example.com/guide")


if __name__ == "__main__":
    unittest.main()
