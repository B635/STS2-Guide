"""Exercise the batch command without touching committed assembly facts."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import merge_fact_reviews as merge


class FactBatchMergeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / "facts.json"
        self.batch = self.root / "batch.json"
        (self.root / "data").mkdir()
        (self.root / "data" / "knowledge.json").write_text('{"cards":[]}', encoding="utf-8")
        self.entry = {
            "entity_key": "cards:TEST", "fingerprint": "abc", "status": "converted",
            "card_type": "Skill", "cost": None, "tags": [], "keywords": [],
            "effects": [], "evidence": ["OnPlay"],
            "reviewed_by": "drafter",
            "audit": {"by": "reviewer", "verdict": "resolved", "notes": "checked OnPlay"},
        }
        prior = dict(self.entry, **{"class": "Test", "hooks": ["OnPlay"],
                                    "cost_modifier": {"old": True}, "multiplayer_only": True,
                                    "reviewed_fingerprint": "prior"})
        self.target.write_text(json.dumps({"entities": [prior]}), encoding="utf-8")

    def invoke(self, entries, *options):
        self.batch.write_text(json.dumps(entries), encoding="utf-8")
        with patch.object(sys, "argv", ["merge", str(self.batch), *options]), \
                patch.object(merge, "ROOT_DIR", self.root), \
                patch.object(merge, "DEFAULT_FACTS_PATH", self.target), \
                patch.object(merge, "load_facts", lambda path: json.loads(path.read_text())), \
                patch.object(merge, "load_extract", return_value={}), \
                patch.object(merge, "extract_index", return_value={"cards:TEST": {"fingerprint": "abc"}}), \
                contextlib.redirect_stdout(io.StringIO()):
            return merge.main()

    def test_draft_can_be_checked_but_not_merged(self):
        original = self.target.read_bytes()
        for audit in (None, {"by": "other", "verdict": "disagree", "notes": "wrong timing"},
                      {"by": "other", "verdict": "agree"}):
            with self.subTest(audit=audit):
                draft = dict(self.entry, audit=audit)
                self.assertEqual(self.invoke([draft], "--check"), 0)
                self.assertEqual(self.invoke([draft]), 1)
                self.assertEqual(self.target.read_bytes(), original)

    def test_duplicate_input_and_partial_card_do_not_write(self):
        original = self.target.read_bytes()
        self.assertEqual(self.invoke([self.entry, self.entry]), 1)
        self.assertEqual(self.invoke([{k: v for k, v in self.entry.items() if k != "effects"}]), 1)
        self.assertEqual(self.target.read_bytes(), original)

    def test_complete_rereview_replaces_old_rules_and_preserves_extract_fields(self):
        self.assertEqual(self.invoke([self.entry]), 0)
        row, = json.loads(self.target.read_text())["entities"]
        self.assertEqual(row, {**self.entry, "class": "Test", "hooks": ["OnPlay"]})
