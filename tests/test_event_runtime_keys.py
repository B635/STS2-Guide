"""Real EventOption.TextKey identities must resolve without suffix guessing."""
import unittest
from unittest.mock import patch

from advisor.decision_core import DecisionCandidate
import test_event_decision as fixtures


PAGE = "PAGE_0123456789ABCDEF"
KEY = "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN"


class EventRuntimeKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.EventPolicyTests.setUpClass()
        cls.repository = fixtures.EventPolicyTests.repository
        cls.policy = fixtures.EventPolicyTests.policy

    @classmethod
    def tearDownClass(cls):
        fixtures.EventPolicyTests.tearDownClass()

    def test_production_full_key_scores_and_preserves_identity(self):
        candidate = DecisionCandidate.create("event:stable-id", {
            "candidate_kind": "event_option", "event_id": "ABYSSAL_BATHS",
            "page_id": PAGE, "option_id": KEY, "costs": [],
            "effects": [fixtures._effect("hp_delta", amount=10)],
        }, label="敬而远之")
        result = self.policy.recommend(fixtures._request(
            [candidate], world=fixtures._world(hp=25),
        ))
        self.assertEqual(result.recommended_candidate_id, "event:stable-id")
        self.assertIsNotNone(result.candidates[0].score)
        self.assertEqual(candidate.payload["option_id"], KEY)

    def test_wrong_event_page_and_ambiguous_separators_do_not_match(self):
        for key in (
            "OTHER.pages.INITIAL.options.ABSTAIN",
            "ABYSSAL_BATHS.pages.UNKNOWN.options.ABSTAIN",
            "ABYSSAL_BATHS.pages.INITIAL.options.NOT_AN_OPTION",
            "ABYSSAL_BATHS.pages.INITIAL.extra.options.ABSTAIN",
            "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN.extra",
            "ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN.options.ABSTAIN",
            "prefix.ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN",
        ):
            with self.subTest(key=key):
                self.assertFalse(self.policy._known_option("ABYSSAL_BATHS", PAGE, key))

    def test_explicit_page_cannot_disagree_with_full_key(self):
        self.assertTrue(self.policy._known_option("ABYSSAL_BATHS", "INITIAL", KEY))
        self.assertFalse(self.policy._known_option("ABYSSAL_BATHS", "ALL", KEY))
        self.assertFalse(self.policy._known_option("ABYSSAL_BATHS", "UNKNOWN", KEY))

    def test_legacy_fingerprint_requires_unique_catalog_page(self):
        tree = {"_pages": [{"id": "ONE", "options": [{"id": "CHOICE"}]}]}
        with patch.object(self.repository, "event_tree", return_value=tree):
            self.assertTrue(self.policy._known_option("FIXTURE", PAGE, "CHOICE"))
            self.assertFalse(self.policy._known_option("FIXTURE", "UNKNOWN", "CHOICE"))
            tree["_pages"].append({"id": "TWO", "options": [{"id": "CHOICE"}]})
            self.assertFalse(self.policy._known_option("FIXTURE", PAGE, "CHOICE"))
            self.assertTrue(self.policy._known_option(
                "FIXTURE", PAGE, "FIXTURE.pages.ONE.options.CHOICE",
            ))

    def test_full_key_does_not_guess_from_root_catalog_summary(self):
        tree = {"_pages": [{"id": "__ROOT__", "options": [{"id": "ABSTAIN"}]}]}
        with patch.object(self.repository, "event_tree", return_value=tree):
            self.assertFalse(self.policy._known_option("ABYSSAL_BATHS", PAGE, KEY))


if __name__ == "__main__":
    unittest.main()
