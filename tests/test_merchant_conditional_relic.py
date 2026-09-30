"""Conditional relic purchases through the release catalog and public policy."""
import tempfile
import unittest
from pathlib import Path

from advisor.decision_core import DecisionCandidate, DecisionRequest, WorldState
from advisor.merchant import MerchantPolicy
from advisor.scoring_preferences import ScoringPreferences
from storage.relational import RelationalRepository
from storage.release_database import build_release_database


ROOT = Path(__file__).resolve().parents[1]


class MerchantConditionalRelicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        target = Path(cls.tempdir.name) / "release.db"
        build_release_database(
            target, ROOT / "data/knowledge.json", ROOT / "data/community_scores.json",
        )
        cls.repository = RelationalRepository(str(target))
        cls.policy = MerchantPolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def request(self, deck, *, entity_id="TOUGH_BANDAGES", price=30,
                eligible=True, costs=None, **state_updates):
        state = {
            "character": "SILENT", "act": 1, "floor": 6,
            "hp": 20, "max_hp": 80, "gold": 120, "energy": 3,
            "deck": deck, "relics": [], "potions": [], "max_potion_slots": 3,
            **state_updates,
        }
        return DecisionRequest.create(
            decision_id="merchant:conditional-relic", decision_type="merchant_choice",
            world=WorldState.create(run_id="conditional-relic", sequence=1, state=state),
            candidates=[
                DecisionCandidate.create("relic", {
                    "candidate_kind": "merchant_offer", "offer_kind": "relic",
                    "is_stocked": eligible, "entity_id": entity_id,
                    "costs": (costs if costs is not None else
                              [{"kind": "gold", "amount": price, "resource_id": None}]),
                }, eligible=eligible),
                DecisionCandidate.create("leave", {"candidate_kind": "leave", "costs": []}),
            ],
        )

    def test_tough_bandages_needs_active_discard_for_survival_adaptation(self):
        without = self.policy.recommend(self.request([{"card": "STRIKE_SILENT", "count": 10}]))
        with_source = self.policy.recommend(self.request([{"card": "DAGGER_THROW", "count": 10}]))
        # The verified trigger must gate all 8 points of existing survival
        # adaptation; the +14 long-term prior and -6 actual price stay intact.
        self.assertEqual(without.candidates[0].score, 58.0)
        self.assertEqual(without.candidates[0].dimensions["survival"], 0.0)
        healthy_without = self.policy.recommend(self.request(
            [{"card": "STRIKE_SILENT", "count": 10}], hp=80,
        ))
        self.assertEqual(healthy_without.candidates[0].score, without.candidates[0].score)
        self.assertEqual(with_source.candidates[0].score, 66.0)
        self.assertEqual(with_source.candidates[0].dimensions["survival"], 8.0)
        self.assertEqual(without.data_gaps, ())
        self.assertEqual(with_source.data_gaps, ())
        scopes = [factor for factor in with_source.candidates[0].factors if factor["code"] == "estimate_scope"]
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0]["delta"], 0.0)
        self.assertIn("潜在配合", scopes[0]["message"])
        self.assertIn("触发频率", scopes[0]["message"])

    def test_cross_color_active_sources_and_upgrades_use_existing_discard_facts(self):
        for card in ("DAGGER_THROW", "SCRAPE"):
            for upgrades in (0, 1):
                with self.subTest(card=card, upgrades=upgrades):
                    result = self.policy.recommend(self.request(
                        [{"card": card, "count": 1, "upgrades": upgrades}],
                        character="IRONCLAD",
                    ))
                    self.assertEqual(result.candidates[0].score, 66.0)
                    self.assertEqual(result.data_gaps, ())

    def test_sly_payoffs_negation_and_discard_pile_nouns_are_not_sources(self):
        # Real release records: Sly payoff, explicit "do not discard", and
        # returning a card from the discard pile are not active providers.
        for card in ("REFLEX", "TACTICIAN", "WELL_LAID_PLANS", "HOLOGRAM"):
            with self.subTest(card=card):
                result = self.policy.recommend(self.request([{"card": card}]))
                self.assertEqual(result.candidates[0].score, 58.0)
                self.assertEqual(result.candidates[0].dimensions["survival"], 0.0)
                self.assertEqual(result.data_gaps, ())
                messages = " ".join(f["message"] for f in result.candidates[0].factors)
                self.assertIn("未观察到主动弃牌来源", messages)

    def test_unknown_or_partial_deck_is_not_treated_as_known_absent_source(self):
        for deck, warnings, gap in (
            ([], [], "merchant:relic_trigger_deck_unknown:TOUGH_BANDAGES"),
            ([{"card": "DAGGER_THROW"}], ["deck_capture_incomplete"],
             "merchant:relic_trigger_deck_unknown:TOUGH_BANDAGES"),
            ([{"card": "NOT_A_CARD"}], [], "unknown_card:NOT_A_CARD"),
            ([{"card": "DAGGER_THROW"}, {"card": "NOT_A_CARD"}], [], "unknown_card:NOT_A_CARD"),
        ):
            with self.subTest(deck=deck, warnings=warnings):
                result = self.policy.recommend(self.request(deck, capture_warnings=warnings))
                self.assertIsNone(result.candidates[0].score)
                self.assertIsNone(result.recommended_candidate_id)
                self.assertIn(gap, result.data_gaps)
                self.assertFalse(any(f["code"] == "estimate_scope" for f in result.candidates[0].factors))

    def test_unmodeled_deck_instances_do_not_claim_a_verified_trigger(self):
        for change in (
            {"enchantment": "fixture"}, {"affliction": "fixture"},
            {"upgrades": 2}, {"count": 0},
        ):
            with self.subTest(change=change):
                result = self.policy.recommend(self.request([{"card": "DAGGER_THROW", **change}]))
                self.assertIsNone(result.candidates[0].score)
                self.assertIn("merchant:relic_trigger_deck_unmodeled:TOUGH_BANDAGES", result.data_gaps)

    def test_survival_preference_applies_once_without_changing_price(self):
        configured = MerchantPolicy(self.repository, ScoringPreferences(survival=1.25))
        request = self.request([{"card": "DAGGER_THROW", "count": 10}])
        baseline = self.policy.recommend(request).candidates[0]
        adjusted = configured.recommend(request).candidates[0]
        before = {f["code"]: f["delta"] for f in baseline.factors}
        after = {f["code"]: f["delta"] for f in adjusted.factors}
        self.assertEqual(baseline.score, 66.0)
        self.assertEqual(adjusted.score, 68.0)
        self.assertEqual(after["gold_liquidity_cost"], before["gold_liquidity_cost"])
        self.assertEqual(after["merchant_relic_permanence"], before["merchant_relic_permanence"])
        self.assertEqual(adjusted.dimensions["survival"], 10.0)
        without = configured.recommend(self.request([{"card": "STRIKE_SILENT"}]))
        self.assertEqual(without.candidates[0].score, 58.0)

    def test_non_allowlisted_and_unknown_relics_keep_existing_behavior(self):
        anchor = self.policy.recommend(self.request([{"card": "STRIKE_SILENT"}], entity_id="ANCHOR"))
        self.assertEqual(anchor.candidates[0].score, 66.0)
        self.assertFalse(any(f["code"] == "estimate_scope" for f in anchor.candidates[0].factors))
        unknown = self.policy.recommend(self.request([{"card": "DAGGER_THROW"}], entity_id="NOT_A_RELIC"))
        self.assertIsNone(unknown.candidates[0].score)
        self.assertIn("unknown_relic:NOT_A_RELIC", unknown.data_gaps)

    def test_price_affordability_and_extra_cost_safety_are_unchanged(self):
        deck = [{"card": "DAGGER_THROW"}]
        cheap = self.policy.recommend(self.request(deck, price=30))
        expensive = self.policy.recommend(self.request(deck, price=90))
        self.assertGreater(cheap.candidates[0].score, expensive.candidates[0].score)
        unaffordable = self.policy.recommend(self.request(deck, price=121))
        self.assertIsNone(unaffordable.candidates[0].score)
        self.assertIn("resource:eligibility_gold_inconsistent", unaffordable.data_gaps)
        extra = self.policy.recommend(self.request(deck, costs=[
            {"kind": "gold", "amount": 30, "resource_id": None},
            {"kind": "hp", "amount": 1, "resource_id": None},
        ]))
        self.assertIsNone(extra.candidates[0].score)
        self.assertTrue(any(gap.startswith("merchant:costs_invalid:") for gap in extra.data_gaps))

    def test_ineligible_offer_does_not_require_deck_or_health_facts(self):
        result = self.policy.recommend(self.request([], eligible=False, hp=None))
        self.assertIsNone(result.candidates[0].score)
        self.assertEqual(result.recommended_candidate_id, "leave")
        self.assertEqual(result.data_gaps, ())


if __name__ == "__main__":
    unittest.main()
