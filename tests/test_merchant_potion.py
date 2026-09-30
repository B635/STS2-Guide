"""Merchant healing opportunity through the shipped SQLite and policy seams."""
import tempfile
import unittest
import shutil
from pathlib import Path

from advisor.decision_core import DecisionCandidate
from advisor.merchant import MerchantPolicy
from advisor.scoring_preferences import ScoringPreferences
from storage.relational import RelationalRepository
from storage.release_database import build_release_database
from test_merchant_decision import _candidate, _request, _world


ROOT = Path(__file__).resolve().parents[1]


class MerchantPotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        target = Path(cls.tempdir.name) / "release.db"
        build_release_database(
            target, ROOT / "data/knowledge.json", ROOT / "data/community_scores.json"
        )
        cls.repository = RelationalRepository(str(target))
        cls.template = target
        cls.policy = MerchantPolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_blood_potion_values_current_healing_opportunity_from_release_data(self):
        result = self.policy.recommend(_request([
            _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", price=10),
            _candidate("leave", "leave"),
        ], world=_world(hp=40)))
        # Verified 20% of 80 HP = 16 recoverable; existing recovery value 8,
        # actual 10/120 gold price -4. This is a purchase opportunity, not use.
        self.assertEqual(result.candidates[0].score, 54.0)
        self.assertEqual(result.recommended_candidate_id, "blood")
        messages = " ".join(f["message"] for f in result.candidates[0].factors)
        self.assertIn("当前可恢复16点", messages)
        self.assertIn("购买不等于饮用", messages)
        self.assertIn("estimate_scope", {f["code"] for f in result.candidates[0].factors})
        self.assertEqual(result.data_gaps, ())

    def test_healing_truncates_percentage_and_caps_at_current_missing_hp(self):
        offer = _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", price=10)
        for hp, recovery, delta in ((40, 16, 7.9012), (70, 11, 5.4321), (81, 0, 0.0)):
            with self.subTest(hp=hp):
                result = self.policy.recommend(_request([
                    offer, _candidate("leave", "leave"),
                ], world=_world(hp=hp, max_hp=81)))
                factors = result.candidates[0].factors
                self.assertEqual(sum(f["delta"] for f in factors if f["code"] == "hp_recovery"), delta)
                self.assertIn(f"当前可恢复{recovery}点", " ".join(f["message"] for f in factors))
                self.assertIsNotNone(result.candidates[0].score)
                self.assertEqual(result.data_gaps, ())
                if hp == 81:
                    self.assertEqual(result.recommended_candidate_id, "leave")
                    self.assertEqual(result.candidates[0].score, 46.0)

    def test_blood_potion_without_valid_health_remains_unknown(self):
        offer = _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", price=0)
        for hp, max_hp in ((None, 80), (40, None), (True, 80), (81, 80), (-1, 80), (0, 0)):
            with self.subTest(hp=hp, max_hp=max_hp):
                result = self.policy.recommend(_request([
                    offer, _candidate("leave", "leave"),
                ], world=_world(hp=hp, max_hp=max_hp)))
                self.assertIsNone(result.candidates[0].score)
                self.assertIn("resource:hp_missing", result.data_gaps)
                self.assertIsNone(result.recommended_candidate_id)

    def test_other_potions_and_unknown_entities_are_not_inferred(self):
        for entity in ("STRENGTH_POTION", "REGEN_POTION", "FRUIT_JUICE", "NOT_A_POTION"):
            with self.subTest(entity=entity):
                result = self.policy.recommend(_request([
                    _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", price=10),
                    _candidate("other", "potion_offer", entity_id=entity, price=0),
                    _candidate("leave", "leave"),
                ], world=_world(hp=40)))
                self.assertIsNotNone(result.candidates[0].score)
                self.assertIsNone(result.candidates[1].score)
                self.assertEqual(result.status, "uncertain")
                self.assertIsNone(result.recommended_candidate_id)

    def test_missing_or_malformed_verified_fact_is_not_scoreable(self):
        for magnitude, source in (
            (None, None), (-1, "verified"), (0, "verified"),
            (21, "verified"), (float("inf"), "verified"), (20, "description_rule"),
        ):
            with self.subTest(magnitude=magnitude, source=source), tempfile.TemporaryDirectory() as directory:
                database = Path(directory) / "malformed.db"
                shutil.copyfile(self.template, database)
                repository = RelationalRepository(str(database))
                # Corrupt an external SQLite input fixture, then observe only
                # the public policy result (not private helper behavior).
                with repository.connect() as connection:
                    if magnitude is None:
                        connection.execute("DELETE FROM entity_effect_tags WHERE tag = 'potion_heal_max_hp_percent'")
                    else:
                        connection.execute(
                            "UPDATE entity_effect_tags SET magnitude = ?, source_field = ? WHERE tag = 'potion_heal_max_hp_percent'",
                            (magnitude, source if source != "verified" else "assembly:0.111.0:BloodPotion.CanonicalVars.HealPercent"),
                        )
                result = MerchantPolicy(repository).recommend(_request([
                    _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", price=0),
                    _candidate("leave", "leave"),
                ], world=_world(hp=40)))
                self.assertIsNone(result.candidates[0].score)
                self.assertIn("merchant:potion_mechanism_unmodeled:BLOOD_POTION", result.data_gaps)

    def test_price_and_preference_apply_once_to_healing_only(self):
        request = _request([
            _candidate("cheap", "potion_offer", entity_id="BLOOD_POTION", price=10),
            _candidate("expensive", "potion_offer", entity_id="BLOOD_POTION", price=60),
        ], world=_world(hp=40))
        baseline = self.policy.recommend(request)
        configured = MerchantPolicy(
            self.repository, ScoringPreferences(survival=1.25),
        ).recommend(request)
        self.assertGreater(baseline.candidates[0].score, baseline.candidates[1].score)
        self.assertEqual(baseline.candidates[0].score, 54.0)
        self.assertEqual(configured.candidates[0].score, 56.0)
        before = {f["code"]: f["delta"] for f in baseline.candidates[0].factors}
        after = {f["code"]: f["delta"] for f in configured.candidates[0].factors}
        self.assertEqual(after["gold_liquidity_cost"], before["gold_liquidity_cost"])
        self.assertEqual(after["hp_recovery"], 10.0)

    def test_full_or_unknown_slots_still_block_blood_potion(self):
        offer = _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", price=0)
        for slots, inventory, gap in (
            (None, (), "resource:potion_slots_missing"),
            (1, ({"potion": "STRENGTH_POTION", "slot": 0},), "resource:potion_replacement_unresolved"),
        ):
            with self.subTest(slots=slots):
                result = self.policy.recommend(_request([
                    offer, _candidate("leave", "leave"),
                ], world=_world(hp=40, max_potion_slots=slots, potions=inventory)))
                self.assertIsNone(result.candidates[0].score)
                self.assertIn(gap, result.data_gaps)

    def test_unavailable_blood_potion_does_not_block_leaving(self):
        offer = _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", price=0)
        unavailable = DecisionCandidate.create(
            offer.candidate_id, offer.payload, eligible=False,
        )
        result = self.policy.recommend(_request([
            unavailable, _candidate("leave", "leave"),
        ], world=_world(hp=None)))
        self.assertIsNone(result.candidates[0].score)
        self.assertEqual(result.recommended_candidate_id, "leave")
        self.assertEqual(result.data_gaps, ())

    def test_additional_costs_or_unaffordable_price_remain_unknown(self):
        for costs, gap in (
            ([{"kind": "gold", "amount": 0}, {"kind": "hp", "amount": 1}], "merchant:costs_invalid:"),
            ([{"kind": "gold", "amount": 200}], "resource:eligibility_gold_inconsistent"),
        ):
            with self.subTest(costs=costs):
                result = self.policy.recommend(_request([
                    _candidate("blood", "potion_offer", entity_id="BLOOD_POTION", costs=costs),
                    _candidate("leave", "leave"),
                ], world=_world(hp=40)))
                self.assertIsNone(result.candidates[0].score)
                self.assertTrue(any(value.startswith(gap) for value in result.data_gaps))
                self.assertIsNone(result.recommended_candidate_id)


if __name__ == "__main__":
    unittest.main()
