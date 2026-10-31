import copy
import math
import tempfile
import unittest
from pathlib import Path

from advisor.decision_core import DecisionCandidate, DecisionRequest, PolicyRegistry
from advisor.golden_set import load_golden_set, world_request
from advisor.merchant import MERCHANT_CHOICE, MerchantPolicy
from storage.relational import RelationalRepository
from storage.release_database import build_release_database


ROOT = Path(__file__).resolve().parents[1]


def card(card_id, count=1, upgrades=0):
    return {
        "card": card_id, "count": count, "upgrades": upgrades,
        "enchantment": None, "enchantment_amount": None,
        "affliction": None, "affliction_amount": None,
    }


STARTER = [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4), card("BASH")]


def _candidate(candidate_id, kind, *, entity_id=None, price=None, **extra):
    if kind == "leave":
        payload = {"candidate_kind": "leave", "costs": [], **extra}
    else:
        payload = {
            "candidate_kind": "merchant_offer",
            "offer_kind": kind,
            "is_stocked": True,
            **({"upgrades": 0} if kind == "card" else {}),
            "costs": [{"kind": "gold", "amount": price, "resource_id": None}] if price is not None else [],
            **extra,
        }
    if entity_id is not None:
        payload["entity_id"] = entity_id
    return DecisionCandidate.create(candidate_id, payload, label=candidate_id)


def _targets(deck):
    """Every card instance removable, as the Mod reports it (verified targets)."""

    targets = []
    for entry in deck:
        for _ in range(entry["count"]):
            targets.append({"target_id": f"fixture:{len(targets)}", "card": entry["card"], "upgrades": entry["upgrades"]})
    return {"operation": "remove", "source_code": "fixture:merchant", "targets": targets}


class MerchantPolicyTests(unittest.TestCase):
    """Shop v2 on a real release database and a real mapped situation."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        release = Path(cls.temporary.name) / "release.db"
        build_release_database(release, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json")
        cls.repository = RelationalRepository(str(release))
        cls.policy = MerchantPolicy(cls.repository)
        cls.view = next(s for s in load_golden_set() if s["id"] == "ironclad-f15-pommel-havoc-unrelenting")["scoring_view"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def world(self, *, deck=None, gold=300, hp=60):
        view = copy.deepcopy(self.view)
        view["state"].update(deck=deck or STARTER, gold=gold, hp=hp, max_hp=80, relics=[], relic_states=[], potions=[])
        return world_request(view).world

    def recommend(self, candidates, **world):
        return self.policy.recommend(DecisionRequest.create(
            decision_id="merchant:visit:1", decision_type=MERCHANT_CHOICE,
            world=self.world(**world), candidates=candidates,
        ))

    @staticmethod
    def row(result, candidate_id):
        return next(c for c in result.candidates if c.candidate_id == candidate_id)

    def test_registry_dispatch_and_the_same_card_cheaper_is_better(self):
        request = DecisionRequest.create(
            decision_id="merchant:visit:1", decision_type=MERCHANT_CHOICE, world=self.world(),
            candidates=[
                _candidate("cheap", "card", entity_id="SHRUG_IT_OFF", price=40),
                _candidate("dear", "card", entity_id="SHRUG_IT_OFF", price=90),
                _candidate("leave", "leave"),
            ],
        )
        result = PolicyRegistry([self.policy]).recommend(request)
        self.assertGreater(self.row(result, "cheap").score, self.row(result, "dear").score)
        codes = [f["code"] for f in self.row(result, "cheap").factors]
        self.assertIn("merchant:deck", codes)
        self.assertIn("merchant:price", codes)
        self.assertEqual(self.row(result, "leave").score, 50.0)

    def test_unknown_eligible_offers_block_and_ineligible_ones_do_not(self):
        blocked = self.recommend([
            _candidate("known", "card", entity_id="SHRUG_IT_OFF", price=50),
            _candidate("mystery", "card", entity_id="NOT_A_CARD", price=50),
        ])
        self.assertEqual(blocked.status, "uncertain")
        self.assertIn("unknown_card:NOT_A_CARD", blocked.data_gaps)
        unavailable = DecisionCandidate.create(
            "mystery", {**_candidate("x", "card", entity_id="NOT_A_CARD", price=50).payload},
            label="mystery", eligible=False,
        )
        fine = self.recommend([_candidate("known", "card", entity_id="SHRUG_IT_OFF", price=50), unavailable])
        self.assertIsNotNone(fine.recommended_candidate_id)

    def test_malformed_offers_fail_closed_with_their_own_gap(self):
        for candidate, gap in (
            (_candidate("a", "card", entity_id="SHRUG_IT_OFF"), "merchant:costs_invalid"),
            (_candidate("b", "card", entity_id="SHRUG_IT_OFF", price=50, upgrades=None), "merchant:card_upgrades_unknown"),
            (_candidate("c", "mystery_kind", entity_id="SHRUG_IT_OFF", price=50), "merchant:unknown_offer_kind"),
            (_candidate("d", "card", price=50), "merchant:entity_id_missing"),
        ):
            with self.subTest(gap=gap):
                result = self.recommend([candidate, _candidate("leave", "leave")])
                self.assertEqual(result.status, "uncertain")
                self.assertTrue(any(g.startswith(gap) for g in result.data_gaps), result.data_gaps)

    def test_non_gold_costs_are_not_silently_ignored(self):
        for cost in ({"kind": "hp", "amount": 10, "resource_id": None}, {"kind": "max_hp", "amount": 5, "resource_id": None}):
            with self.subTest(cost=cost["kind"]):
                offer = _candidate("a", "card", entity_id="SHRUG_IT_OFF", price=50)
                payload = {**offer.payload, "costs": [*offer.payload["costs"], cost]}
                result = self.recommend([DecisionCandidate.create("a", payload, label="a")])
                self.assertIsNone(self.row(result, "a").score)

    def test_a_potion_is_shown_unscored_and_does_not_block_the_shop(self):
        result = self.recommend([
            _candidate("potion", "potion", entity_id="FIRE_POTION", price=50),
            _candidate("card", "card", entity_id="SHRUG_IT_OFF", price=50),
            _candidate("leave", "leave"),
        ])
        self.assertIsNone(self.row(result, "potion").score)
        self.assertIsNone(self.row(result, "potion").rank)
        self.assertIn("merchant:potion_unmodeled:FIRE_POTION", result.data_gaps)
        self.assertNotEqual(result.recommended_candidate_id, "potion")
        self.assertIsNotNone(result.recommended_candidate_id)

    def test_removing_a_curse_beats_leaving(self):
        deck = [*STARTER, card("DOUBT")]
        result = self.recommend(
            [_candidate("remove", "card_removal", price=75, deck_edit_targets=_targets(deck)), _candidate("leave", "leave")],
            deck=deck,
        )
        self.assertEqual(result.recommended_candidate_id, "remove")
        deck_factor = next(f for f in self.row(result, "remove").factors if f["code"] == "merchant:deck")
        self.assertGreater(deck_factor["delta"], 0)

    def test_removal_only_takes_a_verified_target_and_needs_targets(self):
        deck = [*STARTER, card("DOUBT")]
        only_strike = {"operation": "remove", "source_code": "fixture:merchant",
                       "targets": [{"target_id": "fixture:0", "card": "STRIKE_IRONCLAD", "upgrades": 0}]}
        result = self.recommend([_candidate("remove", "card_removal", price=75, deck_edit_targets=only_strike)], deck=deck)
        deck_factor = next(f for f in self.row(result, "remove").factors if f["code"] == "merchant:deck")
        self.assertNotIn("DOUBT", deck_factor["message"])
        missing = self.recommend([_candidate("remove", "card_removal", price=75)], deck=deck)
        self.assertIn("deck_edit:targets_unknown", missing.data_gaps)

    def test_a_relic_with_unreviewed_out_of_combat_hooks_is_only_a_lower_bound(self):
        result = self.recommend([_candidate("relic", "relic", entity_id="WAR_PAINT", price=150), _candidate("leave", "leave")])
        self.assertIn("relic_choice:out_of_combat_unreviewed:WAR_PAINT", result.data_gaps)
        self.assertIsNotNone(self.row(result, "relic").score)

    def test_scores_are_finite_and_bounded(self):
        result = self.recommend([
            _candidate("card", "card", entity_id="INFLAME", price=75),
            _candidate("relic", "relic", entity_id="ANCHOR", price=175),
            _candidate("leave", "leave"),
        ])
        for row in result.candidates:
            if row.score is not None:
                self.assertTrue(math.isfinite(row.score) and 0 <= row.score <= 100)


if __name__ == "__main__":
    unittest.main()
