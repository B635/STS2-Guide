import math
import os
import tempfile
import unittest

from advisor.decision_core import (
    DecisionCandidate,
    DecisionRequest,
    PolicyRegistry,
    WorldState,
)
from advisor.merchant import MERCHANT_CHOICE, MerchantPolicy
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _world(
    *,
    gold=120,
    deck=None,
    potions=(),
    max_potion_slots=3,
):
    return WorldState.create(
        run_id="merchant-run",
        sequence=3,
        state={
            "character": "IRONCLAD",
            "act": 1,
            "floor": 6,
            "hp": 60,
            "max_hp": 80,
            "gold": gold,
            "energy": 3,
            "deck": deck or [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
            ],
            "relics": [],
            "potions": list(potions),
            "max_potion_slots": max_potion_slots,
        },
    )


def _candidate(candidate_id, kind, *, entity_id=None, price=None, **extra):
    legacy_to_v9 = {
        "card_offer": "card",
        "relic_offer": "relic",
        "potion_offer": "potion",
        "card_remove_service": "card_removal",
    }
    if kind == "leave":
        payload = {
            "candidate_kind": "leave",
            "costs": [],
            **extra,
        }
    else:
        payload = {
            "candidate_kind": "merchant_offer",
            "offer_kind": legacy_to_v9.get(kind, kind),
            "is_stocked": True,
            "costs": (
                [{"kind": "gold", "amount": price, "resource_id": None}]
                if price is not None
                else []
            ),
            **extra,
        }
    if entity_id is not None:
        payload["entity_id"] = entity_id
    return DecisionCandidate.create(
        candidate_id,
        payload,
        label=candidate_id,
    )


def _request(candidates, *, world=None):
    return DecisionRequest.create(
        decision_id="merchant:visit:1",
        decision_type=MERCHANT_CHOICE,
        world=world or _world(),
        candidates=candidates,
    )


class MerchantPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "merchant.db")
        )
        cls.repository.sync_catalog(
            os.path.join(ROOT, "data", "knowledge.json")
        )
        cls.policy = MerchantPolicy(cls.repository)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_registry_dispatch_and_same_card_cheaper_is_better(self):
        request = _request([
            _candidate(
                "slot:0",
                "card_offer",
                entity_id="SHRUG_IT_OFF",
                price=30,
            ),
            _candidate(
                "slot:1",
                "card_offer",
                entity_id="SHRUG_IT_OFF",
                price=90,
            ),
            _candidate("leave", "leave"),
        ])
        result = PolicyRegistry([self.policy]).recommend(request)
        by_id = {row.candidate_id: row for row in result.candidates}
        self.assertGreater(by_id["slot:0"].score, by_id["slot:1"].score)
        self.assertEqual(result.recommended_candidate_id, "slot:0")

    def test_unknown_eligible_offer_forces_uncertain(self):
        result = self.policy.recommend(_request([
            _candidate(
                "unknown",
                "relic_offer",
                entity_id="NOT_A_RELIC",
                price=10,
            ),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIn("unknown_relic:NOT_A_RELIC", result.data_gaps)

    def test_unknown_ineligible_offer_does_not_block(self):
        unknown = DecisionCandidate.create(
            "sold",
            {
                "candidate_kind": "merchant_offer",
                "offer_kind": "relic",
                "is_stocked": False,
                "entity_id": "NOT_A_RELIC",
                "costs": [{
                    "kind": "gold",
                    "amount": 10,
                    "resource_id": None,
                }],
            },
            label="sold",
            eligible=False,
        )
        result = self.policy.recommend(_request([
            unknown,
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "recommend")
        self.assertEqual(result.recommended_candidate_id, "leave")

    def test_missing_actual_price_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate(
                "card",
                "card_offer",
                entity_id="SHRUG_IT_OFF",
            ),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("merchant:costs_invalid")
            for gap in result.data_gaps
        ))

    def test_affordability_inconsistency_fails_closed(self):
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "card",
                    "card_offer",
                    entity_id="SHRUG_IT_OFF",
                    price=200,
                ),
                _candidate("leave", "leave"),
            ],
            world=_world(gold=50),
        ))
        self.assertEqual(result.status, "uncertain")
        self.assertIn(
            "resource:eligibility_gold_inconsistent",
            result.data_gaps,
        )

    def test_full_potion_slots_require_explicit_replacement_support(self):
        full = _world(
            potions=(
                {"potion": "STRENGTH_POTION", "slot": 0},
                {"potion": "REGEN_POTION", "slot": 1},
                {"potion": "GLOWWATER_POTION", "slot": 2},
            )
        )
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "potion",
                    "potion_offer",
                    entity_id="STRENGTH_POTION",
                    price=20,
                ),
                _candidate("leave", "leave"),
            ],
            world=full,
        ))
        self.assertEqual(result.status, "uncertain")
        self.assertIn(
            "resource:potion_replacement_unresolved",
            result.data_gaps,
        )

    def test_remove_status_card_can_beat_leaving(self):
        result = self.policy.recommend(_request(
            [
                _candidate(
                    "remove",
                    "card_remove_service",
                    price=30,
                ),
                _candidate("leave", "leave"),
            ],
            world=_world(deck=[
                {"card": "WOUND", "count": 1},
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
            ]),
        ))
        self.assertEqual(result.recommended_candidate_id, "remove")

    def test_unknown_offer_kind_fails_closed(self):
        result = self.policy.recommend(_request([
            _candidate("mystery", "mystery_offer", price=0),
            _candidate("leave", "leave"),
        ]))
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(any(
            gap.startswith("merchant:unknown_offer_kind")
            for gap in result.data_gaps
        ))

    def test_scores_are_finite_and_bounded(self):
        result = self.policy.recommend(_request([
            _candidate(
                "relic",
                "relic_offer",
                entity_id="SHURIKEN",
                price=50,
            ),
            _candidate("leave", "leave"),
        ]))
        for row in result.candidates:
            if row.score is not None:
                self.assertTrue(math.isfinite(row.score))
                self.assertGreaterEqual(row.score, 0)
                self.assertLessEqual(row.score, 100)


if __name__ == "__main__":
    unittest.main()
