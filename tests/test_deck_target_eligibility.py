import copy
import os
import tempfile
import unittest

from advisor.campfire import CAMPFIRE_ACTION, CampfirePolicy
from advisor.deck_edit import DECK_EDIT, DeckEditPolicy
from advisor.decision_core import DecisionCandidate, DecisionRequest, WorldState
from advisor.merchant import MERCHANT_CHOICE, MerchantPolicy
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _targets(operation, *cards):
    return {
        "operation": operation,
        "source_code": "fixture:verified_game_targets",
        "targets": [
            {"target_id": f"target:{index}", "card": card, "upgrades": upgrades}
            for index, (card, upgrades) in enumerate(cards)
        ],
    }


class DeckTargetEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(os.path.join(cls.tempdir.name, "targets.db"))
        cls.repository.sync_catalog(os.path.join(ROOT, "data", "knowledge.json"))

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def _request(self, operation, *, targets=None, deck=None, child=False,
                 upgrades=None, sequence=1, include_targets=True):
        world = WorldState.create(
            run_id="target-run", sequence=sequence,
            state={
                "character": "IRONCLAD", "act": 1, "floor": 7,
                "hp": 80, "max_hp": 80, "gold": 120, "energy": 3,
                "deck": deck if deck is not None else [
                    {"card": "STRIKE_IRONCLAD"}, {"card": "WOUND"},
                ],
                "relics": [], "potions": [], "max_potion_slots": 3,
            },
        )
        if child:
            decision_type = DECK_EDIT
            payload = {"candidate_kind": "deck_edit", "operation": operation,
                       "entity_id": "STRIKE_IRONCLAD", "costs": []}
            if upgrades is not None:
                payload["upgrades"] = upgrades
        elif operation == "remove":
            decision_type = MERCHANT_CHOICE
            payload = {
                "candidate_kind": "merchant_offer", "offer_kind": "card_removal",
                "is_stocked": True, "costs": [{"kind": "gold", "amount": 30}],
            }
        else:
            decision_type = CAMPFIRE_ACTION
            payload = {
                "candidate_kind": "rest_action", "action_id": "SMITH", "costs": [],
                "effects": [{"kind": "upgrade_card", "entity_type": "cards",
                             "target_mode": "choose", "source_code": "fixture:smith"}],
            }
        if not child and include_targets:
            payload["deck_edit_targets"] = targets
        return DecisionRequest.create(
            decision_id="targets:decision", decision_type=decision_type, world=world,
            candidates=[DecisionCandidate.create("deck:0:upgrades:0", payload)],
        )

    def _recommend(self, operation, **kwargs):
        request = self._request(operation, **kwargs)
        policy = {
            MERCHANT_CHOICE: MerchantPolicy,
            CAMPFIRE_ACTION: CampfirePolicy,
            DECK_EDIT: DeckEditPolicy,
        }[request.decision_type](self.repository)
        return policy.recommend(request)

    def test_missing_or_null_parent_targets_never_fall_back_to_deck(self):
        for operation in ("remove", "upgrade"):
            for include_targets in (False, True):
                with self.subTest(operation=operation, explicit_null=include_targets):
                    result = self._recommend(operation, include_targets=include_targets)
                    self.assertIsNone(result.candidates[0].score)
                    self.assertIn("deck_edit:targets_unknown", result.data_gaps)

    def test_complete_empty_parent_targets_are_known_unavailable(self):
        for operation in ("remove", "upgrade"):
            with self.subTest(operation=operation):
                result = self._recommend(operation, targets=_targets(operation))
                self.assertIsNone(result.candidates[0].score)
                self.assertIn(f"deck_edit:no_{operation}_target", result.data_gaps)
                self.assertNotIn("deck_edit:targets_unknown", result.data_gaps)

    def test_smith_ignores_unmodeled_cards_outside_verified_targets(self):
        result = self._recommend(
            "upgrade", targets=_targets("upgrade", ("STRIKE_IRONCLAD", 0)),
            deck=[{"card": "STRIKE_IRONCLAD"}, {"card": "BASH"},
                  {"card": "NOT_IN_CATALOG"}],
        )
        self.assertEqual(result.data_gaps, ())
        self.assertIsNotNone(result.candidates[0].score)

    def test_parent_target_set_rejects_invalid_or_mismatched_instances(self):
        valid = _targets("remove", ("STRIKE_IRONCLAD", 0))
        cases = []
        for field, value in (("operation", "upgrade"), ("source_code", " ")):
            case = copy.deepcopy(valid)
            case[field] = value
            cases.append(case)
        for field, value in (("card", "DEFEND_IRONCLAD"), ("upgrades", 1),
                             ("upgrades", True), ("target_id", " ")):
            case = copy.deepcopy(valid)
            case["targets"][0][field] = value
            cases.append(case)
        duplicate_id = copy.deepcopy(valid)
        duplicate_id["targets"].append(copy.deepcopy(duplicate_id["targets"][0]))
        cases.append(duplicate_id)
        cases.append(_targets("remove", ("STRIKE_IRONCLAD", 0), ("STRIKE_IRONCLAD", 0)))
        for targets in cases:
            with self.subTest(targets=targets):
                result = self._recommend("remove", targets=targets)
                self.assertIsNone(result.candidates[0].score)
                self.assertTrue(result.data_gaps)

    def test_distinct_same_card_targets_use_current_instance_counts(self):
        targets = _targets("remove", ("STRIKE_IRONCLAD", 0), ("STRIKE_IRONCLAD", 0))
        valid = self._recommend(
            "remove", targets=targets,
            deck=[{"card": "STRIKE_IRONCLAD", "count": 2}],
        )
        stale = self._recommend(
            "remove", targets=targets,
            deck=[{"card": "STRIKE_IRONCLAD", "count": 1}], sequence=2,
        )
        # Two Strikes in the deck: both targets are valid (shop v2 needs the
        # full engine to score the removal itself; see test_merchant_decision).
        self.assertFalse([g for g in valid.data_gaps if g.startswith("deck_edit:")])
        self.assertIsNone(stale.candidates[0].score)
        self.assertIn("deck_edit:targets_state_mismatch", stale.data_gaps)

    def test_parent_repeated_upgrade_is_unknown_not_silently_excluded(self):
        result = self._recommend(
            "upgrade", targets=_targets("upgrade", ("STRIKE_IRONCLAD", 0),
                                         ("STRIKE_IRONCLAD", 1)),
            deck=[{"card": "STRIKE_IRONCLAD", "upgrades": 0},
                  {"card": "STRIKE_IRONCLAD", "upgrades": 1}],
        )
        self.assertIsNone(result.candidates[0].score)
        self.assertIn("deck_edit:repeat_upgrade_unmodeled", result.data_gaps)

    def test_child_upgrade_requires_explicit_state_not_candidate_id_parsing(self):
        unknown = self._recommend("upgrade", child=True)
        known = self._recommend("upgrade", child=True, upgrades=0)
        self.assertIsNone(unknown.candidates[0].score)
        self.assertIn("deck_edit:target_upgrades_unknown", unknown.data_gaps)
        self.assertEqual(known.data_gaps, ())
        self.assertIsNotNone(known.candidates[0].score)

    def test_child_repeated_upgrade_is_unknown(self):
        result = self._recommend(
            "upgrade", child=True, upgrades=1,
            deck=[{"card": "STRIKE_IRONCLAD", "upgrades": 1}],
        )
        self.assertIsNone(result.candidates[0].score)
        self.assertIn("deck_edit:repeat_upgrade_unmodeled", result.data_gaps)

    def test_child_removal_uses_matching_upgrade_group(self):
        result = self._recommend(
            "remove", child=True, upgrades=1,
            deck=[{"card": "STRIKE_IRONCLAD", "upgrades": 0, "count": 1},
                  {"card": "STRIKE_IRONCLAD", "upgrades": 1, "count": 3}],
        )
        self.assertEqual(result.data_gaps, ())
        factor = next(f for f in result.candidates[0].factors
                      if f["code"] == "deck_remove_opportunity")
        self.assertEqual(factor["delta"], 7.5)

    def test_unknown_card_inside_legal_targets_blocks_comparison(self):
        result = self._recommend(
            "upgrade", targets=_targets("upgrade", ("STRIKE_IRONCLAD", 0),
                                         ("NOT_IN_CATALOG", 0)),
            deck=[{"card": "STRIKE_IRONCLAD"}, {"card": "NOT_IN_CATALOG"}],
        )
        self.assertIsNone(result.candidates[0].score)
        self.assertIn("unknown_card:NOT_IN_CATALOG", result.data_gaps)

    def test_multi_selection_does_not_reuse_a_single_target_set(self):
        for mode in ("amount", "effects"):
            request = self._request("upgrade", targets=_targets("upgrade", ("STRIKE_IRONCLAD", 0)))
            payload = dict(request.candidates[0].payload)
            effects = [dict(effect) for effect in payload["effects"]]
            if mode == "amount":
                effects[0]["amount"] = 2
            else:
                effects *= 2
            payload["effects"] = effects
            request = DecisionRequest.create(
                decision_id=request.decision_id, decision_type=request.decision_type,
                world=request.world, candidates=[DecisionCandidate.create("smith", payload)],
            )
            result = CampfirePolicy(self.repository).recommend(request)
            self.assertIsNone(result.candidates[0].score)


if __name__ == "__main__":
    unittest.main()
