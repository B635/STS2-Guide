import copy
import tempfile
import unittest
from pathlib import Path

from advisor.decision_core import DecisionCandidate, DecisionRequest, PolicyRegistry
from advisor.golden_set import load_golden_set, world_request
from advisor.neow import NEOW_BLESSING, NeowPolicy
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


def option(index, relic, *, key=None, eligible=True):
    key = key or relic
    return DecisionCandidate.create(
        f"{index}:{key}",
        {"candidate_kind": "neow_blessing", "entity_id": relic, "blessing_id": key,
         "stage_id": "INITIAL", "option_id": key, "effects": []},
        label=key, display_index=index, eligible=eligible,
    )


class NeowPolicyTests(unittest.TestCase):
    """Neow / ancient relic choices on the real release database (TASK-027)."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        release = Path(cls.temporary.name) / "release.db"
        build_release_database(release, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json")
        cls.repository = RelationalRepository(str(release))
        cls.policy = NeowPolicy(cls.repository)
        cls.view = next(s for s in load_golden_set() if s["id"] == "ironclad-f15-pommel-havoc-unrelenting")["scoring_view"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def request(self, candidates, *, gold=99, hp=80, decision="neow:run:1"):
        view = copy.deepcopy(self.view)
        view["state"].update(deck=STARTER, gold=gold, hp=hp, max_hp=80, relics=["BURNING_BLOOD"],
                             relic_states=[{"relic": "BURNING_BLOOD", "display_amount": None,
                                            "stack_count": 1, "status": "NORMAL"}], potions=[])
        return DecisionRequest.create(
            decision_id=decision, decision_type=NEOW_BLESSING, world=world_request(view).world,
            candidates=candidates,
        )

    @staticmethod
    def row(result, candidate_id):
        return next(c for c in result.candidates if c.candidate_id == candidate_id)

    def test_relic_options_are_scored_from_their_pickup_effects(self):
        result = PolicyRegistry([self.policy]).recommend(self.request([
            option(0, "GOLDEN_PEARL"), option(1, "PRECISE_SCISSORS"), option(2, "NUTRITIOUS_OYSTER"),
        ]))
        self.assertEqual(result.status, "recommend")
        codes = {c.candidate_id: [f["code"] for f in c.factors] for c in result.candidates}
        self.assertIn("resource:gold", codes["0:GOLDEN_PEARL"])
        self.assertIn("resource:max_hp", codes["2:NUTRITIOUS_OYSTER"])
        scissors = self.row(result, "1:PRECISE_SCISSORS")
        deck = next(f for f in scissors.factors if f["code"] == "relic_choice:deck")
        self.assertIn("移除", deck["message"])
        for row in result.candidates:
            self.assertIsNotNone(row.score)
            self.assertTrue(0 <= row.score <= 100)

    def test_a_relic_that_changes_nothing_in_the_deck_shows_no_deck_change(self):
        result = self.policy.recommend(self.request([option(0, "GOLDEN_PEARL"), option(1, "NUTRITIOUS_OYSTER")]))
        deck = [f for f in self.row(result, "0:GOLDEN_PEARL").factors if f["code"] == "relic_choice:deck"]
        self.assertTrue(not deck or abs(deck[0]["delta"]) < 1e-6)

    def test_gold_is_worth_less_when_already_rich(self):
        poor = self.policy.recommend(self.request([option(0, "GOLDEN_PEARL"), option(1, "NUTRITIOUS_OYSTER")], gold=0))
        rich = self.policy.recommend(self.request([option(0, "GOLDEN_PEARL"), option(1, "NUTRITIOUS_OYSTER")], gold=900))

        def gold_points(result):
            return next(f["delta"] for f in self.row(result, "0:GOLDEN_PEARL").factors if f["code"] == "resource:gold")
        self.assertGreater(gold_points(poor), gold_points(rich))

    def test_options_without_a_relic_or_a_fact_fail_closed(self):
        no_relic = DecisionCandidate.create(
            "x", {"candidate_kind": "neow_blessing", "blessing_id": "X", "stage_id": "INITIAL",
                  "option_id": "X", "effects": []}, label="X",
        )
        result = self.policy.recommend(self.request([option(0, "GOLDEN_PEARL"), no_relic]))
        self.assertEqual(result.status, "uncertain")
        self.assertIn("neow:relic_id_missing", result.data_gaps)
        unknown = self.policy.recommend(self.request([option(0, "GOLDEN_PEARL"), option(1, "NOT_A_RELIC")]))
        self.assertEqual(unknown.status, "uncertain")
        self.assertIn("relic_fact_missing:NOT_A_RELIC", unknown.data_gaps)

    def test_an_ineligible_option_does_not_block(self):
        result = self.policy.recommend(self.request([
            option(0, "GOLDEN_PEARL"), option(1, "NOT_A_RELIC", eligible=False),
        ]))
        self.assertEqual(result.recommended_candidate_id, "0:GOLDEN_PEARL")

    def test_sea_glass_reads_its_character_from_the_option_key(self):
        result = self.policy.recommend(self.request([option(0, "SEA_GLASS", key="SILENT"), option(1, "GOLDEN_PEARL")]))
        sea = self.row(result, "0:SILENT")
        self.assertIsNotNone(sea.score)
        self.assertFalse([g for g in sea.data_gaps if "add_card:pool" in g])

    def test_the_same_choice_gives_the_same_answer(self):
        candidates = [option(0, "NEW_LEAF"), option(1, "ARCANE_SCROLL"), option(2, "LEAD_PAPERWEIGHT")]
        first = self.policy.recommend(self.request(candidates)).as_dict()
        second = self.policy.recommend(self.request(candidates)).as_dict()
        self.assertEqual(first, second)

    def test_another_decision_type_is_refused(self):
        request = self.request([option(0, "GOLDEN_PEARL")])
        with self.assertRaises(ValueError):
            self.policy.recommend(DecisionRequest.create(
                decision_id="x", decision_type="card_reward", world=request.world, candidates=request.candidates,
            ))


if __name__ == "__main__":
    unittest.main()
