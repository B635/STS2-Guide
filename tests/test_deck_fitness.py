import copy
import json
import tempfile
import unittest
from pathlib import Path

from advisor.decision_core import PolicyRegistry
from advisor.deck_fitness import DECK_FITNESS_POLICY_VERSION, DeckFitnessPolicy
from advisor.golden_set import card_reward_request, load_golden_set, scoring_view
from advisor.scoring_preferences import ScoringPreferences
from storage.relational import RelationalRepository
from storage.release_database import build_release_database


ROOT = Path(__file__).resolve().parents[1]


def card(card_id, count=1, upgrades=0):
    return {
        "card": card_id, "count": count, "upgrades": upgrades,
        "enchantment": None, "enchantment_amount": None,
        "affliction": None, "affliction_amount": None,
    }


class DeckFitnessPolicyTests(unittest.TestCase):
    """v2 behaviour through the public policy entry and a real release DB."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        release = Path(cls.temporary.name) / "release.db"
        build_release_database(
            release, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json"
        )
        cls.repository = RelationalRepository(str(release))
        example = json.loads(
            (ROOT / "protocol" / "state-event.example.json").read_text(encoding="utf-8")
        )
        cls.view = scoring_view(example)
        mapped = {s["id"]: s for s in load_golden_set()}
        cls.mapped_view = mapped["ironclad-f15-pommel-havoc-unrelenting"]["scoring_view"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def recommend(self, deck, offered, *, hp=80, relics=(), view=None, preferences=None):
        view = copy.deepcopy(view or self.mapped_view)
        view["state"].update({"deck": deck, "hp": hp, "max_hp": 80})
        view["state"]["relics"] = list(relics)
        view["state"]["relic_states"] = []
        template = view["candidates"][0]
        skip = next(c for c in view["candidates"] if c["candidate_id"] == "skip")
        candidates = []
        for index, card_id in enumerate(offered):
            candidate = copy.deepcopy(template)
            candidate.update(candidate_id=f"{index}:{card_id}", entity_id=card_id, label=card_id)
            candidate["payload"]["card"] = card_id
            candidates.append(candidate)
        view["candidates"] = [*candidates, copy.deepcopy(skip)]
        policy = DeckFitnessPolicy(self.repository, preferences or ScoringPreferences())
        return PolicyRegistry([policy]).recommend(card_reward_request(view))

    @staticmethod
    def scores(recommendation):
        return {row.candidate_id: row.score for row in recommendation.candidates}

    def test_missing_defense_makes_a_block_card_the_best_choice(self):
        result = self.recommend(
            [card("STRIKE_IRONCLAD", 9), card("BASH")], ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"]
        )
        self.assertEqual(result.recommended_candidate_id, "1:DEFEND_IRONCLAD")
        top = next(r for r in result.candidates if r.candidate_id == "1:DEFEND_IRONCLAD").factors[0]
        self.assertIn("DEFEND_IRONCLAD", top["message"])
        self.assertEqual(top["code"], "deck_fitness:defense")
        self.assertGreater(top["delta"], 0)

    def test_low_hp_makes_defense_matter_more_without_inventing_a_bigger_need(self):
        deck = [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4), card("BASH")]
        full = self.recommend(deck, ["DEFEND_IRONCLAD"], hp=80)
        low = self.recommend(deck, ["DEFEND_IRONCLAD"], hp=20)

        def gain(result):
            return self.scores(result)["0:DEFEND_IRONCLAD"] - self.scores(result)["skip"]

        self.assertGreater(gain(low), gain(full))
        self.assertEqual(low.payload["needs"]["defense"], full.payload["needs"]["defense"])
        self.assertGreater(low.payload["weights"]["defense"], full.payload["weights"]["defense"])

    def test_skip_is_computed_and_a_marginal_card_can_fall_below_it(self):
        small = self.recommend(
            [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4), card("BASH")],
            ["STRIKE_IRONCLAD"],
        )
        bloated = self.recommend(
            [card("STRIKE_IRONCLAD", 12), card("DEFEND_IRONCLAD", 12), card("BASH", 2)],
            ["STRIKE_IRONCLAD"],
        )
        self.assertNotEqual(self.scores(small)["skip"], self.scores(bloated)["skip"])
        for result in (small, bloated):
            self.assertLess(self.scores(result)["0:STRIKE_IRONCLAD"], self.scores(result)["skip"])
            self.assertEqual(result.recommended_candidate_id, "skip")
            self.assertEqual(result.status, "skip")

    def test_scores_have_one_decimal_and_no_community_prior(self):
        result = self.recommend([card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)], ["BASH"])
        self.assertEqual(result.policy_version, DECK_FITNESS_POLICY_VERSION)
        for row in result.candidates:
            self.assertEqual(row.score, round(row.score, 1))
            self.assertTrue(0 <= row.score <= 100)
            self.assertFalse(any("community" in f["code"] or "tier" in f["code"] for f in row.factors))

    def test_unknown_candidate_gets_no_score_and_is_not_ranked(self):
        result = self.recommend(
            [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)], ["TAUNT", "DEFEND_IRONCLAD"]
        )
        taunt = next(r for r in result.candidates if r.candidate_id == "0:TAUNT")
        self.assertIsNone(taunt.score)
        self.assertIsNone(taunt.rank)
        self.assertEqual(taunt.factors[0]["code"], "insufficient_data")
        self.assertIn("card_fact_missing:TAUNT", result.data_gaps)
        defend = next(r for r in result.candidates if r.candidate_id == "1:DEFEND_IRONCLAD")
        self.assertIsNotNone(defend.rank)
        self.assertNotEqual(result.recommended_candidate_id, "0:TAUNT")

    def test_all_unknown_candidates_leave_the_decision_uncertain(self):
        result = self.recommend([card("STRIKE_IRONCLAD", 5)], ["TAUNT", "STONE_ARMOR"])
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIsNotNone(self.scores(result)["skip"])

    def test_unknown_deck_cards_only_dilute_and_are_disclosed(self):
        known = self.recommend([card("STRIKE_IRONCLAD", 5)], ["DEFEND_IRONCLAD"])
        diluted = self.recommend([card("STRIKE_IRONCLAD", 5), card("TAUNT", 5)], ["DEFEND_IRONCLAD"])
        self.assertLess(self.scores(diluted)["skip"], self.scores(known)["skip"])
        defend = next(r for r in diluted.candidates if r.candidate_id == "0:DEFEND_IRONCLAD")
        self.assertEqual(defend.factors[0]["code"], "unmodeled_deck_cards")
        self.assertIn("TAUNT", defend.factors[0]["message"])
        self.assertEqual(defend.dimensions["data_completeness"], 0.0)

    def test_relics_without_verified_facts_are_disclosed_not_scored(self):
        # War Paint and Cloak Clasp were the live counter-examples; their
        # reviewed facts are exercised with ticket 04.
        result = self.recommend(
            [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)],
            ["DEFEND_IRONCLAD"],
            relics=("WAR_PAINT", "CLOAK_CLASP"),
        )
        defend = next(r for r in result.candidates if r.candidate_id == "0:DEFEND_IRONCLAD")
        self.assertEqual(defend.factors[0]["code"], "unmodeled_relics")
        self.assertIn("WAR_PAINT", defend.factors[0]["message"])
        self.assertFalse(any(f["code"].startswith("deck_fitness:") and "WAR_PAINT" in f["message"]
                             for f in defend.factors))
        self.assertIn("relic_fact_missing:WAR_PAINT", result.data_gaps)
        self.assertIn("relic_fact_missing:CLOAK_CLASP", result.data_gaps)

    def test_survival_preference_raises_the_value_of_defense_once(self):
        deck = [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 2)]
        neutral = self.recommend(deck, ["DEFEND_IRONCLAD"])
        survival = self.recommend(
            deck, ["DEFEND_IRONCLAD"],
            preferences=ScoringPreferences(survival=1.25),
        )
        # Exactly one multiplication of the defense need weight, nothing else.
        self.assertAlmostEqual(
            survival.payload["weights"]["defense"], neutral.payload["weights"]["defense"] * 1.25
        )
        self.assertEqual(
            {k: v for k, v in survival.payload["weights"].items() if k != "defense"},
            {k: v for k, v in neutral.payload["weights"].items() if k != "defense"},
        )

        def gain(result):
            return self.scores(result)["0:DEFEND_IRONCLAD"] - self.scores(result)["skip"]

        self.assertGreater(gain(survival), gain(neutral))

    def test_needs_follow_the_recommended_route_when_the_map_is_known(self):
        deck = [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)]
        mapped = self.recommend(deck, ["DEFEND_IRONCLAD"])
        self.assertNotIn("route_context_missing", mapped.data_gaps)
        self.assertIn("BOSS", mapped.payload["fight_mix"])

    def test_without_a_map_needs_are_unknown_and_nothing_is_scored(self):
        result = self.recommend(
            [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)], ["DEFEND_IRONCLAD"],
            view=self.view,
        )
        self.assertEqual(result.status, "uncertain")
        self.assertIn("situation_needs_unavailable", result.data_gaps)
        self.assertTrue(all(row.score is None for row in result.candidates))


if __name__ == "__main__":
    unittest.main()


class DeckFitnessRuleTests(unittest.TestCase):
    """Two spec rules checked directly, as no single deck isolates them."""

    def test_a_shortfall_always_costs_more_than_the_same_surplus_earns(self):
        from advisor.deck_fitness import DIMENSIONS, Needs, fitness

        params = {"satisfaction_cap": 3.0, "surplus_slope": 0.25}
        needs = Needs(
            values={d: 10.0 for d in DIMENSIONS},
            weights={d: (1.0 if d == "defense" else 0.0) for d in DIMENSIONS},
            fight_turns=4.0, extra_enemies=0.0, incoming_attack=0.0, fight_mix={},
        )

        def value(defense):
            return fitness({d: (defense if d == "defense" else 0.0) for d in DIMENSIONS},
                           needs, params).raw

        for below in (0.0, 5.0, 8.0, 9.0, 9.5):
            step = 0.5
            self.assertGreater(value(below + step) - value(below),
                               value(10.0 + step) - value(10.0))

    def test_fallback_counts_only_fights_on_paths_to_the_boss(self):
        from advisor.deck_fitness import _reachable_fight_share
        from advisor.decision_core import WorldMap

        world_map = WorldMap.create({
            "nodes": [
                {"node_id": "1:0", "kind": "MONSTER", "row": 1, "col": 0, "edges": ["2:0"]},
                {"node_id": "1:1", "kind": "ELITE", "row": 1, "col": 1, "edges": ["2:0"]},
                {"node_id": "1:2", "kind": "ELITE", "row": 1, "col": 2, "edges": []},
                {"node_id": "2:0", "kind": "BOSS", "row": 2, "col": 0, "edges": []},
            ],
            "player_row": 0,
            "available_next_node_ids": ["1:0", "1:1"],
            "boss_node_ids": ["2:0"],
        })
        share = _reachable_fight_share(world_map)
        # Two equally likely paths: half a monster, half an elite, one boss.
        # The dead-end elite 1:2 is unreachable from the next nodes' paths.
        self.assertEqual(share, {"MONSTER": 0.5, "ELITE": 0.5, "BOSS": 1.0})
