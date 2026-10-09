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
            [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)], ["UNREVIEWED_CARD", "DEFEND_IRONCLAD"]
        )
        taunt = next(r for r in result.candidates if r.candidate_id == "0:UNREVIEWED_CARD")
        self.assertIsNone(taunt.score)
        self.assertIsNone(taunt.rank)
        self.assertEqual(taunt.factors[0]["code"], "insufficient_data")
        self.assertIn("card_fact_missing:UNREVIEWED_CARD", result.data_gaps)
        defend = next(r for r in result.candidates if r.candidate_id == "1:DEFEND_IRONCLAD")
        self.assertIsNotNone(defend.rank)
        self.assertNotEqual(result.recommended_candidate_id, "0:UNREVIEWED_CARD")

    def test_all_unknown_candidates_leave_the_decision_uncertain(self):
        result = self.recommend([card("STRIKE_IRONCLAD", 5)], ["UNREVIEWED_CARD", "UNREVIEWED_CARD_B"])
        self.assertEqual(result.status, "uncertain")
        self.assertIsNone(result.recommended_candidate_id)
        self.assertIsNotNone(self.scores(result)["skip"])

    def test_unknown_deck_cards_only_dilute_and_are_disclosed(self):
        known = self.recommend([card("STRIKE_IRONCLAD", 5)], ["DEFEND_IRONCLAD"])
        diluted = self.recommend([card("STRIKE_IRONCLAD", 5), card("UNREVIEWED_CARD", 5)], ["DEFEND_IRONCLAD"])
        self.assertLess(self.scores(diluted)["skip"], self.scores(known)["skip"])
        defend = next(r for r in diluted.candidates if r.candidate_id == "0:DEFEND_IRONCLAD")
        self.assertEqual(defend.factors[0]["code"], "unmodeled_deck_cards")
        self.assertIn("UNREVIEWED_CARD", defend.factors[0]["message"])
        self.assertEqual(defend.dimensions["data_completeness"], 0.0)

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


class RelicMechanicTests(unittest.TestCase):
    """Relic facts change capabilities only through their reviewed mechanics.

    Facts here are patched copies of real catalog entries (same fingerprints),
    so each rule is exercised without depending on later review batches.
    """

    PATCHES = {
        # Combat-start block, like Anchor.
        "relics:ANCHOR": {"category": "baseline", "mechanics": [
            {"when": "combat_start", "effect": {"kind": "block", "amount": 30}},
        ]},
        # Block every third attack in a turn, like Ornamental Fan.
        "relics:ORNAMENTAL_FAN": {"category": "trigger", "mechanics": [
            {"when": "card_played", "filter": {"card_type": "Attack"}, "every": 3,
             "counter": "turn", "effect": {"kind": "block", "amount": 4}},
        ]},
        # Skills are upgraded as they are added to the deck, like Toxic Egg.
        "relics:TOXIC_EGG": {"category": "trigger", "mechanics": [
            {"modifier": "card_added_to_deck", "filter": {"card_type": "Skill"}, "upgrade": True},
        ]},
        # Never reviewed: disclosed as missing, contributes nothing.
        "relics:SNECKO_SKULL": {"status": "unreviewed"},
        # A mechanic the engine cannot convert must be disclosed, not guessed.
        "relics:CRACKED_CORE": {"category": "baseline", "mechanics": [
            {"when": "combat_start", "effect": {"kind": "channel_orb", "orb": "Lightning", "amount": 1}},
        ]},
    }

    @classmethod
    def setUpClass(cls):
        from storage.assembly_facts import load_facts

        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        facts = load_facts()
        for entry in facts["entities"]:
            patch = cls.PATCHES.get(entry["entity_key"])
            if patch:
                entry.update({"status": "converted", "evidence": ["test patch"], **patch})
        facts_path = root / "facts.json"
        facts_path.write_text(json.dumps(facts), encoding="utf-8")
        release = root / "release.db"
        build_release_database(
            release, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json",
            assembly_facts_path=facts_path,
        )
        cls.helper = DeckFitnessPolicyTests()
        cls.helper.repository = RelationalRepository(str(release))
        cls.helper.mapped_view = {
            s["id"]: s for s in load_golden_set()
        }["ironclad-f15-pommel-havoc-unrelenting"]["scoring_view"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def gain(self, offered, relics, deck=None):
        deck = deck or [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4), card("BASH")]
        result = self.helper.recommend(deck, [offered], relics=relics)
        scores = self.helper.scores(result)
        return scores[f"0:{offered}"] - scores["skip"], result

    def test_irrelevant_relics_leave_scores_unchanged(self):
        without, _ = self.gain("DEFEND_IRONCLAD", ())
        with_paint, result = self.gain("DEFEND_IRONCLAD", ("WAR_PAINT",))
        self.assertAlmostEqual(without, with_paint)
        self.assertFalse(any("WAR_PAINT" in gap for gap in result.data_gaps))

    def test_baseline_block_makes_more_block_less_valuable(self):
        without, _ = self.gain("DEFEND_IRONCLAD", ())
        with_anchor, _ = self.gain("DEFEND_IRONCLAD", ("ANCHOR",))
        self.assertLess(with_anchor, without)

    def test_attack_trigger_relic_raises_the_value_of_attacks(self):
        deck = [card("STRIKE_IRONCLAD", 4), card("DEFEND_IRONCLAD", 1)]
        without, _ = self.gain("STRIKE_IRONCLAD", (), deck)
        with_fan, _ = self.gain("STRIKE_IRONCLAD", ("ORNAMENTAL_FAN",), deck)
        self.assertGreater(with_fan, without)

    def test_card_added_modifier_scores_the_upgraded_card(self):
        without, _ = self.gain("DEFEND_IRONCLAD", ())
        with_egg, _ = self.gain("DEFEND_IRONCLAD", ("TOXIC_EGG",))
        self.assertGreater(with_egg, without)

    def test_unsupported_relic_mechanics_are_disclosed(self):
        _, result = self.gain("DEFEND_IRONCLAD", ("CRACKED_CORE",))
        self.assertIn("relic_mechanic_unsupported:CRACKED_CORE:effect:channel_orb", result.data_gaps)
        defend = next(r for r in result.candidates if r.candidate_id == "0:DEFEND_IRONCLAD")
        self.assertEqual(defend.factors[0]["code"], "unmodeled_relics")

    def test_unreviewed_relics_are_disclosed_and_never_scored(self):
        without, _ = self.gain("DEFEND_IRONCLAD", ())
        with_skull, result = self.gain("DEFEND_IRONCLAD", ("SNECKO_SKULL",))
        self.assertAlmostEqual(without, with_skull)
        self.assertIn("relic_fact_missing:SNECKO_SKULL", result.data_gaps)

    def test_relics_that_change_the_take_or_skip_choice_are_disclosed(self):
        # Pael's Wing rewards skipping; its value is a random future relic.
        _, result = self.gain("DEFEND_IRONCLAD", ("PAELS_WING",))
        self.assertIn("relic_mechanic_unsupported:PAELS_WING:card_reward_choice", result.data_gaps)
        skip = next(r for r in result.candidates if r.candidate_id == "skip")
        self.assertEqual(skip.factors[0]["code"], "unmodeled_relics")
        self.assertIn("PAELS_WING", skip.factors[0]["message"])


class CardMechanicTests(unittest.TestCase):
    """Reviewed IRONCLAD facts: powers and state-dependent amounts."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.helper = DeckFitnessPolicyTests()

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def gain(self, offered, deck):
        result = self.helper.recommend(deck, [offered])
        scores = self.helper.scores(result)
        return scores[f"0:{offered}"] - scores["skip"], result

    def test_perfected_strike_is_worth_more_with_more_strikes(self):
        few, _ = self.gain("PERFECTED_STRIKE", [card("DEFEND_IRONCLAD", 6), card("STRIKE_IRONCLAD", 1)])
        many, _ = self.gain("PERFECTED_STRIKE", [card("DEFEND_IRONCLAD", 2), card("STRIKE_IRONCLAD", 5)])
        self.assertGreater(many, few)

    def test_body_slam_is_worth_more_in_a_block_heavy_deck(self):
        block, _ = self.gain("BODY_SLAM", [card("DEFEND_IRONCLAD", 7), card("STRIKE_IRONCLAD", 1)])
        attack, _ = self.gain("BODY_SLAM", [card("DEFEND_IRONCLAD", 1), card("STRIKE_IRONCLAD", 7)])
        self.assertGreater(block, attack)

    def test_demon_form_counts_as_growth_through_its_power(self):
        _, result = self.gain("DEMON_FORM", [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)])
        demon = next(r for r in result.candidates if r.candidate_id == "0:DEMON_FORM")
        scaling = [f for f in demon.factors if f["code"] == "deck_fitness:scaling"]
        self.assertTrue(scaling and scaling[0]["delta"] > 0)
        self.assertFalse(demon.data_gaps)

    def test_partially_modeled_cards_are_scored_and_disclosed(self):
        _, result = self.gain("INFERNAL_BLADE", [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)])
        blade = next(r for r in result.candidates if r.candidate_id == "0:INFERNAL_BLADE")
        self.assertIsNotNone(blade.score)
        self.assertEqual(blade.factors[0]["code"], "unmodeled_mechanic")
        self.assertIn("card_effect_unsupported:INFERNAL_BLADE:effect:add_card", blade.data_gaps)
        self.assertEqual(blade.dimensions["data_completeness"], 0.0)

    def test_a_repeated_unsupported_effect_is_one_gap(self):
        # Severance adds Souls to two piles; both add_card effects are one gap.
        _, result = self.gain("SEVERANCE", [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4)])
        severance = next(r for r in result.candidates if r.candidate_id == "0:SEVERANCE")
        self.assertEqual(
            severance.data_gaps.count("card_effect_unsupported:SEVERANCE:effect:add_card"), 1
        )
