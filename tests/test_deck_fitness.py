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

    def test_reward_after_the_act_boss_uses_the_next_act_pools(self):
        # On the boss node no fight is left on this act's map; the next act's
        # pools and possible bosses are catalog facts, so the choice is scored.
        self.assertEqual(
            self.repository.next_act_boss_ids(["VANTOM_BOSS"]),
            ["THE_INSATIABLE_BOSS", "KNOWLEDGE_DEMON_BOSS", "KAISER_CRAB_BOSS"],
        )
        after_boss = {s["id"]: s for s in load_golden_set()}["silent-f17-adrenaline"]["scoring_view"]
        result = self.recommend(
            [card("STRIKE_SILENT", 5), card("DEFEND_SILENT", 5)], ["ADRENALINE"], view=after_boss,
        )
        self.assertNotEqual(result.status, "uncertain")
        self.assertIn("next_act_needs_from_pools", result.data_gaps)
        self.assertNotIn("situation_needs_unavailable", result.data_gaps)

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
            {"when": "combat_start", "effect": {"kind": "test_only_unmodeled", "amount": 1}},
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
        self.assertIn("relic_mechanic_unsupported:CRACKED_CORE:effect:test_only_unmodeled", result.data_gaps)
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

    def test_perfected_strike_deals_more_with_more_strikes(self):
        # Compared on the damage it adds: the score gain also depends on how
        # much damage the deck still lacks.
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        repository = self.helper.repository
        params = repository.mechanic_constant(PARAMETERS_KEY)
        situation = Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0)

        def added_damage(strikes):
            deck = [card_profile(repository, cid, 0)[0]
                    for cid in ["STRIKE_IRONCLAD"] * strikes + ["DEFEND_IRONCLAD"] * (7 - strikes)]
            perfected = card_profile(repository, "PERFECTED_STRIKE", 0)[0]
            base = deck_capabilities(deck, (), situation, params)[0]["frontload"]
            return deck_capabilities([*deck, perfected], (), situation, params)[0]["frontload"] - base

        self.assertGreater(added_damage(5), added_damage(1))

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
        self.assertIn("card_effect_unsupported:INFERNAL_BLADE:add_card:pool", blade.data_gaps)
        self.assertEqual(blade.dimensions["data_completeness"], 0.0)

    def test_a_repeated_unsupported_effect_is_one_gap(self):
        # Malaise records its upgrade bonus on two effects; one gap.
        _, result = self.gain("MALAISE", [card("STRIKE_SILENT", 4), card("DEFEND_SILENT", 4)])
        malaise = next(r for r in result.candidates if r.candidate_id == "0:MALAISE")
        self.assertEqual(
            malaise.data_gaps.count("card_effect_unsupported:MALAISE:effect_key:bonus_if_upgraded"), 1
        )


class GeneratedCardTests(unittest.TestCase):
    """add_card: named generated cards are played or dilute the deck."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.helper = DeckFitnessPolicyTests()

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def candidate(self, offered, deck):
        result = self.helper.recommend(deck, [offered])
        return next(r for r in result.candidates if r.candidate_id == f"0:{offered}")

    def test_shivs_sent_to_hand_count_as_attacks(self):
        dance = self.candidate("BLADE_DANCE", [card("STRIKE_SILENT", 5), card("DEFEND_SILENT", 5)])
        self.assertFalse([g for g in dance.data_gaps if "add_card" in g])
        frontload = [f for f in dance.factors if f["code"] == "deck_fitness:frontload"]
        self.assertTrue(frontload and frontload[0]["delta"] > 0)

    def test_status_cards_sent_to_a_pile_dilute_the_deck(self):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.helper.repository.mechanic_constant(PARAMETERS_KEY)
        boost, _ = card_profile(self.helper.repository, "BOOST_AWAY", 0)
        self.assertEqual([g.profile.card_id for g in boost.generates], ["DAZED"])
        self.assertEqual(boost.generates[0].destination, "discard")
        situation = Situation(6.0, 0.5, 10.0, {"MONSTER": 1.0}, 1.0, 0)
        strike, _ = card_profile(self.helper.repository, "STRIKE_SILENT", 0)
        deck = [strike] * 8
        base, _ = deck_capabilities([*deck, boost], (), situation, params)
        without, _ = deck_capabilities([*deck, replace_generates(boost)], (), situation, params)
        # Dazed copies in the discard pile lower every other card's draw rate.
        self.assertLess(base["frontload"], without["frontload"])


def replace_generates(profile):
    import dataclasses
    return dataclasses.replace(profile, generates=())


class NextTurnPowerTests(unittest.TestCase):
    """Until-triggered powers that fire once at the next turn start."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.helper = DeckFitnessPolicyTests()

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def test_next_turn_energy_draw_and_block_resolve_like_this_turn(self):
        from advisor.deck_mechanics import power_use

        repository = self.helper.repository
        energy, gaps = power_use(repository, "EnergyNextTurn", 2)
        self.assertEqual((energy.duration, gaps), ("this_turn", ()))
        self.assertEqual(energy.mechanics[0]["effect"], {"kind": "energy", "amount": 2.0})
        draw, _ = power_use(repository, "DrawCardsNextTurn", 2)
        self.assertEqual(draw.mechanics, ({"modifier": "hand_draw", "add": 2.0},))
        # Consumed by a later play, not by the next turn: still a gap.
        free, gaps = power_use(repository, "FreeAttack", 1)
        self.assertIsNone(free)
        self.assertEqual(gaps, ("power_duration_unsupported:FREE_ATTACK:until_triggered",))


class MoveCardTests(unittest.TestCase):
    def test_choosing_cards_into_the_hand_counts_as_drawing_them(self):
        from advisor.deck_mechanics import _move_to_hand

        dredge = {"kind": "move_card", "choice": "choose", "count": 3, "destination": "hand", "from": "discard"}
        self.assertEqual(_move_to_hand(dredge, False), {"kind": "draw", "amount": 3.0})
        neow = {**dredge, "count": 2, "count_upgrade": 1}
        self.assertEqual(_move_to_hand(neow, True)["amount"], 3.0)
        # Top-decking, moving itself, or filtered sets are not converted.
        headbutt = {"kind": "move_card", "choice": "choose", "count": 1, "destination": "draw",
                    "position": "top", "source": "discard"}
        self.assertIsNone(_move_to_hand(headbutt, False))
        all_for_one = {"kind": "move_card", "choice": "all", "count": None, "destination": "hand",
                       "filter": {"cost": 0}, "from": "discard"}
        self.assertIsNone(_move_to_hand(all_for_one, False))


class StarTests(unittest.TestCase):
    """Stars carry over between turns; star cards share the fight's stars."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def frontload(self, extra):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0] for cid in ["STRIKE_REGENT"] * 5 + extra]
        capabilities, _ = deck_capabilities(deck, (), Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0), params)
        return capabilities["frontload"]

    def test_a_star_card_needs_star_income(self):
        base = self.frontload([])
        self.assertAlmostEqual(self.frontload(["GAMMA_BLAST"]), base, places=6)
        self.assertGreater(
            self.frontload(["GLOW", "GAMMA_BLAST"]), self.frontload(["GLOW"])
        )


class OrbTests(unittest.TestCase):
    """Channeled orbs from reviewed orb facts (OrbCmd.Channel semantics)."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def capabilities(self, extra, slots=3.0):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import ORB_TYPES, Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        orbs = {orb: self.repository.find_assembly_fact("orbs", orb) for orb in ORB_TYPES.values()}
        deck = [card_profile(self.repository, cid, 0)[0] for cid in ["STRIKE_DEFECT"] * 4 + extra]
        situation = Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0, orb_slots=slots, orbs=orbs)
        capabilities, gaps = deck_capabilities(deck, (), situation, params)
        self.assertFalse([g for g in gaps if "orb" in g])
        return capabilities

    def test_lightning_adds_damage_and_frost_adds_block(self):
        base = self.capabilities([])
        self.assertGreater(self.capabilities(["ZAP"])["frontload"], base["frontload"])
        self.assertGreater(self.capabilities(["GLACIER"])["defense"], base["defense"] + 0.1)

    def test_focus_raises_orb_values(self):
        # Same size and energy: Defragment versus a 1-cost card with no damage.
        zaps = self.capabilities(["ZAP", "ZAP", "DEFEND_DEFECT"])
        focused = self.capabilities(["ZAP", "ZAP", "DEFRAGMENT"])
        self.assertGreater(focused["frontload"], zaps["frontload"])

    def test_a_character_without_slots_holds_one_orb(self):
        self.assertGreater(
            self.capabilities(["ZAP", "ZAP", "ZAP"], slots=3.0)["frontload"],
            self.capabilities(["ZAP", "ZAP", "ZAP"], slots=0.0)["frontload"],
        )


class CostRuleTests(unittest.TestCase):
    """Cost changes driven by the card's own plays, draws or same-turn plays."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def test_rules_parse_from_reviewed_facts(self):
        from advisor.deck_mechanics import card_profile

        expected = {
            "MOMENTUM_STRIKE": ("after_first", 0.0),
            "MODDED": ("per_own_play", 1.0),
            "KINGLY_KICK": ("per_own_draw", -1.0),
            "PINPOINT": ("per_type_this_turn", -1.0, "Skill"),
        }
        for card_id, rule in expected.items():
            profile, gaps = card_profile(self.repository, card_id, 0)
            self.assertEqual(profile.cost_rule, rule, card_id)
            self.assertNotIn(f"card_effect_unsupported:{card_id}:cost_modifier", gaps)
        # The recorded IL call settles the sign: AddThisCombat(-1).
        sleeve, _ = card_profile(self.repository, "UP_MY_SLEEVE", 0)
        self.assertEqual(sleeve.cost_rule, ("per_own_play", -1.0))
        self.assertEqual(card_profile(self.repository, "FLATTEN", 0)[0].cost_rule, ("free_if_osty_attacked",))
        self.assertEqual(card_profile(self.repository, "ROCKET_PUNCH", 0)[0].cost_rule, ("per_status_generated", -1.0))
        # A rule the engine has no source for stays a disclosed gap.
        _, gaps = card_profile(self.repository, "BANSHEES_CRY", 0)
        self.assertIn("card_effect_unsupported:BANSHEES_CRY:cost_modifier", gaps)

    def test_a_card_that_becomes_free_frees_energy_for_the_rest(self):
        import dataclasses

        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        momentum, _ = card_profile(self.repository, "MOMENTUM_STRIKE", 0)
        strike, _ = card_profile(self.repository, "STRIKE_DEFECT", 0)
        situation = Situation(7.0, 0.0, 10.0, {"BOSS": 1.0}, 1.0, 0)
        with_rule, _ = deck_capabilities([momentum, momentum, strike, strike, strike], (), situation, params)
        plain = dataclasses.replace(momentum, cost_rule=None)
        without, _ = deck_capabilities([plain, plain, strike, strike, strike], (), situation, params)
        self.assertGreater(with_rule["frontload"], without["frontload"])

    def test_upgrade_adds_to_a_card_count(self):
        from advisor.deck_mechanics import card_profile

        prepared, _ = card_profile(self.repository, "PREPARED", 1)
        discard = next(e for e in prepared.effects if e["kind"] == "discard")
        self.assertEqual(discard["count"], 2.0)


class OstyAndDoomTests(unittest.TestCase):
    """Summoned Osty soaks unblocked attacks and attacks only while alive."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def capabilities(self, extra, relics=()):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities, relic_profile

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0]
                for cid in ["STRIKE_NECROBINDER"] * 4 + ["DEFEND_NECROBINDER"] * 4 + extra]
        relic_profiles = [relic_profile(self.repository, r)[0] for r in relics]
        situation = Situation(6.0, 0.0, 14.0, {"MONSTER": 1.0}, 1.0, 0)
        return deck_capabilities(deck, relic_profiles, situation, params)[0]

    def test_summoned_osty_takes_unblocked_attack_damage(self):
        # Summoning 5 HP soaks about as much as 5 block for the same energy.
        self.assertGreater(self.capabilities(["BODYGUARD"])["defense"], self.capabilities(["STRIKE_NECROBINDER"])["defense"])
        self.assertAlmostEqual(
            self.capabilities(["BODYGUARD"])["defense"], self.capabilities(["DEFEND_NECROBINDER"])["defense"], places=6
        )

    def test_osty_attacks_need_a_summoned_osty(self):
        without = self.capabilities(["POKE", "POKE"])
        with_osty = self.capabilities(["POKE", "POKE"], relics=("BOUND_PHYLACTERY",))
        self.assertGreater(with_osty["frontload"], without["frontload"])

    def test_doom_stands_in_for_damage(self):
        self.assertGreater(self.capabilities(["SCOURGE"])["frontload"], self.capabilities(["DEFEND_NECROBINDER"])["frontload"])


class SlyTests(unittest.TestCase):
    """A Sly card discarded by an effect is played for free."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def frontload(self, extra):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0]
                for cid in ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5 + extra]
        return deck_capabilities(deck, (), Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0), params)[0]["frontload"]

    def test_discard_effects_raise_the_value_of_sly_cards(self):
        sly_gain_without = self.frontload(["RICOCHET"]) - self.frontload([])
        sly_gain_with = self.frontload(["PREPARED", "RICOCHET"]) - self.frontload(["PREPARED"])
        self.assertGreater(sly_gain_with, sly_gain_without)


class EnemyStrengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def test_temporary_strength_loss_reduces_incoming_damage(self):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        wail, gaps = card_profile(self.repository, "PIERCING_WAIL", 0)
        self.assertEqual(gaps, ())
        self.assertEqual(wail.effects[0]["kind"], "enemy_strength_down")
        self.assertEqual(wail.effects[0]["lasting"], "turn")
        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        strike, _ = card_profile(self.repository, "STRIKE_SILENT", 0)
        situation = Situation(6.0, 1.0, 12.0, {"MONSTER": 1.0}, 1.0, 0)
        with_wail, _ = deck_capabilities([strike] * 6 + [wail], (), situation, params)
        without, _ = deck_capabilities([strike] * 7, (), situation, params)
        self.assertGreater(with_wail["defense"], without["defense"])


class LookaheadAndSynergyTests(unittest.TestCase):
    """Spec 2.8 addendum (2026-10-09): lookahead and synergy notes."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.helper = DeckFitnessPolicyTests()

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def test_lookahead_is_reproducible_and_disclosed(self):
        view = {s["id"]: s for s in load_golden_set()}["silent-f3-ricochet"]["scoring_view"]
        first = PolicyRegistry([DeckFitnessPolicy(self.helper.repository)]).recommend(card_reward_request(view))
        again = PolicyRegistry([DeckFitnessPolicy(self.helper.repository)]).recommend(card_reward_request(view))
        self.assertEqual(
            [(c.candidate_id, c.score) for c in first.candidates],
            [(c.candidate_id, c.score) for c in again.candidates],
        )
        skip = next(c for c in first.candidates if c.candidate_id == "skip")
        self.assertTrue(any(f["code"] == "deck_fitness:lookahead" for f in skip.factors))

    def test_synergy_is_explained_but_never_scored(self):
        view = {s["id"]: s for s in load_golden_set()}["silent-f3-ricochet"]["scoring_view"]
        result = PolicyRegistry([DeckFitnessPolicy(self.helper.repository)]).recommend(card_reward_request(view))
        ricochet = next(c for c in result.candidates if c.candidate_id == "2:RICOCHET")
        notes = [f for f in ricochet.factors if f["code"] == "deck_fitness:synergy"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["delta"], 0.0)
        self.assertNotIn("STRIKE_SILENT", notes[0]["message"])


class GapRoundOneTests(unittest.TestCase):
    """X-hit attacks, applies_to notes and card_drawn triggers."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def run_deck(self, ids):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0] for cid in ids]
        return deck_capabilities(deck, (), Situation(6.0, 1.0, 10.0, {"MONSTER": 1.0}, 1.0, 0), params)

    def test_whirlwind_hits_once_per_energy_and_each_hit_takes_strength(self):
        from advisor.deck_mechanics import card_profile

        whirlwind, gaps = card_profile(self.repository, "WHIRLWIND", 0)
        self.assertEqual(gaps, ())
        base, _ = self.run_deck(["WHIRLWIND", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD"])
        self.assertGreater(base["aoe"], 0.0)

    def test_self_applied_temporary_focus_is_supported(self):
        from advisor.deck_mechanics import power_use

        hotfix, gaps = power_use(self.repository, "Hotfix", 2)
        self.assertEqual(gaps, ())
        self.assertEqual(hotfix.mechanics[0]["modifier"], "focus")

    def test_card_drawn_triggers_follow_the_drawn_share(self):
        from advisor.deck_mechanics import unsupported_reason

        self.assertIsNone(unsupported_reason({
            "when": "card_drawn", "filter": {"card_type": "Status"},
            "effect": {"kind": "draw", "amount": 1}, "limit": "once_per_turn",
        }))


class ForgeTests(unittest.TestCase):
    """ForgeCmd: the first forge creates a Retain Sovereign Blade in hand."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def frontload(self, extra, blade=True):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0]
                for cid in ["STRIKE_REGENT"] * 4 + ["DEFEND_REGENT"] * 4 + extra]
        situation = Situation(
            6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0,
            sovereign_blade=card_profile(self.repository, "SOVEREIGN_BLADE", 0)[0] if blade else None,
        )
        return deck_capabilities(deck, (), situation, params)[0]["frontload"]

    def test_forging_creates_and_powers_a_blade(self):
        with_forge = self.frontload(["REFINE_BLADE"])
        self.assertGreater(with_forge, self.frontload(["DEFEND_REGENT"]))
        # Without the blade's facts the forge has nothing to power.
        self.assertGreater(with_forge, self.frontload(["REFINE_BLADE"], blade=False))


class MechanicGeneratedCardTests(unittest.TestCase):
    """Relics/powers that create named cards."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def capabilities(self, relics):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities, relic_profile

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0] for cid in ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5]
        profiles = [relic_profile(self.repository, r)[0] for r in relics]
        return deck_capabilities(deck, profiles, Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0), params)[0]

    def test_shivs_from_a_relic_add_damage(self):
        self.assertGreater(self.capabilities(["NINJA_SCROLL"])["frontload"], self.capabilities([])["frontload"])

    def test_status_cards_from_a_relic_dilute_the_deck(self):
        # Blessed Antler shuffles Dazed into the draw pile every turn.
        from advisor.deck_mechanics import relic_profile

        antler, gaps = relic_profile(self.repository, "BLESSED_ANTLER")
        self.assertEqual(gaps, ())
        energy_only = {"relic_id": "X", "category": "baseline", "mechanics": tuple(
            m for m in antler.mechanics if "modifier" in m
        )}
        import dataclasses
        without_dazed = dataclasses.replace(antler, mechanics=energy_only["mechanics"])
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0] for cid in ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5]
        situation = Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0)
        with_dazed = deck_capabilities(deck, [antler], situation, params)[0]
        without = deck_capabilities(deck, [without_dazed], situation, params)[0]
        self.assertLess(with_dazed["frontload"], without["frontload"])


class ScaleSourceTests(unittest.TestCase):
    """Amounts that scale with turn or fight state estimated by the engine."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def frontload(self, ids, slots=0.0):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import ORB_TYPES, Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        profiles = []
        for cid in ids:
            profile, gaps = card_profile(self.repository, cid, 0)
            self.assertEqual(gaps, (), cid)
            profiles.append(profile)
        orbs = {orb: self.repository.find_assembly_fact("orbs", orb) for orb in ORB_TYPES.values()}
        situation = Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0, orb_slots=slots, orbs=orbs)
        return deck_capabilities(profiles, (), situation, params)[0]["frontload"]

    def test_finisher_hits_once_per_other_attack_played(self):
        many = self.frontload(["FINISHER"] + ["STRIKE_SILENT"] * 4 + ["DEFEND_SILENT"] * 2)
        few = self.frontload(["FINISHER"] + ["STRIKE_SILENT"] * 1 + ["DEFEND_SILENT"] * 5)
        self.assertGreater(
            many - self.frontload(["DEFEND_SILENT"] + ["STRIKE_SILENT"] * 4 + ["DEFEND_SILENT"] * 2),
            few - self.frontload(["DEFEND_SILENT"] + ["STRIKE_SILENT"] * 1 + ["DEFEND_SILENT"] * 5),
        )

    def test_barrage_hits_once_per_orb_held(self):
        with_orbs = self.frontload(["BARRAGE", "ZAP", "ZAP", "STRIKE_DEFECT"], slots=3.0)
        without = self.frontload(["BARRAGE", "DEFEND_DEFECT", "DEFEND_DEFECT", "STRIKE_DEFECT"], slots=3.0)
        zaps = self.frontload(["DEFEND_DEFECT", "ZAP", "ZAP", "STRIKE_DEFECT"], slots=3.0)
        plain = self.frontload(["DEFEND_DEFECT", "DEFEND_DEFECT", "DEFEND_DEFECT", "STRIKE_DEFECT"], slots=3.0)
        self.assertGreater(with_orbs - zaps, without - plain)


class GapRoundTwoTests(unittest.TestCase):
    """Retaliation, hit triggers, replays, Buffer, relic-held powers."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def capabilities(self, ids, relics=()):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities, relic_profile

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        deck = [card_profile(self.repository, cid, 0)[0] for cid in ids]
        profiles = []
        for relic in relics:
            profile, gaps = relic_profile(self.repository, relic)
            self.assertEqual(gaps, (), relic)
            profiles.append(profile)
        return deck_capabilities(deck, profiles, Situation(6.0, 1.0, 12.0, {"MONSTER": 1.0}, 1.0, 0), params)[0]

    def test_thorns_from_a_relic_hits_every_attacker(self):
        deck = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 5
        self.assertGreater(
            self.capabilities(deck, ["BRONZE_SCALES"])["frontload"], self.capabilities(deck)["frontload"]
        )

    def test_replays_cost_no_energy(self):
        from advisor.deck_mechanics import power_use

        burst, gaps = power_use(self.repository, "Burst", 1)
        self.assertEqual(gaps, ())
        self.assertEqual(burst.mechanics[0]["uses"], 1.0)
        echo, gaps = power_use(self.repository, "EchoForm", 1)
        self.assertEqual((echo.duration, gaps), ("combat", ()))

    def test_buffer_is_held_until_spent(self):
        from advisor.deck_mechanics import power_use

        buffer, gaps = power_use(self.repository, "Buffer", 2)
        self.assertEqual((buffer.duration, gaps), ("combat", ()))
        self.assertEqual(buffer.mechanics, ({"modifier": "hp_loss_negate", "uses": 2.0},))


class FactBasedNeedsTests(unittest.TestCase):
    """Situation needs come from encounter and monster facts."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def test_pools_use_real_line_ups_and_split_weak_fights(self):
        from advisor.deck_fitness import PARAMETERS_KEY, _fact_pools

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        pools = _fact_pools(self.repository, ["VANTOM_BOSS"], params, 0)
        self.assertEqual(set(pools), {"MONSTER", "MONSTER_WEAK", "ELITE", "BOSS"})
        # The act's opening fights are easier than its regular ones.
        self.assertLess(pools["MONSTER_WEAK"]["attack"], pools["MONSTER"]["attack"])
        self.assertLess(pools["MONSTER_WEAK"]["hp"], pools["MONSTER"]["hp"])
        # Per-turn attack walked from the move machines is far below the
        # "strongest move every turn" of the catalog statistic.
        old = self.repository.encounter_pool_expectations(["VANTOM_BOSS"])
        self.assertLess(pools["MONSTER"]["attack"], old["MONSTER"]["average_attack"])
        self.assertEqual(self.repository.act_index(["VANTOM_BOSS"]), 0)

    def test_needs_carry_enemy_hits_and_player_debuffs(self):
        from advisor.deck_fitness import PARAMETERS_KEY, compute_needs
        from advisor.route import RoutePolicy

        view = {s["id"]: s for s in load_golden_set()}["silent-f3-ricochet"]["scoring_view"]
        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        needs = compute_needs(
            self.repository, RoutePolicy(self.repository), card_reward_request(view), params, ScoringPreferences()
        )
        self.assertGreater(needs.enemy_hits, 0.0)
        self.assertGreaterEqual(needs.player_frail, 0.0)
        self.assertFalse([g for g in needs.gaps if g.startswith("encounter_facts_missing")])


class GapRoundThreeTests(unittest.TestCase):
    """Run-only card mechanics, enchantments on a card, orb evoke triggers."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def test_card_counters_that_run_between_fights_are_not_gaps(self):
        from advisor.deck_mechanics import card_profile

        for card_id in ("GUILTY", "DOWSING"):
            _, gaps = card_profile(self.repository, card_id, 0)
            self.assertNotIn(f"card_effect_unsupported:{card_id}:mechanics", gaps)
        # Returning to the hand during a fight is still unmodeled.
        _, gaps = card_profile(self.repository, "RIGHT_HAND_HAND", 0)
        self.assertIn("card_effect_unsupported:RIGHT_HAND_HAND:mechanics", gaps)

    def test_sharp_adds_its_amount_to_the_cards_damage(self):
        from advisor.deck_mechanics import card_profile, enchant

        star, _ = card_profile(self.repository, "FALLING_STAR", 0)
        sharp = self.repository.find_assembly_fact("enchantments", "SHARP")
        enchanted, gaps = enchant(star, sharp, 2.0)
        self.assertEqual(gaps, ())
        base = next(e for e in star.effects if e["kind"] == "damage")["amount"]
        self.assertEqual(next(e for e in enchanted.effects if e["kind"] == "damage")["amount"], base + 2.0)
        # A once-per-combat rider is disclosed, not spread over every play.
        _, gaps = enchant(star, self.repository.find_assembly_fact("enchantments", "SWIFT"), 1.0)
        self.assertEqual(gaps, ("enchantment_unsupported:FALLING_STAR:card_played",))

    def test_thunder_triggers_on_lightning_evokes(self):
        from advisor.deck_mechanics import power_use

        thunder, gaps = power_use(self.repository, "Thunder", 1)
        self.assertEqual(gaps, ())
        self.assertEqual(thunder.mechanics[0]["when"], "orb_evoked")
