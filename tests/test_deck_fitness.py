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
        scores = self.scores(result)
        self.assertGreater(scores["1:DEFEND_IRONCLAD"], scores["0:STRIKE_IRONCLAD"])
        top = next(r for r in result.candidates if r.candidate_id == "1:DEFEND_IRONCLAD").factors[0]
        self.assertIn("防御", top["message"])          # the card's name, not its id
        self.assertNotIn("DEFEND_IRONCLAD", top["message"])
        self.assertEqual(top["code"], "deck_fitness:defense")
        self.assertGreater(top["delta"], 0)

    def test_a_card_must_beat_skipping_by_the_acts_margin(self):
        from advisor.deck_fitness import PARAMETERS_KEY, _pick_margin

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        self.assertEqual([_pick_margin(params, act) for act in (1, 2, 3, 4)], [1.0, 2.0, 3.0, 3.0])
        self.assertEqual(_pick_margin({}, 2), 0.0)
        deck = [card("STRIKE_IRONCLAD", 9), card("BASH")]
        result = self.recommend(deck, ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"])
        rows = {row.candidate_id: row for row in result.candidates}
        margin = next(f for f in rows["skip"].factors if f["code"] == "deck_fitness:pick_margin")
        self.assertEqual(margin["delta"], 1.0)
        self.assertIn("1 分", margin["message"])
        # Defend helps, but by less than the margin: skipping is recommended,
        # and the cards keep their own order below it.
        self.assertEqual((result.status, result.recommended_candidate_id), ("skip", "skip"))
        self.assertEqual([rows[c].rank for c in ("skip", "1:DEFEND_IRONCLAD", "0:STRIKE_IRONCLAD")], [1, 2, 3])
        # A card far better than the deck clears the margin.
        strong = self.recommend(
            [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4), card("BASH")], ["OFFERING"]
        )
        self.assertEqual(strong.recommended_candidate_id, "0:OFFERING")

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
        # One line for the drawer, then the detailed note naming the cards.
        self.assertEqual(defend.factors[0]["code"], "estimate_scope")
        self.assertIn("未完整计入评分", defend.factors[0]["message"])
        detail = next(f for f in defend.factors if f["code"] == "unmodeled_deck_cards")
        self.assertIn("UNREVIEWED_CARD", detail["message"])
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

    def test_killing_faster_shrinks_the_unblocked_damage(self):
        from advisor.deck_fitness import DIMENSIONS, Needs, fight_tempo, fitness

        params = {"satisfaction_cap": 3.0, "surplus_slope": 0.25, "fight_tempo": {"min": 0.5, "max": 2.0}}
        needs = Needs(
            values={d: 10.0 for d in DIMENSIONS},
            weights={d: (1.0 if d == "defense" else 0.0) for d in DIMENSIONS},
            fight_turns=4.0, extra_enemies=0.0, incoming_attack=0.0, fight_mix={},
        )

        def deck(damage, block):
            return {d: {"frontload": damage, "defense": block}.get(d, 0.0) for d in DIMENSIONS}

        # Half the block need is missing. Damage equal to the need: nominal length.
        nominal = fitness(deck(10.0, 5.0), needs, params)
        self.assertEqual(nominal.tempo, 1.0)
        # Twice the damage halves the fight and the damage taken; half doubles it.
        self.assertGreater(fitness(deck(20.0, 5.0), needs, params).terms["defense"], nominal.terms["defense"])
        self.assertLess(fitness(deck(5.0, 5.0), needs, params).terms["defense"], nominal.terms["defense"])
        # Bounded, and growth damage counts as damage.
        self.assertEqual(fight_tempo(deck(1000.0, 0.0), needs, params), 0.5)
        self.assertEqual(fight_tempo(deck(0.0, 0.0), needs, params), 2.0)
        self.assertEqual(fight_tempo({**deck(5.0, 0.0), "scaling": 5.0}, needs, params), 1.0)
        # A fully met block need takes no damage whatever the length.
        self.assertEqual(fitness(deck(5.0, 10.0), needs, params).terms["defense"],
                         fitness(deck(20.0, 10.0), needs, params).terms["defense"])
        # Without the parameter the length has no effect.
        plain = {"satisfaction_cap": 3.0, "surplus_slope": 0.25}
        self.assertEqual(fitness(deck(20.0, 5.0), needs, plain).terms["defense"],
                         fitness(deck(5.0, 5.0), needs, plain).terms["defense"])

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
        self.assertEqual(defend.factors[0]["code"], "estimate_scope")
        self.assertIn("1 件遗物", defend.factors[0]["message"])
        self.assertTrue(any(f["code"] == "unmodeled_relics" for f in defend.factors))

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
        self.assertEqual(skip.factors[0]["code"], "estimate_scope")
        detail = next(f for f in skip.factors if f["code"] == "unmodeled_relics")
        self.assertIn("佩尔之翼", detail["message"])


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
        # A card may repeat one unsupported effect: the gap is listed once.
        _, result = self.gain("MALAISE", [card("STRIKE_SILENT", 4), card("DEFEND_SILENT", 4)])
        malaise = next(r for r in result.candidates if r.candidate_id == "0:MALAISE")
        self.assertTrue(malaise.data_gaps)
        self.assertEqual(len(malaise.data_gaps), len(set(malaise.data_gaps)))


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


class ConfidenceAndNearTieTests(unittest.TestCase):
    """Confidence is low only when gaps could change the pick; near ties are disclosed."""

    PARAMS = {"confidence_gap_allowance": 1.0, "near_tie_points": 0.5}

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.helper = DeckFitnessPolicyTests()
        cls.golden = {s["id"]: s for s in load_golden_set()}

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def confidence(self, shown, partly=(), unscored=False, warnings=False):
        from advisor.deck_fitness import _recommendation_confidence

        ranked = sorted(shown, key=lambda cid: -shown[cid])
        return _recommendation_confidence(shown, ranked, list(partly), unscored, warnings, self.PARAMS)

    def recommend(self, situation_id):
        view = self.golden[situation_id]["scoring_view"]
        return PolicyRegistry([DeckFitnessPolicy(self.helper.repository)]).recommend(card_reward_request(view))

    def test_a_clear_lead_is_medium_even_with_gaps_on_the_pick_itself(self):
        self.assertEqual(self.confidence({"a": 60.0, "skip": 58.0, "b": 57.0}), "medium")
        # Missing effects only add value: a gap on the pick cannot overturn it.
        self.assertEqual(self.confidence({"a": 60.0, "skip": 58.0}, partly=["a"]), "medium")
        # A partly scored rival out of reach of the allowance does not lower it.
        self.assertEqual(self.confidence({"a": 60.0, "skip": 58.0, "b": 58.9}, partly=["b"]), "medium")

    def test_gaps_that_could_change_the_pick_make_it_low(self):
        self.assertEqual(self.confidence({"a": 60.0, "skip": 59.6}), "low")          # near tie
        self.assertEqual(self.confidence({"a": 60.0, "skip": 58.0, "b": 59.0}, partly=["b"]), "low")
        self.assertEqual(self.confidence({"a": 60.0, "skip": 58.0}, unscored=True), "low")
        self.assertEqual(self.confidence({"a": 60.0, "skip": 58.0}, warnings=True), "low")
        self.assertEqual(self.confidence({}), "low")

    def test_near_tie_note_on_both_top_rows(self):
        # ADRENALINE 61.9 against skip 61.5 (with the act's pick margin).
        result = self.recommend("silent-f17-adrenaline")
        rows = {row.candidate_id: row for row in result.candidates}
        best = result.recommended_candidate_id
        runner = next(r.candidate_id for r in result.candidates if r.rank == 2)
        gap = round(rows[best].score - rows[runner].score, 1)
        self.assertLess(gap, 0.5)
        self.assertEqual(result.confidence, "low")
        for cid in (best, runner):
            note = rows[cid].factors[0]
            self.assertEqual((note["code"], note["delta"]), ("deck_fitness:near_tie", 0.0))
            self.assertIn(f"{gap:.1f} 分", note["message"])
            self.assertNotIn("ADRENALINE", note["message"])           # card names, not ids
        self.assertIn("跳过", rows[best].factors[0]["message"])
        for row in result.candidates:
            if row.candidate_id not in (best, runner):
                self.assertFalse(any(f["code"] == "deck_fitness:near_tie" for f in row.factors))

    def test_a_clear_lead_has_no_near_tie_note_and_medium_confidence(self):
        result = self.recommend("defect-8080-f17-skip")
        self.assertTrue(result.data_gaps)           # gaps exist, yet none can change the pick
        self.assertEqual(result.confidence, "medium")
        self.assertFalse(any(
            f["code"] == "deck_fitness:near_tie" for row in result.candidates for f in row.factors
        ))

    def test_confidence_parameters_are_required_and_non_negative(self):
        from advisor.deck_fitness import PARAMETERS_KEY, _parameter_gaps

        params = dict(self.helper.repository.mechanic_constant(PARAMETERS_KEY))
        self.assertEqual(_parameter_gaps(params), ())
        for key in ("confidence_gap_allowance", "near_tie_points"):
            missing = {k: v for k, v in params.items() if k != key}
            self.assertEqual(_parameter_gaps(missing), (f"deck_fitness_parameter_missing:{key}",))
            for bad in (-0.1, "1"):
                self.assertEqual(
                    _parameter_gaps({**params, key: bad}), (f"deck_fitness_parameter_invalid:{key}",)
                )


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

    def test_a_hit_repeating_on_kills_is_worth_more_against_groups(self):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, card_profile, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        slash, gaps = card_profile(self.repository, "ECHOING_SLASH", 0)
        self.assertEqual(gaps, ())
        base = [card_profile(self.repository, cid, 0)[0] for cid in ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5]

        def added(extra_enemies, turns=4.0):
            situation = Situation(turns, extra_enemies, 10.0, {"MONSTER": 1.0}, 1.0, 0)
            with_slash = deck_capabilities([*base, slash], (), situation, params)[0]["aoe"]
            return with_slash / max(extra_enemies, 1e-9) if extra_enemies else with_slash

        # Damage to all per extra enemy: more enemies give more kills, so more repeats.
        self.assertGreater(added(3.0), added(1.0))

    def test_replays_work_with_several_copies_of_the_replayed_card(self):
        # Found on the run history: copies of one card share a profile, and
        # the replay step used to put profiles in a set (they are unhashable).
        with_echo = self.frontload(["ECHO_FORM"] + ["STRIKE_DEFECT"] * 5 + ["DEFEND_DEFECT"] * 4)
        without = self.frontload(["DEFEND_DEFECT"] + ["STRIKE_DEFECT"] * 5 + ["DEFEND_DEFECT"] * 4)
        self.assertGreater(with_echo, without)

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
            # Every mechanic converts; a registered "cannot be known" note of
            # an applied power is a disclosure, not a conversion failure.
            self.assertEqual([g for g in gaps if not g.startswith("fact_unknowable:")], [], relic)
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

class IndependentReviewFixTests(unittest.TestCase):
    """Defects found by the independent review of 2026-10-13."""

    @classmethod
    def setUpClass(cls):
        DeckFitnessPolicyTests.setUpClass()
        cls.repository = DeckFitnessPolicyTests.repository

    @classmethod
    def tearDownClass(cls):
        DeckFitnessPolicyTests.tearDownClass()

    def profile(self, card_id, upgrades=0):
        from advisor.deck_mechanics import card_profile

        return card_profile(self.repository, card_id, upgrades)

    def capabilities(self, cards, turns=6.0):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import ORB_TYPES, Situation, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        orbs = {orb: self.repository.find_assembly_fact("orbs", orb) for orb in ORB_TYPES.values()}
        situation = Situation(turns, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0, orb_slots=3.0, orbs=orbs)
        return deck_capabilities(cards, (), situation, params)[0]

    def starter(self, character="SILENT"):
        return [self.profile(f"STRIKE_{character}")[0] for _ in range(5)] + [
            self.profile(f"DEFEND_{character}")[0] for _ in range(5)
        ]

    def test_copies_sharing_one_profile_score_like_separate_copies(self):
        # The policy builds a deck as [profile] * count.
        for card_id, character in (("GAMMA_BLAST", "REGENT"), ("FLICK_FLACK", "SILENT"), ("REFLEX", "SILENT")):
            base = self.starter(character) + [self.profile("ACROBATICS")[0], self.profile("VENERATE")[0]]
            shared = self.capabilities(base + [self.profile(card_id)[0]] * 3)
            separate = self.capabilities(base + [self.profile(card_id)[0] for _ in range(3)])
            for dimension in shared:
                self.assertAlmostEqual(shared[dimension], separate[dimension], places=6, msg=card_id)

    def test_upgrade_keywords_change_the_card(self):
        self.assertTrue(self.profile("HOLOGRAM", 0)[0].exhausts_self)
        self.assertFalse(self.profile("HOLOGRAM", 1)[0].exhausts_self)
        self.assertTrue(self.profile("ECHO_FORM", 0)[0].ethereal)
        self.assertFalse(self.profile("ECHO_FORM", 1)[0].ethereal)
        plain, upgraded = self.profile("BIG_BANG", 0)[0], self.profile("BIG_BANG", 1)[0]
        self.assertEqual((plain.innate, upgraded.innate), (False, True))

    def test_an_innate_one_off_card_is_always_seen(self):
        from dataclasses import replace

        big_bang = self.profile("BIG_BANG", 0)[0]
        deck = self.starter("REGENT") * 5   # too big to count on drawing one card
        late = self.capabilities(deck + [big_bang])["cycle"]
        opening = self.capabilities(deck + [replace(big_bang, innate=True)])["cycle"]
        self.assertGreater(opening, late)

    def test_a_value_given_only_as_text_is_a_gap_and_keeps_its_base(self):
        # Genetic Algorithm's growth lasts the whole run (DeckVersion): not a fight quantity.
        algorithm, gaps = self.profile("GENETIC_ALGORITHM")
        self.assertEqual([e["amount"] for e in algorithm.effects if e["kind"] == "block"], [1.0])
        self.assertIn("card_effect_unsupported:GENETIC_ALGORITHM:effect_text_only:amount_text", gaps)
        # Values the reviewed text states and a fight source can estimate are scored.
        for card_id in ("MURDER", "REND", "MIND_BLAST", "CHILL", "DOUBLE_ENERGY"):
            self.assertFalse(any("effect_text_only" in gap for gap in self.profile(card_id)[1]), card_id)
        # A plain remark is not a gap.
        self.assertFalse(any("effect_text_only" in gap for gap in self.profile("DODGE_AND_ROLL")[1]))

    def test_a_scaled_or_foreign_target_effect_is_not_dropped_silently(self):
        self.assertTrue(any("power_effect_key" in gap for gap in self.profile("PROLONG")[1]))
        self.assertTrue(any("power_effect_key" in gap for gap in self.profile("SYNCHRONIZE")[1]))
        self.assertIn("card_effect_unsupported:SPUR:effect_target:heal:osty", self.profile("SPUR")[1])

    def test_a_one_turn_power_is_worth_less_than_the_permanent_one(self):
        # Blur keeps block for one turn per play; Barricade for the whole fight.
        from dataclasses import replace

        defends = [self.profile("DEFEND_SILENT")[0] for _ in range(9)]
        blur = replace(self.profile("BLUR")[0], effects=(), cost=0.0)
        barricade = replace(self.profile("BARRICADE")[0], cost=0.0)
        self.assertGreater(
            self.capabilities(defends + [barricade])["defense"],
            self.capabilities(defends + [blur])["defense"],
        )

    def test_what_the_review_could_not_know_is_disclosed_with_the_card(self):
        self.assertIn("fact_unknowable:powers:THORNS", self.profile("CALTROPS")[1])
        # A card applying Doom carries Doom's own note.
        self.assertIn("fact_unknowable:powers:DOOM", self.profile("SCOURGE")[1])
        self.assertFalse(any("fact_unknowable" in gap for gap in self.profile("STRIKE_SILENT")[1]))

    def test_every_parameter_is_required_and_checked(self):
        import copy

        from advisor.deck_fitness import PARAMETERS_KEY, REQUIRED_PARAMETERS, _parameter_gaps

        params = dict(self.repository.mechanic_constant(PARAMETERS_KEY))
        self.assertEqual(_parameter_gaps(params), ())
        for key in REQUIRED_PARAMETERS:
            without = {k: v for k, v in params.items() if k != key}
            self.assertEqual(_parameter_gaps(without), (f"deck_fitness_parameter_missing:{key}",))

        def broken(path, value):
            changed = copy.deepcopy(params)
            target = changed
            for step in path[:-1]:
                target = target[step]
            if value is None:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            return _parameter_gaps(changed)

        for path, value in (
            (("fight_tempo", "max"), None),
            (("dimension_weights", "cycle"), None),
            (("fight_importance",), {"MONSTER": 0, "ELITE": 0, "BOSS": 0}),
            (("pick_margin_by_act",), {"1": "x"}),
            (("pick_margin_by_act",), {"1": -1}),
            (("pick_margin_by_act",), {}),
            (("card_reward_odds", "base"), None),
            (("lookahead", "scenarios"), "many"),
            (("frail_block_reduction",), "a quarter"),
            (("hand_size",), True),
        ):
            gaps = broken(path, value)
            self.assertEqual(len(gaps), 1, path)
            self.assertTrue(gaps[0].startswith("deck_fitness_parameter_invalid:" + path[0]), (path, gaps))

    def test_tempest_channels_one_orb_per_energy_and_one_more_upgraded(self):
        self.assertEqual(self.profile("TEMPEST")[1], ())
        deck = [self.profile("DEFEND_DEFECT")[0] for _ in range(2)]
        nothing = self.capabilities(deck)["frontload"]
        plain = self.capabilities(deck + [self.profile("TEMPEST", 0)[0]])["frontload"]
        upgraded = self.capabilities(deck + [self.profile("TEMPEST", 1)[0]])["frontload"]
        self.assertGreater(plain, nothing)
        self.assertGreater(upgraded, plain)

    def test_a_free_card_is_played_whenever_it_is_drawn(self):
        shiv = self.profile("SHIV")[0]
        self.assertEqual(shiv.cost, 0.0)
        heavy = [self.profile("BLUDGEON")[0] for _ in range(9)]
        with_shiv = self.capabilities(heavy + [shiv])["frontload"]
        without = self.capabilities(heavy)["frontload"]
        # Nine three-cost cards use all the energy either way; the free card
        # (4 damage, exhausts: once in a six-turn fight) adds all of it.
        self.assertAlmostEqual(with_shiv - without, 4.0 / 6.0, places=3)

    def test_a_finisher_with_no_other_attack_hits_nothing(self):
        deck = [self.profile("DEFEND_SILENT")[0] for _ in range(6)]
        self.assertEqual(self.capabilities(deck + [self.profile("FINISHER")[0]])["frontload"], 0.0)

class SecondReviewFixTests(IndependentReviewFixTests):
    """Defects the re-review of 2026-10-13 found in the first round of fixes."""

    def test_unknown_hp_is_disclosed_and_still_scored(self):
        import copy

        from advisor.deck_fitness import DeckFitnessPolicy
        from advisor.golden_set import card_reward_request, load_golden_set

        view = copy.deepcopy(load_golden_set()[0]["scoring_view"])
        view["state"]["hp"] = None
        result = DeckFitnessPolicy(self.repository).recommend(card_reward_request(view))
        self.assertIn("hp_unknown", result.data_gaps)
        self.assertNotEqual(result.status, "uncertain")

    def test_retained_cards_do_not_spend_energy_the_deck_lacks(self):
        from dataclasses import replace

        strike = self.profile("STRIKE_SILENT")[0]
        plain = [replace(strike, cost=2.0) for _ in range(10)]
        kept = [replace(strike, cost=2.0, retain=True) for _ in range(10)]
        # Three energy pays for 1.5 two-cost cards a turn, retained or not.
        self.assertAlmostEqual(self.capabilities(plain)["frontload"], 9.0, places=3)
        self.assertAlmostEqual(self.capabilities(kept)["frontload"], 9.0, places=3)
        # One retained card among them is played every time it is drawn; the
        # others get what is left.
        mixed = self.capabilities([replace(strike, cost=2.0, retain=True)] + plain[:9])
        self.assertAlmostEqual(mixed["frontload"], 9.0, places=3)

    def test_an_innate_power_works_from_the_first_turn(self):
        from dataclasses import replace

        deck = self.starter("SILENT")
        power = self.profile("AFTERIMAGE", 0)[0]
        late = self.capabilities(deck + [power])["defense"]
        opening = self.capabilities(deck + [replace(power, innate=True)])["defense"]
        self.assertGreater(opening, late)

    def test_a_move_counted_only_in_words_keeps_its_gap(self):
        self.assertTrue(any("move_card" in gap for gap in self.profile("DREDGE")[1]))

    def test_out_of_range_parameters_stop_scoring(self):
        import copy

        from advisor.deck_fitness import PARAMETERS_KEY, _parameter_gaps

        params = dict(self.repository.mechanic_constant(PARAMETERS_KEY))
        for path, value in (
            (("lookahead", "scenarios"), 0),
            (("card_reward_odds", "base"), [0.6, 0.37, 0.03]),
            (("preference_dimensions",), {"luck": ["defense"]}),
            (("surplus_slope",), float("nan")),
            (("hand_size",), 0),
        ):
            changed = copy.deepcopy(params)
            target = changed
            for step in path[:-1]:
                target = target[step]
            target[path[-1]] = value
            gaps = _parameter_gaps(changed)
            self.assertEqual(len(gaps), 1, path)
            self.assertIn("deck_fitness_parameter_invalid", gaps[0])


class MissingMechanicTests(IndependentReviewFixTests):
    """Mechanics added because the player kept taking cards that need them."""

    def regent(self):
        names = ["STRIKE_REGENT"] * 4 + ["DEFEND_REGENT"] * 4 + ["FALLING_STAR", "VENERATE", "GLOW"]
        return [self.profile(name)[0] for name in names]

    def test_a_card_put_back_on_top_is_there_again_later(self):
        from dataclasses import replace

        card, gaps = self.profile("SHINING_STRIKE")
        self.assertEqual((card.returns, gaps), ("top", ()))
        once = self.capabilities(self.regent() + [replace(card, returns=None)])
        again = self.capabilities(self.regent() + [card])
        self.assertGreater(again["frontload"], once["frontload"])
        # Copies share one profile: the boost is not applied once per copy.
        shared = self.capabilities(self.regent() + [card] * 3)
        separate = self.capabilities(self.regent() + [self.profile("SHINING_STRIKE")[0] for _ in range(3)])
        for dimension in shared:
            self.assertAlmostEqual(shared[dimension], separate[dimension], places=6, msg=dimension)

    def test_a_card_back_in_hand_is_a_gap(self):
        # Particle Wall returns to the hand but is still discarded at the end
        # of the turn; its gain (replays the same turn) is not modeled.
        card, gaps = self.profile("PARTICLE_WALL")
        self.assertIsNone(card.returns)
        self.assertEqual(gaps, ("card_effect_unsupported:PARTICLE_WALL:returns_to_hand_replays",))

    def test_a_power_moving_only_some_cards_is_not_a_plain_draw(self):
        from advisor.deck_mechanics import power_use

        use, gaps = power_use(self.repository, "Aggression", 1)
        self.assertEqual(use.mechanics, ())
        self.assertTrue(any("AGGRESSION" in gap for gap in gaps))

    def test_counts_of_played_cards_and_lightning_follow_the_game(self):
        base = self.starter("IRONCLAD")
        axe = self.profile("GOLD_AXE")[0]
        shared = self.capabilities(base + [axe] * 3)
        separate = self.capabilities(base + [self.profile("GOLD_AXE")[0] for _ in range(3)])
        self.assertAlmostEqual(shared["frontload"], separate["frontload"], places=6)
        defect = [self.profile(c)[0] for c in ["STRIKE_DEFECT"] * 4 + ["DEFEND_DEFECT"] * 4]
        frost = [self.profile("COOLHEADED")[0] for _ in range(4)]
        voltaic = self.profile("VOLTAIC")[0]
        # Frost channels do not feed Voltaic (it counts Lightning only).
        self.assertAlmostEqual(
            self.capabilities(defect + frost + [voltaic])["frontload"]
            - self.capabilities(defect + frost)["frontload"],
            self.capabilities(defect + [self.profile("DEFEND_DEFECT")[0]] * 4 + [voltaic])["frontload"]
            - self.capabilities(defect + [self.profile("DEFEND_DEFECT")[0]] * 4)["frontload"],
            delta=1.0,
        )

class TransformTests(IndependentReviewFixTests):
    """transform_cards: chosen cards become a named card for the rest of the fight."""

    # card -> (result card, where the results go, capability the result adds to)
    MODELED = {
        "BEGONE": ("MINION_STRIKE", "hand", "frontload"),
        "CHARGE": ("MINION_DIVE_BOMB", "draw", "frontload"),
        "GUARDS": ("MINION_SACRIFICE", "hand", "defense"),
        "COMPACT": ("FUEL", "hand", "energy"),
        "SEANCE": ("SOUL", "draw", "cycle"),
    }

    def without_transform(self, card):
        from dataclasses import replace

        return replace(card, generates=())

    def with_enemy_status(self, cards, status_cards):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        situation = Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0, status_cards=status_cards)
        return deck_capabilities(cards, (), situation, params)[0]

    def test_modeled_transforms_convert_without_a_gap(self):
        for card_id, (result, destination, _) in self.MODELED.items():
            for upgrades in (0, 1):
                card, gaps = self.profile(card_id, upgrades)
                self.assertEqual(gaps, (), (card_id, upgrades))
                self.assertEqual(
                    [(g.profile.card_id, g.destination) for g in card.generates], [(result, destination)],
                )
                self.assertEqual(self.profile(result, upgrades)[1], (), result)

    def test_each_transform_adds_capability(self):
        wounds = [self.profile("WOUND")[0] for _ in range(3)]
        for card_id, (_, _, dimension) in self.MODELED.items():
            # Compact needs Status cards in hand to transform.
            deck = self.starter("REGENT") + (wounds if card_id == "COMPACT" else [])
            card = self.profile(card_id)[0]
            with_results = self.capabilities(deck + [card])[dimension]
            without = self.capabilities(deck + [self.without_transform(card)])[dimension]
            self.assertGreater(with_results, without, card_id)

    def test_the_upgraded_result_is_used_when_the_transform_says_so(self):
        for card_id, kind, plain, upgraded in (
            ("BEGONE", "damage", 6.0, 9.0),
            ("CHARGE", "damage", 13.0, 16.0),
            ("GUARDS", "block", 7.0, 10.0),     # written upgraded_result_if_upgraded
            ("COMPACT", "energy", 1.0, 2.0),
        ):
            for upgrades, amount in ((0, plain), (1, upgraded)):
                result = self.profile(card_id, upgrades)[0].generates[0].profile
                self.assertEqual(result.upgraded, upgrades == 1, card_id)
                self.assertEqual([e["amount"] for e in result.effects if e["kind"] == kind], [amount], card_id)
        # Seance's upgrade only lowers its cost: the Soul stays plain.
        self.assertFalse(self.profile("SEANCE", 1)[0].generates[0].profile.upgraded)
        deck = self.starter("REGENT")
        for card_id, dimension in (("BEGONE", "frontload"), ("CHARGE", "frontload"), ("GUARDS", "defense")):
            plain = self.capabilities(deck + [self.profile(card_id, 0)[0]])[dimension]
            better = self.capabilities(deck + [self.profile(card_id, 1)[0]])[dimension]
            self.assertGreater(better, plain, card_id)

    def test_compact_transforms_the_status_cards_drawn(self):
        deck = self.starter("DEFECT")
        compact = self.profile("COMPACT")[0]
        # No Status card anywhere: nothing to transform.
        self.assertEqual(
            self.with_enemy_status(deck + [compact], 0.0),
            self.with_enemy_status(deck + [self.without_transform(compact)], 0.0),
        )
        # Status cards enemies push into the piles are drawn and become Fuel.
        self.assertGreater(
            self.with_enemy_status(deck + [compact], 1.0)["energy"],
            self.with_enemy_status(deck + [self.without_transform(compact)], 1.0)["energy"],
        )

    def test_guards_takes_only_the_cards_the_turn_leaves_unplayed(self):
        from dataclasses import replace

        guards = self.profile("GUARDS")[0]
        # Free cards are all played: nothing is left in hand to give up.
        free = [replace(self.profile("DEFEND_REGENT")[0], cost=0.0) for _ in range(10)]
        self.assertAlmostEqual(
            self.capabilities(free + [guards])["defense"],
            self.capabilities(free + [self.without_transform(guards)])["defense"],
            places=9,
        )
        # The starter cannot pay for its whole hand: those cards become Minion Sacrifices.
        deck = self.starter("REGENT")
        self.assertGreater(
            self.capabilities(deck + [guards])["defense"],
            self.capabilities(deck + [self.without_transform(guards)])["defense"],
        )

    def test_a_transform_does_not_dilute_the_deck(self):
        # The Dive Bombs take the place of cards in the draw pile: the Defends
        # are drawn as often as before (Boost Away's added Dazed lower it).
        defends = [self.profile("DEFEND_REGENT")[0] for _ in range(9)]
        charge = self.profile("CHARGE")[0]
        self.assertAlmostEqual(
            self.capabilities(defends + [charge])["defense"],
            self.capabilities(defends + [self.without_transform(charge)])["defense"],
            places=9,
        )

    def test_other_transforms_stay_gaps(self):
        from advisor.deck_mechanics import _transformed, power_use

        # Every Attack in hand, which the turn would have played.
        self.assertEqual(
            self.profile("PRIMAL_FORCE")[1],
            ("card_effect_unsupported:PRIMAL_FORCE:transform_cards:hand:filter:Attack",),
        )
        # A random result.
        self.assertEqual(
            power_use(self.repository, "Entropy", 1)[1],
            ("power_mechanic_unsupported:ENTROPY:effect:transform_cards",),
        )
        base = {"kind": "transform_cards", "choice": "choose", "count": 1, "into": "MINION_STRIKE", "scope": "hand"}
        for change, reason in (
            ({"into": "random"}, "into:random"),
            ({"count": "any", "scope": "draw"}, "draw:choose"),
            ({"scope": "discard"}, "discard:choose"),
            ({"filter": {"card_type": "Skill"}}, "hand:filter:Skill"),
            ({"position": "top"}, "key:position"),
        ):
            generated, gaps = _transformed(self.repository, {**base, **change}, False, "TEST", 0)
            self.assertIsNone(generated)
            self.assertEqual(gaps, (f"card_effect_unsupported:TEST:transform_cards:{reason}",))


class FilteredExhaustAndUpgradeTests(IndependentReviewFixTests):
    """Exhausts restricted to one card type (Flak Cannon, Thrash) and cards
    upgraded during a fight (Drain Power, Armaments, Apotheosis, relics)."""

    def fight(self, cards, relics=(), status_cards=0.0, turns=6.0):
        from advisor.deck_fitness import PARAMETERS_KEY
        from advisor.deck_mechanics import Situation, deck_capabilities

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        situation = Situation(turns, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0, status_cards=status_cards)
        return deck_capabilities(list(cards), list(relics), situation, params)[0]

    def relic(self, relic_id):
        from advisor.deck_mechanics import relic_profile

        return relic_profile(self.repository, relic_id)

    @staticmethod
    def without(card, kind):
        from dataclasses import replace

        return replace(card, effects=tuple(e for e in card.effects if e["kind"] != kind))

    def assertSameFight(self, first, second):
        for dimension, value in first.items():
            self.assertAlmostEqual(value, second[dimension], places=9, msg=dimension)

    def test_flak_cannon_hits_once_per_status_card_and_shares_them_between_plays(self):
        for upgrades in (0, 1):
            self.assertEqual(self.profile("FLAK_CANNON", upgrades)[1], ())
        flak, upgraded = self.profile("FLAK_CANNON", 0)[0], self.profile("FLAK_CANNON", 1)[0]
        wounds = [self.profile("WOUND")[0] for _ in range(3)]
        # Drawn and played every turn: the first play exhausts the three
        # Wounds, one hit each; later plays find none. 3 hits over 6 turns.
        self.assertAlmostEqual(self.fight([flak] + wounds)["frontload"], 8.0 * 3 / 6.0, places=6)
        self.assertAlmostEqual(self.fight([upgraded] + wounds)["frontload"], 11.0 * 3 / 6.0, places=6)
        self.assertEqual(self.fight([flak])["frontload"], 0.0)
        # Status cards the enemies add count too: more of them, more hits.
        deck = self.starter("DEFECT")
        gains = [
            self.fight(deck + [flak], status_cards=s)["frontload"] - self.fight(deck, status_cards=s)["frontload"]
            for s in (0.0, 1.0, 2.0)
        ]
        self.assertLess(gains[0], gains[1])
        self.assertLess(gains[1], gains[2])

    def test_status_cards_flak_cannon_exhausts_trigger_exhausts_and_stop_diluting_draws(self):
        flak = self.profile("FLAK_CANNON")[0]
        kept = self.without(flak, "exhaust_other")
        # Energy to spare: every Defend drawn is played.
        deck = [self.profile("DEFEND_IRONCLAD")[0] for _ in range(3)] + [
            self.profile("WOUND")[0] for _ in range(6)
        ]
        # Gone from the piles, the Wounds no longer take draws from Defends.
        self.assertGreater(self.fight(deck + [flak])["defense"], self.fight(deck + [kept])["defense"])
        # Each exhausted Status card is an exhaust event (Feel No Pain).
        pain = self.profile("FEEL_NO_PAIN")[0]
        plain_gain = self.fight(deck + [flak])["defense"] - self.fight(deck + [kept])["defense"]
        pain_gain = self.fight(deck + [flak, pain])["defense"] - self.fight(deck + [kept, pain])["defense"]
        self.assertGreater(pain_gain, plain_gain)

    def test_thrash_exhausts_an_attack_and_keeps_its_damage(self):
        from dataclasses import replace

        for upgrades in (0, 1):
            self.assertEqual(self.profile("THRASH", upgrades)[1], ())
        thrash = self.profile("THRASH")[0]
        pain = self.profile("FEEL_NO_PAIN")[0]
        deck = self.starter("IRONCLAD")
        # The random Attack it exhausts from hand is an exhaust event.
        self.assertGreater(
            self.fight(deck + [thrash, pain])["defense"],
            self.fight(deck + [self.without(thrash, "exhaust_other"), pain])["defense"],
        )
        # Later plays keep the exhausted card's damage.
        flat = replace(thrash, effects=tuple(
            {k: v for k, v in e.items() if k != "scale"} for e in thrash.effects
        ))
        self.assertGreater(self.fight(deck + [thrash])["frontload"], self.fight(deck + [flat])["frontload"])
        # With no other Attack there is nothing to exhaust or keep.
        defends = [self.profile("DEFEND_IRONCLAD")[0] for _ in range(8)]
        self.assertSameFight(
            self.fight(defends + [thrash, pain]),
            self.fight(defends + [self.without(thrash, "exhaust_other"), pain]),
        )

    def test_drain_power_upgrades_cards_only_when_the_deck_has_upgradable_ones(self):
        for upgrades in (0, 1):
            self.assertEqual(self.profile("DRAIN_POWER", upgrades)[1], ())
        drain = self.profile("DRAIN_POWER")[0]
        plain = self.without(drain, "upgrade_cards")
        deck = self.starter("NECROBINDER")
        upgrading, flat = self.fight(deck + [drain]), self.fight(deck + [plain])
        self.assertGreater(upgrading["frontload"], flat["frontload"])
        self.assertGreater(upgrading["defense"], flat["defense"])
        # Upgrading three cards a play is worth more than two.
        upgraded_drain = self.profile("DRAIN_POWER", 1)[0]
        self.assertGreater(
            self.fight(deck + [upgraded_drain])["defense"],
            self.fight(deck + [self.without(upgraded_drain, "upgrade_cards")])["defense"]
            + upgrading["defense"] - flat["defense"],
        )
        # Every other card already upgraded (and Drain Power is not its own
        # target): nothing to gain.
        done = [self.profile("STRIKE_NECROBINDER", 1)[0] for _ in range(5)] + [
            self.profile("DEFEND_NECROBINDER", 1)[0] for _ in range(5)
        ]
        self.assertSameFight(self.fight(done + [drain]), self.fight(done + [plain]))

    def test_upgrades_build_up_over_the_fight_and_stop_at_every_card(self):
        from advisor.deck_mechanics import _held_share

        # Half built up on average, as Strength gained during a fight.
        self.assertAlmostEqual(_held_share(0.1, 6.0), 0.1 * 5 / 2)
        # Never more than every card, and later when they must come back first.
        self.assertLessEqual(_held_share(10.0, 6.0), 1.0)
        self.assertLess(_held_share(0.1, 6.0, delay=2.0), _held_share(0.1, 6.0))
        self.assertEqual(_held_share(0.5, 1.0), 0.0)
        # Many Drain Powers in a small deck cannot upgrade beyond it.
        drains = [self.profile("DRAIN_POWER")[0] for _ in range(6)]
        strikes = [self.profile("STRIKE_NECROBINDER")[0] for _ in range(3)]
        everything = [self.profile("STRIKE_NECROBINDER", 1)[0] for _ in range(3)] + [
            self.profile("DRAIN_POWER", 1)[0] for _ in range(6)
        ]
        self.assertLessEqual(self.fight(strikes + drains)["frontload"], self.fight(everything)["frontload"])

    def test_other_in_fight_upgrades_that_fit_are_valued(self):
        deck = self.starter("IRONCLAD")
        for card_id in ("ARMAMENTS", "APOTHEOSIS"):
            for upgrades in (0, 1):
                card, gaps = self.profile(card_id, upgrades)
                self.assertEqual(gaps, (), card_id)
                self.assertGreater(
                    self.fight(deck + [card])["frontload"],
                    self.fight(deck + [self.without(card, "upgrade_cards")])["frontload"], card_id,
                )
        # Upgraded Armaments upgrades every card in hand instead of one.
        self.assertEqual(
            [e["count"] for e in self.profile("ARMAMENTS", 1)[0].effects if e["kind"] == "upgrade_cards"], ["all"],
        )
        for relic_id in ("STONE_CRACKER", "BELLOWS"):
            relic, gaps = self.relic(relic_id)
            self.assertEqual(gaps, (), relic_id)
            self.assertGreater(self.fight(deck, [relic])["frontload"], self.fight(deck)["frontload"], relic_id)

    def test_variants_the_engine_cannot_value_stay_gaps(self):
        from advisor.deck_mechanics import effect_unsupported, unsupported_reason

        # A relic or power picking a filtered card from hand: not resolved
        # per event, so it would always find one.
        self.assertEqual(unsupported_reason({"when": "turn_start", "effect": {
            "kind": "exhaust_other", "choice": "random", "count": 1, "filter": {"card_type": "Status"},
        }}), "effect_filter:mechanic")
        # A type filter the count does not already apply.
        self.assertEqual(effect_unsupported({
            "kind": "exhaust_other", "choice": "all", "count_scale": {"per": "hand_size"},
            "filter": {"card_type": "Attack"},
        }), "effect_filter:count_scale")
        # "all" with no count source, any other ``then``, other filters.
        self.assertEqual(effect_unsupported({"kind": "exhaust_other", "choice": "all", "count": None}),
                         "effect_choice:all")
        self.assertEqual(effect_unsupported({
            "kind": "exhaust_other", "choice": "random", "count": 1, "then": "something else",
        }), "effect_key:then")
        self.assertIn("card_effect_unsupported:SCRAPE:effect_filter:unsupported", self.profile("SCRAPE")[1])
        self.assertIn("relic_mechanic_unsupported:POWER_CELL:effect_key:filter", self.relic("POWER_CELL")[1])
        # Upgrades of the played card or of a generated card, limited uses.
        self.assertEqual(effect_unsupported({"kind": "upgrade_cards", "scope": "played_card", "count": 1}),
                         "upgrade_cards:scope:played_card")
        self.assertTrue(self.relic("RAZOR_TOOTH")[1])
        self.assertEqual(self.relic("BONE_TEA")[1], ("relic_mechanic_unsupported:BONE_TEA:key:uses",))
