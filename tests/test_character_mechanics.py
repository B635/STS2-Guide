import math
import os
import tempfile
import unittest

from advisor.card_reward import recommend_card_reward
from advisor.character_mechanics import (
    MechanicSignal,
    signals_for_card,
)
from advisor.decision_core import (
    CARD_REWARD,
    DecisionCandidate,
    DecisionRequest,
    WorldState,
)
from advisor.policies import CardRewardPolicy
from storage.effect_tags import (
    EFFECT_TAG_VERSION,
    derive_effect_tags,
    derive_mechanic_effect_tags,
)
from storage.relational import RelationalRepository


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_PATH = os.path.join(ROOT, "data", "knowledge.json")


class CharacterMechanicSignalTests(unittest.TestCase):
    def test_signal_contract_rejects_unknown_roles(self):
        with self.assertRaises(ValueError):
            MechanicSignal(
                domain="test",
                role="bonus",
                magnitude=1,
                source_code="test",
                confidence="structured",
            )

    def test_effect_tag_version_is_4(self):
        self.assertEqual(EFFECT_TAG_VERSION, "4")

    def test_known_false_positive_tags_are_removed(self):
        rupture = derive_effect_tags(
            "cards",
            {
                "id": "RUPTURE",
                "color": "ironclad",
                "type_key": "Power",
                "description": "每当你在你的回合失去生命值时，获得1点力量。",
                "vars": {"Strength": 1},
            },
        )
        blade_dance = derive_effect_tags(
            "cards",
            {
                "id": "BLADE_DANCE",
                "color": "silent",
                "cards_draw": 3,
                "spawns_cards": ["SHIV"],
                "description": "添加3张小刀到你的手牌。",
                "vars": {"Cards": 3},
            },
        )
        shroud = derive_effect_tags(
            "cards",
            {
                "id": "SHROUD",
                "color": "necrobinder",
                "type_key": "Skill",
                "description": "每当你给予灾厄时，获得2点格挡。",
            },
        )

        self.assertNotIn("self_harm", rupture)
        self.assertNotIn("draw", blade_dance)
        self.assertNotIn("scaling", shroud)
        self.assertIn("mechanic:blood:payoff", rupture)
        self.assertIn("mechanic:shiv:provider", blade_dance)
        self.assertIn("mechanic:doom:payoff", shroud)

    def test_structured_fields_take_priority_over_description_fallback(self):
        accuracy = derive_mechanic_effect_tags({
            "id": "ACCURACY",
            "color": "silent",
            "spawns_cards": ["SHIV"],
            "vars": {"Accuracy": 4},
            "description": "小刀额外造成4点伤害。",
        })
        comet = derive_mechanic_effect_tags({
            "id": "COMET",
            "color": "regent",
            "cost": 0,
            "star_cost": 5,
            "vars": {"StarCost": 5},
            "description": "造成33点伤害。",
        })
        unleash = derive_mechanic_effect_tags({
            "id": "UNLEASH",
            "color": "necrobinder",
            "tags": ["OstyAttack"],
            "description": "额外造成等量于奥斯提当前生命值的伤害。",
        })

        self.assertNotIn("mechanic:shiv:provider", accuracy)
        self.assertIn("mechanic:shiv:multiplier", accuracy)
        self.assertEqual(comet["mechanic:star:spender"][0], 5.0)
        self.assertIn("mechanic:osty:payoff", unleash)

    def test_ambiguous_reference_fields_are_disambiguated(self):
        knife_trap = derive_mechanic_effect_tags({
            "id": "KNIFE_TRAP",
            "color": "silent",
            "spawns_cards": ["SHIV"],
            "description": "将你消耗牌堆中的所有小刀对一名敌人打出。",
        })
        hyperbeam = derive_mechanic_effect_tags({
            "id": "HYPERBEAM",
            "color": "defect",
            "vars": {"Focus": 3},
            "description": "造成28点伤害。失去3点集中。",
        })
        haunt = derive_effect_tags(
            "cards",
            {
                "id": "HAUNT",
                "color": "necrobinder",
                "hp_loss": 6,
                "spawns_cards": ["SOUL"],
                "description": (
                    "每当你打出一张灵魂时，"
                    "随机一名敌人失去6点生命值。"
                ),
            },
        )

        self.assertIn("mechanic:shiv:payoff", knife_trap)
        self.assertNotIn("mechanic:shiv:provider", knife_trap)
        self.assertNotIn("mechanic:orb:multiplier", hyperbeam)
        self.assertIn("mechanic:soul:payoff", haunt)
        self.assertNotIn("self_harm", haunt)


class CharacterMechanicScoringTests(unittest.TestCase):
    CHARACTER_STARTERS = {
        "IRONCLAD": ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD"),
        "SILENT": ("STRIKE_SILENT", "DEFEND_SILENT"),
        "DEFECT": ("STRIKE_DEFECT", "DEFEND_DEFECT"),
        "REGENT": ("STRIKE_REGENT", "DEFEND_REGENT"),
        "NECROBINDER": (
            "STRIKE_NECROBINDER",
            "DEFEND_NECROBINDER",
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            os.path.join(cls.tempdir.name, "mechanics.db")
        )
        cls.repository.sync_catalog(KNOWLEDGE_PATH)

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def _row(
        self,
        character,
        candidate,
        deck,
        *,
        hp=70,
        max_hp=80,
        guide_preferences=None,
        map_context=None,
        relics=None,
        can_skip=False,
    ):
        state = {
            "character": character,
            "act": 1,
            "floor": 6,
            "hp": hp,
            "max_hp": max_hp,
            "energy": 3,
            "deck": [
                (
                    {"card": entry[0], "count": entry[1]}
                    if isinstance(entry, tuple)
                    else {"card": entry}
                )
                for entry in deck
            ],
            "relics": list(relics or []),
        }
        if guide_preferences is not None:
            state["guide_preferences"] = guide_preferences
        if map_context is not None:
            state["map_context"] = map_context
        result = recommend_card_reward(
            state,
            [{"card": candidate}],
            self.repository,
            can_skip=can_skip,
        )
        return result["recommendations"][0], result

    @staticmethod
    def _codes(row):
        return {factor["code"] for factor in row["factors"]}

    def _assert_pair(
        self,
        *,
        character,
        candidate,
        with_provider,
        without_provider,
        expected_code,
    ):
        present, _ = self._row(character, candidate, with_provider)
        absent, _ = self._row(character, candidate, without_provider)
        self.assertIn(expected_code, self._codes(present))
        self.assertNotIn(expected_code, self._codes(absent))
        self.assertGreater(present["state_score"], absent["state_score"])

    # Ironclad: blood, exhaust, and strength/multi-hit.
    def test_ironclad_rupture_is_payoff_not_self_harm(self):
        self._assert_pair(
            character="IRONCLAD",
            candidate="RUPTURE",
            with_provider=["HEMOKINESIS", "STRIKE_IRONCLAD"],
            without_provider=["DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
            expected_code="mechanic_synergy_payoff:blood",
        )
        row, _ = self._row(
            "IRONCLAD",
            "RUPTURE",
            ["HEMOKINESIS"],
        )
        self.assertNotIn("self_harm_cost", self._codes(row))

    def test_ironclad_blood_provider_needs_payoff(self):
        self._assert_pair(
            character="IRONCLAD",
            candidate="HEMOKINESIS",
            with_provider=["RUPTURE", "STRIKE_IRONCLAD"],
            without_provider=["DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
            expected_code="mechanic_synergy_provider:blood",
        )

    def test_ironclad_exhaust_provider_needs_payoff(self):
        self._assert_pair(
            character="IRONCLAD",
            candidate="IMPERVIOUS",
            with_provider=["DARK_EMBRACE", "STRIKE_IRONCLAD"],
            without_provider=["DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
            expected_code="mechanic_synergy_provider:exhaust",
        )

    def test_ironclad_multi_hit_needs_strength_provider(self):
        self._assert_pair(
            character="IRONCLAD",
            candidate="SWORD_BOOMERANG",
            with_provider=["INFLAME", "STRIKE_IRONCLAD"],
            without_provider=["DEFEND_IRONCLAD", "STRIKE_IRONCLAD"],
            expected_code="mechanic_synergy_payoff:strength",
        )

    # Silent: discard/Sly, Shiv, and poison.
    def test_silent_blade_dance_is_shiv_generation_not_draw(self):
        row, _ = self._row(
            "SILENT",
            "BLADE_DANCE",
            ["STRIKE_SILENT", "DEFEND_SILENT"],
        )
        codes = self._codes(row)
        self.assertNotIn("draw_value", codes)
        self.assertNotIn("draw_coverage", codes)
        domains = {
            signal["domain"] for signal in row["mechanic_signals"]
        }
        self.assertIn("shiv", domains)

    def test_silent_sly_payoff_needs_discard_provider(self):
        self._assert_pair(
            character="SILENT",
            candidate="TACTICIAN",
            with_provider=["SURVIVOR", "STRIKE_SILENT"],
            without_provider=["DEFEND_SILENT", "STRIKE_SILENT"],
            expected_code="mechanic_synergy_payoff:discard",
        )

    def test_silent_shiv_multiplier_needs_generator(self):
        self._assert_pair(
            character="SILENT",
            candidate="ACCURACY",
            with_provider=["BLADE_DANCE", "STRIKE_SILENT"],
            without_provider=["DEFEND_SILENT", "STRIKE_SILENT"],
            expected_code="mechanic_synergy_multiplier:shiv",
        )

    def test_silent_poison_payoff_needs_poison_provider(self):
        self._assert_pair(
            character="SILENT",
            candidate="ACCELERANT",
            with_provider=["DEADLY_POISON", "STRIKE_SILENT"],
            without_provider=["DEFEND_SILENT", "STRIKE_SILENT"],
            expected_code="mechanic_synergy_payoff:poison",
        )

    # Defect: channel/evoke, Focus, capacity, and orb subtypes.
    def test_defect_channel_provider_needs_evoke_payoff(self):
        self._assert_pair(
            character="DEFECT",
            candidate="ZAP",
            with_provider=["DUALCAST", "STRIKE_DEFECT"],
            without_provider=["DEFEND_DEFECT", "STRIKE_DEFECT"],
            expected_code="mechanic_synergy_provider:orb",
        )

    def test_defect_evoke_payoff_needs_channel_provider(self):
        self._assert_pair(
            character="DEFECT",
            candidate="DUALCAST",
            with_provider=["ZAP", "STRIKE_DEFECT"],
            without_provider=["DEFEND_DEFECT", "STRIKE_DEFECT"],
            expected_code="mechanic_synergy_payoff:orb",
        )

    def test_defect_focus_multiplier_needs_orb_provider(self):
        self._assert_pair(
            character="DEFECT",
            candidate="DEFRAGMENT",
            with_provider=["ZAP", "STRIKE_DEFECT"],
            without_provider=["DEFEND_DEFECT", "STRIKE_DEFECT"],
            expected_code="mechanic_synergy_multiplier:orb",
        )

    def test_defect_capacity_needs_orb_provider(self):
        self._assert_pair(
            character="DEFECT",
            candidate="CAPACITOR",
            with_provider=["ZAP", "STRIKE_DEFECT"],
            without_provider=["DEFEND_DEFECT", "STRIKE_DEFECT"],
            expected_code="mechanic_synergy_capacity:orb",
        )

    def test_defect_orb_subtypes_are_preserved(self):
        expected = {
            "ZAP": "orb.lightning",
            "FUSION": "orb.plasma",
            "DARKNESS": "orb.dark",
            "GLACIER": "orb.frost",
        }
        for card_id, domain in expected.items():
            with self.subTest(card=card_id):
                card = self.repository.find_card(card_id)
                domains = {
                    signal.domain
                    for signal in signals_for_card("DEFECT", card)
                }
                self.assertIn(domain, domains)

    # Regent: Stars and Forge/Sovereign Blade.
    def test_regent_star_spender_is_penalized_without_provider(self):
        absent, _ = self._row(
            "REGENT",
            "COMET",
            ["DEFEND_REGENT", "STRIKE_REGENT"],
        )
        supported, _ = self._row(
            "REGENT",
            "COMET",
            [("VENERATE", 3), "STRIKE_REGENT"],
        )
        self.assertIn(
            "mechanic_resource_unsupported:star",
            self._codes(absent),
        )
        self.assertIn(
            "mechanic_resource_supported:star",
            self._codes(supported),
        )
        self.assertGreater(supported["state_score"], absent["state_score"])

    def test_regent_venerate_provider_needs_star_demand(self):
        self._assert_pair(
            character="REGENT",
            candidate="VENERATE",
            with_provider=["COMET", "STRIKE_REGENT"],
            without_provider=["DEFEND_REGENT", "STRIKE_REGENT"],
            expected_code="mechanic_synergy_provider:star",
        )

    def test_regent_x_star_spender_uses_star_provider(self):
        supported, _ = self._row(
            "REGENT",
            "STARDUST",
            ["VENERATE", "STRIKE_REGENT"],
        )
        absent, _ = self._row(
            "REGENT",
            "STARDUST",
            ["DEFEND_REGENT", "STRIKE_REGENT"],
        )
        self.assertIn(
            "mechanic_resource_supported:star",
            self._codes(supported),
        )
        self.assertIn(
            "mechanic_resource_unsupported:star",
            self._codes(absent),
        )

    def test_regent_forge_provider_needs_sovereign_blade_payoff(self):
        self._assert_pair(
            character="REGENT",
            candidate="WROUGHT_IN_WAR",
            with_provider=["PARRY", "STRIKE_REGENT"],
            without_provider=["DEFEND_REGENT", "STRIKE_REGENT"],
            expected_code="mechanic_synergy_provider:forge",
        )

    # Necrobinder: Summon/Osty, Doom, and Soul.
    def test_necrobinder_summon_provider_needs_osty_payoff(self):
        self._assert_pair(
            character="NECROBINDER",
            candidate="BODYGUARD",
            with_provider=["UNLEASH", "STRIKE_NECROBINDER"],
            without_provider=["DEFEND_NECROBINDER", "STRIKE_NECROBINDER"],
            expected_code="mechanic_synergy_provider:osty",
        )

    def test_necrobinder_unleash_reports_missing_dynamic_osty_hp(self):
        self._assert_pair(
            character="NECROBINDER",
            candidate="UNLEASH",
            with_provider=["BODYGUARD", "STRIKE_NECROBINDER"],
            without_provider=["DEFEND_NECROBINDER", "STRIKE_NECROBINDER"],
            expected_code="mechanic_synergy_payoff:osty",
        )
        row, result = self._row(
            "NECROBINDER",
            "UNLEASH",
            ["BODYGUARD", "STRIKE_NECROBINDER"],
        )
        self.assertIn(
            "character_state:osty_current_hp",
            row["data_gaps"],
        )
        self.assertIn(
            "character_state:osty_current_hp",
            result["data_gaps"],
        )

    def test_necrobinder_doom_payoff_needs_provider(self):
        self._assert_pair(
            character="NECROBINDER",
            candidate="SHROUD",
            with_provider=["END_OF_DAYS", "STRIKE_NECROBINDER"],
            without_provider=["DEFEND_NECROBINDER", "STRIKE_NECROBINDER"],
            expected_code="mechanic_synergy_payoff:doom",
        )

    def test_necrobinder_soul_payoff_needs_generator(self):
        self._assert_pair(
            character="NECROBINDER",
            candidate="SOUL_STORM",
            with_provider=["CAPTURE_SPIRIT", "STRIKE_NECROBINDER"],
            without_provider=[
                "DEFEND_NECROBINDER",
                "STRIKE_NECROBINDER",
            ],
            expected_code="mechanic_synergy_payoff:soul",
        )

    # Shared rules must stay character-agnostic.  These use each character's
    # real starter IDs rather than cross-character synthetic cards.
    def test_shared_attack_and_defense_coverage_for_all_characters(self):
        for character, (strike, defend) in self.CHARACTER_STARTERS.items():
            with self.subTest(character=character, rule="attack"):
                attack, _ = self._row(
                    character,
                    strike,
                    [defend] * 4,
                )
                self.assertIn("attack_coverage", self._codes(attack))
            with self.subTest(character=character, rule="defense"):
                defense, _ = self._row(
                    character,
                    defend,
                    [strike] * 4,
                )
                self.assertIn("defense_coverage", self._codes(defense))

    def test_shared_low_hp_rule_for_all_characters(self):
        for character, (strike, defend) in self.CHARACTER_STARTERS.items():
            with self.subTest(character=character):
                row, _ = self._row(
                    character,
                    defend,
                    [strike] * 4,
                    hp=20,
                    max_hp=80,
                )
                self.assertIn("critical_hp_defense", self._codes(row))

    def test_shared_relic_rule_for_all_characters(self):
        for character, (strike, defend) in self.CHARACTER_STARTERS.items():
            with self.subTest(character=character):
                row, _ = self._row(
                    character,
                    strike,
                    [defend] * 4,
                    relics=["SHURIKEN"],
                )
                self.assertIn("relic_synergy", self._codes(row))

    def test_shared_route_rule_for_all_characters(self):
        route = {
            "nodes": [
                {
                    "node_id": "1:0",
                    "kind": "ELITE",
                    "row": 1,
                    "col": 0,
                    "edges": [],
                },
            ],
            "available_next_node_ids": ["1:0"],
        }
        for character, (strike, defend) in self.CHARACTER_STARTERS.items():
            with self.subTest(character=character):
                row, _ = self._row(
                    character,
                    strike,
                    [defend] * 4,
                    map_context=route,
                )
                self.assertIn("route_elite_fit", self._codes(row))

    def test_shared_skip_and_numeric_bounds_for_all_characters(self):
        for character, (strike, defend) in self.CHARACTER_STARTERS.items():
            with self.subTest(character=character):
                state = {
                    "character": character,
                    "act": 1,
                    "floor": 6,
                    "hp": 70,
                    "max_hp": 80,
                    "energy": 3,
                    "deck": [
                        {"card": strike, "count": 20},
                        {"card": defend, "count": 20},
                    ],
                    "relics": [],
                }
                result = recommend_card_reward(
                    state,
                    [{"card": strike}, {"card": defend}],
                    self.repository,
                    can_skip=True,
                )
                self.assertEqual(result["decision_status"], "skip")
                values = [
                    result["skip_score"],
                    *[
                        recommendation["score"]
                        for recommendation in result["recommendations"]
                    ],
                ]
                for value in values:
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 100.0)

    def test_unknown_character_mechanics_becomes_data_gap(self):
        row, result = self._row(
            "UNSUPPORTED_HERO",
            "STRIKE_IRONCLAD",
            ["DEFEND_IRONCLAD"],
        )
        expected = "character_mechanics:unsupported:UNSUPPORTED_HERO"
        self.assertIn(expected, row["data_gaps"])
        self.assertIn(expected, result["data_gaps"])
        self.assertEqual(result["confidence"], "low")
        self.assertTrue(math.isfinite(row["score"]))
        self.assertGreaterEqual(row["score"], 0.0)
        self.assertLessEqual(row["score"], 100.0)

    def test_policy_contract_preserves_mechanic_gaps_and_dimensions(self):
        necrobinder_world = WorldState.create(
            run_id="mechanic-policy-necrobinder",
            sequence=1,
            state={
                "character": "NECROBINDER",
                "act": 1,
                "floor": 6,
                "hp": 70,
                "max_hp": 80,
                "energy": 3,
                "deck": [
                    {"card": "BODYGUARD", "count": 1},
                    {"card": "STRIKE_NECROBINDER", "count": 1},
                ],
                "relics": [],
            },
        )
        necrobinder_request = DecisionRequest.create(
            decision_id="mechanic-policy-necrobinder:reward",
            decision_type=CARD_REWARD,
            world=necrobinder_world,
            candidates=[
                DecisionCandidate.create(
                    "0:UNLEASH",
                    {"card": "UNLEASH"},
                    label="UNLEASH",
                    display_index=0,
                ),
            ],
            constraints={"can_skip": False},
        )
        necrobinder_result = CardRewardPolicy(
            self.repository
        ).recommend(necrobinder_request)
        osty_gap = "character_state:osty_current_hp"
        self.assertIn(osty_gap, necrobinder_result.data_gaps)
        self.assertIn(
            osty_gap,
            necrobinder_result.candidates[0].data_gaps,
        )
        self.assertEqual(
            necrobinder_result.candidates[0]
            .dimensions["data_completeness"],
            0.0,
        )

        regent_world = WorldState.create(
            run_id="mechanic-policy-regent",
            sequence=1,
            state={
                "character": "REGENT",
                "act": 1,
                "floor": 6,
                "hp": 70,
                "max_hp": 80,
                "energy": 3,
                "deck": [
                    {"card": "STRIKE_REGENT", "count": 1},
                    {"card": "DEFEND_REGENT", "count": 1},
                ],
                "relics": [],
            },
        )
        regent_request = DecisionRequest.create(
            decision_id="mechanic-policy-regent:reward",
            decision_type=CARD_REWARD,
            world=regent_world,
            candidates=[
                DecisionCandidate.create(
                    "0:COMET",
                    {"card": "COMET"},
                    label="COMET",
                    display_index=0,
                ),
            ],
            constraints={"can_skip": False},
        )
        regent_result = CardRewardPolicy(
            self.repository
        ).recommend(regent_request)
        self.assertLess(
            regent_result.candidates[0]
            .dimensions["resource_efficiency"],
            0.0,
        )

    @staticmethod
    def _route_context():
        return {
            "nodes": [
                {
                    "node_id": "1:0",
                    "kind": "ELITE",
                    "row": 1,
                    "col": 0,
                    "edges": ["2:0"],
                },
                {
                    "node_id": "1:1",
                    "kind": "CAMPFIRE",
                    "row": 1,
                    "col": 1,
                    "edges": ["2:0"],
                },
                {
                    "node_id": "2:0",
                    "kind": "MONSTER",
                    "row": 2,
                    "col": 0,
                    "edges": [],
                },
            ],
            "available_next_node_ids": ["1:0", "1:1"],
        }

    def test_card_route_mode_balanced_has_no_extra_preference_factor(self):
        row, result = self._row(
            "IRONCLAD",
            "SHRUG_IT_OFF",
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            guide_preferences={"route_mode": "balanced"},
            map_context=self._route_context(),
        )
        self.assertEqual(result["profile"]["route_mode"], "balanced")
        self.assertFalse(
            any(code.startswith("route_mode_") for code in self._codes(row))
        )

    def test_card_route_mode_survival_rewards_defense_under_pressure(self):
        balanced, _ = self._row(
            "IRONCLAD",
            "SHRUG_IT_OFF",
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            guide_preferences={"route_mode": "balanced"},
            map_context=self._route_context(),
        )
        survival, _ = self._row(
            "IRONCLAD",
            "SHRUG_IT_OFF",
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            guide_preferences={"route_mode": "survival"},
            map_context=self._route_context(),
        )
        self.assertIn("route_mode_survival", self._codes(survival))
        self.assertGreater(survival["state_score"], balanced["state_score"])

    def test_card_route_mode_growth_rewards_scaling_above_safety_floor(self):
        balanced, _ = self._row(
            "IRONCLAD",
            "DEMON_FORM",
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            guide_preferences={"route_mode": "balanced"},
            map_context=self._route_context(),
        )
        growth, _ = self._row(
            "IRONCLAD",
            "DEMON_FORM",
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            guide_preferences={"route_mode": "growth"},
            map_context=self._route_context(),
        )
        self.assertIn("route_mode_growth", self._codes(growth))
        self.assertGreater(growth["state_score"], balanced["state_score"])

    def test_card_route_mode_growth_does_not_cross_critical_hp_floor(self):
        growth, _ = self._row(
            "IRONCLAD",
            "DEMON_FORM",
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            hp=20,
            max_hp=80,
            guide_preferences={"route_mode": "growth"},
            map_context=self._route_context(),
        )
        self.assertNotIn("route_mode_growth", self._codes(growth))
        self.assertIn(
            "route_mode_growth_safety_floor",
            self._codes(growth),
        )

    def test_route_mode_without_map_does_not_fabricate_route_fit(self):
        row, _ = self._row(
            "IRONCLAD",
            "DEMON_FORM",
            ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"],
            guide_preferences={"route_mode": "growth"},
        )
        self.assertFalse(
            any(code.startswith("route_") for code in self._codes(row))
        )

    def test_invalid_route_mode_is_not_silently_defaulted(self):
        with self.assertRaisesRegex(ValueError, "route_mode"):
            self._row(
                "IRONCLAD",
                "DEMON_FORM",
                ["STRIKE_IRONCLAD"],
                guide_preferences={"route_mode": "speedrun"},
                map_context=self._route_context(),
            )


if __name__ == "__main__":
    unittest.main()
