"""Known Boss facts are shared, bounded preparation evidence, never future actions."""
from collections import Counter
from types import SimpleNamespace
import unittest

from advisor.boss_context import build_boss_preparation, monster_peak_attack
from advisor.contextual_scoring import build_context, score_state_factors
from advisor.route import RoutePolicy


def member(hp=160, attack=12):
    return {"id": "BOSS", "max_hp": hp, "max_hp_ascension": hp + 40,
            "moves": [{"intent": "ATTACK", "damage_normal": attack,
                       "damage_ascension": attack + 3, "hit_count": 2}]}


class BossPreparationTests(unittest.TestCase):
    def repository(self, monsters=None):
        return SimpleNamespace(
            encounter_profile=lambda key: {"_monsters": monsters or [member()]}
            if key == "BOSS_ENCOUNTER" else None,
        )

    def test_hp_and_damage_have_separate_ascension_thresholds(self):
        profiles = [build_boss_preparation(self.repository(), ["BOSS_ENCOUNTER"], ascension=a)
                    for a in (0, 8, 9)]
        self.assertEqual([p.encounters[0].total_hp for p in profiles], [160, 200, 200])
        self.assertEqual([p.encounters[0].peak_attack_pressure for p in profiles], [24, 24, 30])
        self.assertTrue(all(p.complete for p in profiles))

    def test_multi_member_and_duplicate_encounter_ids(self):
        facts = build_boss_preparation(self.repository([member(), member(80, 5)]),
                                      ["BOSS_ENCOUNTER"] * 2, ascension=0)
        self.assertEqual(len(facts.encounters), 1)
        self.assertEqual(facts.encounters[0].total_hp, 240)
        self.assertEqual(facts.encounters[0].peak_attack_pressure, 34)
        self.assertEqual(facts.encounters[0].member_count, 2)

    def test_alternative_attacks_are_not_summed_or_read_from_pattern(self):
        moves = [member()["moves"][0], {"intent": "ATTACK", "damage_normal": 7},
                 {"intent": "BUFF", "damage_normal": 999}]
        self.assertEqual(monster_peak_attack(moves, ascension=0), (24, True))
        self.assertEqual(monster_peak_attack([], ascension=0), (0, False))

    def test_missing_or_partial_catalog_discloses_gaps(self):
        for ids in ([], ["UNKNOWN"], ["BOSS_ENCOUNTER", "UNKNOWN"]):
            facts = build_boss_preparation(self.repository(), ids, ascension=0)
            self.assertFalse(facts.complete)
            self.assertTrue(facts.data_gaps)
            self.assertGreaterEqual(facts.route_threat(9.5), 9.5)
        facts = build_boss_preparation(self.repository([{"id": "NO_DATA"}]),
                                      ["BOSS_ENCOUNTER"], ascension=0)
        self.assertFalse(facts.complete)
        self.assertEqual(len(facts.data_gaps), 2)
        self.assertEqual(facts.route_threat(9.5), 9.5)

    def test_route_formula_stays_equivalent(self):
        request = SimpleNamespace(world=SimpleNamespace(
            map=SimpleNamespace(boss_encounter_ids=("BOSS_ENCOUNTER",)), ascension=9))
        expected = 9.5
        threat, gaps = RoutePolicy(self.repository())._boss_threat(
            request, {"encounter_expected_risk": {"BOSS": expected}})
        self.assertEqual(threat, round(expected + min(12, 200 / 45 + 30 / 18), 3))
        self.assertFalse(gaps)

    def context(self, *, hp=160, attack=12, known=True):
        state = {"character": "IRONCLAD", "hp": 70, "max_hp": 80,
                 "map_context": {"boss_encounter_ids": ["BOSS_ENCOUNTER" if known else "MISSING"],
                                 "available_next_node_ids": ["B"],
                                 "nodes": [{"node_id": "B", "kind": "BOSS", "edges": []}]}}
        return build_context(state, [], self.repository([member(hp, attack)]))

    def score(self, context, **card):
        profile = {"size": 10, "names": Counter(), "high_cost_ratio": 0}
        return {f["code"]: f["delta"] for f in score_state_factors(
            {"id": "C", "cost": 1, "type_key": "Skill", **card}, {}, profile, context,
            mechanic_assessment={"factors": []})}

    def test_known_boss_replaces_generic_proximity_not_double_reward(self):
        rows = self.score(self.context(), _effect_tags={"block": 8})
        self.assertIn("boss_prepare_defense", rows)
        self.assertNotIn("route_boss_fit", rows)
        stronger = self.score(self.context(attack=24), _effect_tags={"block": 8})
        self.assertGreater(stronger["boss_prepare_defense"], rows["boss_prepare_defense"])

    def test_preparation_only_rewards_actual_deck_gap(self):
        context = self.context()
        context["deck_effect_counts"]["block"] = 6
        self.assertNotIn("boss_prepare_defense", self.score(context, _effect_tags={"block": 8}))
        small = self.score(self.context(hp=100), type_key="Power", _effect_tags={"scaling": 1})
        large = self.score(self.context(hp=400), type_key="Power", _effect_tags={"scaling": 1})
        self.assertGreater(large["boss_prepare_scaling"], small["boss_prepare_scaling"])

    def test_unknown_boss_uses_generic_context_and_exposes_gap(self):
        context = self.context(known=False)
        self.assertIn("unknown_boss_encounter:MISSING", context["data_gaps"])
        factors = self.score(context, _effect_tags={"block": 8})
        self.assertIn("route_boss_fit", factors)
        self.assertFalse(any(k.startswith("boss_prepare_") for k in factors))

    def test_growth_coverage_counts_power_or_scaling_once_per_card(self):
        context = self.context()
        state = {"character": "IRONCLAD"}
        rows = [{"card": {"type_key": "Power", "_effect_tags": {"strength": 2}}}] * 3
        rows += [{"card": {"type_key": "Power", "_effect_tags": {"scaling": 1}}}]
        observed = build_context(state, rows, self.repository())
        self.assertEqual(observed["growth_card_count"], 4)
        context["growth_card_count"] = observed["growth_card_count"]
        self.assertNotIn("boss_prepare_scaling", self.score(
            context, type_key="Power", _effect_tags={"strength": 2}))

    def test_no_boss_proximity_does_not_invent_new_pressure(self):
        context = build_context({"character": "IRONCLAD"}, [], self.repository())
        self.assertIsNone(context["boss_preparation"])
        self.assertFalse(context["data_gaps"])
        self.assertFalse(any(k.startswith("boss_prepare_") for k in self.score(
            context, _effect_tags={"block": 8})))


if __name__ == "__main__":
    unittest.main()
