import copy
import tempfile
import unittest
from pathlib import Path

from advisor.deck_fitness import DeckFitnessPolicy
from advisor.golden_set import load_golden_set, world_request
from advisor.relic_choice import RelicChoiceValuer, _entity_id, _parameter_gaps
from storage.relational import RelationalRepository
from storage.release_database import build_release_database


ROOT = Path(__file__).resolve().parents[1]


def card(card_id, count=1, upgrades=0):
    return {
        "card": card_id, "count": count, "upgrades": upgrades,
        "enchantment": None, "enchantment_amount": None,
        "affliction": None, "affliction_amount": None,
    }


class RelicChoiceHelperTests(unittest.TestCase):
    def test_cards_and_relics_may_be_named_by_class(self):
        self.assertEqual(
            [_entity_id(x) for x in ("Bash", "FallingStar", "CurseOfTheBell", "STRIKE_IRONCLAD")],
            ["BASH", "FALLING_STAR", "CURSE_OF_THE_BELL", "STRIKE_IRONCLAD"],
        )

    def test_parameters_fail_closed(self):
        self.assertTrue(_parameter_gaps(None))
        self.assertTrue(_parameter_gaps({"samples": 0, "max_variants": 4}))


class RelicChoiceValuerTests(unittest.TestCase):
    """The valuation on the real release database and reviewed facts."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        release = Path(cls.temporary.name) / "release.db"
        build_release_database(release, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json")
        cls.repository = RelationalRepository(str(release))
        cls.valuer = RelicChoiceValuer(DeckFitnessPolicy(cls.repository))
        cls.view = next(s for s in load_golden_set() if s["id"] == "ironclad-f15-pommel-havoc-unrelenting")["scoring_view"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def evaluate(self, relic, *, deck=None, relics=("BURNING_BLOOD",), params=None):
        view = copy.deepcopy(self.view)
        view["state"].update(
            deck=deck or [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4), card("BASH")],
            relics=list(relics), potions=[], gold=99, hp=80, max_hp=80,
            relic_states=[{"relic": r, "display_amount": None, "stack_count": 1, "status": "NORMAL"} for r in relics],
        )
        baseline, values, _ = self.valuer.evaluate(world_request(view), [(relic, params or {})])
        return baseline, values[relic]

    def test_a_chosen_removal_takes_the_fully_valued_card_the_engine_values_least(self):
        # DOUBT's end-of-turn Weak is not valued yet: removing it is not
        # weighed against fully valued cards (its removal gain is understated
        # or overstated by what is missing), so the choice is among the others.
        deck = [card("STRIKE_IRONCLAD", 5), card("DEFEND_IRONCLAD", 4), card("BASH"), card("DOUBT")]
        _, value = self.evaluate("PRECISE_SCISSORS", deck=deck)
        view = copy.deepcopy(self.view)
        view["state"].update(deck=deck, relics=["BURNING_BLOOD"], relic_states=[], potions=[], gold=99, hp=80, max_hp=80)
        context, _ = self.valuer.policy.deck_context(world_request(view))
        gains = {}
        for i, profile in enumerate(context.deck):
            if self.valuer.policy._card(profile.card_id, 0)[1]:
                continue
            gains.setdefault(profile.card_id, context.score(context.deck[:i] + context.deck[i + 1:])[0].raw)
        best = max(gains, key=gains.get)
        self.assertIn("移除 " + self.valuer.policy._card_name(best), value.notes)
        self.assertNotIn("移除 " + self.valuer.policy._card_name("DOUBT"), value.notes)

    def test_gold_and_max_hp_come_back_as_resources(self):
        _, pearl = self.evaluate("GOLDEN_PEARL")
        _, oyster = self.evaluate("NUTRITIOUS_OYSTER")
        self.assertEqual(dict(pearl.resources), {"gold": 150.0})
        self.assertEqual(dict(oyster.resources), {"max_hp": 11.0})

    def test_archaic_tooth_turns_the_starter_card_into_its_ancient_version(self):
        _, value = self.evaluate("ARCHAIC_TOOTH")
        self.assertIn("变化 " + self.valuer.policy._card_name("BASH"), value.notes)
        self.assertGreater(value.raw, 0)

    def test_touch_of_orobas_upgrades_the_starter_relic(self):
        _, value = self.evaluate("TOUCH_OF_OROBAS")
        expected = f"{self.valuer.policy._relic_name('BURNING_BLOOD')} 换成 {self.valuer.policy._relic_name('BLACK_BLOOD')}"
        self.assertIn(expected, value.notes)

    def test_a_relic_with_an_unvalued_part_is_a_lower_bound(self):
        _, value = self.evaluate("PHIAL_HOLSTER")          # potions have no reviewed facts yet
        self.assertTrue(any("gain_potion" in g or "potion_slots" in g for g in value.gaps))

    def test_the_same_situation_gives_the_same_values(self):
        first = self.evaluate("NEW_LEAF")[1].raw
        second = self.evaluate("NEW_LEAF")[1].raw
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
