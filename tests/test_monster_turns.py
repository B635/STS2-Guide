import unittest

from advisor.monster_turns import mean, monster_turns
from storage.assembly_facts import load_facts


def chomper():
    return {
        "hp": {"min": 60, "max": 64, "min_ascension": 63, "max_ascension": 67, "ascension_level": 8},
        "moves": [
            {"id": "CLAMP", "effects": [{
                "kind": "damage", "amount": 8, "amount_ascension": 9, "ascension_level": 9,
                "hits": 2, "target": "player",
            }]},
            {"id": "SCREECH", "effects": [{
                "kind": "add_card", "card": "Dazed", "count": 3, "destination": "discard", "target": "player",
            }]},
        ],
        "pattern": {"initial": "CLAMP_MOVE", "states": [
            {"id": "CLAMP_MOVE", "type": "move", "move": "CLAMP", "next": "SCREECH_MOVE"},
            {"id": "SCREECH_MOVE", "type": "move", "move": "SCREECH", "next": "CLAMP_MOVE"},
        ]},
    }


class MonsterTurnsTests(unittest.TestCase):
    def test_a_fixed_cycle_is_walked_turn_by_turn(self):
        turns = monster_turns(chomper(), 4)
        self.assertEqual(turns.attack, (16.0, 0.0, 16.0, 0.0))
        self.assertEqual(turns.hits, (2.0, 0.0, 2.0, 0.0))
        self.assertEqual(turns.cards, (0.0, 3.0, 0.0, 3.0))
        self.assertEqual(turns.hp, 62.0)
        self.assertEqual(turns.gaps, ())

    def test_an_encounter_can_fix_the_opening_state(self):
        # ChompersNormal sets ScreamFirst on its second Chomper.
        turns = monster_turns(chomper(), 2, initial_state="SCREECH_MOVE")
        self.assertEqual(turns.attack, (0.0, 16.0))

    def test_ascension_values_apply_from_their_level(self):
        self.assertEqual(monster_turns(chomper(), 1, ascension=8).attack, (16.0,))
        self.assertEqual(monster_turns(chomper(), 1, ascension=8).hp, 65.0)
        self.assertEqual(monster_turns(chomper(), 1, ascension=9).attack, (18.0,))

    def test_strength_gained_adds_to_every_later_hit(self):
        fact = {
            "hp": {"min": 10, "max": 10},
            "moves": [
                {"id": "HISS", "effects": [{"kind": "apply_power", "power": "Strength", "amount": 2, "target": "self"}]},
                {"id": "BITE", "effects": [{"kind": "damage", "amount": 5, "hits": 3, "target": "player"}]},
            ],
            "pattern": {"initial": "HISS_MOVE", "states": [
                {"id": "HISS_MOVE", "type": "move", "move": "HISS", "next": "BITE_MOVE"},
                {"id": "BITE_MOVE", "type": "move", "move": "BITE", "next": "BITE_MOVE"},
            ]},
        }
        self.assertEqual(monster_turns(fact, 3).attack, (0.0, 21.0, 21.0))

    def test_random_branches_respect_repeat_limits_and_are_reproducible(self):
        fact = {
            "hp": {"min": 10, "max": 10},
            "moves": [
                {"id": "A", "effects": [{"kind": "damage", "amount": 10, "hits": 1, "target": "player"}]},
                {"id": "B", "effects": [{"kind": "block", "amount": 6, "target": "self"}]},
            ],
            "pattern": {"initial": "RAND", "states": [
                {"id": "RAND", "type": "random", "branches": [
                    {"to": "A_MOVE", "weight": 1, "max_repeat": 1},
                    {"to": "B_MOVE", "weight": 1, "repeat_type": "CannotRepeat"},
                ]},
                {"id": "A_MOVE", "type": "move", "move": "A", "next": "RAND"},
                {"id": "B_MOVE", "type": "move", "move": "B", "next": "RAND"},
            ]},
        }
        first = monster_turns(fact, 6, seed=3)
        self.assertEqual(first, monster_turns(fact, 6, seed=3))
        # Neither move may repeat, so the two alternate: every pair of turns
        # holds exactly one attack.
        for turn in range(0, 6, 2):
            self.assertAlmostEqual(first.attack[turn] + first.attack[turn + 1], 10.0)

    def test_undetermined_branches_and_openings_are_gaps(self):
        fact = {
            "hp": {"min": 10, "max": 10},
            "moves": [{"id": "A", "effects": []}, {"id": "B", "effects": []}],
            "pattern": {"initial": None, "states": [
                {"id": "A_MOVE", "type": "move", "move": "A", "next": "PICK"},
                {"id": "B_MOVE", "type": "move", "move": "B", "next": "PICK"},
                {"id": "PICK", "type": "conditional", "branches": [
                    {"to": "A_MOVE", "condition": "player below half HP", "default": False},
                    {"to": "B_MOVE", "condition": "otherwise", "default": False},
                ]},
            ]},
        }
        gaps = monster_turns(fact, 4).gaps
        self.assertIn("monster_opening_unknown", gaps)
        self.assertIn("monster_branch_undetermined:PICK", gaps)

    def test_every_reviewed_monster_fact_can_be_walked(self):
        for entry in load_facts()["entities"]:
            if not entry["entity_key"].startswith("monsters:") or entry["status"] != "converted":
                continue
            with self.subTest(monster=entry["entity_key"]):
                turns = monster_turns(entry, 6)
                self.assertEqual(len(turns.attack), 6)
                self.assertGreaterEqual(mean(turns.attack), 0.0)
                self.assertNotIn("monster_pattern_unresolved", turns.gaps)


if __name__ == "__main__":
    unittest.main()
