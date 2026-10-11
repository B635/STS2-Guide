import unittest

from advisor.encounter_turns import encounter_turns
from tests.test_monster_turns import chomper


def facts(monster_id):
    return {"CHOMPER": chomper()}.get(monster_id)


CHOMPERS = {
    "variants": [{"weight": 1.0, "spawns": [
        {"monster": "CHOMPER", "slot": None, "set": {}, "initial_state": "CLAMP_MOVE"},
        {"monster": "CHOMPER", "slot": None, "set": {"ScreamFirst": True}, "initial_state": "SCREECH_MOVE"},
    ]}],
}


class EncounterTurnsTests(unittest.TestCase):
    def test_each_spawned_monster_counts_with_its_own_opening(self):
        turns = encounter_turns(CHOMPERS, facts, 4)
        self.assertEqual(turns.enemies, 2.0)
        self.assertEqual(turns.hp, 124.0)
        # Turn 1: the first Chomper clamps (16) while the second screeches.
        self.assertEqual(turns.attack[0], 16.0)
        self.assertEqual(turns.hits[0], 2.0)
        self.assertEqual(turns.cards[0], 3.0)

    def test_monsters_stop_acting_as_the_group_is_killed(self):
        turns = encounter_turns(CHOMPERS, facts, 4)
        # The first of two monsters dies half way: from turn 3 only the
        # second acts (it clamps on even turns: 2 and 4).
        self.assertEqual(turns.attack[2], 0.0)
        self.assertEqual(turns.attack[3], 16.0)

    def test_random_line_ups_are_weighted(self):
        encounter = {"variants": [
            {"weight": 0.5, "spawns": [{"monster": "CHOMPER", "initial_state": "CLAMP_MOVE"}]},
            {"weight": 0.5, "spawns": [{"monster": "CHOMPER", "initial_state": "SCREECH_MOVE"}]},
        ]}
        turns = encounter_turns(encounter, facts, 2)
        self.assertEqual(turns.enemies, 1.0)
        self.assertEqual(turns.attack, (8.0, 8.0))

    def test_missing_monster_facts_are_gaps_not_guesses(self):
        encounter = {"variants": [{"weight": 1.0, "spawns": [{"monster": "UNREVIEWED"}]}]}
        turns = encounter_turns(encounter, facts, 2)
        self.assertEqual(turns.attack, (0.0, 0.0))
        self.assertIn("monster_fact_missing:UNREVIEWED", turns.gaps)
        self.assertIsNone(encounter_turns({"variants": []}, facts, 2))


if __name__ == "__main__":
    unittest.main()
