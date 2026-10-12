import copy
import unittest

from advisor.run_history import RunHistoryError, replay_run

STARTER = ["STRIKE", "STRIKE", "DEFEND"]


def stats(**fields):
    base = {"current_hp": 60, "max_hp": 70, "current_gold": 100, "damage_taken": 0}
    base.update(fields)
    return [base]


def choice(card, picked=False, **extra):
    return {"card": {"id": f"CARD.{card}", **extra}, "was_picked": picked}


def run():
    return {
        "build_id": "v0.111.0", "ascension": 0, "win": True,
        "players": [{
            "character": "CHARACTER.TEST_HERO", "max_potion_slot_count": 3,
            "deck": [
                {"id": "CARD.STRIKE", "floor_added_to_deck": 1, "current_upgrade_level": 1},
                {"id": "CARD.DEFEND", "floor_added_to_deck": 1},
                {"id": "CARD.BASH", "floor_added_to_deck": 2},
            ],
            "relics": [
                {"id": "RELIC.START", "floor_added_to_deck": 1},
                {"id": "RELIC.LATE", "floor_added_to_deck": 3},
            ],
        }],
        "map_point_history": [[
            {"map_point_type": "ancient", "player_stats": stats(), "rooms": []},
            {"map_point_type": "monster",
             "player_stats": stats(
                 current_hp=52, damage_taken=8,
                 card_choices=[choice("BASH", True, floor_added_to_deck=2), choice("CLEAVE"), choice("ANGER")],
                 cards_gained=[{"id": "CARD.BASH"}],
             ),
             "rooms": [{"model_id": "ENCOUNTER.SLIMES", "room_type": "monster", "turns_taken": 4}]},
            {"map_point_type": "rest_site",
             "player_stats": stats(upgraded_cards=["CARD.STRIKE"],
                                   cards_removed=[{"id": "CARD.STRIKE", "floor_added_to_deck": 1}]),
             "rooms": []},
            {"map_point_type": "elite",
             "player_stats": stats(card_choices=[choice("CLEAVE"), choice("ANGER"), choice("SHRUG")]),
             "rooms": [{"model_id": "ENCOUNTER.ELITE", "room_type": "elite", "turns_taken": 6}]},
            {"map_point_type": "boss", "player_stats": stats(),
             "rooms": [{"model_id": "ENCOUNTER.BIG_BOSS", "room_type": "boss", "turns_taken": 9}]},
        ]],
    }


class RunHistoryTests(unittest.TestCase):
    def test_each_reward_becomes_a_situation_with_the_deck_of_that_moment(self):
        result = replay_run(run(), STARTER, "1")
        self.assertEqual(result["problems"], [])
        first, second = result["situations"]
        self.assertEqual(first["best"], ["0:BASH"])
        self.assertEqual(second["best"], ["skip"])
        view = first["scoring_view"]
        # Before the pick: the starting deck, without the card being offered.
        self.assertEqual(
            {(row["card"], row["upgrades"]): row["count"] for row in view["state"]["deck"]},
            {("STRIKE", 0): 2, ("DEFEND", 0): 1},
        )
        self.assertEqual(view["state"]["floor"], 2)
        self.assertEqual(view["state"]["hp"], 52)
        self.assertEqual([c["candidate_id"] for c in view["candidates"]], ["0:BASH", "1:CLEAVE", "2:ANGER", "skip"])
        # After the rest site: one Strike removed, the other upgraded, Bash kept.
        self.assertEqual(
            {(row["card"], row["upgrades"]): row["count"] for row in second["scoring_view"]["state"]["deck"]},
            {("STRIKE", 1): 1, ("DEFEND", 0): 1, ("BASH", 0): 1},
        )

    def test_relics_and_the_path_ahead_are_those_of_that_moment(self):
        first, second = replay_run(run(), STARTER, "1")["situations"]
        self.assertEqual(first["scoring_view"]["state"]["relics"], ["START"])
        self.assertEqual(second["scoring_view"]["state"]["relics"], ["START", "LATE"])
        context = first["scoring_view"]["map_context"]
        self.assertEqual([node["kind"] for node in context["nodes"]],
                         ["EVENT", "MONSTER", "CAMPFIRE", "ELITE", "BOSS"])
        self.assertEqual(context["current_node_id"], "1:0")
        self.assertEqual(context["available_next_node_ids"], ["2:0"])
        self.assertEqual(context["boss_encounter_ids"], ["BIG_BOSS"])

    def test_fights_keep_what_really_happened(self):
        fights = replay_run(run(), STARTER, "1")["fights"]
        self.assertEqual([(f["encounter"], f["turns_taken"]) for f in fights],
                         [("SLIMES", 4), ("ELITE", 6), ("BIG_BOSS", 9)])
        self.assertEqual(fights[0]["damage_taken"], 8)

    def test_a_deck_that_does_not_replay_is_a_problem_not_a_guess(self):
        broken = copy.deepcopy(run())
        broken["players"][0]["deck"].append({"id": "CARD.MYSTERY", "floor_added_to_deck": 3})
        self.assertTrue(replay_run(broken, STARTER, "1")["problems"])

    def test_an_unrecorded_upgrade_is_a_warning_and_starting_curses_are_kept(self):
        data = copy.deepcopy(run())
        data["players"][0]["deck"][1]["current_upgrade_level"] = 1   # Defend upgraded, never recorded
        data["players"][0]["deck"].append({"id": "CARD.ASCENDERS_BANE", "floor_added_to_deck": 1})
        result = replay_run(data, STARTER, "1")
        self.assertEqual(result["problems"], [])
        self.assertTrue(any("unrecorded upgrades" in w for w in result["warnings"]))
        deck = {row["card"] for row in result["situations"][0]["scoring_view"]["state"]["deck"]}
        self.assertIn("ASCENDERS_BANE", deck)

    def test_multiplayer_runs_are_refused(self):
        data = copy.deepcopy(run())
        data["players"].append(data["players"][0])
        with self.assertRaises(RunHistoryError):
            replay_run(data, STARTER, "1")


if __name__ == "__main__":
    unittest.main()
