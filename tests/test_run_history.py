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

    def test_the_first_ancient_choice_is_seen_before_the_relic_is_taken(self):
        data = run()
        data["players"][0]["relics"].append({"id": "RELIC.GIFT", "floor_added_to_deck": 1})
        data["map_point_history"][0][0]["player_stats"] = stats(ancient_choice=[
            {"TextKey": "RELIC.GIFT", "was_chosen": True}, {"TextKey": "RELIC.OTHER", "was_chosen": False},
        ])
        decision = next(d for d in replay_run(data, STARTER, "1")["decisions"] if d["kind"] == "ancient_relic")
        self.assertEqual((decision["chosen"], decision["options"]), ("GIFT", ["GIFT", "OTHER"]))
        state = decision["scoring_view"]["state"]
        self.assertEqual(state["relics"], ["START"])
        self.assertEqual([s["relic"] for s in state["relic_states"]], ["START"])

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

    def test_a_relic_lost_later_is_still_held_before_it_left(self):
        # Independent review 2026-10-13: such relics were missing everywhere.
        data = copy.deepcopy(run())
        points = data["map_point_history"][0]
        points[0]["player_stats"][0]["relic_choices"] = [{"choice": "RELIC.LOST_LATER", "was_picked": True}]
        points[2]["player_stats"][0]["relics_removed"] = ["RELIC.LOST_LATER"]
        result = replay_run(data, STARTER, "1")
        first, second = result["situations"]
        self.assertIn("LOST_LATER", first["scoring_view"]["state"]["relics"])      # floor 2, lost at 3
        self.assertNotIn("LOST_LATER", second["scoring_view"]["state"]["relics"])  # floor 4
        self.assertEqual(result["problems"], [])

    def test_the_starting_relic_is_there_in_a_first_floor_fight(self):
        data = copy.deepcopy(run())
        points = data["map_point_history"][0]
        points[0] = {
            "map_point_type": "monster",
            "player_stats": stats(card_choices=[choice("CLEAVE"), choice("ANGER"), choice("SHRUG")]),
            "rooms": [{"model_id": "ENCOUNTER.FIRST", "room_type": "monster", "turns_taken": 3}],
        }
        first = replay_run(data, STARTER, "1")["situations"][0]
        self.assertEqual(first["scoring_view"]["state"]["floor"], 1)
        self.assertEqual(first["scoring_view"]["state"]["relics"], ["START"])

    def test_the_removed_copy_is_the_one_the_record_names(self):
        data = copy.deepcopy(run())
        points = data["map_point_history"][0]
        # Floor 2: one Strike is upgraded. Floor 3: the other, not upgraded, is removed.
        points[1]["player_stats"][0]["upgraded_cards"] = ["CARD.STRIKE"]
        points[2]["player_stats"] = stats(cards_removed=[{"id": "CARD.STRIKE", "floor_added_to_deck": 1}])
        second = replay_run(data, STARTER, "1")["situations"][1]
        deck = {(row["card"], row["upgrades"]): row["count"] for row in second["scoring_view"]["state"]["deck"]}
        self.assertEqual(deck[("STRIKE", 1)], 1)
        self.assertNotIn(("STRIKE", 0), deck)

    def test_a_potion_used_in_the_fight_is_gone_at_its_reward(self):
        data = copy.deepcopy(run())
        points = data["map_point_history"][0]
        points[0]["player_stats"][0]["potion_choices"] = [{"choice": "POTION.FIRE_POTION", "was_picked": True}]
        points[1]["player_stats"][0]["potion_used"] = ["POTION.FIRE_POTION"]
        first = replay_run(data, STARTER, "1")["situations"][0]
        self.assertEqual(first["scoring_view"]["state"]["potions"], [])

    def test_rest_sites_and_shop_removals_are_kept_for_checking_other_decisions(self):
        data = copy.deepcopy(run())
        points = data["map_point_history"][0]
        points[2]["player_stats"][0]["rest_site_choices"] = ["SMITH"]
        points[2]["player_stats"][0]["cards_removed"] = []
        points[3]["map_point_type"] = "shop"
        points[3]["player_stats"][0]["cards_removed"] = [{"id": "CARD.STRIKE", "floor_added_to_deck": 1}]
        points[3]["player_stats"][0].update(current_gold=40, gold_spent=75)
        result = replay_run(data, STARTER, "1")
        rest, removal, shop = result["decisions"]
        self.assertEqual((rest["kind"], rest["chosen"], rest["upgraded"]), ("rest", "SMITH", "STRIKE"))
        self.assertEqual((removal["kind"], removal["chosen"], removal["chosen_upgrades"]), ("remove", "STRIKE", 0))
        # The whole visit: what was on sale, what was bought, the gold it started with.
        self.assertEqual(shop["kind"], "shop")
        self.assertEqual([c["card"] for c in shop["offers"]["cards"]], ["CLEAVE", "ANGER", "SHRUG"])
        self.assertEqual([c["card"] for c in shop["bought"]["removed"]], ["STRIKE"])
        self.assertEqual((shop["gold_before"], shop["gold_spent"], shop["removals_before"]), (115, 75, 0))
        # The deck of the moment, before the point's own changes.
        before = {(r["card"], r["upgrades"]): r["count"] for r in rest["scoring_view"]["state"]["deck"]}
        self.assertEqual(before, {("STRIKE", 0): 2, ("DEFEND", 0): 1, ("BASH", 0): 1})

    def test_multiplayer_runs_are_refused(self):
        data = copy.deepcopy(run())
        data["players"].append(data["players"][0])
        with self.assertRaises(RunHistoryError):
            replay_run(data, STARTER, "1")


if __name__ == "__main__":
    unittest.main()
