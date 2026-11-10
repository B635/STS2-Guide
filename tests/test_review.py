import copy
import tempfile
import unittest
from pathlib import Path

from advisor.run_history import load_run
from review import tools
from review.engine import Engine
from review.run_review import STRICT_GAP, ReviewError, RunReview, _tier, review_file


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "coach" / "eval" / "runs" / "test" / "1791474137.run"
DEV_RUN = ROOT / "coach" / "eval" / "runs" / "dev" / "1773034032.run"


def option(option_id, deck_score, fully_scored=True):
    return {"id": option_id, "deck_score": deck_score, "fully_scored": fully_scored}


class TierTests(unittest.TestCase):
    def test_agreement_is_tier_zero(self):
        self.assertEqual(_tier([option("a", 5)], "a", "a", "medium", 1.0), (0, 0.0, []))

    def test_strict_disagreement_is_tier_one(self):
        tier, gap, reasons = _tier([option("a", 10), option("b", 10 - STRICT_GAP)], "a", "b", "medium", 3.0)
        self.assertEqual((tier, gap, reasons), (1, STRICT_GAP, []))

    def test_a_gap_above_the_act_margin_is_enough(self):
        self.assertEqual(_tier([option("a", 10), option("b", 8.5)], "a", "b", "medium", 1.0)[0], 1)
        self.assertEqual(_tier([option("a", 10), option("b", 8.5)], "a", "b", "medium", 2.0)[2], ["small_gap"])

    def test_partial_scores_and_low_confidence_stay_out_of_tier_one(self):
        tier, _, reasons = _tier([option("a", 20), option("b", 5, fully_scored=False)], "a", "b", "low", 1.0)
        self.assertEqual((tier, reasons), (2, ["partly_scored", "low_confidence"]))

    def test_no_engine_pick_is_not_judged(self):
        self.assertEqual(_tier([option("a", 5)], None, "a", "low", 1.0)[0], None)


class RunReviewTests(unittest.TestCase):
    """One real v0.111.0 run (the developer's own, committed as an evaluation run)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.engine = Engine(Path(cls.temp.name) / "release.db")
        cls.review = review_file(cls.engine, RUN)

    @classmethod
    def tearDownClass(cls):
        del cls.review, cls.engine
        cls.temp.cleanup()

    def test_summary(self):
        summary = self.review.summary()
        self.assertEqual(summary["character"], "NECROBINDER")
        self.assertEqual(summary["build_id"], "v0.111.0")
        self.assertEqual(dict(summary["decisions_scored"]), {"card_reward": 8, "rest": 3, "shop": 1})
        # Act 2 of a lost run: the walked map never reached its boss, so there is nothing to score against.
        self.assertEqual(dict(summary["decisions_skipped"]), {"no_boss": 5})
        self.assertEqual(summary["tier1"], ["F15-card", "F09-card", "F02-card"])

    def test_tier_one_flags_meet_every_condition(self):
        for decision_id in self.review.summary()["tier1"]:
            record = self.review.decision(decision_id).record
            by_id = {o["id"]: o for o in record["options"]}
            top, chosen = by_id[record["engine_top"]], by_id[record["chosen"]]
            self.assertNotEqual(top["id"], chosen["id"])
            self.assertTrue(top["fully_scored"] and chosen["fully_scored"])
            self.assertNotEqual(record["confidence"], "low")
            self.assertAlmostEqual(record["gap"], top["deck_score"] - chosen["deck_score"], places=2)
            self.assertTrue(record["gap"] >= STRICT_GAP or record["gap"] > record["pick_margin"])

    def test_gaps_leave_out_the_pick_margin(self):
        record = self.review.decision("F15-card").record
        skip = next(o for o in record["options"] if o["id"] == "skip")
        self.assertEqual(round(skip["score"] - skip["deck_score"], 2), record["pick_margin"])
        self.assertEqual(record["chosen"], "skip")

    def test_compare_lists_the_largest_difference_first_without_policy_leanings(self):
        result = tools.compare_options(self.review, "F15-card", "0:DELAY", "skip")
        self.assertEqual(result["factors"][0]["factor"], "deck_fitness:defense")
        self.assertNotIn("deck_fitness:pick_margin", [f["factor"] for f in result["factors"]])
        self.assertEqual(result["deck_score_difference"], self.review.decision("F15-card").record["gap"])

    def test_deck_at_a_decision(self):
        deck = tools.deck_at(self.review, "F15-card")
        self.assertEqual(deck["floor"], 15)
        self.assertEqual(deck["deck_size"], sum(c["count"] for c in deck["deck"]))
        self.assertTrue(all(c["name"] for c in deck["deck"]))

    def test_what_if_rescores_with_real_alternatives_only(self):
        result = tools.what_if(self.review, "F15-card", [{"operation": "remove", "card": "STRIKE_NECROBINDER"}])
        self.assertEqual(result["engine_top_before"], "0:DELAY")
        self.assertIn("pick_changed", result)
        with self.assertRaisesRegex(ReviewError, "not offered before floor 15"):
            tools.what_if(self.review, "F15-card", [{"operation": "add", "card": "NO_ESCAPE"}])
        with self.assertRaisesRegex(ReviewError, "has no upgrade"):
            tools.what_if(self.review, "F15-card", [{"operation": "upgrade", "card": "GUILTY"}])
        with self.assertRaisesRegex(ReviewError, "no SHROUD in the deck"):
            tools.what_if(self.review, "F15-card", [{"operation": "remove", "card": "SHROUD"}])
        with self.assertRaisesRegex(ReviewError, "card rewards only"):
            tools.what_if(self.review, "F07-rest", [{"operation": "remove", "card": "STRIKE_NECROBINDER"}])
        with self.assertRaisesRegex(ReviewError, "1 to 3"):
            tools.what_if(self.review, "F15-card", [])

    def test_what_if_leaves_the_decision_unchanged(self):
        before = copy.deepcopy(self.review.decision("F15-card").view["state"]["deck"])
        tools.what_if(self.review, "F15-card", [{"operation": "add", "card": "SHROUD"}])
        self.assertEqual(self.review.decision("F15-card").view["state"]["deck"], before)

    def test_unknown_decision_names_the_scored_ones(self):
        with self.assertRaisesRegex(ReviewError, "F15-card"):
            self.review.decision("F99-card")

    def test_only_finished_single_player_runs(self):
        run = load_run(RUN)
        abandoned = dict(run, was_abandoned=True)
        with self.assertRaisesRegex(ReviewError, "abandoned"):
            RunReview(self.engine, abandoned, "x")
        multiplayer = dict(run, players=run["players"] * 2)
        with self.assertRaisesRegex(ReviewError, "single-player"):
            RunReview(self.engine, multiplayer, "x")


class DeckTrendTests(unittest.TestCase):
    """deck_trend on a development-set run (IRONCLAD, v0.98.2)."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.engine = Engine(Path(cls.temp.name) / "release.db")
        cls.review = review_file(cls.engine, DEV_RUN)

    @classmethod
    def tearDownClass(cls):
        del cls.review, cls.engine
        cls.temp.cleanup()

    def test_one_row_per_card_reward_with_the_numbers_the_factor_messages_show(self):
        trend = tools.deck_trend(self.review, "defense")
        cards = [d.id for d in self.review.decisions.values() if d.kind == "card_reward"]
        self.assertEqual([r["decision_id"] for r in trend["decisions"]], cards)
        row = next(r for r in trend["decisions"] if r["decision_id"] == "F14-card")
        defense = row["values"]["defense"]
        messages = [f["message"] for o in self.review.decision("F14-card").record["options"] for f in o["factors"]]
        self.assertTrue(any(f"每回合 {defense['have']} →" in m and f"需求 {defense['need']}" in m for m in messages))
        self.assertEqual(defense["below_need"], defense["have"] < defense["need"])
        self.assertEqual(trend["labels"], {"defense": "防御"})

    def test_all_dimensions_by_default_and_unknown_ones_rejected(self):
        trend = tools.deck_trend(self.review)
        self.assertEqual(set(trend["decisions"][0]["values"]),
                         {"frontload", "aoe", "defense", "scaling", "cycle", "energy"})
        with self.assertRaisesRegex(ReviewError, "dimension must be one of"):
            tools.deck_trend(self.review, "luck")


if __name__ == "__main__":
    unittest.main()
