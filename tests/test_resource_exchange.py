import json
import unittest
from pathlib import Path

from advisor.decision_core import WorldState
from advisor.resource_exchange import CAMPFIRE_KEY, PARAMETERS_KEY, parameter_gaps, resource_points

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = json.loads((ROOT / "data" / "guide_mechanics.json").read_text(encoding="utf-8"))
EXCHANGE = PARAMETERS[PARAMETERS_KEY]
CAMPFIRE = PARAMETERS[CAMPFIRE_KEY]


def world(*, hp=80, max_hp=80, gold=99, act=1):
    return WorldState.create(
        run_id="exchange", sequence=1,
        state={
            "character": "IRONCLAD", "act": act, "floor": 1, "hp": hp, "max_hp": max_hp,
            "gold": gold, "energy": 3, "deck": [{"card": "STRIKE_IRONCLAD", "count": 5}],
            "relics": [], "potions": [], "max_potion_slots": 3,
        },
    )


def points(resources, **state):
    total, reasons, gaps = resource_points(resources, world(**state), EXCHANGE, CAMPFIRE)
    return total, reasons, gaps


class ResourceExchangeTests(unittest.TestCase):
    def test_shipped_parameters_are_valid(self):
        self.assertEqual(parameter_gaps(EXCHANGE, CAMPFIRE), ())

    def test_gold_is_worth_less_when_rich_and_in_later_acts(self):
        poor, reasons, gaps = points({"gold": 150}, gold=50, act=1)
        rich, _, _ = points({"gold": 150}, gold=900, act=1)
        third, _, _ = points({"gold": 150}, gold=50, act=3)
        self.assertGreater(poor, rich)
        self.assertGreater(poor, third)       # fewer shops ahead
        self.assertGreaterEqual(rich, 0)
        self.assertEqual((reasons[0]["code"], gaps), ("resource:gold", ()))

    def test_spending_costs_what_the_gold_would_buy_and_less_when_rich(self):
        from advisor.resource_exchange import gold_points
        poor = gold_points(150, -75, world(gold=150), EXCHANGE)
        rich = gold_points(900, -75, world(gold=900), EXCHANGE)
        self.assertLess(poor, rich)            # both negative; the rich lose less
        self.assertLess(poor, 0)

    def test_only_the_gold_there_is_can_be_lost(self):
        lost, _, _ = points({"gold": -300}, gold=40)
        everything, _, _ = points({"gold": -40}, gold=40)
        self.assertAlmostEqual(lost, everything)
        self.assertTrue(resource_points({"gold": 50}, world(gold=None), EXCHANGE, CAMPFIRE)[2])

    def test_hp_is_worth_more_when_low_and_a_heal_stops_at_max(self):
        low, _, _ = points({"hp": 10}, hp=20)
        high, _, _ = points({"hp": 10}, hp=60)
        self.assertGreater(low, high)
        capped, reasons, _ = points({"hp": 30}, hp=75)
        self.assertAlmostEqual(capped, points({"hp": 5}, hp=75)[0])
        self.assertIn("+5", reasons[0]["message"])

    def test_losing_hp_costs_points(self):
        cost, _, _ = points({"hp": -16}, hp=80)
        self.assertLess(cost, 0)

    def test_max_hp_heals_and_lasts_through_the_rests_ahead(self):
        # GainMaxHp raises the maximum and heals the same amount; every later
        # rest heal is 30% of max HP, so the gain also lasts.
        from advisor.resource_exchange import rests_ahead
        gain, reasons, _ = points({"max_hp": 11}, hp=80)
        heal_only, _, _ = points({"hp": 11}, hp=80, max_hp=91)
        lasting = 11 * EXCHANGE["rest_heal_share"] * rests_ahead(world(), EXCHANGE) * EXCHANGE["hp_points_at_threshold"]
        self.assertAlmostEqual(gain, heal_only + lasting)
        self.assertGreater(lasting, 0)
        self.assertEqual(reasons[0]["code"], "resource:max_hp")
        late, _, _ = points({"max_hp": 11}, hp=80, act=3)
        self.assertLess(late, gain)      # fewer rests left

    def test_unknown_hp_is_a_gap_and_bad_parameters_fail_closed(self):
        self.assertEqual(
            resource_points({"hp": 5}, world(hp=None), EXCHANGE, CAMPFIRE)[2], ("resource:hp_missing",),
        )
        self.assertTrue(parameter_gaps({**EXCHANGE, "hp_value_scale": {"min": 2, "max": 1}}, CAMPFIRE))
        self.assertTrue(parameter_gaps(None, CAMPFIRE))
        self.assertTrue(parameter_gaps(EXCHANGE, {}))


if __name__ == "__main__":
    unittest.main()
