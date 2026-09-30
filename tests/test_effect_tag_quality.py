import json
from pathlib import Path
import tempfile
import unittest

from advisor.card_reward import recommend_card_reward
from storage.effect_tags import derive_effect_tags, derive_mechanic_effect_tags
from storage.relational import RelationalRepository


KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge.json"


class DiscardTagQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.repository = RelationalRepository(
            str(Path(cls.tempdir.name) / "tag-quality.db")
        )
        cls.repository.sync_catalog(str(KNOWLEDGE_PATH))
        with KNOWLEDGE_PATH.open(encoding="utf-8") as source:
            cls.cards = {
                card["id"]: card for card in json.load(source)["cards"]
            }

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_reflex_needs_actual_discard_action_in_deck(self):
        for deck_card, expected in (
            ("WELL_LAID_PLANS", False),
            ("DAGGER_THROW", True),
        ):
            with self.subTest(deck_card=deck_card):
                result = recommend_card_reward(
                    {
                        "character": "SILENT",
                        "act": 1,
                        "floor": 6,
                        "hp": 70,
                        "max_hp": 80,
                        "energy": 3,
                        "deck": [{"card": deck_card}],
                        "relics": [],
                    },
                    [{"card": "REFLEX"}],
                    self.repository,
                    can_skip=False,
                )
                codes = {
                    factor["code"]
                    for factor in result["recommendations"][0]["factors"]
                }
                self.assertEqual(
                    "mechanic_synergy_payoff:discard" in codes, expected
                )

    def test_negated_discard_is_not_an_action_or_provider(self):
        for description in (
            "你不丢弃手牌。",
            "不要丢弃手牌。",
            "你不会丢弃手牌。",
            "你不再自动丢弃所有手牌。",
            "你不能丢弃手牌。",
            "无需丢弃任何牌。",
            "保留的牌不会在回合结束时被弃掉。",
            "You no longer discard your hand.",
            "Do not discard any cards.",
            "You don't discard your hand.",
            "You cannot discard any cards.",
            "Cards cannot be discarded.",
            "Retained cards should not be discarded.",
            "Cards cannot be automatically discarded.",
            "Cards cannot automatically be discarded.",
            "Keep this card without being discarded.",
            "You can’t discard any cards.",
            "You never discard your hand.",
            "Draw 2 cards without discarding a card.",
        ):
            with self.subTest(description=description):
                card = {"color": "silent", "description": description}
                tags = derive_effect_tags("cards", card)
                self.assertNotIn("discard", tags)
                self.assertNotIn("mechanic:discard:provider", tags)
                self.assertNotIn(
                    "mechanic:discard:provider", derive_mechanic_effect_tags(card)
                )

    def test_discard_condition_is_not_its_own_provider(self):
        for description in (
            "每当你丢弃一张牌，抽1张牌。",
            "当你丢弃一张牌时，获得3点格挡。",
            "如果你本回合丢弃过一张牌，获得3点格挡。",
            "若你丢弃一张牌，获得3点格挡。",
            "丢弃一张牌时，获得3点格挡。",
            "Whenever you discard a card, draw 1 card.",
            "When a card is discarded, gain 3 Block.",
            "If you discarded a card this turn, gain 3 Block.",
            "After you discard a card, draw 1 card.",
        ):
            with self.subTest(description=description):
                tags = derive_effect_tags(
                    "cards", {"color": "silent", "description": description}
                )
                self.assertNotIn("discard", tags)
                self.assertNotIn("mechanic:discard:provider", tags)

    def test_discard_pile_references_are_not_providers(self):
        for description in (
            "将弃牌堆中的1张牌放入你的手牌。",
            "将1张牌置于你的弃牌堆。",
            "Return a card from your discard pile to your hand.",
            "Place a card in your discard pile.",
            "Choose one of your discard-pile cards.",
        ):
            with self.subTest(description=description):
                tags = derive_effect_tags(
                    "cards", {"color": "silent", "description": description}
                )
                self.assertNotIn("discard", tags)
                self.assertNotIn("mechanic:discard:provider", tags)

    def test_real_catalog_retention_and_active_discard_stay_distinct(self):
        for card_id, expected in (
            ("WELL_LAID_PLANS", False),
            ("DAGGER_THROW", True),
        ):
            with self.subTest(card_id=card_id):
                card = self.cards[card_id]
                tags = derive_effect_tags("cards", card)
                self.assertEqual("discard" in tags, expected)
                self.assertEqual("mechanic:discard:provider" in tags, expected)
                self.assertEqual(
                    "mechanic:discard:provider" in derive_mechanic_effect_tags(card),
                    expected,
                )

    def test_active_discard_survives_other_negated_or_triggered_clauses(self):
        for description in (
            "丢弃抽到的牌中耗能不为0的牌。",
            "不再丢弃你的手牌，但丢弃1张牌。",
            "如果手牌超过5张，丢弃1张牌。",
            "每当你抽牌，丢弃1张牌。",
            "丢弃所有手牌。每丢弃一张牌，获得1点格挡。",
            "If you have more than 5 cards, discard a card.",
            "Whenever you draw a card, discard a card.",
            "Do not discard your hand. Discard 1 card instead.",
            "Discard a card that cannot be played.",
        ):
            with self.subTest(description=description):
                tags = derive_effect_tags(
                    "cards", {"color": "silent", "description": description}
                )
                self.assertIn("discard", tags)
                self.assertIn("mechanic:discard:provider", tags)

    def test_upgraded_active_discard_is_still_a_provider(self):
        tags = derive_effect_tags("cards", {
            "color": "silent",
            "description": "不要丢弃手牌。",
            "upgrade_description": "丢弃1张牌。",
        })
        self.assertIn("discard", tags)
        self.assertIn("mechanic:discard:provider", tags)


if __name__ == "__main__":
    unittest.main()
