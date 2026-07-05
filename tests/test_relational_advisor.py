import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from advisor.card_reward import (
    BASELINE_METHOD,
    COMMUNITY_PRIOR_METHOD,
    recommend_card_reward,
)
from api import app
from rag.knowledge import load_runtime_knowledge
from storage.effect_tags import derive_effect_tags
from storage.relational import RelationalRepository


def _knowledge_payload():
    return {
        "characters": [
            {
                "id": "TEST_HERO",
                "name": "测试角色",
                "description": "",
                "embed_text": "角色测试角色",
            }
        ],
        "cards": [
            {
                "id": "HEAVY_ATTACK",
                "name": "重击测试牌",
                "description": "造成较高伤害。",
                "cost": 3,
                "type_key": "Attack",
                "rarity_key": "Common",
                "color": "test",
                "damage": 18,
                "block": None,
                "keywords_key": [],
                "embed_text": "卡牌重击测试牌：造成较高伤害。",
            },
            {
                "id": "CHEAP_BLOCK",
                "name": "轻防测试牌",
                "description": "获得格挡。",
                "cost": 1,
                "type_key": "Skill",
                "rarity_key": "Common",
                "color": "test",
                "damage": None,
                "block": 8,
                "cards_draw": 1,
                "keywords_key": [],
                "embed_text": "卡牌轻防测试牌：获得8点格挡。",
            },
            {
                "id": "NEUTRAL_SKILL",
                "name": "中性测试牌",
                "description": "没有基础攻防数值。",
                "cost": 1,
                "type_key": "Skill",
                "rarity_key": "Common",
                "color": "test",
                "damage": None,
                "block": None,
                "keywords_key": [],
                "embed_text": "卡牌中性测试牌。",
            },
            {
                "id": "EXHAUST_PROVIDER",
                "name": "耗竭提供牌",
                "description": "造成9点伤害。",
                "cost": 1,
                "type_key": "Attack",
                "rarity_key": "Common",
                "color": "test",
                "damage": 9,
                "block": None,
                "keywords_key": ["Exhaust"],
                "embed_text": "卡牌耗竭提供牌。",
            },
            {
                "id": "EXHAUST_PAYOFF",
                "name": "耗竭收益牌",
                "description": "每当一张牌被耗竭时，获得2点格挡。",
                "cost": 1,
                "type_key": "Power",
                "rarity_key": "Uncommon",
                "color": "test",
                "damage": None,
                "block": None,
                "keywords_key": [],
                "embed_text": "卡牌耗竭收益牌。",
            },
        ],
        "relics": [
            {
                "id": "TEST_RELIC",
                "name": "测试遗物",
                "description": "技能牌提供格挡时，获得额外格挡。",
                "pool": "test",
                "rarity_key": "Common",
                "embed_text": "遗物测试遗物。",
            }
        ],
        "potions": [
            {
                "id": "TEST_DAMAGE_POTION",
                "name": "测试伤害药水",
                "description": "造成20点伤害。",
                "embed_text": "药水测试伤害药水。",
            }
        ],
        "monsters": [],
    }


class RelationalAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.knowledge_path = os.path.join(self.tempdir.name, "knowledge.json")
        self.guides_path = os.path.join(self.tempdir.name, "guides.json")
        self.database_path = os.path.join(self.tempdir.name, "sts2.db")
        with open(self.knowledge_path, "w", encoding="utf-8") as file:
            json.dump(_knowledge_payload(), file, ensure_ascii=False)
        with open(self.guides_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "guides": [
                        {
                            "slug": "test-guide",
                            "title": "Test Guide",
                            "content": "# Plan\nKeep a balanced deck.",
                        }
                    ]
                },
                file,
                ensure_ascii=False,
            )
        self.repository = RelationalRepository(self.database_path)
        self.repository.sync_catalog(self.knowledge_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def _state(self):
        return {
            "character": "TEST_HERO",
            "ascension": 0,
            "act": 1,
            "floor": 3,
            "energy": 3,
            "deck": [
                {"card": "HEAVY_ATTACK", "count": 3, "upgrades": 0},
            ],
            "relics": ["TEST_RELIC"],
        }

    def test_runtime_storage_boundary_only_embeds_guides(self):
        docs, items, index = load_runtime_knowledge(
            self.repository,
            path=self.knowledge_path,
            guides_path=self.guides_path,
        )
        self.assertEqual(len(docs), 1)
        self.assertEqual(items[0]["_type"], "guides")
        self.assertEqual(index["cards"][0]["name"], "重击测试牌")
        self.assertNotIn("卡牌重击测试牌", docs)

    def test_effect_tags_are_materialized_in_sqlite(self):
        card = self.repository.find_card("CHEAP_BLOCK")
        relic = self.repository.find_relic("TEST_RELIC")
        self.assertEqual(card["_effect_tags"]["block"], 8.0)
        self.assertEqual(card["_effect_tags"]["draw"], 1.0)
        self.assertIn("supports_skill", relic["_effect_tags"])
        self.assertIn("supports_block", relic["_effect_tags"])

        with self.repository.connect() as connection:
            stored = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM entity_effect_tags
                WHERE entity_key = 'cards:CHEAP_BLOCK'
                """
            ).fetchone()["count"]
        self.assertGreaterEqual(stored, 3)

    def test_baseline_uses_state_without_claiming_trained_prediction(self):
        result = recommend_card_reward(
            self._state(),
            [
                {"card": "HEAVY_ATTACK", "upgrades": 0},
                {"card": "CHEAP_BLOCK", "upgrades": 0},
            ],
            self.repository,
        )
        self.assertEqual(result["method"], BASELINE_METHOD)
        self.assertEqual(result["recommended_option"], "轻防测试牌")
        self.assertIn("baseline", result["disclaimer"])
        self.assertEqual(result["confidence"], "medium")
        self.assertEqual(result["decision_status"], "recommend")
        self.assertEqual(result["skip_candidate"]["choice"], "skip")
        self.assertNotEqual(result["skip_score"], 50.0)
        self.assertEqual(result["recommended_option_index"], 1)

    def test_hp_relic_and_route_context_add_traceable_factors(self):
        state = {
            **self._state(),
            "hp": 20,
            "max_hp": 80,
            "map_context": {
                "available_next_node_ids": ["next"],
                "nodes": [
                    {
                        "node_id": "next",
                        "kind": "ELITE",
                        "edges": [],
                    }
                ],
            },
        }
        result = recommend_card_reward(
            state,
            [{"card": "CHEAP_BLOCK"}],
            self.repository,
        )
        codes = {
            factor["code"]
            for factor in result["recommendations"][0]["factors"]
        }
        self.assertIn("critical_hp_defense", codes)
        self.assertIn("relic_synergy", codes)
        self.assertIn("route_elite_fit", codes)
        self.assertEqual(result["profile"]["hp_ratio"], 0.25)
        self.assertEqual(
            result["profile"]["route"]["elite_path_ratio"],
            1.0,
        )

    def test_skip_score_rises_as_deck_becomes_large(self):
        small = recommend_card_reward(
            {
                **self._state(),
                "deck": [{"card": "HEAVY_ATTACK", "count": 4}],
            },
            [{"card": "NEUTRAL_SKILL"}],
            self.repository,
        )
        large = recommend_card_reward(
            {
                **self._state(),
                "deck": [{"card": "HEAVY_ATTACK", "count": 28}],
            },
            [{"card": "NEUTRAL_SKILL"}],
            self.repository,
        )
        self.assertGreater(large["skip_score"], small["skip_score"])
        self.assertTrue(large["skip_candidate"]["factors"])


    def test_skip_is_an_explicit_option_with_state_evidence(self):
        result = recommend_card_reward(
            self._state(),
            [{"card": "HEAVY_ATTACK", "upgrades": 0}],
            self.repository,
        )

        self.assertTrue(result["skip_recommended"])
        self.assertEqual(result["decision_status"], "skip")
        self.assertIsNone(result["recommended_option"])
        self.assertTrue(result["skip_candidate"]["eligible"])
        self.assertEqual(result["skip_candidate"]["rank"], 1)

    def test_uncertain_is_distinct_from_skip(self):
        state = {
            **self._state(),
            "deck": [
                {"card": "CHEAP_BLOCK", "count": 3, "upgrades": 0}
            ],
        }
        result = recommend_card_reward(
            state,
            [{"card": "NEUTRAL_SKILL"}],
            self.repository,
        )

        self.assertFalse(result["skip_recommended"])
        self.assertEqual(result["decision_status"], "uncertain")
        self.assertEqual(result["recommended_option"], "中性测试牌")
        self.assertFalse(result["skip_candidate"]["eligible"])

    def test_state_and_decision_are_persisted_relationally(self):
        result = recommend_card_reward(
            self._state(),
            [{"card": "CHEAP_BLOCK", "upgrades": 1}],
            self.repository,
        )
        state_id = self.repository.save_run_state(self._state())
        decision_id = self.repository.save_card_reward_decision(
            state_id,
            result["recommendations"],
            result["recommended_option"],
            result["confidence"],
            result["method"],
        )
        stored = self.repository.load_decision(decision_id)
        self.assertEqual(stored["state_id"], state_id)
        self.assertEqual(stored["method"], BASELINE_METHOD)
        self.assertEqual(stored["recommendations"][0]["card"], "轻防测试牌")

        self.repository.save_decision_outcome(
            decision_id,
            chosen_option="CHEAP_BLOCK",
        )
        self.repository.save_decision_outcome(
            decision_id,
            run_won=False,
            final_floor=20,
        )
        stored = self.repository.load_decision(decision_id)
        self.assertEqual(stored["outcome"]["chosen_option"], "CHEAP_BLOCK")
        self.assertFalse(stored["outcome"]["run_won"])
        with self.assertRaises(ValueError):
            self.repository.save_decision_outcome(
                decision_id,
                chosen_option="NOT_A_CANDIDATE",
            )

    def test_approved_community_score_is_a_traceable_weak_prior(self):
        scores_path = os.path.join(self.tempdir.name, "scores.json")
        with open(scores_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "source": {
                        "id": "spire_codex_api",
                        "endpoints": {
                            "cards": (
                                "https://spire-codex.com/api/"
                                "runs/scores/cards"
                            )
                        },
                    },
                    "fetched_at": "2026-07-01T00:00:00+00:00",
                    "game_version": "mixed",
                    "methodology": "test aggregate",
                    "entities": {
                        "cards": {
                            "CHEAP_BLOCK": {
                                "score": 90,
                                "elo": 2000,
                                "picks": 10000,
                                "wins": 6000,
                                "win_rate": 60.0,
                            }
                        }
                    },
                },
                file,
            )
        imported = self.repository.sync_entity_statistics(scores_path)
        self.assertEqual(imported["cards"], 1)
        imported_again = self.repository.sync_entity_statistics(scores_path)
        self.assertEqual(imported_again["cards"], 1)
        self.assertEqual(
            self.repository.statistics_status()["cards"],
            1,
        )

        result = recommend_card_reward(
            self._state(),
            [{"card": "CHEAP_BLOCK"}],
            self.repository,
        )
        self.assertEqual(result["method"], COMMUNITY_PRIOR_METHOD)
        prior = next(
            factor
            for factor in result["recommendations"][0]["factors"]
            if factor["code"] == "community_prior"
        )
        self.assertEqual(prior["source_name"], "Spire Codex API")
        self.assertEqual(prior["sample_size"], 10000)
        self.assertLess(prior["delta"], 10)

    def test_weak_community_prior_cannot_recommend_skip_by_itself(self):
        scores_path = os.path.join(self.tempdir.name, "negative-scores.json")
        with open(scores_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "source": {
                        "id": "spire_codex_api",
                        "endpoints": {
                            "cards": "https://spire-codex.com/api/test"
                        },
                    },
                    "fetched_at": "2026-07-01T00:00:00+00:00",
                    "game_version": "mixed",
                    "methodology": "test aggregate",
                    "entities": {
                        "cards": {
                            "NEUTRAL_SKILL": {
                                "score": 10,
                                "elo": 1000,
                                "picks": 10000,
                                "wins": 1000,
                                "win_rate": 10.0,
                            }
                        }
                    },
                },
                file,
            )
        self.repository.sync_entity_statistics(scores_path)
        state = {
            **self._state(),
            "deck": [
                {"card": "CHEAP_BLOCK", "count": 3, "upgrades": 0}
            ],
        }
        result = recommend_card_reward(
            state,
            [{"card": "NEUTRAL_SKILL"}],
            self.repository,
        )
        self.assertLess(result["recommendations"][0]["score"], 50)
        self.assertGreater(
            result["recommendations"][0]["state_score"],
            result["recommendations"][0]["score"],
        )
        self.assertFalse(result["skip_recommended"])
        self.assertEqual(result["recommended_option"], "中性测试牌")
        self.assertEqual(result["decision_status"], "uncertain")

    def test_unknown_card_keeps_stable_id_for_panel_matching(self):
        result = recommend_card_reward(
            self._state(),
            [{"card": "UNKNOWN_STABLE_CARD_ID"}],
            self.repository,
        )
        recommendation = result["recommendations"][0]
        self.assertFalse(recommendation["known"])
        self.assertEqual(
            recommendation["card_id"],
            "UNKNOWN_STABLE_CARD_ID",
        )

    def test_card_reward_api_does_not_persist_decision_history(self):
        with patch(
            "api.get_relational_repository",
            return_value=self.repository,
        ), patch(
            "api.KNOWLEDGE_FILE",
            self.knowledge_path,
        ), patch(
            "api.COMMUNITY_SCORES_FILE",
            os.path.join(self.tempdir.name, "missing-scores.json"),
        ):
            client = TestClient(app)
            response = client.post(
                "/recommend/card-reward",
                json={
                    "state": {
                        "character": "TEST_HERO",
                        "act": 1,
                        "floor": 3,
                        "deck": [
                            {"card": "HEAVY_ATTACK", "count": 3}
                        ],
                        "relics": ["TEST_RELIC"],
                    },
                    "options": ["HEAVY_ATTACK", "CHEAP_BLOCK"],
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIsNone(payload["decision_id"])
            self.assertIsNone(payload["state_id"])
            self.assertEqual(payload["recommended_option"], "轻防测试牌")

            with self.repository.connect() as connection:
                states = connection.execute(
                    "SELECT COUNT(*) AS count FROM run_states"
                ).fetchone()["count"]
                decisions = connection.execute(
                    "SELECT COUNT(*) AS count FROM decision_events"
                ).fetchone()["count"]
            self.assertEqual(states, 0)
            self.assertEqual(decisions, 0)

            audit_response = client.get("/decisions/legacy-id")
            self.assertEqual(audit_response.status_code, 410)

            outcome_response = client.post(
                "/decisions/legacy-id/outcome",
                json={
                    "chosen_option": "轻防测试牌",
                    "run_won": True,
                    "final_floor": 52,
                },
            )
            self.assertEqual(outcome_response.status_code, 410)

            invalid_response = client.post(
                "/recommend/card-reward",
                json={
                    "state": {
                        "character": "UNKNOWN",
                        "act": 1,
                        "floor": 1,
                    },
                    "options": ["CHEAP_BLOCK"],
                },
            )
            self.assertEqual(invalid_response.status_code, 400)

    # ── v2 contextual scoring factors ─────────────────────────────

    @staticmethod
    def _state_with_map():
        return {
            "character": "TEST_HERO",
            "ascension": 0,
            "act": 1,
            "floor": 5,
            "energy": 3,
            "hp": 60,
            "max_hp": 80,
            "deck": [
                {"card": "HEAVY_ATTACK", "count": 5, "upgrades": 0},
            ],
            "relics": ["TEST_RELIC"],
            "map_context": {
                "nodes": [
                    {
                        "node_id": "1:0:ELITE",
                        "kind": "ELITE",
                        "row": 1,
                        "col": 0,
                        "edges": ["2:0:CAMPFIRE"],
                    },
                    {
                        "node_id": "1:1:MONSTER",
                        "kind": "MONSTER",
                        "row": 1,
                        "col": 1,
                        "edges": ["2:1:MONSTER"],
                    },
                    {
                        "node_id": "2:0:CAMPFIRE",
                        "kind": "CAMPFIRE",
                        "row": 2,
                        "col": 0,
                        "edges": [],
                    },
                    {
                        "node_id": "2:1:MONSTER",
                        "kind": "MONSTER",
                        "row": 2,
                        "col": 1,
                        "edges": [],
                    },
                ],
                "available_next_node_ids": ["1:0:ELITE", "1:1:MONSTER"],
                "player_row": 0,
            },
        }

    @classmethod
    def _state_with_boss(cls):
        state = cls._state_with_map()
        state["map_context"] = {
            "nodes": [
                {
                    "node_id": "1:0:BOSS",
                    "kind": "BOSS",
                    "row": 1,
                    "col": 0,
                    "edges": [],
                }
            ],
            "available_next_node_ids": ["1:0:BOSS"],
            "player_row": 0,
        }
        return state

    def test_optional_elite_produces_one_path_scaled_factor(self):
        result = recommend_card_reward(
            self._state_with_map(),
            [
                {"card": "CHEAP_BLOCK", "upgrades": 0},
                {"card": "HEAVY_ATTACK", "upgrades": 0},
            ],
            self.repository,
        )
        codes = {
            factor["code"]
            for rec in result["recommendations"]
            for factor in rec["factors"]
            if factor["code"].startswith("route_")
        }
        self.assertEqual(codes, {"route_elite_fit"})
        route = result["profile"]["route"]
        self.assertEqual(route["path_count"], 2)
        self.assertEqual(route["elite_path_ratio"], 0.5)

    def test_boss_proximity_produces_one_boss_factor(self):
        result = recommend_card_reward(
            self._state_with_boss(),
            [{"card": "HEAVY_ATTACK", "upgrades": 0}],
            self.repository,
        )
        route_codes = {
            factor["code"]
            for rec in result["recommendations"]
            for factor in rec["factors"]
            if factor["code"].startswith("route_")
        }
        self.assertEqual(route_codes, {"route_boss_fit"})

    def test_route_without_map_context_shows_no_threat_factors(self):
        result = recommend_card_reward(
            self._state(),
            [{"card": "CHEAP_BLOCK", "upgrades": 0}],
            self.repository,
        )
        route = result["profile"]["route"]
        self.assertFalse(route["known"])
        self.assertEqual(route["path_count"], 0)
        self.assertEqual(route["elite_path_ratio"], 0)
        self.assertEqual(route["boss_path_ratio"], 0)

    def test_exhaust_synergy_requires_provider_and_payoff(self):
        without_payoff = recommend_card_reward(
            self._state(),
            [{"card": "EXHAUST_PROVIDER", "upgrades": 0}],
            self.repository,
        )
        without_codes = {
            factor["code"]
            for factor in without_payoff["recommendations"][0]["factors"]
        }
        self.assertNotIn("effect_engine_synergy", without_codes)

        state = {
            **self._state(),
            "deck": [{"card": "EXHAUST_PAYOFF", "count": 1}],
        }
        result = recommend_card_reward(
            state,
            [{"card": "EXHAUST_PROVIDER", "upgrades": 0}],
            self.repository,
        )
        codes = {
            factor["code"]
            for factor in result["recommendations"][0]["factors"]
        }
        self.assertIn("effect_engine_synergy", codes)

    def test_potions_are_captured_but_do_not_score_cards(self):
        state = {
            **self._state(),
            "potions": [
                {"potion": "TEST_DAMAGE_POTION", "slot": 0}
            ],
        }
        result = recommend_card_reward(
            state,
            [{"card": "HEAVY_ATTACK", "upgrades": 0}],
            self.repository,
        )
        codes = {
            factor["code"]
            for factor in result["recommendations"][0]["factors"]
        }
        self.assertNotIn("potion_synergy", codes)
        self.assertNotIn("potion_effects", result["profile"])
        potion = self.repository.find_entity(
            "potions",
            "TEST_DAMAGE_POTION",
        )
        self.assertEqual(potion["_effect_tags"], {})

    def test_threat_model_is_path_aware(self):
        result = recommend_card_reward(
            self._state_with_map(),
            [{"card": "HEAVY_ATTACK", "upgrades": 0}],
            self.repository,
        )
        route = result["profile"]["route"]
        self.assertIn("danger", route)
        self.assertEqual(route["path_count"], 2)
        self.assertEqual(route["elite_path_ratio"], 0.5)
        self.assertEqual(route["boss_path_ratio"], 0.0)
        self.assertLess(route["min_danger"], route["max_danger"])

    def test_effect_tag_negative_and_positive_examples(self):
        cinder = derive_effect_tags(
            "cards",
            {"description": "消耗你的抽牌堆顶部的牌。"},
        )
        pagestorm = derive_effect_tags(
            "cards",
            {"description": "每当你抽到一张虚无牌时，抽1张牌。"},
        )
        doubt = derive_effect_tags(
            "cards",
            {"description": "如果这张牌在手牌中，获得1层虚弱。"},
        )
        eyes = derive_effect_tags(
            "cards",
            {"description": "如果敌人的意图是攻击，则给予1层虚弱。"},
        )
        dismantle = derive_effect_tags(
            "cards",
            {"description": "如果该敌人有易伤状态，则攻击两次。"},
        )
        bubble = derive_effect_tags(
            "cards",
            {"description": "如果敌方拥有中毒，则给予9层中毒。"},
        )
        exhaust = derive_effect_tags(
            "cards",
            {"keywords_key": ["Exhaust"]},
        )
        ethereal = derive_effect_tags(
            "cards",
            {"keywords_key": ["Ethereal"]},
        )

        self.assertNotIn("self_exhaust", cinder)
        self.assertNotIn("ethereal", pagestorm)
        self.assertIn("ethereal_interaction", pagestorm)
        self.assertNotIn("cond:weak_target", doubt)
        self.assertNotIn("cond:weak_target", eyes)
        self.assertIn("cond:vulnerable_target", dismantle)
        self.assertIn("cond:poison_target", bubble)
        self.assertIn("self_exhaust", exhaust)
        self.assertNotIn("ethereal", exhaust)
        self.assertIn("ethereal", ethereal)
        self.assertNotIn("self_exhaust", ethereal)

    def test_effect_tag_v2_database_migrates_to_v3(self):
        with self.repository.connect() as connection:
            connection.execute(
                """
                UPDATE schema_metadata SET value = '2'
                WHERE key = 'effect_tag_version'
                """
            )
            connection.execute(
                """
                INSERT INTO entity_effect_tags(
                    entity_key, tag, magnitude, source_field
                ) VALUES(
                    'potions:TEST_DAMAGE_POTION',
                    'supports_attack',
                    1,
                    'stale_test'
                )
                """
            )
        changed = self.repository.sync_catalog(self.knowledge_path)
        self.assertFalse(changed)
        with self.repository.connect() as connection:
            version = connection.execute(
                """
                SELECT value FROM schema_metadata
                WHERE key = 'effect_tag_version'
                """
            ).fetchone()["value"]
            stale = connection.execute(
                """
                SELECT COUNT(*) AS count FROM entity_effect_tags
                WHERE entity_key = 'potions:TEST_DAMAGE_POTION'
                """
            ).fetchone()["count"]
        self.assertEqual(version, "3")
        self.assertEqual(stale, 0)

    def test_effect_tag_version_is_3(self):
        from storage.effect_tags import EFFECT_TAG_VERSION
        self.assertEqual(EFFECT_TAG_VERSION, "3")


if __name__ == "__main__":
    unittest.main()
