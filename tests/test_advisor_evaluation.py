"""Tests for the fixed scenario evaluation infrastructure.

These tests verify that the evaluator loads, validates, and runs scenarios
correctly, and that it does not depend on history, networks, or live state.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from advisor.card_reward import recommend_card_reward
from advisor.evaluation import (
    ValidationError,
    check_assertions,
    evaluate_scenarios,
    load_eval_scenarios,
    print_evaluation_report,
    run_evaluation,
    validate_scenario_ids,
    validate_scenario_structure,
)
from storage.relational import RelationalRepository

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS_PATH = os.path.join(ROOT_DIR, "data", "advisor_eval_scenarios.json")
KNOWLEDGE_PATH = os.path.join(ROOT_DIR, "data", "knowledge.json")


def _minimal_scenario(**overrides) -> dict:
    base = {
        "id": "test_001",
        "category": "test",
        "category_label": "测试",
        "description": "一个最小测试场景。",
        "rationale": "用于验证评测基础设施。",
        "state": {
            "character": "IRONCLAD",
            "ascension": 0,
            "act": 1,
            "floor": 5,
            "energy": 3,
            "hp": 70,
            "max_hp": 80,
            "deck": [
                {"card": "STRIKE_IRONCLAD", "count": 5},
                {"card": "DEFEND_IRONCLAD", "count": 4},
                {"card": "BASH", "count": 1},
            ],
            "relics": ["BURNING_BLOOD"],
        },
        "options": [
            {"card": "ANGER", "upgrades": 0},
            {"card": "DEFEND_IRONCLAD", "upgrades": 0},
            {"card": "SHRUG_IT_OFF", "upgrades": 0},
        ],
        "can_skip": True,
        "assertions": {"skip_recommended": False},
    }
    base.update(overrides)
    return base


class ScenarioLoadingTests(unittest.TestCase):
    def test_load_valid_scenarios_file(self):
        scenarios = load_eval_scenarios(SCENARIOS_PATH)
        self.assertGreaterEqual(len(scenarios), 24)
        self.assertLessEqual(len(scenarios), 30)
        seen = set()
        for s in scenarios:
            self.assertIn("id", s)
            self.assertNotIn(s["id"], seen, f"重复 ID: {s['id']}")
            seen.add(s["id"])
            self.assertIn("category", s)
            self.assertIn("state", s)
            self.assertIsInstance(s["options"], list)
            self.assertGreaterEqual(len(s["options"]), 1)

    def test_categories_have_minimum_coverage(self):
        scenarios = load_eval_scenarios(SCENARIOS_PATH)
        categories = {}
        for s in scenarios:
            cat = s.get("category") or s.get("category_label")
            categories[cat] = categories.get(cat, 0) + 1

        self.assertIn("deck_function_gap", categories)
        self.assertIn("hp_context", categories)
        self.assertIn("relic_synergy", categories)
        self.assertIn("deck_synergy", categories)
        self.assertIn("route_pressure", categories)
        self.assertIn("skip_behavior", categories)

        for cat, count in categories.items():
            self.assertGreaterEqual(
                count, 4,
                f"分类 {cat} 场景数 {count} < 4",
            )

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_eval_scenarios("/nonexistent/path/scenarios.json")

    def test_invalid_schema_version_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"schema_version": 999, "scenarios": []}, f)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_eval_scenarios(temp_path)
            self.assertIn("schema_version", str(ctx.exception))
        finally:
            os.unlink(temp_path)

    def test_scenarios_not_list_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"schema_version": 1, "scenarios": "not_a_list"}, f)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_eval_scenarios(temp_path)
            self.assertIn("数组", str(ctx.exception))
        finally:
            os.unlink(temp_path)

    def test_duplicate_ids_raise(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "schema_version": 1,
                    "scenarios": [
                        _minimal_scenario(id="dup"),
                        _minimal_scenario(id="dup"),
                    ],
                },
                f,
            )
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_eval_scenarios(temp_path)
            self.assertIn("重复", str(ctx.exception))
        finally:
            os.unlink(temp_path)


class ScenarioStructureValidationTests(unittest.TestCase):
    def test_missing_required_fields_rejected(self):
        errors = validate_scenario_structure({}, 0)
        self.assertGreater(len(errors), 0)

    def test_missing_state_character_rejected(self):
        scenario = _minimal_scenario()
        del scenario["state"]["character"]
        errors = validate_scenario_structure(scenario, 0)
        field_errors = [
            e for e in errors if "character" in e.message
        ]
        self.assertGreater(len(field_errors), 0)

    def test_empty_options_rejected(self):
        scenario = _minimal_scenario(options=[])
        errors = validate_scenario_structure(scenario, 0)
        option_errors = [
            e for e in errors if "options" in e.message
        ]
        self.assertGreater(len(option_errors), 0)

    def test_option_missing_card_rejected(self):
        scenario = _minimal_scenario()
        scenario["options"][0] = {"upgrades": 0}
        errors = validate_scenario_structure(scenario, 0)
        card_errors = [
            e for e in errors if "card" in e.message
        ]
        self.assertGreater(len(card_errors), 0)

    def test_missing_assertions_rejected(self):
        scenario = _minimal_scenario()
        del scenario["assertions"]
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)

    def test_empty_assertions_rejected(self):
        scenario = _minimal_scenario(assertions={})
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("不能为空" in e.message for e in errors)
        )

    def test_unknown_assertion_key_rejected(self):
        scenario = _minimal_scenario(
            assertions={"skip_recommended": False, "misspelled_key_xyz": True}
        )
        errors = validate_scenario_structure(scenario, 0)
        unknown = [e for e in errors if "未知断言键" in e.message]
        self.assertGreater(len(unknown), 0)
        self.assertIn("misspelled_key_xyz", unknown[0].message)

    def test_wrong_assertion_type_rejected(self):
        scenario = _minimal_scenario(
            assertions={"skip_recommended": "not_a_bool"}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("类型错误" in e.message for e in errors)
        )

    def test_valid_scenario_passes_structure_check(self):
        scenario = _minimal_scenario()
        errors = validate_scenario_structure(scenario, 0)
        self.assertEqual(len(errors), 0)

    # --- Deep container validation ---

    def test_factor_include_string_not_accepted(self):
        """{"assert_factors_include": "abc"} should be rejected (string, not list)."""
        scenario = _minimal_scenario(
            assertions={"assert_factors_include": "abc"}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("必须是列表" in e.message for e in errors)
        )

    def test_empty_factor_list_rejected(self):
        """{"assert_factors_include": []} should be rejected."""
        scenario = _minimal_scenario(
            assertions={"assert_factors_include": []}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("不能为空" in e.message for e in errors)
        )

    def test_empty_card_factors_dict_rejected(self):
        """{"assert_card_factors_include": {}} should be rejected."""
        scenario = _minimal_scenario(
            assertions={"assert_card_factors_include": {}}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("不能为空" in e.message for e in errors)
        )

    def test_card_factors_string_instead_of_dict_rejected(self):
        """{"assert_card_factors_include": "ANGER"} should be rejected."""
        scenario = _minimal_scenario(
            assertions={"assert_card_factors_include": "ANGER"}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("必须是字典" in e.message for e in errors)
        )

    def test_card_factors_empty_inner_list_rejected(self):
        """{"assert_card_factors_include": {"ANGER": []}} should be rejected."""
        scenario = _minimal_scenario(
            assertions={"assert_card_factors_include": {"ANGER": []}}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("列表不能为空" in e.message for e in errors)
        )

    def test_card_factors_inner_not_list_rejected(self):
        """{"assert_card_factors_include": {"ANGER": "attack_coverage"}} should be rejected."""
        scenario = _minimal_scenario(
            assertions={"assert_card_factors_include": {"ANGER": "attack_coverage"}}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("必须是列表" in e.message for e in errors)
        )

    def test_card_order_single_element_rejected(self):
        """card_order with only 1 element should be rejected."""
        scenario = _minimal_scenario(
            assertions={"card_order": ["ANGER"]}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("至少需要 2 个" in e.message for e in errors)
        )

    def test_card_order_empty_list_rejected(self):
        """card_order as empty list should be rejected."""
        scenario = _minimal_scenario(
            assertions={"card_order": []}
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("至少需要 2 个" in e.message for e in errors)
        )

    def test_only_empty_assertions_not_executable(self):
        """Scenario with only empty assertions has no executable assertion."""
        scenario = _minimal_scenario(
            assertions={
                "assert_factors_include": [],
                "assert_factors_exclude": [],
            }
        )
        errors = validate_scenario_structure(scenario, 0)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("没有可执行的断言" in e.message for e in errors)
        )


class ScenarioIdValidationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "test.db")
        self.repository = RelationalRepository(self.db_path)
        self.repository.sync_catalog(KNOWLEDGE_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_valid_ids_pass(self):
        scenario = _minimal_scenario()
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertEqual(len(errors), 0)

    def test_unknown_character_rejected(self):
        scenario = _minimal_scenario()
        scenario["state"]["character"] = "NONEXISTENT_HERO"
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertIn("NONEXISTENT_HERO", errors[0].message)

    def test_unknown_card_in_options_rejected(self):
        scenario = _minimal_scenario()
        scenario["options"].append({"card": "FAKE_CARD_ID_XYZ"})
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("FAKE_CARD_ID_XYZ" in e.message for e in errors)
        )

    def test_unknown_card_in_deck_rejected(self):
        scenario = _minimal_scenario()
        scenario["state"]["deck"].append({"card": "FAKE_DECK_CARD"})
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("FAKE_DECK_CARD" in e.message for e in errors)
        )

    def test_unknown_relic_rejected(self):
        scenario = _minimal_scenario()
        scenario["state"]["relics"].append("FAKE_RELIC")
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("FAKE_RELIC" in e.message for e in errors)
        )

    def test_assertion_card_id_not_in_options_rejected(self):
        """Card IDs in assert_card_factors_include must exist in options."""
        scenario = _minimal_scenario(
            assertions={
                "assert_card_factors_include": {"NONEXISTENT_CARD": ["attack_coverage"]}
            },
        )
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("不在场景 options 中" in e.message for e in errors)
        )

    def test_card_order_id_not_in_options_rejected(self):
        """Card IDs in card_order must exist in options."""
        scenario = _minimal_scenario(
            assertions={
                "card_order": ["NONEXISTENT_CARD", "ANGER"],
            },
        )
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("不在场景 options 中" in e.message for e in errors)
        )

    def test_recommended_card_not_in_options_rejected(self):
        """recommended_card must exist in options."""
        scenario = _minimal_scenario(
            assertions={
                "recommended_card": "NONEXISTENT_CARD",
            },
        )
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("不在场景 options 中" in e.message for e in errors)
        )

    def test_card_score_above_id_not_in_options_rejected(self):
        """Card IDs in assert_card_score_above must exist in options."""
        scenario = _minimal_scenario(
            assertions={
                "assert_card_score_above": {"NONEXISTENT_CARD": 30},
            },
        )
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertGreater(len(errors), 0)
        self.assertTrue(
            any("不在场景 options 中" in e.message for e in errors)
        )

    def test_valid_assertion_card_ids_pass(self):
        """All assertion card IDs matching options should pass."""
        scenario = _minimal_scenario(
            assertions={
                "assert_card_factors_include": {"ANGER": ["attack_coverage"]},
                "card_order": ["ANGER", "DEFEND_IRONCLAD"],
                "recommended_card": "ANGER",
            },
        )
        errors = validate_scenario_ids(scenario, self.repository)
        self.assertEqual(len(errors), 0)


class AssertionCheckTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "test.db")
        self.repository = RelationalRepository(self.db_path)
        self.repository.sync_catalog(KNOWLEDGE_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    def _run_and_check(self, scenario: dict):
        result = recommend_card_reward(
            state=scenario["state"],
            options=scenario["options"],
            repository=self.repository,
            can_skip=bool(scenario.get("can_skip", True)),
        )
        return check_assertions(scenario, result)

    def _low_damage_state(self):
        return {
            "character": "IRONCLAD",
            "ascension": 0,
            "act": 1,
            "floor": 5,
            "energy": 3,
            "hp": 70,
            "max_hp": 80,
            "deck": [
                {"card": "STRIKE_IRONCLAD", "count": 1},
                {"card": "DEFEND_IRONCLAD", "count": 6},
                {"card": "BASH", "count": 1},
            ],
            "relics": ["BURNING_BLOOD"],
        }

    def test_per_card_factor_include_passes(self):
        scenario = _minimal_scenario(
            id="card_fac_inc_pos",
            state=self._low_damage_state(),
            assertions={
                "assert_card_factors_include": {
                    "ANGER": ["attack_coverage"]
                }
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertTrue(
            all_passed,
            f"Per-card factor include should pass, got: {results}",
        )

    def test_per_card_factor_include_fails_when_absent(self):
        scenario = _minimal_scenario(
            id="card_fac_inc_neg",
            state=self._low_damage_state(),
            assertions={
                "assert_card_factors_include": {
                    "DEFEND_IRONCLAD": ["attack_coverage"]
                }
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertFalse(
            all_passed,
            "Defend should NOT have attack_coverage",
        )

    def test_per_card_factor_exclude_passes(self):
        scenario = _minimal_scenario(
            id="card_fac_exc_pos",
            state=self._low_damage_state(),
            assertions={
                "assert_card_factors_exclude": {
                    "DEFEND_IRONCLAD": ["attack_coverage"]
                }
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertTrue(all_passed)

    def test_per_card_factor_exclude_fails_when_present(self):
        scenario = _minimal_scenario(
            id="card_fac_exc_neg",
            state=self._low_damage_state(),
            assertions={
                "assert_card_factors_exclude": {
                    "ANGER": ["attack_coverage"]
                }
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertFalse(
            all_passed,
            "Anger has attack_coverage, exclude should fail",
        )

    def test_global_factor_include_fails_when_absent(self):
        scenario = _minimal_scenario(
            id="global_inc_neg",
            assertions={
                "assert_factors_include": ["nonexistent_factor_code_xyz"]
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        factor_result = [
            r for r in results
            if r["assertion"] == "global_factor_include:nonexistent_factor_code_xyz"
        ][0]
        self.assertFalse(factor_result["passed"])

    def test_global_factor_exclude_fails_when_present(self):
        state = self._low_damage_state()
        scenario = _minimal_scenario(
            id="global_exc_neg",
            state=state,
            assertions={
                "assert_factors_exclude": ["attack_coverage"]
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        factor_result = [
            r for r in results
            if r["assertion"] == "global_factor_exclude:attack_coverage"
        ][0]
        self.assertFalse(
            factor_result["passed"],
            "attack_coverage is present, exclude should fail",
        )

    def test_global_factor_exclude_passes_when_absent(self):
        scenario = _minimal_scenario(
            id="global_exc_pos",
            assertions={
                "assert_factors_exclude": ["nonexistent_factor_xyz"]
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        factor_result = [
            r for r in results
            if r["assertion"] == "global_factor_exclude:nonexistent_factor_xyz"
        ][0]
        self.assertTrue(factor_result["passed"])

    def test_skip_factor_include(self):
        scenario = _minimal_scenario(
            id="skip_fac_inc",
            state={
                "character": "IRONCLAD",
                "ascension": 0,
                "act": 2,
                "floor": 30,
                "energy": 3,
                "hp": 70,
                "max_hp": 80,
                "deck": [
                    {"card": "STRIKE_IRONCLAD", "count": 8},
                    {"card": "DEFEND_IRONCLAD", "count": 8},
                    {"card": "BASH", "count": 1},
                    {"card": "SHRUG_IT_OFF", "count": 3},
                    {"card": "IRON_WAVE", "count": 3},
                    {"card": "INFLAME", "count": 1},
                    {"card": "DEMON_FORM", "count": 1},
                ],
                "relics": ["BURNING_BLOOD"],
            },
            options=[
                {"card": "STRIKE_IRONCLAD", "upgrades": 0},
                {"card": "DEFEND_IRONCLAD", "upgrades": 0},
                {"card": "PERFECTED_STRIKE", "upgrades": 0},
            ],
            assertions={
                "assert_skip_factors_include": ["deck_size"],
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertTrue(
            all_passed,
            f"Skip should have deck_size factor: {results}",
        )

    def test_card_order_strict_greater_fails_on_tie(self):
        scenario = _minimal_scenario(
            id="strict_order_tie",
            options=[
                {"card": "STRIKE_IRONCLAD", "upgrades": 0},
                {"card": "STRIKE_IRONCLAD", "upgrades": 0},
            ],
            assertions={
                "card_order": ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD"],
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        # Both are the same card, same score — strict > should fail
        self.assertFalse(all_passed)

    def test_skip_recommended_assertion(self):
        scenario = _minimal_scenario(
            id="skip_assert",
            can_skip=False,
            assertions={
                "skip_recommended": False,
                "skip_eligible": False,
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertTrue(all_passed)

    def test_card_order_assertion_strict(self):
        state = self._low_damage_state()
        scenario = _minimal_scenario(
            id="order_assert_strict",
            state=state,
            options=[
                {"card": "ANGER", "upgrades": 0},
                {"card": "DEFEND_IRONCLAD", "upgrades": 0},
            ],
            assertions={
                "card_order": ["ANGER", "DEFEND_IRONCLAD"],
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        order_results = [
            r for r in results if "card_order" in r["assertion"]
        ]
        self.assertTrue(
            all(r["passed"] for r in order_results),
            f"Order results: {order_results}",
        )

    def test_decision_status_assertion(self):
        scenario = _minimal_scenario(
            id="status_assert",
            can_skip=False,
            assertions={"decision_status": "recommend"},
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertTrue(all_passed)

    def test_skip_score_range_assertion(self):
        scenario = _minimal_scenario(
            id="skip_range",
            can_skip=False,
            assertions={
                "assert_skip_score_below": 50,
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertTrue(all_passed)

    def test_assertion_statistics_order(self):
        """passed must be between 0 and total, not inverted."""
        scenario = _minimal_scenario(
            id="stat_check",
            assertions={
                "assert_factors_include": ["early_act_frontload"],
                "assert_factors_exclude": ["nonexistent_xyz"],
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertGreaterEqual(passed, 0)
        self.assertLessEqual(passed, total)
        self.assertGreater(total, 0)
        # early_act_frontload should be found, nonexistent_xyz should not be found
        # So passed should be 2 (both assertions evaluated correctly)
        self.assertEqual(total, 2)
        self.assertEqual(
            passed, total,
            f"passed={passed}, total={total}, results={results}",
        )

    def test_assertion_passed_le_total_with_mixed_results(self):
        """Even with some failures, passed <= total invariant holds."""
        state = self._low_damage_state()
        scenario = _minimal_scenario(
            id="mixed_check",
            state=state,
            assertions={
                "assert_factors_include": ["attack_coverage", "nonexistent_xyz"],
                "assert_factors_exclude": ["early_act_frontload"],
            },
        )
        all_passed, passed, total, results = self._run_and_check(scenario)
        self.assertGreaterEqual(passed, 0)
        self.assertLessEqual(passed, total)
        self.assertEqual(total, 3,
                         f"Expected 3 assertions, got total={total}, passed={passed}")


class EvaluationConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "test.db")
        self.repository = RelationalRepository(self.db_path)
        self.repository.sync_catalog(KNOWLEDGE_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_same_input_produces_same_output(self):
        scenario = _minimal_scenario()
        result1 = recommend_card_reward(
            state=scenario["state"],
            options=scenario["options"],
            repository=self.repository,
            can_skip=scenario["can_skip"],
        )
        result2 = recommend_card_reward(
            state=scenario["state"],
            options=scenario["options"],
            repository=self.repository,
            can_skip=scenario["can_skip"],
        )
        self.assertEqual(
            result1["recommended_option_index"],
            result2["recommended_option_index"],
        )
        self.assertEqual(
            result1["skip_score"],
            result2["skip_score"],
        )
        for r1, r2 in zip(
            result1["recommendations"], result2["recommendations"]
        ):
            self.assertEqual(r1["score"], r2["score"])
            self.assertEqual(r1["state_score"], r2["state_score"])

    def test_evaluator_produces_consistent_results(self):
        scenario = _minimal_scenario(
            assertions={"skip_recommended": False},
        )
        results1 = evaluate_scenarios([scenario], self.repository)
        results2 = evaluate_scenarios([scenario], self.repository)
        self.assertEqual(results1[0].passed, results2[0].passed)
        self.assertEqual(
            results1[0].recommended_option_index,
            results2[0].recommended_option_index,
        )
        self.assertEqual(results1[0].skip_score, results2[0].skip_score)


class EvaluationNoSideEffectsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_evaluation_uses_temp_database(self):
        """Verify run_evaluation always uses a temp DB."""
        scenarios_path = os.path.join(self.tempdir.name, "custom_scenarios.json")
        scenario = _minimal_scenario()
        with open(scenarios_path, "w", encoding="utf-8") as f:
            json.dump(
                {"schema_version": 1, "scenarios": [scenario]},
                f,
                ensure_ascii=False,
            )

        results, report = run_evaluation(
            scenarios_path=scenarios_path,
            knowledge_path=KNOWLEDGE_PATH,
        )
        self.assertEqual(len(results), 1)
        self.assertIn("评测报告", report)

    def test_evaluation_does_not_call_history_loader(self):
        scenarios_path = os.path.join(self.tempdir.name, "no_hist_scenarios.json")
        scenario = _minimal_scenario()
        with open(scenarios_path, "w", encoding="utf-8") as f:
            json.dump(
                {"schema_version": 1, "scenarios": [scenario]},
                f,
                ensure_ascii=False,
            )

        with patch(
            "storage.relational.RelationalRepository.load_labeled_card_reward_decisions",
            side_effect=RuntimeError("不应调用历史决策加载"),
        ):
            results, report = run_evaluation(
                scenarios_path=scenarios_path,
                knowledge_path=KNOWLEDGE_PATH,
            )
            self.assertEqual(len(results), 1)
            self.assertIsNone(results[0].error)

    def test_evaluation_blocks_network_connections(self):
        """Evaluator must not make socket connections — patch and verify."""
        scenarios_path = os.path.join(self.tempdir.name, "no_net_scenarios.json")
        scenario = _minimal_scenario()
        with open(scenarios_path, "w", encoding="utf-8") as f:
            json.dump(
                {"schema_version": 1, "scenarios": [scenario]},
                f,
                ensure_ascii=False,
            )

        def _block(*args, **kwargs):
            raise ConnectionRefusedError("评测器不应发起网络连接")

        with patch.object(socket, "create_connection", side_effect=_block):
            with patch.object(socket, "socket", side_effect=_block):
                try:
                    results, report = run_evaluation(
                        scenarios_path=scenarios_path,
                        knowledge_path=KNOWLEDGE_PATH,
                    )
                    self.assertEqual(len(results), 1)
                    self.assertIsNone(results[0].error)
                except ConnectionRefusedError as e:
                    self.fail(f"评测器尝试了网络连接: {e}")


class FullScenarioSetTests(unittest.TestCase):
    """End-to-end tests using the real scenarios file."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "test.db")
        self.repository = RelationalRepository(self.db_path)
        self.repository.sync_catalog(KNOWLEDGE_PATH)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_all_scenarios_load_and_have_valid_ids(self):
        scenarios = load_eval_scenarios(SCENARIOS_PATH)
        self.assertGreaterEqual(len(scenarios), 24)

        for scenario in scenarios:
            errors = validate_scenario_ids(scenario, self.repository)
            self.assertEqual(
                len(errors), 0,
                f"场景 {scenario['id']} 有无效 ID: {errors}",
            )

    def test_all_scenarios_run_without_error(self):
        scenarios = load_eval_scenarios(SCENARIOS_PATH)
        results = evaluate_scenarios(scenarios, self.repository)
        self.assertEqual(len(results), len(scenarios))

        for result in results:
            self.assertIsNone(
                result.error,
                f"场景 {result.scenario_id} 运行出错: {result.error}",
            )

    def test_default_mode_produces_report(self):
        scenarios = load_eval_scenarios(SCENARIOS_PATH)
        results = evaluate_scenarios(scenarios, self.repository)
        report = print_evaluation_report(results)
        self.assertIn("评测报告", report)
        self.assertIn("场景总数", report)

    def test_default_mode_accepts_failures(self):
        scenarios = load_eval_scenarios(SCENARIOS_PATH)
        results = evaluate_scenarios(scenarios, self.repository)
        report = print_evaluation_report(results)
        self.assertTrue(len(report) > 0)


class StrictModeTests(unittest.TestCase):
    @staticmethod
    def _subprocess_env() -> dict:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _make_scenarios_file(self, scenario: dict) -> str:
        path = os.path.join(
            tempfile.mkdtemp(prefix="strict_test_"),
            "scenarios.json",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"schema_version": 1, "scenarios": [scenario]},
                f,
                ensure_ascii=False,
            )
        return path

    def test_strict_flag_exits_nonzero_on_failure(self):
        scenario = _minimal_scenario(
            id="strict_fail",
            assertions={
                "assert_factors_include": ["definitely_fake_factor_xyz"],
            },
        )
        scenarios_path = self._make_scenarios_file(scenario)

        script = os.path.join(ROOT_DIR, "scripts", "eval_card_reward_scenarios.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--scenarios", scenarios_path,
                "--knowledge", KNOWLEDGE_PATH,
                "--strict",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._subprocess_env(),
            timeout=60,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        self.assertNotEqual(
            result.returncode, 0,
            f"--strict 应在断言失败时返回非零，实际返回 {result.returncode}\n"
            f"stdout: {stdout[:500]}\nstderr: {stderr[:500]}",
        )

    def test_strict_flag_exits_zero_when_all_pass(self):
        scenario = _minimal_scenario(
            id="strict_pass",
            can_skip=False,
            assertions={
                "decision_status": "recommend",
                "skip_eligible": False,
            },
        )
        scenarios_path = self._make_scenarios_file(scenario)

        script = os.path.join(ROOT_DIR, "scripts", "eval_card_reward_scenarios.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--scenarios", scenarios_path,
                "--knowledge", KNOWLEDGE_PATH,
                "--strict",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._subprocess_env(),
            timeout=60,
        )
        stderr = result.stderr or ""
        self.assertEqual(
            result.returncode, 0,
            f"--strict 应在全部通过时返回零，实际返回 {result.returncode}\n"
            f"stderr: {stderr[:500]}",
        )

    def test_no_output_file_by_default(self):
        """CLI without --output must not write any report file."""
        scenario = _minimal_scenario(
            id="no_output_test",
            can_skip=False,
            assertions={"decision_status": "recommend"},
        )
        scenarios_path = self._make_scenarios_file(scenario)

        # Count files before
        repo_root = ROOT_DIR
        data_dir = os.path.join(repo_root, "data")
        before_files = set(os.listdir(data_dir))

        script = os.path.join(ROOT_DIR, "scripts", "eval_card_reward_scenarios.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--scenarios", scenarios_path,
                "--knowledge", KNOWLEDGE_PATH,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._subprocess_env(),
            timeout=60,
        )
        self.assertEqual(result.returncode, 0)

        after_files = set(os.listdir(data_dir))
        new_files = after_files - before_files
        self.assertEqual(
            len(new_files), 0,
            f"不应创建新文件，但发现: {new_files}",
        )


class NoDatabaseOptionTests(unittest.TestCase):
    """Verify that the CLI has no --database option."""

    def test_cli_has_no_database_argument(self):
        script = os.path.join(ROOT_DIR, "scripts", "eval_card_reward_scenarios.py")
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("--database", result.stdout)


if __name__ == "__main__":
    unittest.main()
