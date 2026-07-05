"""Offline evaluation harness for fixed card-reward scenarios.

Loads a curated scenario set, validates every ID against the live catalog,
runs the production ``recommend_card_reward()``, and checks assertions.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from advisor.card_reward import recommend_card_reward
from storage.relational import RelationalRepository

SUPPORTED_SCHEMA_VERSION = 1
REQUIRED_SCENARIO_FIELDS = frozenset(
    {"id", "category", "category_label", "description", "rationale",
     "state", "options", "can_skip", "assertions"}
)
REQUIRED_STATE_FIELDS = frozenset({"character"})
REQUIRED_OPTION_FIELDS = frozenset({"card"})

KNOWN_ASSERTION_KEYS = frozenset({
    "assert_factors_include",
    "assert_factors_exclude",
    "assert_card_factors_include",
    "assert_card_factors_exclude",
    "assert_skip_factors_include",
    "assert_skip_factors_exclude",
    "recommended_card",
    "recommended_card_id",
    "skip_recommended",
    "decision_status",
    "skip_eligible",
    "assert_skip_score_above",
    "assert_skip_score_below",
    "assert_card_score_above",
    "assert_card_score_below",
    "card_order",
    "assert_route_elite_ratio",
    "assert_route_boss_ratio_gt",
})

# Assertion keys whose value must be a non-empty list of non-empty strings.
_LIST_FACTOR_KEYS = frozenset({
    "assert_factors_include",
    "assert_factors_exclude",
    "assert_skip_factors_include",
    "assert_skip_factors_exclude",
})

# Assertion keys whose value must be dict[str, non-empty-list[str]].
_CARD_FACTOR_DICT_KEYS = frozenset({
    "assert_card_factors_include",
    "assert_card_factors_exclude",
})

# Assertion keys whose value must be dict[str, int|float].
_CARD_SCORE_DICT_KEYS = frozenset({
    "assert_card_score_above",
    "assert_card_score_below",
})

# Scalar assertion type map (top-level value type).
_SCALAR_TYPE_MAP = {
    "recommended_card": (str, type(None)),
    "recommended_card_id": (str,),
    "skip_recommended": (bool,),
    "decision_status": (str,),
    "skip_eligible": (bool,),
    "assert_skip_score_above": (int, float),
    "assert_skip_score_below": (int, float),
    "assert_route_elite_ratio": (int, float),
    "assert_route_boss_ratio_gt": (int, float),
}


@dataclass
class ValidationError:
    scenario_id: str
    message: str


@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    description: str
    passed: bool
    assertions_total: int
    assertions_passed: int
    assertion_results: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: Optional[Dict] = None
    error: Optional[str] = None
    skip_score: Optional[float] = None
    recommended_option_index: Optional[int] = None
    decision_status: Optional[str] = None
    card_scores: Dict[str, float] = field(default_factory=dict)
    factor_codes_all: List[str] = field(default_factory=list)
    skip_factor_codes: List[str] = field(default_factory=list)


def _scenario_error(scenario_id: str, message: str) -> ValidationError:
    return ValidationError(scenario_id=scenario_id, message=message)


def validate_scenario_structure(
    scenario: Dict,
    index: int,
) -> List[ValidationError]:
    """Check required fields, types, and assertion validity."""
    errors: List[ValidationError] = []
    sid = scenario.get("id") or f"<index {index}>"

    for field in REQUIRED_SCENARIO_FIELDS:
        if field not in scenario:
            errors.append(
                _scenario_error(sid, f"缺少必填字段 {field!r}")
            )

    if not isinstance(scenario.get("state"), dict):
        errors.append(_scenario_error(sid, "state 必须是对象"))
    else:
        state = scenario["state"]
        for field in REQUIRED_STATE_FIELDS:
            if not state.get(field):
                errors.append(
                    _scenario_error(sid, f"state 缺少必填字段 {field!r}")
                )

    if not isinstance(scenario.get("options"), list):
        errors.append(_scenario_error(sid, "options 必须是数组"))
    elif len(scenario["options"]) == 0:
        errors.append(_scenario_error(sid, "options 至少需要一个候选"))
    else:
        for idx, opt in enumerate(scenario["options"]):
            if not isinstance(opt, dict):
                errors.append(
                    _scenario_error(sid, f"options[{idx}] 必须是对象")
                )
                continue
            for field in REQUIRED_OPTION_FIELDS:
                if field not in opt:
                    errors.append(
                        _scenario_error(
                            sid,
                            f"options[{idx}] 缺少必填字段 {field!r}",
                        )
                    )

    # --- Assertion validation ---
    assertions = scenario.get("assertions")
    if not isinstance(assertions, dict):
        errors.append(_scenario_error(sid, "assertions 必须是对象"))
        return errors

    if not assertions:
        errors.append(_scenario_error(sid, "assertions 不能为空"))
        return errors

    has_executable = False

    for key, value in assertions.items():
        if key not in KNOWN_ASSERTION_KEYS:
            errors.append(
                _scenario_error(sid, f"未知断言键 {key!r}")
            )
            continue

        # --- List-of-strings factor assertions ---
        if key in _LIST_FACTOR_KEYS:
            if not isinstance(value, list):
                errors.append(
                    _scenario_error(
                        sid,
                        f"断言 {key!r} 必须是列表，"
                        f"实际 {type(value).__name__}",
                    )
                )
                continue
            if len(value) == 0:
                errors.append(
                    _scenario_error(sid, f"断言 {key!r} 列表不能为空")
                )
                continue
            for i, item in enumerate(value):
                if not isinstance(item, str) or not item.strip():
                    errors.append(
                        _scenario_error(
                            sid,
                            f"断言 {key!r}[{i}] 必须是非空字符串",
                        )
                    )
            has_executable = True
            continue

        # --- card_order (list of strings, at least 2) ---
        if key == "card_order":
            if not isinstance(value, list):
                errors.append(
                    _scenario_error(
                        sid,
                        f"card_order 必须是列表，实际 {type(value).__name__}",
                    )
                )
                continue
            if len(value) < 2:
                errors.append(
                    _scenario_error(sid, "card_order 至少需要 2 个卡牌 ID")
                )
                continue
            for i, item in enumerate(value):
                if not isinstance(item, str) or not item.strip():
                    errors.append(
                        _scenario_error(
                            sid,
                            f"card_order[{i}] 必须是非空字符串",
                        )
                    )
            has_executable = True
            continue

        # --- Card-factor dict assertions (dict[str, list[str]]) ---
        if key in _CARD_FACTOR_DICT_KEYS:
            if not isinstance(value, dict):
                errors.append(
                    _scenario_error(
                        sid,
                        f"断言 {key!r} 必须是字典，实际 {type(value).__name__}",
                    )
                )
                continue
            if len(value) == 0:
                errors.append(
                    _scenario_error(sid, f"断言 {key!r} 字典不能为空")
                )
                continue
            for card_id, codes in value.items():
                if not isinstance(card_id, str) or not card_id.strip():
                    errors.append(
                        _scenario_error(
                            sid,
                            f"断言 {key!r} 的键必须是非空字符串",
                        )
                    )
                if not isinstance(codes, list):
                    errors.append(
                        _scenario_error(
                            sid,
                            f"断言 {key!r}['{card_id}'] 必须是列表，"
                            f"实际 {type(codes).__name__}",
                        )
                    )
                elif len(codes) == 0:
                    errors.append(
                        _scenario_error(
                            sid,
                            f"断言 {key!r}['{card_id}'] 列表不能为空",
                        )
                    )
                else:
                    for i, item in enumerate(codes):
                        if not isinstance(item, str) or not item.strip():
                            errors.append(
                                _scenario_error(
                                    sid,
                                    f"断言 {key!r}['{card_id}'][{i}] 必须是非空字符串",
                                )
                            )
            has_executable = True
            continue

        # --- Card-score dict assertions (dict[str, int|float]) ---
        if key in _CARD_SCORE_DICT_KEYS:
            if not isinstance(value, dict):
                errors.append(
                    _scenario_error(
                        sid,
                        f"断言 {key!r} 必须是字典，实际 {type(value).__name__}",
                    )
                )
                continue
            if len(value) == 0:
                errors.append(
                    _scenario_error(sid, f"断言 {key!r} 字典不能为空")
                )
                continue
            for card_id, threshold in value.items():
                if not isinstance(card_id, str) or not card_id.strip():
                    errors.append(
                        _scenario_error(
                            sid,
                            f"断言 {key!r} 的键必须是非空字符串",
                        )
                    )
                if not isinstance(threshold, (int, float)):
                    errors.append(
                        _scenario_error(
                            sid,
                            f"断言 {key!r}['{card_id}'] 必须是数值，"
                            f"实际 {type(threshold).__name__}",
                        )
                    )
            has_executable = True
            continue

        # --- Scalar assertions ---
        expected_types = _SCALAR_TYPE_MAP.get(key)
        if expected_types and not isinstance(value, expected_types):
            type_names = " | ".join(
                t.__name__ for t in expected_types
            )
            errors.append(
                _scenario_error(
                    sid,
                    f"断言 {key!r} 类型错误：期望 {type_names}，"
                    f"实际 {type(value).__name__}",
                )
            )
            continue
        # Any scalar assertion with a valid key is executable.
        has_executable = True

    if not has_executable:
        errors.append(
            _scenario_error(sid, "assertions 没有可执行的断言键")
        )

    return errors


def validate_scenario_ids(
    scenario: Dict,
    repository: RelationalRepository,
) -> List[ValidationError]:
    """Verify that character, card, and relic IDs exist in the catalog."""
    errors: List[ValidationError] = []
    sid = scenario["id"]
    state = scenario.get("state") or {}

    character_id = state.get("character")
    if character_id:
        entity = repository.find_entity("characters", str(character_id))
        if entity is None:
            errors.append(
                _scenario_error(
                    sid,
                    f"角色 ID {character_id!r} 在 catalog 中未找到",
                )
            )

    for idx, opt in enumerate(scenario.get("options") or []):
        card_id = opt.get("card")
        if not card_id:
            continue
        cid = str(card_id)
        card = repository.find_card(cid)
        if card is None:
            errors.append(
                _scenario_error(
                    sid,
                    f"options[{idx}] 卡牌 ID {card_id!r} 在 catalog 中未找到",
                )
            )

    for relic_id in state.get("relics") or []:
        if repository.find_relic(str(relic_id)) is None:
            errors.append(
                _scenario_error(
                    sid,
                    f"遗物 ID {relic_id!r} 在 catalog 中未找到",
                )
            )

    for entry in state.get("deck") or []:
        deck_card_id = entry.get("card")
        if deck_card_id and repository.find_card(str(deck_card_id)) is None:
            errors.append(
                _scenario_error(
                    sid,
                    f"牌组中卡牌 ID {deck_card_id!r} 在 catalog 中未找到",
                )
            )

    if isinstance(state.get("relic_states"), list):
        for relic_state in state["relic_states"]:
            rid = relic_state.get("relic")
            if rid and repository.find_relic(str(rid)) is None:
                errors.append(
                    _scenario_error(
                        sid,
                        f"relic_states 中遗物 ID {rid!r} 在 catalog 中未找到",
                    )
                )

    # Validate that card IDs referenced in assertions exist in scenario options.
    option_card_ids: set = set()
    for opt in scenario.get("options") or []:
        cid = opt.get("card")
        if cid:
            option_card_ids.add(str(cid).upper())

    assertions = scenario.get("assertions") or {}

    for key in ("assert_card_factors_include", "assert_card_factors_exclude"):
        for card_id in (assertions.get(key) or {}):
            if str(card_id).upper() not in option_card_ids:
                errors.append(
                    _scenario_error(
                        sid,
                        f"断言 {key} 引用的卡牌 {card_id!r} 不在场景 options 中",
                    )
                )

    for key in ("assert_card_score_above", "assert_card_score_below"):
        for card_id in (assertions.get(key) or {}):
            if str(card_id).upper() not in option_card_ids:
                errors.append(
                    _scenario_error(
                        sid,
                        f"断言 {key} 引用的卡牌 {card_id!r} 不在场景 options 中",
                    )
                )

    for card_id in (assertions.get("card_order") or []):
        if str(card_id).upper() not in option_card_ids:
            errors.append(
                _scenario_error(
                    sid,
                    f"card_order 引用的卡牌 {card_id!r} 不在场景 options 中",
                )
            )

    if assertions.get("recommended_card"):
        rec_name = str(assertions["recommended_card"]).upper()
        if rec_name not in option_card_ids:
            errors.append(
                _scenario_error(
                    sid,
                    f"recommended_card {assertions['recommended_card']!r} 不在场景 options 中",
                )
            )

    return errors


def _factor_codes_for_card(
    recommendations: List[Dict],
    card_id: str,
) -> List[str]:
    cid = str(card_id).upper()
    for rec in recommendations:
        rec_cid = str(rec.get("card_id") or rec.get("card") or "").upper()
        if rec_cid == cid:
            return [
                str(f.get("code"))
                for f in (rec.get("factors") or [])
                if f.get("code")
            ]
    return []


def _collect_factor_codes(
    recommendations: List[Dict],
) -> List[str]:
    codes: List[str] = []
    for rec in recommendations:
        for factor in rec.get("factors") or []:
            code = factor.get("code")
            if code:
                codes.append(str(code))
    return codes


def _collect_skip_factor_codes(skip_candidate: Dict) -> List[str]:
    codes: List[str] = []
    for factor in skip_candidate.get("factors") or []:
        code = factor.get("code")
        if code:
            codes.append(str(code))
    return codes


def check_assertions(
    scenario: Dict,
    result: Dict,
) -> Tuple[bool, int, int, List[Dict]]:
    """Check each assertion type against the recommendation output."""
    assertions = scenario["assertions"]
    results: List[Dict] = []
    passed = 0
    total = 0

    def record(
        key: str,
        ok: bool,
        expected: Any,
        actual: Any,
        detail: str = "",
    ) -> None:
        nonlocal passed, total
        total += 1
        if ok:
            passed += 1
        results.append(
            {
                "assertion": key,
                "passed": ok,
                "expected": expected,
                "actual": actual,
                "detail": detail,
            }
        )

    recommendations = result["recommendations"]
    skip_candidate = result["skip_candidate"]
    all_factor_codes = _collect_factor_codes(recommendations)
    skip_factor_codes_list = _collect_skip_factor_codes(skip_candidate)

    # --- Global factor inclusion assertions ---
    for code in assertions.get("assert_factors_include") or []:
        found = code in all_factor_codes or code in skip_factor_codes_list
        record(
            f"global_factor_include:{code}",
            found,
            code,
            all_factor_codes,
            "全局因子应出现在推荐或跳过因子中",
        )

    # --- Global factor exclusion assertions ---
    for code in assertions.get("assert_factors_exclude") or []:
        found = code in all_factor_codes or code in skip_factor_codes_list
        record(
            f"global_factor_exclude:{code}",
            not found,
            f"不应出现 {code}",
            all_factor_codes if found else "未出现",
            "全局因子不应出现在任何推荐中",
        )

    # --- Per-card factor include assertions ---
    for card_id, codes in (assertions.get("assert_card_factors_include") or {}).items():
        card_codes = _factor_codes_for_card(recommendations, str(card_id))
        for code in codes:
            found = code in card_codes
            record(
                f"card_factor_include:{card_id}:{code}",
                found,
                code,
                card_codes if found else f"未出现（实际: {card_codes}）",
                f"卡牌 {card_id} 应包含因子 {code}",
            )

    # --- Per-card factor exclude assertions ---
    for card_id, codes in (assertions.get("assert_card_factors_exclude") or {}).items():
        card_codes = _factor_codes_for_card(recommendations, str(card_id))
        for code in codes:
            found = code in card_codes
            record(
                f"card_factor_exclude:{card_id}:{code}",
                not found,
                f"不应出现 {code}",
                card_codes if found else "未出现",
                f"卡牌 {card_id} 不应包含因子 {code}",
            )

    # --- Skip factor assertions ---
    for code in assertions.get("assert_skip_factors_include") or []:
        found = code in skip_factor_codes_list
        record(
            f"skip_factor_include:{code}",
            found,
            code,
            skip_factor_codes_list,
            "跳过因子应包含该 code",
        )

    for code in assertions.get("assert_skip_factors_exclude") or []:
        found = code in skip_factor_codes_list
        record(
            f"skip_factor_exclude:{code}",
            not found,
            f"不应出现 {code}",
            skip_factor_codes_list if found else "未出现",
            "跳过因子不应包含该 code",
        )

    # --- Recommended card assertion ---
    expected_card_name = assertions.get("recommended_card")
    expected_card_id = assertions.get("recommended_card_id")
    if expected_card_name is not None or expected_card_id is not None:
        actual_name = result.get("recommended_option")
        actual_id = None
        for rec in recommendations:
            if rec.get("card") == actual_name:
                actual_id = rec.get("card_id")
                break
        if expected_card_name is not None:
            record(
                "recommended_card",
                actual_name == expected_card_name,
                expected_card_name,
                actual_name,
            )
        if expected_card_id is not None:
            record(
                "recommended_card_id",
                actual_id == expected_card_id,
                expected_card_id,
                actual_id,
            )

    # --- Skip recommended assertion ---
    if "skip_recommended" in assertions:
        expected = bool(assertions["skip_recommended"])
        actual = bool(result.get("skip_recommended"))
        record("skip_recommended", actual == expected, expected, actual)

    # --- Decision status assertion ---
    if "decision_status" in assertions:
        expected = str(assertions["decision_status"])
        actual = str(result.get("decision_status"))
        record("decision_status", actual == expected, expected, actual)

    # --- Skip eligible assertion ---
    if "skip_eligible" in assertions:
        expected = bool(assertions["skip_eligible"])
        actual = bool(skip_candidate.get("eligible"))
        record("skip_eligible", actual == expected, expected, actual)

    # --- Skip score range assertions ---
    skip_score = float(skip_candidate.get("score", 0))
    if "assert_skip_score_above" in assertions:
        threshold = float(assertions["assert_skip_score_above"])
        actual = skip_score
        record(
            f"skip_score > {threshold}",
            actual > threshold,
            f"> {threshold}",
            actual,
        )

    if "assert_skip_score_below" in assertions:
        threshold = float(assertions["assert_skip_score_below"])
        actual = skip_score
        record(
            f"skip_score < {threshold}",
            actual < threshold,
            f"< {threshold}",
            actual,
        )

    # --- Card score range assertions ---
    for key in ("assert_card_score_above", "assert_card_score_below"):
        card_thresholds = assertions.get(key) or {}
        for card_id, threshold in card_thresholds.items():
            threshold = float(threshold)
            found_score = None
            for rec in recommendations:
                if str(rec.get("card_id") or "").upper() == str(card_id).upper():
                    found_score = float(rec["score"])
                    break
                if str(rec.get("card") or "").upper() == str(card_id).upper():
                    found_score = float(rec["score"])
                    break
            if found_score is None:
                record(
                    f"card_score:{card_id}",
                    False,
                    f"score {key.split('_')[-1]} {threshold}",
                    "card not found in recommendations",
                )
            elif key == "assert_card_score_above":
                record(
                    f"card_score:{card_id} > {threshold}",
                    found_score > threshold,
                    f"> {threshold}",
                    found_score,
                )
            else:
                record(
                    f"card_score:{card_id} < {threshold}",
                    found_score < threshold,
                    f"< {threshold}",
                    found_score,
                )

    # --- Card relative order assertion (strict >) ---
    card_order = assertions.get("card_order") or []
    card_score_map: Dict[str, float] = {}
    for rec in recommendations:
        cid = str(rec.get("card_id") or rec.get("card") or "")
        card_score_map[cid.upper()] = float(rec["score"])
    for i in range(len(card_order) - 1):
        a_id = str(card_order[i]).upper()
        b_id = str(card_order[i + 1]).upper()
        a_score = card_score_map.get(a_id)
        b_score = card_score_map.get(b_id)
        if a_score is None:
            record(
                f"card_order:{card_order[i]}",
                False,
                f"{card_order[i]} 严格高于 {card_order[i+1]}",
                f"card {card_order[i]} 未在推荐结果中找到",
            )
        elif b_score is None:
            record(
                f"card_order:{card_order[i+1]}",
                False,
                f"{card_order[i]} 严格高于 {card_order[i+1]}",
                f"card {card_order[i+1]} 未在推荐结果中找到",
            )
        else:
            ok = a_score > b_score
            record(
                f"card_order:{card_order[i]} > {card_order[i+1]}",
                ok,
                f"{card_order[i]}({a_score}) > {card_order[i+1]}({b_score})",
                f"{'PASS' if ok else 'FAIL'}: {a_score} vs {b_score}",
            )

    # --- Route ratio assertions ---
    route = result.get("profile", {}).get("route", {})
    if "assert_route_elite_ratio" in assertions:
        expected = float(assertions["assert_route_elite_ratio"])
        actual = float(route.get("elite_path_ratio", 0))
        record(
            f"route_elite_ratio == {expected}",
            abs(actual - expected) < 0.001,
            expected,
            actual,
        )

    if "assert_route_boss_ratio_gt" in assertions:
        threshold = float(assertions["assert_route_boss_ratio_gt"])
        actual = float(route.get("boss_path_ratio", 0))
        record(
            f"route_boss_ratio > {threshold}",
            actual > threshold,
            f"> {threshold}",
            actual,
        )

    return passed == total, passed, total, results


def _build_card_score_map(result: Dict) -> Dict[str, float]:
    score_map: Dict[str, float] = {}
    for rec in result.get("recommendations") or []:
        name = str(rec.get("card") or rec.get("card_id") or "")
        score_map[name] = float(rec.get("score", 0))
    return score_map


def evaluate_scenarios(
    scenarios: List[Dict],
    repository: RelationalRepository,
) -> List[ScenarioResult]:
    """Run all scenarios through the production recommender and check assertions."""
    results: List[ScenarioResult] = []
    for scenario in scenarios:
        sid = scenario["id"]
        state = copy.deepcopy(scenario["state"])
        options = copy.deepcopy(scenario["options"])
        can_skip = bool(scenario.get("can_skip", True))

        try:
            recommendation = recommend_card_reward(
                state=state,
                options=options,
                repository=repository,
                can_skip=can_skip,
            )
        except Exception as exc:
            results.append(
                ScenarioResult(
                    scenario_id=sid,
                    category=str(scenario.get("category_label") or scenario.get("category")),
                    description=str(scenario.get("description")),
                    passed=False,
                    assertions_total=0,
                    assertions_passed=0,
                    error=f"推荐器异常: {exc}",
                )
            )
            continue

        try:
            all_passed, passed_count, total, assertion_results = check_assertions(
                scenario, recommendation
            )
        except Exception as exc:
            results.append(
                ScenarioResult(
                    scenario_id=sid,
                    category=str(scenario.get("category_label") or scenario.get("category")),
                    description=str(scenario.get("description")),
                    passed=False,
                    assertions_total=0,
                    assertions_passed=0,
                    recommendation=recommendation,
                    error=f"断言检查异常: {exc}",
                )
            )
            continue

        results.append(
            ScenarioResult(
                scenario_id=sid,
                category=str(scenario.get("category_label") or scenario.get("category")),
                description=str(scenario.get("description")),
                passed=all_passed,
                assertions_total=total,
                assertions_passed=passed_count,
                assertion_results=assertion_results,
                recommendation=recommendation,
                skip_score=float(recommendation["skip_score"]),
                recommended_option_index=int(recommendation.get("recommended_option_index", -1)),
                decision_status=str(recommendation.get("decision_status")),
                card_scores=_build_card_score_map(recommendation),
                factor_codes_all=_collect_factor_codes(recommendation["recommendations"]),
                skip_factor_codes=_collect_skip_factor_codes(recommendation["skip_candidate"]),
            )
        )

    return results


def print_evaluation_report(
    scenario_results: List[ScenarioResult],
) -> str:
    """Render a human-readable evaluation report."""
    lines: List[str] = []
    total = len(scenario_results)
    passed = sum(1 for r in scenario_results if r.passed)
    failed = total - passed
    error_count = sum(1 for r in scenario_results if r.error)
    assertion_total = sum(r.assertions_total for r in scenario_results)
    assertion_passed = sum(r.assertions_passed for r in scenario_results)

    lines.append("=" * 72)
    lines.append("  固定选牌场景评测报告")
    lines.append("=" * 72)
    lines.append(f"  场景总数: {total}")
    lines.append(f"  场景通过: {passed}")
    lines.append(f"  场景失败: {failed}")
    lines.append(f"  运行错误: {error_count}")
    lines.append(
        f"  断言总数: {assertion_total}  通过: {assertion_passed}  "
        f"通过率: {assertion_passed / max(1, assertion_total) * 100:.1f}%"
    )
    lines.append("")

    # Group by category
    categories: Dict[str, List[ScenarioResult]] = {}
    for r in scenario_results:
        cat = r.category or "未分类"
        categories.setdefault(cat, []).append(r)

    for cat, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r.passed)
        lines.append(f"-- {cat} ({cat_passed}/{len(cat_results)} 通过) --")
        for r in cat_results:
            status = "PASS" if r.passed else "FAIL"
            if r.error:
                status = "ERROR"
            lines.append(f"  [{status}] {r.scenario_id}: {r.description}")
            if r.error:
                lines.append(f"         错误: {r.error}")
            if r.decision_status:
                lines.append(
                    f"         决策状态: {r.decision_status}  "
                    f"推荐索引: {r.recommended_option_index}  "
                    f"跳过分: {r.skip_score}"
                )
            if r.card_scores:
                score_str = " | ".join(
                    f"{name}: {score:.1f}"
                    for name, score in sorted(
                        r.card_scores.items(),
                        key=lambda x: -x[1],
                    )
                )
                lines.append(f"         候选分: {score_str}")

            # Show failed assertions
            failed_assertions = [
                a for a in r.assertion_results if not a.get("passed")
            ]
            if failed_assertions:
                lines.append(
                    f"         失败断言 ({len(failed_assertions)}/{r.assertions_total}):"
                )
                for fa in failed_assertions:
                    lines.append(
                        f"           X {fa['assertion']}: "
                        f"期望={fa['expected']} 实际={fa['actual']}"
                    )
                    if fa.get("detail"):
                        lines.append(f"             {fa['detail']}")

            if not r.passed and not failed_assertions and not r.error:
                lines.append("          （无断言或全部通过）")
        lines.append("")

    lines.append("=" * 72)
    if passed == total:
        lines.append("  所有场景通过 [PASS]")
    else:
        lines.append(f"  {failed}/{total} 场景失败（基线报告，非错误）")
    lines.append("=" * 72)

    return "\n".join(lines)


def load_eval_scenarios(path: str) -> List[Dict]:
    """Load and perform structural validation on a scenario file."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"场景文件未找到: {path}")

    with open(source_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError("场景文件根元素必须是对象")

    schema_version = payload.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"不支持的 schema_version: {schema_version!r}，"
            f"期望: {SUPPORTED_SCHEMA_VERSION}"
        )

    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("scenarios 必须是数组")

    scenarios: List[Dict] = []
    seen_ids: set = set()
    for index, raw in enumerate(raw_scenarios):
        if not isinstance(raw, dict):
            raise ValueError(f"scenarios[{index}] 必须是对象")

        # Structural validation
        struct_errors = validate_scenario_structure(raw, index)
        if struct_errors:
            raise ValueError(
                "场景结构校验失败:\n"
                + "\n".join(
                    f"  - {e.scenario_id}: {e.message}" for e in struct_errors
                )
            )

        sid = raw["id"]
        if sid in seen_ids:
            raise ValueError(f"重复的场景 ID: {sid!r}")
        seen_ids.add(sid)

        scenarios.append(raw)

    return scenarios


def run_evaluation(
    scenarios_path: str,
    knowledge_path: str,
) -> Tuple[List[ScenarioResult], str]:
    """Full evaluation pipeline: load, validate, evaluate, report.

    Always uses a temporary SQLite database — never touches the real one.
    Returns (results, report_text).
    """
    scenarios = load_eval_scenarios(scenarios_path)

    temp_dir = tempfile.mkdtemp(prefix="advisor_eval_")
    db_path = os.path.join(temp_dir, "eval.db")

    try:
        repository = RelationalRepository(db_path)
        repository.sync_catalog(knowledge_path)

        # Validate IDs against catalog
        all_errors: List[ValidationError] = []
        for scenario in scenarios:
            all_errors.extend(validate_scenario_ids(scenario, repository))

        if all_errors:
            error_text = "\n".join(
                f"  - {e.scenario_id}: {e.message}" for e in all_errors
            )
            raise ValueError(f"场景 ID 校验失败:\n{error_text}")

        scenario_results = evaluate_scenarios(scenarios, repository)
        report = print_evaluation_report(scenario_results)
        return scenario_results, report

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
