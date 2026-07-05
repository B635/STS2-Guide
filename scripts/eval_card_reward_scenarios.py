"""P0 固定选牌场景评测入口。

使用仅包含固定场景的离线评测器，检查推荐结果、动态跳过和关键评分因子。
不读取历史选择、用户行为、AppData 当前局、本机 tier 文件或网络。
始终使用临时 SQLite，不接触真实数据库。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from advisor.evaluation import run_evaluation
from config import KNOWLEDGE_FILE


DEFAULT_SCENARIOS = os.path.join(ROOT_DIR, "data", "advisor_eval_scenarios.json")


def main() -> None:
    # Ensure stdout can handle the full Unicode report on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="P0 固定选牌场景评测 —— 离线检查推荐结果与关键因子。"
    )
    parser.add_argument(
        "--scenarios",
        default=DEFAULT_SCENARIOS,
        help=f"场景文件路径（默认: {DEFAULT_SCENARIOS}）",
    )
    parser.add_argument(
        "--knowledge",
        default=KNOWLEDGE_FILE,
        help=f"知识库文件路径（默认: {KNOWLEDGE_FILE}）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="可选：将评测结果 JSON 写入指定路径。不传则不写文件。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="有任何场景断言失败时以非零状态退出",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="以 JSON 格式输出报告",
    )
    args = parser.parse_args()

    print(f"场景文件: {args.scenarios}")
    print(f"知识库:   {args.knowledge}")
    print(f"数据库:   (临时 SQLite，评测后自动删除)")
    print()

    try:
        scenario_results, report = run_evaluation(
            scenarios_path=args.scenarios,
            knowledge_path=args.knowledge,
        )
    except Exception as exc:
        print(f"评测运行失败: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.json_output:
        print(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scenarios_path": args.scenarios,
                    "knowledge_path": args.knowledge,
                    "summary": {
                        "total": len(scenario_results),
                        "passed": sum(1 for r in scenario_results if r.passed),
                        "failed": sum(
                            1 for r in scenario_results if not r.passed
                        ),
                        "errors": sum(
                            1 for r in scenario_results if r.error
                        ),
                    },
                    "results": [
                        {
                            "scenario_id": r.scenario_id,
                            "category": r.category,
                            "description": r.description,
                            "passed": r.passed,
                            "assertions_total": r.assertions_total,
                            "assertions_passed": r.assertions_passed,
                            "error": r.error,
                            "decision_status": r.decision_status,
                            "recommended_option_index": r.recommended_option_index,
                            "skip_score": r.skip_score,
                            "card_scores": r.card_scores,
                            "assertion_results": r.assertion_results,
                        }
                        for r in scenario_results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(report)

    # Write output file only when explicitly requested.
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scenarios_path": args.scenarios,
                    "knowledge_path": args.knowledge,
                    "summary": {
                        "total": len(scenario_results),
                        "passed": sum(1 for r in scenario_results if r.passed),
                        "failed": sum(
                            1 for r in scenario_results if not r.passed
                        ),
                        "errors": sum(
                            1 for r in scenario_results if r.error
                        ),
                    },
                    "results": [
                        {
                            "scenario_id": r.scenario_id,
                            "category": r.category,
                            "description": r.description,
                            "passed": r.passed,
                            "assertions_total": r.assertions_total,
                            "assertions_passed": r.assertions_passed,
                            "error": r.error,
                            "decision_status": r.decision_status,
                            "recommended_option_index": r.recommended_option_index,
                            "skip_score": r.skip_score,
                            "card_scores": r.card_scores,
                            "assertion_results": r.assertion_results,
                        }
                        for r in scenario_results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n评测结果已写入: {args.output}")

    # Strict mode exit
    if args.strict:
        failed = sum(1 for r in scenario_results if not r.passed)
        if failed > 0:
            print(
                f"\n--strict: {failed} 个场景断言失败，以非零状态退出。",
                file=sys.stderr,
            )
            sys.exit(1)

    print("\n评测完成。")


if __name__ == "__main__":
    main()
