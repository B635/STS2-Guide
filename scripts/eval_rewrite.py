"""Evaluate the multi-turn query rewriter.

Each case provides (history, follow-up query) and a set of keywords the
rewritten query should contain (`expected_keywords`) plus, optionally,
keywords it must NOT contain (`forbidden_keywords` — used for topic switches
and hallucination traps where the rewriter shouldn't smuggle in stale
entities).

Prints per-case pass/fail plus aggregate pass rate.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag.chat import create_client
from rag.knowledge import load_knowledge
from rag.query_rewriter import rewrite_query

DEFAULT_EVAL_FILE = "./data/multiturn_eval.json"


def load_cases(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def check_case(rewritten: str, case: Dict) -> Dict:
    missing = [kw for kw in case.get("expected_keywords", []) if kw not in rewritten]
    leaked = [kw for kw in case.get("forbidden_keywords", []) if kw in rewritten]
    return {"passed": not missing and not leaked, "missing": missing, "leaked": leaked}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-turn query rewriter.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    args = parser.parse_args()

    _, _, index = load_knowledge()
    client = create_client()
    cases = load_cases(args.eval_file)

    passed = 0
    for i, case in enumerate(cases, 1):
        history = case["history"]
        query = case["query"]
        rewritten = rewrite_query(query, history, client, index)
        result = check_case(rewritten, case)

        mark = "✓" if result["passed"] else "✗"
        print(f"\n[{i}/{len(cases)}] [{case['tag']}] {mark}")
        print(f"  原 query:   {query}")
        print(f"  改写后:     {rewritten}")
        if result["missing"]:
            print(f"  缺少关键词: {result['missing']}")
        if result["leaked"]:
            print(f"  泄漏关键词: {result['leaked']}")
        if result["passed"]:
            passed += 1

    print("\n" + "=" * 48)
    print(f"通过率: {passed}/{len(cases)} ({passed/len(cases):.2%})")
    print("=" * 48)


if __name__ == "__main__":
    main()
