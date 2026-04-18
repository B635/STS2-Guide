"""End-to-end answer quality eval.

Runs the full RAG pipeline (retrieve -> generate) for each labeled case and
scores the final answer. Two eval modes per case:

  - keyword   : answer must contain every string in `expected_keywords`.
                Cheap, no extra API call. Good for factual / numeric questions.

  - llm_judge : DeepSeek judges (问题, 参考答案, 模型答案) across three axes
                (faithful / relevant / complete, each 0-2). Covers open-ended
                and comparison questions where string match would miss.

Limitation: the judge model == the generator model (both DeepSeek), so scores
skew high (self-bias). Production would use a stronger external judge.
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    DEEPSEEK_MODEL, BM25_TOP_N, VECTOR_TOP_N_FOR_HYBRID, RRF_K,
    RERANKER_CANDIDATE_N, RETRIEVE_TOP_N,
)
from rag.embedder import load_model, load_or_compute_embeddings
from rag.knowledge import load_knowledge
from rag.retriever import retrieve, hybrid_retrieve, format_context
from rag.chat import create_client, rag_chat
from rag.router import structured_query

DEFAULT_EVAL_FILE = "./data/answer_eval.json"


def load_eval_cases(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def retrieve_context(query, docs, items, index, store, model, bm25_idx, reranker,
                     use_hybrid, use_reranker, use_router) -> List[Dict]:
    if use_router:
        routed = structured_query(query, index, items)
        if routed is not None:
            return routed[:RETRIEVE_TOP_N]

    if use_hybrid:
        candidate_n = RERANKER_CANDIDATE_N if use_reranker else RETRIEVE_TOP_N
        candidates = hybrid_retrieve(
            query, docs, store, model, bm25_idx,
            vector_n=VECTOR_TOP_N_FOR_HYBRID, bm25_n=BM25_TOP_N,
            rrf_k=RRF_K, top_n=candidate_n,
        )
    else:
        candidate_n = RERANKER_CANDIDATE_N if use_reranker else RETRIEVE_TOP_N
        candidates = retrieve(query, docs, store, model, n=candidate_n)

    if use_reranker:
        from rag.reranker import rerank
        return rerank(query, candidates, reranker, top_n=RETRIEVE_TOP_N)
    return candidates


def score_keyword(answer: str, keywords: List[str]) -> Dict:
    missing = [kw for kw in keywords if kw not in answer]
    return {"passed": len(missing) == 0, "missing": missing}


JUDGE_PROMPT = """你是一个严格的答案评分员。从三个维度打分：

- faithful (忠实度, 0-2): 模型答案是否与参考答案矛盾或编造了参考答案中没有的事实？
    0 = 有明显矛盾或编造, 1 = 部分对齐, 2 = 完全忠实
- relevant (相关性, 0-2): 是否直接回答了问题而不是答非所问？
    0 = 答非所问, 1 = 部分相关, 2 = 直接正面回答
- complete (完整性, 0-2): 参考答案中的关键事实是否都覆盖到了？
    0 = 缺失关键信息, 1 = 覆盖部分, 2 = 关键信息完整

只输出合法 JSON，格式：{"faithful":<int>,"relevant":<int>,"complete":<int>,"reason":"<一句话>"}
不要输出其他任何文字，不要用 markdown 代码块。"""


def score_llm_judge(query: str, expected: str, answer: str, client) -> Dict:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"问题: {query}\n参考答案: {expected}\n模型答案: {answer}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Strip ```json fences if the judge ignored instructions
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {"error": "no_json", "raw": raw[:200]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"error": "bad_json", "raw": raw[:200]}


def evaluate(cases: List[Dict], pipeline_kwargs: Dict) -> Dict:
    docs, items, index = load_knowledge()
    model = load_model()
    store = load_or_compute_embeddings(docs, model)
    client = create_client()

    bm25_idx = None
    if pipeline_kwargs["use_hybrid"]:
        from rag.bm25 import build_bm25_index
        bm25_idx = build_bm25_index(docs)

    reranker = None
    if pipeline_kwargs["use_reranker"]:
        from rag.reranker import load_reranker
        reranker = load_reranker()

    records = []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        print(f"\n[{i}/{len(cases)}] {query}")

        results = retrieve_context(
            query, docs, items, index, store, model, bm25_idx, reranker,
            **pipeline_kwargs,
        )
        context = format_context(results)
        answer = rag_chat(query, context, history=[], client=client)
        print(f"  答案: {answer[:100]}{'...' if len(answer) > 100 else ''}")

        if case["eval_type"] == "keyword":
            score = score_keyword(answer, case["expected_keywords"])
            print(f"  [keyword] {'✓' if score['passed'] else '✗'} missing={score['missing']}")
        else:
            score = score_llm_judge(query, case["expected_answer"], answer, client)
            if "error" in score:
                print(f"  [llm_judge] 解析失败: {score['error']}")
            else:
                print(f"  [llm_judge] faithful={score['faithful']} relevant={score['relevant']} complete={score['complete']}")

        records.append({"case": case, "answer": answer, "score": score})

    return {"records": records}


def summarize(records: List[Dict]) -> None:
    keyword_total = keyword_passed = 0
    judge_total = 0
    sums = {"faithful": 0, "relevant": 0, "complete": 0}
    judge_errors = 0

    for r in records:
        eval_type = r["case"]["eval_type"]
        score = r["score"]
        if eval_type == "keyword":
            keyword_total += 1
            if score["passed"]:
                keyword_passed += 1
        else:
            if "error" in score:
                judge_errors += 1
                continue
            judge_total += 1
            for axis in sums:
                sums[axis] += score[axis]

    print("\n" + "=" * 56)
    print("端到端答案评测结果")
    print("-" * 56)
    if keyword_total:
        print(f"关键词匹配  : {keyword_passed}/{keyword_total} ({keyword_passed/keyword_total:.2%})")
    if judge_total:
        print(f"LLM-Judge   : {judge_total} cases (裁判模型 = 生成模型 → 有自评偏差)")
        for axis, total in sums.items():
            print(f"  avg {axis:<9}: {total/judge_total:.2f} / 2.00")
    if judge_errors:
        print(f"  裁判解析失败: {judge_errors}")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAG answer eval.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--hybrid", action="store_true", help="Hybrid (BM25 + 向量 RRF)")
    parser.add_argument("--reranker", action="store_true", help="两阶段精排")
    parser.add_argument("--router", action="store_true", help="结构化查询路由")
    parser.add_argument("--output", help="可选：把完整结果写入 JSON 文件")
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_file)
    pipeline = {"use_hybrid": args.hybrid, "use_reranker": args.reranker, "use_router": args.router}

    mode = " + ".join([k for k, v in pipeline.items() if v]) or "baseline"
    print(f"Pipeline: {mode} | 样本数: {len(cases)}")

    result = evaluate(cases, pipeline)
    summarize(result["records"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已写入 {args.output}")


if __name__ == "__main__":
    main()
