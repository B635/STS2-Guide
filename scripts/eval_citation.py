"""Rule-based citation quality evaluation (no LLM-as-Judge).

Measures three things over the retrieval_eval.json set:

1. Citation Rate       — fraction of "substantive" sentences that end with [n] or [?].
                          Catches a model that silently drops the citation rule.
2. Source Validity     — fraction of [n] markers pointing to an ID that the LLM
                          actually saw (context had [1]..[k]; any [j] with j>k is
                          a hallucinated citation).
3. Number Grounding    — for each cited sentence, take all standalone integers in
                          the sentence text. A citation is "grounded" when every
                          such number also appears in at least one cited source.
                          Crude but zero-LLM — numbers are the most frequently
                          hallucinated tokens in this domain.
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import BM25_TOP_N, VECTOR_TOP_N_FOR_HYBRID, RRF_K, RETRIEVE_TOP_N
from rag.knowledge import load_knowledge
from rag.embedder import load_model, load_or_compute_embeddings
from rag.chat import create_client, rag_chat
from rag.retriever import hybrid_retrieve, format_context
from rag.router import structured_query
from rag.bm25 import build_bm25_index
from rag.hyde import generate_hypothetical
from rag.reranker import load_reranker, rerank


DEFAULT_EVAL_FILE = "./data/retrieval_eval.json"

CITATION_RE = re.compile(r"\[(\d+|\?)\]")
NUMBER_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
    # If LLM put the citation after the sentence-ending period ("...格挡。[1]"),
    # the "[1]" becomes its own fragment. Glue it back to the previous sentence.
    merged: List[str] = []
    for p in parts:
        if p.startswith("[") and merged:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def is_substantive(sentence: str) -> bool:
    # Skip pure Markdown scaffolding (table separators, list bullets alone, etc.).
    stripped = sentence.strip("*- |：:#").strip()
    if not stripped:
        return False
    if re.fullmatch(r"[\s\-\|=:]+", stripped):
        return False
    return len(stripped) >= 4


def numbers_in(text: str) -> List[str]:
    return NUMBER_RE.findall(text)


def evaluate_answer(answer: str, sources: List[str]) -> Dict:
    sentences = split_sentences(answer)
    total_substantive = 0
    cited_sentences = 0
    total_citations = 0
    invalid_citations = 0
    grounded_sentences = 0
    ungroundable_number_sentences = 0

    max_valid_id = len(sources)

    for sent in sentences:
        if not is_substantive(sent):
            continue
        total_substantive += 1

        cites = CITATION_RE.findall(sent)
        if not cites:
            continue
        cited_sentences += 1

        valid_cited_texts = []
        for c in cites:
            total_citations += 1
            if c == "?":
                continue
            cid = int(c)
            if cid < 1 or cid > max_valid_id:
                invalid_citations += 1
            else:
                valid_cited_texts.append(sources[cid - 1])

        # Number grounding: every number in the sentence must appear in at least
        # one cited source. Skip if no numbers or no valid citations.
        nums = numbers_in(sent)
        nums = [n for n in nums if n not in {c for c in cites if c.isdigit()}]
        if nums and valid_cited_texts:
            merged = " ".join(valid_cited_texts)
            if all(n in merged for n in nums):
                grounded_sentences += 1
            else:
                ungroundable_number_sentences += 1

    return {
        "substantive_sentences": total_substantive,
        "cited_sentences": cited_sentences,
        "citation_tokens": total_citations,
        "invalid_citations": invalid_citations,
        "grounded_sentences": grounded_sentences,
        "ungroundable_number_sentences": ungroundable_number_sentences,
    }


def run(cases: List[Dict], top_k: int, limit: int, verbose: bool) -> None:
    docs, items, index = load_knowledge()
    model = load_model()
    emb = load_or_compute_embeddings(docs, model)
    client = create_client()
    bm25 = build_bm25_index(docs)
    reranker = load_reranker()

    agg = {
        "substantive_sentences": 0,
        "cited_sentences": 0,
        "citation_tokens": 0,
        "invalid_citations": 0,
        "grounded_sentences": 0,
        "ungroundable_number_sentences": 0,
    }
    answered = 0
    selected = cases[:limit] if limit else cases

    for case in selected:
        q = case["query"]
        routed = structured_query(q, index, items)
        if routed is not None:
            results = routed[:top_k]
        else:
            vq = generate_hypothetical(q, client)
            cands = hybrid_retrieve(
                q, docs, emb, model, bm25,
                vector_n=VECTOR_TOP_N_FOR_HYBRID, bm25_n=BM25_TOP_N,
                rrf_k=RRF_K, top_n=20, vector_query=vq,
            )
            results = rerank(q, cands, reranker, top_n=top_k)

        ctx = format_context(results)
        answer = rag_chat(q, ctx, [], client)
        answered += 1

        source_texts = [r["text"] for r in results]
        stats = evaluate_answer(answer, source_texts)
        for k, v in stats.items():
            agg[k] += v

        if verbose:
            print(f"\nQ: {q}")
            print(f"A: {answer}")
            print(f"  -> substantive={stats['substantive_sentences']} "
                  f"cited={stats['cited_sentences']} "
                  f"invalid={stats['invalid_citations']} "
                  f"grounded={stats['grounded_sentences']}/"
                  f"{stats['grounded_sentences'] + stats['ungroundable_number_sentences']} "
                  f"(num-bearing)")

    print("\n" + "=" * 50)
    print(f"评测完成，共 {answered} 条样本")
    print("-" * 50)
    sub = max(agg["substantive_sentences"], 1)
    tok = max(agg["citation_tokens"], 1)
    num_sent = max(agg["grounded_sentences"] + agg["ungroundable_number_sentences"], 1)

    print(f"Citation Rate      (句级引用覆盖): {agg['cited_sentences']}/{sub} = "
          f"{agg['cited_sentences'] / sub:.2%}")
    print(f"Source Validity    (引用编号有效): "
          f"{(tok - agg['invalid_citations'])}/{tok} = "
          f"{(tok - agg['invalid_citations']) / tok:.2%}")
    print(f"Number Grounding   (数字可追溯率): "
          f"{agg['grounded_sentences']}/{num_sent} = "
          f"{agg['grounded_sentences'] / num_sent:.2%}  "
          f"（仅统计含数字的被引用句）")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate citation quality (rule-based).")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=RETRIEVE_TOP_N)
    parser.add_argument("--limit", type=int, default=0, help="Only run first N cases (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.eval_file, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    run(cases, args.top_k, args.limit, args.verbose)


if __name__ == "__main__":
    main()
