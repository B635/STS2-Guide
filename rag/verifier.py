"""Rule-based answer verification for grounded RAG responses."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


CITATION_RE = re.compile(r"\[(\d+|\?)\]")
NUMBER_RE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")


@dataclass
class VerificationResult:
    passed: bool
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
    merged: List[str] = []
    for part in parts:
        if part.startswith("[") and merged:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


def is_substantive(sentence: str) -> bool:
    stripped = sentence.strip("*- |：:#").strip()
    if not stripped:
        return False
    if sentence.lstrip().startswith("#"):
        return False
    if sentence.strip().startswith("**") and sentence.strip().endswith("**"):
        return False
    if stripped.startswith("基于当前检索到的证据"):
        return False
    if stripped.endswith("如下") or stripped.endswith("如下："):
        return False
    if re.fullmatch(r"[\s\-\|=:]+", stripped):
        return False
    return len(stripped) >= 4


def numbers_in(text: str) -> List[str]:
    return NUMBER_RE.findall(text)


def verify_answer(answer: str, results: List[Dict]) -> VerificationResult:
    """Verify citation shape and simple numeric grounding without an LLM judge."""
    sources = [str(r.get("text", "")) for r in results]
    max_valid_id = len(sources)

    substantive_sentences = 0
    cited_sentences = 0
    invalid_citations = 0
    ungrounded_number_sentences = 0
    unknown_markers = 0

    for sentence in split_sentences(answer):
        if not is_substantive(sentence):
            continue
        substantive_sentences += 1

        cites = CITATION_RE.findall(sentence)
        if not cites:
            continue
        cited_sentences += 1

        valid_cited_texts = []
        citation_numbers = {c for c in cites if c.isdigit()}
        for cite in cites:
            if cite == "?":
                unknown_markers += 1
                continue
            citation_id = int(cite)
            if citation_id < 1 or citation_id > max_valid_id:
                invalid_citations += 1
            else:
                valid_cited_texts.append(sources[citation_id - 1])

        nums = [n for n in numbers_in(sentence) if n not in citation_numbers]
        if nums and valid_cited_texts:
            merged_sources = " ".join(valid_cited_texts)
            if not all(n in merged_sources for n in nums):
                ungrounded_number_sentences += 1

    uncited_sentences = max(substantive_sentences - cited_sentences, 0)
    notes: List[str] = []
    if uncited_sentences:
        notes.append(f"{uncited_sentences} substantive sentence(s) missing citation markers.")
    if invalid_citations:
        notes.append(f"{invalid_citations} citation marker(s) point outside provided sources.")
    if ungrounded_number_sentences:
        notes.append(f"{ungrounded_number_sentences} cited sentence(s) contain numbers not found in cited sources.")
    if unknown_markers:
        notes.append(f"{unknown_markers} [?] marker(s) indicate insufficient evidence.")
    if not notes:
        notes.append("Citation markers and numeric grounding passed rule checks.")

    passed = uncited_sentences == 0 and invalid_citations == 0 and ungrounded_number_sentences == 0
    return VerificationResult(
        passed=passed,
        notes=notes,
        stats={
            "substantive_sentences": substantive_sentences,
            "cited_sentences": cited_sentences,
            "uncited_sentences": uncited_sentences,
            "invalid_citations": invalid_citations,
            "ungrounded_number_sentences": ungrounded_number_sentences,
            "unknown_markers": unknown_markers,
        },
    )


def format_verification_summary(result: VerificationResult) -> str:
    status = "passed" if result.passed else "failed"
    return f"Verification {status}: " + " ".join(result.notes)
