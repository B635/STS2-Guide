"""Stratified dev/test split creation and reproducible offline evaluation metrics."""

from __future__ import annotations

import collections
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from fact_pipeline.compare import ComparisonResult, compare_facts
from fact_pipeline.pipeline import EntityPipelineResult
from storage.assembly_facts import REVIEWED_STATUSES, extract_index, load_extract, load_facts

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = ROOT_DIR / "fact_pipeline" / "eval" / "split.json"
DEFAULT_V0113_INPUTS_DIR = ROOT_DIR / "build" / "fact-pipeline" / "v0113-inputs"

DEFAULT_DEV_QUOTAS: Dict[str, int] = {
    "cards_simple": 4,
    "cards_complex": 4,
    "cards_fool": 3,
    "cards_hard": 2,
    "powers_all": 5,
    "relics_combat": 3,
    "relics_out_of_combat": 3,
}

DEFAULT_TEST_QUOTAS: Dict[str, int] = {
    "cards_simple": 10,
    "cards_complex": 12,
    "cards_fool": 6,
    "cards_hard": 6,
    "powers_all": 12,
    "relics_combat": 6,
    "relics_out_of_combat": 5,
    "relics_hard": 3,
}


def export_git_branch_inputs(branch: str = "feature/v0113", out_dir: Path = DEFAULT_V0113_INPUTS_DIR) -> Dict[str, Path]:
    """Export versioned extract/facts/knowledge JSONs from a git ref into an ignored build dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "extract": ("data/assembly/sts2-extract.json", out_dir / "sts2-extract.json"),
        "facts": ("data/assembly/facts.json", out_dir / "facts.json"),
        "knowledge": ("data/knowledge.json", out_dir / "knowledge.json"),
    }
    paths: Dict[str, Path] = {}
    for name, (git_path, target) in mapping.items():
        content = subprocess.check_output(
            ["git", "show", f"{branch}:{git_path}"],
            cwd=ROOT_DIR,
            text=True,
            encoding="utf-8",
        )
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        paths[name] = target
    return paths


def classify_stratum(entry: Mapping[str, Any]) -> Optional[str]:
    """Assign a reviewed ground-truth entity into one mutually exclusive difficulty stratum."""
    key = str(entry.get("entity_key", ""))
    if entry.get("status") not in REVIEWED_STATUSES:
        return None
    entity_type = key.split(":", 1)[0]
    verdict = (entry.get("audit") or {}).get("verdict")
    reviewed_by = str(entry.get("reviewed_by") or "")

    if entity_type == "cards":
        if verdict == "resolved":
            return "cards_hard"
        if "TASK-029" in reviewed_by or any(t in ("Shoot", "Wah", "Cho", "Ki") for t in (entry.get("tags") or ())):
            return "cards_fool"
        is_complex = (
            len(entry.get("effects") or ()) >= 2
            or bool(entry.get("mechanics"))
            or bool(entry.get("cost_modifier"))
            or bool(entry.get("star_cost"))
            or any(
                isinstance(e, dict) and (e.get("condition") or e.get("scale") or e.get("hits_scale"))
                for e in (entry.get("effects") or ())
            )
        )
        return "cards_complex" if is_complex else "cards_simple"

    if entity_type == "powers":
        return "powers_all"

    if entity_type == "relics":
        if verdict == "resolved":
            return "relics_hard"
        if entry.get("category") == "irrelevant":
            return "relics_out_of_combat"
        return "relics_combat"

    return None


def create_stratified_split(
    *,
    facts_path: Path,
    extract_path: Path,
    seed: int = 42,
    dev_quotas: Mapping[str, int] = DEFAULT_DEV_QUOTAS,
    test_quotas: Mapping[str, int] = DEFAULT_TEST_QUOTAS,
) -> Dict[str, Any]:
    """Build a deterministic, non-overlapping stratified dev/test split."""
    extract = load_extract(extract_path)
    facts = load_facts(facts_path)
    by_extract = extract_index(extract)

    buckets: Dict[str, List[str]] = collections.defaultdict(list)
    for entry in facts["entities"]:
        key = entry["entity_key"]
        if key not in by_extract:
            continue
        if entry.get("fingerprint") != by_extract[key].get("fingerprint"):
            continue
        stratum = classify_stratum(entry)
        if stratum is not None:
            buckets[stratum].append(key)

    rng = random.Random(seed)
    dev_items: List[Dict[str, str]] = []
    test_items: List[Dict[str, str]] = []
    strata_summary: Dict[str, Dict[str, int]] = {}

    for stratum in sorted(set(dev_quotas) | set(test_quotas)):
        pool = sorted(buckets.get(stratum, []))
        rng.shuffle(pool)
        n_dev = int(dev_quotas.get(stratum, 0))
        n_test = int(test_quotas.get(stratum, 0))
        if n_dev + n_test > len(pool):
            raise ValueError(
                f"Stratum {stratum!r} has only {len(pool)} entities, need {n_dev + n_test}"
            )
        chosen_dev = sorted(pool[:n_dev])
        chosen_test = sorted(pool[n_dev : n_dev + n_test])
        dev_items.extend({"entity_key": k, "stratum": stratum} for k in chosen_dev)
        test_items.extend({"entity_key": k, "stratum": stratum} for k in chosen_test)
        strata_summary[stratum] = {"pool": len(pool), "dev": len(chosen_dev), "test": len(chosen_test)}

    return {
        "seed": seed,
        "assembly_sha256": extract.get("assembly_sha256"),
        "strata": strata_summary,
        "dev": dev_items,
        "test": test_items,
    }


def _rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return round(num / den, 4)


def compute_eval_metrics(
    results: Sequence[EntityPipelineResult],
    ground_truth_by_key: Mapping[str, Mapping[str, Any]],
    strata_by_key: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Compute reproducible offline accuracy, safety, auditor recall, revision, and cost metrics."""
    strata_by_key = strata_by_key or {}
    total = len(results)
    schema_first_pass = 0
    schema_final_pass = 0
    auto_approved = 0
    auto_core_ok = 0
    auto_strict_ok = 0
    overall_core_ok = 0
    overall_strict_ok = 0
    initial_core_ok = 0
    initial_strict_ok = 0
    needs_human = 0
    needs_new_vocab = 0

    # Auditor confusion matrix on initial_draft vs ground truth (entities that reached audit)
    audited_total = 0
    buggy_initial = 0
    auditor_caught = 0
    good_initial = 0
    auditor_false_alarm = 0

    # Revision net effect
    revisions = 0
    rev_fixed_bug = 0
    rev_broke_good = 0
    rev_still_buggy = 0
    rev_kept_good = 0

    mismatch_fields = collections.Counter()
    reason_counts = collections.Counter()
    by_type: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {
            "total": 0,
            "auto_approved": 0,
            "auto_core_ok": 0,
            "auto_strict_ok": 0,
            "overall_core_ok": 0,
            "overall_strict_ok": 0,
        }
    )
    by_stratum: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"total": 0, "auto_approved": 0, "auto_core_ok": 0, "overall_core_ok": 0}
    )

    per_entity: List[Dict[str, Any]] = []
    total_prompt = 0
    total_comp = 0
    total_cache_hit = 0
    total_cost = 0.0
    total_latency = 0

    for res in results:
        key = res.entity_key
        gold = ground_truth_by_key[key]
        stratum = strata_by_key.get(key, res.entity_type)

        final_cmp: ComparisonResult = compare_facts(key, res.fact, gold)
        init_cmp: ComparisonResult = compare_facts(key, res.initial_draft, gold)

        final_validation_ok = (
            res.initial_draft is not None
            and res.fact is not None
            and not res.validation_problems
            and res.reason not in ("schema_validation_failed", "revision_validation_failed", "il_extract_failed", "llm_error")
        )
        if final_validation_ok:
            schema_final_pass += 1
            if res.schema_repairs == 0:
                schema_first_pass += 1
        if init_cmp.core_match:
            initial_core_ok += 1
        if init_cmp.strict_match:
            initial_strict_ok += 1
        if final_cmp.core_match:
            overall_core_ok += 1
        if final_cmp.strict_match:
            overall_strict_ok += 1

        reason_counts[res.reason] += 1
        if res.needs_new_vocabulary:
            needs_new_vocab += 1

        t_bucket = by_type[res.entity_type]
        s_bucket = by_stratum[stratum]
        t_bucket["total"] += 1
        s_bucket["total"] += 1
        if final_cmp.core_match:
            t_bucket["overall_core_ok"] += 1
            s_bucket["overall_core_ok"] += 1
        if final_cmp.strict_match:
            t_bucket["overall_strict_ok"] += 1

        if res.outcome == "auto_approved":
            auto_approved += 1
            t_bucket["auto_approved"] += 1
            s_bucket["auto_approved"] += 1
            if final_cmp.core_match:
                auto_core_ok += 1
                t_bucket["auto_core_ok"] += 1
                s_bucket["auto_core_ok"] += 1
            if final_cmp.strict_match:
                auto_strict_ok += 1
                t_bucket["auto_strict_ok"] += 1
        else:
            needs_human += 1

        if res.initial_audit is not None and res.initial_draft is not None:
            audited_total += 1
            disagreed = res.initial_audit.verdict != "agree" or res.initial_audit.needs_new_vocabulary
            if not init_cmp.core_match:
                buggy_initial += 1
                if disagreed:
                    auditor_caught += 1
            else:
                good_initial += 1
                if disagreed:
                    auditor_false_alarm += 1

        if res.revised:
            revisions += 1
            if not init_cmp.core_match and final_cmp.core_match:
                rev_fixed_bug += 1
            elif init_cmp.core_match and not final_cmp.core_match:
                rev_broke_good += 1
            elif not init_cmp.core_match and not final_cmp.core_match:
                rev_still_buggy += 1
            else:
                rev_kept_good += 1

        if not final_cmp.core_match:
            for field_name in final_cmp.field_diffs:
                mismatch_fields[field_name] += 1

        total_prompt += res.usage.prompt_tokens
        total_comp += res.usage.completion_tokens
        total_cache_hit += res.usage.prompt_cache_hit_tokens
        total_cost += res.usage.cost_usd
        total_latency += res.latency_ms

        per_entity.append(
            {
                "entity_key": key,
                "stratum": stratum,
                "outcome": res.outcome,
                "reason": res.reason,
                "revised": res.revised,
                "initial_core_match": init_cmp.core_match,
                "final_core_match": final_cmp.core_match,
                "final_strict_match": final_cmp.strict_match,
                "initial_audit_verdict": res.initial_audit.verdict if res.initial_audit else None,
                "final_audit_verdict": res.final_audit.verdict if res.final_audit else None,
                "diff_fields": sorted(final_cmp.field_diffs.keys()),
                "field_diffs": final_cmp.field_diffs if not final_cmp.core_match else {},
                "cost_usd": round(res.usage.cost_usd, 6),
                "latency_ms": res.latency_ms,
            }
        )

    def _finalize_bucket(raw: Mapping[str, Dict[str, int]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, b in sorted(raw.items()):
            entry = dict(b)
            entry["auto_approved_rate"] = _rate(b["auto_approved"], b["total"])
            entry["auto_core_precision"] = _rate(b["auto_core_ok"], b["auto_approved"])
            entry["overall_core_accuracy"] = _rate(b["overall_core_ok"], b["total"])
            out[name] = entry
        return out

    return {
        "total": total,
        "schema_first_pass": schema_first_pass,
        "schema_first_pass_rate": _rate(schema_first_pass, total),
        "schema_final_pass": schema_final_pass,
        "schema_final_pass_rate": _rate(schema_final_pass, total),
        "auto_approved": auto_approved,
        "auto_approved_rate": _rate(auto_approved, total),
        "needs_human": needs_human,
        "needs_human_rate": _rate(needs_human, total),
        "needs_new_vocabulary": needs_new_vocab,
        "auto_approved_core_ok": auto_core_ok,
        "auto_approved_core_precision": _rate(auto_core_ok, auto_approved),
        "auto_approved_strict_ok": auto_strict_ok,
        "auto_approved_strict_precision": _rate(auto_strict_ok, auto_approved),
        "initial_draft_core_ok": initial_core_ok,
        "initial_draft_core_accuracy": _rate(initial_core_ok, total),
        "initial_draft_strict_ok": initial_strict_ok,
        "overall_core_ok": overall_core_ok,
        "overall_core_accuracy": _rate(overall_core_ok, total),
        "overall_strict_ok": overall_strict_ok,
        "overall_strict_accuracy": _rate(overall_strict_ok, total),
        "auditor": {
            "audited_total": audited_total,
            "buggy_initial": buggy_initial,
            "caught_bugs": auditor_caught,
            "bug_recall": _rate(auditor_caught, buggy_initial),
            "good_initial": good_initial,
            "false_alarms": auditor_false_alarm,
            "false_alarm_rate": _rate(auditor_false_alarm, good_initial),
        },
        "revision": {
            "attempted": revisions,
            "fixed_bug": rev_fixed_bug,
            "broke_good": rev_broke_good,
            "still_buggy": rev_still_buggy,
            "kept_good": rev_kept_good,
        },
        "reasons": dict(sorted(reason_counts.items())),
        "mismatch_fields": dict(mismatch_fields.most_common()),
        "by_type": _finalize_bucket(by_type),
        "by_stratum": _finalize_bucket(by_stratum),
        "usage": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_comp,
            "prompt_cache_hit_tokens": total_cache_hit,
            "cache_hit_rate": _rate(total_cache_hit, total_prompt),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / total, 6) if total else 0.0,
            "avg_latency_ms": int(total_latency / total) if total else 0,
        },
        "per_entity": per_entity,
    }


def load_saved_results(run_dir: Path, entity_keys: Optional[Sequence[str]] = None) -> List[EntityPipelineResult]:
    results_dir = Path(run_dir) / "results"
    by_key: Dict[str, EntityPipelineResult] = {}
    for p in sorted(results_dir.glob("*.json")):
        item = EntityPipelineResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
        by_key[item.entity_key] = item
    if entity_keys is None:
        return [by_key[k] for k in sorted(by_key)]
    return [by_key[k] for k in entity_keys if k in by_key]
