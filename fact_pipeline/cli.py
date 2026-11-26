"""Command-line entry point for fact_pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fact_pipeline.evaluate import (
    DEFAULT_SPLIT_PATH,
    DEFAULT_V0113_INPUTS_DIR,
    compute_eval_metrics,
    create_stratified_split,
    export_git_branch_inputs,
    load_saved_results,
)
from fact_pipeline.extractor import (
    DEFAULT_IL_CACHE_DIR,
    CachedAssemblyProvider,
    resolve_default_dll,
    resolve_default_extractor,
)
from fact_pipeline.llm import COMPARISON_MODEL, DEFAULT_MODEL, OpenAIChatClient
from fact_pipeline.pipeline import (
    EntityPipelineResult,
    PipelineConfig,
    run_batch,
    write_review_queue_markdown,
)
from fact_pipeline.router import COMPLEXITY_TIERS, ROUTING_MODES
from fact_pipeline.validator import SUPPORTED_ENTITY_TYPES
from storage.assembly_facts import (
    DEFAULT_EXTRACT_PATH,
    DEFAULT_FACTS_PATH,
    evaluate_facts,
    extract_index,
    load_extract,
    load_facts,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_PATH = ROOT_DIR / "data" / "knowledge.json"


def load_workspace_inputs(
    *,
    extract_path: Path = DEFAULT_EXTRACT_PATH,
    facts_path: Path = DEFAULT_FACTS_PATH,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
) -> tuple:
    extract = load_extract(extract_path)
    facts = load_facts(facts_path)
    knowledge = json.loads(Path(knowledge_path).read_text(encoding="utf-8"))
    by_extract = extract_index(extract)
    by_facts = {entry["entity_key"]: entry for entry in facts["entities"]}
    catalog_by_type: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for entity_type in SUPPORTED_ENTITY_TYPES:
        rows = knowledge.get(entity_type) or []
        catalog_by_type[entity_type] = {row["id"]: row for row in rows if "id" in row}
    return extract, facts, by_extract, by_facts, catalog_by_type


def sanitize_facts_for_eval(
    facts_by_key: Mapping[str, Mapping[str, Any]],
    held_out_keys: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Strip ground-truth fields from every split entity so lookup_fact cannot leak answers."""
    blocked = set(held_out_keys)
    clean: Dict[str, Dict[str, Any]] = {}
    for key, entry in facts_by_key.items():
        if key in blocked:
            stub: Dict[str, Any] = {
                "entity_key": key,
                "class": entry.get("class"),
                "status": "unreviewed",
            }
            if "triage" in entry:
                stub["triage"] = entry["triage"]
            clean[key] = stub
        else:
            clean[key] = dict(entry)
    return clean


def select_target_entities(
    *,
    extract: Mapping[str, Any],
    facts: Mapping[str, Any],
    by_extract: Mapping[str, Mapping[str, Any]],
    explicit: Optional[Sequence[str]] = None,
    states: Sequence[str] = ("unreviewed", "stale"),
    entity_types: Sequence[str] = SUPPORTED_ENTITY_TYPES,
    limit: Optional[int] = None,
) -> List[str]:
    if explicit:
        keys = [k.strip() for k in explicit if k.strip()]
        missing = [k for k in keys if k not in by_extract]
        if missing:
            raise ValueError(f"Entities not found in extract: {missing}")
        return keys[:limit] if limit is not None else keys

    evaluated = evaluate_facts(facts, extract)
    want_states = set(states)
    want_types = set(entity_types)
    selected = [
        key
        for key, info in sorted(evaluated.items())
        if info["state"] in want_states
        and key.split(":", 1)[0] in want_types
        and key in by_extract
    ]
    return selected[:limit] if limit is not None else selected


def _cmd_run(args: argparse.Namespace) -> int:
    extract, facts, by_extract, by_facts, catalog_by_type = load_workspace_inputs(
        extract_path=args.extract,
        facts_path=args.facts,
        knowledge_path=args.knowledge,
    )
    keys = select_target_entities(
        extract=extract,
        facts=facts,
        by_extract=by_extract,
        explicit=args.entities,
        states=args.states,
        entity_types=args.types,
        limit=args.limit,
    )
    if not keys:
        print("No matching entities to process.")
        return 0

    dll_path = args.dll or resolve_default_dll()
    extractor_path = args.extractor or resolve_default_extractor()
    assembly = CachedAssemblyProvider(
        dll_path=dll_path,
        extractor_path=extractor_path,
        cache_dir=args.il_cache,
        assembly_tag=str(extract.get("assembly_sha256", "default"))[:12],
    )
    llm = OpenAIChatClient()
    config = PipelineConfig(
        drafter_model=args.model,
        auditor_model=args.auditor_model or args.model,
        thinking=args.thinking,
        auditor_thinking=True if args.auditor_thinking else None,
        skip_audit=args.skip_audit,
        max_revision_rounds=0 if args.no_revision else 1,
        routing=args.routing,
        hard_drafter_model=args.hard_model,
        auto_approve_max_tier=args.auto_approve_max_tier,
        allow_revised_auto_approve=args.allow_revised_auto_approve,
    )

    def _on_done(res: EntityPipelineResult, done: int, total: int) -> None:
        print(
            f"[{done}/{total}] {res.entity_key:<28} {res.outcome:<13} {res.reason:<32} "
            f"${res.usage.cost_usd:.5f} ({res.latency_ms} ms)",
            flush=True,
        )

    results = run_batch(
        keys,
        extract_by_key=by_extract,
        facts_by_key=by_facts,
        catalog_by_type=catalog_by_type,
        assembly=assembly,
        llm=llm,
        out_dir=args.out,
        config=config,
        workers=args.workers,
        force=args.force,
        progress_cb=_on_done,
    )
    auto = sum(1 for r in results if r.outcome == "auto_approved")
    cost = sum(r.usage.cost_usd for r in results)
    print(f"Done: {len(results)} entities, {auto} auto-approved, {len(results) - auto} needs_human, ${cost:.4f} -> {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results_dir = args.run_dir / "results"
    if not results_dir.is_dir():
        print(f"No results directory at {results_dir}", file=sys.stderr)
        return 2
    items = [
        EntityPipelineResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(results_dir.glob("*.json"))
    ]
    out_path = args.out or (args.run_dir / "review-queue.md")
    write_review_queue_markdown(items, out_path)
    print(f"Wrote {out_path} ({len(items)} entities)")
    return 0


def _cmd_eval_export_branch(args: argparse.Namespace) -> int:
    paths = export_git_branch_inputs(branch=args.branch, out_dir=args.out)
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


def _cmd_eval_split(args: argparse.Namespace) -> int:
    split = create_stratified_split(
        facts_path=args.facts,
        extract_path=args.extract,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(split, ensure_ascii=False, indent=1) + "\n")
    print(
        f"Wrote {args.out}: dev={len(split['dev'])}, test={len(split['test'])}, "
        f"strata={json.dumps(split['strata'])}"
    )
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    items = split[args.set_name]
    keys = [row["entity_key"] for row in items]
    if args.only:
        allowed = {x.strip() for x in args.only.split(",") if x.strip()}
        keys = [k for k in keys if k in allowed]
    if args.limit is not None:
        keys = keys[: args.limit]
    strata = {row["entity_key"]: row["stratum"] for row in items}

    extract, _facts, by_extract, by_facts, catalog_by_type = load_workspace_inputs(
        extract_path=args.extract,
        facts_path=args.facts,
        knowledge_path=args.knowledge,
    )
    all_split_keys = [row["entity_key"] for row in split["dev"]] + [row["entity_key"] for row in split["test"]]
    sanitized_facts = sanitize_facts_for_eval(by_facts, all_split_keys)

    dll_path = args.dll or resolve_default_dll()
    extractor_path = args.extractor or resolve_default_extractor()
    assembly = CachedAssemblyProvider(
        dll_path=dll_path,
        extractor_path=extractor_path,
        cache_dir=args.il_cache,
        assembly_tag=str(extract.get("assembly_sha256", "default"))[:12],
    )
    llm = OpenAIChatClient()
    config = PipelineConfig(
        drafter_model=args.model,
        auditor_model=args.auditor_model or args.model,
        thinking=args.thinking,
        auditor_thinking=True if args.auditor_thinking else None,
        skip_audit=args.skip_audit,
        max_revision_rounds=0 if args.no_revision else 1,
        routing=args.routing,
        hard_drafter_model=args.hard_model,
        auto_approve_max_tier=args.auto_approve_max_tier,
        allow_revised_auto_approve=args.allow_revised_auto_approve,
    )

    def _on_done(res: EntityPipelineResult, done: int, total: int) -> None:
        print(
            f"[{done}/{total}] {res.entity_key:<28} {res.outcome:<13} {res.reason:<32} "
            f"${res.usage.cost_usd:.5f} ({res.latency_ms} ms)",
            flush=True,
        )

    results = run_batch(
        keys,
        extract_by_key=by_extract,
        facts_by_key=sanitized_facts,
        catalog_by_type=catalog_by_type,
        assembly=assembly,
        llm=llm,
        out_dir=args.out,
        config=config,
        workers=args.workers,
        force=args.force,
        progress_cb=_on_done,
    )
    metrics = compute_eval_metrics(results, by_facts, strata)
    metrics_path = args.metrics_out or (args.out / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n")
    _print_metrics_summary(metrics)
    print(f"Metrics saved: {metrics_path}")
    return 0


def _cmd_eval_metrics(args: argparse.Namespace) -> int:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    items = split[args.set_name]
    keys = [row["entity_key"] for row in items]
    strata = {row["entity_key"]: row["stratum"] for row in items}
    facts = load_facts(args.facts)
    by_facts = {entry["entity_key"]: entry for entry in facts["entities"]}
    results = load_saved_results(args.run_dir, entity_keys=keys)
    metrics = compute_eval_metrics(results, by_facts, strata)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n")
    _print_metrics_summary(metrics)
    return 0


def _print_metrics_summary(m: Mapping[str, Any]) -> None:
    aud = m["auditor"]
    rev = m["revision"]
    u = m["usage"]
    print(
        f"total={m['total']} | schema_pass={m['schema_final_pass']}/{m['total']} "
        f"(first={m['schema_first_pass']}) | auto_approved={m['auto_approved']}/{m['total']} "
        f"({m['auto_approved_rate']}) | auto_core_prec={m['auto_approved_core_ok']}/{m['auto_approved']} "
        f"({m['auto_approved_core_precision']}) | auto_strict_prec={m['auto_approved_strict_ok']}/{m['auto_approved']} "
        f"({m['auto_approved_strict_precision']})"
    )
    print(
        f"initial_core={m['initial_draft_core_ok']}/{m['total']} ({m['initial_draft_core_accuracy']}) "
        f"-> final_core={m['overall_core_ok']}/{m['total']} ({m['overall_core_accuracy']}) | "
        f"auditor_bug_recall={aud['caught_bugs']}/{aud['buggy_initial']} ({aud['bug_recall']}), "
        f"false_alarms={aud['false_alarms']}/{aud['good_initial']} ({aud['false_alarm_rate']})"
    )
    print(
        f"revision: attempted={rev['attempted']}, fixed={rev['fixed_bug']}, broke={rev['broke_good']}, "
        f"still_buggy={rev['still_buggy']} | cost=${u['total_cost_usd']:.4f} "
        f"(avg=${u['avg_cost_usd']:.5f}, cache_hit={u['cache_hit_rate']})"
    )
    if m["mismatch_fields"]:
        print(f"top_mismatch_fields: {json.dumps(m['mismatch_fields'])}")


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extract", type=Path, default=DEFAULT_EXTRACT_PATH)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS_PATH)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE_PATH)
    parser.add_argument("--dll", type=Path, default=None)
    parser.add_argument("--extractor", type=Path, default=None)
    parser.add_argument("--il-cache", type=Path, default=DEFAULT_IL_CACHE_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fact_pipeline", description="STS2 automated IL-to-fact pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Draft, validate, and audit entities.")
    _add_input_args(p_run)
    p_run.add_argument("--entities", nargs="*", default=None, help="Explicit entity keys (e.g. cards:ABRASIVE).")
    p_run.add_argument("--states", nargs="+", default=["unreviewed", "stale"], choices=["unreviewed", "stale", "converted", "unknowable"])
    p_run.add_argument("--types", nargs="+", default=list(SUPPORTED_ENTITY_TYPES), choices=list(SUPPORTED_ENTITY_TYPES))
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument("--auditor-model", default=None)
    p_run.add_argument("--thinking", action="store_true")
    p_run.add_argument("--auditor-thinking", action="store_true")
    p_run.add_argument("--skip-audit", action="store_true")
    p_run.add_argument("--no-revision", action="store_true")
    p_run.add_argument("--routing", default="fixed", choices=list(ROUTING_MODES))
    p_run.add_argument("--hard-model", default=COMPARISON_MODEL)
    p_run.add_argument("--auto-approve-max-tier", default="medium", choices=["none"] + list(COMPLEXITY_TIERS))
    p_run.add_argument("--allow-revised-auto-approve", action="store_true")
    p_run.add_argument("--workers", type=int, default=2)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--out", type=Path, required=True)
    p_run.set_defaults(func=_cmd_run)

    p_report = sub.add_parser("report", help="Regenerate human review queue Markdown from saved results.")
    p_report.add_argument("run_dir", type=Path)
    p_report.add_argument("--out", type=Path, default=None)
    p_report.set_defaults(func=_cmd_report)

    p_eval = sub.add_parser("eval", help="Offline benchmark evaluation commands.")
    eval_sub = p_eval.add_subparsers(dest="eval_command", required=True)

    p_exp = eval_sub.add_parser("export-branch", help="Export extract/facts/knowledge from a git branch into build/.")
    p_exp.add_argument("--branch", default="feature/v0113")
    p_exp.add_argument("--out", type=Path, default=DEFAULT_V0113_INPUTS_DIR)
    p_exp.set_defaults(func=_cmd_eval_export_branch)

    p_split = eval_sub.add_parser("split", help="Generate fixed stratified dev/test split JSON.")
    p_split.add_argument("--extract", type=Path, required=True)
    p_split.add_argument("--facts", type=Path, required=True)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--out", type=Path, default=DEFAULT_SPLIT_PATH)
    p_split.set_defaults(func=_cmd_eval_split)

    p_erun = eval_sub.add_parser("run", help="Run pipeline on dev or test split with anti-leakage sanitization.")
    p_erun.add_argument("set_name", choices=["dev", "test"])
    _add_input_args(p_erun)
    p_erun.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    p_erun.add_argument("--only", default=None, help="Comma-separated entity keys subset.")
    p_erun.add_argument("--limit", type=int, default=None)
    p_erun.add_argument("--model", default=DEFAULT_MODEL)
    p_erun.add_argument("--auditor-model", default=None)
    p_erun.add_argument("--thinking", action="store_true")
    p_erun.add_argument("--auditor-thinking", action="store_true")
    p_erun.add_argument("--skip-audit", action="store_true")
    p_erun.add_argument("--no-revision", action="store_true")
    p_erun.add_argument("--routing", default="fixed", choices=list(ROUTING_MODES))
    p_erun.add_argument("--hard-model", default=COMPARISON_MODEL)
    p_erun.add_argument("--auto-approve-max-tier", default="medium", choices=["none"] + list(COMPLEXITY_TIERS))
    p_erun.add_argument("--allow-revised-auto-approve", action="store_true")
    p_erun.add_argument("--workers", type=int, default=3)
    p_erun.add_argument("--force", action="store_true")
    p_erun.add_argument("--out", type=Path, required=True)
    p_erun.add_argument("--metrics-out", type=Path, default=None)
    p_erun.set_defaults(func=_cmd_eval_run)

    p_emet = eval_sub.add_parser("metrics", help="Recompute metrics offline from saved run results.")
    p_emet.add_argument("set_name", choices=["dev", "test"])
    p_emet.add_argument("run_dir", type=Path)
    p_emet.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    p_emet.add_argument("--facts", type=Path, required=True)
    p_emet.add_argument("--out", type=Path, default=None)
    p_emet.set_defaults(func=_cmd_eval_metrics)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
