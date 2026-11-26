"""Core orchestration state machine: Drafter -> Deterministic Gate -> Independent Auditor -> Triage."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from fact_pipeline import PIPELINE_VERSION
from fact_pipeline.extractor import AssemblyProvider, AssemblyProviderError
from fact_pipeline.llm import DEFAULT_MODEL, LLMClient, LLMResponse, LLMUsage, ToolCall
from fact_pipeline.prompts import (
    AUDITOR_TOOLS,
    DRAFTER_TOOLS,
    build_auditor_prompt,
    build_auditor_system_prompt,
    build_drafter_system_prompt,
    build_entity_context,
    build_revision_prompt,
)
from fact_pipeline.validator import ValidationResult, prepare_mergeable_entry, validate_fact

# Explicit named budgets per entity.
MAX_DRAFTER_TOOL_CALLS = 6
MAX_REVISION_TOOL_CALLS = 4
MAX_SCHEMA_REPAIRS = 2
MAX_AUDITOR_TOOL_CALLS = 4
MAX_REVISION_ROUNDS = 1


@dataclass(frozen=True)
class PipelineConfig:
    drafter_model: str = DEFAULT_MODEL
    auditor_model: str = DEFAULT_MODEL
    thinking: bool = False
    auditor_thinking: Optional[bool] = None
    skip_audit: bool = False
    max_drafter_tools: int = MAX_DRAFTER_TOOL_CALLS
    max_revision_tools: int = MAX_REVISION_TOOL_CALLS
    max_schema_repairs: int = MAX_SCHEMA_REPAIRS
    max_auditor_tools: int = MAX_AUDITOR_TOOL_CALLS
    max_revision_rounds: int = MAX_REVISION_ROUNDS
    reviewed_by_tag: str = PIPELINE_VERSION

    @property
    def use_auditor_thinking(self) -> bool:
        return self.thinking if self.auditor_thinking is None else self.auditor_thinking


@dataclass
class AuditOutcome:
    verdict: str  # "agree" | "disagree" | "error"
    issues: List[str] = field(default_factory=list)
    needs_new_vocabulary: bool = False
    notes: str = ""
    tool_calls: int = 0
    usage: LLMUsage = field(default_factory=LLMUsage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "issues": list(self.issues),
            "needs_new_vocabulary": self.needs_new_vocabulary,
            "notes": self.notes,
            "tool_calls": self.tool_calls,
            "usage": self.usage.to_dict(),
        }


@dataclass
class EntityPipelineResult:
    entity_key: str
    entity_type: str
    class_name: str
    fingerprint: str
    outcome: str  # "auto_approved" | "needs_human"
    reason: str
    fact: Optional[Dict[str, Any]]
    initial_draft: Optional[Dict[str, Any]]
    initial_audit: Optional[AuditOutcome]
    final_audit: Optional[AuditOutcome]
    revised: bool = False
    needs_new_vocabulary: bool = False
    vocabulary_gap_note: str = ""
    codex_conflicts: List[str] = field(default_factory=list)
    validation_problems: List[str] = field(default_factory=list)
    schema_repairs: int = 0
    drafter_tool_calls: int = 0
    auditor_tool_calls: int = 0
    usage: LLMUsage = field(default_factory=LLMUsage)
    stage_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latency_ms: int = 0
    drafter_model: str = DEFAULT_MODEL
    auditor_model: str = DEFAULT_MODEL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "entity_key": self.entity_key,
            "entity_type": self.entity_type,
            "class_name": self.class_name,
            "fingerprint": self.fingerprint,
            "outcome": self.outcome,
            "reason": self.reason,
            "fact": self.fact,
            "initial_draft": self.initial_draft,
            "initial_audit": self.initial_audit.to_dict() if self.initial_audit else None,
            "final_audit": self.final_audit.to_dict() if self.final_audit else None,
            "revised": self.revised,
            "needs_new_vocabulary": self.needs_new_vocabulary,
            "vocabulary_gap_note": self.vocabulary_gap_note,
            "codex_conflicts": list(self.codex_conflicts),
            "validation_problems": list(self.validation_problems),
            "schema_repairs": self.schema_repairs,
            "drafter_tool_calls": self.drafter_tool_calls,
            "auditor_tool_calls": self.auditor_tool_calls,
            "usage": self.usage.to_dict(),
            "stage_usage": dict(self.stage_usage),
            "latency_ms": self.latency_ms,
            "drafter_model": self.drafter_model,
            "auditor_model": self.auditor_model,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityPipelineResult":
        def _load_usage(raw: Optional[Mapping[str, Any]]) -> LLMUsage:
            if not raw:
                return LLMUsage()
            return LLMUsage(
                prompt_tokens=int(raw.get("prompt_tokens", 0)),
                completion_tokens=int(raw.get("completion_tokens", 0)),
                prompt_cache_hit_tokens=int(raw.get("prompt_cache_hit_tokens", 0)),
                cost_usd=float(raw.get("cost_usd", 0.0)),
            )

        def _load_audit(raw: Optional[Mapping[str, Any]]) -> Optional[AuditOutcome]:
            if not raw:
                return None
            return AuditOutcome(
                verdict=str(raw.get("verdict", "error")),
                issues=list(raw.get("issues") or ()),
                needs_new_vocabulary=bool(raw.get("needs_new_vocabulary", False)),
                notes=str(raw.get("notes", "")),
                tool_calls=int(raw.get("tool_calls", 0)),
                usage=_load_usage(raw.get("usage")),
            )

        return cls(
            entity_key=str(data["entity_key"]),
            entity_type=str(data["entity_type"]),
            class_name=str(data["class_name"]),
            fingerprint=str(data["fingerprint"]),
            outcome=str(data["outcome"]),
            reason=str(data["reason"]),
            fact=data.get("fact"),
            initial_draft=data.get("initial_draft"),
            initial_audit=_load_audit(data.get("initial_audit")),
            final_audit=_load_audit(data.get("final_audit")),
            revised=bool(data.get("revised", False)),
            needs_new_vocabulary=bool(data.get("needs_new_vocabulary", False)),
            vocabulary_gap_note=str(data.get("vocabulary_gap_note", "")),
            codex_conflicts=list(data.get("codex_conflicts") or ()),
            validation_problems=list(data.get("validation_problems") or ()),
            schema_repairs=int(data.get("schema_repairs", 0)),
            drafter_tool_calls=int(data.get("drafter_tool_calls", 0)),
            auditor_tool_calls=int(data.get("auditor_tool_calls", 0)),
            usage=_load_usage(data.get("usage")),
            stage_usage=dict(data.get("stage_usage") or {}),
            latency_ms=int(data.get("latency_ms", 0)),
            drafter_model=str(data.get("drafter_model", DEFAULT_MODEL)),
            auditor_model=str(data.get("auditor_model", DEFAULT_MODEL)),
        )


def _safe_key(entity_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", entity_key)


def _parse_args(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON tool arguments: {exc}"
    if not isinstance(parsed, dict):
        return None, "Tool arguments must be a JSON object"
    return parsed, None


def _coerce_fact_object(raw_fact: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # Defensive parsing: some models occasionally stringify nested JSON objects.
    if isinstance(raw_fact, str):
        try:
            raw_fact = json.loads(raw_fact)
        except json.JSONDecodeError as exc:
            return None, f"submit_fact.fact was a string that failed JSON parsing: {exc}"
    if not isinstance(raw_fact, dict):
        return None, "submit_fact.fact must be a JSON object"
    return dict(raw_fact), None


def _assistant_message(resp: LLMResponse) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in resp.tool_calls
        ]
    return msg


def _execute_read_tool(
    tc: ToolCall,
    *,
    target_entity_key: str,
    assembly: AssemblyProvider,
    facts_by_key: Mapping[str, Mapping[str, Any]],
) -> str:
    args, err = _parse_args(tc.arguments)
    if err or args is None:
        return json.dumps({"error": err}, ensure_ascii=False)
    try:
        if tc.name == "dump_class":
            class_name = str(args.get("class_name") or "").strip()
            return assembly.dump_class(class_name)
        if tc.name == "read_enum":
            enum_name = str(args.get("enum_name") or "").strip()
            return assembly.read_enum(enum_name)
        if tc.name == "lookup_fact":
            key = str(args.get("entity_key") or "").strip()
            if key == target_entity_key:
                return json.dumps(
                    {"error": "Cannot look up the target entity itself; derive facts directly from the IL."},
                    ensure_ascii=False,
                )
            entry = facts_by_key.get(key)
            if entry is None:
                return json.dumps({"error": f"No entry found for {key!r}"}, ensure_ascii=False)
            return json.dumps(entry, ensure_ascii=False, indent=1)
        return json.dumps({"error": f"Unknown tool {tc.name!r}"}, ensure_ascii=False)
    except AssemblyProviderError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _run_drafter_stage(
    *,
    stage_name: str,
    entity_key: str,
    user_prompt: str,
    system_prompt: str,
    llm: LLMClient,
    assembly: AssemblyProvider,
    extract_by_key: Mapping[str, Mapping[str, Any]],
    facts_by_key: Mapping[str, Mapping[str, Any]],
    catalog_cards: Mapping[str, Mapping[str, Any]],
    config: PipelineConfig,
    max_read_tools: int,
    trace: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], ValidationResult, bool, str, int, int, LLMUsage, int]:
    """Run Drafter (or revision) loop with deterministic schema validation gate."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    usage = LLMUsage()
    latency_ms = 0
    read_calls = 0
    repairs = 0
    last_validation = ValidationResult(ok=False, problems=["no fact submitted"])
    last_fact: Optional[Dict[str, Any]] = None
    needs_new_vocab = False
    vocab_note = ""

    max_turns = max_read_tools + config.max_schema_repairs + 2
    for turn in range(max_turns):
        force_submit = read_calls >= max_read_tools or turn == max_turns - 1
        tool_choice: Any = (
            {"type": "function", "function": {"name": "submit_fact"}}
            if force_submit and not config.thinking
            else "auto"
        )
        resp = llm.chat(
            model=config.drafter_model,
            messages=messages,
            tools=DRAFTER_TOOLS,
            tool_choice=tool_choice,
            temperature=0.0,
            thinking=config.thinking,
        )
        usage = usage.add(resp.usage)
        latency_ms += resp.latency_ms
        trace.append(
            {
                "stage": stage_name,
                "turn": turn,
                "model": config.drafter_model,
                "forced_submit": force_submit,
                "tool_calls": [tc.name for tc in resp.tool_calls],
                "usage": resp.usage.to_dict(),
                "latency_ms": resp.latency_ms,
            }
        )
        messages.append(_assistant_message(resp))

        if not resp.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": "You must submit the complete structured fact by calling the `submit_fact` tool.",
                }
            )
            continue

        submitted_this_turn = False
        for tc in resp.tool_calls:
            if tc.name == "submit_fact":
                submitted_this_turn = True
                args, err = _parse_args(tc.arguments)
                if err or args is None:
                    last_validation = ValidationResult(ok=False, problems=[err or "bad arguments"])
                    repairs += 1
                    trace.append({"stage": stage_name, "event": "validation_failed", "problems": last_validation.problems})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"ok": False, "problems": last_validation.problems}, ensure_ascii=False),
                        }
                    )
                    if repairs > config.max_schema_repairs:
                        return None, last_validation, needs_new_vocab, vocab_note, read_calls, repairs, usage, latency_ms
                    continue

                fact_obj, coerce_err = _coerce_fact_object(args.get("fact"))
                needs_new_vocab = bool(args.get("needs_new_vocabulary", False))
                vocab_note = str(args.get("vocabulary_gap_note") or "").strip()

                if coerce_err or fact_obj is None:
                    last_validation = ValidationResult(ok=False, problems=[coerce_err or "missing fact"])
                    repairs += 1
                    trace.append({"stage": stage_name, "event": "validation_failed", "problems": last_validation.problems})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"ok": False, "problems": last_validation.problems}, ensure_ascii=False),
                        }
                    )
                    if repairs > config.max_schema_repairs:
                        return None, last_validation, needs_new_vocab, vocab_note, read_calls, repairs, usage, latency_ms
                    continue

                # Stamp canonical reviewed_by & fingerprint if omitted only when entity matches?
                # No — keep strict validation so the model learns to preserve exact fingerprint!
                validation = validate_fact(
                    fact_obj,
                    extract_by_key,
                    facts_by_key,
                    expected_entity_key=entity_key,
                    catalog_cards=catalog_cards,
                )
                last_fact = fact_obj
                last_validation = validation
                if validation.ok:
                    trace.append(
                        {
                            "stage": stage_name,
                            "event": "validation_passed",
                            "codex_conflicts": validation.codex_conflicts,
                            "needs_new_vocabulary": needs_new_vocab,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"ok": True}, ensure_ascii=False),
                        }
                    )
                    return fact_obj, validation, needs_new_vocab, vocab_note, read_calls, repairs, usage, latency_ms

                if needs_new_vocab:
                    trace.append(
                        {
                            "stage": stage_name,
                            "event": "needs_new_vocabulary",
                            "note": vocab_note,
                            "problems": validation.problems,
                        }
                    )
                    return fact_obj, validation, True, vocab_note, read_calls, repairs, usage, latency_ms

                repairs += 1
                trace.append({"stage": stage_name, "event": "validation_failed", "problems": validation.problems})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "ok": False,
                                "problems": validation.problems,
                                "instruction": "Fix every listed problem and call submit_fact again.",
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                if repairs > config.max_schema_repairs:
                    return None, validation, needs_new_vocab, vocab_note, read_calls, repairs, usage, latency_ms
            else:
                read_calls += 1
                if read_calls > max_read_tools:
                    tool_out = json.dumps(
                        {"error": "Tool budget exhausted. Call submit_fact now."},
                        ensure_ascii=False,
                    )
                else:
                    tool_out = _execute_read_tool(
                        tc,
                        target_entity_key=entity_key,
                        assembly=assembly,
                        facts_by_key=facts_by_key,
                    )
                trace.append({"stage": stage_name, "event": "read_tool", "tool": tc.name, "args": tc.arguments})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_out})

        if submitted_this_turn and last_validation.ok:
            return last_fact, last_validation, needs_new_vocab, vocab_note, read_calls, repairs, usage, latency_ms

    return None, last_validation, needs_new_vocab, vocab_note, read_calls, repairs, usage, latency_ms


def _run_auditor_stage(
    *,
    stage_name: str,
    entity_key: str,
    entity_context: str,
    candidate_fact: Mapping[str, Any],
    codex_conflicts: Sequence[str],
    system_prompt: str,
    llm: LLMClient,
    assembly: AssemblyProvider,
    facts_by_key: Mapping[str, Mapping[str, Any]],
    config: PipelineConfig,
    trace: List[Dict[str, Any]],
) -> Tuple[AuditOutcome, int]:
    """Run independent Auditor in a completely isolated conversation context."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_auditor_prompt(entity_context, candidate_fact, codex_conflicts)},
    ]
    usage = LLMUsage()
    latency_ms = 0
    read_calls = 0

    max_turns = config.max_auditor_tools + 2
    aud_thinking = config.use_auditor_thinking
    for turn in range(max_turns):
        force_submit = read_calls >= config.max_auditor_tools or turn == max_turns - 1
        tool_choice: Any = (
            {"type": "function", "function": {"name": "submit_audit"}}
            if force_submit and not aud_thinking
            else "auto"
        )
        resp = llm.chat(
            model=config.auditor_model,
            messages=messages,
            tools=AUDITOR_TOOLS,
            tool_choice=tool_choice,
            temperature=0.0,
            thinking=aud_thinking,
        )
        usage = usage.add(resp.usage)
        latency_ms += resp.latency_ms
        trace.append(
            {
                "stage": stage_name,
                "turn": turn,
                "model": config.auditor_model,
                "forced_submit": force_submit,
                "tool_calls": [tc.name for tc in resp.tool_calls],
                "usage": resp.usage.to_dict(),
                "latency_ms": resp.latency_ms,
            }
        )
        messages.append(_assistant_message(resp))

        if not resp.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": "Call `submit_audit` now with your verdict ('agree' or 'disagree'), issues, and notes.",
                }
            )
            continue

        for tc in resp.tool_calls:
            if tc.name == "submit_audit":
                args, err = _parse_args(tc.arguments)
                if err or args is None:
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"error": err}, ensure_ascii=False)}
                    )
                    continue
                verdict = str(args.get("verdict") or "").strip().lower()
                if verdict not in ("agree", "disagree"):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps({"error": "verdict must be 'agree' or 'disagree'"}, ensure_ascii=False),
                        }
                    )
                    continue
                raw_issues = args.get("issues") or []
                issues = [str(x).strip() for x in raw_issues if str(x).strip()] if isinstance(raw_issues, list) else [str(raw_issues)]
                notes = str(args.get("notes") or "").strip() or ("Verified against IL." if verdict == "agree" else "; ".join(issues))
                needs_new_vocab = bool(args.get("needs_new_vocabulary", False))
                outcome = AuditOutcome(
                    verdict=verdict,
                    issues=issues,
                    needs_new_vocabulary=needs_new_vocab,
                    notes=notes,
                    tool_calls=read_calls,
                    usage=usage,
                )
                trace.append({"stage": stage_name, "event": "audit_submitted", "audit": outcome.to_dict()})
                return outcome, latency_ms
            else:
                read_calls += 1
                if read_calls > config.max_auditor_tools:
                    tool_out = json.dumps(
                        {"error": "Tool budget exhausted. Call submit_audit now."},
                        ensure_ascii=False,
                    )
                else:
                    tool_out = _execute_read_tool(
                        tc,
                        target_entity_key=entity_key,
                        assembly=assembly,
                        facts_by_key=facts_by_key,
                    )
                trace.append({"stage": stage_name, "event": "read_tool", "tool": tc.name, "args": tc.arguments})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_out})

    return (
        AuditOutcome(
            verdict="error",
            issues=["auditor exhausted turn budget without calling submit_audit"],
            notes="Auditor did not submit a verdict within budget.",
            tool_calls=read_calls,
            usage=usage,
        ),
        latency_ms,
    )


def process_entity(
    entity_key: str,
    *,
    extract_by_key: Mapping[str, Mapping[str, Any]],
    facts_by_key: Mapping[str, Mapping[str, Any]],
    catalog_by_type: Mapping[str, Mapping[str, Mapping[str, Any]]],
    assembly: AssemblyProvider,
    llm: LLMClient,
    config: PipelineConfig = PipelineConfig(),
    drafter_system: Optional[str] = None,
    auditor_system: Optional[str] = None,
) -> Tuple[EntityPipelineResult, List[Dict[str, Any]]]:
    """Run the full Drafter -> Gate -> Auditor -> (max 1 Revision) -> Triage loop for one entity."""
    trace: List[Dict[str, Any]] = []
    entity_type, entity_id = entity_key.split(":", 1)
    extract_entry = extract_by_key[entity_key]
    facts_entry = facts_by_key[entity_key]
    class_name = str(extract_entry["class"])
    fingerprint = str(extract_entry["fingerprint"])
    catalog_cards = catalog_by_type.get("cards", {})
    catalog_entry = catalog_by_type.get(entity_type, {}).get(entity_id)

    drafter_sys = drafter_system or build_drafter_system_prompt()
    auditor_sys = auditor_system or build_auditor_system_prompt()

    try:
        il_dump = assembly.dump_class(class_name)
    except AssemblyProviderError as exc:
        trace.append({"stage": "prefetch", "event": "il_error", "error": str(exc)})
        return (
            EntityPipelineResult(
                entity_key=entity_key,
                entity_type=entity_type,
                class_name=class_name,
                fingerprint=fingerprint,
                outcome="needs_human",
                reason="il_extract_failed",
                fact=None,
                initial_draft=None,
                initial_audit=None,
                final_audit=None,
                validation_problems=[str(exc)],
                drafter_model=config.drafter_model,
                auditor_model=config.auditor_model,
            ),
            trace,
        )

    entity_context = build_entity_context(
        entity_key=entity_key,
        extract_entry=extract_entry,
        facts_entry=facts_entry,
        il_dump=il_dump,
        catalog_entry=catalog_entry,
    )

    total_usage = LLMUsage()
    stage_usage: Dict[str, Dict[str, Any]] = {}
    total_latency = 0

    try:
        # Stage 1: Initial Draft
        (
            initial_draft,
            val1,
            draft_new_vocab,
            draft_vocab_note,
            d_tools,
            d_repairs,
            d_usage,
            d_ms,
        ) = _run_drafter_stage(
            stage_name="drafter",
            entity_key=entity_key,
            user_prompt=entity_context,
            system_prompt=drafter_sys,
            llm=llm,
            assembly=assembly,
            extract_by_key=extract_by_key,
            facts_by_key=facts_by_key,
            catalog_cards=catalog_cards,
            config=config,
            max_read_tools=config.max_drafter_tools,
            trace=trace,
        )
        total_usage = total_usage.add(d_usage)
        stage_usage["drafter"] = d_usage.to_dict()
        total_latency += d_ms

        if draft_new_vocab:
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="needs_human",
                    reason="needs_new_vocabulary",
                    fact=initial_draft,
                    initial_draft=initial_draft,
                    initial_audit=None,
                    final_audit=None,
                    needs_new_vocabulary=True,
                    vocabulary_gap_note=draft_vocab_note,
                    codex_conflicts=val1.codex_conflicts,
                    validation_problems=val1.problems,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        if initial_draft is None or not val1.ok:
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="needs_human",
                    reason="schema_validation_failed",
                    fact=None,
                    initial_draft=None,
                    initial_audit=None,
                    final_audit=None,
                    validation_problems=val1.problems,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        if config.skip_audit:
            merged = prepare_mergeable_entry(
                initial_draft,
                reviewed_by=f"{config.reviewed_by_tag} ({config.drafter_model})",
                audit_by="skipped_ablation",
                audit_verdict="agree",
                audit_notes="Audit skipped (drafter-only ablation run).",
            )
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="auto_approved",
                    reason="audit_skipped",
                    fact=merged,
                    initial_draft=initial_draft,
                    initial_audit=None,
                    final_audit=None,
                    codex_conflicts=val1.codex_conflicts,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        # Stage 2: Independent Audit (isolated context!)
        audit1, a1_ms = _run_auditor_stage(
            stage_name="auditor",
            entity_key=entity_key,
            entity_context=entity_context,
            candidate_fact=initial_draft,
            codex_conflicts=val1.codex_conflicts,
            system_prompt=auditor_sys,
            llm=llm,
            assembly=assembly,
            facts_by_key=facts_by_key,
            config=config,
            trace=trace,
        )
        total_usage = total_usage.add(audit1.usage)
        stage_usage["auditor"] = audit1.usage.to_dict()
        total_latency += a1_ms
        auditor_tools = audit1.tool_calls

        if audit1.needs_new_vocabulary:
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="needs_human",
                    reason="needs_new_vocabulary",
                    fact=initial_draft,
                    initial_draft=initial_draft,
                    initial_audit=audit1,
                    final_audit=audit1,
                    needs_new_vocabulary=True,
                    vocabulary_gap_note=audit1.notes,
                    codex_conflicts=val1.codex_conflicts,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    auditor_tool_calls=auditor_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        if audit1.verdict == "agree":
            if val1.codex_conflicts:
                return (
                    EntityPipelineResult(
                        entity_key=entity_key,
                        entity_type=entity_type,
                        class_name=class_name,
                        fingerprint=fingerprint,
                        outcome="needs_human",
                        reason="codex_conflict",
                        fact=initial_draft,
                        initial_draft=initial_draft,
                        initial_audit=audit1,
                        final_audit=audit1,
                        codex_conflicts=val1.codex_conflicts,
                        schema_repairs=d_repairs,
                        drafter_tool_calls=d_tools,
                        auditor_tool_calls=auditor_tools,
                        usage=total_usage,
                        stage_usage=stage_usage,
                        latency_ms=total_latency,
                        drafter_model=config.drafter_model,
                        auditor_model=config.auditor_model,
                    ),
                    trace,
                )
            merged = prepare_mergeable_entry(
                initial_draft,
                reviewed_by=f"{config.reviewed_by_tag} ({config.drafter_model})",
                audit_by=f"{config.reviewed_by_tag}-auditor ({config.auditor_model})",
                audit_verdict="agree",
                audit_notes=audit1.notes,
            )
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="auto_approved",
                    reason="auditor_agreed",
                    fact=merged,
                    initial_draft=initial_draft,
                    initial_audit=audit1,
                    final_audit=audit1,
                    codex_conflicts=val1.codex_conflicts,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    auditor_tool_calls=auditor_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        if audit1.verdict != "disagree" or config.max_revision_rounds <= 0:
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="needs_human",
                    reason="auditor_disagreed" if audit1.verdict == "disagree" else "auditor_budget_exhausted",
                    fact=initial_draft,
                    initial_draft=initial_draft,
                    initial_audit=audit1,
                    final_audit=audit1,
                    codex_conflicts=val1.codex_conflicts,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    auditor_tool_calls=auditor_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        # Stage 3: One bounded revision round + fresh re-audit
        rev_prompt = build_revision_prompt(entity_context, initial_draft, audit1.issues, audit1.notes)
        (
            revised_draft,
            val2,
            rev_new_vocab,
            rev_vocab_note,
            r_tools,
            r_repairs,
            r_usage,
            r_ms,
        ) = _run_drafter_stage(
            stage_name="revision",
            entity_key=entity_key,
            user_prompt=rev_prompt,
            system_prompt=drafter_sys,
            llm=llm,
            assembly=assembly,
            extract_by_key=extract_by_key,
            facts_by_key=facts_by_key,
            catalog_cards=catalog_cards,
            config=config,
            max_read_tools=config.max_revision_tools,
            trace=trace,
        )
        total_usage = total_usage.add(r_usage)
        stage_usage["revision"] = r_usage.to_dict()
        total_latency += r_ms
        d_tools += r_tools
        d_repairs += r_repairs

        if rev_new_vocab:
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="needs_human",
                    reason="needs_new_vocabulary",
                    fact=revised_draft or initial_draft,
                    initial_draft=initial_draft,
                    initial_audit=audit1,
                    final_audit=audit1,
                    revised=True,
                    needs_new_vocabulary=True,
                    vocabulary_gap_note=rev_vocab_note,
                    codex_conflicts=val2.codex_conflicts,
                    validation_problems=val2.problems,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    auditor_tool_calls=auditor_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        if revised_draft is None or not val2.ok:
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="needs_human",
                    reason="revision_validation_failed",
                    fact=initial_draft,
                    initial_draft=initial_draft,
                    initial_audit=audit1,
                    final_audit=audit1,
                    revised=True,
                    codex_conflicts=val1.codex_conflicts,
                    validation_problems=val2.problems,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    auditor_tool_calls=auditor_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        audit2, a2_ms = _run_auditor_stage(
            stage_name="reaudit",
            entity_key=entity_key,
            entity_context=entity_context,
            candidate_fact=revised_draft,
            codex_conflicts=val2.codex_conflicts,
            system_prompt=auditor_sys,
            llm=llm,
            assembly=assembly,
            facts_by_key=facts_by_key,
            config=config,
            trace=trace,
        )
        total_usage = total_usage.add(audit2.usage)
        stage_usage["reaudit"] = audit2.usage.to_dict()
        total_latency += a2_ms
        auditor_tools += audit2.tool_calls

        if audit2.verdict == "agree" and not audit2.needs_new_vocabulary and not val2.codex_conflicts:
            merged = prepare_mergeable_entry(
                revised_draft,
                reviewed_by=f"{config.reviewed_by_tag} ({config.drafter_model}, revised)",
                audit_by=f"{config.reviewed_by_tag}-auditor ({config.auditor_model})",
                audit_verdict="agree",
                audit_notes=audit2.notes,
            )
            return (
                EntityPipelineResult(
                    entity_key=entity_key,
                    entity_type=entity_type,
                    class_name=class_name,
                    fingerprint=fingerprint,
                    outcome="auto_approved",
                    reason="revised_auditor_agreed",
                    fact=merged,
                    initial_draft=initial_draft,
                    initial_audit=audit1,
                    final_audit=audit2,
                    revised=True,
                    codex_conflicts=val2.codex_conflicts,
                    schema_repairs=d_repairs,
                    drafter_tool_calls=d_tools,
                    auditor_tool_calls=auditor_tools,
                    usage=total_usage,
                    stage_usage=stage_usage,
                    latency_ms=total_latency,
                    drafter_model=config.drafter_model,
                    auditor_model=config.auditor_model,
                ),
                trace,
            )

        if audit2.needs_new_vocabulary:
            reason = "needs_new_vocabulary"
        elif audit2.verdict == "agree" and val2.codex_conflicts:
            reason = "codex_conflict"
        else:
            reason = "auditor_disagreed_after_revision"
        return (
            EntityPipelineResult(
                entity_key=entity_key,
                entity_type=entity_type,
                class_name=class_name,
                fingerprint=fingerprint,
                outcome="needs_human",
                reason=reason,
                fact=revised_draft,
                initial_draft=initial_draft,
                initial_audit=audit1,
                final_audit=audit2,
                revised=True,
                needs_new_vocabulary=audit2.needs_new_vocabulary,
                vocabulary_gap_note=audit2.notes if audit2.needs_new_vocabulary else "",
                codex_conflicts=val2.codex_conflicts,
                schema_repairs=d_repairs,
                drafter_tool_calls=d_tools,
                auditor_tool_calls=auditor_tools,
                usage=total_usage,
                stage_usage=stage_usage,
                latency_ms=total_latency,
                drafter_model=config.drafter_model,
                auditor_model=config.auditor_model,
            ),
            trace,
        )
    except Exception as exc:
        trace.append({"stage": "pipeline", "event": "llm_error", "error": repr(exc)})
        return (
            EntityPipelineResult(
                entity_key=entity_key,
                entity_type=entity_type,
                class_name=class_name,
                fingerprint=fingerprint,
                outcome="needs_human",
                reason="llm_error",
                fact=None,
                initial_draft=None,
                initial_audit=None,
                final_audit=None,
                validation_problems=[str(exc)],
                usage=total_usage,
                stage_usage=stage_usage,
                latency_ms=total_latency,
                drafter_model=config.drafter_model,
                auditor_model=config.auditor_model,
            ),
            trace,
        )


def write_review_queue_markdown(results: Sequence[EntityPipelineResult], out_path: Path) -> None:
    """Render a concise human adjudication queue for all needs_human entries."""
    pending = [r for r in results if r.outcome != "auto_approved"]
    lines = [
        "# Fact Pipeline Human Review Queue",
        "",
        f"- Total processed: {len(results)}",
        f"- Auto-approved: {len(results) - len(pending)}",
        f"- Needs human review: {len(pending)}",
        "",
    ]
    if not pending:
        lines.append("All entries passed validation and independent audit.")
    for item in pending:
        lines.append(f"## `{item.entity_key}` (`{item.class_name}`)")
        lines.append(f"- Reason: `{item.reason}` | Revised: `{item.revised}` | Cost: `${item.usage.cost_usd:.5f}`")
        if item.needs_new_vocabulary and item.vocabulary_gap_note:
            lines.append(f"- Vocabulary gap: {item.vocabulary_gap_note}")
        if item.validation_problems:
            lines.append("- Validation problems:")
            for prob in item.validation_problems:
                lines.append(f"  - {prob}")
        if item.codex_conflicts:
            lines.append("- Codex conflicts:")
            for c in item.codex_conflicts:
                lines.append(f"  - {c}")
        if item.final_audit and item.final_audit.issues:
            lines.append("- Auditor issues:")
            for issue in item.final_audit.issues:
                lines.append(f"  - {issue}")
        if item.final_audit and item.final_audit.notes:
            lines.append(f"- Auditor notes: {item.final_audit.notes}")
        if item.fact:
            lines.append("```json")
            lines.append(json.dumps(item.fact, ensure_ascii=False, indent=1))
            lines.append("```")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))


def run_batch(
    entity_keys: Sequence[str],
    *,
    extract_by_key: Mapping[str, Mapping[str, Any]],
    facts_by_key: Mapping[str, Mapping[str, Any]],
    catalog_by_type: Mapping[str, Mapping[str, Mapping[str, Any]]],
    assembly: AssemblyProvider,
    llm: LLMClient,
    out_dir: Path,
    config: PipelineConfig = PipelineConfig(),
    workers: int = 1,
    force: bool = False,
    progress_cb: Optional[Callable[[EntityPipelineResult, int, int], None]] = None,
) -> List[EntityPipelineResult]:
    """Run the pipeline over a batch of entities with checkpoint/resume and summary outputs."""
    out_dir = Path(out_dir)
    results_dir = out_dir / "results"
    traces_dir = out_dir / "traces"
    results_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    drafter_sys = build_drafter_system_prompt()
    auditor_sys = build_auditor_system_prompt()

    cached_results: Dict[str, EntityPipelineResult] = {}
    todo: List[str] = []
    for key in entity_keys:
        res_file = results_dir / f"{_safe_key(key)}.json"
        if not force and res_file.is_file():
            loaded = EntityPipelineResult.from_dict(json.loads(res_file.read_text(encoding="utf-8")))
            expected_fp = extract_by_key.get(key, {}).get("fingerprint")
            if loaded.fingerprint == expected_fp and loaded.reason != "llm_error":
                cached_results[key] = loaded
                continue
        todo.append(key)

    def _work(key: str) -> EntityPipelineResult:
        res, trace = process_entity(
            key,
            extract_by_key=extract_by_key,
            facts_by_key=facts_by_key,
            catalog_by_type=catalog_by_type,
            assembly=assembly,
            llm=llm,
            config=config,
            drafter_system=drafter_sys,
            auditor_system=auditor_sys,
        )
        res_file = results_dir / f"{_safe_key(key)}.json"
        res_file.write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        trace_file = traces_dir / f"{_safe_key(key)}.jsonl"
        trace_file.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace),
            encoding="utf-8",
        )
        return res

    total = len(entity_keys)
    done_count = len(cached_results)
    if workers <= 1:
        for key in todo:
            res = _work(key)
            cached_results[key] = res
            done_count += 1
            if progress_cb:
                progress_cb(res, done_count, total)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_work, key): key for key in todo}
            for fut in as_completed(futures):
                res = fut.result()
                cached_results[res.entity_key] = res
                done_count += 1
                if progress_cb:
                    progress_cb(res, done_count, total)

    ordered = [cached_results[k] for k in entity_keys]
    auto_approved = [r.fact for r in ordered if r.outcome == "auto_approved" and r.fact is not None]
    needs_human = [r.to_dict() for r in ordered if r.outcome != "auto_approved"]

    (out_dir / "auto_approved.json").write_text(
        json.dumps(auto_approved, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    (out_dir / "needs_human.json").write_text(
        json.dumps(needs_human, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    write_review_queue_markdown(ordered, out_dir / "review-queue.md")

    total_usage = LLMUsage()
    for r in ordered:
        total_usage = total_usage.add(r.usage)
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "drafter_model": config.drafter_model,
        "auditor_model": config.auditor_model,
        "skip_audit": config.skip_audit,
        "total": len(ordered),
        "auto_approved": len(auto_approved),
        "needs_human": len(needs_human),
        "revised": sum(1 for r in ordered if r.revised),
        "needs_new_vocabulary": sum(1 for r in ordered if r.needs_new_vocabulary),
        "usage": total_usage.to_dict(),
        "avg_cost_usd": round(total_usage.cost_usd / len(ordered), 6) if ordered else 0.0,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return ordered
