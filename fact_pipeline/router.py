"""Deterministic static IL complexity scorer and adaptive routing policy for fact_pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from fact_pipeline.llm import COMPARISON_MODEL

ROUTING_MODES = ("fixed", "auto")
COMPLEXITY_TIERS = ("simple", "medium", "hard")
_TIER_RANK: Dict[str, int] = {"simple": 0, "medium": 1, "hard": 2}

# Non-hook metadata/UI overrides that do not represent gameplay trigger hooks.
_NON_HOOK_METHODS = frozenset(
    {
        ".ctor",
        ".cctor",
        "OnPlay",
        "OnUpgrade",
        "OnDowngrade",
        "ShouldGlowGoldInternal",
        "ShouldGlowRedInternal",
        "DoActivateVisuals",
        "CreateVisuals",
        "Flash",
    }
)

# Lifecycle hooks that span outside a single combat or mutate run-level state.
_OUT_OF_COMBAT_HOOK_PREFIXES = (
    "AfterObtained",
    "AfterRemoved",
    "AfterCombatEnd",
    "AfterRoomEntered",
    "BeforeRoomEntered",
    "AfterRestSite",
    "TryModifyRestSite",
    "ModifyRestSite",
    "AfterMapGenerated",
    "BeforeMapGenerated",
    "TryModifyRewards",
    "ModifyRewards",
    "TryModifyCardReward",
    "ModifyCardReward",
    "ModifyMerchant",
    "TryModifyMerchant",
    "TryModifyCardBeingAddedToDeck",
    "AfterCardAddedToDeck",
    "AfterActEntered",
    "BeforeActEntered",
    "AfterCampfire",
    "AfterShop",
    "AfterTreasure",
    "AfterEvent",
    "TryModifyPotion",
    "ModifyPotion",
    "ModifyMaxHp",
)

_BRANCH_OPCODE_RE = re.compile(
    r"^\s*(?:brtrue|brfalse|beq|bne|bgt|bge|blt|ble|switch)(?:\.[a-z0-9.]+)?\b",
    re.MULTILINE,
)
_STATE_MACHINE_TYPE_RE = re.compile(r"^type\s+.*<[^>]+>d__\d+", re.MULTILINE)
_CLOSURE_TYPE_RE = re.compile(r"^type\s+.*<>c(?:__DisplayClass[^\s]*)?", re.MULTILINE)
_PERSISTENT_IL_RE = re.compile(
    r"\b(?:SavedProperty|InternalData|ShouldResetBetweenCombats|RunState|CardPile::Deck|"
    r"RemoveCardFromDeck|AddCardToDeck|PlayerCmd::GainMaxHp|PlayerCmd::GainGold|"
    r"PotionCmd::|RelicCmd::Obtain|CardCmd::Upgrade)\b"
)
_INSTANCE_STFLD_RE = re.compile(r"^\s*stfld\s+[^\n]*::_(?!<)", re.MULTILINE)


@dataclass(frozen=True)
class ComplexitySignals:
    """Static structural signals extracted from sts2-extract metadata and IL dump."""

    entity_type: str
    hook_methods: Tuple[str, ...]
    out_of_combat_hooks: Tuple[str, ...]
    persistent_state_markers: Tuple[str, ...]
    state_machine_types: int
    closure_types: int
    branch_count: int
    il_lines: int
    branch_density: float
    call_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "hook_methods": list(self.hook_methods),
            "hook_count": len(self.hook_methods),
            "out_of_combat_hooks": list(self.out_of_combat_hooks),
            "persistent_state_markers": list(self.persistent_state_markers),
            "state_machine_types": self.state_machine_types,
            "closure_types": self.closure_types,
            "branch_count": self.branch_count,
            "il_lines": self.il_lines,
            "branch_density": round(self.branch_density, 2),
            "call_count": self.call_count,
        }


@dataclass(frozen=True)
class ComplexityAssessment:
    """Deterministic complexity score and tier for an entity prior to drafting."""

    score: int
    tier: str  # "simple" | "medium" | "hard"
    reasons: Tuple[str, ...]
    signals: ComplexitySignals

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "tier": self.tier,
            "reasons": list(self.reasons),
            "signals": self.signals.to_dict(),
        }


@dataclass(frozen=True)
class RouteDecision:
    """Concrete model, thinking, tool-budget, and auto-approval gate settings."""

    routing_mode: str  # "fixed" | "auto"
    stage: str  # "initial" | "revision"
    tier: str  # "simple" | "medium" | "hard"
    score: int
    drafter_model: str
    auditor_model: str
    drafter_thinking: bool
    auditor_thinking: bool
    max_drafter_tools: int
    max_revision_tools: int
    max_auditor_tools: int
    allow_auto_approve: bool
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_mode": self.routing_mode,
            "stage": self.stage,
            "tier": self.tier,
            "score": self.score,
            "drafter_model": self.drafter_model,
            "auditor_model": self.auditor_model,
            "drafter_thinking": self.drafter_thinking,
            "auditor_thinking": self.auditor_thinking,
            "max_drafter_tools": self.max_drafter_tools,
            "max_revision_tools": self.max_revision_tools,
            "max_auditor_tools": self.max_auditor_tools,
            "allow_auto_approve": self.allow_auto_approve,
            "reasons": list(self.reasons),
        }


def _is_hook_method(name: str) -> bool:
    if not name or name in _NON_HOOK_METHODS:
        return False
    if name.startswith("get_") or name.startswith("set_"):
        return False
    return True


def _is_out_of_combat_hook(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _OUT_OF_COMBAT_HOOK_PREFIXES)


def extract_complexity_signals(
    entity_type: str,
    extract_entry: Mapping[str, Any],
    il_dump: str,
) -> ComplexitySignals:
    """Scan extract metadata and IL dump for deterministic complexity signals."""
    overrides = [str(m) for m in (extract_entry.get("overrides") or ())]
    calls = [str(c) for c in (extract_entry.get("calls") or ())]

    hook_methods = tuple(sorted(m for m in overrides if _is_hook_method(m)))
    out_of_combat_hooks = tuple(sorted(m for m in overrides if _is_out_of_combat_hook(m)))

    persistent_hits = sorted(set(_PERSISTENT_IL_RE.findall(il_dump)))
    if _INSTANCE_STFLD_RE.search(il_dump):
        persistent_hits.append("instance_field_write")
    persistent_markers = tuple(sorted(set(persistent_hits)))

    state_machines = len(_STATE_MACHINE_TYPE_RE.findall(il_dump))
    closures = len(_CLOSURE_TYPE_RE.findall(il_dump))
    branches = len(_BRANCH_OPCODE_RE.findall(il_dump))
    il_lines = len(il_dump.splitlines()) if il_dump else 0
    density = (branches / max(1, il_lines)) * 100.0 if il_lines > 0 else 0.0

    return ComplexitySignals(
        entity_type=entity_type,
        hook_methods=hook_methods,
        out_of_combat_hooks=out_of_combat_hooks,
        persistent_state_markers=persistent_markers,
        state_machine_types=state_machines,
        closure_types=closures,
        branch_count=branches,
        il_lines=il_lines,
        branch_density=density,
        call_count=len(calls),
    )


def score_complexity(
    entity_type: str,
    extract_entry: Mapping[str, Any],
    il_dump: str,
) -> ComplexityAssessment:
    """Compute deterministic complexity score and tier ('simple' | 'medium' | 'hard')."""
    sig = extract_complexity_signals(entity_type, extract_entry, il_dump)
    score = 0
    reasons: List[str] = []

    # 1. Entity type prior (powers/relics/monsters have higher structural risk than simple cards).
    if entity_type == "monsters":
        score += 4
        reasons.append("entity_type=monsters (+4)")
    elif entity_type == "powers":
        # Trivial marker powers with zero hooks and tiny IL stay simple.
        if len(sig.hook_methods) == 0 and sig.il_lines < 30 and sig.branch_count == 0:
            score += 1
            reasons.append("entity_type=powers_trivial_marker (+1)")
        else:
            score += 3
            reasons.append("entity_type=powers (+3)")
    elif entity_type == "relics":
        score += 2
        reasons.append("entity_type=relics (+2)")

    # 2. Gameplay hook method count.
    n_hooks = len(sig.hook_methods)
    if n_hooks >= 3:
        score += 4
        reasons.append(f"hooks={n_hooks}>=3 (+4)")
    elif n_hooks == 2:
        score += 3
        reasons.append("hooks=2 (+3)")
    elif n_hooks == 1:
        score += 1
        reasons.append("hooks=1 (+1)")

    # 3. Cross-fight persistent state / out-of-combat hooks.
    n_ooc = len(sig.out_of_combat_hooks)
    if n_ooc >= 2:
        score += 3
        reasons.append(f"out_of_combat_hooks={n_ooc} (+3)")
    elif n_ooc == 1:
        score += 2
        reasons.append(f"out_of_combat_hook={sig.out_of_combat_hooks[0]} (+2)")

    if sig.persistent_state_markers:
        score += 2
        reasons.append(f"persistent_state={','.join(sig.persistent_state_markers[:3])} (+2)")

    # 4. Async state machines, lambda closures, and conditional branch density.
    if sig.closure_types >= 2:
        score += 3
        reasons.append(f"closures={sig.closure_types} (+3)")
    elif sig.closure_types == 1:
        score += 2
        reasons.append("closures=1 (+2)")

    if sig.state_machine_types >= 2:
        score += 2
        reasons.append(f"state_machines={sig.state_machine_types} (+2)")

    if sig.branch_count >= 12 or (sig.il_lines >= 80 and sig.branch_density >= 4.5):
        score += 2
        reasons.append(f"branches={sig.branch_count},density={sig.branch_density:.1f}% (+2)")
    elif sig.branch_count >= 8 or sig.il_lines >= 300:
        score += 1
        reasons.append(f"branches={sig.branch_count},il_lines={sig.il_lines} (+1)")

    if sig.call_count >= 6:
        score += 1
        reasons.append(f"calls={sig.call_count} (+1)")

    if score >= 6:
        tier = "hard"
    elif score >= 3:
        tier = "medium"
    else:
        tier = "simple"

    return ComplexityAssessment(
        score=score,
        tier=tier,
        reasons=tuple(reasons),
        signals=sig,
    )


class ComplexityRouter:
    """Adaptive stage router mapping static IL complexity + audit state to execution config."""

    def __init__(
        self,
        *,
        mode: str = "auto",
        hard_drafter_model: Optional[str] = COMPARISON_MODEL,
        hard_auditor_model: Optional[str] = None,
        auto_approve_max_tier: str = "medium",
        allow_revised_auto_approve: bool = False,
        medium_drafter_tools: int = 8,
        hard_drafter_tools: int = 10,
        medium_auditor_tools: int = 5,
        hard_auditor_tools: int = 6,
        escalated_revision_tools: int = 6,
    ) -> None:
        if mode not in ROUTING_MODES:
            raise ValueError(f"Unsupported routing mode {mode!r}; expected one of {ROUTING_MODES}")
        if auto_approve_max_tier not in ("none",) + COMPLEXITY_TIERS:
            raise ValueError(f"Unsupported auto_approve_max_tier {auto_approve_max_tier!r}")
        self.mode = mode
        self.hard_drafter_model = hard_drafter_model
        self.hard_auditor_model = hard_auditor_model
        self.auto_approve_max_tier = auto_approve_max_tier
        self.allow_revised_auto_approve = allow_revised_auto_approve
        self.medium_drafter_tools = medium_drafter_tools
        self.hard_drafter_tools = hard_drafter_tools
        self.medium_auditor_tools = medium_auditor_tools
        self.hard_auditor_tools = hard_auditor_tools
        self.escalated_revision_tools = escalated_revision_tools

    def assess(
        self,
        entity_type: str,
        extract_entry: Mapping[str, Any],
        il_dump: str,
    ) -> ComplexityAssessment:
        return score_complexity(entity_type, extract_entry, il_dump)

    def _tier_allows_auto_approve(self, tier: str) -> bool:
        if self.auto_approve_max_tier == "none":
            return False
        return _TIER_RANK[tier] <= _TIER_RANK[self.auto_approve_max_tier]

    def route_initial(
        self,
        assessment: ComplexityAssessment,
        *,
        drafter_model: str,
        auditor_model: str,
        thinking: bool,
        auditor_thinking: bool,
        max_drafter_tools: int,
        max_revision_tools: int,
        max_auditor_tools: int,
    ) -> RouteDecision:
        if self.mode == "fixed":
            return RouteDecision(
                routing_mode="fixed",
                stage="initial",
                tier=assessment.tier,
                score=assessment.score,
                drafter_model=drafter_model,
                auditor_model=auditor_model,
                drafter_thinking=thinking,
                auditor_thinking=auditor_thinking,
                max_drafter_tools=max_drafter_tools,
                max_revision_tools=max_revision_tools,
                max_auditor_tools=max_auditor_tools,
                allow_auto_approve=True,
                reasons=("fixed_routing",),
            )

        tier = assessment.tier
        if tier == "simple":
            return RouteDecision(
                routing_mode="auto",
                stage="initial",
                tier=tier,
                score=assessment.score,
                drafter_model=drafter_model,
                auditor_model=auditor_model,
                drafter_thinking=thinking,
                auditor_thinking=auditor_thinking,
                max_drafter_tools=max_drafter_tools,
                max_revision_tools=max_revision_tools,
                max_auditor_tools=max_auditor_tools,
                allow_auto_approve=self._tier_allows_auto_approve(tier),
                reasons=("auto_simple_fast_path",) + assessment.reasons,
            )

        if tier == "medium":
            return RouteDecision(
                routing_mode="auto",
                stage="initial",
                tier=tier,
                score=assessment.score,
                drafter_model=drafter_model,
                auditor_model=auditor_model,
                drafter_thinking=True,
                auditor_thinking=True,
                max_drafter_tools=max(max_drafter_tools, self.medium_drafter_tools),
                max_revision_tools=max(max_revision_tools, self.escalated_revision_tools),
                max_auditor_tools=max(max_auditor_tools, self.medium_auditor_tools),
                allow_auto_approve=self._tier_allows_auto_approve(tier),
                reasons=("auto_medium_enable_thinking_and_budget",) + assessment.reasons,
            )

        # tier == "hard": escalate thinking, model tier, tool budget, and tighten auto-approval.
        return RouteDecision(
            routing_mode="auto",
            stage="initial",
            tier=tier,
            score=assessment.score,
            drafter_model=self.hard_drafter_model or drafter_model,
            auditor_model=self.hard_auditor_model or auditor_model,
            drafter_thinking=True,
            auditor_thinking=True,
            max_drafter_tools=max(max_drafter_tools, self.hard_drafter_tools),
            max_revision_tools=max(max_revision_tools, self.escalated_revision_tools),
            max_auditor_tools=max(max_auditor_tools, self.hard_auditor_tools),
            allow_auto_approve=self._tier_allows_auto_approve(tier),
            reasons=("auto_hard_escalate_model_thinking_and_human_gate",) + assessment.reasons,
        )

    def route_revision(
        self,
        assessment: ComplexityAssessment,
        initial_route: RouteDecision,
        *,
        audit_issues: Sequence[str] = (),
    ) -> RouteDecision:
        """Escalate reasoning budget on Auditor-disagreed revision rounds."""
        if self.mode == "fixed":
            return RouteDecision(
                routing_mode="fixed",
                stage="revision",
                tier=initial_route.tier,
                score=initial_route.score,
                drafter_model=initial_route.drafter_model,
                auditor_model=initial_route.auditor_model,
                drafter_thinking=initial_route.drafter_thinking,
                auditor_thinking=initial_route.auditor_thinking,
                max_drafter_tools=initial_route.max_drafter_tools,
                max_revision_tools=initial_route.max_revision_tools,
                max_auditor_tools=initial_route.max_auditor_tools,
                allow_auto_approve=True,
                reasons=("fixed_routing_revision",),
            )

        escalate_model = (
            assessment.tier in ("medium", "hard") or len(audit_issues) >= 2
        ) and bool(self.hard_drafter_model)
        rev_model = (
            self.hard_drafter_model
            if escalate_model and self.hard_drafter_model
            else initial_route.drafter_model
        )
        allow_auto = self.allow_revised_auto_approve and self._tier_allows_auto_approve(
            assessment.tier
        )
        reasons = ["auto_revision_escalate_thinking"]
        if escalate_model:
            reasons.append(f"auto_revision_model={rev_model}")
        if not allow_auto:
            reasons.append("auto_revision_require_human_check")
        return RouteDecision(
            routing_mode="auto",
            stage="revision",
            tier=assessment.tier,
            score=assessment.score,
            drafter_model=rev_model,
            auditor_model=initial_route.auditor_model,
            drafter_thinking=True,
            auditor_thinking=True,
            max_drafter_tools=initial_route.max_drafter_tools,
            max_revision_tools=max(initial_route.max_revision_tools, self.escalated_revision_tools),
            max_auditor_tools=max(initial_route.max_auditor_tools, self.medium_auditor_tools),
            allow_auto_approve=allow_auto,
            reasons=tuple(reasons),
        )


def simulate_routing_policy(
    per_entity_rows: Sequence[Mapping[str, Any]],
    assessments_by_key: Mapping[str, ComplexityAssessment],
    *,
    auto_approve_max_tier: str = "medium",
    allow_revised_auto_approve: bool = False,
) -> Dict[str, Any]:
    """Reproducible offline risk & gate-threshold analysis over persisted eval metrics."""
    tier_stats: Dict[str, Dict[str, Any]] = {}
    for tier in COMPLEXITY_TIERS:
        tier_stats[tier] = {
            "total": 0,
            "baseline_auto": 0,
            "baseline_auto_core_ok": 0,
            "baseline_overall_core_ok": 0,
            "baseline_cost_usd": 0.0,
        }

    base_auto = 0
    base_auto_ok = 0
    gated_auto = 0
    gated_auto_ok = 0
    gated_human = 0
    gated_human_buggy = 0
    escalated_medium = 0
    escalated_hard = 0
    escalated_revisions = 0

    for row in per_entity_rows:
        key = str(row["entity_key"])
        assessment = assessments_by_key[key]
        tier = assessment.tier
        bucket = tier_stats[tier]
        bucket["total"] += 1
        core_ok = bool(row.get("final_core_match"))
        was_auto = row.get("outcome") == "auto_approved"
        was_revised = bool(row.get("revised"))
        bucket["baseline_overall_core_ok"] += int(core_ok)
        bucket["baseline_cost_usd"] += float(row.get("cost_usd", 0.0))
        if tier == "medium":
            escalated_medium += 1
        elif tier == "hard":
            escalated_hard += 1
        if was_revised:
            escalated_revisions += 1

        if was_auto:
            bucket["baseline_auto"] += 1
            bucket["baseline_auto_core_ok"] += int(core_ok)
            base_auto += 1
            base_auto_ok += int(core_ok)

        tier_ok = (
            auto_approve_max_tier != "none"
            and _TIER_RANK[tier] <= _TIER_RANK[auto_approve_max_tier]
        )
        rev_ok = allow_revised_auto_approve or not was_revised
        if was_auto and tier_ok and rev_ok:
            gated_auto += 1
            gated_auto_ok += int(core_ok)
        else:
            gated_human += 1
            if not core_ok:
                gated_human_buggy += 1

    for tier, bucket in tier_stats.items():
        ba = bucket["baseline_auto"]
        tot = bucket["total"]
        bucket["baseline_auto_core_precision"] = (
            round(bucket["baseline_auto_core_ok"] / ba, 4) if ba else None
        )
        bucket["baseline_overall_core_accuracy"] = (
            round(bucket["baseline_overall_core_ok"] / tot, 4) if tot else None
        )
        bucket["baseline_cost_usd"] = round(bucket["baseline_cost_usd"], 4)

    total = len(per_entity_rows)
    return {
        "total": total,
        "policy": {
            "auto_approve_max_tier": auto_approve_max_tier,
            "allow_revised_auto_approve": allow_revised_auto_approve,
        },
        "by_tier": tier_stats,
        "routing_volume": {
            "simple": tier_stats["simple"]["total"],
            "medium": escalated_medium,
            "hard": escalated_hard,
            "revisions_escalated": escalated_revisions,
        },
        "baseline": {
            "auto_approved": base_auto,
            "auto_approved_rate": round(base_auto / total, 4) if total else 0.0,
            "auto_core_ok": base_auto_ok,
            "auto_core_precision": round(base_auto_ok / base_auto, 4) if base_auto else None,
        },
        "gated": {
            "auto_approved": gated_auto,
            "auto_approved_rate": round(gated_auto / total, 4) if total else 0.0,
            "auto_core_ok": gated_auto_ok,
            "auto_core_precision": round(gated_auto_ok / gated_auto, 4) if gated_auto else None,
            "needs_human": gated_human,
            "needs_human_bug_rate": round(gated_human_buggy / gated_human, 4) if gated_human else None,
            "bugs_intercepted_vs_baseline": (base_auto - base_auto_ok) - (gated_auto - gated_auto_ok),
            "good_diverted_vs_baseline": base_auto_ok - gated_auto_ok,
        },
    }
