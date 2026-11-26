"""Self-contained, zero-dependency Chinese HTML post-run review report generator.

Renders any finished single-player ``.run`` file into one offline HTML file
with inline CSS (parchment light & obsidian dark themes), inline 6-dimension
capability SVG charts from ``review.tools.deck_trend``, Tier-1/Tier-2
counterfactual ``what_if`` cards with honest disclosure of unscored effects,
and a complete floor-by-floor decision ledger.

CLI usage:
    python -m review.html_report <run_path> [-o out.html]
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from advisor.deck_fitness import DIMENSION_LABELS, DIMENSIONS
from review import tools
from review.engine import KNOWLEDGE_PATH, ROOT_DIR, Engine
from review.run_review import ReviewError, RunReview, review_file

_DEFAULT_ENGINE_LOCK = threading.Lock()
_DEFAULT_ENGINE: Optional[Engine] = None

# Canonical display order for the 6 capability charts.
CHART_DIMENSIONS: Tuple[str, ...] = (
    "frontload",
    "scaling",
    "aoe",
    "defense",
    "cycle",
    "energy",
)

CHART_DIM_SUBTITLES: Mapping[str, str] = {
    "frontload": "前期输出 · frontload",
    "scaling": "成长上限 · scaling",
    "aoe": "群体伤害 · aoe",
    "defense": "生存防御 · defense",
    "cycle": "过牌循环 · cycle",
    "energy": "能量运转 · energy",
}

CHARACTER_NAMES_FALLBACK: Mapping[str, str] = {
    "IRONCLAD": "铁甲战士",
    "SILENT": "静默猎手",
    "DEFECT": "故障机器人",
    "NECROBINDER": "亡灵契约师",
    "REGENT": "储君",
    "FOOL": "愚者",
}

KIND_LABELS: Mapping[str, str] = {
    "card_reward": "选牌",
    "shop": "商店",
    "rest": "篝火",
}

KIND_SORT_ORDER: Mapping[str, int] = {
    "card_reward": 0,
    "shop": 1,
    "rest": 2,
}

CONFIDENCE_LABELS: Mapping[str, str] = {
    "high": "高把握",
    "medium": "中把握",
    "low": "低把握",
}

FACTOR_LABELS: Mapping[str, str] = {
    "deck_fitness:frontload": "前期输出",
    "deck_fitness:aoe": "群伤",
    "deck_fitness:defense": "防御",
    "deck_fitness:scaling": "成长",
    "deck_fitness:cycle": "循环（过牌）",
    "deck_fitness:energy": "能量",
    "deck_fitness:synergy": "牌组配合",
    "deck_fitness:lookahead": "后续选牌空间",
    "merchant:deck": "牌组变化",
    "merchant:price": "金币机会成本",
    "merchant_leave": "保留金币",
    "campfire:upgrade": "升级牌组增益",
    "campfire:heal": "回血生存增益",
    "insufficient_data": "数据不足",
    "unmodeled_mechanic": "含未建模效果",
}

NOT_STRICT_LABELS: Mapping[str, str] = {
    "partly_scored": "有选项只计入部分效果",
    "low_confidence": "引擎把握较低",
    "small_gap": "分差较小",
    "choice_not_scored": "实际选择无法计分",
    "engine_no_pick": "引擎未给出首选",
    "choice_unknown": "记录未明确玩家选择",
}

SKIPPED_LABELS: Mapping[str, str] = {
    "no_boss": "本幕未到达 Boss，缺少对照目标",
    "unknown_boss": "Boss 不在当前数据目录中（旧版本）",
    "unknown_card": "含当前数据目录未收录的卡牌（旧版本）",
    "rest_option_not_modeled": "篝火使用了未建模的特殊选项",
}

RARITY_PRIORITY: Mapping[str, int] = {
    "rare": 3,
    "uncommon": 2,
    "common": 1,
    "basic": 0,
}

MAX_WHAT_IF_DECISIONS = 5
MAX_WHAT_IF_PROBES_PER_DECISION = 6
MAX_WHAT_IF_SHOWN = 4


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _get_default_engine() -> Engine:
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        if _DEFAULT_ENGINE is None:
            _DEFAULT_ENGINE = Engine()
        return _DEFAULT_ENGINE


def _read_current_game_version() -> Optional[str]:
    compat_path = ROOT_DIR / "packaging" / "compatibility.json"
    try:
        raw = json.loads(compat_path.read_text(encoding="utf-8"))
        version = str((raw.get("game") or {}).get("version") or "").strip()
        if not version:
            return None
        return version if version.startswith("v") else f"v{version}"
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _character_display_name(character_id: str) -> str:
    key = str(character_id or "").upper()
    try:
        data = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        for row in data.get("characters") or ():
            if str(row.get("id") or "").upper() == key and row.get("name"):
                return str(row["name"])
    except (OSError, ValueError, TypeError):
        pass
    return CHARACTER_NAMES_FALLBACK.get(key, key or "未知角色")


def _factor_label(code: Optional[str]) -> str:
    if not code:
        return "其他因子"
    return FACTOR_LABELS.get(code, code)


def _not_strict_text(reasons: Sequence[str]) -> str:
    if not reasons:
        return ""
    return "、".join(NOT_STRICT_LABELS.get(r, r) for r in reasons)


def _skipped_text(reason: str) -> str:
    head = str(reason or "").split(":", 1)[0]
    base = SKIPPED_LABELS.get(head, reason)
    if ":" in str(reason or "") and head in ("unknown_card", "unknown_boss", "rest_option_not_modeled"):
        detail = str(reason).split(":", 1)[1]
        return f"{base}（{detail}）"
    return base


def _confidence_label(confidence: Optional[str]) -> str:
    return CONFIDENCE_LABELS.get(str(confidence or "").lower(), str(confidence or "--"))


def _format_score(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def _format_signed(value: Optional[float]) -> str:
    if value is None:
        return "--"
    num = float(value)
    if abs(num) < 0.005:
        return "0.00"
    return f"{num:+.2f}"


def _option_name(record: Mapping[str, Any], option_id: Optional[str]) -> str:
    if option_id is None:
        return "未知"
    if option_id == "bought_unscored":
        count = int(record.get("bought_unscored") or 0)
        if count > 0:
            return f"已购买（{count} 项未计分商品）"
        return "已购买（未计分项）"
    for opt in record.get("options") or ():
        if opt.get("id") == option_id:
            return str(opt.get("name") or option_id)
    fallback = {
        "skip": "跳过",
        "leave": "离开商店",
        "removal": "删牌",
        "rest:HEAL": "休息",
        "rest:SMITH": "锻造",
    }
    return fallback.get(option_id, option_id)


def _faithful_player_choice_text(record: Mapping[str, Any]) -> str:
    """Render the player's actual choice without ever misreporting shop purchases."""
    kind = record.get("kind")
    chosen = record.get("chosen")
    if kind == "shop":
        bought_ids = list(record.get("bought") or ())
        unscored = int(record.get("bought_unscored") or 0)
        if not bought_ids:
            return _option_name(record, chosen or "leave")
        names = [_option_name(record, bid) for bid in bought_ids]
        if unscored > 0:
            names.append(f"{unscored} 项未计分商品")
        primary = _option_name(record, chosen)
        if len(names) == 1:
            return names[0]
        joined = "、".join(names)
        if chosen in bought_ids:
            return f"{primary}（共购：{joined}）"
        return f"已购买：{joined}"
    if kind == "rest":
        base = _option_name(record, chosen)
        upgraded = record.get("upgraded")
        if chosen == "rest:SMITH" and isinstance(upgraded, Mapping) and upgraded.get("name"):
            return f"{base}（{upgraded['name']}）"
        return base
    return _option_name(record, chosen)


def _decision_outcome_label(record: Mapping[str, Any]) -> Tuple[str, str]:
    """Return ``(badge_class, Chinese status text)`` for one scored decision."""
    tier = record.get("tier")
    kind = record.get("kind")
    chosen = record.get("chosen")
    engine_top = record.get("engine_top")
    if tier == 1:
        return "tier1", "值得想一想 (Tier-1)"
    if tier == 2:
        reasons = _not_strict_text(record.get("not_strict_because") or ())
        suffix = f"：{reasons}" if reasons else ""
        return "tier2", f"其他分歧 (Tier-2{suffix})"
    if tier == 0:
        if kind == "shop" and chosen != engine_top:
            if chosen == "bought_unscored":
                return "agree-soft", "一致（均选择消费，实际购买为未计分项）"
            return "agree-soft", "一致（均选择消费，具体购买项不同）"
        return "agree", "一致"
    reasons = _not_strict_text(record.get("not_strict_because") or ())
    suffix = f"（{reasons}）" if reasons else ""
    return "unjudged", f"未判定{suffix}"


def _agreement_breakdown(review: RunReview) -> Dict[str, Any]:
    """Compute honest overall and per-kind agreement metrics."""
    records = [d.record for d in review.decisions.values()]
    by_kind: Dict[str, Dict[str, int]] = {}
    for kind in ("card_reward", "rest", "shop"):
        subset = [r for r in records if r.get("kind") == kind]
        judged = [r for r in subset if r.get("tier") is not None]
        agree = [r for r in judged if r.get("tier") == 0]
        exact = [
            r for r in judged
            if r.get("chosen") is not None and r.get("chosen") == r.get("engine_top")
        ]
        info: Dict[str, int] = {
            "total": len(subset),
            "judged": len(judged),
            "agree": len(agree),
            "exact": len(exact),
        }
        if kind == "shop":
            buy_or_leave_judged = [r for r in subset if r.get("buy_or_leave_agree") is not None]
            buy_or_leave_agree = [r for r in buy_or_leave_judged if r.get("buy_or_leave_agree") is True]
            both_bought_diff = [
                r for r in buy_or_leave_agree
                if r.get("chosen") != "leave" and r.get("chosen") != r.get("engine_top")
            ]
            info["buy_or_leave_judged"] = len(buy_or_leave_judged)
            info["buy_or_leave_agree"] = len(buy_or_leave_agree)
            info["both_bought_different_item"] = len(both_bought_diff)
        by_kind[kind] = info

    overall_judged = sum(r.get("tier") is not None for r in records)
    overall_agree = sum(r.get("tier") == 0 for r in records if r.get("tier") is not None)
    return {
        "overall": {
            "agree": overall_agree,
            "judged": overall_judged,
            "rate": round(overall_agree / overall_judged, 3) if overall_judged else None,
        },
        "by_kind": by_kind,
    }


def _rate_text(agree: int, judged: int) -> str:
    if judged <= 0:
        return "--"
    pct = round(agree * 100.0 / judged)
    return f"{agree}/{judged} ({pct}%)"


# ------------------------------------------------------------------ SVG Charts

def _nice_axis_max(raw_max: float) -> float:
    if raw_max <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw_max))
    scaled = raw_max / magnitude
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if scaled <= step:
            return step * magnitude
    return 10.0 * magnitude


def _fmt_tick(value: float, dim: str) -> str:
    if dim in ("cycle", "energy") and abs(value - round(value)) > 0.05:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.0f}"
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _render_dimension_svg(
    dim: str,
    label: str,
    points: Sequence[Mapping[str, Any]],
) -> str:
    width = 360.0
    height = 178.0
    pad_l = 40.0
    pad_r = 14.0
    pad_t = 16.0
    pad_b = 28.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    if not points:
        return (
            f'<div class="chart-card" data-dim="{_esc(dim)}">'
            f'<div class="chart-head">'
            f'<h4>{_esc(CHART_DIM_SUBTITLES.get(dim, label))}</h4>'
            f'<span class="pill pill-muted">无选牌点</span>'
            f'</div>'
            f'<svg class="dim-svg" viewBox="0 0 {int(width)} {int(height)}" '
            f'role="img" aria-label="{_esc(label)}能力曲线">'
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
            f'class="svg-empty">本局无可评分选牌记录</text>'
            f'</svg></div>'
        )

    have_vals = [float(p["values"][dim]["have"]) for p in points]
    need_vals = [float(p["values"][dim]["need"]) for p in points]
    below_flags = [bool(p["values"][dim]["below_need"]) for p in points]
    floors = [int(p["floor"]) for p in points]
    acts = [int(p["act"]) for p in points]

    raw_max = max(max(have_vals), max(need_vals), 0.1)
    y_max = _nice_axis_max(raw_max * 1.12)
    n = len(points)

    def x_at(idx: int) -> float:
        if n == 1:
            return pad_l + plot_w / 2.0
        return pad_l + (plot_w * idx) / (n - 1)

    def y_at(val: float) -> float:
        clamped = max(0.0, min(y_max, val))
        return pad_t + plot_h * (1.0 - clamped / y_max)

    parts: List[str] = []
    # Horizontal gridlines + Y ticks (4 intervals).
    for step in range(5):
        frac = step / 4.0
        yv = y_max * frac
        yy = y_at(yv)
        parts.append(
            f'<line x1="{pad_l:.1f}" y1="{yy:.1f}" x2="{width - pad_r:.1f}" y2="{yy:.1f}" class="svg-grid"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6:.1f}" y="{yy + 3.5:.1f}" text-anchor="end" class="svg-axis">'
            f'{_esc(_fmt_tick(yv, dim))}</text>'
        )

    # Act boundary vertical markers.
    for idx in range(1, n):
        if acts[idx] != acts[idx - 1]:
            bx = (x_at(idx - 1) + x_at(idx)) / 2.0
            parts.append(
                f'<line x1="{bx:.1f}" y1="{pad_t:.1f}" x2="{bx:.1f}" y2="{pad_t + plot_h:.1f}" class="svg-act"/>'
            )
            parts.append(
                f'<text x="{bx + 3:.1f}" y="{pad_t + 10:.1f}" class="svg-act-label">Act {acts[idx]}</text>'
            )

    # Shortfall area polygons between adjacent points where below need.
    if n >= 2:
        for idx in range(n - 1):
            if below_flags[idx] or below_flags[idx + 1]:
                x0, x1 = x_at(idx), x_at(idx + 1)
                yh0, yh1 = y_at(have_vals[idx]), y_at(have_vals[idx + 1])
                yn0, yn1 = y_at(need_vals[idx]), y_at(need_vals[idx + 1])
                top0, bot0 = min(yn0, yh0), max(yn0, yh0)
                top1, bot1 = min(yn1, yh1), max(yn1, yh1)
                # Fill only the shortfall band where need > have.
                if need_vals[idx] > have_vals[idx] and need_vals[idx + 1] > have_vals[idx + 1]:
                    poly = f"{x0:.1f},{yn0:.1f} {x1:.1f},{yn1:.1f} {x1:.1f},{yh1:.1f} {x0:.1f},{yh0:.1f}"
                    parts.append(f'<polygon points="{poly}" class="svg-shortfall"/>')
                elif need_vals[idx] > have_vals[idx] or need_vals[idx + 1] > have_vals[idx + 1]:
                    poly = f"{x0:.1f},{top0:.1f} {x1:.1f},{top1:.1f} {x1:.1f},{bot1:.1f} {x0:.1f},{bot0:.1f}"
                    parts.append(f'<polygon points="{poly}" class="svg-shortfall-soft"/>')

    # Need & Have polylines.
    if n >= 2:
        need_pts = " ".join(f"{x_at(i):.1f},{y_at(need_vals[i]):.1f}" for i in range(n))
        have_pts = " ".join(f"{x_at(i):.1f},{y_at(have_vals[i]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{need_pts}" class="svg-need"/>')
        parts.append(f'<polyline points="{have_pts}" class="svg-have"/>')
    else:
        # Single data point: render subtle horizontal reference line for need.
        yn = y_at(need_vals[0])
        parts.append(
            f'<line x1="{pad_l + 24:.1f}" y1="{yn:.1f}" x2="{width - pad_r - 24:.1f}" y2="{yn:.1f}" class="svg-need"/>'
        )

    # Data markers with native browser <title> tooltips.
    for idx in range(n):
        cx = x_at(idx)
        cy_have = y_at(have_vals[idx])
        cy_need = y_at(need_vals[idx])
        status = "低于需求" if below_flags[idx] else "达标"
        tip = (
            f"第 {floors[idx]} 层 (第 {acts[idx]} 幕) · {label}: "
            f"供给 {have_vals[idx]} / 需求 {need_vals[idx]}（{status}）"
        )
        dot_cls = "svg-dot-below" if below_flags[idx] else "svg-dot-ok"
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy_need:.1f}" r="2.2" class="svg-dot-need">'
            f'<title>{_esc(tip)}</title></circle>'
        )
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy_have:.1f}" r="3.4" class="{dot_cls}">'
            f'<title>{_esc(tip)}</title></circle>'
        )

    # X-axis floor labels (pick up to 6 evenly spaced indices including first & last).
    if n <= 6:
        tick_indices = list(range(n))
    else:
        step = max(1, (n - 1) // 5)
        tick_indices = list(range(0, n, step))
        if tick_indices[-1] != n - 1:
            if (n - 1) - tick_indices[-1] < max(1, step // 2):
                tick_indices[-1] = n - 1
            else:
                tick_indices.append(n - 1)
    for idx in tick_indices:
        cx = x_at(idx)
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 8:.1f}" text-anchor="middle" class="svg-axis">F{floors[idx]}</text>'
        )

    below_count = sum(1 for flag in below_flags if flag)
    if below_count == 0:
        badge_cls = "pill pill-ok"
        badge_text = "全程达标"
    else:
        badge_cls = "pill pill-warn"
        badge_text = f"缺口 {below_count}/{n} 层"

    last_have = have_vals[-1]
    last_need = need_vals[-1]
    footer_note = f"末层 F{floors[-1]}：供给 {last_have} / 需求 {last_need}"

    return (
        f'<div class="chart-card" data-dim="{_esc(dim)}">'
        f'<div class="chart-head">'
        f'<div><h4>{_esc(CHART_DIM_SUBTITLES.get(dim, label))}</h4>'
        f'<div class="chart-sub">{_esc(footer_note)}</div></div>'
        f'<span class="{badge_cls}">{_esc(badge_text)}</span>'
        f'</div>'
        f'<svg class="dim-svg" viewBox="0 0 {int(width)} {int(height)}" '
        f'role="img" aria-label="{_esc(label)}供需随楼层演进曲线">'
        + "".join(parts)
        + "</svg></div>"
    )


def _render_trend_section(trend: Mapping[str, Any]) -> str:
    labels = trend.get("labels") or DIMENSION_LABELS
    points = list(trend.get("decisions") or ())
    dims = [d for d in CHART_DIMENSIONS if d in labels] or list(DIMENSIONS)
    cards = [_render_dimension_svg(dim, str(labels.get(dim, dim)), points) for dim in dims]
    return (
        '<section class="section" id="capability-trend">'
        '<div class="section-head">'
        '<div><h2>6 维能力随楼层演进</h2>'
        '<p class="section-desc">基于 <code>deck_trend</code> 在每次可评分选牌前的牌组实测产出（每回合供给）与本幕后续战斗需求对比。</p></div>'
        '<div class="legend">'
        '<span class="legend-item"><i class="sw sw-have"></i>当前供给 (have)</span>'
        '<span class="legend-item"><i class="sw sw-need"></i>战斗需求 (need)</span>'
        '<span class="legend-item"><i class="sw sw-below"></i>低于需求</span>'
        '<span class="legend-item"><i class="sw sw-ok"></i>达到需求</span>'
        '</div>'
        '</div>'
        '<div class="chart-grid">'
        + "".join(cards)
        + "</div></section>"
    )


# ------------------------------------------------------------ What-If Explorer

def _card_relevance_score(
    review: RunReview,
    card_id: str,
    top_factor: Optional[str],
) -> Tuple[int, int, str]:
    """Deterministic priority key for selecting informative earlier-offered cards."""
    facts = review.engine.catalog.card(card_id) or {}
    rarity_key = review.engine.rarity.get("cards", {}).get(card_id, "")
    rarity_rank = RARITY_PRIORITY.get(rarity_key, 0)
    match = 0
    if top_factor == "deck_fitness:defense" and (facts.get("block") or 0) > 0:
        match = 2
    elif top_factor == "deck_fitness:cycle" and (facts.get("cards_draw") or 0) > 0:
        match = 2
    elif top_factor == "deck_fitness:energy" and (facts.get("energy_gain") or 0) > 0:
        match = 2
    elif top_factor == "deck_fitness:aoe" and str(facts.get("target") or "").lower() in ("allenemies", "all_enemies"):
        match = 2
    elif top_factor == "deck_fitness:scaling" and str(facts.get("type") or "") in ("能力", "Power"):
        match = 2
    elif top_factor == "deck_fitness:frontload" and (facts.get("damage") or 0) >= 8:
        match = 2
    elif rarity_rank >= 2:
        match = 1
    return (match, rarity_rank, card_id)


def _select_what_if_probes(
    review: RunReview,
    decision_id: str,
    top_factor: Optional[str],
) -> List[Dict[str, str]]:
    """Choose up to ``MAX_WHAT_IF_PROBES_PER_DECISION`` real-run legal perturbations."""
    choices = tools.what_if_choices(review, decision_id)
    probes: List[Dict[str, str]] = []
    seen = set()

    def push(operation: str, card_id: str) -> None:
        key = (operation, card_id)
        if key in seen or len(probes) >= MAX_WHAT_IF_PROBES_PER_DECISION:
            return
        seen.add(key)
        probes.append({"operation": operation, "card": card_id})

    # 1. Earlier-offered cards ("若此前拿到某张核心牌"):
    #    pick up to 2 factor-matched cards + 1 highest-rarity card for diversity.
    add_pool = [str(entry["card"]) for entry in choices.get("add") or ()]
    add_by_factor = sorted(
        add_pool,
        key=lambda cid: _card_relevance_score(review, cid, top_factor),
        reverse=True,
    )
    add_by_rarity = sorted(
        add_pool,
        key=lambda cid: (
            RARITY_PRIORITY.get(review.engine.rarity.get("cards", {}).get(cid, ""), 0),
            cid,
        ),
        reverse=True,
    )
    for cid in add_by_factor[:2]:
        push("add", cid)
    for cid in add_by_rarity:
        if len(probes) >= 3:
            break
        push("add", cid)
    for cid in add_by_factor[2:]:
        if len(probes) >= 3:
            break
        push("add", cid)

    # 2. Unupgraded cards currently in deck ("若此前已升级某张牌") — up to 2.
    upgrade_candidates = sorted(
        (str(cid) for cid in choices.get("upgrade") or ()),
        key=lambda cid: (
            RARITY_PRIORITY.get(review.engine.rarity.get("cards", {}).get(cid, ""), 0),
            0 if cid.startswith(("STRIKE_", "DEFEND_")) else 1,
            cid,
        ),
        reverse=True,
    )
    for cid in upgrade_candidates[:2]:
        push("upgrade", cid)

    # 3. Basic starter removal ("若此前已删掉基础牌") — up to 1.
    remove_candidates = list(choices.get("remove") or ())
    basics = [c for c in remove_candidates if str(c).startswith(("STRIKE_", "DEFEND_"))]
    if basics:
        push("remove", basics[0])
    elif remove_candidates:
        push("remove", str(remove_candidates[0]))

    # Fill any remaining slots with extra earlier-offered cards if available.
    for cid in add_by_factor:
        if len(probes) >= MAX_WHAT_IF_PROBES_PER_DECISION:
            break
        push("add", cid)

    return probes


def _run_what_if_probes(
    review: RunReview,
    decision_id: str,
    top_factor: Optional[str],
) -> List[Dict[str, Any]]:
    decision = review.decision(decision_id)
    if decision.kind != "card_reward":
        return []
    probes = _select_what_if_probes(review, decision_id, top_factor)
    results: List[Dict[str, Any]] = []
    before_gap = decision.record.get("gap")
    for probe in probes:
        try:
            outcome = tools.what_if(review, decision_id, [probe])
        except ReviewError:
            continue
        after_gap = outcome.get("gap_after")
        gap_shrink = 0.0
        if before_gap is not None and after_gap is not None:
            gap_shrink = round(float(before_gap) - float(after_gap), 2)
        results.append({
            "changes": [probe],
            "outcome": outcome,
            "gap_shrink": gap_shrink,
        })

    def _sort_key(item: Mapping[str, Any]) -> Tuple[int, int, float, int]:
        out = item["outcome"]
        return (
            1 if out.get("agrees_with_player_after") else 0,
            1 if out.get("pick_changed") else 0,
            float(item["gap_shrink"]),
            -len(item["changes"]),
        )

    results.sort(key=_sort_key, reverse=True)

    # If no single-card perturbation flipped the pick, test one combined
    # 2-change perturbation from the two strongest gap-shrinking moves.
    if results and not any(r["outcome"].get("pick_changed") for r in results):
        shrinkers = [r["changes"][0] for r in results if r["gap_shrink"] > 0]
        if len(shrinkers) >= 2:
            combo = [shrinkers[0], shrinkers[1]]
            try:
                combo_out = tools.what_if(review, decision_id, combo)
            except ReviewError:
                combo_out = None
            if combo_out is not None:
                after_gap = combo_out.get("gap_after")
                gap_shrink = 0.0
                if before_gap is not None and after_gap is not None:
                    gap_shrink = round(float(before_gap) - float(after_gap), 2)
                results.append({
                    "changes": combo,
                    "outcome": combo_out,
                    "gap_shrink": gap_shrink,
                })
                results.sort(key=_sort_key, reverse=True)

    return results[:MAX_WHAT_IF_SHOWN]


def _describe_probe_actions(changes: Sequence[Mapping[str, Any]]) -> str:
    phrases: List[str] = []
    for change in changes:
        op = change.get("operation")
        name = str(change.get("name") or change.get("card") or "")
        if op == "add":
            phrases.append(f"此前已拿到「{name}」")
        elif op == "upgrade":
            phrases.append(f"此时已升级「{name}」")
        elif op == "remove":
            phrases.append(f"此时已移除一张「{name}」")
        else:
            phrases.append(f"{op} {name}")
    if not phrases:
        return "未指定改动"
    return "若" + "且".join(phrases)


# ----------------------------------------------------------- Disagreement Cards

def _render_unscored_disclosure(record: Mapping[str, Any]) -> str:
    """Honest disclosure of every option that has unmodeled/unscored effects."""
    issues: List[str] = []
    for opt in record.get("options") or ():
        gaps = list(opt.get("gaps") or ())
        if not opt.get("fully_scored") or gaps:
            name = str(opt.get("name") or opt.get("id") or "")
            if gaps:
                short_gaps = ", ".join(str(g) for g in gaps[:2])
                if len(gaps) > 2:
                    short_gaps += f" 等 {len(gaps)} 项"
                issues.append(f"<li><b>{_esc(name)}</b>：存在未建模/未计分效果（<code>{_esc(short_gaps)}</code>），当前分值仅反映已计入部分。</li>")
            else:
                issues.append(f"<li><b>{_esc(name)}</b>：部分效果未计入评分，当前分值仅为下限。</li>")
    bought_unscored = int(record.get("bought_unscored") or 0)
    if record.get("kind") == "shop" and bought_unscored > 0:
        issues.append(
            f"<li><b>玩家实际购买</b>：包含 {bought_unscored} 项当前引擎无法定价/评分的商品（如药水或旧版未收录项），未强行估算分值。</li>"
        )
    if not issues:
        return ""
    return (
        '<div class="disclosure">'
        '<div class="disclosure-title">未计分效果诚实披露</div>'
        '<ul>' + "".join(issues) + '</ul>'
        '</div>'
    )


def _render_card_glossary(review: RunReview, record: Mapping[str, Any], extra_card_ids: Iterable[str] = ()) -> str:
    seen_cards = set()
    items: List[str] = []
    candidate_ids: List[str] = []
    for opt in record.get("options") or ():
        if opt.get("kind") == "card" and opt.get("entity_id"):
            candidate_ids.append(str(opt["entity_id"]))
    for cid in extra_card_ids:
        if cid:
            candidate_ids.append(str(cid))

    for cid in candidate_ids:
        if cid in seen_cards:
            continue
        seen_cards.add(cid)
        facts = review.engine.catalog.card(cid)
        if not facts:
            continue
        meta_parts: List[str] = []
        if facts.get("cost") is not None:
            meta_parts.append(f"{facts['cost']} 费")
        if facts.get("star_cost"):
            meta_parts.append(f"耗 {facts['star_cost']} 星")
        if facts.get("type"):
            meta_parts.append(str(facts["type"]))
        if facts.get("rarity"):
            meta_parts.append(str(facts["rarity"]))
        meta_str = " · ".join(meta_parts)
        desc = str(facts.get("description") or "").strip()
        items.append(
            f'<li><b>{_esc(facts.get("name") or cid)}</b>'
            f'<span class="gmeta">{_esc(meta_str)}</span>'
            f'<span class="gdesc">{_esc(desc)}</span></li>'
        )
    if not items:
        return ""
    return '<ul class="glossary">' + "".join(items) + "</ul>"


def _render_disagreement_card(
    review: RunReview,
    decision_id: str,
    index: int,
    *,
    tier_label: str,
    run_what_if: bool,
) -> str:
    decision = review.decision(decision_id)
    record = decision.record
    kind = str(record.get("kind") or "")
    kind_cn = KIND_LABELS.get(kind, kind)
    chosen_id = record.get("chosen")
    top_id = record.get("engine_top")
    chosen_text = _faithful_player_choice_text(record)
    top_text = _option_name(record, top_id)
    gap = record.get("gap")
    confidence = _confidence_label(record.get("confidence"))
    deck_snapshot = tools.deck_at(review, decision_id)

    # Compare options factor-by-factor when both options exist in the record.
    option_ids = {str(o.get("id")) for o in record.get("options") or ()}
    comparison: Optional[Dict[str, Any]] = None
    if top_id in option_ids and chosen_id in option_ids and top_id != chosen_id:
        try:
            comparison = tools.compare_options(review, decision_id, str(top_id), str(chosen_id))
        except ReviewError:
            comparison = None

    top_factor_code: Optional[str] = None
    factor_rows_html = ""
    if comparison and comparison.get("factors"):
        meaningful = [f for f in comparison["factors"] if abs(float(f.get("difference") or 0.0)) >= 0.05]
        shown_factors = meaningful[:5] if meaningful else comparison["factors"][:3]
        if shown_factors:
            top_factor_code = str(shown_factors[0].get("factor") or "")
        max_abs = max((abs(float(f.get("difference") or 0.0)) for f in shown_factors), default=1.0) or 1.0
        rows = []
        for f in shown_factors:
            diff = float(f.get("difference") or 0.0)
            pct = min(100.0, max(6.0, abs(diff) / max_abs * 100.0))
            bar_cls = "bar-pos" if diff >= 0 else "bar-neg"
            msg = f.get("a_message") or f.get("b_message") or ""
            rows.append(
                '<li>'
                '<div class="factor-top">'
                f'<span class="tag">{_esc(_factor_label(f.get("factor")))}</span>'
                f'<span class="factor-nums">首选 {_esc(_format_signed(f.get("a")))} vs 实选 {_esc(_format_signed(f.get("b")))} '
                f'<b>（差值 {_esc(_format_signed(diff))}）</b></span>'
                '</div>'
                f'<div class="bar-track"><div class="bar-fill {bar_cls}" style="width:{pct:.1f}%"></div></div>'
                + (f'<div class="factor-msg">{_esc(msg)}</div>' if msg else "")
                + '</li>'
            )
        factor_rows_html = (
            '<div class="subblock">'
            '<h4>核心维度差异因子（不含策略偏好项）</h4>'
            '<ul class="factor-list">' + "".join(rows) + '</ul>'
            '</div>'
        )

    # All candidate options table for full transparency.
    opt_rows = []
    for opt in record.get("options") or ():
        oid = opt.get("id")
        badges: List[str] = []
        if oid == top_id:
            badges.append('<span class="mini-tag mini-engine">引擎首选</span>')
        if oid == chosen_id or (kind == "shop" and oid in (record.get("bought") or ())):
            badges.append('<span class="mini-tag mini-player">玩家选择</span>')
        if not opt.get("fully_scored"):
            badges.append('<span class="mini-tag mini-warn">部分计分</span>')
        if opt.get("eligible") is False:
            badges.append('<span class="mini-tag mini-muted">不可用/金币不足</span>')
        opt_rows.append(
            f'<tr>'
            f'<td class="mono">{_esc(opt.get("rank") if opt.get("rank") is not None else "-")}</td>'
            f'<td>{_esc(opt.get("name") or oid)} {" ".join(badges)}</td>'
            f'<td class="mono num">{_esc(_format_score(opt.get("deck_score")))}</td>'
            f'<td class="mono num">{_esc(_format_score(opt.get("score")))}</td>'
            f'</tr>'
        )
    options_html = (
        '<div class="subblock">'
        '<h4>全部候选评分对照</h4>'
        '<div class="table-wrap"><table class="compact-table">'
        '<thead><tr><th>排名</th><th>候选</th><th class="num">牌组分 (deck_score)</th><th class="num">含偏好总分</th></tr></thead>'
        '<tbody>' + "".join(opt_rows) + '</tbody></table></div>'
        '</div>'
    )

    # Deterministic what_if counterfactual probes (card_reward only).
    what_if_html = ""
    extra_cards: List[str] = []
    if run_what_if and kind == "card_reward":
        probes = _run_what_if_probes(review, decision_id, top_factor_code)
        if probes:
            flipped_to_player = any(p["outcome"].get("agrees_with_player_after") for p in probes)
            flipped_any = any(p["outcome"].get("pick_changed") for p in probes)
            if flipped_to_player:
                summary_line = "微扰实验发现：改变本局真实出现过的 1–2 张牌后，引擎首选会翻转并同意你的选择。"
            elif flipped_any:
                summary_line = "微扰实验发现：改变本局真实出现过的 1–2 张牌后，引擎首选发生切换（属于临界局面）。"
            else:
                summary_line = "微扰实验显示：对当时牌组做微调后引擎首选保持不变，结论相对稳固。"
            probe_items = []
            for item in probes:
                out = item["outcome"]
                applied_changes = list(out.get("changes") or ())
                for ch in applied_changes:
                    if ch.get("card"):
                        extra_cards.append(str(ch["card"]))
                after_top = out.get("engine_top_after")
                after_name = _option_name(record, after_top)
                after_gap = out.get("gap_after")
                if out.get("agrees_with_player_after"):
                    res_badge = '<span class="pill pill-ok">翻转并同意玩家</span>'
                elif out.get("pick_changed"):
                    res_badge = f'<span class="pill pill-warn">首选变为「{_esc(after_name)}」</span>'
                else:
                    res_badge = '<span class="pill pill-muted">首选不变</span>'
                gap_str = f"分差 {_format_score(gap)} → {_format_score(after_gap)}" if gap is not None else f"新分差 {_format_score(after_gap)}"
                probe_items.append(
                    f'<li><span class="probe-cond">{_esc(_describe_probe_actions(applied_changes))}</span>'
                    f'{res_badge}'
                    f'<span class="probe-meta">新首选：{_esc(after_name)} · {_esc(gap_str)}</span></li>'
                )
            what_if_html = (
                '<div class="subblock whatif-box">'
                '<h4>反事实微扰实验（<code>what_if</code> · 仅限本局真实牌池）</h4>'
                f'<p class="whatif-summary">{_esc(summary_line)}</p>'
                '<ul class="probe-list">' + "".join(probe_items) + '</ul>'
                '</div>'
            )

    not_strict_html = ""
    reasons = record.get("not_strict_because") or ()
    if reasons:
        not_strict_html = (
            f'<div class="not-strict-note">未进入 Tier-1 的原因：{_esc(_not_strict_text(reasons))}</div>'
        )

    gap_display = f"+{gap:.2f}" if gap is not None else "--"
    headline = (
        f"引擎更看好「{top_text}」（比「{chosen_text}」高 {gap:.2f} 分）"
        if gap is not None
        else f"引擎首选「{top_text}」，你选择了「{chosen_text}」"
    )
    if top_factor_code:
        headline += f" · 主因：{_factor_label(top_factor_code)}"

    deck_pills = " ".join(
        f'<span class="deck-chip">{_esc(c["name"])}'
        + ("+" if int(c.get("upgrades") or 0) > 0 else "")
        + (f"×{c['count']}" if int(c.get("count") or 1) > 1 else "")
        + "</span>"
        for c in deck_snapshot.get("deck") or ()
    )
    relic_pills = " ".join(
        f'<span class="deck-chip relic-chip">{_esc(r["name"])}</span>'
        for r in deck_snapshot.get("relics") or ()
    )

    tier_cls = "card-tier1" if record.get("tier") == 1 else "card-tier2"
    return (
        f'<article class="decision-card {tier_cls}" id="decision-{_esc(decision_id)}" '
        f'data-tier="{_esc(record.get("tier"))}" data-kind="{_esc(kind)}">'
        '<div class="card-meta">'
        f'<span class="pill {"pill-accent" if record.get("tier") == 1 else "pill-muted"}">{_esc(tier_label)}</span>'
        f'<span>第 {int(record["act"])} 幕 · 第 {int(record["floor"])} 层 · {_esc(kind_cn)}</span>'
        f'<span>生命 {int(record["hp"])}/{int(record["max_hp"])} · 金币 {int(record["gold"])} · 牌组 {int(deck_snapshot["deck_size"])} 张</span>'
        f'<span class="pill pill-outline">{_esc(confidence)}</span>'
        '</div>'
        f'<h3>{index}. {_esc(headline)}</h3>'
        '<div class="vs-strip">'
        f'<div class="vs-box vs-player"><div class="vs-label">玩家实际选择</div><div class="vs-val">{_esc(chosen_text)}</div></div>'
        f'<div class="vs-gap"><div class="vs-label">牌组分差</div><div class="vs-num mono">{_esc(gap_display)}</div></div>'
        f'<div class="vs-box vs-engine"><div class="vs-label">引擎首选</div><div class="vs-val">{_esc(top_text)}</div></div>'
        '</div>'
        + not_strict_html
        + _render_card_glossary(review, record, extra_cards)
        + factor_rows_html
        + what_if_html
        + _render_unscored_disclosure(record)
        + options_html
        + '<details class="snapshot-details"><summary>查看当时牌组与遗物快照</summary>'
        + f'<div class="snapshot-body"><div class="snapshot-group"><b>牌组（{int(deck_snapshot["deck_size"])} 张）：</b>{deck_pills}</div>'
        + (f'<div class="snapshot-group"><b>遗物：</b>{relic_pills}</div>' if relic_pills else "")
        + '</div></details>'
        + '</article>'
    )


# --------------------------------------------------------------- Full Ledger

def _render_ledger_section(review: RunReview) -> str:
    entries: List[Tuple[int, int, str, str, Any]] = []
    for decision in review.decisions.values():
        r = decision.record
        entries.append((
            int(r["floor"]),
            KIND_SORT_ORDER.get(str(r.get("kind")), 9),
            str(r["id"]),
            "scored",
            r,
        ))
    for item in review.skipped:
        entries.append((
            int(item["floor"]),
            KIND_SORT_ORDER.get(str(item.get("kind")), 9),
            str(item["id"]),
            "skipped",
            item,
        ))
    entries.sort(key=lambda e: (e[0], e[1], e[2]))

    rows_html: List[str] = []
    for floor, _order, ident, entry_type, payload in entries:
        if entry_type == "scored":
            r = payload
            kind = str(r.get("kind") or "")
            kind_cn = KIND_LABELS.get(kind, kind)
            player_str = _faithful_player_choice_text(r)
            top_str = _option_name(r, r.get("engine_top")) if r.get("engine_top") else "--"
            gap_str = _format_score(r.get("gap")) if r.get("gap") is not None else "--"
            badge_cls, status_text = _decision_outcome_label(r)
            tier = r.get("tier")
            filter_tier = "tier1" if tier == 1 else ("tier2" if tier == 2 else "other")
            state_str = f'Act {int(r["act"])} · {int(r["hp"])}/{int(r["max_hp"])} HP · {int(r["gold"])}G'
            anchor = f'<a href="#decision-{_esc(ident)}" class="jump-link">F{floor:02d}</a>' if tier in (1, 2) else f"F{floor:02d}"
            rows_html.append(
                f'<tr class="ledger-row row-{badge_cls}" data-kind="{_esc(kind)}" data-filter="{filter_tier}">'
                f'<td class="mono">{anchor}</td>'
                f'<td><span class="kind-badge kind-{_esc(kind)}">{_esc(kind_cn)}</span></td>'
                f'<td class="muted-cell">{_esc(state_str)}</td>'
                f'<td class="choice-cell">{_esc(player_str)}</td>'
                f'<td>{_esc(top_str)}</td>'
                f'<td class="mono num">{_esc(gap_str)}</td>'
                f'<td><span class="status-pill status-{badge_cls}">{_esc(status_text)}</span></td>'
                f'</tr>'
            )
        else:
            s = payload
            kind = str(s.get("kind") or "")
            kind_cn = KIND_LABELS.get(kind, kind)
            reason_cn = _skipped_text(str(s.get("reason") or ""))
            rows_html.append(
                f'<tr class="ledger-row row-skipped" data-kind="{_esc(kind)}" data-filter="skipped">'
                f'<td class="mono">F{floor:02d}</td>'
                f'<td><span class="kind-badge kind-{_esc(kind)}">{_esc(kind_cn)}</span></td>'
                f'<td class="muted-cell">--</td>'
                f'<td class="muted-cell">--</td>'
                f'<td class="muted-cell">--</td>'
                f'<td class="mono num muted-cell">--</td>'
                f'<td><span class="status-pill status-skipped">未评分：{_esc(reason_cn)}</span></td>'
                f'</tr>'
            )

    return (
        '<section class="section" id="decision-ledger">'
        '<div class="section-head">'
        '<div><h2>全楼层决策明细账本</h2>'
        '<p class="section-desc">按楼层完整记录选牌、商店与篝火的已评分及未评分决策。商店严格区分实际购买项与“买或离开”一致性。</p></div>'
        '<div class="filter-bar" role="group" aria-label="筛选决策">'
        '<button type="button" class="filter-btn active" data-mode="all">全部</button>'
        '<button type="button" class="filter-btn" data-mode="tier1">Tier-1 重点</button>'
        '<button type="button" class="filter-btn" data-mode="tier2">Tier-2 分歧</button>'
        '<button type="button" class="filter-btn" data-mode="card_reward">选牌</button>'
        '<button type="button" class="filter-btn" data-mode="shop">商店</button>'
        '<button type="button" class="filter-btn" data-mode="rest">篝火</button>'
        '<button type="button" class="filter-btn" data-mode="skipped">未评分</button>'
        '</div>'
        '</div>'
        '<div class="table-wrap"><table class="ledger-table">'
        '<thead><tr><th>楼层</th><th>类型</th><th>当时状态</th><th>玩家实际选择</th><th>引擎首选</th><th class="num">分差</th><th>一致性 / 判定</th></tr></thead>'
        '<tbody>' + "\n".join(rows_html) + '</tbody>'
        '</table></div>'
        '</section>'
    )


# --------------------------------------------------------------- CSS & Script

REPORT_CSS = """
:root {
  --bg: #f7f3eb;
  --bg-subtle: #efe9dd;
  --card: #fffdf9;
  --fg: #1f1b16;
  --muted: #685f55;
  --line: #e2d9c9;
  --accent: #9c4a1d;
  --accent-soft: #f5e6d7;
  --ok: #25683d;
  --ok-soft: #e3f2e8;
  --warn: #b05d14;
  --warn-soft: #fae9d6;
  --danger: #a63228;
  --danger-soft: #fbe5e2;
  --have: #225d7a;
  --need: #9a7b4f;
  --shadow: 0 2px 10px rgba(31, 27, 22, 0.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #151412;
    --bg-subtle: #1e1c19;
    --card: #211f1b;
    --fg: #eee8de;
    --muted: #a69e92;
    --line: #36322b;
    --accent: #e5985b;
    --accent-soft: #3a2a1d;
    --ok: #79c793;
    --ok-soft: #1d3325;
    --warn: #ebb063;
    --warn-soft: #3a2b18;
    --danger: #ec7f73;
    --danger-soft: #3b201d;
    --have: #66b8de;
    --need: #c9ab78;
    --shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
  }
}
:root[data-theme="dark"] {
  --bg: #151412;
  --bg-subtle: #1e1c19;
  --card: #211f1b;
  --fg: #eee8de;
  --muted: #a69e92;
  --line: #36322b;
  --accent: #e5985b;
  --accent-soft: #3a2a1d;
  --ok: #79c793;
  --ok-soft: #1d3325;
  --warn: #ebb063;
  --warn-soft: #3a2b18;
  --danger: #ec7f73;
  --danger-soft: #3b201d;
  --have: #66b8de;
  --need: #c9ab78;
  --shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 14.5px/1.65 system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
}
code, .mono { font-family: ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace; font-size: 0.92em; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 24px 18px 64px; }
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
.brand { font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.brand-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); display: inline-block; }
.theme-btn {
  border: 1px solid var(--line);
  background: var(--card);
  color: var(--fg);
  border-radius: 999px;
  padding: 5px 12px;
  font-size: 12.5px;
  cursor: pointer;
}
.theme-btn:hover { border-color: var(--accent); color: var(--accent); }
.hero {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 20px 22px;
  box-shadow: var(--shadow);
}
.hero-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.hero h1 { margin: 0; font-size: 24px; line-height: 1.3; letter-spacing: -0.01em; }
.hero-meta { margin-top: 6px; color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 8px 14px; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 12px; margin-top: 16px; }
.kpi {
  background: var(--bg-subtle);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 14px;
}
.kpi-label { font-size: 12px; color: var(--muted); }
.kpi-val { font-size: 20px; font-weight: 700; margin-top: 2px; line-height: 1.3; }
.kpi-sub { font-size: 12.5px; color: var(--muted); margin-top: 4px; }
.kpi-win { color: var(--ok); }
.kpi-loss { color: var(--danger); }
.banner {
  margin-top: 14px;
  padding: 10px 14px;
  border-radius: 8px;
  border-left: 3px solid var(--accent);
  background: var(--bg-subtle);
  font-size: 13px;
  color: var(--muted);
}
.banner-warn { border-left-color: var(--warn); background: var(--warn-soft); color: var(--fg); }
.section { margin-top: 30px; }
.section-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.section h2 { margin: 0; font-size: 18px; }
.section-desc { margin: 3px 0 0; font-size: 13px; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 10px 14px; font-size: 12.5px; color: var(--muted); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.sw { display: inline-block; width: 14px; height: 3px; border-radius: 2px; }
.sw-have { background: var(--have); }
.sw-need { border-top: 2px dashed var(--need); height: 0; }
.sw-below { width: 8px; height: 8px; border-radius: 50%; background: var(--danger); }
.sw-ok { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }
.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 14px; }
.chart-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 14px 10px;
  box-shadow: var(--shadow);
}
.chart-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 6px; }
.chart-head h4 { margin: 0; font-size: 14px; }
.chart-sub { font-size: 12px; color: var(--muted); }
.dim-svg { width: 100%; height: auto; display: block; }
.svg-grid { stroke: var(--line); stroke-width: 1; }
.svg-axis { fill: var(--muted); font-size: 10px; }
.svg-act { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; opacity: 0.65; }
.svg-act-label { fill: var(--muted); font-size: 9.5px; font-weight: 600; }
.svg-need { fill: none; stroke: var(--need); stroke-width: 1.8; stroke-dasharray: 5 4; }
.svg-have { fill: none; stroke: var(--have); stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
.svg-shortfall { fill: var(--danger); opacity: 0.12; }
.svg-shortfall-soft { fill: var(--warn); opacity: 0.08; }
.svg-dot-below { fill: var(--danger); stroke: var(--card); stroke-width: 1.2; }
.svg-dot-ok { fill: var(--ok); stroke: var(--card); stroke-width: 1.2; }
.svg-dot-need { fill: var(--need); opacity: 0.85; }
.svg-empty { fill: var(--muted); font-size: 12px; }
.pill {
  display: inline-block;
  border-radius: 999px;
  padding: 2px 9px;
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}
.pill-ok { background: var(--ok-soft); color: var(--ok); }
.pill-warn { background: var(--warn-soft); color: var(--warn); }
.pill-accent { background: var(--accent-soft); color: var(--accent); }
.pill-muted { background: var(--bg-subtle); color: var(--muted); }
.pill-outline { border: 1px solid var(--line); color: var(--muted); font-weight: 500; }
.decision-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: var(--shadow);
}
.card-tier1 { border-left: 4px solid var(--accent); }
.card-tier2 { border-left: 4px solid var(--line); }
.card-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; font-size: 12.5px; color: var(--muted); margin-bottom: 6px; }
.decision-card h3 { margin: 4px 0 12px; font-size: 16.5px; line-height: 1.4; }
.vs-strip { display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items: stretch; margin-bottom: 12px; }
@media (max-width: 640px) { .vs-strip { grid-template-columns: 1fr; } }
.vs-box, .vs-gap { background: var(--bg-subtle); border: 1px solid var(--line); border-radius: 9px; padding: 9px 12px; }
.vs-engine { border-color: var(--accent); background: var(--accent-soft); }
.vs-gap { display: flex; flex-direction: column; justify-content: center; align-items: center; min-width: 96px; }
.vs-label { font-size: 11.5px; color: var(--muted); }
.vs-val { font-size: 15px; font-weight: 700; margin-top: 2px; }
.vs-num { font-size: 17px; font-weight: 700; color: var(--accent); }
.not-strict-note {
  margin: 8px 0 10px;
  padding: 7px 10px;
  border-radius: 7px;
  background: var(--bg-subtle);
  font-size: 13px;
  color: var(--muted);
}
.glossary {
  list-style: none;
  margin: 10px 0;
  padding: 9px 12px;
  border: 1px dashed var(--line);
  border-radius: 8px;
  font-size: 13px;
}
.glossary li { margin: 3px 0; }
.glossary b { margin-right: 6px; }
.gmeta { color: var(--muted); font-size: 12px; margin-right: 8px; }
.gdesc { color: var(--muted); }
.subblock { margin-top: 12px; }
.subblock h4 { margin: 0 0 8px; font-size: 13.5px; color: var(--muted); font-weight: 600; }
.factor-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.factor-list li { background: var(--bg-subtle); border-radius: 8px; padding: 8px 10px; }
.factor-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 13px; }
.tag { display: inline-block; background: var(--accent-soft); color: var(--accent); border-radius: 5px; padding: 1px 7px; font-size: 12px; font-weight: 600; }
.factor-nums { font-size: 12.5px; color: var(--muted); }
.factor-nums b { color: var(--fg); }
.bar-track { height: 5px; background: var(--line); border-radius: 999px; overflow: hidden; margin: 5px 0 4px; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-pos { background: var(--accent); }
.bar-neg { background: var(--have); }
.factor-msg { font-size: 12.5px; color: var(--muted); }
.whatif-box { background: var(--bg-subtle); border: 1px solid var(--line); border-radius: 9px; padding: 11px 13px; }
.whatif-summary { margin: 0 0 8px; font-size: 13.5px; }
.probe-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.probe-list li { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; font-size: 13px; padding: 5px 8px; background: var(--card); border: 1px solid var(--line); border-radius: 7px; }
.probe-cond { font-weight: 600; }
.probe-meta { color: var(--muted); font-size: 12.5px; margin-left: auto; }
.disclosure {
  margin-top: 12px;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--warn-soft);
  border: 1px solid var(--warn);
  font-size: 13px;
}
.disclosure-title { font-weight: 700; color: var(--warn); margin-bottom: 4px; }
.disclosure ul { margin: 0; padding-left: 18px; }
.disclosure li { margin: 2px 0; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th, td { border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: middle; }
th { font-size: 12px; color: var(--muted); font-weight: 600; white-space: nowrap; }
.num { text-align: right; white-space: nowrap; }
.compact-table { font-size: 13px; }
.mini-tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px; margin-left: 4px; vertical-align: middle; }
.mini-engine { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.mini-player { background: var(--ok-soft); color: var(--ok); font-weight: 600; }
.mini-warn { background: var(--warn-soft); color: var(--warn); }
.mini-muted { background: var(--bg-subtle); color: var(--muted); }
.snapshot-details { margin-top: 10px; font-size: 13px; }
.snapshot-details summary { cursor: pointer; color: var(--muted); }
.snapshot-body { margin-top: 8px; padding: 10px 12px; background: var(--bg-subtle); border-radius: 8px; display: grid; gap: 8px; }
.deck-chip { display: inline-block; background: var(--card); border: 1px solid var(--line); border-radius: 5px; padding: 1px 7px; margin: 2px 4px 2px 0; font-size: 12px; }
.relic-chip { border-color: var(--accent); }
.filter-bar { display: flex; flex-wrap: wrap; gap: 6px; }
.filter-btn {
  border: 1px solid var(--line);
  background: var(--card);
  color: var(--fg);
  border-radius: 999px;
  padding: 4px 11px;
  font-size: 12.5px;
  cursor: pointer;
}
.filter-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.ledger-table { background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.row-tier1 td { background: var(--accent-soft); font-weight: 600; }
.row-skipped td { opacity: 0.78; }
.muted-cell { color: var(--muted); font-size: 12.5px; }
.choice-cell { font-weight: 600; }
.jump-link { color: var(--accent); text-decoration: none; font-weight: 700; }
.jump-link:hover { text-decoration: underline; }
.kind-badge { display: inline-block; font-size: 11.5px; padding: 1px 7px; border-radius: 5px; font-weight: 600; background: var(--bg-subtle); }
.kind-card_reward { background: var(--accent-soft); color: var(--accent); }
.kind-shop { background: var(--warn-soft); color: var(--warn); }
.kind-rest { background: var(--ok-soft); color: var(--ok); }
.status-pill { display: inline-block; font-size: 12px; padding: 2px 7px; border-radius: 5px; }
.status-tier1 { background: var(--accent); color: #fff; font-weight: 600; }
.status-tier2 { background: var(--warn-soft); color: var(--warn); }
.status-agree { background: var(--ok-soft); color: var(--ok); }
.status-agree-soft { background: var(--bg-subtle); color: var(--ok); border: 1px solid var(--line); }
.status-unjudged, .status-skipped { background: var(--bg-subtle); color: var(--muted); }
.empty-card { background: var(--card); border: 1px dashed var(--line); border-radius: 10px; padding: 16px; color: var(--muted); }
footer { margin-top: 36px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12.5px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
"""

REPORT_JS = """
(function() {
  var root = document.documentElement;
  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", function() {
      var cur = root.getAttribute("data-theme");
      if (!cur) {
        cur = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      }
      var next = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      themeBtn.textContent = next === "dark" ? "切换羊皮纸浅色" : "切换深色模式";
    });
  }
  var buttons = document.querySelectorAll(".filter-btn");
  var rows = document.querySelectorAll(".ledger-row");
  buttons.forEach(function(btn) {
    btn.addEventListener("click", function() {
      var mode = btn.getAttribute("data-mode");
      buttons.forEach(function(b) { b.classList.toggle("active", b === btn); });
      rows.forEach(function(row) {
        var kind = row.getAttribute("data-kind");
        var filt = row.getAttribute("data-filter");
        var show = (
          mode === "all" ||
          mode === filt ||
          mode === kind
        );
        row.style.display = show ? "" : "none";
      });
    });
  });
})();
"""


def render_review_html(
    review: RunReview,
    *,
    started_at: Optional[float] = None,
    elapsed_seconds: Optional[float] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Render an already-scored ``RunReview`` into a self-contained HTML string."""
    t0 = time.perf_counter() if started_at is None else started_at
    summary = review.summary()
    trend = tools.deck_trend(review)
    breakdown = _agreement_breakdown(review)
    char_cn = _character_display_name(summary["character"])
    ascension = summary.get("ascension")
    asc_text = f"进阶 {ascension}" if ascension is not None else "普通模式"
    build_id = summary.get("build_id") or "未知版本"
    win = bool(summary.get("win"))
    killed_by = summary.get("killed_by") or "未知遭遇"
    outcome_title = "胜利通关" if win else f"阵亡于 {killed_by}"
    outcome_cls = "kpi-win" if win else "kpi-loss"

    overall = breakdown["overall"]
    by_kind = breakdown["by_kind"]
    card_stat = by_kind["card_reward"]
    rest_stat = by_kind["rest"]
    shop_stat = by_kind["shop"]

    shop_note_parts = [f"商店（买或离开）：{_rate_text(shop_stat['buy_or_leave_agree'], shop_stat['buy_or_leave_judged'])}"]
    if shop_stat["both_bought_different_item"] > 0:
        shop_note_parts.append(f"其中 {shop_stat['both_bought_different_item']} 次同为消费但具体项不同")
    if shop_stat["judged"] > 0:
        shop_note_parts.append(f"精确首选 {shop_stat['exact']}/{shop_stat['judged']}")

    tier1_ids = list(summary.get("tier1") or ())
    tier2_ids = list(summary.get("tier2") or ())
    skipped_count = len(review.skipped)

    # Decide which disagreement cards receive deterministic what_if probes.
    what_if_budget = MAX_WHAT_IF_DECISIONS
    what_if_targets = set()
    for did in tier1_ids:
        if review.decision(did).kind == "card_reward" and len(what_if_targets) < what_if_budget:
            what_if_targets.add(did)
    if len(what_if_targets) < 3:
        for did in tier2_ids:
            if review.decision(did).kind == "card_reward" and len(what_if_targets) < 3:
                what_if_targets.add(did)

    tier1_cards = [
        _render_disagreement_card(
            review,
            did,
            idx + 1,
            tier_label="Tier-1 重点分歧",
            run_what_if=(did in what_if_targets),
        )
        for idx, did in enumerate(tier1_ids)
    ]
    tier2_cards = [
        _render_disagreement_card(
            review,
            did,
            idx + 1,
            tier_label="Tier-2 次要分歧",
            run_what_if=(did in what_if_targets),
        )
        for idx, did in enumerate(tier2_ids)
    ]

    current_build = _read_current_game_version()
    version_banner = ""
    if current_build and summary.get("build_id") and str(summary["build_id"]) != current_build:
        version_banner = (
            f'<div class="banner banner-warn">版本差异提示：本局记录来自 <b>{_esc(build_id)}</b>，'
            f'当前本地评分数据目录对应 <b>{_esc(current_build)}</b>，个别卡牌或遭遇数值可能与当时略有出入。</div>'
        )
    warnings_banner = ""
    if review.warnings:
        joined = "；".join(str(w) for w in review.warnings)
        warnings_banner = f'<div class="banner banner-warn">回放提示：{_esc(joined)}</div>'

    ts = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    total_ms = round(
        (elapsed_seconds if elapsed_seconds is not None else (time.perf_counter() - t0)) * 1000.0,
        1,
    )

    tier1_html = (
        '<section class="section" id="tier1-flags">'
        '<div class="section-head">'
        f'<div><h2>重点分歧反事实卡片 · Tier-1（{len(tier1_cards)}）</h2>'
        '<p class="section-desc">仅列出双方完整计分、引擎把握非低且纯牌组分差明显的关键抉择；措辞只呈现引擎视角，不替代玩家流派意图。</p></div>'
        '</div>'
        + ("".join(tier1_cards) if tier1_cards else '<div class="empty-card">本局未发现引擎高把握且分差明显的 Tier-1 分歧。</div>')
        + '</section>'
    )

    tier2_html = (
        '<section class="section" id="tier2-flags">'
        '<div class="section-head">'
        f'<div><h2>其他分歧附录 · Tier-2（{len(tier2_cards)}）</h2>'
        '<p class="section-desc">含分差较小、引擎把握较低或部分效果未建模的分歧，供参考并诚实标注限制原因。</p></div>'
        '</div>'
        + ("".join(tier2_cards) if tier2_cards else '<div class="empty-card">本局无额外的 Tier-2 分歧记录。</div>')
        + '</section>'
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>STS2 赛后复盘 · {_esc(char_cn)} · {_esc(summary['run_id'])}</title>\n"
        f"<style>{REPORT_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<div class="wrap">\n'
        '<div class="topbar">'
        '<div class="brand"><span class="brand-dot"></span><span>STS2 Guide · 本地确定性赛后复盘（零网络 / 只读）</span></div>'
        '<button type="button" id="theme-toggle" class="theme-btn">切换主题（羊皮纸 / 深色）</button>'
        "</div>\n"
        '<header class="hero">'
        '<div class="hero-top">'
        '<div>'
        f'<h1>{_esc(char_cn)} · {_esc(asc_text)} · {_esc(outcome_title)}</h1>'
        '<div class="hero-meta">'
        f'<span>对局 ID：<code>{_esc(summary["run_id"])}</code></span>'
        f'<span>角色代码：<code>{_esc(summary["character"])}</code></span>'
        f'<span>游戏版本：<code>{_esc(build_id)}</code></span>'
        f'<span>生成时间：{_esc(ts)}</span>'
        "</div>"
        "</div>"
        f'<span class="pill {"pill-ok" if win else "pill-warn"}">{_esc("VICTORY" if win else "DEFEAT")}</span>'
        "</div>"
        '<div class="kpi-grid">'
        '<div class="kpi">'
        '<div class="kpi-label">对局结局</div>'
        f'<div class="kpi-val {outcome_cls}">{_esc(outcome_title)}</div>'
        f'<div class="kpi-sub">到达第 {int(summary["acts_reached"])} 幕 · 最高评分楼层 F{int(summary["last_scored_floor"])}</div>'
        "</div>"
        '<div class="kpi">'
        '<div class="kpi-label">终盘牌组规模</div>'
        f'<div class="kpi-val">{int(summary["final_deck_size"])} 张</div>'
        f'<div class="kpi-sub">已评分 {len(review.decisions)} 项 · 未评分 {skipped_count} 项</div>'
        "</div>"
        '<div class="kpi">'
        '<div class="kpi-label">与引擎总一致率</div>'
        f'<div class="kpi-val">{_esc(_rate_text(overall["agree"], overall["judged"]))}</div>'
        f'<div class="kpi-sub">选牌 {_esc(_rate_text(card_stat["agree"], card_stat["judged"]))} · '
        f'篝火 {_esc(_rate_text(rest_stat["agree"], rest_stat["judged"]))}</div>'
        "</div>"
        '<div class="kpi">'
        '<div class="kpi-label">商店“买或离开”一致性</div>'
        f'<div class="kpi-val">{_esc(_rate_text(shop_stat["buy_or_leave_agree"], shop_stat["buy_or_leave_judged"]))}</div>'
        f'<div class="kpi-sub">{_esc("；".join(shop_note_parts[1:]) if len(shop_note_parts) > 1 else "商店仅按消费或离开判定一致")}</div>'
        "</div>"
        "</div>"
        '<div class="banner">说明：本报告完全由本地确定性 <code>deck_fitness</code> 评分引擎即时计算生成，不调用网络或外部模型。'
        '推荐分代表当前牌组面对后续战斗需求的适合度而非绝对胜率；未建模卡牌/药水效果已在对应卡片中明确标注。</div>'
        + version_banner
        + warnings_banner
        + "</header>\n"
        + _render_trend_section(trend)
        + "\n"
        + tier1_html
        + "\n"
        + tier2_html
        + "\n"
        + _render_ledger_section(review)
        + "\n"
        + '<footer>'
        + f'<span>STS2 Guide 赛后复盘 · 引擎数据本地只读运行 · 耗时 {total_ms:.1f} ms</span>'
        + f'<span>Tier-1: {len(tier1_ids)} · Tier-2: {len(tier2_ids)} · 未评分: {skipped_count}</span>'
        + "</footer>\n"
        + "</div>\n"
        + f"<script>{REPORT_JS}</script>\n"
        + "</body>\n"
        + "</html>\n"
    )


def render_run_review_html(
    run_path: str | Path,
    engine: Optional[Engine] = None,
) -> str:
    """Load, score, and render a finished single-player ``.run`` file to HTML."""
    t0 = time.perf_counter()
    resolved = Path(run_path)
    active_engine = engine if engine is not None else _get_default_engine()
    review = review_file(active_engine, resolved)
    return render_review_html(review, started_at=t0)


def _atomic_write_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m review.html_report",
        description="Render a finished single-player STS2 .run file to a self-contained HTML review report.",
    )
    parser.add_argument("run_path", type=Path, help="Path to a finished .run file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output HTML path. Prints HTML to stdout when omitted.",
    )
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    try:
        html_doc = render_run_review_html(args.run_path)
    except (FileNotFoundError, OSError, ReviewError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if args.output is not None:
        _atomic_write_utf8(args.output, html_doc)
        print(f"Wrote {args.output} ({len(html_doc)} chars, {elapsed_ms:.1f} ms)")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        sys.stdout.write(html_doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
