"""Semantic comparison between candidate facts and ground-truth facts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROSE_KEYS = frozenset({"note", "notes", "comment", "comments", "description", "reason", "what"})
TEXT_FALLBACK_KEYS = frozenset({"amount_text", "hits_text", "count_text"})


@dataclass(frozen=True)
class ComparisonResult:
    entity_key: str
    strict_match: bool
    core_match: bool
    field_diffs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_key": self.entity_key,
            "strict_match": self.strict_match,
            "core_match": self.core_match,
            "field_diffs": self.field_diffs,
        }


def _norm_number(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def _strip_power_suffix(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    clean = name.strip()
    if len(clean) > 5 and clean.endswith("Power"):
        return clean[:-5]
    return clean


def _norm_value(val: Any, *, core_only: bool = False) -> Any:
    if isinstance(val, dict):
        out: Dict[str, Any] = {}
        for k, v in sorted(val.items()):
            if k in PROSE_KEYS:
                continue
            if v is None:
                continue
            if core_only and k in TEXT_FALLBACK_KEYS and any(nk in val for nk in ("amount", "hits", "count", "scale", "hits_scale")):
                continue
            if k == "power":
                out[k] = _strip_power_suffix(v)
            else:
                normed = _norm_value(v, core_only=core_only)
                if normed == {} or normed == []:
                    continue
                out[k] = normed
        return out
    if isinstance(val, list):
        return [_norm_value(x, core_only=core_only) for x in val]
    return _norm_number(val)


def _norm_effect(effect: Mapping[str, Any], *, core_only: bool = False) -> Dict[str, Any]:
    out = _norm_value(dict(effect), core_only=core_only)
    if not isinstance(out, dict):
        return {}
    if out.get("upgrade") == 0:
        out.pop("upgrade", None)
    if out.get("hits") == 1:
        out.pop("hits", None)
    if out.get("hits_upgrade") == 0:
        out.pop("hits_upgrade", None)
    if out.get("kind") in ("block", "draw", "energy", "heal", "hp_loss") and out.get("target") == "self":
        out.pop("target", None)
    if out.get("kind") == "block" and "times" in out and "hits" not in out:
        out["hits"] = out.pop("times")
    if isinstance(out.get("card"), str):
        out["card"] = out["card"].strip().upper()
    if out.get("count") == 1 and out.get("kind") in ("add_card", "discard", "exhaust_other"):
        out["count"] = 1
    if core_only:
        if out.get("free") is False:
            out.pop("free", None)
        free_text = str(out.pop("free_text", "") or "").lower()
        if out.get("free") is True and "this turn" in free_text:
            out["free"] = "this_turn"
        if out.get("kind") == "add_card":
            if "pool" in out:
                out["pool"] = "<pool>"
                if out.get("choice") == "random":
                    out.pop("choice", None)
            if out.get("position") == "random":
                out.pop("position", None)
        if out.get("kind") == "auto_play":
            if "from" in out and "source" not in out:
                out["source"] = out.pop("from")
            if out.get("choice") == "all":
                out.pop("choice", None)
        if out.get("kind") == "upgrade_cards" and out.get("scope") == "merchant_stock":
            out["scope"] = "merchant_cards"
        if out.get("kind") == "cost_reduction":
            if out.get("duration") == "combat":
                out.pop("duration", None)
            if out.get("target") == "this_card":
                out.pop("target", None)
        if isinstance(out.get("into"), str) and " " in out["into"]:
            out["into"] = "<into>"
        if isinstance(out.get("amount_from"), str):
            out["amount_from"] = "<derived>"
        if isinstance(out.get("amount_text"), str):
            out["amount_text"] = "<text>"
        if isinstance(out.get("filter"), dict):
            flt = dict(out["filter"])
            if "tags" in flt and "tag" not in flt:
                flt["tag"] = flt.pop("tags")
            out["filter"] = {
                fk: ("<text>" if isinstance(fv, str) and " " in fv else fv)
                for fk, fv in sorted(flt.items())
            }
        if isinstance(out.get("condition"), dict):
            # Keep condition keys as structural check without brittle free-text wording.
            out["condition"] = {
                ck: ("<text>" if isinstance(cv, str) and " " in cv else cv)
                for ck, cv in sorted(out["condition"].items())
            }
    return out


def _norm_mechanic(mechanic: Mapping[str, Any], *, core_only: bool = False) -> Dict[str, Any]:
    out = _norm_value(dict(mechanic), core_only=core_only)
    if not isinstance(out, dict):
        return {}
    if isinstance(out.get("effect"), dict):
        out["effect"] = _norm_effect(out["effect"], core_only=core_only)
    if core_only:
        if out.get("counter") == "combat" and isinstance(out.get("effect"), dict) and out["effect"].get("kind") == "cost_reduction":
            out.pop("counter", None)
        if isinstance(out.get("filter"), dict):
            flt = dict(out["filter"])
            if flt.get("card") == "self":
                flt.pop("card")
                flt["self"] = True
            if flt.get("owner") == "self":
                flt.pop("owner")
            if "tags" in flt and "tag" not in flt:
                flt["tag"] = flt.pop("tags")
            cleaned_flt = {
                fk: ("<text>" if isinstance(fv, str) and " " in fv else fv)
                for fk, fv in sorted(flt.items())
            }
            if cleaned_flt:
                out["filter"] = cleaned_flt
            else:
                out.pop("filter", None)
        if isinstance(out.get("condition"), dict):
            out["condition"] = {
                ck: ("<text>" if isinstance(cv, str) and " " in cv else cv)
                for ck, cv in sorted(out["condition"].items())
            }
        if "modifier" in out and isinstance(out.get("amount_text"), str):
            out.pop("amount_text", None)
        elif isinstance(out.get("amount_text"), str):
            out["amount_text"] = "<text>"
        if isinstance(out.get("amount_from"), str):
            out["amount_from"] = "<derived>"
    return out


def _canonical_sort(items: Sequence[Any]) -> List[Any]:
    return sorted(items, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))


def _norm_cost(cost: Any) -> Any:
    if cost is None:
        return None
    if isinstance(cost, dict):
        return {
            "base": _norm_number(cost.get("base")),
            "upgraded": _norm_number(cost.get("upgraded")),
        }
    return _norm_number(cost)


def _norm_upgrade_keywords(uk: Any) -> Dict[str, List[str]]:
    if not isinstance(uk, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k in ("add", "remove"):
        vals = sorted({str(x) for x in (uk.get(k) or ())})
        if vals:
            out[k] = vals
    return out


def _extract_card_view(entry: Mapping[str, Any], *, core_only: bool) -> Dict[str, Any]:
    keywords = {str(k) for k in (entry.get("keywords") or ())}
    effects: List[Dict[str, Any]] = []
    for eff in entry.get("effects") or ():
        if not isinstance(eff, dict):
            continue
        if eff.get("kind") == "exhaust_self":
            keywords.add("Exhaust")
            continue
        effects.append(_norm_effect(eff, core_only=core_only))

    mechanics = _canonical_sort(
        [_norm_mechanic(m, core_only=core_only) for m in (entry.get("mechanics") or ()) if isinstance(m, dict)]
    )
    view: Dict[str, Any] = {
        "status": entry.get("status"),
        "card_type": entry.get("card_type"),
        "cost": _norm_cost(entry.get("cost")),
        "star_cost": _norm_cost(entry.get("star_cost")),
        "tags": sorted({str(t) for t in (entry.get("tags") or ())}),
        "keywords": sorted(keywords),
        "upgrade_keywords": _norm_upgrade_keywords(entry.get("upgrade_keywords")),
        "multiplayer_only": bool(entry.get("multiplayer_only", False)),
        "can_be_generated_in_combat": bool(entry.get("can_be_generated_in_combat", True)),
        "effects": _canonical_sort(effects) if core_only else effects,
        "mechanics": mechanics,
        "has_cost_modifier": bool(entry.get("cost_modifier")),
        "has_unknowable": bool(entry.get("unknowable")),
    }
    if not core_only and entry.get("cost_modifier"):
        view["cost_modifier"] = _norm_value(entry["cost_modifier"], core_only=False)
    return view


def _extract_power_view(entry: Mapping[str, Any], *, core_only: bool) -> Dict[str, Any]:
    dur = entry.get("duration") if isinstance(entry.get("duration"), dict) else {}
    mechanics = _canonical_sort(
        [_norm_mechanic(m, core_only=core_only) for m in (entry.get("mechanics") or ()) if isinstance(m, dict)]
    )
    view: Dict[str, Any] = {
        "status": entry.get("status"),
        "category": entry.get("category"),
        "duration_kind": dur.get("kind"),
        "mechanics": mechanics,
        "has_unknowable": bool(entry.get("unknowable")),
    }
    if not core_only:
        view["duration"] = _norm_value(dur, core_only=False)
    return view


def _extract_relic_view(entry: Mapping[str, Any], *, core_only: bool) -> Dict[str, Any]:
    mechanics = _canonical_sort(
        [_norm_mechanic(m, core_only=core_only) for m in (entry.get("mechanics") or ()) if isinstance(m, dict)]
    )
    obtain = [
        _norm_effect(o, core_only=core_only)
        for o in (entry.get("obtain") or ())
        if isinstance(o, dict)
    ]
    between = _canonical_sort(
        [_norm_mechanic(b, core_only=core_only) for b in (entry.get("between_fights") or ()) if isinstance(b, dict)]
    )
    return {
        "status": entry.get("status"),
        "category": entry.get("category"),
        "mechanics": mechanics,
        "obtain": _canonical_sort(obtain) if core_only else obtain,
        "between_fights": between,
        "has_unknowable": bool(entry.get("unknowable")),
    }


def _build_view(entity_key: str, entry: Mapping[str, Any], *, core_only: bool) -> Dict[str, Any]:
    entity_type = entity_key.split(":", 1)[0]
    if entity_type == "cards":
        return _extract_card_view(entry, core_only=core_only)
    if entity_type == "powers":
        return _extract_power_view(entry, core_only=core_only)
    if entity_type == "relics":
        return _extract_relic_view(entry, core_only=core_only)
    raise ValueError(f"Unsupported entity type for comparison: {entity_type}")


def compare_facts(
    entity_key: str,
    candidate: Optional[Mapping[str, Any]],
    ground_truth: Mapping[str, Any],
) -> ComparisonResult:
    """Compare candidate fact against ground truth on strict and core semantic levels."""
    if candidate is None:
        return ComparisonResult(
            entity_key=entity_key,
            strict_match=False,
            core_match=False,
            field_diffs={"fact": {"expected": "present", "actual": None}},
        )

    strict_cand = _build_view(entity_key, candidate, core_only=False)
    strict_gold = _build_view(entity_key, ground_truth, core_only=False)
    core_cand = _build_view(entity_key, candidate, core_only=True)
    core_gold = _build_view(entity_key, ground_truth, core_only=True)

    strict_diffs: Dict[str, Dict[str, Any]] = {}
    for field_name in sorted(set(strict_gold) | set(strict_cand)):
        if strict_cand.get(field_name) != strict_gold.get(field_name):
            strict_diffs[field_name] = {
                "expected": strict_gold.get(field_name),
                "actual": strict_cand.get(field_name),
            }

    core_diffs: Dict[str, Dict[str, Any]] = {}
    for field_name in sorted(set(core_gold) | set(core_cand)):
        if core_cand.get(field_name) != core_gold.get(field_name):
            core_diffs[field_name] = {
                "expected": core_gold.get(field_name),
                "actual": core_cand.get(field_name),
            }

    # Report core_diffs first if core mismatches exist, else strict_diffs for full diagnostics.
    return ComparisonResult(
        entity_key=entity_key,
        strict_match=len(strict_diffs) == 0,
        core_match=len(core_diffs) == 0,
        field_diffs=core_diffs if core_diffs else strict_diffs,
    )
