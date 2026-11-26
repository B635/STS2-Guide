"""Deterministic validation gate wrapping scripts/merge_fact_reviews.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from advisor.deck_mechanics import unsupported_reason
from scripts import merge_fact_reviews as mfr
from storage.assembly_facts import REVIEWED_STATUSES

SUPPORTED_ENTITY_TYPES = ("cards", "powers", "relics")

# Backward-compatible superset including TASK-029 (v0.113.1) vocabulary so the
# pipeline validates both 0.111.0 (main) and 0.113.1 (feature/v0113) extracts.
EFFECT_KINDS: Set[str] = set(mfr.EFFECT_KINDS) | {
    "arrow_slots",
    "dislodge_arrow",
}
WHEN: Set[str] = set(mfr.WHEN) | {
    "arrow_shot",
    "arrow_impaled",
    "arrow_dislodged",
}
MODIFIERS: Set[str] = set(mfr.MODIFIERS)
CATEGORIES: Set[str] = set(mfr.CATEGORIES)
OBTAIN_KINDS: Set[str] = set(mfr.OBTAIN_KINDS)
BETWEEN_WHEN: Set[str] = set(mfr.BETWEEN_WHEN) | {
    "lethal_damage",
}
BETWEEN_MODIFIERS: Set[str] = set(mfr.BETWEEN_MODIFIERS)
REVIEW_FIELDS: Set[str] = set(mfr.REVIEW_FIELDS)

CORE_FIELDS: Dict[str, tuple] = {
    "cards": ("card_type", "cost", "tags", "keywords", "effects"),
    "powers": ("category", "duration", "mechanics"),
    "relics": ("category", "mechanics"),
}

VALID_CARD_TYPES = {"Attack", "Skill", "Power", "Status", "Curse", "Quest"}
DAMAGE_TARGETS = {"single", "all", "random"}
APPLY_POWER_TARGETS = {"self", "enemy", "all_enemies", "random"}
SELF_ONLY_EFFECT_KINDS = {"block", "draw", "energy", "heal", "hp_loss"}
VALID_DESTINATIONS = {"hand", "draw", "discard", "deck"}
VALID_TRIGGER_MECHANIC_KEYS = {
    "when", "filter", "every", "counter", "turns", "every_turns", "turns_from",
    "limit", "condition", "effect", "owner", "scope", "times", "note", "notes",
    "amount_text", "amount_from",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    problems: List[str] = field(default_factory=list)
    codex_conflicts: List[str] = field(default_factory=list)
    engine_unsupported: List[str] = field(default_factory=list)


DYNAMIC_CARD_REF_PREFIXES = ("copy_of_", "copy ", "one_card_")


def _check_add_card_ref(
    effect: Mapping[str, Any],
    where: str,
    valid_card_ids: Optional[Set[str]],
    valid_card_classes: Optional[Set[str]],
) -> List[str]:
    card_ref = effect.get("card")
    if not isinstance(card_ref, str) or valid_card_ids is None:
        return []
    text = card_ref.strip()
    if not text:
        return [f"{where}: add_card.card must not be empty"]
    if text in valid_card_ids:
        return []
    if valid_card_classes is not None and text in valid_card_classes:
        return []
    if text.startswith(DYNAMIC_CARD_REF_PREFIXES):
        return []
    return [f"{where}: add_card.card {card_ref!r} is not in catalog_cards"]


def _check_effect(
    effect: Any,
    where: str,
    *,
    allow_when: bool = False,
    valid_card_ids: Optional[Set[str]] = None,
    valid_card_classes: Optional[Set[str]] = None,
) -> List[str]:
    if not isinstance(effect, dict):
        return [f"{where}: effect is not an object"]
    problems: List[str] = []
    if not allow_when and ("when" in effect or "modifier" in effect):
        problems.append(f"{where}: triggered/modifier hooks belong in 'mechanics', not 'effects'")
    kind = effect.get("kind")
    if kind == "generate":
        return problems + [f"{where}: use canonical kind 'add_card' instead of 'generate'"]
    if kind not in EFFECT_KINDS:
        return problems + [f"{where}: unknown effect kind {kind!r}"]
    amount = effect.get("amount")
    if amount is not None and (isinstance(amount, bool) or not isinstance(amount, (int, float))):
        problems.append(f"{where}: amount must be a number (use amount_from for derived amounts)")
    upgrade = effect.get("upgrade")
    if upgrade is not None and (isinstance(upgrade, bool) or not isinstance(upgrade, (int, float))):
        problems.append(f"{where}: upgrade must be a number")
    if kind == "damage" and "target" in effect and effect["target"] not in DAMAGE_TARGETS:
        problems.append(f"{where}: damage target must be one of {sorted(DAMAGE_TARGETS)} (got {effect['target']!r})")
    if kind == "apply_power":
        power_name = effect.get("power")
        if not isinstance(power_name, str) or not power_name.strip():
            problems.append(f"{where}: apply_power requires non-empty 'power' name (without 'Power' suffix)")
        elif len(power_name) > 5 and power_name.endswith("Power"):
            problems.append(f"{where}: strip trailing 'Power' suffix from power name {power_name!r}")
        if "target" in effect and effect["target"] not in APPLY_POWER_TARGETS:
            problems.append(
                f"{where}: apply_power target must be one of {sorted(APPLY_POWER_TARGETS)} (got {effect['target']!r})"
            )
    if kind in SELF_ONLY_EFFECT_KINDS and "target" in effect:
        problems.append(f"{where}: omit 'target' on self-only kind {kind!r}")
    if kind == "add_card":
        if "pile" in effect:
            problems.append(f"{where}: use 'destination' ('hand'|'draw'|'discard'|'deck') instead of 'pile'")
        dest = effect.get("destination")
        if dest is not None and dest not in VALID_DESTINATIONS:
            problems.append(f"{where}: destination must be one of {sorted(VALID_DESTINATIONS)} (got {dest!r})")
        problems.extend(_check_add_card_ref(effect, where, valid_card_ids, valid_card_classes))
    return problems


def _check_obtain_and_between(
    entry: Mapping[str, Any],
    key: str,
    *,
    valid_card_ids: Optional[Set[str]] = None,
    valid_card_classes: Optional[Set[str]] = None,
) -> List[str]:
    problems: List[str] = []
    if (entry.get("obtain") or entry.get("between_fights") or entry.get("obtain_params")) and not key.startswith("relics:"):
        problems.append(f"{key}: obtain / between_fights are relic fields")
    for i, effect in enumerate(entry.get("obtain") or ()):
        where = f"{key} obtain[{i}]"
        if not isinstance(effect, dict) or effect.get("kind") not in OBTAIN_KINDS:
            kind = effect.get("kind") if isinstance(effect, dict) else effect
            problems.append(f"{where}: unknown obtain kind {kind!r}")
            continue
        amount = effect.get("amount")
        if amount is not None and (isinstance(amount, bool) or not isinstance(amount, (int, float))):
            problems.append(f"{where}: amount must be a number (use amount_from for derived amounts)")
        count = effect.get("count")
        if count is not None and (isinstance(count, bool) or (not isinstance(count, (int, float)) and count != "all")):
            problems.append(f"{where}: count must be a number or \"all\"")
        if effect.get("kind") == "add_card":
            problems.extend(_check_add_card_ref(effect, where, valid_card_ids, valid_card_classes))
        if effect.get("kind") == "offer_rewards":
            rewards = effect.get("rewards")
            if not isinstance(rewards, list) or not rewards:
                problems.append(f"{where}: offer_rewards requires a non-empty 'rewards' list of reward objects")
    for i, mechanic in enumerate(entry.get("between_fights") or ()):
        where = f"{key} between_fights[{i}]"
        if not isinstance(mechanic, dict):
            problems.append(f"{where}: not an object")
        elif "modifier" in mechanic:
            if "when" in mechanic:
                problems.append(f"{where}: static modifier should not also include 'when'")
            if mechanic["modifier"] not in BETWEEN_MODIFIERS:
                problems.append(f"{where}: unknown modifier {mechanic['modifier']!r}")
        elif mechanic.get("when") not in BETWEEN_WHEN:
            problems.append(f"{where}: unknown when {mechanic.get('when')!r}")
        elif not isinstance(mechanic.get("effect"), dict) or mechanic["effect"].get("kind") not in OBTAIN_KINDS:
            problems.append(f"{where}: effect needs an obtain kind")
        elif mechanic["effect"].get("kind") == "add_card":
            problems.extend(_check_add_card_ref(mechanic["effect"], where, valid_card_ids, valid_card_classes))
    params = entry.get("obtain_params")
    if params is not None and not isinstance(params, dict):
        problems.append(f"{key}: obtain_params must be an object")
    return problems


def _check_card_cost(cost: Any, key: str) -> List[str]:
    if cost is None:
        return []
    if not isinstance(cost, dict):
        return [f"{key}: cost must be null or an object with base and upgraded"]
    problems: List[str] = []
    for field_name in ("base", "upgraded"):
        if field_name not in cost:
            problems.append(f"{key}: cost.{field_name} is required")
            continue
        val = cost[field_name]
        if val is None or val == "X":
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            problems.append(f"{key}: cost.{field_name} must be int, \"X\", or null (got {val!r})")
    return problems


def validate_fact(
    entry: Any,
    extract_by_key: Mapping[str, Mapping[str, Any]],
    facts_by_key: Mapping[str, Mapping[str, Any]],
    *,
    expected_entity_key: Optional[str] = None,
    catalog_cards: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ValidationResult:
    """Validate one candidate fact against extract, catalog, and vocabulary rules."""
    if not isinstance(entry, dict):
        return ValidationResult(ok=False, problems=["fact must be a JSON object"])

    key = entry.get("entity_key")
    if not isinstance(key, str) or ":" not in key:
        return ValidationResult(ok=False, problems=["entity_key must be a string like 'cards:STRIKE'"])
    if expected_entity_key is not None and key != expected_entity_key:
        return ValidationResult(
            ok=False,
            problems=[f"entity_key mismatch: expected {expected_entity_key!r}, got {key!r}"],
        )
    if key not in facts_by_key:
        return ValidationResult(ok=False, problems=[f"{key}: not a catalog entity"])

    entity_type = key.split(":", 1)[0]
    if entity_type not in SUPPORTED_ENTITY_TYPES:
        return ValidationResult(
            ok=False,
            problems=[f"{key}: entity type {entity_type!r} is not supported by fact_pipeline"],
        )

    problems: List[str] = []
    source = extract_by_key.get(key)
    if source is None or entry.get("fingerprint") != source.get("fingerprint"):
        problems.append(f"{key}: fingerprint does not match the current extract")

    if entry.get("status") not in REVIEWED_STATUSES:
        problems.append(f"{key}: status must be 'converted' or 'unknowable'")

    evidence = entry.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(isinstance(x, str) and x.strip() for x in evidence):
        problems.append(f"{key}: evidence must be a non-empty list of strings citing methods/IL calls")

    extra = set(entry) - REVIEW_FIELDS - {"entity_key", "class", "fingerprint"}
    if extra:
        problems.append(f"{key}: unknown fields {sorted(extra)}")

    if entry.get("status") == "converted":
        for req in CORE_FIELDS.get(entity_type, ()):
            if req not in entry:
                problems.append(f"{key}: a complete review requires {req}")

    if entity_type == "cards":
        if "card_type" in entry and entry["card_type"] is not None:
            if entry["card_type"] not in VALID_CARD_TYPES:
                problems.append(f"{key}: invalid card_type {entry['card_type']!r}")
        if "cost" in entry:
            problems.extend(_check_card_cost(entry.get("cost"), key))
        for list_field in ("tags", "keywords"):
            if list_field in entry:
                val = entry[list_field]
                if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
                    problems.append(f"{key}: {list_field} must be a list of strings")
        if "effects" in entry and not isinstance(entry["effects"], list):
            problems.append(f"{key}: effects must be a list")

    if entity_type in ("relics", "powers"):
        if entry.get("status") == "converted" or "category" in entry:
            if entry.get("category") not in CATEGORIES:
                problems.append(f"{key}: category must be one of {sorted(CATEGORIES)}")
        if "mechanics" in entry and not isinstance(entry["mechanics"], list):
            problems.append(f"{key}: mechanics must be a list")

    if entity_type == "powers" and entry.get("status") == "converted":
        duration = entry.get("duration")
        if not isinstance(duration, dict) or not isinstance(duration.get("kind"), str) or not duration["kind"].strip():
            problems.append(f"{key}: duration must be an object with a non-empty 'kind'")

    valid_card_ids: Optional[Set[str]] = None
    valid_card_classes: Optional[Set[str]] = None
    if catalog_cards is not None:
        valid_card_ids = {str(cid) for cid in catalog_cards}
        valid_card_classes = set()
        for src in (facts_by_key, extract_by_key):
            for k, v in src.items():
                if k.startswith("cards:") and isinstance(v, Mapping):
                    cid = k.split(":", 1)[1]
                    cls_name = v.get("class")
                    if cid in valid_card_ids and isinstance(cls_name, str) and cls_name:
                        valid_card_classes.add(cls_name)

    problems.extend(
        _check_obtain_and_between(
            entry,
            key,
            valid_card_ids=valid_card_ids,
            valid_card_classes=valid_card_classes,
        )
    )

    effects = entry.get("effects")
    if isinstance(effects, list):
        for i, effect in enumerate(effects):
            problems.extend(
                _check_effect(
                    effect,
                    f"{key} effects[{i}]",
                    valid_card_ids=valid_card_ids,
                    valid_card_classes=valid_card_classes,
                )
            )

    mechanics = entry.get("mechanics")
    engine_unsupported: List[str] = []
    if isinstance(mechanics, list):
        for i, mechanic in enumerate(mechanics):
            where = f"{key} mechanics[{i}]"
            if not isinstance(mechanic, dict):
                problems.append(f"{where}: mechanic is not an object")
                continue
            if "modifier" in mechanic:
                if "when" in mechanic:
                    problems.append(f"{where}: static modifier should not also include 'when'")
                if mechanic["modifier"] not in MODIFIERS:
                    problems.append(f"{where}: unknown modifier {mechanic['modifier']!r}")
            else:
                if "count" in mechanic:
                    problems.append(f"{where}: use 'every' + 'counter' ('turn'|'combat') instead of 'count'")
                bad_keys = set(mechanic) - VALID_TRIGGER_MECHANIC_KEYS
                if bad_keys:
                    problems.append(f"{where}: unexpected trigger mechanic keys {sorted(bad_keys)}")
                if mechanic.get("when") not in WHEN:
                    problems.append(f"{where}: unknown when {mechanic.get('when')!r}")
                every = mechanic.get("every")
                if (
                    mechanic.get("when") == "turn_start"
                    and mechanic.get("counter") == "turn"
                    and isinstance(every, (int, float))
                    and not isinstance(every, bool)
                    and every > 1
                ):
                    problems.append(
                        f"{where}: impossible trigger combination when='turn_start' with counter='turn' and every={every} "
                        "(turn_start fires at most once per turn; use counter='combat' or every_turns)"
                    )
                problems.extend(
                    _check_effect(
                        mechanic.get("effect"),
                        where,
                        valid_card_ids=valid_card_ids,
                        valid_card_classes=valid_card_classes,
                    )
                )
            reason = unsupported_reason(mechanic)
            if reason:
                engine_unsupported.append(f"{key}: {reason}")

    unknowable = entry.get("unknowable")
    if unknowable is not None:
        if not isinstance(unknowable, list):
            problems.append(f"{key}: unknowable must be a list")
        else:
            for item in unknowable:
                if not isinstance(item, dict) or not item.get("what") or not item.get("reason"):
                    problems.append(f"{key}: unknowable items need 'what' and 'reason'")

    conflicts: List[str] = []
    if not problems and catalog_cards is not None and entity_type == "cards":
        conflicts = mfr.codex_conflicts(entry, dict(catalog_cards))

    return ValidationResult(
        ok=len(problems) == 0,
        problems=problems,
        codex_conflicts=conflicts,
        engine_unsupported=engine_unsupported,
    )


def prepare_mergeable_entry(
    fact: Mapping[str, Any],
    *,
    reviewed_by: str,
    audit_by: str,
    audit_verdict: str,
    audit_notes: str,
) -> Dict[str, Any]:
    """Return a clean dict ready for scripts/merge_fact_reviews.py."""
    entry = dict(fact)
    entry["reviewed_by"] = reviewed_by
    entry["audit"] = {
        "by": audit_by,
        "verdict": audit_verdict,
        "notes": audit_notes,
    }
    return entry


def vocabulary_snapshot() -> Dict[str, Sequence[str]]:
    """Sorted vocabulary lists shared with prompts so rules and checks never drift."""
    return {
        "effect_kinds": sorted(EFFECT_KINDS),
        "when": sorted(WHEN),
        "modifiers": sorted(MODIFIERS),
        "categories": sorted(CATEGORIES),
        "obtain_kinds": sorted(OBTAIN_KINDS),
        "between_when": sorted(BETWEEN_WHEN),
        "between_modifiers": sorted(BETWEEN_MODIFIERS),
        "review_fields": sorted(REVIEW_FIELDS),
    }
