"""Assembly-verified mechanic facts for cards and relics.

Two committed inputs feed the release database:

* the extract: produced by the read-only ``tools/Sts2FactExtractor`` from the
  local ``sts2.dll``; one entry per card/relic class with overridden hooks,
  referenced game members and a per-class code fingerprint;
* the facts: one entry per catalog card/relic with a review status. A fact is
  only valid while its recorded fingerprint equals the extract's fingerprint
  for the same class, so a game update invalidates exactly the classes whose
  code changed.

Nothing here reads descriptions; relic categories are triaged from hooks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

ASSEMBLY_FACTS_VERSION = "assembly-facts-v1"
FACTS_SCHEMA_VERSION = 1

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACT_PATH = ROOT_DIR / "data" / "assembly" / "sts2-extract.json"
DEFAULT_FACTS_PATH = ROOT_DIR / "data" / "assembly" / "facts.json"

KIND_TO_ENTITY_TYPE = {"card": "cards", "relic": "relics", "power": "powers"}

# Review states. Only reviewed states can be valid.
STATUS_UNREVIEWED = "unreviewed"
STATUS_CONVERTED = "converted"
STATUS_UNKNOWABLE = "unknowable"
STATUS_STALE = "stale"
REVIEWED_STATUSES = frozenset({STATUS_CONVERTED, STATUS_UNKNOWABLE})
# Effective coverage states: a reviewed fact whose class code no longer
# matches its recorded fingerprint is stale, wherever it is evaluated.
COVERAGE_STATES = (STATUS_CONVERTED, STATUS_UNKNOWABLE, STATUS_STALE, STATUS_UNREVIEWED)

# Relic categories (see GLOSSARY: 基线遗物 / 触发遗物 / 无关遗物).
CATEGORY_TRIGGER = "trigger"
CATEGORY_BASELINE = "baseline"
CATEGORY_IRRELEVANT = "irrelevant"
CATEGORY_NEEDS_REVIEW = "needs_review"

# Hooks whose firing is conditioned on a card or card-driven action.
TRIGGER_HOOKS = frozenset({
    "AfterAttack",
    "AfterCardChangedPiles",
    "AfterCardDiscarded",
    "AfterCardEnteredCombat",
    "AfterCardExhausted",
    "AfterCardGeneratedForCombat",
    "AfterCardPlayed",
    "AfterHandEmptied",
    "AfterModifyingBlockAmount",
    "AfterModifyingCardPlayCount",
    "AfterModifyingOrbPassiveTriggerCount",
    "AfterModifyingPowerAmountGiven",
    "AfterOrbChanneled",
    "AfterShuffle",
    "AfterStarsSpent",
    "BeforeCardPlayed",
    "ModifyBlockMultiplicative",
    "ModifyCardPlayCount",
    "ModifyDamageAdditive",
    "ModifyDamageMultiplicative",
    "ModifyOrbPassiveTriggerCounts",
    "ModifyOrbValue",
    "ModifyPowerAmountGivenAdditive",
    "ModifyPowerAmountGivenMultiplicative",
    "ModifyXValue",
    "ShouldPlay",
    "TryModifyEnergyCostInCombat",
    "TryModifyEnergyCostInCombatLate",
    "TryModifyStarCost",
})

# In-combat hooks that change what the deck can do regardless of which card.
BASELINE_HOOKS = frozenset({
    "AfterAutoPrePlayPhaseEntered",
    "AfterAutoPrePlayPhaseEnteredLate",
    "AfterBlockBroken",
    "AfterBlockCleared",
    "AfterCreatureAddedToCombat",
    "AfterCurrentHpChanged",
    "AfterDamageReceived",
    "AfterDeath",
    "AfterDiedToDoom",
    "AfterEnergyReset",
    "AfterEnergyResetLate",
    "AfterFlush",
    "AfterModifyingHandDraw",
    "AfterModifyingHpLostAfterOsty",
    "AfterModifyingPowerAmountReceived",
    "AfterPlayerTurnStart",
    "AfterPlayerTurnStartLate",
    "AfterPotionDiscarded",
    "AfterPotionUsed",
    "AfterPreventingBlockClear",
    "AfterPreventingDeath",
    "AfterPreventingDraw",
    "AfterSideTurnEnd",
    "AfterSideTurnStart",
    "AfterTakingExtraTurn",
    "BeforeCombatStart",
    "BeforeCombatStartLate",
    "BeforeHandDraw",
    "BeforePowerAmountChanged",
    "BeforeSideTurnEnd",
    "BeforeSideTurnEndEarly",
    "BeforeSideTurnEndVeryEarly",
    "BeforeSideTurnStart",
    "ModifyHandDraw",
    "ModifyHpLostAfterOsty",
    "ModifyHpLostAfterOstyLate",
    "ModifyMaxEnergy",
    "ShouldClearBlock",
    "ShouldDieLate",
    "ShouldDraw",
    "ShouldFlush",
    "ShouldPlayerResetEnergy",
    "ShouldTakeExtraTurn",
    "TryModifyPowerAmountReceived",
})

# Hooks that change card rewards or cards being added: they can affect the
# skip option or the offered candidates, so they always need a human review.
REWARD_HOOKS = frozenset({
    "AfterModifyingCardRewardOptions",
    "ModifyCardRewardCreationOptions",
    "TryModifyCardBeingAddedToDeck",
    "TryModifyCardRewardAlternatives",
    "TryModifyCardRewardOptions",
    "TryModifyCardRewardOptionsLate",
})

# Out-of-combat hooks: pickup, rooms, rewards, shop, rest site, map.
RUN_HOOKS = frozenset({
    "AfterCloned",
    "AfterCombatEnd",
    "AfterCombatVictory",
    "AfterCombatVictoryEarly",
    "AfterGoldGained",
    "AfterItemPurchased",
    "AfterModifyingGoldGained",
    "AfterModifyingRewards",
    "AfterObtained",
    "AfterPotionProcured",
    "AfterRestSiteHeal",
    "AfterRoomEntered",
    "BeforeCombatRewardOffered",
    "IsAllowed",
    "IsAllowedAtNeow",
    "ModifyExtraRestSiteHealText",
    "ModifyGeneratedMap",
    "ModifyGeneratedMapLate",
    "ModifyGoldGained",
    "ModifyMerchantCardCreationResults",
    "ModifyMerchantPrice",
    "ModifyRestSiteHealAmount",
    "ModifyUnknownMapPointRoomTypes",
    "ShouldAllowFreeTravel",
    "ShouldDisableRemainingRestSiteOptions",
    "ShouldForcePotionReward",
    "ShouldGenerateTreasure",
    "ShouldProcurePotion",
    "ShouldRefillMerchantEntry",
    "TryModifyRestSiteHealRewards",
    "TryModifyRestSiteOptions",
    "TryModifyRewards",
    "TryModifyRewardsLate",
})

# Presentation/bookkeeping overrides that carry no mechanic.
IGNORED_OVERRIDES = frozenset({
    "DoActivateVisuals",
    "SetupForPlayer",
    "SetupForTests",
    "SetTestEnergyCostOverride",
    "SummonPet",
    "UpdateDisplay",
})


class AssemblyFactsError(ValueError):
    """Raised when the committed extract or facts are malformed."""


def class_to_entity_id(class_name: str) -> str:
    """Map a model class name to its catalog id (``WarPaint`` -> ``WAR_PAINT``)."""

    return re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        class_name,
    ).upper()


def entity_key(kind: str, class_name: str) -> str:
    # Power classes carry a ``Power`` suffix their catalog ids do not.
    if kind == "power" and class_name.endswith("Power"):
        class_name = class_name[: -len("Power")]
    return f"{KIND_TO_ENTITY_TYPE[kind]}:{class_to_entity_id(class_name)}"


def _hook_names(overrides: Iterable[str]) -> list[str]:
    return sorted(
        name for name in overrides
        if not name.startswith("get_") and name not in IGNORED_OVERRIDES
    )


# Game commands that act inside a fight. Entering a combat room is how some
# relics grant fight-start powers, so a room hook calling one is not "run only".
COMBAT_COMMAND_PATTERN = re.compile(
    r"Commands\.(CreatureCmd::(GainBlock|Damage|LoseBlock)|PowerCmd::Apply|DamageCmd"
    r"|CardPileCmd::(Draw|AddGeneratedCard)|PlayerCmd::GainEnergy|OrbCmd|OstyCmd)"
)


def triage_relic(overrides: Iterable[str], calls: Iterable[str] = ()) -> Dict[str, object]:
    """Initial, structure-only relic category; review may still change it."""

    hooks = _hook_names(overrides)
    groups = {
        "trigger": [h for h in hooks if h in TRIGGER_HOOKS],
        "baseline": [h for h in hooks if h in BASELINE_HOOKS],
        "reward": [h for h in hooks if h in REWARD_HOOKS],
        "run": [h for h in hooks if h in RUN_HOOKS],
        "unknown": [
            h for h in hooks
            if h not in TRIGGER_HOOKS | BASELINE_HOOKS | REWARD_HOOKS | RUN_HOOKS
        ],
    }
    if groups["reward"] or groups["unknown"]:
        category = CATEGORY_NEEDS_REVIEW
    elif groups["trigger"]:
        category = CATEGORY_TRIGGER
    elif groups["baseline"]:
        category = CATEGORY_BASELINE
    elif "AfterRoomEntered" in groups["run"] and any(
        COMBAT_COMMAND_PATTERN.search(call) for call in calls
    ):
        category = CATEGORY_BASELINE
    else:
        category = CATEGORY_IRRELEVANT
    return {"category": category, "hook_groups": groups}


def _read_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssemblyFactsError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise AssemblyFactsError(f"{label} must be a JSON object: {path}")
    return payload


def load_extract(path: Path | str = DEFAULT_EXTRACT_PATH) -> dict:
    payload = _read_json(Path(path), "assembly extract")
    types = payload.get("types")
    if not isinstance(types, list) or not payload.get("assembly_sha256"):
        raise AssemblyFactsError("assembly extract has no types or assembly hash")
    return payload


def load_facts(path: Path | str = DEFAULT_FACTS_PATH) -> dict:
    payload = _read_json(Path(path), "assembly facts")
    if payload.get("schema_version") != FACTS_SCHEMA_VERSION:
        raise AssemblyFactsError("assembly facts schema version is unsupported")
    if not isinstance(payload.get("entities"), list):
        raise AssemblyFactsError("assembly facts have no entities")
    return payload


def catalog_keys(knowledge_path: Path | str) -> list[str]:
    """Catalog card and relic entity keys that must each have one fact."""

    knowledge = _read_json(Path(knowledge_path), "catalog")
    return [
        f"{entity_type}:{item['id']}"
        for entity_type in KIND_TO_ENTITY_TYPE.values()
        for item in knowledge[entity_type]
    ]


def extract_index(extract: Mapping) -> Dict[str, dict]:
    """Index extract entries by catalog entity key."""

    index: Dict[str, dict] = {}
    for entry in extract["types"]:
        kind = entry.get("kind")
        if kind not in KIND_TO_ENTITY_TYPE or entry.get("abstract"):
            continue
        index[entity_key(kind, entry["class"])] = entry
    return index


def refresh_facts(
    extract: Mapping,
    catalog_keys: Iterable[str],
    previous: Optional[Mapping] = None,
) -> dict:
    """Rebuild the facts file for one extract, keeping still-valid reviews.

    Reviewed entries whose class fingerprint changed become ``stale`` and keep
    their previous content for re-review; they are never silently re-used.
    """

    index = extract_index(extract)
    previous_by_key = {
        entry["entity_key"]: entry
        for entry in (previous or {}).get("entities", [])
    }
    entities = []
    for key in sorted(set(catalog_keys)):
        source = index.get(key)
        if source is None:
            entities.append({
                "entity_key": key,
                "class": None,
                "fingerprint": None,
                "status": STATUS_UNREVIEWED,
                "note": "no model class found in the assembly",
            })
            continue
        prior = dict(previous_by_key.get(key) or {})
        hooks = _hook_names(source.get("overrides") or ())
        tokens = source.get("override_tokens") or {}
        entry = {
            "entity_key": key,
            "class": source["class"],
            "fingerprint": source["fingerprint"],
            "hooks": hooks,
            "hook_tokens": {hook: tokens[hook] for hook in hooks if hook in tokens},
        }
        if key.startswith("relics:"):
            entry["triage"] = triage_relic(
                source.get("overrides") or (), source.get("calls") or ()
            )
        status = prior.get("status", STATUS_UNREVIEWED)
        if status in REVIEWED_STATUSES | {STATUS_STALE}:
            entry.update({
                k: v for k, v in prior.items()
                if k not in {
                    "entity_key", "class", "fingerprint", "hooks",
                    "hook_tokens", "triage",
                }
            })
            if prior.get("fingerprint") != source["fingerprint"] or status == STATUS_STALE:
                entry["status"] = STATUS_STALE
                entry["reviewed_fingerprint"] = prior.get(
                    "reviewed_fingerprint", prior.get("fingerprint")
                )
        else:
            entry["status"] = STATUS_UNREVIEWED
        entities.append(entry)
    return {
        "schema_version": FACTS_SCHEMA_VERSION,
        "facts_version": ASSEMBLY_FACTS_VERSION,
        "assembly_sha256": extract["assembly_sha256"],
        "extractor_version": extract.get("extractor_version"),
        "entities": entities,
    }


def evaluate_facts(facts: Mapping, extract: Mapping) -> Dict[str, dict]:
    """Return every fact keyed by entity with its coverage state and validity."""

    index = extract_index(extract)
    result: Dict[str, dict] = {}
    for entry in facts["entities"]:
        key = entry["entity_key"]
        source = index.get(key)
        current = source["fingerprint"] if source else None
        status = entry.get("status", STATUS_UNREVIEWED)
        if status in REVIEWED_STATUSES and current is not None and entry.get("fingerprint") == current:
            state = status
        elif status in REVIEWED_STATUSES | {STATUS_STALE}:
            state = STATUS_STALE
        else:
            state = STATUS_UNREVIEWED
        result[key] = {
            **entry,
            "current_fingerprint": current,
            "state": state,
            "valid": state in REVIEWED_STATUSES,
        }
    return result


def coverage_summary(states: Iterable[tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """Count ``(entity_key, coverage_state)`` pairs per entity type."""

    summary: Dict[str, Dict[str, int]] = {}
    for key, state in states:
        bucket = summary.setdefault(
            key.split(":", 1)[0],
            {"total": 0, **{name: 0 for name in COVERAGE_STATES}},
        )
        bucket["total"] += 1
        bucket[state] += 1
    return summary
