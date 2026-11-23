"""Prompt templates, shared cacheable system spec, and OpenAI tool schemas."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from fact_pipeline.validator import CORE_FIELDS, vocabulary_snapshot

VERIFIED_ENUMS = """Verified enum values (from Sts2FactExtractor --enum; use read_enum for any other enum):
- CardType: 1 Attack, 2 Skill, 3 Power, 4 Status, 5 Curse, 6 Quest
- PileType: 1 Draw, 2 Hand, 3 Discard, 4 Exhaust, 5 Play, 6 Deck
- CardKeyword: 1 Exhaust, 2 Ethereal, 3 Innate, 4 Unplayable, 5 Retain, 6 Sly, 7 Eternal
- RoomType: 1 Monster, 2 Elite, 3 Boss, 4 Treasure, 5 Shop, 6 Event, 7 RestSite, 8 Map
- MapPointType: 1 Unknown, 2 Shop, 3 Treasure, 4 RestSite, 5 Monster, 6 Elite, 7 Boss, 8 Ancient"""

CANONICAL_EXAMPLES = """Canonical fact examples (match this exact structure and style):

1) Card example (cards:ABRASIVE):
{
  "entity_key": "cards:ABRASIVE",
  "fingerprint": "<exact fingerprint from extract>",
  "status": "converted",
  "card_type": "Power",
  "cost": {"base": 3, "upgraded": 3},
  "tags": [],
  "keywords": ["Sly"],
  "effects": [
    {"kind": "apply_power", "power": "Dexterity", "amount": 1, "upgrade": 0, "target": "self"},
    {"kind": "apply_power", "power": "Thorns", "amount": 4, "upgrade": 2, "target": "self"}
  ],
  "unknowable": [],
  "evidence": [
    "ctor CardModel(cost=3, Power, Rare, Self)",
    "CanonicalKeywords [Sly(6)]",
    "CanonicalVars PowerVar<ThornsPower>(4), PowerVar<DexterityPower>(1); OnUpgrade ThornsPower +2",
    "OnPlay <OnPlay>d__7: PowerCmd.Apply<DexterityPower>(owner, Dexterity); PowerCmd.Apply<ThornsPower>(owner, ThornsPower)"
  ]
}

2) Power example (powers:ACCURACY):
{
  "entity_key": "powers:ACCURACY",
  "fingerprint": "<exact fingerprint from extract>",
  "status": "converted",
  "category": "trigger",
  "duration": {"kind": "combat", "stack_type": "Counter"},
  "mechanics": [
    {"modifier": "damage", "add_per_stack": 1, "filter": {"dealer": "owner", "powered_attack": true, "tag": "Shiv"}}
  ],
  "unknowable": [],
  "evidence": [
    "get_StackType=1 (Counter), get_Type=1 (Buff); no remove/decrement -> all combat",
    "ModifyDamageAdditive: dealer==Owner && ValueProp.IsPoweredAttack && cardSource.Tags contains 5 (CardTag.Shiv) -> +Amount"
  ]
}

3) Relic examples:
- Combat baseline/trigger relic (relics:AKABEKO):
{
  "entity_key": "relics:AKABEKO",
  "fingerprint": "<exact fingerprint from extract>",
  "status": "converted",
  "category": "baseline",
  "mechanics": [
    {"when": "turn_start", "turns": [1], "effect": {"kind": "apply_power", "power": "Vigor", "amount": 8, "target": "self"}}
  ],
  "unknowable": [],
  "evidence": [
    "get_CanonicalVars: PowerVar<VigorPower>(8)",
    "AfterSideTurnStart <AfterSideTurnStart>d__6: TurnNumber<=1 -> PowerCmd.Apply<VigorPower>(owner, 8)"
  ]
}
- Pickup / out-of-combat relic (only AfterObtained or between-fight hooks -> category="irrelevant", mechanics=[]):
{
  "entity_key": "relics:GOLD_PEARL",
  "fingerprint": "<exact fingerprint from extract>",
  "status": "converted",
  "category": "irrelevant",
  "mechanics": [],
  "obtain": [{"kind": "gain_gold", "amount": 150}],
  "between_fights": [],
  "unknowable": [],
  "evidence": [
    "get_CanonicalVars: GoldVar(150)",
    "AfterObtained: PlayerCmd.GainGold(150, Owner)"
  ]
}"""


def _shared_spec_block() -> str:
    vocab = vocabulary_snapshot()
    canonical_effect_kinds = [k for k in vocab["effect_kinds"] if k != "generate"]
    return f"""# STS2 Assembly Mechanic-Facts Specification

You extract verified structured game-mechanic facts for Slay the Spire 2 strictly from .NET IL and assembly metadata.

## Ground Rules
1. **Assembly IL is the ONLY source of truth.** Spire Codex catalog fields/descriptions are provided strictly for cross-checking. If IL disagrees with Spire Codex, follow IL and note the discrepancy in `evidence`. Never infer unverified mechanics from descriptions.
2. **Read async state machines.** Actual logic of async hooks (`OnPlay`, `AfterCardPlayed`, `AfterObtained`, etc.) lives in nested `<MethodName>d__N::MoveNext`.
3. **Decode numbers and SIGNS from IL.**
   - Card base energy cost is the first integer passed to `CardModel::.ctor` (`-1` = X cost -> `"X"`, unplayable/no cost -> `null`). Check `OnUpgrade` for `EnergyCost::UpgradeBy` (delta!).
   - Base values come from `get_CanonicalVars` (`DamageVar`, `BlockVar`, `CardsVar`, `PowerVar<T>`, `GoldVar`, etc.). `amount` is the base number; `upgrade` is the **upgrade delta** (`0` if unchanged). Multi-hit uses `hits` (and `hits_upgrade` if upgraded).
   - **Check signs carefully:** `PlayerCmd.LoseEnergy(1)` -> `{{"kind": "energy", "amount": -1}}` (NEGATIVE!). `PlayerCmd.GainEnergy(1)` -> `{{"kind": "energy", "amount": 1}}`.
   - **Check indices carefully:** `Orbs[0]` / `FirstOrDefault` is the **first (front)** orb in queue, NEVER rightmost.
   - Never guess integer enum constants; use `read_enum` if an enum value is not listed below.
   - Ignore Dread (`GameMode 3`) and Blitz (`GameMode 4`) exclusive branches; mention them only in `evidence` if seen.
4. **Core required fields when `status == "converted"`:**
   - `cards`: {list(CORE_FIELDS["cards"])} (optional: `mechanics` for passive/triggered hooks on the card like `AfterCardDrawn` or `AfterCardExhausted`, `upgrade_keywords`, `star_cost`, `cost_modifier`, `can_be_generated_in_combat`, `multiplayer_only`, `open_questions`)
   - `powers`: {list(CORE_FIELDS["powers"])}
   - `relics`: {list(CORE_FIELDS["relics"])} (plus `obtain` for `AfterObtained`/`SetupForPlayer`, `between_fights` for out-of-combat ongoing hooks, `obtain_params` if option-generated)
5. **Exact EFFECT conventions (strictly enforced by validator):**
   - `damage`: `amount`, `upgrade`, `hits` (if >1), `target` in `"single" | "all" | "random"` (NEVER `"enemy"` or `"all_enemies"`).
   - `block`, `draw`, `energy`, `heal`, `hp_loss`: `amount`, `upgrade`; **OMIT `target`** (always self).
   - `apply_power`: `power` (strip trailing `"Power"` suffix, e.g. `"Thorns"`), `amount`, `upgrade`, `target` in `"self" | "enemy" | "all_enemies" | "random"`.
   - `add_card` (**NEVER use `"generate"`**): `card` (uppercase ID) or `pool`, `count` (`count_upgrade` if upgraded; for `"choice": "choose_1_of_3"`, `count` is **1** because the player gains 1 card, NOT 3!), `destination` in `"hand" | "draw" | "discard" | "deck"` (NEVER `pile`/`draw_pile`), `free` (`"this_turn"` | `"combat"` | `false`), `choice`, `upgraded_if_self_upgraded`.
   - `move_card`: `from`, `destination` (`"hand"|"draw"|"discard"`), `choice` (`"all"|"choose"|"random"`), `filter` (check `IsXCost` / `x_cost: false` if IL checks it).
   - `exhaust_other` / `discard`: `count`, `choice` (`"random"|"choose"|"all"`), `filter`.
   - `transform_cards` / `upgrade_cards`: `scope` (`"hand"|"draw"|"discard"|"deck"|"merchant_cards"`), `count`, `choice`, `filter`, `into`.
   - **Dynamic scaling** (use `scale`/`hits_scale` instead of splitting into a bogus second effect):
     `scale: {{"per": "<source>", "factor": n, "factor_upgrade": m, "min": 0}}` where `per` is one of `own_previous_plays`, `exhaust_pile`, `block`, `enemy_vulnerable`, `strength`, `strike_cards`, `hand_size`, `hp_loss_events`, `cards_drawn_this_combat`, `draw_pile`, `discards_this_turn`, `enemy_count`, `orbs_channeled_this_combat`, `cards_played_this_combat`, `energy_spent_this_turn`, `skills_in_hand`, `energy_after_cost`, `melodies_this_turn`, `melody_cards_in_combat`, `sadness_generated_this_combat`.
   - Triggered hooks on a card (e.g. `AfterCardDrawn` on Void, `AfterCardExhausted` cost reduction) MUST go in `mechanics[]`, **NEVER** in `effects[]`.
6. **Exact MECHANIC, `category`, and `duration.kind` conventions:**
   - Triggered mechanic: `{{"when": "...", "filter": {{...}}, "every": n, "counter": "turn"|"combat", "turns": [1], "every_turns": n, "limit": "once_per_turn"|"once_per_combat", "condition": {{...}}, "effect": EFFECT}}`. **NEVER** use `"count"` at mechanic top level (use `"every"` + `"counter"`). **NEVER** add `"turns": [1]` unless IL explicitly checks `TurnNumber == 1` / `TurnNumber <= 1`. For powers whose effect scales with stack `Amount`, use `amount_per_stack` (NOT fixed `count: 1`).
   - Static modifier: `{{"modifier": "...", "add"|"add_per_stack"|"multiply": n, "filter": {{...}}}}`. **NEVER** mix `"when"` into a static `"modifier"` object.
   - Temporary stat powers (e.g. `TemporaryStrengthPower` / `TemporaryDexterityPower` subclasses): record as `category: "baseline"`, `duration: {{"kind": "this_turn"}}`, `mechanics: [{{"modifier": "strength"|"dexterity", "add_per_stack": 1, "applies_to": "self"}}]`.
   - `category` (for powers & relics):
     - `irrelevant`: ONLY out-of-combat hooks (`AfterObtained`, `between_fights`, reward pools); `mechanics` MUST be `[]`.
     - `baseline`: combat baseline effects not triggered by playing specific cards (turn/combat start/end hooks, orb passives like Loop, static Strength/Dexterity/Focus/hand_draw/max_energy/TemporaryStrengthPower).
     - `trigger`: triggered by specific cards/play/discard/exhaust/arrow actions or modifying specific card subsets.
   - `duration.kind` (for powers):
     - `"this_turn"`: removed at the end of the current turn (`AfterTurnEnd` calling `PowerCmd.Remove(this)` without decrementing turns, e.g., `TemporaryStrengthPower` subclasses or 1-turn debuffs).
     - `"turns_decrement"`: decrements by 1 each turn (`PowerCmd.Decrement`, e.g., Vulnerable/Weak).
     - `"combat"`: persists for the whole combat (no remove/decrement hook).
     - `"until_triggered"`: consumed when triggered (e.g., Aim, Stretch, Buffer, Vigor).
7. **Relic `obtain` and `between_fights` conventions:**
   - `obtain`: executed in `AfterObtained`/`SetupForPlayer`.
     - `offer_rewards`: MUST have `"rewards": [{{"type": "potion"}}, ...]` (one object per reward; card rewards include `"options"` and `"pool"`).
     - `gain_relic`: `{{"kind": "gain_relic", "count": n, "source": "next_in_pool"|"neow_options"|"<rarity>"}}`.
     - `add_card`: `{{"kind": "add_card", "count": n, "destination": "deck", "pool": {{...}}, "upgraded": true}}`.
     - `gain_gold`/`lose_gold`/`gain_max_hp`/`heal`/`remove_from_deck`/`transform_cards`/`upgrade_cards`/`enchant_cards`/`gain_potion`/`potion_slots`/`replace_relic`/`remove_relic`.
   - `between_fights`: out-of-combat ongoing hooks (`when` in `BETWEEN_WHEN`, or static `modifier` in `BETWEEN_MODIFIERS` without `when`). Merchant upgrades use `"when": "merchant"` + `upgrade_cards` with `"scope": "merchant_cards"`.
8. **Status & unknowable:**
   - Put items in `unknowable` (`[{{"what": "...", "reason": "..."}}]`) ONLY for facts genuinely unknowable at card-selection time (e.g., unknown future enemy intent or multiplayer-dependent ally actions). **NEVER** mark random card/relic/potion generation from a known game pool as `unknowable` (random pools are modeled by expectation!).
   - If a mechanic genuinely requires a NEW vocabulary kind/when/modifier not in the allowed list below, do NOT invent an illegal keyword—set `needs_new_vocabulary=true` when submitting and explain what is missing in `open_questions`.
9. **Fool (v0.113.1) conventions:**
   - Tags: `"Shoot"`, `"Wah"`, `"Cho"`, `"Ki"` (check `get_Tags` overrides as well as `CanonicalTags`).
   - Shots: `damage` with `"source": "arrow"`. Extra slot: `arrow_slots`. Loosen: `dislodge_arrow` (`count`: number or `"all"`, `owner_only`).
   - Arrow/impale events: `when` in `arrow_shot` / `arrow_impaled` / `arrow_dislodged`; `filter.from_arrow: true`; `condition.target_impaled` (`"any"` / `"owner"`).
   - Aim/Stretch: `apply_power` `"Aim"` / `"Stretch"`. Percentage per stack: `multiply_percent_per_stack`.
10. **`evidence`:** Non-empty list of concise factual strings citing concrete method names, `CanonicalVars`, `OnUpgrade`, and key command calls observed in IL.

## Allowed Vocabulary (strictly enforced by validator)
- `EFFECT_KINDS`: {json.dumps(canonical_effect_kinds)}
- `WHEN` (combat mechanics): {json.dumps(vocab["when"])}
- `MODIFIERS` (combat static modifiers): {json.dumps(vocab["modifiers"])}
- `CATEGORIES`: {json.dumps(vocab["categories"])}
- `OBTAIN_KINDS` (relic `obtain` & `between_fights` effects): {json.dumps(vocab["obtain_kinds"])}
- `BETWEEN_WHEN`: {json.dumps(vocab["between_when"])}
- `BETWEEN_MODIFIERS`: {json.dumps(vocab["between_modifiers"])}
- `REVIEW_FIELDS`: {json.dumps(vocab["review_fields"])}

{VERIFIED_ENUMS}

{CANONICAL_EXAMPLES}"""


def build_drafter_system_prompt() -> str:
    return (
        _shared_spec_block()
        + """

## Your Role: Drafter Agent
- Read the pre-fetched extract metadata and IL dump carefully.
- Only call `dump_class`, `read_enum`, or `lookup_fact` when the pre-fetched IL references a helper class, base class (e.g. `TemporaryStrengthPower`), generated card, referenced power, or unknown enum constant that affects the mechanic.
- When ready, call `submit_fact` with the complete fact object. The validator runs immediately; if it returns validation problems, fix every reported problem and call `submit_fact` again.
- Keep `entity_key` and `fingerprint` identical to the input."""
    )


def build_auditor_system_prompt() -> str:
    return (
        _shared_spec_block()
        + """

## Your Role: Independent Adversarial Auditor Agent
You are an adversarial code auditor. Treat the candidate fact as UNTRUSTED—drafts frequently contain subtle IL misreadings (wrong sign, missed upgrade, wrong duration kind, wrong category, wrong orb/pile index, missed second effect in MoveNext, or omitted obtain/between_fights hook).

Follow this strict two-step verification protocol:
1. **Read the IL FIRST independently** (ignore the draft while reading IL; call `dump_class` on base/helper classes or `read_enum` if anything is unclear):
   - Record exact cost/upgrade cost, `CanonicalVars`, `OnUpgrade` deltas, keywords, tags.
   - Trace EVERY method in `overrides` and its nested `<Method>d__N::MoveNext` state machine.
   - Verify exact sign (`LoseEnergy` = negative, `GainEnergy` = positive), target/owner (`Owner` vs `Applier` vs `Dealer`), index (`[0]` = first/front), `category` (`baseline` vs `trigger` vs `irrelevant`), and `duration.kind` (`this_turn` vs `turns_decrement` vs `combat` vs `until_triggered`).
   - For relics, check both combat `mechanics` AND out-of-combat `obtain` / `between_fights`.
2. **Diff your independent IL reading against the candidate fact** field by field.
   - If ANY factual mismatch, wrong sign/number/upgrade/cost, wrong category/duration, wrong target/index, missed hook, or missed obtain/between_fights effect exists -> call `submit_audit` with `verdict="disagree"` and list EVERY concrete issue with IL citations.
   - Only return `verdict="agree"` when EVERY factual field matches your independent IL reading. Ignore only trivial differences in `evidence` prose wording.

Finish by calling `submit_audit` (populate `il_reading` FIRST with your independent IL trace, then `issues` and `verdict`)."""
    )


def build_entity_context(
    *,
    entity_key: str,
    extract_entry: Mapping[str, Any],
    facts_entry: Mapping[str, Any],
    il_dump: str,
    catalog_entry: Optional[Mapping[str, Any]] = None,
) -> str:
    """Assemble the pre-fetched entity dossier shared by Drafter and Auditor."""
    compact_extract = {
        "class": extract_entry.get("class"),
        "kind": extract_entry.get("kind"),
        "namespace": extract_entry.get("namespace"),
        "base": extract_entry.get("base"),
        "overrides": extract_entry.get("overrides", []),
        "calls": extract_entry.get("calls", []),
        "canonical_vars": extract_entry.get("canonical_vars", []),
        "canonical_tags": extract_entry.get("canonical_tags", []),
        "card_ctor": extract_entry.get("card_ctor"),
        "upgrade_deltas": extract_entry.get("upgrade_deltas", []),
        "card_type_checks": extract_entry.get("card_type_checks", []),
        "strings": extract_entry.get("strings", []),
        "fingerprint": extract_entry.get("fingerprint"),
    }
    header_parts = [
        f"## Target Entity: `{entity_key}`",
        f"- Class: `{extract_entry.get('class')}`",
        f"- Fingerprint: `{extract_entry.get('fingerprint')}`",
    ]
    if "triage" in facts_entry:
        header_parts.append(f"- Hook triage hint: `{json.dumps(facts_entry['triage'], ensure_ascii=False)}`")

    sections = [
        "\n".join(header_parts),
        "### Pre-fetched Extract Metadata\n```json\n"
        + json.dumps(compact_extract, ensure_ascii=False, indent=1)
        + "\n```",
    ]
    if catalog_entry is not None:
        compact_cat = {
            k: v
            for k, v in catalog_entry.items()
            if k
            in (
                "id",
                "name",
                "description",
                "upgrade_description",
                "cost",
                "is_x_cost",
                "star_cost",
                "type",
                "type_key",
                "rarity",
                "color",
                "target",
                "damage",
                "block",
                "hit_count",
                "cards_draw",
                "energy_gain",
                "keywords",
                "tags",
                "vars",
                "pool",
            )
            and v is not None
        }
        sections.append(
            "### Spire Codex Catalog Entry (CROSS-CHECK ONLY — IL WINS ON CONFLICT)\n```json\n"
            + json.dumps(compact_cat, ensure_ascii=False, indent=1)
            + "\n```"
        )
    sections.append(f"### Pre-fetched Assembly IL (`{extract_entry.get('class')}`)\n```il\n{il_dump.strip()}\n```")
    return "\n\n".join(sections)


def build_revision_prompt(
    entity_context: str,
    previous_draft: Mapping[str, Any],
    audit_issues: Sequence[str],
    audit_notes: str,
) -> str:
    issues_md = "\n".join(f"- {item}" for item in audit_issues) if audit_issues else "- (see auditor notes)"
    return (
        f"{entity_context}\n\n"
        "### Previous Draft (Rejected by Independent Auditor)\n```json\n"
        + json.dumps(previous_draft, ensure_ascii=False, indent=1)
        + "\n```\n\n"
        "### Independent Auditor Findings\n"
        f"{issues_md}\n\n"
        f"Auditor notes: {audit_notes}\n\n"
        "Re-check the IL carefully against EVERY auditor point (call `dump_class` / `read_enum` if needed). "
        "Fix every confirmed error and submit the complete corrected fact via `submit_fact`. "
        "Only disagree with an auditor point if the IL clearly proves your reading, and document the exact IL evidence."
    )


def build_auditor_prompt(
    entity_context: str,
    candidate_fact: Mapping[str, Any],
    codex_conflicts: Sequence[str],
) -> str:
    conflict_block = ""
    if codex_conflicts:
        conflict_lines = "\n".join(f"- {c}" for c in codex_conflicts)
        conflict_block = (
            "\n\n### Automated Spire Codex Cross-Check Warnings\n"
            "Verify from IL whether the candidate fact is right (Codex wrong) or the candidate fact has a bug:\n"
            f"{conflict_lines}"
        )
    return (
        f"{entity_context}{conflict_block}\n\n"
        "### Candidate Fact to Audit\n```json\n"
        + json.dumps(candidate_fact, ensure_ascii=False, indent=1)
        + "\n```\n\n"
        "Independently verify every field against the IL and submit your verdict via `submit_audit`."
    )


_SHARED_READ_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dump_class",
            "description": (
                "Dump the .NET IL of another game class (helper class, base class, Power class, "
                "generated Card class, or command type) from the read-only assembly extractor."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "Simple or full C# class name, e.g. 'ThornsPower' or 'ArrowHelper'.",
                    }
                },
                "required": ["class_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_enum",
            "description": "Read exact integer-to-name values of a game enum from the assembly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enum_name": {
                        "type": "string",
                        "description": "Simple enum name, e.g. 'CardTag', 'PowerStackType', 'ValueProp'.",
                    }
                },
                "required": ["enum_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_fact",
            "description": "Inspect an existing reviewed fact in facts.json to follow established conventions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_key": {
                        "type": "string",
                        "description": "Entity key such as 'powers:THORNS' or 'cards:STRIKE_IRONCLAD'.",
                    }
                },
                "required": ["entity_key"],
            },
        },
    },
]

DRAFTER_TOOLS: List[Dict[str, Any]] = [
    *_SHARED_READ_TOOLS,
    {
        "type": "function",
        "function": {
            "name": "submit_fact",
            "description": (
                "Submit the complete structured fact JSON for deterministic validation. "
                "If validation fails, errors are returned so you can fix and re-submit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "object",
                        "description": "Complete fact object following the mechanic-facts specification.",
                    },
                    "needs_new_vocabulary": {
                        "type": "boolean",
                        "description": "Set true ONLY if the IL mechanic cannot be expressed without adding new schema vocabulary.",
                    },
                    "vocabulary_gap_note": {
                        "type": "string",
                        "description": "Short explanation when needs_new_vocabulary is true.",
                    },
                },
                "required": ["fact"],
            },
        },
    },
]

AUDITOR_TOOLS: List[Dict[str, Any]] = [
    *_SHARED_READ_TOOLS,
    {
        "type": "function",
        "function": {
            "name": "submit_audit",
            "description": "Submit the independent audit verdict after writing your own IL reading and comparing every field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "il_reading": {
                        "type": "string",
                        "description": "Your independent summary of what the IL actually does (costs, signs, numbers, hooks, category, duration, obtain/between_fights) BEFORE deciding verdict.",
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete list of factual errors/omissions found by comparing il_reading against the candidate fact (empty if agree).",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["agree", "disagree"],
                        "description": "'agree' ONLY if every factual field matches il_reading; 'disagree' if any factual error or omission exists.",
                    },
                    "needs_new_vocabulary": {
                        "type": "boolean",
                        "description": "Set true ONLY if expressing this IL mechanic faithfully requires adding new schema vocabulary.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Concise IL verification summary (written into audit.notes when agreed).",
                    },
                },
                "required": ["il_reading", "issues", "verdict", "notes"],
            },
        },
    },
]
