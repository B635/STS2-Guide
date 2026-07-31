"""Versioned contract between the read-only game mod and the local agent."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator


MIN_SUPPORTED_SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = 9
STRICT_SNAPSHOT_SCHEMA_VERSION = 6
AUTHORITATIVE_SNAPSHOT_SCHEMA_VERSION = 7


def _expect_json_type(
    payload: Mapping[str, Any],
    field: str,
    expected: type | tuple[type, ...],
    *,
    path: str,
    required: bool = False,
) -> Any:
    """Validate an unparsed JSON value without Python/Pydantic coercion."""
    if field not in payload:
        if required:
            raise ValueError(f"{path}.{field} is required")
        return None
    value = payload[field]
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if value is None and type(None) in expected_types:
        return value
    # bool is an int subclass in Python, while JSON Schema treats it as a
    # distinct primitive.  Exact type comparison preserves that distinction.
    if type(value) not in expected_types:
        names = "/".join(item.__name__ for item in expected_types)
        raise ValueError(f"{path}.{field} must be JSON {names}")
    return value


def _expect_object(value: Any, *, path: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{path} must be a JSON object")
    return value


def _expect_array(value: Any, *, path: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{path} must be a JSON array")
    return value


def _require_nonblank_id(value: Any, *, path: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path} must be a non-empty ID")


def _reject_duplicate_values(values: list[Any], *, path: str) -> None:
    for index, value in enumerate(values):
        if value in values[:index]:
            raise ValueError(f"{path} must contain unique values")


def _validate_json_annotation(annotation: Any, value: Any, *, path: str) -> None:
    """Apply JSON primitive semantics to one resolved Pydantic annotation."""
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Union:
        if value is None and type(None) in arguments:
            return
        candidates = tuple(item for item in arguments if item is not type(None))
        errors = []
        for candidate in candidates:
            try:
                _validate_json_annotation(candidate, value, path=path)
                return
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(errors[0] if errors else f"{path} has invalid JSON type")
    if origin is list:
        values = _expect_array(value, path=path)
        item_type = arguments[0] if arguments else Any
        for index, item in enumerate(values):
            _validate_json_annotation(item_type, item, path=f"{path}[{index}]")
        return
    if annotation is Any:
        return
    if annotation is datetime:
        if type(value) is not str:
            raise ValueError(f"{path} must be a JSON datetime string")
        return
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        value_type = type(next(iter(annotation)).value)
        if type(value) is not value_type:
            raise ValueError(f"{path} must be JSON {value_type.__name__}")
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _validate_model_json_types(
            annotation,
            _expect_object(value, path=path),
            path=path,
        )
        return
    if annotation in (str, int, bool, float):
        if type(value) is not annotation:
            raise ValueError(f"{path} must be JSON {annotation.__name__}")


def _validate_model_json_types(
    model: type[BaseModel],
    payload: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Recursively reject coercion for fields supplied by a live producer."""
    for name, value in payload.items():
        field = model.model_fields.get(name)
        if field is not None:
            _validate_json_annotation(
                field.annotation,
                value,
                path=f"{path}.{name}",
            )


def _validate_v6_ids_and_uniqueness(payload: Mapping[str, Any]) -> None:
    """Validate semantic ID shapes that primitive type checking cannot cover."""
    for field in ("event_id", "run_id"):
        _require_nonblank_id(payload.get(field), path=f"$.{field}")
    for field in ("decision_id", "parent_event_id"):
        value = payload.get(field)
        if value is not None:
            _require_nonblank_id(value, path=f"$.{field}")

    state = _expect_object(payload.get("state"), path="$.state")
    _require_nonblank_id(state.get("character"), path="$.state.character")
    for index, card in enumerate(state.get("deck", [])):
        _require_nonblank_id(
            card.get("card"),
            path=f"$.state.deck[{index}].card",
        )
    for index, relic_id in enumerate(state.get("relics", [])):
        _require_nonblank_id(
            relic_id,
            path=f"$.state.relics[{index}]",
        )
    for index, relic in enumerate(state.get("relic_states", [])):
        _require_nonblank_id(
            relic.get("relic"),
            path=f"$.state.relic_states[{index}].relic",
        )
    for index, potion in enumerate(state.get("potions", [])):
        _require_nonblank_id(
            potion.get("potion"),
            path=f"$.state.potions[{index}].potion",
        )

    options = payload.get("options", [])
    candidate_ids: list[str] = []
    for index, option in enumerate(options):
        _require_nonblank_id(option.get("card"), path=f"$.options[{index}].card")
        candidate_id = option.get("candidate_id")
        if candidate_id is not None:
            _require_nonblank_id(
                candidate_id,
                path=f"$.options[{index}].candidate_id",
            )
            candidate_ids.append(candidate_id)
        if option in options[:index]:
            raise ValueError("$.options must contain unique values")
    _reject_duplicate_values(candidate_ids, path="$.options.candidate_id")

    candidates = payload.get("candidates", [])
    generic_candidate_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_id = candidate.get("candidate_id")
        _require_nonblank_id(
            candidate_id,
            path=f"$.candidates[{index}].candidate_id",
        )
        generic_candidate_ids.append(candidate_id)
        entity_id = candidate.get("entity_id")
        if entity_id is not None:
            _require_nonblank_id(
                entity_id,
                path=f"$.candidates[{index}].entity_id",
            )
    _reject_duplicate_values(
        generic_candidate_ids,
        path="$.candidates.candidate_id",
    )

    outcome = payload.get("outcome")
    if outcome is not None and outcome.get("selected_candidate_id") is not None:
        _require_nonblank_id(
            outcome["selected_candidate_id"],
            path="$.outcome.selected_candidate_id",
        )

    context = payload.get("map_context")
    if context is None:
        return
    for field in ("current_node_id", "origin_node_id"):
        value = context.get(field)
        if value is not None:
            _require_nonblank_id(value, path=f"$.map_context.{field}")
    for field in (
        "available_next_node_ids",
        "boss_node_ids",
        "boss_encounter_ids",
    ):
        values = context.get(field, [])
        for index, value in enumerate(values):
            _require_nonblank_id(
                value,
                path=f"$.map_context.{field}[{index}]",
            )
        _reject_duplicate_values(values, path=f"$.map_context.{field}")

    node_ids: list[str] = []
    nodes = context.get("nodes", [])
    for index, node in enumerate(nodes):
        node_id = node.get("node_id")
        _require_nonblank_id(
            node_id,
            path=f"$.map_context.nodes[{index}].node_id",
        )
        node_ids.append(node_id)
        edges = node.get("edges", [])
        for edge_index, edge in enumerate(edges):
            _require_nonblank_id(
                edge,
                path=f"$.map_context.nodes[{index}].edges[{edge_index}]",
            )
        _reject_duplicate_values(
            edges,
            path=f"$.map_context.nodes[{index}].edges",
        )
        if node in nodes[:index]:
            raise ValueError("$.map_context.nodes must contain unique values")
    _reject_duplicate_values(node_ids, path="$.map_context.nodes.node_id")


class EventType(str, Enum):
    CARD_REWARD = "card_reward"
    DECISION_CLOSED = "decision_closed"
    MAP_CHOICE = "map_choice"
    ROUTE_CHOICE = "route_choice"
    MERCHANT = "merchant"
    REST_SITE = "rest_site"
    NEOW_CHOICE = "neow_choice"
    EVENT_CHOICE = "event_choice"
    DECK_EDIT = "deck_edit"
    RUN_ENDED = "run_ended"


class RouteMode(str, Enum):
    BALANCED = "balanced"
    SURVIVAL = "survival"
    GROWTH = "growth"


class GuidePreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_mode: RouteMode


class DeckCardState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card: str = Field(..., min_length=1)
    count: int = Field(default=1, ge=1, le=99)
    upgrades: int = Field(default=0, ge=0, le=9)
    enchantment: Optional[str] = Field(default=None, max_length=200)
    enchantment_amount: Optional[int] = Field(default=None, ge=0)
    affliction: Optional[str] = Field(default=None, max_length=200)
    affliction_amount: Optional[int] = Field(default=None, ge=0)


class PotionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    potion: str = Field(..., min_length=1)
    slot: int = Field(..., ge=0, le=99)


class RelicState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relic: str = Field(..., min_length=1)
    display_amount: Optional[int] = None
    stack_count: int = Field(default=1, ge=0)
    status: Optional[str] = Field(default=None, max_length=100)


class RunStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character: str = Field(..., min_length=1)
    ascension: int = Field(default=0, ge=0)
    act: int = Field(default=1, ge=1)
    floor: int = Field(default=0, ge=0)
    hp: Optional[int] = Field(default=None, ge=0)
    max_hp: Optional[int] = Field(default=None, ge=1)
    gold: Optional[int] = Field(default=None, ge=0)
    energy: int = Field(default=3, ge=0, le=99)
    deck: List[DeckCardState] = Field(default_factory=list)
    relics: List[str] = Field(default_factory=list, max_length=99)
    relic_states: List[RelicState] = Field(
        default_factory=list,
        max_length=99,
    )
    potions: List[PotionState] = Field(default_factory=list, max_length=99)
    max_potion_slots: Optional[int] = Field(default=None, ge=0, le=99)
    modifiers: List[str] = Field(default_factory=list, max_length=99)
    capture_warnings: List[str] = Field(default_factory=list)


class DecisionOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: Optional[str] = Field(default=None, min_length=1, max_length=240)
    card: str = Field(..., min_length=1)
    upgrades: int = Field(default=0, ge=0, le=9)
    enchantment: Optional[str] = Field(default=None, max_length=200)
    enchantment_amount: Optional[int] = Field(default=None, ge=0)
    affliction: Optional[str] = Field(default=None, max_length=200)
    affliction_amount: Optional[int] = Field(default=None, ge=0)


class CandidateKind(str, Enum):
    CARD = "card"
    SKIP = "skip"
    ROUTE_NODE = "route_node"
    MERCHANT_OFFER = "merchant_offer"
    LEAVE = "leave"
    REST_ACTION = "rest_action"
    DECK_EDIT = "deck_edit"
    NEOW_BLESSING = "neow_blessing"
    EVENT_OPTION = "event_option"


class DecisionCostKind(str, Enum):
    GOLD = "gold"
    HP = "hp"
    MAX_HP = "max_hp"
    ENERGY = "energy"
    RESOURCE = "resource"


class DecisionCost(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: DecisionCostKind
    amount: int = Field(..., ge=0)
    resource_id: Optional[str] = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def _validate_resource_id(self) -> "DecisionCost":
        if (
            self.kind == DecisionCostKind.RESOURCE
            and self.resource_id is None
        ):
            raise ValueError("resource cost requires resource_id")
        if (
            self.kind != DecisionCostKind.RESOURCE
            and self.resource_id is not None
        ):
            raise ValueError("built-in cost kinds cannot set resource_id")
        return self


class EffectCertainty(str, Enum):
    EXACT = "exact"
    BOUNDED = "bounded"
    UNKNOWN = "unknown"


class EffectTargetMode(str, Enum):
    NONE = "none"
    SPECIFIC = "specific"
    CHOOSE = "choose"
    RANDOM = "random"


class ChoiceEffect(BaseModel):
    """A structured game effect; descriptions are never parsed at runtime."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        ...,
        pattern=(
            "^(hp_delta|max_hp_delta|gold_delta|add_card|remove_card|"
            "upgrade_card|transform_card|add_relic|remove_relic|"
            "add_potion|remove_potion|start_combat|followup_choice|no_op)$"
        ),
    )
    amount: Optional[Union[int, float]] = None
    min_amount: Optional[Union[int, float]] = None
    max_amount: Optional[Union[int, float]] = None
    entity_type: Optional[str] = Field(default=None, min_length=1, max_length=80)
    entity_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    target_mode: EffectTargetMode
    certainty: EffectCertainty
    source_code: str = Field(..., min_length=1, max_length=200)
    child_decision_type: Optional[Literal["card_reward"]] = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "ChoiceEffect":
        numeric = {"hp_delta", "max_hp_delta", "gold_delta"}
        entity_types = {
            "add_card": "cards",
            "remove_card": "cards",
            "upgrade_card": "cards",
            "transform_card": "cards",
            "add_relic": "relics",
            "remove_relic": "relics",
            "add_potion": "potions",
            "remove_potion": "potions",
        }
        if self.kind in numeric:
            if self.certainty == EffectCertainty.EXACT and self.amount is None:
                raise ValueError("exact numeric effect requires amount")
            if self.certainty == EffectCertainty.BOUNDED and (
                self.min_amount is None or self.max_amount is None
            ):
                raise ValueError("bounded numeric effect requires bounds")
            for value in (self.amount, self.min_amount, self.max_amount):
                if value is not None and not float(value).is_integer():
                    raise ValueError(
                        "HP and gold effects require integer amounts"
                    )
        if (
            self.min_amount is not None
            and self.max_amount is not None
            and self.min_amount > self.max_amount
        ):
            raise ValueError("effect amount bounds are inverted")
        if self.kind in entity_types:
            if self.entity_type != entity_types[self.kind]:
                raise ValueError(
                    f"{self.kind} requires "
                    f"entity_type={entity_types[self.kind]}"
                )
            if self.target_mode == EffectTargetMode.NONE:
                raise ValueError(
                    "entity effect requires an explicit target mode"
                )
            if (
                self.target_mode == EffectTargetMode.SPECIFIC
                and self.entity_id is None
            ):
                raise ValueError("specific entity effect requires entity_id")
        if self.child_decision_type is not None:
            if self.kind != "followup_choice":
                raise ValueError(
                    "child_decision_type is only valid for followup_choice"
                )
            if self.certainty != EffectCertainty.EXACT:
                raise ValueError(
                    "child_decision_type requires exact effect certainty"
                )
        return self


class CandidatePayload(BaseModel):
    """Closed payload vocabulary shared by every v9 candidate kind."""

    model_config = ConfigDict(extra="forbid")

    card: Optional[str] = Field(default=None, min_length=1, max_length=200)
    upgrades: Optional[int] = Field(default=None, ge=0, le=9)
    enchantment: Optional[str] = Field(default=None, max_length=200)
    enchantment_amount: Optional[int] = Field(default=None, ge=0)
    affliction: Optional[str] = Field(default=None, max_length=200)
    affliction_amount: Optional[int] = Field(default=None, ge=0)
    node_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    slot_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    offer_kind: Optional[str] = Field(
        default=None,
        pattern="^(card|relic|potion|card_removal)$",
    )
    is_stocked: Optional[bool] = None
    replacement_supported: Optional[bool] = None
    action_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    operation: Optional[str] = Field(
        default=None,
        pattern="^(upgrade|remove|transform)$",
    )
    blessing_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    stage_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    event_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    page_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    option_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    target_candidate_ids: List[
        Annotated[
            str,
            Field(min_length=1, max_length=240, pattern=r".*\S.*"),
        ]
    ] = Field(default_factory=list, max_length=200)
    effects: List[ChoiceEffect] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _validate_target_candidate_ids(self) -> "CandidatePayload":
        if len(self.target_candidate_ids) != len(
            set(self.target_candidate_ids)
        ):
            raise ValueError("target_candidate_ids must be unique")
        return self


class DecisionCandidateEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1, max_length=240)
    kind: CandidateKind
    entity_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    label: str = Field(..., min_length=1, max_length=300)
    eligible: bool
    unavailable_reason: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=300,
    )
    costs: List[DecisionCost] = Field(..., max_length=8)
    payload: CandidatePayload

    @model_validator(mode="after")
    def _validate_kind_payload(self) -> "DecisionCandidateEnvelope":
        present = self.payload.model_fields_set
        required: dict[CandidateKind, set[str]] = {
            CandidateKind.CARD: {"card", "upgrades"},
            CandidateKind.SKIP: set(),
            CandidateKind.ROUTE_NODE: {"node_id"},
            CandidateKind.MERCHANT_OFFER: {
                "slot_id",
                "offer_kind",
                "is_stocked",
            },
            CandidateKind.LEAVE: set(),
            CandidateKind.REST_ACTION: {"action_id"},
            CandidateKind.DECK_EDIT: {"operation"},
            CandidateKind.NEOW_BLESSING: {
                "blessing_id",
                "stage_id",
                "option_id",
                "effects",
            },
            CandidateKind.EVENT_OPTION: {
                "event_id",
                "page_id",
                "option_id",
                "effects",
            },
        }
        missing = required[self.kind] - present
        if missing:
            raise ValueError(
                f"{self.kind.value} candidate missing payload fields: "
                f"{sorted(missing)}"
            )
        meaningful = {
            name
            for name in CandidatePayload.model_fields
            if (
                (value := getattr(self.payload, name)) is not None
                and value != []
            )
        }
        allowed: dict[CandidateKind, set[str]] = {
            CandidateKind.CARD: {
                "card",
                "upgrades",
                "enchantment",
                "enchantment_amount",
                "affliction",
                "affliction_amount",
            },
            CandidateKind.SKIP: set(),
            CandidateKind.ROUTE_NODE: {"node_id"},
            CandidateKind.MERCHANT_OFFER: {
                "slot_id",
                "offer_kind",
                "is_stocked",
                "replacement_supported",
                "effects",
            },
            CandidateKind.LEAVE: set(),
            CandidateKind.REST_ACTION: {
                "action_id",
                "replacement_supported",
                "effects",
            },
            CandidateKind.DECK_EDIT: {
                "operation",
                "target_candidate_ids",
            },
            CandidateKind.NEOW_BLESSING: {
                "blessing_id",
                "stage_id",
                "option_id",
                "replacement_supported",
                "effects",
            },
            CandidateKind.EVENT_OPTION: {
                "event_id",
                "page_id",
                "option_id",
                "replacement_supported",
                "effects",
            },
        }
        unexpected = meaningful - allowed[self.kind]
        if unexpected:
            raise ValueError(
                f"{self.kind.value} candidate has unrelated payload "
                f"fields: {sorted(unexpected)}"
            )
        if self.kind == CandidateKind.MERCHANT_OFFER:
            gold_costs = [
                cost for cost in self.costs
                if cost.kind == DecisionCostKind.GOLD
            ]
            if len(gold_costs) != 1:
                raise ValueError(
                    "merchant offer requires exactly one actual gold cost"
                )
            if self.payload.is_stocked is False and self.eligible:
                raise ValueError("sold-out merchant offer cannot be eligible")
            if (
                self.payload.offer_kind != "card_removal"
                and self.entity_id is None
            ):
                raise ValueError(
                    "merchant entity offer requires entity_id"
                )
            if (
                self.payload.offer_kind == "card_removal"
                and self.entity_id is not None
            ):
                raise ValueError(
                    "merchant card removal cannot carry entity_id"
                )
        if self.kind in {
            CandidateKind.NEOW_BLESSING,
            CandidateKind.EVENT_OPTION,
        } and not self.payload.effects:
            raise ValueError(
                f"{self.kind.value} candidate requires a structured "
                "effect, including an explicit unknown effect"
            )
        if not self.eligible and self.unavailable_reason is None:
            raise ValueError(
                "ineligible candidate requires unavailable_reason"
            )
        if self.eligible and self.unavailable_reason is not None:
            raise ValueError(
                "eligible candidate cannot set unavailable_reason"
            )
        if self.kind == CandidateKind.SKIP and (
            self.entity_id is not None or self.costs
        ):
            raise ValueError("skip candidate cannot carry entity or cost")
        if self.kind == CandidateKind.LEAVE and (
            self.entity_id is not None or self.costs
        ):
            raise ValueError("leave candidate cannot carry entity or cost")
        if self.kind == CandidateKind.CARD and (
            self.entity_id != self.payload.card
        ):
            raise ValueError(
                "card candidate entity_id must match payload.card"
            )
        if self.kind == CandidateKind.ROUTE_NODE and (
            self.entity_id != self.payload.node_id
        ):
            raise ValueError(
                "route candidate entity_id must match payload.node_id"
            )
        if (
            self.kind == CandidateKind.DECK_EDIT
            and self.entity_id is None
        ):
            raise ValueError("deck_edit candidate requires entity_id")
        return self


class DecisionParentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(..., min_length=3, max_length=240)
    candidate_id: Optional[str] = Field(default=None, min_length=1, max_length=240)
    source_type: str = Field(
        ...,
        pattern="^(neow_choice|event_choice|rest_site|merchant)$",
    )
    source_id: str = Field(..., min_length=1, max_length=200)


class DecisionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    can_skip: bool = True
    can_reroll: bool = False
    reward_source: Optional[str] = Field(default=None, max_length=200)


class DecisionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        ...,
        pattern="^(selected|skipped|closed_unknown)$",
    )
    selected_candidate_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=240,
    )
    selected_card: Optional[str] = Field(default=None, max_length=200)
    selected_option_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=9,
    )


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str = Field(..., pattern="^(win|loss|abandon)$")
    final_score: Optional[int] = Field(default=None, ge=0)
    started_at: Optional[datetime] = None
    ended_at: datetime


class MapNodeState(BaseModel):
    """A single node on the act map."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(..., min_length=1)
    kind: str = Field(
        ...,
        pattern="^(MONSTER|ELITE|CAMPFIRE|SHOP|EVENT|BOSS|TREASURE|UNKNOWN)$",
    )
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    edges: List[str] = Field(default_factory=list)
    label: Optional[str] = Field(default=None, max_length=200)


class MapChoiceContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map_name: Optional[str] = Field(default=None, max_length=100)
    player_row: Optional[int] = Field(default=None, ge=0)
    nodes: List[MapNodeState] = Field(default_factory=list)
    node_count: int = Field(default=0, ge=0)
    current_node_id: Optional[str] = Field(default=None, min_length=1)
    origin_node_id: Optional[str] = Field(default=None, min_length=1)
    available_next_node_ids: List[str] = Field(default_factory=list)
    boss_node_ids: List[str] = Field(default_factory=list)
    boss_encounter_ids: List[str] = Field(default_factory=list)


class GameStateEvent(BaseModel):
    """One immutable observation emitted by the read-only mod."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(
        ...,
        ge=MIN_SUPPORTED_SCHEMA_VERSION,
        le=CURRENT_SCHEMA_VERSION,
    )
    event_id: str = Field(..., min_length=3, max_length=200)
    event_type: EventType
    emitted_at: datetime
    source: str = Field(default="sts2-guide-readonly-mod", min_length=1)
    game_version: Optional[str] = Field(default=None, max_length=100)
    snapshot_kind: Optional[str] = Field(default=None, max_length=40)
    state_revision: Optional[int] = Field(default=None, ge=1)
    producer_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    producer_version: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
    )
    game_assembly_sha256: Optional[str] = Field(
        default=None,
        pattern="^[0-9a-f]{64}$",
    )
    release_fingerprint: Optional[str] = Field(
        default=None,
        pattern="^[0-9a-f]{64}$",
    )
    guide_preferences: Optional[GuidePreferences] = None
    run_id: str = Field(..., min_length=3, max_length=200)
    sequence: int = Field(..., ge=1)
    decision_id: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=240,
    )
    state: RunStateSnapshot
    options: List[DecisionOption] = Field(default_factory=list, max_length=10)
    candidates: List[DecisionCandidateEnvelope] = Field(
        default_factory=list,
        max_length=200,
    )
    decision: Optional[DecisionContext] = None
    decision_parent: Optional[DecisionParentContext] = None
    parent_event_id: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=200,
    )
    outcome: Optional[DecisionOutcome] = None
    map_context: Optional[MapChoiceContext] = None
    run_result: Optional[RunResult] = None

    @model_validator(mode="before")
    @classmethod
    def _validate_live_json_types(cls, value: Any) -> Any:
        if type(value) is not dict:
            return value
        raw_version = value.get("schema_version")
        if type(raw_version) is not int:
            raise ValueError("$.schema_version must be JSON int")
        if raw_version >= STRICT_SNAPSHOT_SCHEMA_VERSION:
            _validate_model_json_types(GameStateEvent, value, path="$")
            _validate_v6_ids_and_uniqueness(value)
        return value

    @model_validator(mode="after")
    def _validate_schema_contract(self) -> "GameStateEvent":
        """Keep event-specific runtime requirements aligned with the schema.

        Older fixtures remain replayable, but their versioned requirements
        still cannot be bypassed by calling Pydantic without JSON Schema.
        """
        if (
            self.event_type == EventType.CARD_REWARD
            and self.schema_version < 9
        ):
            if "options" not in self.model_fields_set or not self.options:
                raise ValueError("card_reward requires non-empty options")
            if self.schema_version >= 2 and self.decision is None:
                raise ValueError("card_reward schema v2+ requires decision")
            if self.schema_version >= 5:
                if self.decision_id is None:
                    raise ValueError("card_reward schema v5+ requires decision_id")
                if any(option.candidate_id is None for option in self.options):
                    raise ValueError(
                        "card_reward schema v5+ requires candidate_id"
                    )

        if self.event_type == EventType.DECISION_CLOSED:
            if self.schema_version >= 2 and (
                self.parent_event_id is None or self.outcome is None
            ):
                raise ValueError(
                    "decision_closed schema v2+ requires parent and outcome"
                )
            if self.schema_version >= 5 and self.decision_id is None:
                raise ValueError(
                    "decision_closed schema v5+ requires decision_id"
                )
            if (
                self.schema_version >= 5
                and self.outcome is not None
                and self.outcome.kind == "selected"
                and self.outcome.selected_candidate_id is None
            ):
                raise ValueError(
                    "selected schema v5+ outcome requires selected_candidate_id"
                )

        if (
            self.event_type == EventType.RUN_ENDED
            and self.schema_version >= 3
            and self.run_result is None
        ):
            raise ValueError("run_ended schema v3+ requires run_result")

        if self.schema_version >= STRICT_SNAPSHOT_SCHEMA_VERSION:
            required_state = set(RunStateSnapshot.model_fields)
            if not required_state.issubset(self.state.model_fields_set):
                raise ValueError(
                    "schema v6+ requires an explicit complete run state"
                )
            if self.decision is not None and not {
                "can_skip",
                "can_reroll",
                "reward_source",
            }.issubset(self.decision.model_fields_set):
                raise ValueError(
                    "schema v6+ decision context must be explicit"
                )

        if self.schema_version >= AUTHORITATIVE_SNAPSHOT_SCHEMA_VERSION:
            required_envelope = {
                "source",
                "game_version",
                "snapshot_kind",
                "state_revision",
                "producer_id",
                "producer_version",
                "game_assembly_sha256",
                "release_fingerprint",
                "decision_id",
                "options",
                "decision",
                "parent_event_id",
                "outcome",
                "map_context",
                "run_result",
            }
            if not required_envelope.issubset(self.model_fields_set):
                raise ValueError(
                    "schema v7+ requires an explicit complete event envelope"
                )
            if self.snapshot_kind != "complete":
                raise ValueError("schema v7+ snapshot_kind must be complete")
            if self.state_revision != self.sequence:
                raise ValueError(
                    "schema v7+ state_revision must equal committed sequence"
                )
            if not self.game_version or not self.game_version.strip():
                raise ValueError("schema v7+ requires game_version")
            for name, value in (
                ("source", self.source),
                ("producer_id", self.producer_id),
                ("producer_version", self.producer_version),
                ("game_assembly_sha256", self.game_assembly_sha256),
                ("release_fingerprint", self.release_fingerprint),
            ):
                if value is None or not value.strip():
                    raise ValueError(f"schema v7+ requires {name}")
            self._validate_v7_nested_completeness()

        if self.schema_version == CURRENT_SCHEMA_VERSION:
            required_v9 = {
                "guide_preferences",
                "candidates",
                "decision_parent",
            }
            if not required_v9.issubset(self.model_fields_set):
                raise ValueError(
                    "schema v9 requires explicit guide_preferences, "
                    "candidates and decision_parent"
                )
            if self.guide_preferences is None:
                raise ValueError("schema v9 requires guide_preferences")
            if self.options:
                raise ValueError(
                    "schema v9 production events cannot use legacy options"
                )
            self._validate_v9_decision_contract()

        if self.event_type != EventType.ROUTE_CHOICE:
            return self
        # Pydantic fills optional/defaulted fields before this validator.  A
        # route event still has to contain the same explicit facts as schema
        # v6/v7, so distinguish a producer's omission from an explicit
        # empty list or false value through model_fields_set.
        required_top_level = {
            "schema_version",
            "decision_id",
            "map_context",
            "options",
            "decision",
        }
        if not required_top_level.issubset(self.model_fields_set):
            raise ValueError("route_choice omits required schema-v6+ fields")
        if self.schema_version not in (6, 7, 8, CURRENT_SCHEMA_VERSION):
            raise ValueError(
                "route_choice requires schema_version 6, 7, 8, or 9"
            )
        if self.decision_id is None:
            raise ValueError("route_choice requires decision_id")
        if self.options:
            raise ValueError("route_choice options must be empty")
        if self.decision is None:
            raise ValueError("route_choice requires decision context")
        if not {"can_skip", "can_reroll", "reward_source"}.issubset(
            self.decision.model_fields_set
        ):
            raise ValueError("route_choice decision omits required MAP fields")
        if (
            self.decision.can_skip
            or self.decision.can_reroll
            or self.decision.reward_source != "MAP"
        ):
            raise ValueError("route_choice must use the MAP non-skip decision context")
        context = self.map_context
        if (
            context is None
            or not context.nodes
            or context.node_count < 1
            or not context.origin_node_id
            or not context.available_next_node_ids
            or not context.boss_node_ids
        ):
            raise ValueError("route_choice requires a non-empty route map context")
        if "boss_encounter_ids" not in context.model_fields_set:
            raise ValueError("route_choice requires boss_encounter_ids")
        for name, identifiers, require_non_empty in (
            ("available_next_node_ids", context.available_next_node_ids, True),
            ("boss_node_ids", context.boss_node_ids, True),
            ("boss_encounter_ids", context.boss_encounter_ids, False),
        ):
            # MapChoiceContext intentionally remains permissive for legacy
            # map observations, but a v6 route event is a live production
            # contract.  Mirror the schema's minLength/uniqueItems checks
            # here so direct Pydantic callers cannot bypass them.
            if require_non_empty and not identifiers:
                raise ValueError(f"route_choice requires {name}")
            if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers):
                raise ValueError(f"route_choice {name} must contain non-empty IDs")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"route_choice {name} must be unique")
        return self

    def _validate_v9_decision_contract(self) -> None:
        decision_events = {
            EventType.CARD_REWARD,
            EventType.ROUTE_CHOICE,
            EventType.MERCHANT,
            EventType.REST_SITE,
            EventType.NEOW_CHOICE,
            EventType.EVENT_CHOICE,
            EventType.DECK_EDIT,
        }
        if self.event_type in decision_events:
            if self.decision_id is None or self.decision is None:
                raise ValueError(
                    "schema v9 decision event requires decision identity "
                    "and context"
                )
            if not self.candidates:
                raise ValueError(
                    "schema v9 decision event requires generic candidates"
                )
            if not any(candidate.eligible for candidate in self.candidates):
                raise ValueError(
                    "schema v9 decision event requires an eligible candidate"
                )
        elif self.candidates:
            raise ValueError(
                "schema v9 non-decision event candidates must be empty"
            )

        allowed_kinds = {
            EventType.CARD_REWARD: {
                CandidateKind.CARD,
                CandidateKind.SKIP,
            },
            EventType.ROUTE_CHOICE: {CandidateKind.ROUTE_NODE},
            EventType.MERCHANT: {
                CandidateKind.MERCHANT_OFFER,
                CandidateKind.LEAVE,
            },
            EventType.REST_SITE: {
                CandidateKind.REST_ACTION,
                CandidateKind.LEAVE,
            },
            EventType.NEOW_CHOICE: {CandidateKind.NEOW_BLESSING},
            EventType.EVENT_CHOICE: {
                CandidateKind.EVENT_OPTION,
                CandidateKind.LEAVE,
            },
            EventType.DECK_EDIT: {
                CandidateKind.DECK_EDIT,
                CandidateKind.LEAVE,
            },
        }
        if self.event_type in allowed_kinds:
            invalid = {
                candidate.kind for candidate in self.candidates
            } - allowed_kinds[self.event_type]
            if invalid:
                raise ValueError(
                    f"{self.event_type.value} has invalid candidate kinds: "
                    f"{sorted(item.value for item in invalid)}"
                )
        for index, candidate in enumerate(self.candidates):
            missing = (
                set(DecisionCandidateEnvelope.model_fields)
                - candidate.model_fields_set
            )
            if missing:
                raise ValueError(
                    "schema v9 candidate "
                    f"{index} omits fields: {sorted(missing)}"
                )
            for cost_index, cost in enumerate(candidate.costs):
                cost_missing = (
                    set(DecisionCost.model_fields)
                    - cost.model_fields_set
                )
                if cost_missing:
                    raise ValueError(
                        "schema v9 candidate cost "
                        f"{index}:{cost_index} omits fields: "
                        f"{sorted(cost_missing)}"
                    )
            for effect_index, effect in enumerate(
                candidate.payload.effects
            ):
                effect_missing = (
                    set(ChoiceEffect.model_fields)
                    - effect.model_fields_set
                )
                if effect_missing:
                    raise ValueError(
                        "schema v9 candidate effect "
                        f"{index}:{effect_index} omits fields: "
                        f"{sorted(effect_missing)}"
                    )
        if self.decision_parent is not None:
            missing_parent = (
                set(DecisionParentContext.model_fields)
                - self.decision_parent.model_fields_set
            )
            if missing_parent:
                raise ValueError(
                    "schema v9 decision_parent omits fields: "
                    f"{sorted(missing_parent)}"
                )

        if self.event_type == EventType.CARD_REWARD:
            card_count = sum(
                candidate.kind == CandidateKind.CARD
                for candidate in self.candidates
            )
            if card_count < 1:
                raise ValueError("card_reward requires a card candidate")
            skip_count = sum(
                candidate.kind == CandidateKind.SKIP
                for candidate in self.candidates
            )
            if skip_count > 1:
                raise ValueError("card_reward permits at most one skip")
            can_skip = bool(self.decision and self.decision.can_skip)
            if can_skip != (skip_count == 1):
                raise ValueError(
                    "card_reward skip candidate must match decision.can_skip"
                )
            reward_source = self.decision.reward_source if self.decision else None
            if self.decision_parent is None:
                if reward_source != "CARD":
                    raise ValueError(
                        "parentless card_reward requires reward_source=CARD"
                    )
            else:
                expected_nested_source = {
                    "neow_choice": "NEOW",
                    "event_choice": "EVENT",
                }.get(self.decision_parent.source_type)
                if reward_source != expected_nested_source:
                    raise ValueError(
                        "nested card_reward reward_source must match its "
                        "neow/event parent type"
                    )

        if self.event_type == EventType.ROUTE_CHOICE:
            context = self.map_context
            candidate_ids = {
                candidate.payload.node_id
                for candidate in self.candidates
            }
            if (
                context is not None
                and candidate_ids != set(context.available_next_node_ids)
            ):
                raise ValueError(
                    "route candidates must equal available next node IDs"
                )

        if self.event_type == EventType.DECK_EDIT:
            if self.decision_parent is None:
                raise ValueError("deck_edit requires decision_parent")
        elif (
            self.decision_parent is not None
            and self.event_type not in {
                EventType.CARD_REWARD,
                EventType.DECK_EDIT,
            }
        ):
            raise ValueError(
                "decision_parent is only valid on nested decisions"
            )

    def _validate_v7_nested_completeness(self) -> None:
        """Require the producer to spell out every v7 snapshot fact.

        Optional values may explicitly be ``null``.  Omission is different:
        it would let Pydantic defaults manufacture a seemingly complete
        snapshot and defeat the cross-component handshake.
        """

        def require_all(model: BaseModel, label: str) -> None:
            missing = (
                set(type(model).model_fields)
                - set(model.model_fields_set)
            )
            if missing:
                raise ValueError(
                    f"schema v7 {label} omits fields: {sorted(missing)}"
                )

        require_all(self.state, "state")
        for index, card in enumerate(self.state.deck):
            require_all(card, f"state.deck[{index}]")
        for index, relic in enumerate(self.state.relic_states):
            require_all(relic, f"state.relic_states[{index}]")
        for index, potion in enumerate(self.state.potions):
            require_all(potion, f"state.potions[{index}]")
        for index, option in enumerate(self.options):
            require_all(option, f"options[{index}]")
        if self.decision is not None:
            require_all(self.decision, "decision")
        if self.outcome is not None:
            require_all(self.outcome, "outcome")
        if self.map_context is not None:
            require_all(self.map_context, "map_context")
            for index, node in enumerate(self.map_context.nodes):
                require_all(node, f"map_context.nodes[{index}]")
        if self.run_result is not None:
            require_all(self.run_result, "run_result")
