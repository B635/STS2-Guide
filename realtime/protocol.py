"""Versioned contract between the read-only game mod and the local agent."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


MIN_SUPPORTED_SCHEMA_VERSION = 1
CURRENT_SCHEMA_VERSION = 3


class EventType(str, Enum):
    CARD_REWARD = "card_reward"
    DECISION_CLOSED = "decision_closed"
    MAP_CHOICE = "map_choice"
    MERCHANT = "merchant"
    REST_SITE = "rest_site"
    RUN_ENDED = "run_ended"


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

    card: str = Field(..., min_length=1)
    upgrades: int = Field(default=0, ge=0, le=9)
    enchantment: Optional[str] = Field(default=None, max_length=200)
    enchantment_amount: Optional[int] = Field(default=None, ge=0)
    affliction: Optional[str] = Field(default=None, max_length=200)
    affliction_amount: Optional[int] = Field(default=None, ge=0)


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
    available_next_node_ids: List[str] = Field(default_factory=list)
    boss_node_ids: List[str] = Field(default_factory=list)


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
    run_id: str = Field(..., min_length=3, max_length=200)
    sequence: int = Field(..., ge=1)
    state: RunStateSnapshot
    options: List[DecisionOption] = Field(default_factory=list, max_length=10)
    decision: Optional[DecisionContext] = None
    parent_event_id: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=200,
    )
    outcome: Optional[DecisionOutcome] = None
    map_context: Optional[MapChoiceContext] = None
    run_result: Optional[RunResult] = None
