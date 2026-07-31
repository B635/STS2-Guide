"""Idempotent processing of versioned game-state observations."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, Optional

from advisor.data_sources import LocalCardTierSource
from advisor.decision_core import (
    CARD_REWARD,
    ROUTE_CHOICE,
    ROUTE_MODES,
    DecisionCandidate,
    DecisionLifecycleManager,
    DecisionRequest,
    PolicyRegistry,
    WorldState,
)
from advisor.policies import CardRewardPolicy
from advisor.campfire import CAMPFIRE_ACTION, CampfirePolicy
from advisor.deck_edit import DECK_EDIT, DeckEditPolicy
from advisor.event_policy import EVENT_OPTION, EventPolicy
from advisor.merchant import MERCHANT_CHOICE, MerchantPolicy
from advisor.neow import NEOW_BLESSING, NeowPolicy
from advisor.route import RoutePolicy
from realtime.checkpoint import (
    ActiveRunCheckpointStore,
    can_supersede_checkpoint,
    checkpoint_record_payload_matches,
)
from realtime.compatibility import (
    CompatibilityManifest,
    EventHandshake,
    RuntimeComponents,
    assess_event_handshake,
    assess_local_runtime_compatibility,
)
from realtime.protocol import EventType, GameStateEvent
from realtime.session import TransientSessionStore
from storage.relational import RelationalRepository


class RealtimeEventValidationError(ValueError):
    """The event is valid JSON but cannot safely drive a recommendation."""


class RealtimeCompatibilityError(RealtimeEventValidationError):
    """A live event does not match the packaged compatibility manifest."""

    def __init__(self, reason_codes: tuple[str, ...]):
        self.reason_codes = reason_codes
        super().__init__(
            "Unsupported STS2 Guide runtime: "
            + ", ".join(reason_codes)
        )


class RealtimeEventProcessor:
    def __init__(
        self,
        repository: RelationalRepository,
        sessions: TransientSessionStore | None = None,
        checkpoint: Optional[ActiveRunCheckpointStore] = None,
        local_tiers: Optional[LocalCardTierSource] = None,
        policies: Optional[PolicyRegistry] = None,
        lifecycle: Optional[DecisionLifecycleManager] = None,
        compatibility_manifest: CompatibilityManifest | None = None,
        runtime_components: RuntimeComponents | None = None,
    ):
        self.repository = repository
        self.sessions = sessions or TransientSessionStore()
        self.checkpoint = checkpoint
        self.local_tiers = local_tiers
        self.compatibility_manifest = compatibility_manifest
        self.runtime_components = runtime_components
        self.local_compatibility = (
            assess_local_runtime_compatibility(
                compatibility_manifest,
                runtime_components or RuntimeComponents(),
            )
            if compatibility_manifest is not None
            else None
        )
        self.policies = policies or PolicyRegistry(
            [
                CardRewardPolicy(repository, local_tiers),
                RoutePolicy(repository),
                MerchantPolicy(repository),
                CampfirePolicy(repository),
                NeowPolicy(repository),
                EventPolicy(repository),
                DeckEditPolicy(repository),
            ]
        )
        self.lifecycle = lifecycle or DecisionLifecycleManager()
        self._closed_child_expectations: dict[str, dict] = {}
        self._active_child_bindings: dict[str, dict] = {}
        if self.checkpoint is not None:
            checkpoint_state = self.checkpoint.load()
            current_decision = (
                checkpoint_state.get("current_decision")
                if checkpoint_state is not None
                and self._checkpoint_record_matches_current_release(
                    checkpoint_state,
                    checkpoint_state.get("current_decision"),
                )
                else None
            )
            closed_decision = (
                checkpoint_state.get("closed_decision")
                if checkpoint_state is not None
                and self._checkpoint_record_matches_current_release(
                    checkpoint_state,
                    checkpoint_state.get("closed_decision"),
                )
                else None
            )
            restored_binding = None
            if current_decision is not None and (
                (current_decision.get("payload") or {}).get(
                    "decision_parent"
                )
                is not None
            ):
                restored_binding = self._restore_active_child_binding(
                    current_decision
                )
                if restored_binding is None:
                    current_decision = None
            if (
                current_decision is not None
                and current_decision.get("decision_id")
                and current_decision.get("event_id")
            ):
                current_recommendation = (
                    (current_decision.get("result") or {}).get(
                        "recommendation"
                    )
                    or {}
                )
                self.lifecycle.restore_active(
                    run_id=str(current_decision["run_id"]),
                    decision_id=str(current_decision["decision_id"]),
                    decision_type=str(
                        current_recommendation.get("decision_type")
                        or current_decision.get("decision_type")
                        or current_decision["event_type"]
                    ),
                    sequence=int(current_decision["sequence"]),
                    observation_event_id=str(current_decision["event_id"]),
                )
                if restored_binding is not None:
                    self._active_child_bindings[
                        str(current_decision["run_id"])
                    ] = restored_binding
            elif (
                closed_decision is not None
                and closed_decision.get("decision_id")
                and closed_decision.get("event_id")
            ):
                closed_result = closed_decision.get("result") or {}
                raw_outcome = closed_decision.get("outcome") or {}
                restored_outcome = closed_result.get("chosen_option")
                if restored_outcome is None:
                    if raw_outcome.get("kind") == "skipped":
                        restored_outcome = "skip"
                    else:
                        restored_outcome = (
                            raw_outcome.get("selected_candidate_id")
                            or raw_outcome.get("kind")
                        )
                self.lifecycle.restore_closed(
                    run_id=str(closed_decision["run_id"]),
                    decision_id=str(closed_decision["decision_id"]),
                    decision_type=str(
                        closed_result.get("decision_type") or CARD_REWARD
                    ),
                    sequence=int(closed_decision["sequence"]),
                    outcome=(
                        str(restored_outcome)
                        if restored_outcome is not None
                        else None
                    ),
                    observation_event_id=str(closed_decision["event_id"]),
                )
                expectation = self._restore_closed_child_expectation(
                    closed_decision
                )
                if expectation is not None:
                    self._closed_child_expectations[
                        str(closed_decision["run_id"])
                    ] = expectation

    def _checkpoint_record_matches_current_release(
        self,
        checkpoint: Dict,
        record: Dict | None,
    ) -> bool:
        """Return whether recovery state was produced by this exact release."""

        if record is None:
            return False
        payload = record.get("payload") or {}
        checkpoint_preferences = checkpoint.get("guide_preferences")
        payload_preferences = payload.get("guide_preferences")
        result_preferences = (record.get("result") or {}).get(
            "guide_preferences"
        )
        try:
            schema_version = int(payload.get("schema_version") or 0)
        except (TypeError, ValueError):
            return False
        preferences_match = schema_version < 8 or (
                isinstance(checkpoint_preferences, dict)
                and checkpoint_preferences == payload_preferences
                and checkpoint_preferences == result_preferences
                and set(checkpoint_preferences) == {"route_mode"}
                and checkpoint_preferences.get("route_mode")
                in ROUTE_MODES
            )
        if not preferences_match:
            return False
        if self.compatibility_manifest is None:
            return True
        expected = self.compatibility_manifest.release_fingerprint
        return (
            checkpoint.get("release_fingerprint") == expected
            and (record.get("payload") or {}).get("release_fingerprint")
            == expected
            and (record.get("result") or {})
            .get("compatibility", {})
            .get("release_fingerprint")
            == expected
        )

    def process(self, event: GameStateEvent) -> Dict:
        self._validate(event)
        payload = event.model_dump(mode="json")
        superseded_run_id: str | None = None
        if self.checkpoint is not None:
            checkpoint_before = self.checkpoint.load()
            if (
                checkpoint_before is not None
                and checkpoint_before.get("run_id") != event.run_id
            ):
                if not can_supersede_checkpoint(
                    checkpoint_before,
                    payload,
                ):
                    raise RealtimeEventValidationError(
                        "Stale or non-newer event from another run cannot "
                        "replace the active-run checkpoint"
                    )
                superseded_run_id = str(checkpoint_before["run_id"])
            replay = self.checkpoint.replay_result(
                payload,
                expected_release_fingerprint=(
                    self.compatibility_manifest.release_fingerprint
                    if self.compatibility_manifest is not None
                    else None
                ),
            )
            if replay is not None:
                return self._mark_compatible(
                    event,
                    self._ensure_advice_disposition(event, replay),
                )
        claimed, stored = self.sessions.claim(payload)
        if not claimed:
            if stored.get("result") is None:
                return self._mark_compatible(event, {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "status": "processing",
                    "duplicate": True,
                    "state_id": stored.get("state_id"),
                    "decision_id": stored.get("decision_id"),
                    "advice": None,
                    "advice_disposition": self._advice_disposition(
                        "preserve"
                    ),
                    "message": "同一事件正在由另一个本地消费者处理。",
                })
            result = dict(stored["result"])
            result["duplicate"] = True
            return self._mark_compatible(
                event,
                self._ensure_advice_disposition(event, result),
            )

        try:
            if event.event_type == EventType.CARD_REWARD:
                world = self._build_world(event)
                request = self._build_card_reward_request(event, world)
                result = self._process_decision(event, request)
            elif event.event_type == EventType.ROUTE_CHOICE:
                world = self._build_world(event)
                request = self._build_route_choice_request(event, world)
                result = self._process_decision(event, request)
            elif event.event_type == EventType.MERCHANT:
                result = self._process_decision(
                    event,
                    self._build_generic_decision_request(
                        event,
                        self._build_world(event),
                        MERCHANT_CHOICE,
                    ),
                )
            elif event.event_type == EventType.REST_SITE:
                result = self._process_decision(
                    event,
                    self._build_generic_decision_request(
                        event,
                        self._build_world(event),
                        CAMPFIRE_ACTION,
                    ),
                )
            elif event.event_type == EventType.NEOW_CHOICE:
                result = self._process_decision(
                    event,
                    self._build_generic_decision_request(
                        event,
                        self._build_world(event),
                        NEOW_BLESSING,
                    ),
                )
            elif event.event_type == EventType.EVENT_CHOICE:
                result = self._process_decision(
                    event,
                    self._build_generic_decision_request(
                        event,
                        self._build_world(event),
                        EVENT_OPTION,
                    ),
                )
            elif event.event_type == EventType.DECK_EDIT:
                result = self._process_decision(
                    event,
                    self._build_generic_decision_request(
                        event,
                        self._build_world(event),
                        DECK_EDIT,
                    ),
                )
            elif event.event_type == EventType.MAP_CHOICE:
                result = self._process_map_choice(
                    event,
                    self._build_world(event),
                )
            elif event.event_type == EventType.DECISION_CLOSED:
                result = self._process_decision_closed(event)
            elif event.event_type == EventType.RUN_ENDED:
                result = self._process_run_ended(event)
            else:
                result = self._observed_result(
                    event,
                    "accepted_no_advisor",
                    f"{event.event_type.value} 已进入协议，推荐器尚未接入。",
                )
            result.update(
                {
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "emitted_at": event.emitted_at.isoformat(),
                    "processed_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "guide_preferences": {
                        "route_mode": (
                            event.guide_preferences.route_mode.value
                            if event.guide_preferences is not None
                            else "balanced"
                        ),
                    },
                }
            )
            result = self._mark_compatible(event, result)
            disposition = result.get("advice_disposition") or {}
            if (
                superseded_run_id is not None
                and disposition.get("action") == "preserve"
            ):
                result["advice_disposition"] = self._advice_disposition(
                    "clear_run",
                    run_id=superseded_run_id,
                )
            if self.checkpoint is not None:
                if event.event_type == EventType.RUN_ENDED:
                    self.checkpoint.clear(event.run_id)
                else:
                    self.checkpoint.update(payload, result)
            self.sessions.complete(
                event.event_id,
                status=result["status"],
                decision_id=result.get("decision_id"),
                result=result,
            )
            if superseded_run_id is not None:
                self.lifecycle.end_run(superseded_run_id)
                self._closed_child_expectations.pop(
                    superseded_run_id,
                    None,
                )
                self._active_child_bindings.pop(
                    superseded_run_id,
                    None,
                )
                self.sessions.clear_run(superseded_run_id)
            if event.event_type == EventType.RUN_ENDED:
                self._closed_child_expectations.pop(event.run_id, None)
                self._active_child_bindings.pop(event.run_id, None)
                self.sessions.clear_run(event.run_id)
            return result
        except Exception as exc:
            failure = self._observed_result(
                event,
                "failed",
                str(exc),
            )
            self.sessions.complete(
                event.event_id,
                status="failed",
                result=failure,
                error=str(exc),
            )
            raise

    def _validate(self, event: GameStateEvent) -> None:
        if self.compatibility_manifest is not None:
            if not self.compatibility_manifest.is_enabled:
                raise RealtimeCompatibilityError(("manifest_not_enabled",))
            event_capability = event.event_type.value
            if (
                event_capability
                in self.compatibility_manifest.capabilities
                and not self.compatibility_manifest.capability_is_enabled(
                    event_capability
                )
            ):
                raise RealtimeCompatibilityError(
                    (f"capability_not_enabled:{event_capability}",)
                )
            local_compatibility = self.local_compatibility
            compatibility = assess_event_handshake(
                self.compatibility_manifest,
                EventHandshake(
                    game_version=event.game_version,
                    schema_version=event.schema_version,
                    source=event.source,
                    producer_id=event.producer_id,
                    producer_version=event.producer_version,
                    game_assembly_sha256=event.game_assembly_sha256,
                    release_fingerprint=event.release_fingerprint,
                ),
            )
            issues = (
                tuple(
                    local_compatibility.reason_codes
                    if local_compatibility is not None
                    else ("local_components_missing",)
                )
                + compatibility.reason_codes
            )
            if issues:
                raise RealtimeCompatibilityError(
                    tuple(dict.fromkeys(issues))
                )
        if event.emitted_at.tzinfo is None:
            raise RealtimeEventValidationError(
                "emitted_at must include a timezone"
            )
        if (
            event.event_type == EventType.CARD_REWARD
            and event.schema_version < 9
            and not event.options
        ):
            raise RealtimeEventValidationError(
                "card_reward requires at least one option"
            )
        if (
            event.event_type == EventType.CARD_REWARD
            and event.schema_version < 9
            and any(option.card is None for option in event.options)
        ):
            raise RealtimeEventValidationError(
                "card_reward options require card IDs"
            )
        if event.event_type == EventType.ROUTE_CHOICE:
            if event.schema_version < 6:
                raise RealtimeEventValidationError(
                    "route_choice requires schema v6"
                )
            if event.decision_id is None:
                raise RealtimeEventValidationError(
                    "route_choice requires a stable decision_id"
                )
            if event.options:
                raise RealtimeEventValidationError(
                    "route_choice options must be empty"
                )
            if event.map_context is None:
                raise RealtimeEventValidationError(
                    "route_choice requires map_context"
                )
            if (
                not event.map_context.nodes
                or event.map_context.node_count < 1
                or not event.map_context.origin_node_id
                or not event.map_context.available_next_node_ids
                or not event.map_context.boss_node_ids
            ):
                raise RealtimeEventValidationError(
                    "route_choice requires a complete non-empty map context"
                )
            if len(set(event.map_context.available_next_node_ids)) != len(
                event.map_context.available_next_node_ids
            ):
                raise RealtimeEventValidationError(
                    "route_choice candidates must be unique"
                )
            if (
                event.decision is None
                or event.decision.can_skip
                or event.decision.can_reroll
                or event.decision.reward_source != "MAP"
            ):
                raise RealtimeEventValidationError(
                    "route_choice requires MAP non-skip decision context"
                )
        if (
            event.schema_version >= 2
            and event.event_type == EventType.CARD_REWARD
            and event.decision is None
        ):
            raise RealtimeEventValidationError(
                "schema v2 card_reward requires decision context"
            )
        if (
            event.schema_version >= 2
            and event.event_type == EventType.DECISION_CLOSED
            and (event.parent_event_id is None or event.outcome is None)
        ):
            raise RealtimeEventValidationError(
                "schema v2 decision_closed requires parent_event_id and outcome"
            )
        if (
            event.schema_version >= 3
            and event.event_type == EventType.RUN_ENDED
            and event.run_result is None
        ):
            raise RealtimeEventValidationError(
                "schema v3 run_ended requires run_result"
            )
        if (
            event.schema_version >= 5
            and event.event_type == EventType.CARD_REWARD
            and event.decision_id is None
        ):
            raise RealtimeEventValidationError(
                "schema v5 card_reward requires a stable decision_id"
            )
        if (
            event.schema_version >= 5
            and event.schema_version < 9
            and event.event_type == EventType.CARD_REWARD
            and any(option.candidate_id is None for option in event.options)
        ):
            raise RealtimeEventValidationError(
                "schema v5 card_reward requires stable candidate IDs"
            )
        if (
            event.schema_version >= 5
            and event.event_type == EventType.DECISION_CLOSED
            and event.decision_id is None
        ):
            raise RealtimeEventValidationError(
                "schema v5 decision_closed requires decision_id"
            )
        if (
            event.schema_version >= 5
            and event.event_type == EventType.DECISION_CLOSED
            and event.outcome is not None
            and event.outcome.kind == "selected"
            and event.outcome.selected_candidate_id is None
        ):
            raise RealtimeEventValidationError(
                "schema v5 selected outcome requires selected_candidate_id"
            )
        if self.repository.find_entity(
            "characters", event.state.character
        ) is None:
            raise RealtimeEventValidationError(
                "Unknown character ID or name"
            )

    def _build_world(self, event: GameStateEvent) -> WorldState:
        state = event.state.model_dump()
        map_context = (
            event.map_context.model_dump()
            if event.map_context is not None
            else None
        )
        # A decision must be evaluated from one observation revision.  Never
        # combine a fresh state snapshot with a map captured by an older
        # event: after the player chooses a route that would silently score a
        # later card reward against the previous origin and candidates.
        # Producers should attach a current map to decision events when it is
        # available.  Missing map context is an honest degradable gap.
        return WorldState.create(
            run_id=event.run_id,
            sequence=event.state_revision or event.sequence,
            state=state,
            map_context=map_context,
            route_mode=(
                event.guide_preferences.route_mode.value
                if event.guide_preferences is not None
                else "balanced"
            ),
            game_version=event.game_version,
            scene=event.event_type.value,
        )

    def _mark_compatible(
        self,
        event: GameStateEvent,
        result: Dict,
    ) -> Dict:
        if self.compatibility_manifest is None:
            return result
        marked = dict(result)
        marked["compatibility"] = {
            "status": "compatible",
            "manifest_version": self.compatibility_manifest.manifest_version,
            "release_fingerprint": (
                self.compatibility_manifest.release_fingerprint
            ),
            "guide_version": self.compatibility_manifest.guide.version,
            "state_event_schema_version": event.schema_version,
            "producer_id": event.producer_id,
            "producer_version": event.producer_version,
            "game_version": event.game_version,
            "game_assembly_sha256": event.game_assembly_sha256,
            "capability": event.event_type.value,
            "capability_status": (
                self.compatibility_manifest.capabilities.get(
                    event.event_type.value,
                    "not_applicable",
                )
            ),
        }
        return marked

    @staticmethod
    def _build_card_reward_request(
        event: GameStateEvent,
        world: WorldState,
    ) -> DecisionRequest:
        can_skip = (
            event.decision.can_skip
            if event.decision is not None
            else True
        )
        if event.schema_version >= 9:
            candidates = [
                RealtimeEventProcessor._decision_candidate(
                    candidate,
                    index,
                )
                for index, candidate in enumerate(event.candidates)
            ]
        else:
            candidates = [
                DecisionCandidate.create(
                    option.candidate_id or f"{index}:{option.card}",
                    option.model_dump(),
                    label=option.card or "未知卡牌",
                    display_index=index,
                )
                for index, option in enumerate(event.options)
            ]
        if can_skip and event.schema_version < 9:
            candidates.append(DecisionCandidate.create(
                "skip",
                {"choice": "skip"},
                label="跳过",
                display_index=len(candidates),
            ))
        return DecisionRequest.create(
            decision_id=event.decision_id or event.event_id,
            decision_type=CARD_REWARD,
            world=world,
            candidates=candidates,
            constraints={
                "can_skip": can_skip,
                "can_reroll": (
                    event.decision.can_reroll
                    if event.decision is not None
                    else False
                ),
                "reward_source": (
                    event.decision.reward_source
                    if event.decision is not None
                    else None
                ),
                "decision_parent": (
                    event.decision_parent.model_dump(mode="json")
                    if event.decision_parent is not None
                    else None
                ),
                "route_mode": world.route_mode,
            },
        )

    @staticmethod
    def _build_route_choice_request(
        event: GameStateEvent,
        world: WorldState,
    ) -> DecisionRequest:
        assert event.map_context is not None
        by_id = {
            node.node_id: node
            for node in event.map_context.nodes
        }
        candidates = []
        raw_candidates = (
            [
                (
                    candidate.payload.node_id,
                    candidate,
                )
                for candidate in event.candidates
            ]
            if event.schema_version >= 9
            else [
                (node_id, None)
                for node_id in event.map_context.available_next_node_ids
            ]
        )
        for index, (node_id, envelope) in enumerate(raw_candidates):
            assert node_id is not None
            node = by_id.get(node_id)
            if node is None:
                # Keep the opaque Mod candidate in the request so RoutePolicy
                # can fail closed with an explicit unknown-candidate gap.
                candidates.append(DecisionCandidate.create(
                    node_id,
                    (
                        RealtimeEventProcessor._candidate_payload(envelope)
                        if envelope is not None
                        else {"node_id": node_id}
                    ),
                    label=node_id,
                    display_index=index,
                ))
                continue
            payload = node.model_dump(mode="json")
            if envelope is not None:
                payload.update(
                    RealtimeEventProcessor._candidate_payload(envelope)
                )
            candidates.append(DecisionCandidate.create(
                envelope.candidate_id if envelope is not None else node.node_id,
                payload,
                label=(
                    envelope.label
                    if envelope is not None
                    else node.label or node.node_id
                ),
                display_index=index,
                eligible=(envelope.eligible if envelope is not None else True),
            ))
        return DecisionRequest.create(
            decision_id=event.decision_id or event.event_id,
            decision_type=ROUTE_CHOICE,
            world=world,
            candidates=candidates,
            constraints={
                "origin_node_id": event.map_context.origin_node_id,
                "map_name": event.map_context.map_name,
                "route_mode": world.route_mode,
            },
        )

    @staticmethod
    def _candidate_payload(candidate) -> Dict:
        payload = candidate.payload.model_dump(
            mode="json",
            exclude_none=True,
        )
        payload.update({
            "candidate_kind": candidate.kind.value,
            "entity_id": candidate.entity_id,
            "costs": [
                cost.model_dump(mode="json")
                for cost in candidate.costs
            ],
            "unavailable_reason": candidate.unavailable_reason,
        })
        return payload

    @staticmethod
    def _decision_candidate(candidate, index: int) -> DecisionCandidate:
        return DecisionCandidate.create(
            candidate.candidate_id,
            RealtimeEventProcessor._candidate_payload(candidate),
            label=candidate.label,
            display_index=index,
            eligible=candidate.eligible,
        )

    @staticmethod
    def _build_generic_decision_request(
        event: GameStateEvent,
        world: WorldState,
        decision_type: str,
    ) -> DecisionRequest:
        if event.schema_version < 9:
            raise RealtimeEventValidationError(
                f"{event.event_type.value} requires schema v9 candidates"
            )
        constraints = {
            "can_skip": bool(
                event.decision and event.decision.can_skip
            ),
            "can_reroll": bool(
                event.decision and event.decision.can_reroll
            ),
            "reward_source": (
                event.decision.reward_source
                if event.decision is not None
                else None
            ),
            "decision_parent": (
                event.decision_parent.model_dump(mode="json")
                if event.decision_parent is not None
                else None
            ),
            "route_mode": world.route_mode,
        }
        return DecisionRequest.create(
            decision_id=event.decision_id or event.event_id,
            decision_type=decision_type,
            world=world,
            candidates=[
                RealtimeEventProcessor._decision_candidate(
                    candidate,
                    index,
                )
                for index, candidate in enumerate(event.candidates)
            ],
            constraints=constraints,
        )

    def _process_decision(
        self,
        event: GameStateEvent,
        request: DecisionRequest,
    ) -> Dict:
        verified_parent = self._validate_decision_parent(request)
        closed = self.lifecycle.closed(request.world.run_id)
        if closed is not None and closed.decision_id == request.decision_id:
            raise RealtimeEventValidationError(
                "closed decision cannot be reopened"
            )
        recommendation = self.policies.recommend(request)
        transition = self.lifecycle.open(
            request,
            observation_event_id=event.event_id,
        )
        active_child_binding = None
        if verified_parent is not None:
            active_child_binding = {
                **{
                    key: verified_parent[key]
                    for key in (
                        "parent_decision_id",
                        "parent_candidate_id",
                        "source_type",
                        "source_id",
                        "parent_close_sequence",
                        "child_decision_type",
                        "operation",
                    )
                },
                "child_decision_id": request.decision_id,
            }
            self._active_child_bindings[
                request.world.run_id
            ] = active_child_binding
        if transition.phase.value == "opened":
            self._closed_child_expectations.pop(
                request.world.run_id,
                None,
            )
            if active_child_binding is None:
                self._active_child_bindings.pop(
                    request.world.run_id,
                    None,
                )
        result = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": "processed",
            "duplicate": False,
            "state_id": None,
            "decision_id": request.decision_id,
            "decision_phase": transition.phase.value,
            "recommendation": recommendation.as_dict(),
            "guide_preferences": {
                "route_mode": request.world.route_mode,
            },
            "advice_disposition": self._advice_disposition(
                "publish",
                run_id=event.run_id,
                decision_id=request.decision_id,
            ),
            "message": "已根据当前只读状态生成决策建议。",
        }
        if active_child_binding is not None:
            result["active_child_binding"] = active_child_binding
        if request.decision_type == CARD_REWARD:
            result["advice"] = recommendation.payload_dict()
        return result

    def _process_decision_closed(self, event: GameStateEvent) -> Dict:
        if event.parent_event_id is None or event.outcome is None:
            return self._observed_result(
                event,
                "observed",
                "旧版决策关闭事件已接收，但没有实际选择信息。",
            )

        session_parent = self.sessions.load(event.parent_event_id)
        parent = session_parent
        recovered = False
        decision_id = event.decision_id
        if parent is None and self.checkpoint is not None:
            checkpoint = self.checkpoint.load()
            current = (
                checkpoint.get("current_decision")
                if checkpoint is not None
                and checkpoint.get("run_id") == event.run_id
                and self._checkpoint_record_matches_current_release(
                    checkpoint,
                    checkpoint.get("current_decision"),
                )
                else None
            )
            if (
                current is not None
                and current.get("event_id") == event.parent_event_id
                and (
                    (current.get("payload") or {}).get("decision_parent")
                    is None
                    or self._restore_active_child_binding(current)
                    is not None
                )
            ):
                parent = current
                recovered = True
        if parent is None or parent.get("decision_id") is None:
            if self.checkpoint is not None and decision_id is not None:
                closed = self.checkpoint.find_closed_decision(
                    run_id=event.run_id,
                    decision_id=decision_id,
                )
                if closed is not None and (
                    checkpoint is None
                    or not self._checkpoint_record_matches_current_release(
                        checkpoint,
                        closed,
                    )
                ):
                    closed = None
                if closed is not None:
                    closed_schema_version = int(
                        (closed.get("payload") or {}).get("schema_version")
                        or event.schema_version
                    )
                    previous = self._outcome_signature(
                        closed.get("outcome") or {},
                        schema_version=closed_schema_version,
                    )
                    incoming = self._outcome_signature(
                        event.outcome.model_dump(mode="json"),
                        schema_version=event.schema_version,
                    )
                    if previous != incoming:
                        raise RealtimeEventValidationError(
                            "decision close conflicts with its final outcome"
                        )
                    result = self._observed_result(
                        event,
                        "outcome_already_recorded",
                        "重复关闭与已确认结果一致，已按幂等事件处理。",
                    )
                    result["decision_id"] = decision_id
                    previous_result = closed.get("result") or {}
                    result["decision_type"] = (
                        previous_result.get("decision_type") or CARD_REWARD
                    )
                    result["chosen_option"] = previous_result.get(
                        "chosen_option"
                    )
                    result["decision_phase"] = "closed"
                    result["advice_disposition"] = (
                        self._advice_disposition(
                            "clear",
                            run_id=event.run_id,
                            decision_id=decision_id,
                        )
                    )
                    return result
            return self._observed_result(
                event,
                "outcome_unmatched",
                "未找到对应的决策建议，结果已保留在原始事件中。",
            )
        if parent.get("run_id") != event.run_id:
            raise RealtimeEventValidationError(
                "decision close parent belongs to another run"
            )
        parent_decision_id = str(parent["decision_id"])
        if decision_id is not None and decision_id != parent_decision_id:
            raise RealtimeEventValidationError(
                "decision close targets another decision"
            )
        decision_id = parent_decision_id
        if int(parent.get("sequence", 0)) >= event.sequence:
            raise RealtimeEventValidationError(
                "decision close sequence must follow its parent observation"
            )
        parent_payload = parent.get("payload") or {}
        parent_result = parent.get("result") or {}
        recommendation = parent_result.get("recommendation") or {}
        canonical_candidates = recommendation.get("candidates") or []
        canonical_close = event.schema_version >= 5
        decision_type = str(
            recommendation.get("decision_type")
            or parent_payload.get("event_type")
            or CARD_REWARD
        )
        outcome = event.outcome
        if outcome.kind == "closed_unknown":
            chosen_option = "closed_unknown"
        elif outcome.kind == "skipped":
            chosen_option = "skip"
            legacy_decision = parent_payload.get("decision") or {}
            if canonical_close or canonical_candidates:
                self._require_eligible_candidate(
                    canonical_candidates,
                    chosen_option,
                    required=canonical_close,
                )
            elif not bool(legacy_decision.get("can_skip", True)):
                raise RealtimeEventValidationError(
                    "decision does not allow skip"
                )
        else:
            if canonical_close:
                chosen_option = outcome.selected_candidate_id
                if not chosen_option:
                    raise RealtimeEventValidationError(
                        "canonical selected outcome requires candidate ID"
                    )
            else:
                chosen_option = self._resolve_legacy_card_choice(
                    outcome=outcome,
                    parent_options=parent_payload.get("options") or [],
                )
            self._require_eligible_candidate(
                canonical_candidates,
                chosen_option,
                required=canonical_close,
            )

        previous_outcome = (
            parent.get("result", {}).get("outcome") or {}
        ).get("chosen_option")
        if previous_outcome is not None and previous_outcome != chosen_option:
            raise RealtimeEventValidationError(
                "decision outcome conflicts with its final result"
            )
        transition = self.lifecycle.close(
            run_id=event.run_id,
            decision_id=decision_id,
            decision_type=decision_type,
            sequence=event.sequence,
            outcome=chosen_option,
            allow_recovered=recovered,
            parent_event_id=event.parent_event_id,
        )
        child_expectation = None
        self._closed_child_expectations.pop(event.run_id, None)
        self._active_child_bindings.pop(event.run_id, None)
        if outcome.kind == "selected":
            child_expectation = self._derive_child_expectation(
                parent_payload=parent_payload,
                parent_decision_id=decision_id,
                selected_candidate_id=chosen_option,
                parent_decision_type=decision_type,
                parent_close_sequence=event.sequence,
            )
            if child_expectation is not None:
                self._closed_child_expectations[event.run_id] = (
                    child_expectation
                )
        if outcome.kind == "closed_unknown":
            result = self._observed_result(
                event,
                "observed_unresolved",
                "决策界面已关闭，但 Mod 未确认玩家选择；不会把它误标为跳过。",
            )
            result["decision_id"] = decision_id
            result["decision_type"] = decision_type
            result["decision_phase"] = transition.phase.value
            result["chosen_option"] = chosen_option
            result["child_expectation"] = None
            result["advice_disposition"] = self._advice_disposition(
                "clear",
                run_id=event.run_id,
                decision_id=decision_id,
            )
            return result

        if session_parent is not None:
            self.sessions.record_outcome(
                event.parent_event_id, chosen_option
            )
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": "outcome_recorded",
            "duplicate": False,
            "state_id": parent.get("state_id"),
            "decision_id": decision_id,
            "decision_type": decision_type,
            "decision_phase": transition.phase.value,
            "chosen_option": chosen_option,
            "child_expectation": child_expectation,
            "advice": None,
            "advice_disposition": self._advice_disposition(
                "clear",
                run_id=event.run_id,
                decision_id=decision_id,
            ),
            "message": (
                "已确认当前局决策结果：跳过。"
                if chosen_option == "skip"
                else f"已确认当前局决策结果：{chosen_option}。"
            ),
        }

    @staticmethod
    def _stable_parent_source_id(candidate_id: str) -> str:
        digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        return "SOURCE_" + digest[:24].upper()

    @staticmethod
    def _read_child_expectation(value: object) -> dict | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {
            "parent_decision_id",
            "parent_candidate_id",
            "source_type",
            "source_id",
            "parent_close_sequence",
            "child_decision_type",
            "operation",
        }:
            return None
        if any(
            not isinstance(value.get(field), str)
            or not str(value[field]).strip()
            for field in (
                "parent_decision_id",
                "parent_candidate_id",
                "source_type",
                "source_id",
                "child_decision_type",
            )
        ):
            return None
        sequence = value.get("parent_close_sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
        ):
            return None
        child_type = value["child_decision_type"]
        operation = value.get("operation")
        if value["source_type"] not in {
            "neow_choice",
            "event_choice",
            "rest_site",
            "merchant",
        }:
            return None
        if value["source_id"] != (
            RealtimeEventProcessor._stable_parent_source_id(
                value["parent_candidate_id"]
            )
        ):
            return None
        if child_type == CARD_REWARD:
            if (
                operation is not None
                or value["source_type"]
                not in {"neow_choice", "event_choice"}
            ):
                return None
        elif child_type == DECK_EDIT:
            if operation not in {"upgrade", "remove", "transform"}:
                return None
        else:
            return None
        return dict(value)

    @classmethod
    def _read_active_child_binding(cls, value: object) -> dict | None:
        if not isinstance(value, dict) or set(value) != {
            "parent_decision_id",
            "parent_candidate_id",
            "source_type",
            "source_id",
            "parent_close_sequence",
            "child_decision_type",
            "operation",
            "child_decision_id",
        }:
            return None
        expectation = cls._read_child_expectation({
            key: value[key]
            for key in (
                "parent_decision_id",
                "parent_candidate_id",
                "source_type",
                "source_id",
                "parent_close_sequence",
                "child_decision_type",
                "operation",
            )
        })
        child_decision_id = value.get("child_decision_id")
        if (
            expectation is None
            or not isinstance(child_decision_id, str)
            or not child_decision_id.strip()
        ):
            return None
        return {**expectation, "child_decision_id": child_decision_id}

    @classmethod
    def _restore_closed_child_expectation(
        cls,
        closed_record: object,
    ) -> dict | None:
        """Re-derive a child expectation from retained parent evidence."""

        if not checkpoint_record_payload_matches(closed_record):
            return None
        assert isinstance(closed_record, dict)
        parent_record = closed_record.get("parent_observation")
        if not checkpoint_record_payload_matches(parent_record):
            return None
        assert isinstance(parent_record, dict)
        close_payload = closed_record["payload"]
        parent_payload = parent_record["payload"]
        close_result = closed_record.get("result") or {}
        parent_result = parent_record.get("result") or {}
        recommendation = parent_result.get("recommendation") or {}
        outcome = close_payload.get("outcome") or {}
        stored = cls._read_child_expectation(
            close_result.get("child_expectation")
        )
        if stored is None:
            return None
        try:
            parent_sequence = int(parent_record.get("sequence") or 0)
            close_sequence = int(closed_record.get("sequence") or 0)
        except (TypeError, ValueError):
            return None
        if (
            close_payload.get("event_type") != "decision_closed"
            or outcome.get("kind") != "selected"
            or close_payload.get("run_id") != parent_payload.get("run_id")
            or closed_record.get("run_id") != parent_record.get("run_id")
            or close_payload.get("parent_event_id")
            != parent_record.get("event_id")
            or close_payload.get("decision_id")
            != parent_record.get("decision_id")
            or closed_record.get("decision_id")
            != parent_record.get("decision_id")
            or close_payload.get("decision_id")
            != closed_record.get("decision_id")
            or parent_payload.get("decision_id")
            != parent_record.get("decision_id")
            or parent_result.get("decision_id")
            != parent_record.get("decision_id")
            or recommendation.get("decision_id")
            != parent_record.get("decision_id")
            or close_result.get("decision_type")
            != recommendation.get("decision_type")
            or parent_record.get("decision_id")
            != stored["parent_decision_id"]
            or outcome.get("selected_candidate_id")
            != close_result.get("chosen_option")
            or outcome.get("selected_candidate_id")
            != stored["parent_candidate_id"]
            or close_sequence != stored["parent_close_sequence"]
            or parent_sequence >= close_sequence
            or parent_payload.get("release_fingerprint")
            != close_payload.get("release_fingerprint")
        ):
            return None
        derived = cls._derive_child_expectation(
            parent_payload=parent_payload,
            parent_decision_id=stored["parent_decision_id"],
            selected_candidate_id=str(
                outcome["selected_candidate_id"]
            ),
            parent_decision_type=str(
                recommendation.get("decision_type") or ""
            ),
            parent_close_sequence=close_sequence,
        )
        return derived if derived == stored else None

    @classmethod
    def _restore_active_child_binding(
        cls,
        current_record: object,
    ) -> dict | None:
        if not checkpoint_record_payload_matches(current_record):
            return None
        assert isinstance(current_record, dict)
        current_payload = current_record["payload"]
        current_result = current_record.get("result") or {}
        recommendation = current_result.get("recommendation") or {}
        stored = cls._read_active_child_binding(
            current_result.get("active_child_binding")
        )
        parent_close = current_record.get("parent_close")
        expected = cls._restore_closed_child_expectation(parent_close)
        if stored is None or expected is None:
            return None
        expected_binding = {
            **expected,
            "child_decision_id": current_record.get("decision_id"),
        }
        parent = current_payload.get("decision_parent") or {}
        expected_event_type = {
            CARD_REWARD: "card_reward",
            DECK_EDIT: "deck_edit",
        }.get(expected["child_decision_type"])
        try:
            current_sequence = int(current_record.get("sequence") or 0)
        except (TypeError, ValueError):
            return None
        if (
            stored != expected_binding
            or current_payload.get("event_type") != expected_event_type
            or current_payload.get("decision_id")
            != stored["child_decision_id"]
            or current_result.get("decision_id")
            != stored["child_decision_id"]
            or recommendation.get("decision_id")
            != stored["child_decision_id"]
            or recommendation.get("decision_type")
            != expected["child_decision_type"]
            or current_payload.get("run_id")
            != (parent_close or {}).get("run_id")
            or current_record.get("event_id")
            != current_payload.get("event_id")
            or current_record.get("run_id")
            != current_payload.get("run_id")
            or current_sequence != current_payload.get("sequence")
            or current_payload.get("release_fingerprint")
            != ((parent_close or {}).get("payload") or {}).get(
                "release_fingerprint"
            )
            or current_sequence <= expected["parent_close_sequence"]
        ):
            return None
        if parent != {
            "decision_id": expected["parent_decision_id"],
            "candidate_id": expected["parent_candidate_id"],
            "source_type": expected["source_type"],
            "source_id": expected["source_id"],
        }:
            return None
        decision = current_payload.get("decision") or {}
        if expected["child_decision_type"] == CARD_REWARD and (
            decision.get("reward_source")
            != cls._expected_child_reward_source(expected)
        ):
            return None
        if expected["child_decision_type"] == DECK_EDIT:
            operations = {
                str((candidate.get("payload") or {}).get("operation") or "")
                for candidate in current_payload.get("candidates") or ()
            }
            if operations != {expected["operation"]}:
                return None
        return stored

    @staticmethod
    def _expected_child_reward_source(expectation: dict) -> str | None:
        if expectation.get("child_decision_type") != CARD_REWARD:
            return None
        return {
            "neow_choice": "NEOW",
            "event_choice": "EVENT",
        }.get(expectation.get("source_type"))

    @classmethod
    def _derive_child_expectation(
        cls,
        *,
        parent_payload: Dict,
        parent_decision_id: str,
        selected_candidate_id: str,
        parent_decision_type: str,
        parent_close_sequence: int,
    ) -> dict | None:
        source_type = str(parent_payload.get("event_type") or "")
        expected_parent_type = {
            "neow_choice": NEOW_BLESSING,
            "event_choice": EVENT_OPTION,
            "rest_site": CAMPFIRE_ACTION,
            "merchant": MERCHANT_CHOICE,
        }.get(source_type)
        if expected_parent_type != parent_decision_type:
            return None
        expected_reward_source = {
            "neow_choice": "NEOW",
            "event_choice": "EVENT",
            "rest_site": "REST_SITE",
            "merchant": "MERCHANT",
        }.get(source_type)
        if (
            not isinstance(parent_payload.get("decision"), dict)
            or parent_payload["decision"].get("reward_source")
            != expected_reward_source
        ):
            return None
        selected = next(
            (
                candidate
                for candidate in parent_payload.get("candidates") or ()
                if candidate.get("candidate_id") == selected_candidate_id
            ),
            None,
        )
        if not isinstance(selected, dict) or selected.get("eligible") is not True:
            return None
        payload = selected.get("payload") or {}
        expectations: set[tuple[str, str | None]] = set()
        for effect in payload.get("effects") or ():
            if not isinstance(effect, dict) or effect.get("certainty") != "exact":
                continue
            kind = effect.get("kind")
            target_mode = effect.get("target_mode")
            if (
                kind == "followup_choice"
                and effect.get("child_decision_type") == CARD_REWARD
            ):
                expectations.add((CARD_REWARD, None))
            elif target_mode == "choose" and kind in {
                "upgrade_card",
                "remove_card",
                "transform_card",
            }:
                expectations.add((DECK_EDIT, kind.removesuffix("_card")))
        if len(expectations) != 1:
            return None
        child_type, operation = next(iter(expectations))
        if child_type == CARD_REWARD and source_type not in {
            "neow_choice",
            "event_choice",
        }:
            return None
        return {
            "parent_decision_id": parent_decision_id,
            "parent_candidate_id": selected_candidate_id,
            "source_type": source_type,
            "source_id": cls._stable_parent_source_id(
                selected_candidate_id
            ),
            "parent_close_sequence": parent_close_sequence,
            "child_decision_type": child_type,
            "operation": operation,
        }

    def _validate_decision_parent(
        self,
        request: DecisionRequest,
    ) -> dict | None:
        parent = request.constraints.get("decision_parent")
        if parent is None:
            active = self._active_child_bindings.get(request.world.run_id)
            if (
                active is not None
                and request.decision_id == active.get("child_decision_id")
            ):
                raise RealtimeEventValidationError(
                    "active child decision omitted its verified parent"
                )
            expected = self._closed_child_expectations.get(
                request.world.run_id
            )
            if (
                expected is not None
                and request.decision_type
                == expected.get("child_decision_type")
            ):
                raise RealtimeEventValidationError(
                    "expected child decision omitted its verified parent"
                )
            return None
        if not isinstance(parent, dict):
            # DecisionRequest freezes nested dictionaries as MappingProxyType.
            try:
                parent = dict(parent)
            except (TypeError, ValueError):
                raise RealtimeEventValidationError(
                    "decision parent is not a structured object"
                ) from None
        active_binding = self._active_child_bindings.get(
            request.world.run_id
        )
        closed = self.lifecycle.closed(request.world.run_id)
        if active_binding is not None:
            expected = self._read_active_child_binding(active_binding)
            active = self.lifecycle.active(request.world.run_id)
            if (
                expected is None
                or active is None
                or active.decision_id != expected["child_decision_id"]
                or request.decision_id != expected["child_decision_id"]
            ):
                raise RealtimeEventValidationError(
                    "nested decision active-child binding is invalid"
                )
        else:
            expectation = self._closed_child_expectations.get(
                request.world.run_id
            )
            if closed is None or expectation is None:
                raise RealtimeEventValidationError(
                    "nested decision has no verified recently closed parent"
                )
            expected = self._read_child_expectation(expectation)
        if expected is None:
            raise RealtimeEventValidationError(
                "nested decision parent expectation is invalid"
            )
        identity_matches = (
            parent.get("decision_id") == expected["parent_decision_id"]
            and parent.get("candidate_id")
            == expected["parent_candidate_id"]
            and parent.get("source_type") == expected["source_type"]
            and parent.get("source_id") == expected["source_id"]
            and (
                active_binding is not None
                or (
                    closed is not None
                    and closed.decision_id
                    == expected["parent_decision_id"]
                    and closed.outcome
                    == expected["parent_candidate_id"]
                    and closed.sequence
                    == expected["parent_close_sequence"]
                )
            )
            and request.world.sequence
            > expected["parent_close_sequence"]
            and request.decision_type == expected["child_decision_type"]
        )
        if not identity_matches:
            raise RealtimeEventValidationError(
                "nested decision does not match its latest closed parent"
            )
        if request.decision_type == CARD_REWARD and (
            request.constraints.get("reward_source")
            != self._expected_child_reward_source(expected)
        ):
            raise RealtimeEventValidationError(
                "card reward source does not match parent expectation"
            )
        if request.decision_type == DECK_EDIT:
            operations = {
                str(candidate.payload.get("operation") or "")
                for candidate in request.candidates
            }
            if operations != {expected["operation"]}:
                raise RealtimeEventValidationError(
                    "deck edit operation does not match parent expectation"
                )
        return expected

    @staticmethod
    def _resolve_legacy_card_choice(
        *,
        outcome,
        parent_options: list[Dict],
    ) -> str:
        """Resolve v1-v4 Card Reward close fields into a candidate ID."""
        chosen_option = outcome.selected_candidate_id
        selected_card = outcome.selected_card
        if outcome.selected_option_index is not None:
            option_index = outcome.selected_option_index
            if option_index >= len(parent_options):
                raise RealtimeEventValidationError(
                    "selected_option_index is outside the parent options"
                )
            option = parent_options[option_index]
            indexed_card = option["card"]
            if (
                selected_card is not None
                and selected_card.strip().lower()
                != str(indexed_card).strip().lower()
            ):
                raise RealtimeEventValidationError(
                    "selected_card does not match selected_option_index"
                )
            indexed_candidate = option.get("candidate_id") or (
                f"{option_index}:{indexed_card}"
            )
            if chosen_option is not None and chosen_option != indexed_candidate:
                raise RealtimeEventValidationError(
                    "selected_candidate_id does not match option index"
                )
            chosen_option = indexed_candidate
        elif chosen_option is None and selected_card:
            matching_options = [
                (index, option)
                for index, option in enumerate(parent_options)
                if str(option.get("card") or "").strip().lower()
                == selected_card.strip().lower()
            ]
            if len(matching_options) > 1:
                raise RealtimeEventValidationError(
                    "legacy selected_card is ambiguous without an option index"
                )
            if not matching_options:
                raise RealtimeEventValidationError(
                    "selected_card is not part of the parent decision"
                )
            option_index, option = matching_options[0]
            indexed_card = str(option["card"])
            chosen_option = option.get("candidate_id") or (
                f"{option_index}:{indexed_card}"
            )
        if not chosen_option:
            raise RealtimeEventValidationError(
                "selected outcome requires a candidate or option index"
            )
        return chosen_option

    @staticmethod
    def _require_eligible_candidate(
        candidates: list[Dict],
        candidate_id: str,
        *,
        required: bool,
    ) -> None:
        candidate = next(
            (
                item
                for item in candidates
                if item.get("candidate_id") == candidate_id
            ),
            None,
        )
        if candidate is None:
            if required or candidates:
                raise RealtimeEventValidationError(
                    "selected candidate is not part of the parent decision"
                )
            return
        if not candidate.get("eligible", False):
            raise RealtimeEventValidationError(
                "selected candidate is ineligible"
            )

    @staticmethod
    def _outcome_signature(
        outcome: Dict,
        *,
        schema_version: int,
    ) -> tuple:
        """Compare final outcomes without accepting same-kind substitutions."""
        if schema_version >= 5:
            return (
                outcome.get("kind"),
                outcome.get("selected_candidate_id"),
            )
        return (
            outcome.get("kind"),
            outcome.get("selected_candidate_id"),
            outcome.get("selected_card"),
            outcome.get("selected_option_index"),
        )

    def _process_map_choice(
        self,
        event: GameStateEvent,
        world: WorldState,
    ) -> Dict:
        if event.map_context is None or not event.map_context.nodes:
            return self._observed_result(
                event,
                "accepted_no_advisor",
                "地图事件缺少节点数据，无法生成推荐。",
            )

        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": "map_captured",
            "duplicate": False,
            "state_id": None,
            "decision_id": None,
            "advice": None,
            "advice_disposition": self._advice_disposition("preserve"),
            "message": "地图快照已进入当前局检查点；P0 不生成路线推荐。",
        }

    def _process_run_ended(self, event: GameStateEvent) -> Dict:
        self.lifecycle.end_run(event.run_id)
        if event.run_result is None:
            result = self._observed_result(
                event,
                "session_cleared",
                "旧版结束事件没有最终摘要；仅清理当前局状态。",
            )
            result["session_cleared"] = True
            result["summary_saved"] = False
            result["advice_disposition"] = self._advice_disposition(
                "clear_run",
                run_id=event.run_id,
            )
            return result

        state = event.state.model_dump(mode="json")
        run_result = event.run_result.model_dump(mode="json")
        summary = {
            "run_id": event.run_id,
            "outcome": run_result["outcome"],
            "character": state["character"],
            "ascension": state["ascension"],
            "final_floor": state["floor"],
            "final_score": run_result.get("final_score"),
            "started_at": run_result.get("started_at"),
            "ended_at": run_result["ended_at"],
            "game_version": event.game_version,
            "final_deck": state["deck"],
            "final_relics": (
                state["relic_states"] or state["relics"]
            ),
            "final_potions": state["potions"],
        }
        self.repository.save_run_summary(summary)
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": "run_finalized",
            "duplicate": False,
            "state_id": None,
            "decision_id": None,
            "advice": None,
            "advice_disposition": self._advice_disposition(
                "clear_run",
                run_id=event.run_id,
            ),
            "summary_saved": True,
            "session_cleared": True,
            "message": "本局最终摘要已保存，中间状态已清理。",
        }

    @staticmethod
    def _advice_disposition(
        action: str,
        *,
        run_id: str | None = None,
        decision_id: str | None = None,
    ) -> Dict:
        return {
            "action": action,
            "run_id": run_id,
            "decision_id": decision_id,
        }

    @classmethod
    def _ensure_advice_disposition(
        cls,
        event: GameStateEvent,
        result: Dict,
    ) -> Dict:
        """Normalize replay of checkpoints written before disposition v1."""
        normalized = dict(result)
        if normalized.get("advice_disposition") is not None:
            return normalized
        decision_id = normalized.get("decision_id")
        if normalized.get("recommendation") is not None and decision_id:
            action = "publish"
        elif event.event_type == EventType.DECISION_CLOSED and decision_id:
            action = "clear"
        elif event.event_type == EventType.RUN_ENDED:
            action = "clear_run"
        else:
            action = "preserve"
        normalized["advice_disposition"] = cls._advice_disposition(
            action,
            run_id=(event.run_id if action != "preserve" else None),
            decision_id=(decision_id if action in {"publish", "clear"} else None),
        )
        return normalized

    @classmethod
    def _observed_result(
        cls,
        event: GameStateEvent,
        status: str,
        message: str,
    ) -> Dict:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": status,
            "duplicate": False,
            "state_id": None,
            "decision_id": None,
            "advice": None,
            "advice_disposition": cls._advice_disposition("preserve"),
            "message": message,
        }
