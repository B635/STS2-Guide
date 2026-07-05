"""Idempotent processing of versioned game-state observations."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from advisor.card_reward import recommend_card_reward
from advisor.data_sources import LocalCardTierSource
from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.protocol import EventType, GameStateEvent
from realtime.session import TransientSessionStore
from storage.relational import RelationalRepository


class RealtimeEventValidationError(ValueError):
    """The event is valid JSON but cannot safely drive a recommendation."""


class RealtimeEventProcessor:
    def __init__(
        self,
        repository: RelationalRepository,
        sessions: TransientSessionStore | None = None,
        checkpoint: Optional[ActiveRunCheckpointStore] = None,
        local_tiers: Optional[LocalCardTierSource] = None,
    ):
        self.repository = repository
        self.sessions = sessions or TransientSessionStore()
        self.checkpoint = checkpoint
        self.local_tiers = local_tiers

    def process(self, event: GameStateEvent) -> Dict:
        self._validate(event)
        payload = event.model_dump(mode="json")
        if self.checkpoint is not None:
            replay = self.checkpoint.replay_result(payload)
            if replay is not None:
                return replay
        claimed, stored = self.sessions.claim(payload)
        if not claimed:
            if stored.get("result") is None:
                return {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "status": "processing",
                    "duplicate": True,
                    "state_id": stored.get("state_id"),
                    "decision_id": stored.get("decision_id"),
                    "advice": None,
                    "message": "同一事件正在由另一个本地消费者处理。",
                }
            result = dict(stored["result"])
            result["duplicate"] = True
            return result

        try:
            if event.event_type == EventType.CARD_REWARD:
                result = self._process_card_reward(event)
            elif event.event_type == EventType.MAP_CHOICE:
                result = self._process_map_choice(event)
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
                }
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
            if event.event_type == EventType.RUN_ENDED:
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
        if event.emitted_at.tzinfo is None:
            raise RealtimeEventValidationError(
                "emitted_at must include a timezone"
            )
        if (
            event.event_type == EventType.CARD_REWARD
            and not event.options
        ):
            raise RealtimeEventValidationError(
                "card_reward requires at least one option"
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
        if self.repository.find_entity(
            "characters", event.state.character
        ) is None:
            raise RealtimeEventValidationError(
                "Unknown character ID or name"
            )

    def _process_card_reward(self, event: GameStateEvent) -> Dict:
        state = event.state.model_dump()
        state["game_version"] = event.game_version
        if self.checkpoint is not None:
            checkpoint = self.checkpoint.load()
            if (
                checkpoint is not None
                and checkpoint.get("run_id") == event.run_id
                and checkpoint.get("map_context") is not None
            ):
                state["map_context"] = checkpoint["map_context"]
        options = [option.model_dump() for option in event.options]
        advice = recommend_card_reward(
            state,
            options,
            self.repository,
            local_tiers=self.local_tiers,
            can_skip=(
                event.decision.can_skip
                if event.decision is not None
                else True
            ),
        )
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": "processed",
            "duplicate": False,
            "state_id": None,
            "decision_id": event.event_id,
            "advice": advice,
            "message": "已根据只读 Mod 状态生成选牌建议。",
        }

    def _process_decision_closed(self, event: GameStateEvent) -> Dict:
        if event.parent_event_id is None or event.outcome is None:
            return self._observed_result(
                event,
                "observed",
                "旧版选牌界面关闭事件已接收，但没有实际选择信息。",
            )

        session_parent = self.sessions.load(event.parent_event_id)
        parent = session_parent
        if parent is None and self.checkpoint is not None:
            parent = self.checkpoint.find_event(event.parent_event_id)
        if parent is None or parent.get("decision_id") is None:
            return self._observed_result(
                event,
                "outcome_unmatched",
                "未找到对应的选牌建议，结果已保留在原始事件中。",
            )

        decision_id = parent["decision_id"]
        outcome = event.outcome
        if outcome.kind == "closed_unknown":
            result = self._observed_result(
                event,
                "observed_unresolved",
                "选牌界面已关闭，但 Mod 未确认玩家选择；不会把它误标为跳过。",
            )
            result["decision_id"] = decision_id
            return result

        if outcome.kind == "skipped":
            chosen_option = "skip"
        else:
            chosen_option = outcome.selected_card
            parent_options = parent["payload"].get("options", [])
            if outcome.selected_option_index is not None:
                option_index = outcome.selected_option_index
                if option_index >= len(parent_options):
                    raise RealtimeEventValidationError(
                        "selected_option_index is outside the parent options"
                    )
                indexed_card = parent_options[option_index]["card"]
                if (
                    chosen_option is not None
                    and chosen_option.strip().lower()
                    != str(indexed_card).strip().lower()
                ):
                    raise RealtimeEventValidationError(
                        "selected_card does not match selected_option_index"
                    )
                chosen_option = indexed_card
            if not chosen_option:
                raise RealtimeEventValidationError(
                    "selected outcome requires a card or option index"
                )

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
            "advice": None,
            "message": (
                "已在当前局内确认实际选择：跳过。"
                if chosen_option == "skip"
                else f"已在当前局内确认实际选择：{chosen_option}。"
            ),
        }

    def _process_map_choice(self, event: GameStateEvent) -> Dict:
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
            "message": "地图快照已进入当前局检查点；P0 不生成路线推荐。",
        }

    def _process_run_ended(self, event: GameStateEvent) -> Dict:
        if event.run_result is None:
            result = self._observed_result(
                event,
                "session_cleared",
                "旧版结束事件没有最终摘要；仅清理当前局状态。",
            )
            result["session_cleared"] = True
            result["summary_saved"] = False
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
            "summary_saved": True,
            "session_cleared": True,
            "message": "本局最终摘要已保存，中间状态已清理。",
        }

    @staticmethod
    def _observed_result(
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
            "message": message,
        }
