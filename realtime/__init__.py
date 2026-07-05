"""Real-time game-state ingestion for the local STS2 companion."""

from realtime.processor import RealtimeEventProcessor
from realtime.protocol import GameStateEvent

__all__ = ["GameStateEvent", "RealtimeEventProcessor"]
