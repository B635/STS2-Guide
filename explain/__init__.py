"""Bounded, read-only explanation primitives for STS2 Guide.

This package is intentionally independent from the realtime Host and from the
legacy Web/RAG experiments.  It accepts an already-produced canonical
Recommendation and can only explain that immutable input.
"""

from explain.contracts import (
    ExplainIdentity,
    ExplainRequest,
    ExplainResponse,
    canonical_recommendation_digest,
    canonicalize_recommendation,
)
from explain.worker import ExplainWorker

__all__ = [
    "ExplainIdentity",
    "ExplainRequest",
    "ExplainResponse",
    "ExplainWorker",
    "canonical_recommendation_digest",
    "canonicalize_recommendation",
]
