"""Function-calling schemas for the agent planner."""
from __future__ import annotations

from typing import Any, Dict, List

from config import MAX_SUB_QUERIES


KNOWLEDGE_TYPES = ["", "cards", "relics", "potions", "monsters", "characters"]


def _filters_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": KNOWLEDGE_TYPES,
            }
        },
    }


AGENT_FUNCTION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Default factual retrieval over the STS2 knowledge base using BM25 + vector search + RRF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for the retrieval tool."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": _filters_schema(),
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyde_hybrid_search",
            "description": "Retrieval for descriptive or open-ended questions where user wording may differ from knowledge entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query used to generate the hypothetical document."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": _filters_schema(),
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_query_search",
            "description": "Retrieval for comparison, multi-entity, multi-dimensional, or strategy questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Main search query."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": _filters_schema(),
                    "sub_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Independent sub-queries for complex questions, at most {MAX_SUB_QUERIES}.",
                    },
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Semantic vector-search fallback when BM25/hybrid search is unavailable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for vector retrieval."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": _filters_schema(),
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
]
