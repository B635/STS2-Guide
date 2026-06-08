import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

os.environ["HF_HOME"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import RERANKER_CANDIDATE_N, RETRIEVE_TOP_N
from rag.agent import AgentConfig, run_agent
from rag.bm25 import build_bm25_index
from rag.chat import create_client
from rag.embedder import load_model, load_or_compute_embeddings
from rag.errors import handle_api_error, handle_file_error
from rag.knowledge import load_knowledge
from rag.langgraph_agent import run_langgraph_agent


@dataclass
class AppResources:
    docs: List[str]
    items: List[Dict]
    index: Dict
    store: object
    model: object
    client: object
    bm25_index: object


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: List[ChatMessage] = Field(default_factory=list)
    use_langgraph: bool = False
    use_reranker: bool = False
    top_n: int = Field(default=RETRIEVE_TOP_N, ge=1, le=10)
    candidate_n: int = Field(default=RERANKER_CANDIDATE_N, ge=5, le=50)


class AgentStepResponse(BaseModel):
    tool: str
    detail: str
    observation: str


class SourceResponse(BaseModel):
    id: int
    text: str
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    index: Optional[int] = None


class VerificationResponse(BaseModel):
    passed: bool
    notes: List[str]
    stats: Dict[str, int]


class ChatResponse(BaseModel):
    answer: str
    retrieve_query: str
    selected_tool: str
    reason: str
    sources: List[SourceResponse]
    verification: Optional[VerificationResponse]
    steps: List[AgentStepResponse]


class HealthResponse(BaseModel):
    status: str
    service: str


app = FastAPI(
    title="STS2 Guide Agent API",
    description="FastAPI service wrapper for the STS2 Tool-Using Agent RAG pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_resources() -> AppResources:
    try:
        docs, items, index = load_knowledge()
        model = load_model()
        store = load_or_compute_embeddings(docs, model)
        client = create_client()
        bm25_index = build_bm25_index(docs)
        return AppResources(
            docs=docs,
            items=items,
            index=index,
            store=store,
            model=model,
            client=client,
            bm25_index=bm25_index,
        )
    except Exception as exc:
        raise RuntimeError(handle_file_error(exc, "./data/knowledge.json")) from exc


@lru_cache(maxsize=1)
def get_reranker():
    from rag.reranker import load_reranker

    return load_reranker()


def _serialize_sources(results: List[Dict]) -> List[SourceResponse]:
    sources: List[SourceResponse] = []
    for idx, item in enumerate(results, start=1):
        sources.append(
            SourceResponse(
                id=idx,
                text=str(item.get("text", "")),
                score=item.get("score"),
                rerank_score=item.get("rerank_score"),
                retrieval_score=item.get("retrieval_score"),
                index=item.get("index"),
            )
        )
    return sources


def _serialize_verification(result) -> Optional[VerificationResponse]:
    if result is None:
        return None
    return VerificationResponse(
        passed=bool(result.passed),
        notes=list(result.notes),
        stats=dict(result.stats),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="sts2-guide-agent-api")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        resources = get_resources()
        reranker = get_reranker() if request.use_reranker else None
        runner = run_langgraph_agent if request.use_langgraph else run_agent
        history = [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in request.history
        ]
        result = runner(
            request.question,
            history,
            resources.docs,
            resources.items,
            resources.index,
            resources.store,
            resources.model,
            resources.client,
            bm25_index=resources.bm25_index,
            reranker=reranker,
            config=AgentConfig(
                top_n=request.top_n,
                candidate_n=request.candidate_n,
            ),
        )
        return ChatResponse(
            answer=result.answer,
            retrieve_query=result.retrieve_query,
            selected_tool=result.selected_tool,
            reason=result.reason,
            sources=_serialize_sources(result.results),
            verification=_serialize_verification(result.verification),
            steps=[
                AgentStepResponse(
                    tool=step.tool,
                    detail=step.detail,
                    observation=step.observation,
                )
                for step in result.steps
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=handle_api_error(exc)) from exc
