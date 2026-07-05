import os
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

os.environ["HF_HOME"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from advisor.card_reward import recommend_card_reward
from advisor.data_sources import (
    LocalCardTierSource,
    load_local_card_tiers,
)
from config import (
    COMMUNITY_SCORES_FILE,
    KNOWLEDGE_FILE,
    LOCAL_CARD_TIERS_FILE,
    RELATIONAL_DB_FILE,
    RERANKER_CANDIDATE_N,
    RETRIEVE_TOP_N,
)
from rag.agent import AgentConfig, run_agent
from rag.bm25 import build_bm25_index
from rag.chat import create_client
from rag.embedder import load_model, load_or_compute_embeddings
from rag.errors import handle_api_error, handle_file_error
from rag.knowledge import load_runtime_knowledge
from rag.langgraph_agent import run_langgraph_agent
from realtime.processor import (
    RealtimeEventProcessor,
    RealtimeEventValidationError,
)
from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.file_bridge import default_checkpoint_path, default_output_path
from realtime.protocol import GameStateEvent
from realtime.session import TransientSessionStore
from storage.relational import RelationalRepository


@dataclass
class AppResources:
    docs: List[str]
    items: List[Dict]
    index: Dict
    store: object
    model: object
    client: object
    bm25_index: object
    repository: RelationalRepository


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


class DeckCardInput(BaseModel):
    card: str = Field(..., min_length=1)
    count: int = Field(default=1, ge=1, le=99)
    upgrades: int = Field(default=0, ge=0, le=9)


class CardRewardOptionInput(BaseModel):
    card: str = Field(..., min_length=1)
    upgrades: int = Field(default=0, ge=0, le=9)


class RunStateInput(BaseModel):
    character: str = Field(..., min_length=1)
    ascension: int = Field(default=0, ge=0)
    act: int = Field(default=1, ge=1)
    floor: int = Field(default=1, ge=0)
    hp: Optional[int] = Field(default=None, ge=0)
    max_hp: Optional[int] = Field(default=None, ge=1)
    gold: Optional[int] = Field(default=None, ge=0)
    energy: int = Field(default=3, ge=0, le=99)
    game_version: Optional[str] = None
    deck: List[Union[str, DeckCardInput]] = Field(default_factory=list)
    relics: List[str] = Field(default_factory=list)


class CardRewardRequest(BaseModel):
    state: RunStateInput
    options: List[Union[str, CardRewardOptionInput]] = Field(
        ...,
        min_length=1,
        max_length=10,
    )
    persist: bool = False


class RecommendationFactorResponse(BaseModel):
    code: str
    delta: float
    message: str
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    snapshot_id: Optional[str] = None
    sample_size: Optional[int] = None
    raw_score: Optional[float] = None
    signal: Optional[float] = None
    weight: Optional[float] = None


class CardRecommendationResponse(BaseModel):
    card: str
    card_id: Optional[str] = None
    upgrades: int
    enchantment: Optional[str] = None
    enchantment_amount: Optional[int] = None
    affliction: Optional[str] = None
    affliction_amount: Optional[int] = None
    score: float
    state_score: float
    rank: int
    option_index: int
    factors: List[RecommendationFactorResponse]
    known: bool


class CardRewardResponse(BaseModel):
    state_id: Optional[str] = None
    decision_id: Optional[str] = None
    method: str
    recommended_option: Optional[str] = None
    recommended_option_index: int
    decision_status: str
    skip_recommended: bool
    skip_score: float
    skip_candidate: Dict
    confidence: str
    recommendations: List[CardRecommendationResponse]
    profile: Dict
    disclaimer: str


class DecisionOutcomeRequest(BaseModel):
    chosen_option: Optional[str] = None
    run_won: Optional[bool] = None
    final_floor: Optional[int] = Field(default=None, ge=0)


class RealtimeEventResponse(BaseModel):
    event_id: str
    event_type: str
    status: str
    duplicate: bool
    state_id: Optional[str] = None
    decision_id: Optional[str] = None
    advice: Optional[Dict] = None
    message: str


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
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    original_url: Optional[str] = None
    section: Optional[str] = None
    language: Optional[str] = None
    published_at: Optional[str] = None


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
def get_relational_repository() -> RelationalRepository:
    repository = RelationalRepository(RELATIONAL_DB_FILE)
    repository.ensure_schema()
    return repository


@lru_cache(maxsize=1)
def get_realtime_sessions() -> TransientSessionStore:
    return TransientSessionStore()


@lru_cache(maxsize=1)
def get_realtime_checkpoint() -> ActiveRunCheckpointStore:
    return ActiveRunCheckpointStore(default_checkpoint_path())


@lru_cache(maxsize=1)
def get_local_card_tiers() -> LocalCardTierSource:
    return load_local_card_tiers(LOCAL_CARD_TIERS_FILE)


def _sync_relational_sources(
    repository: RelationalRepository,
) -> None:
    repository.sync_catalog(KNOWLEDGE_FILE)
    if os.path.exists(COMMUNITY_SCORES_FILE):
        repository.sync_entity_statistics(COMMUNITY_SCORES_FILE)


@lru_cache(maxsize=1)
def get_resources() -> AppResources:
    try:
        repository = get_relational_repository()
        docs, items, index = load_runtime_knowledge(repository)
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
            repository=repository,
        )
    except Exception as exc:
        raise RuntimeError(handle_file_error(exc, "./data/knowledge.json")) from exc


@lru_cache(maxsize=1)
def get_reranker():
    from rag.reranker import load_reranker

    return load_reranker()


def _serialize_sources(results: List[Dict]) -> List[SourceResponse]:
    def safe_url(value) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip()
        return text if urlparse(text).scheme in {"http", "https"} else None

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
                source_type=item.get("source_type"),
                source_id=item.get("source_id"),
                title=item.get("title"),
                author=item.get("author"),
                url=safe_url(item.get("url")),
                original_url=safe_url(item.get("original_url")),
                section=item.get("section"),
                language=item.get("language"),
                published_at=item.get("published_at"),
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


def _dump_model(value) -> Dict:
    return value.model_dump() if hasattr(value, "model_dump") else value.dict()


def _normalize_deck(entries: List[Union[str, DeckCardInput]]) -> List[Dict]:
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"card": entry, "count": 1, "upgrades": 0})
        else:
            normalized.append(_dump_model(entry))
    return normalized


def _normalize_options(
    entries: List[Union[str, CardRewardOptionInput]],
) -> List[Dict]:
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"card": entry, "upgrades": 0})
        else:
            normalized.append(_dump_model(entry))
    return normalized


@app.get("/storage/status")
def storage_status() -> Dict:
    repository = get_relational_repository()
    _sync_relational_sources(repository)
    return {
        "relational": {
            "backend": "sqlite",
            "contains": [
                "structured_game_entities",
                "source_snapshots",
                "entity_statistics",
            ],
            "catalog_counts": repository.catalog_counts(),
            "statistics_counts": repository.statistics_status(),
        },
        "vector": {
            "backend": "faiss",
            "contains": ["unstructured_guide_chunks"],
        },
    }


@app.post(
    "/events/game-state",
    response_model=RealtimeEventResponse,
)
def ingest_game_state_event(
    event: GameStateEvent,
) -> RealtimeEventResponse:
    repository = get_relational_repository()
    _sync_relational_sources(repository)
    try:
        result = RealtimeEventProcessor(
            repository,
            sessions=get_realtime_sessions(),
            checkpoint=get_realtime_checkpoint(),
            local_tiers=get_local_card_tiers(),
        ).process(event)
    except RealtimeEventValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        status = 409 if "identity collision" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    return RealtimeEventResponse(**result)


@app.get("/events/game-state/latest")
def get_latest_game_state_event() -> Dict:
    event = get_realtime_sessions().latest()
    if event is None:
        checkpoint = get_realtime_checkpoint().load()
        if checkpoint is not None:
            event = checkpoint.get("latest_event")
    if event is None:
        event = _load_bridge_event()
    return {"event": event}


@app.get("/events/game-state/{event_id}")
def get_game_state_event(event_id: str) -> Dict:
    event = get_realtime_sessions().load(event_id)
    if event is None:
        event = get_realtime_checkpoint().find_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Game-state event not found")
    return event


def _load_bridge_event() -> Optional[Dict]:
    path = default_output_path()
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not result.get("event_id"):
        return None
    return {
        "event_id": result["event_id"],
        "run_id": result.get("run_id"),
        "sequence": result.get("sequence"),
        "event_type": result.get("event_type"),
        "emitted_at": result.get("emitted_at"),
        "processed_at": result.get("processed_at"),
        "status": result.get("status"),
        "state_id": None,
        "decision_id": result.get("decision_id"),
        "result": result,
    }


@app.post("/recommend/card-reward", response_model=CardRewardResponse)
def recommend_card_reward_endpoint(
    request: CardRewardRequest,
) -> CardRewardResponse:
    repository = get_relational_repository()
    _sync_relational_sources(repository)
    state = _dump_model(request.state)
    state["deck"] = _normalize_deck(request.state.deck)
    options = _normalize_options(request.options)
    if repository.find_entity("characters", state["character"]) is None:
        raise HTTPException(
            status_code=400,
            detail="Unknown character ID or name",
        )

    try:
        result = recommend_card_reward(
            state,
            options,
            repository,
            local_tiers=get_local_card_tiers(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state_id = None
    decision_id = None
    if request.persist:
        raise HTTPException(
            status_code=400,
            detail=(
                "P0 does not persist individual run states or decisions; "
                "use the active-run checkpoint instead."
            ),
        )
    return CardRewardResponse(
        state_id=state_id,
        decision_id=decision_id,
        **result,
    )


@app.get("/decisions/{decision_id}")
def get_decision(decision_id: str) -> Dict:
    raise HTTPException(
        status_code=410,
        detail="Per-decision SQLite history is disabled in P0.",
    )


@app.post("/decisions/{decision_id}/outcome")
def record_decision_outcome(
    decision_id: str,
    request: DecisionOutcomeRequest,
) -> Dict:
    raise HTTPException(
        status_code=410,
        detail="Per-decision outcome history is disabled in P0.",
    )


@app.post("/recommend/paths")
def recommend_paths_endpoint(request: Dict) -> Dict:
    raise HTTPException(
        status_code=410,
        detail=(
            "Route recommendation is disabled until real map edges pass "
            "fixture and in-game verification."
        ),
    )


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
