import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import LegalAgent
from src.agent.session_store import SessionStore, SQLiteSessionStore
from src.config import get_settings, setup_logging
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.vector_store import QdrantVectorStore
from src.llm.llm_client import create_helper_llm_client, create_llm_client
from src.rag.rag_pipeline import RAGPipeline

# Basic setup
setup_logging()
settings = get_settings()

# Global singletons
rag_pipeline: RAGPipeline | None = None
legal_agent: LegalAgent | None = None
session_store: SessionStore = SQLiteSessionStore(db_path=settings.CHAT_DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, legal_agent
    print("Initializing RAG pipeline (this might take a moment to load models)...")

    embedder = LegalChunkEmbedder(device="cpu", show_progress=False)
    vector_store = QdrantVectorStore(url=settings.QDRANT_URL)
    llm_client = create_llm_client()

    reranker = None
    if settings.ENABLE_RERANK:
        from src.embedding.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    rag_pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm_client=llm_client,
        reranker=reranker,
        use_graph=settings.USE_GRAPH,
        use_reranker=settings.ENABLE_RERANK,
    )

    # Wire up the agent over the pipeline
    graph_store = rag_pipeline.graph_store  # may be None if Neo4j is down
    legal_agent = LegalAgent(
        rag_pipeline=rag_pipeline,
        graph_store=graph_store,
        llm_client=llm_client,
        helper_llm_client=create_helper_llm_client(),
        session_store=session_store,
    )

    print("RAG Pipeline + Legal Agent initialization complete.")
    try:
        yield
    finally:
        if rag_pipeline:
            rag_pipeline.close()
            rag_pipeline = None
        if hasattr(session_store, "close"):
            session_store.close()  # type: ignore[attr-defined]
        legal_agent = None


app = FastAPI(title="Nepali Legal RAG API", lifespan=lifespan)

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Allow CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


class QueryRequest(BaseModel):
    query: str
    session_id: str = ""   # UUID; frontend generates once per page load
    top_k: int = 5
    regenerate: bool = False


class NavigationInfo(BaseModel):
    process_name: str
    total_steps: int
    current_step: int
    is_complete: bool
    progress_pct: float
    step: dict


class QueryResponse(BaseModel):
    answer: str
    intent: str
    session_id: str
    navigation: NavigationInfo | None = None
    chunks: list
    citations: list = []
    graph_context: list


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not legal_agent:
        raise HTTPException(status_code=503, detail="Agent is still initializing")

    # Assign a session ID when the client sends none
    sid = request.session_id or str(uuid.uuid4())

    try:
        # Run synchronous PyTorch/Neo4j work off the asyncio event loop
        result = await run_in_threadpool(
            legal_agent.respond, request.query, sid, None, request.regenerate
        )

        nav = None
        if result.navigation:
            nav = NavigationInfo(
                process_name=result.navigation.process_name,
                total_steps=result.navigation.total_steps,
                current_step=result.navigation.current_step,
                is_complete=result.navigation.is_complete,
                progress_pct=result.navigation.progress_pct,
                step=result.navigation.step or {},
            )

        return QueryResponse(
            answer=result.answer,
            intent=result.intent.value,
            session_id=result.session_id,
            navigation=nav,
            chunks=result.chunks,
            citations=result.citations,
            graph_context=result.graph,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
async def process_query_stream(request: QueryRequest):
    """Stream intermediate agent steps (SSE) followed by the final response."""
    if not legal_agent:
        raise HTTPException(status_code=503, detail="Agent is still initializing")

    sid = request.session_id or str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    def _run_agent() -> None:
        try:
            for event_name, payload in legal_agent.respond_stream(
                request.query, sid, regenerate=request.regenerate
            ):
                loop.call_soon_threadsafe(queue.put_nowait, (event_name, payload))
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"detail": str(exc)}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def event_generator():
        worker = loop.run_in_executor(None, _run_agent)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event_name, data = item
                yield _format_sse(event_name, data)
        finally:
            await worker

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class RenameConversationRequest(BaseModel):
    title: str


@app.get("/conversations")
async def list_conversations(limit: int = 50):
    safe_limit = max(1, min(limit, 200))
    return {"conversations": session_store.list_sessions(limit=safe_limit)}


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "session_id": session.session_id,
        "title": session.title or "New Chat",
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "active_process": session.active_process,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "citations": m.citations,
                "steps": m.steps,
            }
            for m in session.messages
        ],
    }


@app.patch("/conversations/{session_id}")
async def rename_conversation(session_id: str, body: RenameConversationRequest):
    if not body.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    if not session_store.rename(session_id, body.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"session_id": session_id, "title": body.title.strip()}


@app.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "session_id": session_id}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "pipeline_loaded": rag_pipeline is not None,
        "agent_loaded": legal_agent is not None,
        "sessions_active": len(session_store),
    }
