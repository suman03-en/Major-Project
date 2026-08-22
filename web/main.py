from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

# Add project root to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_settings, setup_logging
from src.rag.rag_pipeline import RAGPipeline
from src.embedding.embedder import LegalChunkEmbedder
from src.embedding.vector_store import QdrantVectorStore
from src.llm.llm_client import MistralLLMClient

# Basic setup
setup_logging()
settings = get_settings()

app = FastAPI(title="Nepali Legal RAG API")

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

# Global RAG Pipeline instance
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    print("Initializing RAG pipeline (this might take a moment to load models)...")
    
    embedder = LegalChunkEmbedder(show_progress=False)
    vector_store = QdrantVectorStore(url=settings.QDRANT_URL)
    llm_client = MistralLLMClient(model=settings.MISTRAL_MODEL)
    
    # Check if reranker is enabled in env
    use_reranker = os.getenv("ENABLE_RERANK", "false").lower() == "true"
    reranker = None
    if use_reranker:
        from src.embedding.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        
    use_graph = os.getenv("RUN_GRAPH_INGEST", "false").lower() == "true"
    
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        llm_client=llm_client,
        reranker=reranker,
        use_graph=use_graph,
        use_reranker=use_reranker
    )
    print("RAG Pipeline initialization complete.")

@app.on_event("shutdown")
async def shutdown_event():
    global rag_pipeline
    if rag_pipeline:
        rag_pipeline.close()

@app.get("/")
async def serve_frontend(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    chunks: list
    graph_context: list

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is still initializing")
        
    try:
        # We use the synchronous answer() method to return a full JSON response
        result = rag_pipeline.answer(query=request.query, top_k=request.top_k)
        
        return QueryResponse(
            answer=result["answer"],
            chunks=result["chunks"],
            graph_context=result["graph"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "pipeline_loaded": rag_pipeline is not None}
