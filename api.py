"""ThermoWaste-RAG FastAPI service (Day 10: errors and timing).

Place this file in the project root as api.py.
Start from the project root:
    python -m uvicorn api:app --reload

Python 3.8+ compatible syntax; model initializations remain the same as app.py.
"""

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.loader import load_folder
from src.splitter import split_pages
from src.embedder import Embedder
from src.vector_store import FaissVectorStore
from src.retriever import Retriever
from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.llm import LLMClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("thermowaste.api")


@dataclass
class RAGPipeline:
    hybrid: HybridRetriever
    reranker: Reranker
    llm: LLMClient
    chunk_count: int
    # Serialize inference on one local pipeline to avoid shared-model races.
    inference_lock: Lock = field(default_factory=Lock)
    llm_lock: Lock = field(default_factory=Lock)


def build_pipeline() -> RAGPipeline:
    """Build resources once per Uvicorn worker, using original app.py settings."""
    logger.info("Loading PDF corpus ...")
    pages = load_folder("data/papers")
    chunks = split_pages(pages, chunk_size=800, chunk_overlap=150)
    if not chunks:
        raise RuntimeError("No chunks found. Check data/papers and the PDF loader.")
    logger.info("Loaded %d pages and %d chunks", len(pages), len(chunks))

    embedder = Embedder()
    embeddings = embedder.encode_documents([chunk["text"] for chunk in chunks])
    store = FaissVectorStore()
    store.build(embeddings)
    dense = Retriever(embedder=embedder, vector_store=store, chunks=chunks)
    bm25 = BM25Retriever(chunks=chunks)

    pipeline = RAGPipeline(
        hybrid=HybridRetriever(dense_retriever=dense, bm25_retriever=bm25),
        reranker=Reranker(),
        llm=LLMClient(),
        chunk_count=len(chunks),
    )
    logger.info("RAG pipeline ready")
    return pipeline


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.rag = build_pipeline()
    try:
        yield
    finally:
        del application.state.rag


app = FastAPI(
    title="ThermoWaste-RAG API",
    description="Scientific literature RAG system",
    version="0.3.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchHit(BaseModel):
    source: str
    page: int
    text: str
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Timings(BaseModel):
    # Milliseconds; stage values measure computation, total_ms includes overhead.
    hybrid_ms: float
    rerank_ms: float
    total_ms: float
    llm_ms: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchHit]
    timings_ms: Timings


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchHit]
    timings_ms: Timings


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def retrieve_evidence(query: str, top_k: int) -> Tuple[List[SearchHit], float, float]:
    """Run Hybrid and Reranker; return evidence and stage latencies."""
    pipeline: RAGPipeline = app.state.rag
    candidate_k = max(20, top_k * 4)
    hybrid_top_k = max(10, top_k * 2)

    with pipeline.inference_lock:
        start = time.perf_counter()
        candidates = pipeline.hybrid.retrieve(
            query,
            top_k=hybrid_top_k,
            candidate_k=candidate_k,
            rrf_k=60,
        )
        hybrid_ms = _ms(start)

        start = time.perf_counter()
        ranked = pipeline.reranker.rerank(query, candidates, top_k=top_k)
        rerank_ms = _ms(start)

    hits = [
        SearchHit(
            source=item["source"],
            page=item["page"],
            text=item["text"],
            hybrid_score=item.get("hybrid_score"),
            rerank_score=item.get("rerank_score"),
        )
        for item in ranked
    ]
    return hits, hybrid_ms, rerank_ms


def _validate_query(req: QueryRequest) -> str:
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")
    return query


@app.get("/health")
def health():
    pipeline: RAGPipeline = app.state.rag
    return {"status": "ok", "service": "ThermoWaste-RAG", "chunks": pipeline.chunk_count}


@app.post("/search", response_model=SearchResponse)
def search(req: QueryRequest):
    query = _validate_query(req)
    total_start = time.perf_counter()
    try:
        hits, hybrid_ms, rerank_ms = retrieve_evidence(query, req.top_k)
    except Exception as exc:
        # Details remain in server logs; avoid leaking files, credentials and tracebacks.
        logger.exception("Search backend failed")
        raise HTTPException(status_code=503, detail="Retrieval service temporarily unavailable") from exc

    total_ms = _ms(total_start)
    logger.info("/search success: hits=%d duration_ms=%.2f", len(hits), total_ms)
    return SearchResponse(
        query=query,
        count=len(hits),
        results=hits,
        timings_ms=Timings(
            hybrid_ms=hybrid_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
        ),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: QueryRequest):
    query = _validate_query(req)
    total_start = time.perf_counter()
    try:
        evidence, hybrid_ms, rerank_ms = retrieve_evidence(query, req.top_k)
    except Exception as exc:
        logger.exception("Retrieval failed while handling /ask")
        raise HTTPException(status_code=503, detail="Retrieval service temporarily unavailable") from exc

    if not evidence:
        return AskResponse(
            query=query,
            answer="No supporting passages were retrieved from the literature corpus.",
            sources=[],
            timings_ms=Timings(
                hybrid_ms=hybrid_ms,
                rerank_ms=rerank_ms,
                total_ms=_ms(total_start),
            ),
        )

    # The existing LLMClient.answer() takes a list of dictionaries.
    passage_dicts = [
        hit.model_dump() if hasattr(hit, "model_dump") else hit.dict()
        for hit in evidence
    ]
    llm_start = time.perf_counter()
    try:
        with app.state.rag.llm_lock:
            answer = app.state.rag.llm.answer(query=query, results=passage_dicts)
        llm_ms = _ms(llm_start)
    except Exception as exc:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail="Language model service temporarily unavailable") from exc

    total_ms = _ms(total_start)
    logger.info(
        "/ask success: sources=%d hybrid_ms=%.2f rerank_ms=%.2f llm_ms=%.2f total_ms=%.2f",
        len(evidence), hybrid_ms, rerank_ms, llm_ms, total_ms,
    )
    return AskResponse(
        query=query,
        answer=str(answer),
        sources=evidence,
        timings_ms=Timings(
            hybrid_ms=hybrid_ms,
            rerank_ms=rerank_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        ),
    )
