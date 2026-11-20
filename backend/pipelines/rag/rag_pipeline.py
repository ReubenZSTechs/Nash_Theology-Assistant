import argparse
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, HTTPException
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from backend.database.neo4j.get_connection import get_neo4j_driver
from backend.database.postgres.get_connection import PGConnect
from backend.pipelines.rag import nodes
from backend.pipelines.rag.states import RAGState
from backend.services.embedding_service import get_embedding_service
from backend.services.rerank_service import get_rerank_service
from backend.utils.rag_configs import (
    DEFAULT_TOP_K,
    GRAPH_HEALTH,
    MAX_RERANK_DOCUMENTS,
    MAX_TOP_K,
    RERANK_ENABLED,
    SERMON_DB,
    WESTMINSTER_DB,
)


# RAG Graph
@lru_cache(maxsize=1)
def build_rag_graph():
    """Wire the nodes together and compile the graph (done once, then cached)."""
    graph = StateGraph(RAGState)

    graph.add_node("understand_query", nodes.understand_query)
    graph.add_node("vector_search", nodes.vector_search)
    graph.add_node("graph_search", nodes.graph_search)
    graph.add_node("build_context", nodes.build_context)

    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "vector_search")              # fan out:
    graph.add_edge("understand_query", "graph_search")               #   both run in parallel
    graph.add_edge(["vector_search", "graph_search"], "build_context")  # fan in: wait for both
    graph.add_edge("build_context", END)

    return graph.compile()


# RAG Services
def check_top_k(name: str, value: int) -> None:
    if not isinstance(value, int) or not 0 <= value <= MAX_TOP_K:
        raise ValueError(f"{name} must be an integer between 0 and {MAX_TOP_K}, got {value!r}")


def retrieve(query: str, vector_top_k: int = DEFAULT_TOP_K, graph_top_k: int = DEFAULT_TOP_K, rerank: bool = RERANK_ENABLED) -> dict:
    """
    Run the full hybrid RAG pipeline.

    Args:
        query:        the question to search for.
        vector_top_k: number of sermon / Westminster passages (0 = skip the vector database).
        graph_top_k:  number of Bible verses from the knowledge graph (0 = skip the graph).
        rerank:       let the LLM re-score the vector candidates before the final selection.

    Returns a dict with: query, context, sources, vector_results, graph_results, errors.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    check_top_k("vector_top_k", vector_top_k)
    check_top_k("graph_top_k", graph_top_k)
    if vector_top_k == 0 and graph_top_k == 0:
        raise ValueError("at least one of vector_top_k and graph_top_k must be greater than 0")

    final_state = build_rag_graph().invoke({
        "query": query.strip(),
        "vector_top_k": vector_top_k,
        "graph_top_k": graph_top_k,
        "rerank": rerank,
        "errors": [],
    })

    return {
        "query": final_state["query"],
        "context": final_state["context"],
        "sources": final_state["sources"],
        "vector_results": final_state.get("vector_results", []),
        "graph_results": final_state.get("graph_results", []),
        "errors": final_state.get("errors", []),
    }


def vector_search(query: str, top_k: int = DEFAULT_TOP_K, rerank: bool = RERANK_ENABLED) -> dict:
    """Only the advanced-RAG branch (sermons + Westminster Standards)."""
    return retrieve(query, vector_top_k=top_k, graph_top_k=0, rerank=rerank)


def graph_search(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Only the graph-RAG branch (Bible verses + cross-references)."""
    return retrieve(query, vector_top_k=0, graph_top_k=top_k)


def rerank(query: str, documents: list[str], top_n: Optional[int] = None) -> dict:
    """
    Rerank any list of texts with the LLM (the same reranker the pipeline uses).

    Returns {"model": ..., "results": [{"index", "relevance_score", "document"}, ...]},
    best first. `index` is the position of the document in the input list.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not documents or len(documents) > MAX_RERANK_DOCUMENTS:
        raise ValueError(f"documents must contain between 1 and {MAX_RERANK_DOCUMENTS} texts")
    if any(not isinstance(document, str) or not document.strip() for document in documents):
        raise ValueError("every document must be a non-empty string")
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1")

    service = get_rerank_service()
    ranking = service.rerank(query.strip(), documents, top_n)
    return {
        "model": service.model,
        "results": [{**item, "document": documents[item["index"]]} for item in ranking],
    }


def check_component(check) -> str:
    """Run one health check and return "ok" or the error message."""
    try:
        check()
        return "ok"
    except Exception as error:
        return f"error: {error}"


def check_postgres(database: str) -> None:
    with PGConnect(dbname=database) as postgres:
        postgres.get_db_connection().execute("SELECT 1")


def health() -> dict:
    """Check every component the pipeline depends on."""
    components = {
        "embedding": check_component(lambda: get_embedding_service().check_server()),
        "reranker_llm": check_component(lambda: get_rerank_service().check_server()),
        "neo4j": check_component(lambda: get_neo4j_driver().execute_query(GRAPH_HEALTH)),
        "sermon_db": check_component(lambda: check_postgres(SERMON_DB)),
        "westminster_db": check_component(lambda: check_postgres(WESTMINSTER_DB)),
    }
    status = "ok" if all(value == "ok" for value in components.values()) else "degraded"
    return {"status": status, "components": components}


# Endpoints
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to search for.")
    vector_top_k: int = Field(DEFAULT_TOP_K, ge=0, le=MAX_TOP_K, description="Passages from the vector database.")
    graph_top_k: int = Field(DEFAULT_TOP_K, ge=0, le=MAX_TOP_K, description="Verses from the knowledge graph.")
    rerank: bool = Field(RERANK_ENABLED, description="Let the LLM re-score the vector candidates.")


class SingleSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    rerank: bool = Field(RERANK_ENABLED, description="Only used by /vector-search.")


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(..., min_length=1, max_length=MAX_RERANK_DOCUMENTS)
    top_n: Optional[int] = Field(None, ge=1, description="Return only the best N (default: all).")


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


class RetrieveResponse(BaseModel):
    query: str
    context: str
    sources: list[dict]
    vector_results: list[dict]
    graph_results: list[dict]
    errors: list[str]


router = APIRouter(prefix="/rag", tags=["rag"])


def run_service(service, *args, **kwargs) -> dict:
    """Call a service function and translate Python errors into HTTP errors."""
    try:
        return service(*args, **kwargs)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"RAG pipeline failed: {error}") from error


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest):
    return run_service(retrieve, request.query, request.vector_top_k, request.graph_top_k, request.rerank)


@router.post("/vector-search", response_model=RetrieveResponse)
def vector_search_endpoint(request: SingleSearchRequest):
    return run_service(vector_search, request.query, request.top_k, request.rerank)


@router.post("/graph-search", response_model=RetrieveResponse)
def graph_search_endpoint(request: SingleSearchRequest):
    return run_service(graph_search, request.query, request.top_k)


@router.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(request: RerankRequest):
    return run_service(rerank, request.query, request.documents, request.top_n)


@router.get("/health")
def health_endpoint():
    return health()


# Demo purposes
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Nash hybrid RAG pipeline for one query")
    parser.add_argument("query")
    parser.add_argument("--vector-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--graph-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--no-rerank", action="store_true", help="skip the LLM reranking step")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = retrieve(args.query, args.vector_top_k, args.graph_top_k, rerank=not args.no_rerank)
    print(f"Context:\n{result['context']}")
    for error in result["errors"]:
        print(f"[warning] {error}")
