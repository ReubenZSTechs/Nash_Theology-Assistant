"""
Node 2a — Advanced RAG over the vector database (Postgres + pgvector).

"Naive" RAG embeds the query and returns the nearest chunks. Advanced RAG improves on
that in four steps:

  1. Hybrid search  — for every source we run TWO searches:
                        * dense  : cosine similarity between embeddings (finds meaning)
                        * keyword: Postgres full-text search (finds exact words / names)
  2. Fusion         — Reciprocal Rank Fusion (RRF) merges all ranked lists into one.
  3. Reranking      — the LLM reads the question together with each of the best fused
                      candidates and scores how useful the passage is (0..1).
  4. Diversity      — Maximal Marginal Relevance (MMR) picks the final top-k so that the
                      results are relevant to the query but not near-duplicates of each other.

If the LLM reranker is unavailable, step 3 is skipped (with a message in `errors`) and MMR
uses plain embedding similarity instead.

Sources searched: sermon transcript chunks and Westminster Standards units.
"""
import numpy as np

from backend.core.decorators.log_node_call import log_node
from backend.database.postgres.get_connection import PGConnect
from backend.pipelines.rag.states import RAGState
from backend.services.rerank_service import get_rerank_service
from backend.utils.rag_configs import (
    CANDIDATE_MULTIPLIER,
    MMR_LAMBDA,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RRF_K,
    SERMON_DB,
    SERMON_DENSE_SEARCH,
    SERMON_KEYWORD_SEARCH,
    WESTMINSTER_DB,
    WESTMINSTER_DENSE_SEARCH,
    WESTMINSTER_KEYWORD_SEARCH,
)

# (database name, dense query, keyword query) for every source we search.
SOURCES = (
    (SERMON_DB, SERMON_DENSE_SEARCH, SERMON_KEYWORD_SEARCH),
    (WESTMINSTER_DB, WESTMINSTER_DENSE_SEARCH, WESTMINSTER_KEYWORD_SEARCH),
)

COLUMNS = ("id", "source", "title", "text", "url", "embedding", "score")


# ---------------------------------------------------------
# Step 1: hybrid search
# ---------------------------------------------------------
def to_numpy(embedding) -> np.ndarray:
    """
    pgvector returns embeddings as `pgvector.Vector` objects, which numpy cannot read
    directly. Convert them (or plain lists) into a float32 numpy array.
    """
    if hasattr(embedding, "to_numpy"):
        embedding = embedding.to_numpy()
    return np.asarray(embedding, dtype=np.float32)


def run_search(connection, sql: str, params: dict) -> list[dict]:
    """Run one search query and return its rows as dictionaries (best first)."""
    rows = connection.execute(sql, params).fetchall()
    documents = [dict(zip(COLUMNS, row)) for row in rows]
    for document in documents:
        document["embedding"] = to_numpy(document["embedding"])
    return documents


def hybrid_search(query: str, query_vector: list[float], limit: int) -> tuple[list[list[dict]], list[str]]:
    """
    Run the dense and the keyword search on every source.

    Returns (ranked_lists, errors). A source that cannot be reached is skipped and
    reported in `errors`, so one broken database does not stop the whole search.
    """
    params = {"query": query, "vector": np.asarray(query_vector, dtype=np.float32), "limit": limit}
    ranked_lists, errors = [], []

    for database, dense_sql, keyword_sql in SOURCES:
        try:
            with PGConnect(dbname=database) as postgres:
                connection = postgres.get_db_connection()
                ranked_lists.append(run_search(connection, dense_sql, params))
                ranked_lists.append(run_search(connection, keyword_sql, params))
        except Exception as error:
            errors.append(f"vector search on '{database}' failed: {error}")

    return ranked_lists, errors


# ---------------------------------------------------------
# Step 2: Reciprocal Rank Fusion
# ---------------------------------------------------------
def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """
    Merge several ranked lists into one.

    Each document gets   RRF(d) = sum over lists of  1 / (k + rank of d in that list)
    (rank starts at 1). A document that appears near the top of many lists wins.
    RRF only looks at ranks, so it does not matter that cosine similarity and
    full-text scores are on completely different scales.
    """
    fused = {}
    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            key = (document["source"], document["id"])
            if key not in fused:
                fused[key] = {**document, "rrf_score": 0.0}
            fused[key]["rrf_score"] += 1.0 / (k + rank)

    return sorted(fused.values(), key=lambda document: document["rrf_score"], reverse=True)


# ---------------------------------------------------------
# Step 3: LLM reranking
# ---------------------------------------------------------
def rerank_text(document: dict) -> str:
    """What the LLM reads for one candidate: its title and its text."""
    return f"{document['title']}\n{document['text']}"


def llm_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Ask the LLM to score every candidate and store the score as `rerank_score`.
    Returns the candidates sorted by that score (best first).
    """
    ranking = get_rerank_service().rerank(query, [rerank_text(document) for document in candidates])
    reranked = []
    for item in ranking:
        document = candidates[item["index"]]
        document["rerank_score"] = item["relevance_score"]
        reranked.append(document)
    return reranked


# ---------------------------------------------------------
# Step 4: Maximal Marginal Relevance
# ---------------------------------------------------------
def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mmr_select(candidates: list[dict], query_vector: list[float], top_k: int, lambda_: float = MMR_LAMBDA) -> list[dict]:
    """
    Pick `top_k` candidates one by one. At every step choose the candidate with the best

        MMR = lambda * relevance(candidate) - (1 - lambda) * max sim(candidate, already chosen)

    `relevance` is the LLM rerank score when available, otherwise the cosine similarity
    between the candidate and the query. The second part punishes repeating what we
    already have (e.g. two almost identical sermon chunks).
    """
    remaining = []
    for candidate in candidates:
        similarity = cosine_similarity(candidate["embedding"], query_vector)
        relevance = candidate.get("rerank_score", similarity)
        remaining.append({**candidate, "similarity": similarity, "relevance": relevance})

    selected = []
    while remaining and len(selected) < top_k:
        best_index, best_score = 0, -float("inf")

        for index, candidate in enumerate(remaining):
            redundancy = max(
                (cosine_similarity(candidate["embedding"], chosen["embedding"]) for chosen in selected),
                default=0.0,
            )
            score = lambda_ * candidate["relevance"] - (1 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = index, score

        selected.append(remaining.pop(best_index))

    return selected


def clean_result(document: dict) -> dict:
    """Drop the large embedding vector and keep only what the caller needs."""
    rerank_score = document.get("rerank_score")
    return {
        "id": document["id"],
        "source": document["source"],
        "title": document["title"],
        "text": document["text"],
        "url": document["url"],
        "similarity": round(document["similarity"], 4),
        "rrf_score": round(document["rrf_score"], 5),
        "rerank_score": round(rerank_score, 4) if rerank_score is not None else None,
    }


# ---------------------------------------------------------
# The LangGraph node
# ---------------------------------------------------------
def search_vector_database(query: str, query_vector: list[float], top_k: int, rerank: bool = RERANK_ENABLED) -> tuple[list[dict], list[str]]:
    """The full advanced-RAG algorithm: hybrid search -> RRF -> LLM rerank -> MMR."""
    if top_k <= 0:
        return [], []

    pool_size = top_k * CANDIDATE_MULTIPLIER
    ranked_lists, errors = hybrid_search(query, query_vector, limit=pool_size)
    candidates = reciprocal_rank_fusion(ranked_lists)[:pool_size]   # keep the best fused candidates

    if rerank and candidates:
        rerank_pool = candidates[:max(RERANK_CANDIDATES, top_k * 2)]
        try:
            candidates = llm_rerank(query, rerank_pool)
        except Exception as error:
            errors.append(f"LLM reranking skipped, using embedding similarity instead: {error}")

    selected = mmr_select(candidates, query_vector, top_k)
    return [clean_result(document) for document in selected], errors


@log_node("rag.vector_search")
def vector_search(state: RAGState) -> dict:
    results, errors = search_vector_database(
        state["query"], state["query_vector"], state["vector_top_k"], state.get("rerank", RERANK_ENABLED)
    )
    return {"vector_results": results, "errors": errors}
