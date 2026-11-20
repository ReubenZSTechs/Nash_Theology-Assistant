import operator
from typing import Annotated, Any, TypedDict


class RAGState(TypedDict, total=False):
    """
    The shared "notebook" that every node of the RAG graph reads from and writes to.

    LangGraph passes this dictionary from node to node. Each node returns only the
    keys it changed, and LangGraph merges them back into the state.
    """

    # ---- Input (given by the caller) ----
    query: str                      # the user's / LLM's question
    vector_top_k: int               # how many passages to take from the vector database
    graph_top_k: int                # how many verses to take from the knowledge graph
    rerank: bool                    # let the LLM re-score the vector candidates

    # ---- Filled by understand_query ----
    query_vector: list[float]       # embedding of the query
    explicit_verse_ids: list[str]   # verses written in the query, e.g. "Rom.3.28"

    # ---- Filled by the two search branches ----
    vector_results: list[dict[str, Any]]   # advanced RAG results (sermons + Westminster)
    graph_results: list[dict[str, Any]]    # graph RAG results (Bible verses)

    # ---- Filled by build_context ----
    context: str                    # numbered text blocks, ready to paste into an LLM prompt
    sources: list[dict[str, Any]]   # the same results as structured data (for citations / UI)

    # ---- Errors ----
    # The two search branches run in parallel and may both report an error.
    # `operator.add` tells LangGraph to concatenate the lists instead of overwriting.
    errors: Annotated[list[str], operator.add]
