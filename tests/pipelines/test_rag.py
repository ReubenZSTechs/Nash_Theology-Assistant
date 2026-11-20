"""
Unit tests for the pure logic of the RAG pipeline and the retrieval tool.
None of these tests need Postgres, Neo4j or the embedding server.

Run with:  pytest tests/pipelines/test_rag.py
"""
import json

import numpy as np
import pytest

from backend.configs.tools import retrieval as retrieval_tool
from backend.pipelines.rag.nodes.build_context import build_context_and_sources, shorten
from backend.pipelines.rag.nodes.graph_search import choose_seeds, score_neighbours, select_verses
from backend.pipelines.rag.nodes.understand_query import extract_verse_ids
from backend.pipelines.rag.nodes.vector_search import mmr_select, reciprocal_rank_fusion


def doc(doc_id, source="sermon", embedding=(1.0, 0.0)):
    return {"id": doc_id, "source": source, "title": doc_id, "text": doc_id, "url": None,
            "embedding": np.array(embedding, dtype=np.float32), "score": 0.0}


# ---------------- Reciprocal Rank Fusion ----------------
def test_rrf_rewards_documents_found_by_several_searches():
    dense = [doc("a"), doc("b"), doc("c")]
    keyword = [doc("c"), doc("d")]
    fused = reciprocal_rank_fusion([dense, keyword], k=60)

    assert fused[0]["id"] == "c"                      # in both lists -> first
    assert {d["id"] for d in fused} == {"a", "b", "c", "d"}
    assert fused[0]["rrf_score"] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_keeps_same_id_from_different_sources_apart():
    fused = reciprocal_rank_fusion([[doc("1", "sermon")], [doc("1", "westminster")]])
    assert len(fused) == 2


# ---------------- Maximal Marginal Relevance ----------------
def test_mmr_avoids_near_duplicates():
    query = [1.0, 0.0]
    candidates = [
        doc("best", embedding=(1.0, 0.05)),
        doc("duplicate", embedding=(1.0, 0.06)),      # almost the same as "best"
        doc("different", embedding=(0.9, -0.44)),     # a bit less relevant, but adds something new
    ]
    selected = mmr_select(candidates, query, top_k=2, lambda_=0.3)
    assert [d["id"] for d in selected] == ["best", "different"]


def test_mmr_returns_at_most_top_k():
    candidates = [doc(str(i), embedding=(1.0, i / 10)) for i in range(5)]
    assert len(mmr_select(candidates, [1.0, 0.0], top_k=3)) == 3
    assert mmr_select([], [1.0, 0.0], top_k=3) == []


# ---------------- Bible reference extraction ----------------
def test_extract_english_reference():
    assert extract_verse_ids("What does Romans 3:28 say?") == ["Rom.3.28"]


def test_extract_range_and_indonesian_reference():
    assert extract_verse_ids("Efesus 2:8-9 dan Yohanes 3:16") == ["Eph.2.8", "Eph.2.9", "John.3.16"]


def test_extract_nothing_from_plain_text():
    assert extract_verse_ids("What is grace?") == []


# ---------------- Graph scoring ----------------
def test_seeds_keep_explicit_references_first():
    seeds = choose_seeds(["Rom.3.28"], [("John.3.16", 0.9), ("Rom.3.28", 0.8), ("Eph.2.8", 0.7)], seed_limit=2)
    assert list(seeds) == ["Rom.3.28", "John.3.16"]
    assert seeds["Rom.3.28"]["score"] == 1.0


def test_neighbour_connected_to_many_seeds_scores_higher():
    seeds = {"A": {"score": 1.0}, "B": {"score": 0.8}}
    links = [("A", "hub"), ("B", "hub"), ("A", "leaf"), ("A", "B")]
    neighbours = score_neighbours(seeds, links, decay=0.5)

    assert "B" not in neighbours                       # seeds are not neighbours
    assert neighbours["hub"]["score"] == pytest.approx(0.9)
    assert neighbours["leaf"]["score"] == pytest.approx(0.5)
    assert neighbours["hub"]["relation"] == "cross-reference of A, B"


def test_select_verses_fills_with_semantic_matches_when_few_neighbours():
    seeds = {"A": {"score": 1.0, "relation": "semantic match"}}
    neighbours = {"N": {"score": 0.5, "relation": "cross-reference of A"}}
    selected = select_verses(seeds, neighbours, [("A", 1.0), ("S", 0.6)], top_k=3)
    assert list(selected) == ["A", "N", "S"]


# ---------------- Context building ----------------
def test_context_labels_vector_and_graph_results():
    vector = [{"id": "WSC.1", "source": "westminster", "title": "WSC.1", "text": "Man's chief end...", "url": None}]
    graph = [{"id": "Rom.3.28", "source": "bible", "reference": "Romans 3:28", "pericope": None,
              "text": "For we maintain...", "text_tb": None, "relation": "referenced in the query",
              "linked_sources": [{"source": "westminster", "id": "WCF.11.1", "title": "WCF 11.1"}]}]
    context, sources = build_context_and_sources(vector, graph)

    assert "[V1] Westminster Standards" in context
    assert "[G1] Bible — Romans 3:28" in context
    assert "Cited by: WCF 11.1" in context
    assert [s["label"] for s in sources] == ["V1", "G1"]


def test_shorten_cuts_long_text():
    assert shorten("word " * 1000, limit=20).endswith(" ...")


# ---------------- Retrieval tool ----------------
def test_tool_accepts_json_string_and_calls_pipeline(monkeypatch):
    calls = {}

    def fake_retrieve(query, vector_top_k, graph_top_k):
        calls.update(query=query, vector_top_k=vector_top_k, graph_top_k=graph_top_k)
        return {"query": query, "context": "ctx", "sources": [], "errors": []}

    monkeypatch.setattr(retrieval_tool, "retrieve", fake_retrieve)
    result = retrieval_tool.execute_retrieval_tool(json.dumps({"query": "grace", "vector_top_k": 3, "graph_top_k": 0}))

    assert result["context"] == "ctx"
    assert calls == {"query": "grace", "vector_top_k": 3, "graph_top_k": 0}


def test_tool_uses_defaults_for_missing_top_k(monkeypatch):
    monkeypatch.setattr(retrieval_tool, "retrieve", lambda q, v, g: {"query": q, "context": f"{v},{g}", "sources": [], "errors": []})
    assert retrieval_tool.execute_retrieval_tool({"query": "grace"})["context"] == "5,5"


@pytest.mark.parametrize("arguments", [
    "not json",
    {"query": ""},
    {"query": "grace", "vector_top_k": 99},
    {"query": "grace", "vector_top_k": 0, "graph_top_k": 0},
])
def test_tool_returns_error_instead_of_raising(arguments):
    assert "error" in retrieval_tool.execute_retrieval_tool(arguments)


def test_tool_schema_matches_limits():
    properties = retrieval_tool.RETRIEVAL_TOOL_SCHEMA["function"]["parameters"]["properties"]
    assert set(properties) == {"query", "vector_top_k", "graph_top_k"}
    assert properties["vector_top_k"]["maximum"] == 20


# ---------------- Full graph wiring (databases replaced by fakes) ----------------
def test_pipeline_runs_both_branches_and_merges_errors(monkeypatch):
    import importlib

    from backend.pipelines.rag import rag_pipeline

    # import the node *modules* (the package itself exports the node functions of the same name)
    understand_query = importlib.import_module("backend.pipelines.rag.nodes.understand_query")
    vector_search = importlib.import_module("backend.pipelines.rag.nodes.vector_search")
    graph_search = importlib.import_module("backend.pipelines.rag.nodes.graph_search")

    class FakeEmbedder:
        def embed_query(self, text):
            return [1.0, 0.0]

    monkeypatch.setattr(understand_query, "get_embedding_service", lambda: FakeEmbedder())
    monkeypatch.setattr(vector_search, "search_vector_database", lambda q, v, k, rerank: (
        [{"id": "WSC.1", "source": "westminster", "title": "WSC.1", "text": "chief end", "url": None}],
        ["vector search on 'nash_sermons' failed: down"],
    ))
    monkeypatch.setattr(graph_search, "search_knowledge_graph", lambda v, ids, k: (
        [], [f"graph search on Neo4j failed: seeds were {ids}"],
    ))

    result = rag_pipeline.retrieve("Romans 3:28", vector_top_k=2, graph_top_k=2)

    assert "[V1] Westminster Standards — WSC.1" in result["context"]
    assert len(result["errors"]) == 2                  # errors from both parallel branches are kept
    assert "['Rom.3.28']" in result["errors"][1] or "['Rom.3.28']" in result["errors"][0]


def test_pipeline_rejects_bad_top_k():
    from backend.pipelines.rag import rag_pipeline

    with pytest.raises(ValueError):
        rag_pipeline.retrieve("grace", vector_top_k=0, graph_top_k=0)
    with pytest.raises(ValueError):
        rag_pipeline.retrieve("grace", vector_top_k=21)


def test_pgvector_vector_objects_are_converted_to_numpy():
    from pgvector import Vector
    from backend.pipelines.rag.nodes.vector_search import to_numpy

    array = to_numpy(Vector([0.6, 0.8]))
    assert isinstance(array, np.ndarray)
    assert mmr_select([doc("a", embedding=array)], [1.0, 0.0], top_k=1)[0]["id"] == "a"



# ---------------- LLM reranker ----------------
class FakeLogprob:
    def __init__(self, token, probability):
        self.token = token
        self.logprob = float(np.log(probability))


def test_expected_relevance_uses_digit_probabilities():
    from backend.services.rerank_service import expected_relevance

    assert expected_relevance([FakeLogprob("3", 1.0)]) == pytest.approx(1.0)
    assert expected_relevance([FakeLogprob("0", 1.0)]) == pytest.approx(0.0)
    # 50% "3" and 50% "1" -> expected rating 2 -> 2/3
    assert expected_relevance([FakeLogprob("3", 0.5), FakeLogprob("1", 0.5), FakeLogprob("Rating", 0.2)]) == pytest.approx(2 / 3)


def test_expected_relevance_fails_without_digits():
    from backend.core.exceptions.RerankExceptions import RerankServiceError
    from backend.services.rerank_service import expected_relevance

    with pytest.raises(RerankServiceError):
        expected_relevance([FakeLogprob("Hello", 1.0)])


class FakeReranker:
    """Scores a document by how many times it contains the word 'grace'."""
    model = "llm"

    def rerank(self, query, documents, top_n=None):
        ranking = sorted(
            ({"index": i, "relevance_score": min(1.0, d.count("grace") / 3)} for i, d in enumerate(documents)),
            key=lambda item: item["relevance_score"], reverse=True,
        )
        return ranking[:top_n] if top_n else ranking


class BrokenReranker:
    def rerank(self, query, documents, top_n=None):
        raise RuntimeError("LLM server unreachable")


def fused_candidates():
    plain = doc("plain", embedding=(1.0, 0.0))                 # closest embedding, but no 'grace'
    plain["text"] = "nothing relevant"
    graceful = doc("graceful", embedding=(0.0, 1.0))           # far embedding, very relevant text
    graceful["text"] = "grace grace grace"
    return [[plain, graceful]]


def test_llm_rerank_changes_the_final_order(monkeypatch):
    import importlib
    vector_search = importlib.import_module("backend.pipelines.rag.nodes.vector_search")

    monkeypatch.setattr(vector_search, "hybrid_search", lambda q, v, limit: (fused_candidates(), []))
    monkeypatch.setattr(vector_search, "get_rerank_service", lambda: FakeReranker())

    results, errors = vector_search.search_vector_database("grace", [1.0, 0.0], top_k=1, rerank=True)
    assert [r["id"] for r in results] == ["graceful"]
    assert results[0]["rerank_score"] == 1.0
    assert errors == []

    results, _ = vector_search.search_vector_database("grace", [1.0, 0.0], top_k=1, rerank=False)
    assert [r["id"] for r in results] == ["plain"]
    assert results[0]["rerank_score"] is None


def test_reranker_failure_falls_back_to_similarity(monkeypatch):
    import importlib
    vector_search = importlib.import_module("backend.pipelines.rag.nodes.vector_search")

    monkeypatch.setattr(vector_search, "hybrid_search", lambda q, v, limit: (fused_candidates(), []))
    monkeypatch.setattr(vector_search, "get_rerank_service", lambda: BrokenReranker())

    results, errors = vector_search.search_vector_database("grace", [1.0, 0.0], top_k=1, rerank=True)
    assert [r["id"] for r in results] == ["plain"]
    assert "LLM reranking skipped" in errors[0]


def test_rerank_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from backend.pipelines.rag import rag_pipeline

    monkeypatch.setattr(rag_pipeline, "get_rerank_service", lambda: FakeReranker())
    client = TestClient(app)

    response = client.post("/api/rag/rerank", json={"query": "grace", "documents": ["law", "grace grace grace", "grace"], "top_n": 2})
    assert response.status_code == 200
    body = response.json()
    assert [r["index"] for r in body["results"]] == [1, 2]
    assert body["results"][0]["document"] == "grace grace grace"

    assert client.post("/api/rag/rerank", json={"query": "grace", "documents": []}).status_code == 422
    assert client.post("/api/rag/rerank", json={"query": "grace", "documents": ["a"], "top_n": 0}).status_code == 422
