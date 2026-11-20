"""
Node 2b — Graph RAG over the knowledge graph (Neo4j).

The Bible graph has one node per verse and a CROSS_REFERENCE_TO edge between verses
that talk about the same thing. Plain vector search only finds verses that *sound like*
the query; the graph also finds verses that are *connected to* the relevant ones.

  1. Seeds      — the starting verses:
                    * verses written in the query ("Romans 3:28")   -> always kept, score 1.0
                    * the verses closest to the query embedding      -> score = similarity
  2. Expansion  — follow one hop of CROSS_REFERENCE_TO from every seed.
                  neighbour score = HOP_DECAY * (sum of the scores of the seeds it touches)
                  so a verse connected to several seeds (a "hub") ranks higher.
  3. Selection  — about half of the slots go to seeds, the rest to the best neighbours.
  4. Bridge     — for every chosen verse, list the Westminster units (proof texts) and
                  sermon chunks (verse mentions) in Postgres that cite it. This links the
                  graph results back to the documents of the vector database.
"""
from backend.core.decorators.log_node_call import log_node
from backend.database.neo4j.get_connection import get_neo4j_driver
from backend.database.postgres.get_connection import PGConnect
from backend.pipelines.rag.states import RAGState
from backend.utils.rag_configs import (
    FIND_CROSS_REFERENCES,
    FIND_VERSES_BY_ID,
    FIND_VERSES_BY_VECTOR,
    GET_VERSE_DETAILS,
    HOP_DECAY,
    LINKED_SOURCES_PER_VERSE,
    SERMON_DB,
    SERMONS_CITING_VERSES,
    WESTMINSTER_CITING_VERSES,
    WESTMINSTER_DB,
)


# ---------------------------------------------------------
# Step 1: seeds
# ---------------------------------------------------------
def find_explicit_seeds(driver, verse_ids: list[str]) -> list[str]:
    """Keep only the referenced verse ids that really exist in the graph (in query order)."""
    if not verse_ids:
        return []
    records, _, _ = driver.execute_query(FIND_VERSES_BY_ID, ids=verse_ids)
    existing = {record["id"] for record in records}
    return [verse_id for verse_id in verse_ids if verse_id in existing]


def find_semantic_matches(driver, query_vector: list[float], limit: int) -> list[tuple[str, float]]:
    """Ask the Neo4j vector index for the verses closest to the query: [(verse_id, score), ...]."""
    records, _, _ = driver.execute_query(FIND_VERSES_BY_VECTOR, limit=limit, vector=query_vector)
    return [(record["id"], record["score"]) for record in records]


def choose_seeds(explicit_ids: list[str], semantic_matches: list[tuple[str, float]], seed_limit: int) -> dict[str, dict]:
    """
    Build the seed dictionary {verse_id: {"score": ..., "relation": ...}}.

    Explicit references are always seeds. Semantic matches fill the remaining seed slots.
    """
    seeds = {verse_id: {"score": 1.0, "relation": "referenced in the query"} for verse_id in explicit_ids}

    for verse_id, score in semantic_matches:
        if len(seeds) >= seed_limit:
            break
        if verse_id not in seeds:
            seeds[verse_id] = {"score": score, "relation": "semantic match"}

    return seeds


# ---------------------------------------------------------
# Step 2: expansion
# ---------------------------------------------------------
def score_neighbours(seeds: dict[str, dict], links: list[tuple[str, str]], decay: float = HOP_DECAY) -> dict[str, dict]:
    """
    Score every neighbour reached from the seeds.

    `links` is a list of (seed_id, neighbour_id) pairs. A neighbour's score is
    decay * (sum of the scores of all seeds pointing to it).
    """
    neighbours = {}
    for seed_id, neighbour_id in links:
        if neighbour_id in seeds:
            continue
        entry = neighbours.setdefault(neighbour_id, {"score": 0.0, "via": []})
        entry["score"] += decay * seeds[seed_id]["score"]
        if seed_id not in entry["via"]:
            entry["via"].append(seed_id)

    for entry in neighbours.values():
        entry["relation"] = "cross-reference of " + ", ".join(entry["via"])

    return neighbours


def expand_seeds(driver, seeds: dict[str, dict]) -> dict[str, dict]:
    """Follow one hop of CROSS_REFERENCE_TO from all seeds and score the neighbours."""
    records, _, _ = driver.execute_query(FIND_CROSS_REFERENCES, seed_ids=list(seeds))
    links = [(record["seed_id"], record["neighbour_id"]) for record in records]
    return score_neighbours(seeds, links)


# ---------------------------------------------------------
# Step 3: selection
# ---------------------------------------------------------
def select_verses(seeds: dict, neighbours: dict, semantic_matches: list[tuple[str, float]], top_k: int) -> dict[str, dict]:
    """
    Pick the final `top_k` verses: all seeds first, then the best neighbours.
    If there are not enough neighbours, fill up with more semantic matches.
    """
    selected = dict(list(seeds.items())[:top_k])

    best_neighbours = sorted(neighbours.items(), key=lambda item: (-item[1]["score"], item[0]))
    for verse_id, info in best_neighbours:
        if len(selected) >= top_k:
            break
        selected[verse_id] = info

    for verse_id, score in semantic_matches:
        if len(selected) >= top_k:
            break
        if verse_id not in selected:
            selected[verse_id] = {"score": score, "relation": "semantic match"}

    return selected


def get_verse_details(driver, verse_ids: list[str]) -> dict[str, dict]:
    records, _, _ = driver.execute_query(GET_VERSE_DETAILS, ids=verse_ids)
    return {record["id"]: dict(record) for record in records}


# ---------------------------------------------------------
# Step 4: bridge back to Postgres
# ---------------------------------------------------------
def find_citing_documents(verse_ids: list[str]) -> tuple[dict[str, list[dict]], list[str]]:
    """
    Return {verse_id: [{"source", "id", "title"}, ...]} listing the Westminster units and
    sermon chunks that cite each verse, plus a list of error messages.
    """
    citing = {verse_id: [] for verse_id in verse_ids}
    errors = []
    lookups = (("westminster", WESTMINSTER_DB, WESTMINSTER_CITING_VERSES), ("sermon", SERMON_DB, SERMONS_CITING_VERSES))

    for source, database, sql in lookups:
        try:
            with PGConnect(dbname=database) as postgres:
                rows = postgres.get_db_connection().execute(sql, {"verse_ids": verse_ids}).fetchall()
        except Exception as error:
            errors.append(f"graph bridge to '{database}' failed: {error}")
            continue

        for verse_id, document_id, title in rows:
            same_source = [item for item in citing[verse_id] if item["source"] == source]
            if len(same_source) < LINKED_SOURCES_PER_VERSE:
                citing[verse_id].append({"source": source, "id": document_id, "title": title})

    return citing, errors


# ---------------------------------------------------------
# The LangGraph node
# ---------------------------------------------------------
def search_knowledge_graph(query_vector: list[float], explicit_verse_ids: list[str], top_k: int) -> tuple[list[dict], list[str]]:
    """The full graph-RAG algorithm: seeds -> 1-hop expansion -> selection -> bridge."""
    if top_k <= 0:
        return [], []

    try:
        driver = get_neo4j_driver()
        explicit_ids = find_explicit_seeds(driver, explicit_verse_ids)
        semantic_matches = find_semantic_matches(driver, query_vector, limit=top_k + len(explicit_ids))

        seeds = choose_seeds(explicit_ids, semantic_matches, seed_limit=max(1, top_k // 2))
        neighbours = expand_seeds(driver, seeds) if seeds else {}
        selected = select_verses(seeds, neighbours, semantic_matches, top_k)
        details = get_verse_details(driver, list(selected))
    except Exception as error:
        return [], [f"graph search on Neo4j failed: {error}"]

    citing, errors = find_citing_documents(list(selected))

    results = []
    for verse_id, info in selected.items():
        verse = details.get(verse_id, {})
        results.append({
            "id": verse_id,
            "source": "bible",
            "reference": verse.get("reference", verse_id),
            "pericope": verse.get("pericope"),
            "text": verse.get("text_bsb", ""),
            "text_tb": verse.get("text_tb"),
            "score": round(info["score"], 4),
            "relation": info["relation"],
            "linked_sources": citing.get(verse_id, []),
        })

    return results, errors


@log_node("rag.graph_search")
def graph_search(state: RAGState) -> dict:
    results, errors = search_knowledge_graph(state["query_vector"], state.get("explicit_verse_ids", []), state["graph_top_k"])
    return {"graph_results": results, "errors": errors}
