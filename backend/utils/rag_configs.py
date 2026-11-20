"""
Settings and query strings used by the RAG pipeline (backend/pipelines/rag).

Keeping every SQL / Cypher query and every tunable number in one file makes the
pipeline nodes short and easy to read: the nodes decide *what* to do, this file
says *how* to ask the databases.
"""
from dotenv import load_dotenv
load_dotenv()

import os


# =========================================================
# DATABASES
# =========================================================
SERMON_DB = os.getenv("SERMON_DB", "nash_sermons")
WESTMINSTER_DB = os.getenv("WESTMINSTER_DB", "nash-westminster")


# =========================================================
# TUNABLE PARAMETERS
# =========================================================
DEFAULT_TOP_K = 5            # results per branch when the caller does not choose
MAX_TOP_K = 20               # hard upper limit for vector_top_k and graph_top_k

# Advanced RAG (vector database)
CANDIDATE_MULTIPLIER = 4     # each search fetches top_k * 4 candidates before re-ranking
RRF_K = 60                   # constant of Reciprocal Rank Fusion (60 is the value from the original paper)
MMR_LAMBDA = 0.7             # 1.0 = only relevance, 0.0 = only diversity

# Reranking with the LLM (backend/services/rerank_service.py)
RERANK_ENABLED = os.getenv("RAG_RERANK", "true").lower() in ("1", "true", "yes")
RERANK_CANDIDATES = 20       # how many fused candidates the LLM re-scores (at least 2 * top_k)
MAX_RERANK_DOCUMENTS = 50    # limit for the /rag/rerank endpoint

# Graph RAG (knowledge graph)
HOP_DECAY = 0.5              # a neighbour is worth half of the seed that leads to it
LINKED_SOURCES_PER_VERSE = 3 # how many citing sermons / Westminster units to list per verse

# Context building
MAX_PASSAGE_CHARS = 1200     # long passages are cut so the LLM context stays small


# =========================================================
# VECTOR DATABASE QUERIES (Postgres + pgvector)
# Every query returns the same columns so the results can be fused together:
#   id, source, title, text, url, embedding, score
# =========================================================
SERMON_DENSE_SEARCH = """
SELECT c.id,
       'sermon' AS source,
       coalesce(v.title, c.video_id) AS title,
       c.retrieval_text AS text,
       v.url || '&t=' || floor(c.start_sec)::int || 's' AS url,
       c.embedding,
       1 - (c.embedding <=> %(vector)s) AS score
FROM chunks c JOIN videos v USING (video_id)
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(vector)s
LIMIT %(limit)s
"""

SERMON_KEYWORD_SEARCH = """
SELECT c.id,
       'sermon' AS source,
       coalesce(v.title, c.video_id) AS title,
       c.retrieval_text AS text,
       v.url || '&t=' || floor(c.start_sec)::int || 's' AS url,
       c.embedding,
       ts_rank_cd(c.search, query) AS score
FROM chunks c JOIN videos v USING (video_id),
     websearch_to_tsquery('simple', %(query)s) AS query
WHERE c.search @@ query AND c.embedding IS NOT NULL
ORDER BY score DESC
LIMIT %(limit)s
"""

WESTMINSTER_DENSE_SEARCH = """
SELECT u.id,
       'westminster' AS source,
       u.document_title || ' ' || u.id AS title,
       coalesce('Q. ' || u.question || E'\n' || 'A. ', '') || u.content AS text,
       NULL AS url,
       u.embedding,
       1 - (u.embedding <=> %(vector)s) AS score
FROM units u
WHERE u.embedding IS NOT NULL
ORDER BY u.embedding <=> %(vector)s
LIMIT %(limit)s
"""

WESTMINSTER_KEYWORD_SEARCH = """
SELECT u.id,
       'westminster' AS source,
       u.document_title || ' ' || u.id AS title,
       coalesce('Q. ' || u.question || E'\n' || 'A. ', '') || u.content AS text,
       NULL AS url,
       u.embedding,
       ts_rank_cd(u.search, query) AS score
FROM units u, websearch_to_tsquery('english', %(query)s) AS query
WHERE u.search @@ query AND u.embedding IS NOT NULL
ORDER BY score DESC
LIMIT %(limit)s
"""


# =========================================================
# BRIDGE QUERIES: from Bible verses back to the documents that cite them
# =========================================================
WESTMINSTER_CITING_VERSES = """
SELECT DISTINCT p.verse_id, u.id, u.document_title || ' ' || u.id AS title
FROM proofs p JOIN units u ON u.id = p.unit_id
WHERE p.verse_id = ANY(%(verse_ids)s)
ORDER BY p.verse_id, u.id
"""

SERMONS_CITING_VERSES = """
SELECT DISTINCT m.verse_id, c.id, coalesce(v.title, c.video_id) AS title
FROM verse_mentions m
JOIN chunks c ON c.id = m.chunk_id
JOIN videos v ON v.video_id = c.video_id
WHERE m.verse_id = ANY(%(verse_ids)s)
ORDER BY m.verse_id, c.id
"""


# =========================================================
# KNOWLEDGE GRAPH QUERIES (Neo4j)
# =========================================================
FIND_VERSES_BY_ID = """
MATCH (v:Verse) WHERE v.id IN $ids
RETURN v.id AS id
"""

# Vector search with the Cypher 25 SEARCH clause (replaces the deprecated
# db.index.vector.queryNodes procedure). "CYPHER 25" makes the query work even on a
# database whose default language is still Cypher 5. `score` is the cosine similarity
# rescaled to 0..1 (higher = closer).
FIND_VERSES_BY_VECTOR = """
CYPHER 25
MATCH (node:Verse)
  SEARCH node IN (
    VECTOR INDEX verse_embedding
    FOR $vector
    LIMIT $limit
  ) SCORE AS score
RETURN node.id AS id, score
"""

FIND_CROSS_REFERENCES = """
UNWIND $seed_ids AS seed_id
MATCH (:Verse {id: seed_id})-[:CROSS_REFERENCE_TO]->(neighbour:Verse)
WHERE NOT neighbour.id IN $seed_ids
RETURN seed_id, neighbour.id AS neighbour_id
"""

GET_VERSE_DETAILS = """
MATCH (v:Verse) WHERE v.id IN $ids
RETURN v.id AS id, v.reference AS reference, v.pericope AS pericope,
       v.text_bsb AS text_bsb, v.text_tb AS text_tb
"""

GRAPH_HEALTH = "MATCH (v:Verse) RETURN count(v) AS verses LIMIT 1"
