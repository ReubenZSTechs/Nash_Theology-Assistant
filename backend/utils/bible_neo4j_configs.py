import re

RELATION_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

SCHEMA = (
    "CREATE CONSTRAINT verse_id IF NOT EXISTS FOR (v:Verse) REQUIRE v.id IS UNIQUE",
    "CREATE FULLTEXT INDEX verse_text IF NOT EXISTS FOR (v:Verse) ON EACH [v.text_bsb, v.text_tb, v.reference, v.pericope]",
)

VECTOR_INDEX = (
    "CREATE VECTOR INDEX verse_embedding IF NOT EXISTS FOR (v:Verse) ON (v.embedding) WITH [v.testament, v.book] "
    "OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}"
)

UPSERT_VERSES = """
UNWIND $rows AS row
MERGE (v:Verse {id: row.id})
WITH v, row, coalesce(v.embed_text <> row.embed_text, false) AS changed
SET v += row
FOREACH (_ IN CASE WHEN changed THEN [1] ELSE [] END | REMOVE v.embedding)
"""

DELETE_LINKS = "MATCH ()-[r:{relation}]->() CALL (r) {{ DELETE r }} IN TRANSACTIONS OF 10000 ROWS"

CREATE_LINKS = """
UNWIND $rows AS row
MATCH (h:Verse {{id: row.head}})
MATCH (t:Verse {{id: row.tail}})
CREATE (h)-[r:{relation}]->(t)
SET r.via_range = row.via_range
"""

RESET = "MATCH (v:Verse) CALL (v) { DETACH DELETE v } IN TRANSACTIONS OF 5000 ROWS"

PENDING_EMBEDDINGS = """
MATCH (v:Verse) WHERE v.embedding IS NULL
RETURN v.id AS id, v.embed_text AS text
ORDER BY v.canonical_order
"""

SET_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (v:Verse {id: row.id})
CALL db.create.setNodeVectorProperty(v, 'embedding', row.vector)
"""

STATS = """
MATCH (v:Verse)
RETURN count(v) AS verses,
       count(v.embedding) AS embedded,
       count(v.text_tb) AS with_tb,
       COUNT { ()-[:CROSS_REFERENCE_TO]->() } AS links
"""