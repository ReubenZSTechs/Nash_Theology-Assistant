from dotenv import load_dotenv
load_dotenv()

import os

DOCUMENTS = (
    ("WCF", "wcf.json"),
    ("WLC", "wlc.json"),
    ("WSC", "wsc.json"),
)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS units (
    id text PRIMARY KEY,
    document text NOT NULL,
    document_title text NOT NULL,
    chapter int,
    chapter_title text,
    section int,
    question_number int,
    question text,
    content text NOT NULL,
    content_with_proofs text,
    embed_text text NOT NULL,
    embedding vector({os.getenv("EMBED_DIM", "1024")}),
    search tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(question, '') || ' ' || content)) STORED
);
CREATE TABLE IF NOT EXISTS proofs (
    unit_id text NOT NULL REFERENCES units(id) ON DELETE CASCADE,
    marker int NOT NULL,
    reference text NOT NULL,
    verse_id text NOT NULL,
    via_range text,
    PRIMARY KEY (unit_id, marker, verse_id)
);
CREATE INDEX IF NOT EXISTS units_document_idx ON units (document);
CREATE INDEX IF NOT EXISTS units_search_idx ON units USING gin (search);
CREATE INDEX IF NOT EXISTS units_embedding_idx ON units USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS proofs_verse_idx ON proofs (verse_id);
"""

UPSERT_UNIT = """
INSERT INTO units (id, document, document_title, chapter, chapter_title, section, question_number,
                   question, content, content_with_proofs, embed_text)
VALUES (%(id)s, %(document)s, %(document_title)s, %(chapter)s, %(chapter_title)s, %(section)s,
        %(question_number)s, %(question)s, %(content)s, %(content_with_proofs)s, %(embed_text)s)
ON CONFLICT (id) DO UPDATE SET
    document = EXCLUDED.document,
    document_title = EXCLUDED.document_title,
    chapter = EXCLUDED.chapter,
    chapter_title = EXCLUDED.chapter_title,
    section = EXCLUDED.section,
    question_number = EXCLUDED.question_number,
    question = EXCLUDED.question,
    content = EXCLUDED.content,
    content_with_proofs = EXCLUDED.content_with_proofs,
    embedding = CASE WHEN units.embed_text = EXCLUDED.embed_text THEN units.embedding END,
    embed_text = EXCLUDED.embed_text
"""

INSERT_PROOF = """
INSERT INTO proofs (unit_id, marker, reference, verse_id, via_range)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""