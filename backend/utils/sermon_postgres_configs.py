from dotenv import load_dotenv
load_dotenv()

from backend.utils.common_configs import BOOKS
import os
import re


ACCEPTED = "('native', 'accepted')"
RETRIEVAL_TEXT = f"CASE WHEN translation_status IN {ACCEPTED} THEN text_en ELSE text_original END"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS videos (
    video_id text PRIMARY KEY,
    source text,
    channel_id text,
    title text,
    url text,
    duration_sec int,
    included boolean,
    exclusion_reason text,
    language text,
    language_probability real,
    transcribed_at timestamptz
);
CREATE TABLE IF NOT EXISTS chunks (
    id text PRIMARY KEY,
    video_id text NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    start_sec real NOT NULL,
    end_sec real NOT NULL,
    language text,
    text_original text NOT NULL,
    text_en text,
    translation_status text NOT NULL DEFAULT 'pending'
        CHECK (translation_status IN ('native', 'pending', 'accepted', 'rejected')),
    retrieval_text text GENERATED ALWAYS AS ({RETRIEVAL_TEXT}) STORED,
    embedding vector({os.getenv("EMBED_DIM", 1024)}),
    search tsvector GENERATED ALWAYS AS (to_tsvector('simple', {RETRIEVAL_TEXT})) STORED,
    UNIQUE (video_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS verse_mentions (
    chunk_id text NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    verse_id text NOT NULL,
    mention text NOT NULL,
    via_range text,
    PRIMARY KEY (chunk_id, verse_id, mention)
);
CREATE INDEX IF NOT EXISTS chunks_video_idx ON chunks (video_id);
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin (search);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS verse_mentions_verse_idx ON verse_mentions (verse_id);
CREATE OR REPLACE FUNCTION reset_chunk_embedding() RETURNS trigger AS $$
BEGIN
    IF NEW.text_original IS DISTINCT FROM OLD.text_original
       OR NEW.text_en IS DISTINCT FROM OLD.text_en
       OR NEW.translation_status IS DISTINCT FROM OLD.translation_status THEN
        NEW.embedding := NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE TRIGGER chunks_reset_embedding BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION reset_chunk_embedding();
"""

UPSERT_VIDEO = """
INSERT INTO videos (video_id, source, channel_id, title, url, duration_sec, included, exclusion_reason)
VALUES (%(video_id)s, %(source)s, %(channel_id)s, %(title)s, %(url)s, %(duration_sec)s, %(included)s, %(exclusion_reason)s)
ON CONFLICT (video_id) DO UPDATE SET
    source = EXCLUDED.source,
    channel_id = EXCLUDED.channel_id,
    title = EXCLUDED.title,
    url = EXCLUDED.url,
    duration_sec = EXCLUDED.duration_sec,
    included = EXCLUDED.included,
    exclusion_reason = EXCLUDED.exclusion_reason
"""

MARK_TRANSCRIBED = """
UPDATE videos SET language = %s, language_probability = %s, transcribed_at = now() WHERE video_id = %s
"""

UPSERT_CHUNK = """
INSERT INTO chunks (id, video_id, chunk_index, start_sec, end_sec, language, text_original, text_en, translation_status)
VALUES (%(id)s, %(video_id)s, %(chunk_index)s, %(start_sec)s, %(end_sec)s, %(language)s,
        %(text_original)s, %(text_en)s, %(translation_status)s)
ON CONFLICT (id) DO UPDATE SET
    start_sec = EXCLUDED.start_sec,
    end_sec = EXCLUDED.end_sec,
    language = EXCLUDED.language,
    text_en = CASE WHEN chunks.text_original = EXCLUDED.text_original THEN chunks.text_en ELSE EXCLUDED.text_en END,
    translation_status = CASE WHEN chunks.text_original = EXCLUDED.text_original
                              THEN chunks.translation_status ELSE EXCLUDED.translation_status END,
    text_original = EXCLUDED.text_original
"""

INSERT_MENTION = """
INSERT INTO verse_mentions (chunk_id, verse_id, mention, via_range) VALUES (%s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""

BOOK_ALIASES = {
    "psalm": "Ps",
    "song of songs": "Song",
    "kisah rasul": "Acts",
    "kisah rasul-rasul": "Acts",
    "revelations": "Rev",
}
MAX_MENTION_SPAN = 50


def book_lookup():
    lookup = dict(BOOK_ALIASES)
    for book in BOOKS:
        lookup[book.name.lower()] = book.osis
        lookup[book.name_id.lower()] = book.osis
    return lookup


BOOK_LOOKUP = book_lookup()
BOOK_PATTERN = "|".join(
    re.escape(name).replace(r"\ ", r"\s+") for name in sorted(BOOK_LOOKUP, key=len, reverse=True)
)
MENTION = re.compile(
    rf"(?<![\w-])(?P<book>{BOOK_PATTERN})\s+(?:pasal\s+|chapter\s+)?(?P<chapter>\d{{1,3}})"
    r"\s*(?::|\.|,?\s+ayat\s+|,?\s+verses?\s+)\s*(?P<start>\d{1,3})"
    r"(?:\s*(?:-|–|sampai(?:\s+dengan)?|s/d|to|through)\s*(?P<end>\d{1,3}))?",
    re.IGNORECASE,
)