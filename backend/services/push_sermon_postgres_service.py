from dotenv import load_dotenv
load_dotenv()

import argparse
import os

from tqdm import tqdm

from backend.utils.common_configs import SERMON_METADATA, VerseIndex, batched, read_jsonl
from backend.database.postgres.get_connection import PGConnect
from backend.utils.sermon_postgres_configs import UPSERT_VIDEO, SCHEMA
from backend.utils.sermon_chunk_configs import ChunkSettings, SemanticChunker
from backend.services.chunk_sermon_service import ingest_transcripts, refresh_mentions, video_row
from backend.services.embedding_service import EmbeddingService

VIDEO_BATCH = 500


def progress(total, description, unit):
    return tqdm(total=total, desc=description, unit=unit, unit_scale=total >= 1000, dynamic_ncols=True)


def sync_videos(connection):
    if not SERMON_METADATA.exists():
        print(f"[sermons] {SERMON_METADATA} not found, skipping video sync")
        return
    rows = [video_row(record) for record in read_jsonl(SERMON_METADATA)]
    with progress(len(rows), "[sermons] videos", "video") as bar:
        for batch in batched(rows, VIDEO_BATCH):
            with connection.cursor() as cursor:
                cursor.executemany(UPSERT_VIDEO, batch)
            bar.update(len(batch))
    connection.commit()


def embed_chunks(connection, embedder, batch_size, limit):
    query = """
        SELECT c.id, coalesce(v.title, '') || E'\n' || c.retrieval_text
        FROM chunks c JOIN videos v USING (video_id)
        WHERE c.embedding IS NULL AND c.retrieval_text IS NOT NULL
        ORDER BY c.video_id, c.chunk_index
    """
    pending = connection.execute(query + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    if not pending:
        print("[embed] all chunks already embedded")
        return
    with progress(len(pending), "[embed] chunks", "chunk") as bar:
        for batch in batched(pending, batch_size):
            vectors = embedder.encode(text for _, text in batch)
            with connection.cursor() as cursor:
                cursor.executemany(
                    "UPDATE chunks SET embedding = %s WHERE id = %s",
                    [(vector, chunk_id) for (chunk_id, _), vector in zip(batch, vectors)],
                )
            connection.commit()
            bar.update(len(batch))


def print_stats(connection):
    row = connection.execute(
        """
        SELECT (SELECT count(*) FROM videos),
               (SELECT count(*) FROM videos WHERE transcribed_at IS NOT NULL),
               count(*), count(embedding),
               count(*) FILTER (WHERE translation_status = 'pending'),
               (SELECT count(*) FROM verse_mentions)
        FROM chunks
        """
    ).fetchone()
    print(
        f"[sermons] videos={row[0]} transcribed={row[1]} chunks={row[2]} embedded={row[3]} "
        f"pending_translation={row[4]} verse_mentions={row[5]}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Sermon store: schema, transcript chunks, verse mentions, embeddings")
    parser.add_argument("command", choices=("videos", "chunks", "mentions", "embed", "all", "stats"))
    parser.add_argument("--rechunk", action="store_true", help="re-chunk transcripts that are already loaded (default chunk settings)")
    parser.add_argument("--allow-retranslate", action="store_true", help="re-chunk even if reviewed translations would be reset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="embed at most N chunks (0 = all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embedder = None

    with PGConnect(dbname=os.getenv("SERMON_DB", "nash_sermons")) as database:
        connection = database.get_db_connection()
        connection.execute(SCHEMA)
        connection.commit()

        if args.command in ("chunks", "embed", "all"):
            embedder = EmbeddingService()

        if args.command in ("videos", "all"):
            sync_videos(connection)

        if args.command in ("chunks", "all"):
            chunker = SemanticChunker(embedder, ChunkSettings())
            ingest_transcripts(connection, chunker, VerseIndex.from_file(), args.rechunk, allow_retranslate=args.allow_retranslate)

        if args.command == "mentions":
            refresh_mentions(connection, VerseIndex.from_file())

        if args.command in ("embed", "all"):
            embed_chunks(connection, embedder, args.batch_size, args.limit)

        print_stats(connection)