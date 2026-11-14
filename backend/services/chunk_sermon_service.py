from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from backend.utils.common_configs import TRANSCRIPT_DIR, VerseIndex, batched
from backend.database.postgres.get_connection import PGConnect
from backend.utils.sermon_postgres_configs import MENTION, BOOK_LOOKUP, MAX_MENTION_SPAN, UPSERT_CHUNK, UPSERT_VIDEO, INSERT_MENTION, MARK_TRANSCRIBED, SCHEMA
from backend.utils.sermon_chunk_configs import ChunkSettings, SemanticChunker
from backend.services.embedding_service import EmbeddingService

MENTION_BATCH = 500
TRANSLATED_STATUSES = ("accepted", "rejected")


def extract_mentions(text, index):
    for match in MENTION.finditer(text or ""):
        osis = BOOK_LOOKUP[" ".join(match.group("book").lower().split())]
        chapter = int(match.group("chapter"))
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end < start or end - start >= MAX_MENTION_SPAN:
            end = start
        via_range = f"{osis}.{chapter}.{start}-{osis}.{chapter}.{end}" if end > start else None
        for verse in range(start, end + 1):
            verse_id = f"{osis}.{chapter}.{verse}"
            if verse_id in index:
                yield verse_id, match.group(0), via_range


def video_row(record):
    return {
        "video_id": record["video_id"],
        "source": record.get("source"),
        "channel_id": record.get("channel_id"),
        "title": record.get("title"),
        "url": record.get("url"),
        "duration_sec": record.get("duration_sec"),
        "included": record.get("included", True),
        "exclusion_reason": record.get("exclusion_reason"),
    }


def chunk_rows(transcript, chunker):
    video_id = transcript["video_id"]
    native = transcript["language"] == "en"
    for chunk_index, chunk in enumerate(chunker.chunk(transcript["segments"])):
        yield {
            "id": f"{video_id}:{chunk_index:04d}",
            "video_id": video_id,
            "chunk_index": chunk_index,
            "start_sec": round(chunk["start"], 2),
            "end_sec": round(chunk["end"], 2),
            "language": transcript["language"],
            "text_original": chunk["text"],
            "text_en": chunk["text"] if native else None,
            "translation_status": "native" if native else "pending",
        }


def write_mentions(cursor, chunk_ids, index):
    cursor.execute("DELETE FROM verse_mentions WHERE chunk_id = ANY(%s)", (chunk_ids,))
    cursor.execute("SELECT id, text_original, text_en FROM chunks WHERE id = ANY(%s)", (chunk_ids,))
    rows = []
    for chunk_id, original, english in cursor.fetchall():
        for text in dict.fromkeys((original, english)):
            rows.extend((chunk_id, *mention) for mention in extract_mentions(text, index))
    cursor.executemany(INSERT_MENTION, rows)
    return len(rows)


def ingest_transcript(connection, transcript, chunker, index):
    rows = list(chunk_rows(transcript, chunker))
    with connection.cursor() as cursor:
        cursor.execute(UPSERT_VIDEO, video_row(transcript))
        cursor.execute(MARK_TRANSCRIBED, (transcript["language"], transcript.get("language_probability"), transcript["video_id"]))
        cursor.executemany(UPSERT_CHUNK, rows)
        cursor.execute("DELETE FROM chunks WHERE video_id = %s AND chunk_index >= %s", (transcript["video_id"], len(rows)))
        mentions = write_mentions(cursor, [row["id"] for row in rows], index) if rows else 0
    connection.commit()
    return len(rows), mentions


def translated_chunks(connection, video_ids):
    query = "SELECT count(*) FROM chunks WHERE translation_status = ANY(%s)"
    parameters = [list(TRANSLATED_STATUSES)]
    if video_ids is not None:
        query += " AND video_id = ANY(%s)"
        parameters.append(list(video_ids))
    return connection.execute(query, parameters).fetchone()[0]


def select_transcripts(connection, transcript_dir, rechunk, video_ids):
    paths = sorted(transcript_dir.glob("*.json")) if transcript_dir.exists() else []
    if video_ids:
        paths = [path for path in paths if path.stem in video_ids]
    if rechunk:
        return paths
    chunked = {row[0] for row in connection.execute("SELECT DISTINCT video_id FROM chunks").fetchall()}
    return [path for path in paths if path.stem not in chunked]


def ingest_transcripts(connection, chunker, index, rechunk=False, video_ids=None, transcript_dir=TRANSCRIPT_DIR, allow_retranslate=False):
    paths = select_transcripts(connection, transcript_dir, rechunk, video_ids)
    if not paths:
        print(f"[chunks] nothing to chunk in {transcript_dir}")
        return
    if rechunk and not allow_retranslate:
        translated = translated_chunks(connection, [path.stem for path in paths])
        if translated:
            raise SystemExit(
                f"[chunks] {translated} chunks already have a reviewed translation. Re-chunking changes their text and "
                "resets them to pending. Add --allow-retranslate to continue."
            )
    total_chunks = 0
    total_mentions = 0
    with tqdm(total=len(paths), desc="[chunks] transcripts", unit="file", dynamic_ncols=True) as bar:
        for path in paths:
            transcript = json.loads(path.read_text(encoding="utf-8"))
            chunks, mentions = ingest_transcript(connection, transcript, chunker, index)
            total_chunks += chunks
            total_mentions += mentions
            bar.update(1)
            bar.set_postfix(chunks=total_chunks, mentions=total_mentions)
    print(f"[chunks] {len(paths)} transcripts -> {total_chunks} chunks, {total_mentions} verse mentions ({chunker.settings.as_dict()})")


def refresh_mentions(connection, index):
    chunk_ids = [row[0] for row in connection.execute("SELECT id FROM chunks").fetchall()]
    total = 0
    with tqdm(total=len(chunk_ids), desc="[chunks] mentions", unit="chunk", dynamic_ncols=True) as bar:
        with connection.cursor() as cursor:
            for batch in batched(chunk_ids, MENTION_BATCH):
                total += write_mentions(cursor, batch, index)
                bar.update(len(batch))
                bar.set_postfix(mentions=total)
    connection.commit()


def describe(text, width):
    flat = " ".join(text.split())
    if len(flat) <= width * 2:
        return flat
    return f"{flat[:width]} … {flat[-width:]}"


def preview(paths, chunker, width):
    for path in paths:
        transcript = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = chunker.chunk(transcript["segments"])
        durations = [chunk["end"] - chunk["start"] for chunk in chunks]
        print(f"\n== {transcript.get('title') or transcript['video_id']}")
        print(f"   {len(chunks)} chunks, {min(durations, default=0):.0f}-{max(durations, default=0):.0f} s, mean {sum(durations) / max(1, len(durations)):.0f} s")
        for index, chunk in enumerate(chunks):
            overlap = chunk["core_start"] - chunk["start"]
            print(f"   [{index:02d}] {chunk['start']:7.1f}-{chunk['end']:7.1f} ({chunk['end'] - chunk['start']:5.1f} s, overlap {overlap:4.1f} s, {len(chunk['text'].split())} words)")
            print(f"        {describe(chunk['text'], width)}")


def settings_from(args):
    return ChunkSettings(
        min_seconds=args.min_seconds,
        target_seconds=args.target_seconds,
        max_seconds=args.max_seconds,
        unit_seconds=args.unit_seconds,
        window_units=args.window_units,
        overlap_seconds=args.overlap_seconds,
        strong_shift_percentile=args.strong_shift_percentile,
    )


def parse_args():
    defaults = ChunkSettings()
    parser = argparse.ArgumentParser(description="Split sermon transcripts into topic-coherent chunks with bge-m3 and store them in Postgres")
    parser.add_argument("--rechunk", action="store_true", help="re-chunk transcripts that already have chunks")
    parser.add_argument("--video", nargs="+", metavar="VIDEO_ID", help="only these videos")
    parser.add_argument("--allow-retranslate", action="store_true", help="re-chunk even if reviewed translations would be reset")
    parser.add_argument("--preview", nargs="+", metavar="TRANSCRIPT_JSON", help="print the chunks for these transcript files without touching the database")
    parser.add_argument("--preview-width", type=int, default=70, help="characters shown from the start and end of each previewed chunk")
    parser.add_argument("--min-seconds", type=float, default=defaults.min_seconds, help="shortest chunk")
    parser.add_argument("--target-seconds", type=float, default=defaults.target_seconds, help="preferred chunk length")
    parser.add_argument("--max-seconds", type=float, default=defaults.max_seconds, help="longest chunk")
    parser.add_argument("--unit-seconds", type=float, default=defaults.unit_seconds, help="Whisper segments are grouped into units of about this length before comparing topics")
    parser.add_argument("--window-units", type=int, default=defaults.window_units, help="units on each side compared when looking for a topic shift")
    parser.add_argument("--overlap-seconds", type=float, default=defaults.overlap_seconds, help="seconds of the previous chunk repeated at the start of each chunk (0 = none)")
    parser.add_argument("--strong-shift-percentile", type=float, default=defaults.strong_shift_percentile, help="cuts at or above this percentile of topic-shift strength get no overlap")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        chunker = SemanticChunker(EmbeddingService(), settings_from(args))
    except ValueError as error:
        raise SystemExit(f"[chunks] invalid settings: {error}")

    if args.preview:
        preview(args.preview, chunker, args.preview_width)
    else:
        with PGConnect(dbname=os.getenv("SERMON_DB", "nash_sermons")) as database:
            connection = database.get_db_connection()
            connection.execute(SCHEMA)
            connection.commit()
            ingest_transcripts(connection, chunker, VerseIndex.from_file(), args.rechunk, set(args.video) if args.video else None, allow_retranslate=args.allow_retranslate)