from dotenv import load_dotenv
load_dotenv()

import argparse
from collections import defaultdict
from itertools import islice
import os

from neo4j import GraphDatabase
from tqdm import tqdm

from backend.utils.common_configs import TB_OVERLAY_PATH, TRIPLES_PATH, VERSES_PATH, batched, read_jsonl
from backend.utils.bible_neo4j_configs import SCHEMA, VECTOR_INDEX, UPSERT_VERSES, RELATION_NAME, CREATE_LINKS, DELETE_LINKS, PENDING_EMBEDDINGS, SET_EMBEDDINGS, RESET, STATS
from backend.services.embedding_service import EmbeddingService


def embed_text(verse):
    heading = verse["pericope"] or ""
    if verse["section"] and verse["section"] != heading:
        heading = f"{verse['section']} — {heading}"
    prefix = f"{verse['reference']} ({heading})" if heading else verse["reference"]
    return f"{prefix}: {verse['text_bsb']}"


def load_tb_overlay():
    if not TB_OVERLAY_PATH.exists():
        print("[bible] TB overlay not found, loading BSB only")
        return {}
    return {row["id"]: row["text_tb"] for row in read_jsonl(TB_OVERLAY_PATH) if row["in_bsb"]}


def verse_rows():
    overlay = load_tb_overlay()
    for verse in read_jsonl(VERSES_PATH):
        row = {**verse, "embed_text": embed_text(verse)}
        if verse["id"] in overlay:
            row["text_tb"] = overlay[verse["id"]]
        yield row


def progress(total, description, unit):
    return tqdm(total=total, desc=description, unit=unit, unit_scale=True, dynamic_ncols=True)


def ensure_schema(driver):
    for statement in SCHEMA:
        driver.execute_query(statement)
    driver.execute_query(VECTOR_INDEX.format(dimensions=int(os.getenv("EMBED_DIM", "1024"))))


def load_verses(driver, batch_size):
    rows = list(verse_rows())
    with progress(len(rows), "[bible] verses", "verse") as bar:
        for batch in batched(rows, batch_size):
            driver.execute_query(UPSERT_VERSES, rows=batch)
            bar.update(len(batch))


def load_links(driver, batch_size):
    grouped = defaultdict(list)
    for row in read_jsonl(TRIPLES_PATH):
        grouped[row["relation"]].append({"head": row["head"], "tail": row["tail"], "via_range": row.get("via_range")})
    for relation, rows in grouped.items():
        if not RELATION_NAME.match(relation):
            raise ValueError(f"Invalid relation name: {relation}")
        print(f"[bible] removing existing {relation} links")
        with driver.session() as session:
            session.run(DELETE_LINKS.format(relation=relation)).consume()
        with progress(len(rows), f"[bible] {relation}", "link") as bar:
            for batch in batched(rows, batch_size):
                driver.execute_query(CREATE_LINKS.format(relation=relation), rows=batch)
                bar.update(len(batch))


def embed_verses(driver, batch_size, limit):
    records, _, _ = driver.execute_query(PENDING_EMBEDDINGS)
    pending = list(islice(records, limit)) if limit else records
    if not pending:
        print("[embed] all verses already embedded")
        return
    embedder = EmbeddingService()
    with progress(len(pending), "[embed] verses", "verse") as bar:
        for batch in batched(pending, batch_size):
            vectors = embedder.encode(record["text"] for record in batch)
            rows = [{"id": record["id"], "vector": vector.tolist()} for record, vector in zip(batch, vectors)]
            driver.execute_query(SET_EMBEDDINGS, rows=rows)
            bar.update(len(rows))


def reset(driver):
    with driver.session() as session:
        session.run(RESET).consume()
    print("[bible] removed all Verse nodes")


def print_stats(driver):
    records, _, _ = driver.execute_query(STATS)
    print(f"[bible] {dict(records[0])}")


def parse_args():
    parser = argparse.ArgumentParser(description="Load the Bible graph into Neo4j and embed verses")
    parser.add_argument("command", choices=("load", "embed", "all", "stats"))
    parser.add_argument("--reset", action="store_true", help="delete all Verse nodes before loading")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="embed at most N verses (0 = all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "nash-password"))
    with GraphDatabase.driver(os.getenv("NEO4J_URI", "bolt://localhost:7687"), auth=auth) as driver:
        driver.verify_connectivity()
        if args.reset:
            reset(driver)
        if args.command in ("load", "all"):
            ensure_schema(driver)
            load_verses(driver, args.batch_size)
            load_links(driver, args.batch_size)
        if args.command in ("embed", "all"):
            ensure_schema(driver)
            embed_verses(driver, args.embed_batch_size, args.limit)
        print_stats(driver)