from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
from collections import Counter

from tqdm import tqdm

from backend.utils.common_configs import WESTMINSTER_DIR, VerseIndex, batched
from backend.utils.westminster_postgres_configs import DOCUMENTS, UPSERT_UNIT, INSERT_PROOF, SCHEMA
from backend.services.embedding_service import EmbeddingService
from backend.database.postgres.get_connection import PGConnect


def progress(total, description, unit):
    return tqdm(total=total, desc=description, unit=unit, unit_scale=total >= 1000, dynamic_ncols=True)


def load_document(code, filename):
    payload = json.loads((WESTMINSTER_DIR / filename).read_text(encoding="utf-8"))
    title = payload["Metadata"]["Title"]
    if code == "WCF":
        return title, list(confession_units(payload["Data"], title))
    return title, [catechism_unit(code, title, item) for item in payload["Data"]]


def confession_units(chapters, title):
    for chapter in chapters:
        number = int(chapter["Chapter"])
        for section in chapter["Sections"]:
            section_number = int(section["Section"])
            content = section["Content"]
            yield {
                "id": f"WCF.{number}.{section_number}",
                "document": "WCF",
                "document_title": title,
                "chapter": number,
                "chapter_title": chapter["Title"],
                "section": section_number,
                "question_number": None,
                "question": None,
                "content": content,
                "content_with_proofs": section.get("ContentWithProofs"),
                "embed_text": f"{title} {number}.{section_number} — {chapter['Title']}\n{content}",
                "proofs": section.get("Proofs") or [],
            }


def catechism_unit(code, title, item):
    number = int(item["Number"])
    return {
        "id": f"{code}.{number}",
        "document": code,
        "document_title": title,
        "chapter": None,
        "chapter_title": None,
        "section": None,
        "question_number": number,
        "question": item["Question"],
        "content": item["Answer"],
        "content_with_proofs": item.get("AnswerWithProofs"),
        "embed_text": f"{title} Q{number}. {item['Question']}\nA. {item['Answer']}",
        "proofs": item.get("Proofs") or [],
    }


def proof_references(proof):
    for entry in proof["References"]:
        for reference in entry.split(","):
            if reference.strip():
                yield reference.strip()


def proof_rows(unit, index, unresolved):
    rows = {}
    for proof in unit["proofs"]:
        for reference in proof_references(proof):
            verses = index.resolve(reference)
            if verses is None:
                unresolved[reference] += 1
                continue
            via_range = reference if len(verses) > 1 else None
            for verse_id in verses:
                rows.setdefault((proof["Id"], verse_id), (unit["id"], proof["Id"], reference, verse_id, via_range))
    return list(rows.values())


def load_units(connection, index):
    unresolved = Counter()
    totals = Counter()
    with connection.cursor() as cursor:
        for code, filename in DOCUMENTS:
            _, units = load_document(code, filename)
            with progress(len(units), f"[westminster] {code}", "unit") as bar:
                for unit in units:
                    cursor.execute(UPSERT_UNIT, unit)
                    cursor.execute("DELETE FROM proofs WHERE unit_id = %s", (unit["id"],))
                    rows = proof_rows(unit, index, unresolved)
                    cursor.executemany(INSERT_PROOF, rows)
                    totals[code] += 1
                    totals["proof_rows"] += len(rows)
                    bar.update(1)
                    bar.set_postfix(proofs=totals["proof_rows"])
    connection.commit()
    print(f"[westminster] {totals['proof_rows']} proof verse rows")
    if unresolved:
        print(f"[westminster] references not in BSB: {dict(unresolved)}")


def embed_units(connection, batch_size, limit):
    query = "SELECT id, embed_text FROM units WHERE embedding IS NULL ORDER BY document, chapter, section, question_number"
    pending = connection.execute(query + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    if not pending:
        print("[embed] all units already embedded")
        return
    embedder = EmbeddingService()
    with progress(len(pending), "[embed] units", "unit") as bar:
        for batch in batched(pending, batch_size):
            vectors = embedder.encode(text for _, text in batch)
            with connection.cursor() as cursor:
                cursor.executemany(
                    "UPDATE units SET embedding = %s WHERE id = %s",
                    [(vector, unit_id) for (unit_id, _), vector in zip(batch, vectors)],
                )
            connection.commit()
            bar.update(len(batch))


def print_stats(connection):
    row = connection.execute(
        "SELECT count(*), count(embedding), (SELECT count(*) FROM proofs), (SELECT count(DISTINCT verse_id) FROM proofs) FROM units"
    ).fetchone()
    print(f"[westminster] units={row[0]} embedded={row[1]} proof_rows={row[2]} distinct_verses={row[3]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Load and embed the Westminster Standards into Postgres")
    parser.add_argument("command", choices=("load", "embed", "all", "stats"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="embed at most N units (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()
    with PGConnect(dbname=os.getenv("WESTMINSTER_DB", "nash-westminster")) as database:
        connection = database.get_db_connection()

        connection.execute(SCHEMA)
        connection.commit()

        if args.command in ("load", "all"):
            load_units(connection, VerseIndex.from_file())

        if args.command in ("embed", "all"):
            embed_units(connection, args.batch_size, args.limit)
            
        print_stats(connection)


if __name__ == "__main__":
    main()