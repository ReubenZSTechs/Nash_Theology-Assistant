import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = Path(os.environ.get("NASH_DATA_ROOT", "data"))
PUBLIC_RAW = DATA_ROOT / "public" / "raw"
PRIVATE_RAW = DATA_ROOT / "private" / "raw"
PUBLIC_PROCESSED = DATA_ROOT / "public" / "processed"
PRIVATE_PROCESSED = DATA_ROOT / "private" / "processed"
INTERIM = DATA_ROOT / "interim"

MANIFEST_PATH = DATA_ROOT / "manifest.json"
BSB_USFM_DIR = PUBLIC_RAW / "bible" / "bsb_usfm"
XREF_PATH = PUBLIC_RAW / "crossrefs" / "cross_references.txt"
TB_PDF_PATH = PRIVATE_RAW / "bible" / "tb_1974.pdf"
WESTMINSTER_DIR = PUBLIC_RAW / "westminster"
SERMON_METADATA = PRIVATE_RAW / "sermons" / "metadata.jsonl"
TRANSCRIPT_DIR = INTERIM / "sermons" / "transcripts"

BIBLE_DIR = PUBLIC_PROCESSED / "bible"
VERSES_PATH = BIBLE_DIR / "verses.jsonl"
TRIPLES_PATH = BIBLE_DIR / "triples.jsonl"
GRAPH_META_PATH = BIBLE_DIR / "graph_meta.json"
TB_OVERLAY_PATH = PRIVATE_PROCESSED / "bible" / "verses_tb.jsonl"


@dataclass(frozen=True)
class Book:
    usfm: str
    osis: str
    name: str
    name_id: str
    testament: str


BOOKS = (
    Book("GEN", "Gen", "Genesis", "Kejadian", "OT"),
    Book("EXO", "Exod", "Exodus", "Keluaran", "OT"),
    Book("LEV", "Lev", "Leviticus", "Imamat", "OT"),
    Book("NUM", "Num", "Numbers", "Bilangan", "OT"),
    Book("DEU", "Deut", "Deuteronomy", "Ulangan", "OT"),
    Book("JOS", "Josh", "Joshua", "Yosua", "OT"),
    Book("JDG", "Judg", "Judges", "Hakim-hakim", "OT"),
    Book("RUT", "Ruth", "Ruth", "Rut", "OT"),
    Book("1SA", "1Sam", "1 Samuel", "1 Samuel", "OT"),
    Book("2SA", "2Sam", "2 Samuel", "2 Samuel", "OT"),
    Book("1KI", "1Kgs", "1 Kings", "1 Raja-raja", "OT"),
    Book("2KI", "2Kgs", "2 Kings", "2 Raja-raja", "OT"),
    Book("1CH", "1Chr", "1 Chronicles", "1 Tawarikh", "OT"),
    Book("2CH", "2Chr", "2 Chronicles", "2 Tawarikh", "OT"),
    Book("EZR", "Ezra", "Ezra", "Ezra", "OT"),
    Book("NEH", "Neh", "Nehemiah", "Nehemia", "OT"),
    Book("EST", "Esth", "Esther", "Ester", "OT"),
    Book("JOB", "Job", "Job", "Ayub", "OT"),
    Book("PSA", "Ps", "Psalms", "Mazmur", "OT"),
    Book("PRO", "Prov", "Proverbs", "Amsal", "OT"),
    Book("ECC", "Eccl", "Ecclesiastes", "Pengkhotbah", "OT"),
    Book("SNG", "Song", "Song of Solomon", "Kidung Agung", "OT"),
    Book("ISA", "Isa", "Isaiah", "Yesaya", "OT"),
    Book("JER", "Jer", "Jeremiah", "Yeremia", "OT"),
    Book("LAM", "Lam", "Lamentations", "Ratapan", "OT"),
    Book("EZK", "Ezek", "Ezekiel", "Yehezkiel", "OT"),
    Book("DAN", "Dan", "Daniel", "Daniel", "OT"),
    Book("HOS", "Hos", "Hosea", "Hosea", "OT"),
    Book("JOL", "Joel", "Joel", "Yoel", "OT"),
    Book("AMO", "Amos", "Amos", "Amos", "OT"),
    Book("OBA", "Obad", "Obadiah", "Obaja", "OT"),
    Book("JON", "Jonah", "Jonah", "Yunus", "OT"),
    Book("MIC", "Mic", "Micah", "Mikha", "OT"),
    Book("NAM", "Nah", "Nahum", "Nahum", "OT"),
    Book("HAB", "Hab", "Habakkuk", "Habakuk", "OT"),
    Book("ZEP", "Zeph", "Zephaniah", "Zefanya", "OT"),
    Book("HAG", "Hag", "Haggai", "Hagai", "OT"),
    Book("ZEC", "Zech", "Zechariah", "Zakaria", "OT"),
    Book("MAL", "Mal", "Malachi", "Maleakhi", "OT"),
    Book("MAT", "Matt", "Matthew", "Matius", "NT"),
    Book("MRK", "Mark", "Mark", "Markus", "NT"),
    Book("LUK", "Luke", "Luke", "Lukas", "NT"),
    Book("JHN", "John", "John", "Yohanes", "NT"),
    Book("ACT", "Acts", "Acts", "Kisah Para Rasul", "NT"),
    Book("ROM", "Rom", "Romans", "Roma", "NT"),
    Book("1CO", "1Cor", "1 Corinthians", "1 Korintus", "NT"),
    Book("2CO", "2Cor", "2 Corinthians", "2 Korintus", "NT"),
    Book("GAL", "Gal", "Galatians", "Galatia", "NT"),
    Book("EPH", "Eph", "Ephesians", "Efesus", "NT"),
    Book("PHP", "Phil", "Philippians", "Filipi", "NT"),
    Book("COL", "Col", "Colossians", "Kolose", "NT"),
    Book("1TH", "1Thess", "1 Thessalonians", "1 Tesalonika", "NT"),
    Book("2TH", "2Thess", "2 Thessalonians", "2 Tesalonika", "NT"),
    Book("1TI", "1Tim", "1 Timothy", "1 Timotius", "NT"),
    Book("2TI", "2Tim", "2 Timothy", "2 Timotius", "NT"),
    Book("TIT", "Titus", "Titus", "Titus", "NT"),
    Book("PHM", "Phlm", "Philemon", "Filemon", "NT"),
    Book("HEB", "Heb", "Hebrews", "Ibrani", "NT"),
    Book("JAS", "Jas", "James", "Yakobus", "NT"),
    Book("1PE", "1Pet", "1 Peter", "1 Petrus", "NT"),
    Book("2PE", "2Pet", "2 Peter", "2 Petrus", "NT"),
    Book("1JN", "1John", "1 John", "1 Yohanes", "NT"),
    Book("2JN", "2John", "2 John", "2 Yohanes", "NT"),
    Book("3JN", "3John", "3 John", "3 Yohanes", "NT"),
    Book("JUD", "Jude", "Jude", "Yudas", "NT"),
    Book("REV", "Rev", "Revelation", "Wahyu", "NT"),
)
BOOK_BY_OSIS = {book.osis: book for book in BOOKS}
BOOK_BY_NAME_ID = {book.name_id: book for book in BOOKS}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def batched(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class VerseIndex:
    def __init__(self, verse_ids):
        self.ids = list(verse_ids)
        self.position = {verse_id: index for index, verse_id in enumerate(self.ids)}
        self.chapters = {}
        for index, verse_id in enumerate(self.ids):
            book, chapter, _ = verse_id.split(".")
            key = (book, int(chapter))
            first, _ = self.chapters.get(key, (index, index))
            self.chapters[key] = (first, index)

    @classmethod
    def from_file(cls, path=VERSES_PATH):
        if not Path(path).exists():
            raise SystemExit(f"{path} not found. Run build_graph.py first.")
        return cls(row["id"] for row in read_jsonl(path))

    def __contains__(self, verse_id):
        return verse_id in self.position

    def _bound(self, token, is_end):
        parts = token.split(".")
        if len(parts) == 3:
            return self.position.get(token)
        if len(parts) == 2 and parts[1].isdigit():
            span = self.chapters.get((parts[0], int(parts[1])))
            if span is None:
                return None
            return span[1] if is_end else span[0]
        return None

    def resolve(self, reference):
        start, _, end = reference.partition("-")
        first = self._bound(start, is_end=False)
        last = self._bound(end or start, is_end=True)
        if first is None or last is None or last < first:
            return None
        return self.ids[first:last + 1]