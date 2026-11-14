import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from backend.utils.common_configs import BOOK_BY_NAME_ID, BOOKS, BSB_USFM_DIR, GRAPH_META_PATH, TB_OVERLAY_PATH, TB_PDF_PATH, TRIPLES_PATH, VERSES_PATH, XREF_PATH, VerseIndex, write_jsonl

RELATION = "CROSS_REFERENCE_TO"
SKIPPED_MARKERS = {"id", "usfm", "h", "toc1", "toc2", "toc3", "mt1", "mt2", "ms", "qa", "r", "rem", "ide"}
LINE_MARKER = re.compile(r"^\\(\w+)\s?(.*)$")
VERSE_SPLIT = re.compile(r"\\v (\d+)\s?")
FOOTNOTE = re.compile(r"\\(f|x) .*?\\\1\*")
REFERENCE = re.compile(r"\\ref ([^|\\]*)\|[^\\]*\\ref\*")
INLINE_MARKER = re.compile(r"\\\+?\w+\*?")
TB_HEADER_MULTI = re.compile(r"^(.+?) (\d+)\.(\d+)(?:\s*[–-]\s*(?:\d+\.)?\d+)? \d+$")
TB_HEADER_SINGLE = re.compile(r"^(.+?) (\d+)\s*[–-]\s*\d+ \d+$")
TB_VERSE = re.compile(r"^(\d+)(?:\s*[–-]\s*(\d+))?(?:\s+(.*))?$")
TB_NOISE = re.compile(r"\(\d+\s*[–-]\s*\d+\)|[/*]")


def parse_cap(value):
    if value.lower() == "all":
        return None
    cap = int(value)
    if cap < 0:
        raise argparse.ArgumentTypeError("cap must be >= 0 or 'all'")
    return cap


def clean_usfm(text):
    text = FOOTNOTE.sub("", text)
    text = REFERENCE.sub(r"\1", text)
    text = INLINE_MARKER.sub("", text)
    return " ".join(text.split())


def reference_label(book, chapter, verse):
    name = "Psalm" if book.osis == "Ps" else book.name
    return f"{name} {chapter}:{verse}"


def parse_book(book, order_start):
    verses = []
    chapter = 0
    section = pericope = None
    fragments = None
    for line in (BSB_USFM_DIR / f"{book.usfm}.usfm").read_text(encoding="utf-8").splitlines():
        match = LINE_MARKER.match(line)
        if not match:
            if fragments is not None and line.strip():
                fragments.append(line)
            continue
        marker, rest = match.groups()
        if marker == "c":
            chapter = int(rest.split()[0])
            continue
        if marker == "s1":
            section = pericope = clean_usfm(rest)
            continue
        if marker == "s2":
            pericope = clean_usfm(rest)
            continue
        if marker in SKIPPED_MARKERS or (marker == "d" and "\\v " not in rest):
            continue
        parts = VERSE_SPLIT.split(rest)
        if fragments is not None:
            fragments.append(parts[0])
        for number, text in zip(parts[1::2], parts[2::2]):
            fragments = [text]
            verses.append(
                {
                    "id": f"{book.osis}.{chapter}.{int(number)}",
                    "book": book.osis,
                    "book_name": book.name,
                    "book_name_id": book.name_id,
                    "testament": book.testament,
                    "canonical_order": order_start + len(verses),
                    "chapter": chapter,
                    "verse": int(number),
                    "reference": reference_label(book, chapter, int(number)),
                    "section": section,
                    "pericope": pericope,
                    "text_bsb": fragments,
                }
            )
    for verse in verses:
        verse["text_bsb"] = clean_usfm(" ".join(verse["text_bsb"]))
    return verses


def parse_bsb():
    verses = []
    for book in BOOKS:
        verses.extend(parse_book(book, len(verses)))
    empty = [verse["id"] for verse in verses if not verse["text_bsb"]]
    if empty:
        print(f"[warn] {len(empty)} BSB verses without text: {empty[:10]}")
    return verses


def read_xref():
    with XREF_PATH.open(encoding="utf-8") as file:
        next(file)
        for line in file:
            columns = line.rstrip("\n").split("\t")
            if len(columns) >= 2:
                yield columns[0], columns[1]


def build_pairs(index, cap):
    explicit = set()
    derived = {}
    skipped = Counter()
    for head, tail in read_xref():
        if head not in index:
            skipped["unknown_head"] += 1
            continue
        if "-" not in tail:
            if tail not in index:
                skipped["unknown_tail"] += 1
            elif tail != head:
                pair = frozenset((head, tail))
                explicit.add(pair)
                derived.pop(pair, None)
            continue
        if cap == 0:
            skipped["range_excluded"] += 1
            continue
        targets = index.resolve(tail)
        if targets is None:
            skipped["unknown_range"] += 1
            continue
        if cap is not None and len(targets) > cap:
            skipped["range_over_cap"] += 1
            continue
        for target in targets:
            pair = frozenset((head, target))
            if target != head and pair not in explicit:
                derived.setdefault(pair, tail)
    return explicit, derived, skipped


def triple_rows(index, explicit, derived):
    edges = [(pair, None) for pair in explicit] + list(derived.items())
    rows = []
    for pair, via_range in edges:
        first, second = sorted(pair, key=index.position.get)
        for head, tail in ((first, second), (second, first)):
            row = {"head": head, "relation": RELATION, "tail": tail}
            if via_range:
                row["via_range"] = via_range
            rows.append(row)
    rows.sort(key=lambda row: (index.position[row["head"]], index.position[row["tail"]]))
    return rows


def parse_tb_header(line):
    match = TB_HEADER_MULTI.match(line)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    match = TB_HEADER_SINGLE.match(line)
    if match:
        return match.group(1), 1, int(match.group(2))
    return None


def append_fragment(texts, keys, fragment):
    current = texts[keys[0]]
    if current.endswith("-"):
        texts[keys[0]] = current + fragment
    else:
        texts[keys[0]] = f"{current} {fragment}".strip()


class TbState:
    def __init__(self):
        self.book = None
        self.chapter = 0
        self.expected = 1
        self.keys = None
        self.last = None

    def resync(self, book, chapter, verse):
        if book != self.book:
            self.keys = None
        if (book, chapter, verse) != self.last:
            self.book, self.chapter, self.expected = book, chapter, verse


def page_start(lines, state):
    header = parse_tb_header(lines[0])
    if header and header[0] in BOOK_BY_NAME_ID:
        state.resync(BOOK_BY_NAME_ID[header[0]].osis, header[1], header[2])
        return lines[1:]
    if lines[0] in BOOK_BY_NAME_ID and len(lines) > 1 and lines[1] == "1":
        state.resync(BOOK_BY_NAME_ID[lines[0]].osis, 1, 1)
        return lines[2:]
    return None


def parse_tb_page(lines, state, texts, groups):
    body = page_start(lines, state)
    if body is None:
        return
    for position, line in enumerate(body):
        following = body[position + 1] if position + 1 < len(body) else ""
        if line.isdigit() and following.startswith("1 ") and int(line) in (state.chapter, state.chapter + 1):
            if int(line) == state.chapter + 1:
                state.chapter, state.expected, state.keys = int(line), 1, None
            continue
        match = TB_VERSE.match(line)
        if match and int(match.group(1)) == state.expected:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            state.keys = [f"{state.book}.{state.chapter}.{verse}" for verse in range(start, end + 1)]
            texts[state.keys[0]] = match.group(3) or ""
            groups[state.keys[0]] = state.keys
            state.expected = end + 1
            state.last = (state.book, state.chapter, start)
        elif state.keys:
            append_fragment(texts, state.keys, line)


def parse_tb(index):
    from pypdf import PdfReader

    reader = PdfReader(TB_PDF_PATH)
    state = TbState()
    texts, groups = {}, {}
    for page in reader.pages:
        lines = [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]
        if lines:
            parse_tb_page(lines, state, texts, groups)
    rows = []
    for first, keys in groups.items():
        text = " ".join(TB_NOISE.sub("", texts[first]).split())
        for key in keys:
            rows.append({"id": key, "text_tb": text, "in_bsb": key in index})
    return rows


def report_tb(rows, index):
    found = {row["id"] for row in rows}
    missing = [verse_id for verse_id in index.ids if verse_id not in found]
    extra = [row["id"] for row in rows if not row["in_bsb"]]
    by_book = Counter(verse_id.split(".")[0] for verse_id in missing)
    print(f"[tb] {len(rows)} verses parsed, {len(found & set(index.ids))} matched BSB IDs")
    print(f"[tb] TB-only verses ({len(extra)}): {extra[:20]}")
    print(f"[tb] BSB verses missing in TB ({len(missing)}): {missing[:20]}")
    if by_book:
        print(f"[tb] missing by book: {dict(by_book.most_common(10))}")
    return {"parsed": len(rows), "tb_only": len(extra), "missing": len(missing)}


def degree_summary(rows):
    degree = defaultdict(int)
    for row in rows:
        degree[row["head"]] += 1
    values = sorted(degree.values())
    return {
        "connected_verses": len(values),
        "median_degree": values[len(values) // 2] if values else 0,
        "max_degree": values[-1] if values else 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build the Nash Bible cross-reference graph")
    parser.add_argument("--max-range-verses", type=parse_cap, default=0, help="0 = single-verse links only, N = split ranges up to N verses, 'all' = split every range")
    parser.add_argument("--skip-tb", action="store_true", help="do not build the private TB text overlay")
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    verses = parse_bsb()
    index = VerseIndex(verse["id"] for verse in verses)
    write_jsonl(VERSES_PATH, verses)
    print(f"[bible] {len(verses)} verses -> {VERSES_PATH}")

    explicit, derived, skipped = build_pairs(index, args.max_range_verses)
    rows = triple_rows(index, explicit, derived)
    write_jsonl(TRIPLES_PATH, rows)
    print(f"[graph] {len(explicit)} direct + {len(derived)} via_range links = {len(rows)} triples -> {TRIPLES_PATH}")
    print(f"[graph] skipped source rows: {dict(skipped)}")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "relation": RELATION,
        "max_range_verses": "all" if args.max_range_verses is None else args.max_range_verses,
        "verses": len(verses),
        "direct_links": len(explicit),
        "via_range_links": len(derived),
        "triples": len(rows),
        "skipped_source_rows": dict(skipped),
        **degree_summary(rows),
        "sources": {"bible": "Berean Standard Bible (public domain)", "cross_references": "OpenBible.info (CC-BY)"},
    }
    if not args.skip_tb and TB_PDF_PATH.exists():
        tb_rows = parse_tb(index)
        write_jsonl(TB_OVERLAY_PATH, tb_rows)
        meta["tb_overlay"] = report_tb(tb_rows, index)
    GRAPH_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[graph] summary -> {GRAPH_META_PATH}")