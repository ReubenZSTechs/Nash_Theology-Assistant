import argparse
import hashlib
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

from backend.utils.common_configs import BSB_USFM_DIR, MANIFEST_PATH, PRIVATE_RAW, SERMON_METADATA, WESTMINSTER_DIR, XREF_PATH, TB_PDF_PATH, write_jsonl

USER_AGENT = "NashDataCrawler/0.2 (personal non-commercial research)"
CREEDS_BASE = "https://raw.githubusercontent.com/NonlinearFruit/Creeds.json/master/creeds"
CHUNK_SIZE = 1 << 16
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 5
MIN_SERMON_SECONDS = 15 * 60
CJK_PATTERN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


@dataclass(frozen=True)
class FileSource:
    name: str
    group: str
    url: str
    path: Path
    license: str
    kind: str
    expected_count: int | None = None


@dataclass(frozen=True)
class ChannelSource:
    name: str
    channel_id: str
    tabs: tuple[str, ...] = ("videos", "streams")
    keyword: str | None = None

    @property
    def url(self):
        return f"https://www.youtube.com/channel/{self.channel_id}"


FILE_SOURCES = (
    FileSource("bsb_usfm", "bible", "https://bereanbible.com/bsb_usfm.zip", BSB_USFM_DIR.parent / "bsb_usfm.zip", "Public Domain", "usfm_zip", 66),
    FileSource("openbible_xref", "bible", "https://a.openbible.info/data/cross-references.zip", XREF_PATH.parent / "cross_references.zip", "CC-BY openbible.info", "xref_zip", 340000),
    FileSource("tb", "bible", "https://download.sabda.org/mobile/pdf/1974_TB.pdf", TB_PDF_PATH, "LAI (personal use only)", "pdf"),
    FileSource("wcf", "westminster", f"{CREEDS_BASE}/westminster_confession_of_faith.json", WESTMINSTER_DIR / "wcf.json", "Public Domain", "creed_json", 33),
    FileSource("wlc", "westminster", f"{CREEDS_BASE}/westminster_larger_catechism.json", WESTMINSTER_DIR / "wlc.json", "Public Domain", "creed_json", 196),
    FileSource("wsc", "westminster", f"{CREEDS_BASE}/westminster_shorter_catechism.json", WESTMINSTER_DIR / "wsc.json", "Public Domain", "creed_json", 107),
    FileSource("wcf_opc", "westminster", "https://opc.org/documents/CFLayout.pdf", PRIVATE_RAW / "westminster" / "opc" / "wcf_proofs.pdf", "OPC (reference only)", "pdf"),
    FileSource("wlc_opc", "westminster", "https://opc.org/documents/LCLayout.pdf", PRIVATE_RAW / "westminster" / "opc" / "wlc_proofs.pdf", "OPC (reference only)", "pdf"),
    FileSource("wsc_opc", "westminster", "https://opc.org/documents/SCLayout.pdf", PRIVATE_RAW / "westminster" / "opc" / "wsc_proofs.pdf", "OPC (reference only)", "pdf"),
)

FILE_GROUPS = ("bible", "westminster")
TARGETS = ("all", *FILE_GROUPS, "sermons")

CHANNEL_SOURCES = (
    ChannelSource("grii_reformed_injili", "UC0SAXUuHJznVpu3hntun2vA"),
    ChannelSource("grii_ku_sore", "UCpmveeReGHBGsklok2mMGiA", tabs=("videos",)),
    ChannelSource("cit_chapel", "UCfJa1oP9gpNWt8KbX9dVQ-Q", keyword="chapel"),
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_session():
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest():
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(MANIFEST_PATH)


def download_total(response, start):
    if response.headers.get("Content-Encoding"):
        return None
    length = int(response.headers.get("Content-Length") or 0)
    return start + length if length else None


def stream_to(session, url, partial, label):
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, stream=True, timeout=(15, 60), headers=headers) as response:
        if response.status_code == 416:
            return
        response.raise_for_status()
        resumed = response.status_code == 206
        start = offset if resumed else 0
        bar = tqdm(
            total=download_total(response, start),
            initial=start,
            desc=f"  {label}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            dynamic_ncols=True,
        )
        with partial.open("ab" if resumed else "wb") as file, bar:
            for block in response.iter_content(CHUNK_SIZE):
                file.write(block)
                bar.update(len(block))


def download(session, source):
    source.path.parent.mkdir(parents=True, exist_ok=True)
    partial = source.path.with_suffix(source.path.suffix + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            stream_to(session, source.url, partial, source.name)
            partial.replace(source.path)
            return
        except requests.RequestException as error:
            tqdm.write(f"[retry] {source.name} attempt {attempt}/{MAX_ATTEMPTS}: {error.__class__.__name__}")
            time.sleep(BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"{source.name}: download failed after {MAX_ATTEMPTS} attempts")


def extract_zip(path):
    target = path.parent.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target):
                raise ValueError(f"{path.name}: unsafe member {member.filename}")
        archive.extractall(target)


def count_items(source):
    if source.kind == "pdf":
        with source.path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                raise ValueError(f"{source.name}: not a valid PDF")
        return None
    if source.kind == "creed_json":
        return len(json.loads(source.path.read_text(encoding="utf-8"))["Data"])
    if source.kind == "usfm_zip":
        extract_zip(source.path)
        return len(list(BSB_USFM_DIR.glob("*.usfm")))
    if source.kind == "xref_zip":
        extract_zip(source.path)
        with XREF_PATH.open(encoding="utf-8") as file:
            return sum(1 for line in file if line[:1].isalnum() and "\t" in line) - 1
    raise ValueError(f"Unknown kind: {source.kind}")


def validate(source):
    count = count_items(source)
    if source.expected_count is None:
        return count
    if source.kind in ("creed_json", "usfm_zip") and count != source.expected_count:
        raise ValueError(f"{source.name}: expected {source.expected_count} items, found {count}")
    if source.kind == "xref_zip" and count < source.expected_count:
        raise ValueError(f"{source.name}: only {count} rows parsed")
    return count


def fetch_source(session, source):
    download(session, source)
    count = validate(source)
    return {
        "url": source.url,
        "path": str(source.path),
        "license": source.license,
        "bytes": source.path.stat().st_size,
        "sha256": sha256_of(source.path),
        "item_count": count,
        "retrieved_at": utc_now(),
    }


def crawl_files(force, groups):
    session = build_session()
    manifest = load_manifest()
    failures = []
    selected = [source for source in FILE_SOURCES if source.group in groups]
    with tqdm(total=len(selected), desc="[files]", unit="file", dynamic_ncols=True) as bar:
        for source in selected:
            bar.set_postfix_str(source.name)
            if source.path.exists() and not force and source.name in manifest:
                tqdm.write(f"[skip] {source.name}")
                bar.update(1)
                continue
            try:
                manifest[source.name] = fetch_source(session, source)
            except (RuntimeError, ValueError, zipfile.BadZipFile) as error:
                failures.append(source.name)
                tqdm.write(f"[fail] {source.name}: {error}")
                bar.update(1)
                continue
            save_manifest(manifest)
            tqdm.write(f"[done] {source.name}: {manifest[source.name]['item_count'] or 'pdf'} -> {source.path}")
            bar.update(1)
            time.sleep(1)
    return failures


def list_tab(channel, tab):
    options = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "sleep_interval_requests": 1,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"{channel.url}/{tab}", download=False, process=False)
        try:
            for entry in (info or {}).get("entries") or []:
                if entry and entry.get("id"):
                    yield entry
        except (DownloadError, ExtractorError) as error:
            tqdm.write(f"[warn] {channel.name}/{tab}: listing stopped early ({error.__class__.__name__})")


def exclusion_reason(entry, channel):
    title = entry.get("title") or ""
    duration = entry.get("duration")
    if channel.keyword and channel.keyword not in title.lower():
        return "keyword_mismatch"
    if duration is None:
        return "no_duration"
    if duration < MIN_SERMON_SECONDS:
        return "too_short"
    if CJK_PATTERN.search(title):
        return "cjk_title"
    return None


def build_record(entry, channel, tab, position):
    reason = exclusion_reason(entry, channel)
    return {
        "video_id": entry["id"],
        "source": channel.name,
        "channel_id": channel.channel_id,
        "tab": tab,
        "position": position,
        "title": entry.get("title"),
        "duration_sec": entry.get("duration"),
        "url": f"https://www.youtube.com/watch?v={entry['id']}",
        "included": reason is None,
        "exclusion_reason": reason,
        "harvested_at": utc_now(),
    }


def harvest_channel(channel, seen):
    records = []
    for tab in channel.tabs:
        included = 0
        with tqdm(desc=f"[sermons] {channel.name}/{tab}", unit="video", dynamic_ncols=True) as bar:
            for position, entry in enumerate(list_tab(channel, tab)):
                bar.update(1)
                if entry["id"] in seen:
                    continue
                seen.add(entry["id"])
                record = build_record(entry, channel, tab, position)
                records.append(record)
                included += record["included"]
                bar.set_postfix(included=included)
    return records


def summarize(records):
    totals = defaultdict(lambda: {"videos": 0, "included": 0, "hours": 0.0})
    reasons = Counter()
    for record in records:
        stats = totals[record["source"]]
        stats["videos"] += 1
        if record["included"]:
            stats["included"] += 1
            stats["hours"] += record["duration_sec"] / 3600
        else:
            reasons[(record["source"], record["exclusion_reason"])] += 1
    for source, stats in totals.items():
        print(f"{source}: {stats['included']}/{stats['videos']} included, {stats['hours']:.0f} h")
    for (source, reason), total in sorted(reasons.items()):
        print(f"  {source} excluded {reason}: {total}")


def crawl_sermons():
    seen = set()
    records = []
    for channel in CHANNEL_SOURCES:
        channel_records = harvest_channel(channel, seen)
        if not channel_records:
            tqdm.write(f"[warn] {channel.name}: no entries returned")
        records.extend(channel_records)
    write_jsonl(SERMON_METADATA, records)
    summarize(records)


def parse_args():
    parser = argparse.ArgumentParser(description="Nash dataset crawler")
    parser.add_argument("targets", nargs="+", choices=TARGETS, help="one or more of: " + ", ".join(TARGETS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    targets = set(TARGETS) if "all" in args.targets else set(args.targets)
    groups = targets & set(FILE_GROUPS)
    failures = crawl_files(args.force, groups) if groups else []
    if "sermons" in targets:
        crawl_sermons()
    if failures:
        raise SystemExit(f"Failed sources: {', '.join(failures)}. Re-run to resume.")