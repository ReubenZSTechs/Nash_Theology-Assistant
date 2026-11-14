from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import APIConnectionError, APIError, InternalServerError, OpenAI
from tqdm import tqdm
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from backend.utils.common_configs import SERMON_METADATA, read_jsonl
from backend.utils.transcribe_configs import (
    ALLOWED_LANGUAGES,
    COMPOSE_FILE,
    DEFAULT_AUDIO_BUFFER,
    DEFAULT_DOWNLOAD_WORKERS,
    DEFAULT_MODEL,
    DEFAULT_TRANSCRIBE_WORKERS,
    DEFAULT_URL,
    EMPTY_WINDOW_ERROR,
    LANGUAGE_ALIASES,
    MAIN_WORKSPACE,
    MAX_ATTEMPTS,
    MAX_PART_SECONDS,
    MIN_PART_SECONDS,
    PROBE_POSITIONS,
    PROBE_SECONDS,
    PROMPT_RESCUE_WARNING,
    PROMPTS,
    RETRY_SECONDS,
    SAMPLE_RATE,
    SAMPLE_WORKSPACE,
    STOP_POLL_SECONDS,
    WINDOW_SECONDS,
    SegmentShaper,
    is_prompt_echo,
)
from backend.core.exceptions.ServerExceptions import ServerUnavailable
from backend.core.exceptions.TranscribeExceptions import EmptyWindow

WORK_ERRORS = (DownloadError, APIError, subprocess.CalledProcessError, OSError, ValueError)


class RunStopped(Exception):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def require_ffmpeg():
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Install with: sudo apt install ffmpeg")


def audio_duration(path):
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    return float(output)


def cut_audio(path, start, seconds):
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=path.parent, delete=False) as handle:
        clip = Path(handle.name)
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{seconds:.3f}",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(clip),
    ]
    try:
        subprocess.run(command, capture_output=True, check=True)
        return clip.read_bytes()
    finally:
        clip.unlink(missing_ok=True)


def find_audio(workspace, video_id):
    for candidate in sorted(workspace.audio.glob(f"{video_id}.*")):
        if candidate.suffix not in (".part", ".ytdl", ".wav", ".tmp"):
            return candidate
    return None


def youtube_options(extra):
    options = {"quiet": True, "no_warnings": True}
    browser = os.getenv("YTDLP_COOKIES_BROWSER")
    if browser:
        options["cookiesfrombrowser"] = (browser,)
    options.update(extra)
    return options


def download_audio(workspace, record, stop_event):
    existing = find_audio(workspace, record["video_id"])
    if existing:
        return existing

    def on_progress(status):
        if stop_event.is_set():
            raise RunStopped()

    options = youtube_options(
        {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": str(workspace.audio / "%(id)s.%(ext)s"),
            "noprogress": True,
            "continuedl": True,
            "retries": 5,
            "fragment_retries": 5,
            "progress_hooks": [on_progress],
        }
    )
    with YoutubeDL(options) as ydl:
        ydl.download([record["url"]])
    downloaded = find_audio(workspace, record["video_id"])
    if downloaded is None:
        raise DownloadError(f"No audio file produced for {record['video_id']}")
    return downloaded


def normalize_language(language):
    code = (language or "").strip().lower()
    return LANGUAGE_ALIASES.get(code, code)


class Transcriber:
    def __init__(self, url, model, part_seconds, workers, language, shaper, use_prompt):
        self.url = url
        self.model = model
        self.part_seconds = part_seconds
        self.workers = workers
        self.forced_language = language
        self.shaper = shaper
        self.use_prompt = use_prompt
        self.prompt_rescues = 0
        self.prompt_warned = False
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(workers)
        self.stop_event = threading.Event()
        self.max_retries = int(os.getenv("TRANSCRIBE_MAX_RETRIES", "2"))
        self.client = OpenAI(
            base_url=url,
            api_key=os.getenv("VLLM_API_KEY", "not-needed"),
            timeout=float(os.getenv("TRANSCRIBE_TIMEOUT", "900")),
            max_retries=0,
        )

    def check_server(self):
        try:
            served = [model.id for model in self.client.models.list().data]
        except APIConnectionError as error:
            raise SystemExit(
                f"Transcription server unreachable at {self.url}. Start it with:\n"
                f"  TRANSCRIBE_MODEL={self.model} docker compose -f {COMPOSE_FILE} up -d"
            ) from error
        if self.model not in served:
            raise SystemExit(
                f"Model '{self.model}' is not served at {self.url} (serving {served}). Restart the server with:\n"
                f"  TRANSCRIBE_MODEL={self.model} docker compose -f {COMPOSE_FILE} up -d"
            )
        suffix = " with vocabulary prompt" if self.use_prompt else ""
        print(f"[transcribe] {self.model} via {self.url}{suffix}, {self.workers} parallel requests")

    def prompt_for(self, language):
        if not self.use_prompt:
            return None
        return PROMPTS.get(language)

    def acquire_slot(self):
        while not self.slots.acquire(timeout=STOP_POLL_SECONDS):
            if self.stop_event.is_set():
                raise RunStopped()
        if self.stop_event.is_set():
            self.slots.release()
            raise RunStopped()

    def send(self, arguments):
        self.acquire_slot()
        try:
            return self.client.audio.transcriptions.create(**arguments)
        finally:
            self.slots.release()

    def request(self, audio_bytes, language, prompt=None):
        arguments = {
            "model": self.model,
            "file": ("part.wav", audio_bytes, "audio/wav"),
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
            "temperature": 0.0,
        }
        if language:
            arguments["language"] = language
        if prompt:
            arguments["prompt"] = prompt
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.send(arguments)
            except InternalServerError as error:
                if EMPTY_WINDOW_ERROR in str(error):
                    raise EmptyWindow() from error
                last_error = error
            except APIConnectionError as error:
                last_error = error
            if attempt < self.max_retries and self.stop_event.wait(RETRY_SECONDS * (attempt + 1)):
                raise RunStopped()
        if isinstance(last_error, APIConnectionError):
            raise ServerUnavailable(f"Transcription server unreachable at {self.url}") from last_error
        raise last_error

    def detect_language(self, audio_path, duration):
        if self.forced_language:
            return self.forced_language, None
        votes = Counter()
        for position in PROBE_POSITIONS:
            start = max(0.0, min(duration * position, duration - PROBE_SECONDS))
            seconds = min(PROBE_SECONDS, duration - start)
            if seconds < MIN_PART_SECONDS:
                continue
            try:
                response = self.request(cut_audio(audio_path, start, seconds), None)
            except EmptyWindow:
                continue
            language = normalize_language(response.language)
            if language:
                votes[language] += 1
        if not votes:
            return "unknown", None
        language, count = votes.most_common(1)[0]
        return language, round(count / sum(votes.values()), 2)

    def transcribe_part(self, audio_path, start, seconds, language):
        prompt = self.prompt_for(language)
        try:
            response = self.request(cut_audio(audio_path, start, seconds), language, prompt)
        except EmptyWindow:
            return self.transcribe_windows(audio_path, start, seconds, language, prompt)
        return self.to_segments(response, start, seconds, prompt)

    def transcribe_windows(self, audio_path, start, seconds, language, prompt):
        segments = []
        for index in range(math.ceil(seconds / WINDOW_SECONDS)):
            offset = index * WINDOW_SECONDS
            length = min(WINDOW_SECONDS, seconds - offset)
            if length < MIN_PART_SECONDS:
                continue
            audio = cut_audio(audio_path, start + offset, length)
            response = self.window_response(audio, language, prompt)
            if response is not None:
                segments.extend(self.to_segments(response, start + offset, length, prompt))
        return segments

    def window_response(self, audio, language, prompt):
        try:
            return self.request(audio, language, prompt)
        except EmptyWindow:
            if not prompt:
                return None
        try:
            response = self.request(audio, language)
        except EmptyWindow:
            return None
        self.note_prompt_rescue()
        return response

    def note_prompt_rescue(self):
        with self.lock:
            self.prompt_rescues += 1
            if self.prompt_warned or self.prompt_rescues < PROMPT_RESCUE_WARNING:
                return
            self.prompt_warned = True
        tqdm.write(
            "[warn] prompted requests keep coming back empty but work without the prompt. "
            "The transcribe container is probably not patched; recreate it with:\n"
            f"  docker compose -f {COMPOSE_FILE} up -d --force-recreate"
        )

    @staticmethod
    def to_segments(response, start, seconds, prompt=None):
        timed = []
        for segment in response.segments or []:
            text = (segment.text or "").strip()
            if not text or (prompt and is_prompt_echo(text, prompt)):
                continue
            begin = max(float(segment.start), 0.0)
            timed.append((begin, max(float(segment.end), begin), text))
        latest = max((end for _, end, _ in timed), default=0.0)
        scale = seconds / latest if latest > seconds else 1.0
        segments = []
        for begin, end, text in timed:
            begin = min(begin * scale, seconds)
            end = min(max(end * scale, begin), seconds)
            segments.append({"start": round(start + begin, 2), "end": round(start + end, 2), "text": text})
        full_text = (response.text or "").strip()
        if prompt and is_prompt_echo(full_text, prompt):
            full_text = ""
        if not segments and full_text:
            segments.append({"start": round(start, 2), "end": round(start + seconds, 2), "text": full_text})
        segments.sort(key=lambda segment: segment["start"])
        return segments


def checkpoint_path(workspace, video_id):
    return workspace.checkpoints / f"{video_id}.json"


def load_checkpoint(workspace, record, transcriber, duration):
    path = checkpoint_path(workspace, record["video_id"])
    fresh = {
        "video_id": record["video_id"],
        "model": transcriber.model,
        "part_seconds": transcriber.part_seconds,
        "prompt": transcriber.use_prompt,
        "duration": round(duration, 2),
        "language": None,
        "language_probability": None,
        "parts": {},
    }
    saved = read_json(path, None)
    if saved is None:
        return fresh
    same_run = all(saved.get(key, False) == fresh[key] for key in ("model", "part_seconds", "prompt", "duration"))
    if not same_run:
        tqdm.write(f"[transcribe] {record['video_id']}: settings changed, restarting this video")
        return fresh
    return saved


def part_bounds(index, part_seconds, duration):
    start = index * part_seconds
    return start, min(part_seconds, duration - start)


def build_transcript(record, checkpoint, part_count, shaper):
    raw = []
    for index in range(part_count):
        raw.extend(checkpoint["parts"][str(index)])
    segments = shaper.shape(raw)
    return {
        "video_id": record["video_id"],
        "source": record.get("source"),
        "channel_id": record.get("channel_id"),
        "title": record.get("title"),
        "url": record.get("url"),
        "duration_sec": record.get("duration_sec"),
        "language": checkpoint["language"],
        "language_probability": checkpoint["language_probability"],
        "model": checkpoint["model"],
        "transcribed_seconds": checkpoint["duration"],
        "prompt_used": checkpoint.get("prompt", False),
        "segment_limits": {"target_seconds": shaper.target_seconds, "max_seconds": shaper.max_seconds},
        "transcribed_at": utc_now(),
        "segments": segments,
    }


def cleanup(workspace, video_id, audio_path):
    if audio_path is not None:
        audio_path.unlink(missing_ok=True)
    checkpoint_path(workspace, video_id).unlink(missing_ok=True)


def record_minutes(record, limit_seconds):
    seconds = record.get("duration_sec") or 0
    if limit_seconds:
        seconds = min(seconds, limit_seconds)
    return seconds / 60


def select_pending(records, workspace, progress, retry_failed):
    pending = []
    for record in records:
        video_id = record["video_id"]
        if (workspace.transcripts / f"{video_id}.json").exists():
            continue
        if video_id in progress["skipped"]:
            continue
        attempts = progress["failed"].get(video_id, {}).get("attempts", 0)
        if attempts >= MAX_ATTEMPTS and not retry_failed:
            continue
        pending.append(record)
    return pending


def reset_records(records, workspace, progress):
    for record in records:
        video_id = record["video_id"]
        (workspace.transcripts / f"{video_id}.json").unlink(missing_ok=True)
        progress["skipped"].pop(video_id, None)
        progress["failed"].pop(video_id, None)


class Pipeline:
    def __init__(self, transcriber, workspace, limit_seconds, download_workers, buffer_size):
        self.transcriber = transcriber
        self.workspace = workspace
        self.limit_seconds = limit_seconds
        self.download_workers = download_workers
        self.disk_slots = threading.BoundedSemaphore(buffer_size)
        self.stop_event = transcriber.stop_event
        self.lock = threading.Lock()
        self.counts = Counter()
        self.progress = None
        self.audio_bar = None
        self.download_bar = None
        self.fatal = None

    def credit(self, minutes):
        with self.lock:
            if self.audio_bar.total is not None:
                minutes = min(minutes, self.audio_bar.total - self.audio_bar.n)
            if minutes > 0:
                self.audio_bar.update(minutes)

    def add_audio_total(self, minutes):
        with self.lock:
            self.audio_bar.total = round(self.audio_bar.total + minutes, 2)
            self.audio_bar.refresh()

    def refresh_postfix(self):
        self.audio_bar.set_postfix(
            done=self.counts["done"],
            skipped=self.counts["skipped"],
            failed=self.counts["failed"],
            active=self.counts["active"],
            refresh=False,
        )
        self.download_bar.set_postfix(on_disk=self.counts["on_disk"], refresh=True)

    def acquire_disk_slot(self):
        while not self.disk_slots.acquire(timeout=STOP_POLL_SECONDS):
            if self.stop_event.is_set():
                raise RunStopped()
        if self.stop_event.is_set():
            self.disk_slots.release()
            raise RunStopped()

    def download(self, record):
        self.acquire_disk_slot()
        try:
            path = download_audio(self.workspace, record, self.stop_event)
        except BaseException:
            self.disk_slots.release()
            raise
        with self.lock:
            self.counts["on_disk"] += 1
            self.download_bar.update(1)
            self.refresh_postfix()
        return path

    def release_disk_slot(self):
        with self.lock:
            self.counts["on_disk"] -= 1
            self.refresh_postfix()
        self.disk_slots.release()

    def finish(self, record, status, detail, error=None, credited=0.0):
        video_id = record["video_id"]
        with self.lock:
            if status == "skipped":
                self.progress["skipped"][video_id] = {"reason": detail, "title": record.get("title"), "at": utc_now()}
            if status == "done":
                self.progress["failed"].pop(video_id, None)
            if status == "failed":
                entry = self.progress["failed"].get(video_id, {"attempts": 0})
                entry["attempts"] += 1
                entry["error"] = f"{type(error).__name__}: {str(error)[:300]}"
                entry["title"] = record.get("title")
                entry["at"] = utc_now()
                self.progress["failed"][video_id] = entry
            write_json(self.workspace.progress, self.progress)
            self.counts[status] += 1
            self.refresh_postfix()
        self.credit(record_minutes(record, self.limit_seconds) - credited)
        tqdm.write(f"[{status}] {video_id} {(record.get('title') or '')[:60]} — {detail}")

    def handle(self, record, download_future):
        while not download_future.done():
            if self.stop_event.wait(STOP_POLL_SECONDS):
                return
        try:
            audio_path = download_future.result()
        except (RunStopped, CancelledError):
            return
        except WORK_ERRORS as error:
            if self.stop_event.is_set():
                return
            self.finish(record, "failed", f"{type(error).__name__}: {str(error)[:200]}", error)
            return

        credited = [0.0]

        def on_minutes(minutes):
            credited[0] += minutes
            self.credit(minutes)

        with self.lock:
            self.counts["active"] += 1
            self.refresh_postfix()
        try:
            status, detail = process_video(self.transcriber, self.workspace, record, audio_path, self.limit_seconds, on_minutes, self.add_audio_total)
            self.finish(record, status, detail, credited=credited[0])
        except RunStopped:
            pass
        except ServerUnavailable as error:
            self.fatal = error
            self.stop_event.set()
        except WORK_ERRORS as error:
            if not self.stop_event.is_set():
                remaining = find_audio(self.workspace, record["video_id"])
                if remaining is not None:
                    remaining.unlink(missing_ok=True)
                self.finish(record, "failed", f"{type(error).__name__}: {str(error)[:200]}", error, credited[0])
        finally:
            with self.lock:
                self.counts["active"] -= 1
            self.release_disk_slot()

    def run(self, records, retry_failed, redo):
        self.workspace.prepare()
        self.progress = read_json(self.workspace.progress, {"skipped": {}, "failed": {}})
        if redo:
            reset_records(records, self.workspace, self.progress)
        pending = select_pending(records, self.workspace, self.progress, retry_failed)
        print(f"[transcribe] {self.workspace.name}: {len(pending)} videos to do, {len(records) - len(pending)} already handled")
        if not pending:
            return

        total_minutes = round(sum(record_minutes(record, self.limit_seconds) for record in pending), 2)
        audio_format = "{l_bar}{bar}| {n:.0f}/{total:.0f} audio min [{elapsed}<{remaining}]{postfix}"
        self.audio_bar = tqdm(total=total_minutes, desc="[transcribe]", position=0, dynamic_ncols=True, bar_format=audio_format)
        self.download_bar = tqdm(total=len(pending), desc="[download]  ", unit="video", position=1, dynamic_ncols=True)
        downloads = ThreadPoolExecutor(max_workers=self.download_workers, thread_name_prefix="download")
        videos = ThreadPoolExecutor(max_workers=self.transcriber.workers, thread_name_prefix="video")
        try:
            jobs = []
            for record in pending:
                download_future = downloads.submit(self.download, record)
                jobs.append(videos.submit(self.handle, record, download_future))
            for job in as_completed(jobs):
                job.result()
                if self.fatal:
                    break
        except KeyboardInterrupt:
            self.stop_event.set()
            tqdm.write("[transcribe] stopping: finishing requests already sent, then saving progress...")
        finally:
            if self.fatal:
                self.stop_event.set()
            downloads.shutdown(wait=True, cancel_futures=True)
            videos.shutdown(wait=True, cancel_futures=True)
            self.audio_bar.close()
            self.download_bar.close()

        if self.fatal:
            raise SystemExit(f"[transcribe] {self.fatal}. Progress is saved; run again once the server is back.")
        if self.stop_event.is_set():
            raise SystemExit("[transcribe] stopped. Run the same command again to continue from this point.")
        print(f"[transcribe] finished: {self.counts['done']} done, {self.counts['skipped']} skipped, {self.counts['failed']} failed")
        print(f"[transcribe] transcripts -> {self.workspace.transcripts}")


def run_parts(transcriber, workspace, record, audio_path, checkpoint, pending, duration, on_minutes):
    lock = threading.Lock()
    path = checkpoint_path(workspace, record["video_id"])

    def work(index):
        if transcriber.stop_event.is_set():
            raise RunStopped()
        start, seconds = part_bounds(index, transcriber.part_seconds, duration)
        segments = transcriber.transcribe_part(audio_path, start, seconds, checkpoint["language"])
        with lock:
            checkpoint["parts"][str(index)] = segments
            write_json(path, checkpoint)
        return seconds

    with ThreadPoolExecutor(max_workers=transcriber.workers, thread_name_prefix="part") as executor:
        futures = [executor.submit(work, index) for index in pending]
        try:
            for future in as_completed(futures):
                on_minutes(future.result() / 60)
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def process_video(transcriber, workspace, record, audio_path, limit_seconds, on_minutes, add_audio_total):
    video_id = record["video_id"]
    duration = audio_duration(audio_path)
    if limit_seconds:
        duration = min(duration, limit_seconds)
    if not record.get("duration_sec"):
        record["duration_sec"] = round(audio_duration(audio_path))
        add_audio_total(duration / 60)

    checkpoint = load_checkpoint(workspace, record, transcriber, duration)
    if checkpoint["language"] is None:
        language, probability = transcriber.detect_language(audio_path, duration)
        checkpoint["language"] = language
        checkpoint["language_probability"] = probability
        write_json(checkpoint_path(workspace, video_id), checkpoint)

    if checkpoint["language"] not in ALLOWED_LANGUAGES:
        cleanup(workspace, video_id, audio_path)
        return "skipped", f"language_{checkpoint['language']}"

    part_count = max(1, math.ceil(duration / transcriber.part_seconds))
    pending = []
    for index in range(part_count):
        _, seconds = part_bounds(index, transcriber.part_seconds, duration)
        if seconds < MIN_PART_SECONDS:
            checkpoint["parts"].setdefault(str(index), [])
        if str(index) not in checkpoint["parts"]:
            pending.append(index)

    done_seconds = sum(part_bounds(index, transcriber.part_seconds, duration)[1] for index in range(part_count) if index not in pending)
    on_minutes(done_seconds / 60)
    run_parts(transcriber, workspace, record, audio_path, checkpoint, pending, duration, on_minutes)

    transcript = build_transcript(record, checkpoint, part_count, transcriber.shaper)
    if not transcript["segments"]:
        cleanup(workspace, video_id, audio_path)
        return "skipped", "no_speech"
    write_json(workspace.transcripts / f"{video_id}.json", transcript)
    cleanup(workspace, video_id, audio_path)
    return "done", f"{len(transcript['segments'])} segments, {checkpoint['language']}"


def main_records(source, limit):
    if not SERMON_METADATA.exists():
        raise SystemExit(f"{SERMON_METADATA} not found. Run the crawler with the 'sermons' target first.")
    records = [record for record in read_jsonl(SERMON_METADATA) if record.get("included")]
    if source:
        records = [record for record in records if record.get("source") == source]
    return records[:limit] if limit else records


def sample_records(urls):
    records = []
    with YoutubeDL(youtube_options({"skip_download": True})) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=False)
            except DownloadError as error:
                raise SystemExit(f"Could not read {url}: {str(error)[:300]}") from error
            records.append(
                {
                    "video_id": info["id"],
                    "source": "sample",
                    "channel_id": info.get("channel_id"),
                    "title": info.get("title"),
                    "url": info.get("webpage_url") or url,
                    "duration_sec": info.get("duration"),
                }
            )
    return records


def part_length(value):
    seconds = int(value)
    if not 30 <= seconds <= MAX_PART_SECONDS:
        raise argparse.ArgumentTypeError(f"must be between 30 and {MAX_PART_SECONDS}")
    return seconds


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description="Download, transcribe and delete sermon audio with a vLLM Whisper server")
    parser.add_argument("--model", default=os.getenv("TRANSCRIBE_MODEL", DEFAULT_MODEL), help="served Whisper model name")
    parser.add_argument("--url", default=os.getenv("TRANSCRIBE_URL", DEFAULT_URL), help="vLLM transcription server URL")
    parser.add_argument("--sample", nargs="+", metavar="URL", help="transcribe only these YouTube links into the samples folder")
    parser.add_argument("--sample-minutes", type=float, default=0, help="with --sample: transcribe only the first N minutes (0 = whole video)")
    parser.add_argument("--source", help="main run: only this channel source, e.g. cit_chapel")
    parser.add_argument("--limit", type=int, default=0, help="main run: consider only the first N videos (0 = all)")
    parser.add_argument("--language", choices=ALLOWED_LANGUAGES, help="skip detection and force this language")
    parser.add_argument("--part-seconds", type=part_length, default=300, help="audio seconds per request and per checkpoint")
    parser.add_argument("--prompt", action="store_true", help="send the built-in vocabulary prompt (needs the patched transcribe container)")
    parser.add_argument("--segment-seconds", type=float, default=0.0, help="join segments until at least this many seconds, then cut at the next sentence end (0 = keep Whisper's segments)")
    parser.add_argument("--max-segment-seconds", type=float, default=60.0, help="cut anyway at this length if no sentence end appears (0 = never)")
    parser.add_argument("--workers", type=positive, default=int(os.getenv("TRANSCRIBE_WORKERS", DEFAULT_TRANSCRIBE_WORKERS)), help="parallel requests to the Whisper server")
    parser.add_argument("--download-workers", type=positive, default=int(os.getenv("DOWNLOAD_WORKERS", DEFAULT_DOWNLOAD_WORKERS)), help="parallel YouTube downloads")
    parser.add_argument("--buffer", type=positive, default=int(os.getenv("AUDIO_BUFFER", DEFAULT_AUDIO_BUFFER)), help="most audio files kept on disk at once")
    parser.add_argument("--cookies-browser", help="read YouTube cookies from this browser (firefox, chrome, ...) if YouTube asks to sign in")
    parser.add_argument("--retry-failed", action="store_true", help=f"retry videos that already failed {MAX_ATTEMPTS} times")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    require_ffmpeg()
    if args.cookies_browser:
        os.environ["YTDLP_COOKIES_BROWSER"] = args.cookies_browser
    shaper = SegmentShaper(max(0.0, args.segment_seconds), max(0.0, args.max_segment_seconds))
    transcriber = Transcriber(args.url, args.model, args.part_seconds, args.workers, args.language, shaper, args.prompt)
    transcriber.check_server()
    buffer_size = max(args.buffer, args.workers)
    pipeline = Pipeline(transcriber, SAMPLE_WORKSPACE if args.sample else MAIN_WORKSPACE, args.sample_minutes * 60 if args.sample else 0, args.download_workers, buffer_size)

    if args.sample:
        pipeline.run(sample_records(args.sample), args.retry_failed, redo=True)
    else:
        pipeline.run(main_records(args.source, args.limit), args.retry_failed, redo=False)