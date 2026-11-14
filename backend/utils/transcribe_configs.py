from pathlib import Path
from dataclasses import dataclass
from backend.utils.common_configs import INTERIM, PRIVATE_RAW, TRANSCRIPT_DIR
import re

DEFAULT_MODEL = "whisper-large-v3-turbo"
DEFAULT_URL = "http://localhost:15002/v1"
COMPOSE_FILE = "deployment/compose/docker-compose.transcribe.yaml"
SAMPLE_RATE = 16000
MAX_PART_SECONDS = 600
MIN_PART_SECONDS = 1.0
WINDOW_SECONDS = 30
PROBE_SECONDS = 30
PROBE_POSITIONS = (0.25, 0.5, 0.75)
EMPTY_WINDOW_ERROR = "tuple index out of range"
RETRY_SECONDS = 5
PROMPT_RESCUE_WARNING = 3
ECHO_MIN_WORDS = 6
PROMPTS = {
    "id": (
        " Saudara-saudara, mari kita membaca Alkitab, kitab Wahyu pasal 2 ayat 18. Kepada malaikat jemaat di "
        "Efesus, Smirna, Pergamus, Tiatira, Sardis, Filadelfia, dan Laodikia, Kristus berfirman: bertobatlah "
        "dari zinah dan berhala. Anugerah, pembenaran, dan pengudusan datang dari Allah melalui Roh Kudus."
    ),
    "en": (
        " Brothers and sisters, let us read the Scripture, Romans chapter 8 verse 28. To the churches in Ephesus, "
        "Smyrna, Pergamum, Thyatira, Sardis, Philadelphia, and Laodicea, Christ says: repent. Grace, "
        "justification, and sanctification come from God through the Holy Spirit."
    ),
}
ALLOWED_LANGUAGES = ("id", "en")
LANGUAGE_ALIASES = {"ms": "id", "malay": "id", "indonesian": "id", "english": "en"}
MAX_ATTEMPTS = 3
DEFAULT_TRANSCRIBE_WORKERS = 4
DEFAULT_DOWNLOAD_WORKERS = 6
DEFAULT_AUDIO_BUFFER = 8
STOP_POLL_SECONDS = 1.0


@dataclass(frozen=True)
class Workspace:
    name: str
    transcripts: Path
    checkpoints: Path
    audio: Path
    progress: Path

    def prepare(self):
        for folder in (self.transcripts, self.checkpoints, self.audio):
            folder.mkdir(parents=True, exist_ok=True)


SAMPLE_ROOT = INTERIM / "sermons" / "samples"
MAIN_WORKSPACE = Workspace(
    name="main",
    transcripts=TRANSCRIPT_DIR,
    checkpoints=INTERIM / "sermons" / "checkpoints",
    audio=PRIVATE_RAW / "sermons" / "audio",
    progress=INTERIM / "sermons" / "transcribe_progress.json",
)
SAMPLE_WORKSPACE = Workspace(
    name="sample",
    transcripts=SAMPLE_ROOT / "transcripts",
    checkpoints=SAMPLE_ROOT / "checkpoints",
    audio=SAMPLE_ROOT / "audio",
    progress=SAMPLE_ROOT / "transcribe_progress.json",
)


SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")
NON_SPEECH = re.compile(r"^[\s\[\(♪*]*(music|musik|applause|tepuk tangan|laughter|tawa)?[\s\]\)♪*.]*$", re.IGNORECASE)


@dataclass(frozen=True)
class SegmentShaper:
    target_seconds: float
    max_seconds: float

    def shape(self, segments):
        pieces = []
        for segment in segments:
            if not NON_SPEECH.match(segment["text"]):
                pieces.extend(self.sentences(segment))
        if not self.target_seconds:
            merged = pieces
        else:
            merged = self.merge(pieces)
        return [{"id": index, **segment} for index, segment in enumerate(merged)]

    @staticmethod
    def sentences(segment):
        texts = [text.strip() for text in SENTENCE_SPLIT.split(segment["text"]) if text.strip()]
        total = sum(len(text) for text in texts)
        duration = segment["end"] - segment["start"]
        pieces = []
        position = segment["start"]
        for text in texts:
            end = position + duration * len(text) / total
            pieces.append({"start": round(position, 2), "end": round(end, 2), "text": text})
            position = end
        pieces[-1]["end"] = segment["end"]
        return pieces

    def merge(self, pieces):
        merged = []
        current = None
        for piece in pieces:
            if current is None:
                current = dict(piece)
            else:
                current["end"] = piece["end"]
                current["text"] = f"{current['text']} {piece['text']}"
            duration = current["end"] - current["start"]
            sentence_done = bool(SENTENCE_END.search(piece["text"]))
            reached_target = duration >= self.target_seconds and sentence_done
            reached_limit = bool(self.max_seconds) and duration >= self.max_seconds
            if reached_target or reached_limit:
                merged.append(current)
                current = None
        if current is not None:
            merged.append(current)
        return merged


def normalized_words(text):
    return re.findall(r"\w+", text.lower())


def is_prompt_echo(text, prompt):
    words = normalized_words(text)
    if len(words) < ECHO_MIN_WORDS:
        return False
    return f" {' '.join(words)} " in f" {' '.join(normalized_words(prompt))} "