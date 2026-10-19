import json
import threading
import logging

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict



logger = logging.getLogger(__name__)
write_lock = threading.Lock()

CONFIG = {
    "DEFAULT_LOG_DIR": Path(__file__).resolve().parent.parent / "logging",
    'LLM_LOG': "llm_metrics.jsonl",
    'NODE_LOG': "node_metrics.jsonl",
    'EVALUATION_LOG': "evaluation.jsonl",
    'ERROR_LOG': "error_logs.jsonl"
}


def get_utc_time_now():
    return datetime.now(timezone.utc).isoformat()


def write_record(filename: str, payload: Dict[str, Any]):
    record_file = CONFIG["DEFAULT_LOG_DIR"] / filename

    try:
        record_file.parent.mkdir(parent=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"

        with write_lock:
            with open(record_file, "a", encoding='utf-8') as record_file:
                record_file.write(line)

    except OSError:
        pass


def read_record(filename: str):
    record_file = CONFIG['DEFAULT_LOG_DIR'] / filename

    if not record_file.exists():
        return

    with open(record_file, "r", encoding='utf-8') as record_file:
        for line, i in enumerate(record_file, 1):
            line = line.strip()

            if not line:
                logger.warning(f"Got empty log at entry {i}")
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.error(f"Cannot decode log at entry {i}")
                continue