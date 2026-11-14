from dataclasses import asdict, dataclass

import numpy as np

from backend.utils.transcribe_configs import NON_SPEECH

LENGTH_PENALTY = 0.02


@dataclass(frozen=True)
class ChunkSettings:
    min_seconds: float = 45.0
    target_seconds: float = 90.0
    max_seconds: float = 120.0
    unit_seconds: float = 10.0
    window_units: int = 3
    overlap_seconds: float = 15.0
    strong_shift_percentile: float = 90.0

    def validate(self):
        if not 0 < self.min_seconds <= self.target_seconds <= self.max_seconds:
            raise ValueError("chunk lengths must satisfy 0 < min <= target <= max")
        if self.unit_seconds <= 0 or self.window_units < 1 or self.overlap_seconds < 0:
            raise ValueError("unit_seconds and window_units must be positive, overlap_seconds non-negative")
        if not 0 <= self.strong_shift_percentile <= 100:
            raise ValueError("strong_shift_percentile must be between 0 and 100")
        if self.unit_seconds > self.min_seconds:
            raise ValueError("unit_seconds must not exceed min_seconds")
        return self

    def as_dict(self):
        return asdict(self)


class SemanticChunker:
    def __init__(self, embedder, settings):
        self.embedder = embedder
        self.settings = settings.validate()

    def chunk(self, segments):
        units = self.units(segments)
        if not units:
            return []
        depths = self.depths(self.similarities(units))
        ends = self.boundaries(units, depths)
        return self.assemble(units, ends, depths)

    def units(self, segments):
        units = []
        current = None
        for segment in sorted(segments, key=lambda item: item["start"]):
            text = (segment.get("text") or "").strip()
            if not text or NON_SPEECH.match(text):
                continue
            piece = {"start": segment["start"], "end": segment["end"], "text": text}
            if current is None:
                current = {"start": piece["start"], "end": piece["end"], "text": text, "segments": [piece]}
            else:
                current["end"] = piece["end"]
                current["text"] = f"{current['text']} {text}"
                current["segments"].append(piece)
            if current["end"] - current["start"] >= self.settings.unit_seconds:
                units.append(current)
                current = None
        if current is not None:
            units.append(current)
        return units

    def similarities(self, units):
        if len(units) < 2:
            return np.zeros(0, dtype=np.float32)
        vectors = self.embedder.encode([unit["text"] for unit in units])
        width = self.settings.window_units
        scores = []
        for gap in range(len(units) - 1):
            left = vectors[max(0, gap - width + 1):gap + 1].mean(axis=0)
            right = vectors[gap + 1:gap + 1 + width].mean(axis=0)
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            scores.append(float(left @ right / denominator) if denominator else 0.0)
        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def depths(similarities):
        depths = np.zeros_like(similarities)
        for gap, value in enumerate(similarities):
            left_peak = value
            cursor = gap
            while cursor > 0 and similarities[cursor - 1] >= left_peak:
                left_peak = similarities[cursor - 1]
                cursor -= 1
            right_peak = value
            cursor = gap
            while cursor < len(similarities) - 1 and similarities[cursor + 1] >= right_peak:
                right_peak = similarities[cursor + 1]
                cursor += 1
            depths[gap] = (left_peak - value) + (right_peak - value)
        return depths

    def boundaries(self, units, depths):
        settings = self.settings
        ends = []
        first = 0
        last_index = len(units) - 1
        while first <= last_index:
            if units[last_index]["end"] - units[first]["start"] <= settings.max_seconds:
                ends.append(last_index)
                break
            best = None
            best_score = None
            fallback = first
            for last in range(first, last_index):
                duration = units[last]["end"] - units[first]["start"]
                if duration > settings.max_seconds:
                    break
                fallback = last
                tail = units[last_index]["end"] - units[last + 1]["start"]
                if duration < settings.min_seconds or tail < settings.min_seconds:
                    continue
                score = depths[last] - LENGTH_PENALTY * abs(duration - settings.target_seconds) / settings.target_seconds
                if best_score is None or score > best_score:
                    best = last
                    best_score = score
            cut = best if best is not None else fallback
            ends.append(cut)
            first = cut + 1
        return ends

    def strong_shift(self, depths):
        if not len(depths):
            return float("inf")
        return float(np.percentile(depths, self.settings.strong_shift_percentile))

    def overlap_segments(self, units, first):
        chosen = []
        seconds = 0.0
        for unit in reversed(units[:first]):
            for segment in reversed(unit["segments"]):
                if seconds >= self.settings.overlap_seconds:
                    return chosen
                chosen.insert(0, segment)
                seconds += segment["end"] - segment["start"]
        return chosen

    def assemble(self, units, ends, depths):
        threshold = self.strong_shift(depths)
        chunks = []
        first = 0
        for last in ends:
            core = units[first:last + 1]
            previous_gap = first - 1
            continues_topic = first > 0 and depths[previous_gap] < threshold
            lead = self.overlap_segments(units, first) if self.settings.overlap_seconds and continues_topic else []
            text = " ".join([segment["text"] for segment in lead] + [unit["text"] for unit in core])
            chunks.append(
                {
                    "start": lead[0]["start"] if lead else core[0]["start"],
                    "end": core[-1]["end"],
                    "core_start": core[0]["start"],
                    "text": text,
                }
            )
            first = last + 1
        return chunks