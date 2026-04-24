import re

from .models import SubtitleSegment
from .timecode import format_srt_time, parse_srt_time


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def estimate_speech_seconds(
    text: str,
    speed_multiplier: float = 1.0,
    chars_per_second: float = 17.0,
) -> float:
    cleaned = normalize_text(text)
    if not cleaned:
        return 0.0
    safe_speed = max(0.5, speed_multiplier)
    return max(1.0, len(cleaned) / (chars_per_second * safe_speed))


def fit_text_to_duration(
    text: str,
    duration_seconds: float,
    speed_multiplier: float = 1.0,
    chars_per_second: float = 17.5,
    min_chars: int = 18,
) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""

    budget = max(min_chars, int(duration_seconds * chars_per_second * max(0.6, speed_multiplier)))
    if len(cleaned) <= budget:
        return cleaned

    clauses = [
        normalize_text(part)
        for part in re.split(r"(?<=[.!?;,])\s+|(?:\s+-\s+)|(?:\s+ve\s+)", cleaned)
        if normalize_text(part)
    ]

    selected: list[str] = []
    for clause in clauses:
        candidate = " ".join([*selected, clause]).strip()
        if selected and len(candidate) > budget:
            break
        selected.append(clause)

    if selected:
        compact = " ".join(selected).strip(" ,;:-")
        if compact and len(compact) <= budget:
            return compact

    words: list[str] = []
    for word in cleaned.split():
        candidate = " ".join([*words, word])
        if words and len(candidate) > budget:
            break
        words.append(word)
    return " ".join(words).strip(" ,;:-") or cleaned[:budget].strip(" ,;:-")


def normalize_segment_overlaps(
    segments: list[SubtitleSegment],
    min_gap_ms: int = 0,
    min_duration_ms: int = 350,
) -> list[SubtitleSegment]:
    normalized: list[SubtitleSegment] = []
    previous_end_ms = -1

    for segment in segments:
        start_ms = parse_srt_time(segment.start_time)
        end_ms = parse_srt_time(segment.end_time)
        if previous_end_ms >= 0 and start_ms < previous_end_ms + min_gap_ms:
            start_ms = previous_end_ms + min_gap_ms
        if end_ms <= start_ms:
            end_ms = start_ms + min_duration_ms
        normalized.append(
            SubtitleSegment(
                start_time=format_srt_time(start_ms),
                end_time=format_srt_time(end_ms),
                text=normalize_text(segment.text),
            )
        )
        previous_end_ms = end_ms

    return normalized


def has_overlaps(segments: list[SubtitleSegment]) -> bool:
    previous_end_ms = -1
    for segment in segments:
        start_ms = parse_srt_time(segment.start_time)
        end_ms = parse_srt_time(segment.end_time)
        if previous_end_ms >= 0 and start_ms < previous_end_ms:
            return True
        previous_end_ms = end_ms
    return False
