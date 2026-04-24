import json
import re
from typing import Iterable

from .models import SubtitleSegment
from .timecode import parse_srt_time


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_markdown_fence(raw_text: str) -> str:
    match = JSON_BLOCK_RE.search(raw_text.strip())
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def parse_gemini_segments(raw_text: str) -> list[SubtitleSegment]:
    """Parse and validate the JSON array returned by Gemini."""
    try:
        payload = json.loads(_strip_markdown_fence(raw_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini response is not valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Gemini response must be a JSON array")

    return validate_segments(payload)


def validate_segments(items: Iterable[dict]) -> list[SubtitleSegment]:
    """Validate segment shape and chronological order."""
    segments: list[SubtitleSegment] = []
    previous_start_ms = -1

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Segment {index} must be an object")

        try:
            start_time = str(item["start_time"]).strip()
            end_time = str(item["end_time"]).strip()
            text = str(item["text"]).strip()
        except KeyError as exc:
            raise ValueError(f"Segment {index} is missing key: {exc}") from exc

        if not text:
            raise ValueError(f"Segment {index} text cannot be empty")

        start_ms = parse_srt_time(start_time)
        end_ms = parse_srt_time(end_time)
        if end_ms <= start_ms:
            raise ValueError(f"Segment {index} end_time must be after start_time")
        if start_ms < previous_start_ms:
            raise ValueError(f"Segment {index} starts before a previous segment")

        previous_start_ms = start_ms
        segments.append(
            SubtitleSegment(start_time=start_time, end_time=end_time, text=text)
        )

    if not segments:
        raise ValueError("Gemini returned no narration segments")

    return segments

