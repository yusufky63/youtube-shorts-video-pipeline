from dataclasses import dataclass


@dataclass(frozen=True)
class SubtitleSegment:
    """A timed narration/subtitle segment returned by Gemini."""

    start_time: str
    end_time: str
    text: str

