from pathlib import Path

from .models import SubtitleSegment
from .timecode import format_srt_time, parse_srt_time
from .tts_generator import read_media_duration_seconds


def align_segments_to_audio(
    segments: list[SubtitleSegment],
    audio_paths: list[Path],
    min_gap_ms: int = 0,
    gap_seconds: float = 0.0,
    tail_padding_ms: int = 40,
    max_end_ms: int | None = None,
) -> list[SubtitleSegment]:
    aligned: list[SubtitleSegment] = []
    current_end_ms = -1
    gap_ms = int(max(0.0, gap_seconds) * 1000)

    for segment, audio_path in zip(segments, audio_paths):
        original_start_ms = parse_srt_time(segment.start_time)
        earliest_start_ms = 0
        if current_end_ms >= 0:
            earliest_start_ms = max(earliest_start_ms, current_end_ms + min_gap_ms + gap_ms)
        planned_start_ms = max(original_start_ms, earliest_start_ms)
        audio_duration_ms = max(250, int(read_media_duration_seconds(audio_path) * 1000) + tail_padding_ms)
        planned_end_ms = planned_start_ms + audio_duration_ms
        if max_end_ms is not None:
            if planned_end_ms > max_end_ms:
                latest_start_ms = max(earliest_start_ms, max_end_ms - audio_duration_ms)
                planned_start_ms = latest_start_ms
                planned_end_ms = min(max_end_ms, planned_start_ms + audio_duration_ms)
        aligned.append(
            SubtitleSegment(
                start_time=format_srt_time(planned_start_ms),
                end_time=format_srt_time(max(planned_start_ms + 250, planned_end_ms)),
                text=segment.text,
            )
        )
        current_end_ms = parse_srt_time(aligned[-1].end_time)

    return aligned


def append_outro_segment(
    segments: list[SubtitleSegment],
    video_duration_seconds: float,
    text: str,
    duration_seconds: float = 3.6,
) -> list[SubtitleSegment]:
    if not text.strip():
        return segments

    total_ms = max(1000, int(video_duration_seconds * 1000))
    outro_duration_ms = min(int(duration_seconds * 1000), total_ms)
    start_ms = max(0, total_ms - outro_duration_ms)
    end_ms = total_ms

    if segments and parse_srt_time(segments[-1].end_time) > start_ms:
        trimmed = segments[:-1]
    else:
        trimmed = segments[:]

    trimmed.append(
        SubtitleSegment(
            start_time=format_srt_time(start_ms),
            end_time=format_srt_time(end_ms),
            text=text.strip(),
        )
    )
    return trimmed
