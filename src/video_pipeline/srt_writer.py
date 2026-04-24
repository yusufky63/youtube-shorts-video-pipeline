from pathlib import Path

from .models import SubtitleSegment


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> Path:
    """Write validated narration segments as a standard UTF-8 SRT file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []

        for index, segment in enumerate(segments, start=1):
            lines.extend(
                [
                    str(index),
                    f"{segment.start_time} --> {segment.end_time}",
                    segment.text,
                    "",
                ]
            )

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
    except OSError as exc:
        raise RuntimeError(f"Could not write SRT file to {output_path}: {exc}") from exc

