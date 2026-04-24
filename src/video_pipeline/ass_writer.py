from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

from .models import SubtitleSegment
from .timecode import parse_srt_time


@dataclass(frozen=True)
class AssStyleConfig:
    font_name: str = "Arial"
    font_size: int = 62
    primary_color: str = "#FFFFFF"
    accent_color: str = "#FFE45C"
    outline_color: str = "#000000"
    back_color: str = "#000000"
    back_alpha: int = 150
    bold: bool = True
    outline: int = 7
    shadow: int = 3
    shadow_3d: bool = False
    shadow_3d_depth: int = 0
    alignment: int = 2
    margin_v: int = 150
    boxed_background: bool = False
    box_mode: str = "text"
    box_padding_px: int = 24
    box_blur: int = 0
    glow: bool = True
    animate: bool = True
    karaoke_words: bool = False
    max_chars: int = 84
    line_width: int = 26


POSITION_TO_ASS = {
    "Top": (8, 150),
    "Middle": (5, 0),
    "Bottom": (2, 150),
    "TR: Ust": (8, 150),
    "TR: Orta": (5, 0),
    "TR: Alt": (2, 150),
}


def style_for_position(
    config: AssStyleConfig,
    position: str,
    vertical_offset_px: int = 0,
) -> AssStyleConfig:
    alignment, margin_v = POSITION_TO_ASS.get(position, (config.alignment, config.margin_v))
    margin_v = max(0, margin_v + vertical_offset_px)
    return AssStyleConfig(**{**config.__dict__, "alignment": alignment, "margin_v": margin_v})


def _hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    cleaned = hex_color.strip().lstrip("#")
    if len(cleaned) != 6:
        cleaned = "FFFFFF"
    red = cleaned[0:2]
    green = cleaned[2:4]
    blue = cleaned[4:6]
    return f"&H{alpha:02X}{blue}{green}{red}"


def _ass_header(config: AssStyleConfig) -> str:
    border_style = 1
    bold = 1 if config.bold else 0
    primary = _hex_to_ass_color(config.primary_color)
    outline = _hex_to_ass_color(config.outline_color)
    back = _hex_to_ass_color(config.back_color, 255)
    secondary = _hex_to_ass_color(config.accent_color)

    return f"""[Script Info]
ScriptType: v4.00+
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: AnimatedShorts,{config.font_name},{config.font_size},{primary},{secondary},{outline},{back},{bold},0,0,0,100,100,0,0,{border_style},{config.outline},{config.shadow},{config.alignment},80,80,{config.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _format_ass_time(srt_time: str) -> str:
    return _format_ass_time_from_ms(parse_srt_time(srt_time))


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _wrap_caption_text(text: str, line_width: int) -> str:
    lines = wrap(text, width=line_width, break_long_words=False, break_on_hyphens=False)
    return r"\N".join(lines or [text])


def _caption_lines(text: str, line_width: int) -> list[str]:
    lines = wrap(text, width=line_width, break_long_words=False, break_on_hyphens=False)
    return lines or [text]


def _caption_chunks(text: str, max_chars: int, line_width: int) -> list[str]:
    """Split long narration into readable subtitle cards without dropping words."""
    cleaned = _normalize_text(text)
    if not cleaned:
        return [""]

    max_chars = max(20, max_chars)
    line_width = max(10, line_width)
    chunks: list[str] = []
    current_words: list[str] = []

    def fits(value: str) -> bool:
        lines = wrap(value, width=line_width, break_long_words=False, break_on_hyphens=False)
        return len(value) <= max_chars and len(lines) <= 2

    for word in cleaned.split():
        candidate = " ".join([*current_words, word])
        if current_words and not fits(candidate):
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks or [cleaned]


def _split_ass_times(start: str, end: str, parts: int) -> list[tuple[str, str]]:
    start_ms = parse_srt_time(start)
    end_ms = parse_srt_time(end)
    duration = max(1, end_ms - start_ms)
    parts = max(1, parts)

    ranges = []
    for index in range(parts):
        part_start = start_ms + round(duration * index / parts)
        part_end = start_ms + round(duration * (index + 1) / parts)
        ranges.append((_format_ass_time_from_ms(part_start), _format_ass_time_from_ms(part_end)))
    return ranges


def _format_ass_time_from_ms(milliseconds: int) -> str:
    seconds_total, millis = divmod(max(0, milliseconds), 1000)
    minutes_total, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    centiseconds = millis // 10
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"


def _escape_ass_text(text: str) -> str:
    return text.replace("{", "").replace("}", "")


def _parse_ass_time(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    second_value, centiseconds = seconds.split(".")
    return (
        int(hours) * 3600 * 1000
        + int(minutes) * 60 * 1000
        + int(second_value) * 1000
        + int(centiseconds) * 10
    )


def _distribute_units(total: int, weights: list[int]) -> list[int]:
    if not weights:
        return []
    total = max(len(weights), total)
    weight_sum = max(1, sum(weights))
    values = [max(1, round(total * (weight / weight_sum))) for weight in weights]
    diff = total - sum(values)
    index = 0
    while diff != 0 and values:
        slot = index % len(values)
        if diff > 0:
            values[slot] += 1
            diff -= 1
        elif values[slot] > 1:
            values[slot] -= 1
            diff += 1
        index += 1
    return values


def _karaoke_text(chunk: str, start: str, end: str) -> str:
    words = _normalize_text(chunk).split()
    if not words:
        return ""

    start_ms = _parse_ass_time(start)
    end_ms = _parse_ass_time(end)
    duration_cs = max(len(words) * 6, int(max(250, end_ms - start_ms) / 10))
    durations = _distribute_units(duration_cs, [max(1, len(word)) for word in words])

    parts: list[str] = []
    for word, word_duration in zip(words, durations):
        parts.append(rf"{{\kf{word_duration}}}{_escape_ass_text(word)}")
    return " ".join(parts)


def _karaoke_text_from_word_timings(words: list[dict], display_text: str | None = None) -> str:
    parts: list[str] = []
    display_words = _normalize_text(display_text or "").split()
    for index, word in enumerate(words):
        duration_cs = max(1, int(max(10, word["end_ms"] - word["start_ms"]) / 10))
        visible_word = display_words[index] if index < len(display_words) else str(word.get("text", ""))
        parts.append(rf"{{\kf{duration_cs}}}{_escape_ass_text(visible_word)}")
    return " ".join(parts)


def _text_anchor_y(config: AssStyleConfig) -> int:
    if config.alignment in (7, 8, 9):
        return config.margin_v
    if config.alignment in (4, 5, 6):
        return 960 + config.margin_v
    return 1920 - config.margin_v


def _animation_tags(config: AssStyleConfig) -> str:
    anchor_y = _text_anchor_y(config)
    base = rf"\an{config.alignment}\pos(540,{anchor_y})\fad(90,110)\fscx96\fscy96"
    if not config.animate:
        return "{" + base + r"\blur0.4\fscx100\fscy100}"
    if not config.glow:
        return "{" + base + r"\t(0,160,\fscx102\fscy102)}"

    primary = _hex_to_ass_color(config.primary_color)
    accent = _hex_to_ass_color(config.accent_color)
    outline = _hex_to_ass_color(config.outline_color)
    return (
        "{"
        + base
        + rf"\blur0.6\t(0,170,\fscx108\fscy108\1c{accent}\3c{accent}\blur1.6)"
        + rf"\t(170,520,\fscx100\fscy100\1c{primary}\3c{outline}\blur0.4)"
        + "}"
    )


def _shadow_3d_dialogues(start: str, end: str, config: AssStyleConfig, text: str) -> list[str]:
    depth = max(0, min(36, int(config.shadow_3d_depth)))
    if not config.shadow_3d or depth <= 0:
        return []

    anchor_y = _text_anchor_y(config)
    color = _hex_to_ass_color(config.outline_color)
    outline = max(config.outline, int(round(config.font_size * 0.12)))
    step = max(2, depth // 4)
    offsets = sorted({step, depth // 2, depth}, reverse=True)
    lines = []
    for offset in offsets:
        tags = (
            "{"
            + rf"\an{config.alignment}\pos({540 + offset},{anchor_y + offset})"
            + rf"\fad(90,110)\bord{outline}\shad0\blur0.4\1c{color}\3c{color}"
            + "}"
        )
        lines.append(f"Dialogue: 0,{start},{end},AnimatedShorts,,0,0,0,,{tags}{text}")
    return lines


def _full_width_box_geometry(config: AssStyleConfig, height: int, half_w: int) -> tuple[int, int, str]:
    half_h = height // 2
    padding_shift = max(0, config.box_padding_px)
    text_anchor_y = _text_anchor_y(config)
    if config.alignment in (7, 8, 9):
        y = max(0, text_anchor_y - padding_shift)
        shape = f"m -{half_w} 0 l {half_w} 0 l {half_w} {height} l -{half_w} {height} l -{half_w} 0"
        return config.alignment, y, shape
    if config.alignment in (4, 5, 6):
        y = text_anchor_y
        shape = (
            f"m -{half_w} -{half_h} "
            f"l {half_w} -{half_h} "
            f"l {half_w} {half_h} "
            f"l -{half_w} {half_h} "
            f"l -{half_w} -{half_h}"
        )
        return config.alignment, y, shape
    y = min(1920, text_anchor_y + padding_shift)
    shape = f"m -{half_w} -{height} l {half_w} -{height} l {half_w} 0 l -{half_w} 0 l -{half_w} -{height}"
    return config.alignment, y, shape


def _full_width_box_dialogue(
    start: str,
    end: str,
    config: AssStyleConfig,
    line_count: int = 2,
) -> str:
    content_height = int(config.font_size * (1.18 * max(1, line_count)))
    height = max(78, content_height + config.box_padding_px * 2)
    color = _hex_to_ass_color(config.back_color)
    alpha = f"&H{max(0, min(255, config.back_alpha)):02X}&"
    bleed = 180
    half_w = 540 + bleed
    alignment, y, shape = _full_width_box_geometry(config, height, half_w)
    tags = rf"{{\p1\an{alignment}\pos(540,{y})\bord0\shad0\1c{color}\alpha{alpha}}}"
    return f"Dialogue: 0,{start},{end},AnimatedShorts,,0,0,0,,{tags}{shape}"


def write_animated_ass(
    segments: list[SubtitleSegment],
    output_path: Path,
    config: AssStyleConfig | None = None,
    word_timings_by_segment: list[list[dict]] | None = None,
) -> Path:
    """Write animated, phone-friendly ASS subtitles for FFmpeg hard-sub rendering."""
    active_config = config or AssStyleConfig()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [_ass_header(active_config)]

        for segment_index, segment in enumerate(segments):
            chunks = _caption_chunks(segment.text, active_config.max_chars, active_config.line_width)
            segment_word_timings = (
                word_timings_by_segment[segment_index]
                if word_timings_by_segment and segment_index < len(word_timings_by_segment)
                else []
            )
            chunk_word_cursor = 0
            if segment_word_timings:
                time_ranges: list[tuple[str, str]] = []
                for chunk in chunks:
                    word_count = len(_normalize_text(chunk).split())
                    chunk_words = segment_word_timings[chunk_word_cursor:chunk_word_cursor + word_count]
                    chunk_word_cursor += word_count
                    if chunk_words:
                        segment_start_ms = parse_srt_time(segment.start_time)
                        chunk_start_ms = segment_start_ms + int(chunk_words[0]["start_ms"])
                        chunk_end_ms = segment_start_ms + int(chunk_words[-1]["end_ms"])
                        time_ranges.append(
                            (_format_ass_time_from_ms(chunk_start_ms), _format_ass_time_from_ms(max(chunk_start_ms + 80, chunk_end_ms)))
                        )
                    else:
                        time_ranges.append((_format_ass_time(segment.start_time), _format_ass_time(segment.end_time)))
                chunk_word_cursor = 0
            else:
                time_ranges = _split_ass_times(segment.start_time, segment.end_time, len(chunks))

            for (start, end), chunk in zip(time_ranges, chunks):
                wrapped_lines = _caption_lines(chunk, active_config.line_width)
                if active_config.karaoke_words and segment_word_timings:
                    line_words_source = segment_word_timings[chunk_word_cursor:chunk_word_cursor + len(_normalize_text(chunk).split())]
                    chunk_word_cursor += len(_normalize_text(chunk).split())
                    remaining = line_words_source[:]
                    karaoke_lines: list[str] = []
                    for line in wrapped_lines:
                        line_word_count = len(_normalize_text(line).split())
                        karaoke_lines.append(_karaoke_text_from_word_timings(remaining[:line_word_count], line))
                        remaining = remaining[line_word_count:]
                    text = r"\N".join(karaoke_lines)
                elif active_config.karaoke_words:
                    karaoke_lines = [_karaoke_text(line, start, end) for line in wrapped_lines]
                    text = r"\N".join(karaoke_lines)
                else:
                    text = _escape_ass_text(r"\N".join(wrapped_lines))
                shadow_text = _escape_ass_text(r"\N".join(wrapped_lines))
                lines.extend(_shadow_3d_dialogues(start, end, active_config, shadow_text))
                animated_text = _animation_tags(active_config) + text
                lines.append(f"Dialogue: 1,{start},{end},AnimatedShorts,,0,0,0,,{animated_text}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
    except OSError as exc:
        raise RuntimeError(f"Could not write ASS file to {output_path}: {exc}") from exc
