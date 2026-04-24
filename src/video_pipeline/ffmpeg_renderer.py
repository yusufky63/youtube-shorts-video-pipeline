import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import SubtitleSegment
from .timecode import parse_srt_time


SUBTITLE_STYLE = (
    "FontName=Arial,"
    "FontSize=24,"
    "PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,"
    "BackColour=&H80000000,"
    "BorderStyle=1,"
    "Outline=2,"
    "Shadow=1,"
    "Alignment=2,"
    "MarginV=48"
)


@dataclass(frozen=True)
class VideoLayoutConfig:
    mode: str = "source"
    width: int = 1080
    height: int = 1920
    blur_sigma: int = 24
    crop_offset_y: int = 0


@dataclass(frozen=True)
class SourceCropConfig:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0


@dataclass(frozen=True)
class CensorRegionConfig:
    start_time: str = "00:00:00,000"
    end_time: str = "00:00:05,000"
    x: int = 0
    y: int = 0
    width: int = 200
    height: int = 100
    blur: int = 16
    enabled: bool = True


@dataclass(frozen=True)
class LogoOverlayConfig:
    path: Path
    scale_percent: int = 12
    opacity: float = 1.0
    anchor: str = "Top Right"
    offset_x: int = 36
    offset_y: int = 36


def resolve_ffmpeg_binary() -> str:
    """Return a usable FFmpeg executable from PATH or imageio-ffmpeg."""
    env_binary = shutil.which("ffmpeg")
    if env_binary:
        return env_binary

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg is not installed or available. Install FFmpeg into PATH, "
            "or install the Python fallback with: pip install imageio-ffmpeg"
        ) from exc


def _run_command(command: list[str], label: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{label} failed because FFmpeg is not installed or not available"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"{label} failed: {stderr}") from exc


def ensure_ffmpeg_available() -> None:
    """Fail early if FFmpeg cannot be executed."""
    _run_command([resolve_ffmpeg_binary(), "-version"], "FFmpeg check")


def _read_media_info(video_path: Path) -> str:
    """Read FFmpeg input metadata from stderr without requiring ffprobe."""
    ffmpeg = resolve_ffmpeg_binary()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return f"{result.stdout}\n{result.stderr}"


def video_has_audio_stream(video_path: Path) -> bool:
    """Return whether the input video contains an audio stream."""
    info = _read_media_info(video_path)
    return bool(re.search(r"Stream #\d+:\d+.*Audio:", info))


def get_video_duration_seconds(video_path: Path) -> float:
    """Read the container duration from FFmpeg input metadata."""
    info = _read_media_info(video_path)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if not match:
        raise RuntimeError(f"Could not read video duration for {video_path}")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _escape_subtitles_path(path: Path) -> str:
    # FFmpeg subtitles filter uses a filter expression, so Windows drive colons
    # and backslashes need escaping even though subprocess receives argv safely.
    escaped = path.resolve().as_posix()
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    return escaped


def _build_subtitle_filter(subtitle_path: Path, subtitle_style: str, fonts_dir: Path | None = None) -> str:
    escaped_path = _escape_subtitles_path(subtitle_path)
    fonts_suffix = ""
    if fonts_dir and fonts_dir.exists():
        fonts_suffix = f":fontsdir='{_escape_subtitles_path(fonts_dir)}'"
    if subtitle_path.suffix.lower() == ".ass":
        return f"subtitles='{escaped_path}'{fonts_suffix}"
    return f"subtitles='{escaped_path}'{fonts_suffix}:force_style='{subtitle_style}'"


def _build_tts_mix_filter(
    segments: list[SubtitleSegment],
    audio_count: int,
) -> tuple[str, str]:
    delayed_labels: list[str] = []
    filters: list[str] = []

    for index, segment in enumerate(segments, start=1):
        start_ms = parse_srt_time(segment.start_time)
        input_index = index
        label = f"tts{index}"
        filters.append(
            f"[{input_index}:a]asetpts=PTS-STARTPTS,adelay={start_ms}|{start_ms},volume=1.0[{label}]"
        )
        delayed_labels.append(f"[{label}]")

    mix_label = "tts_mix"
    filters.append(
        f"{''.join(delayed_labels)}amix=inputs={audio_count}:duration=longest:dropout_transition=0[{mix_label}]"
    )
    return ";".join(filters), mix_label


def _build_video_filter_chain(
    subtitle_filter: str,
    layout: VideoLayoutConfig | None = None,
    source_crop: SourceCropConfig | None = None,
    censor_regions: list[CensorRegionConfig] | None = None,
    logo_input_index: int | None = None,
    logo_config: LogoOverlayConfig | None = None,
) -> str:
    active_layout = layout or VideoLayoutConfig()
    width = active_layout.width
    height = active_layout.height
    crop = source_crop or SourceCropConfig()
    crop_prefix = ""
    input_label = "[0:v]"
    if any(value > 0 for value in (crop.top, crop.bottom, crop.left, crop.right)):
        crop_width = f"max(2,iw-{max(0, crop.left)}-{max(0, crop.right)})"
        crop_height = f"max(2,ih-{max(0, crop.top)}-{max(0, crop.bottom)})"
        crop_prefix = (
            f"[0:v]crop=w='{crop_width}':h='{crop_height}':"
            f"x={max(0, crop.left)}:y={max(0, crop.top)}[source_cropped];"
        )
        input_label = "[source_cropped]"
    crop_offset_y = active_layout.crop_offset_y

    if active_layout.mode == "fit_blur":
        bg_input = input_label
        fg_input = input_label
        if crop_prefix:
            crop_prefix += f"{input_label}split[source_bg][source_fg];"
            bg_input = "[source_bg]"
            fg_input = "[source_fg]"
        base_chain = (
            f"{crop_prefix}"
            f"{bg_input}scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},gblur=sigma={active_layout.blur_sigma}[bg];"
            f"{fg_input}scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[base];"
        )
    elif active_layout.mode == "fit_black":
        base_chain = (
            f"{crop_prefix}"
            f"{input_label}scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p[base];"
        )
    elif active_layout.mode == "fill_crop":
        base_chain = (
            f"{crop_prefix}"
            f"{input_label}scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2+{crop_offset_y},format=yuv420p[base];"
        )
    else:
        base_chain = f"{crop_prefix}{input_label}format=yuv420p[base];"

    chain = base_chain + _build_censor_filter_chain("[base]", "[censored]", censor_regions, width, height)
    chain += f";[censored]{subtitle_filter}[subbed]"

    if logo_input_index is None or logo_config is None or not logo_config.path.exists():
        return chain + ";[subbed]copy[final_video]"

    logo_width = max(48, int(width * max(4, logo_config.scale_percent) / 100))
    opacity = max(0.0, min(1.0, logo_config.opacity))
    if logo_config.anchor == "Top Left":
        x_expr = str(max(0, logo_config.offset_x))
        y_expr = str(max(0, logo_config.offset_y))
    elif logo_config.anchor == "Top Center":
        x_expr = f"(W-w)/2+{logo_config.offset_x}"
        y_expr = str(max(0, logo_config.offset_y))
    elif logo_config.anchor == "Bottom Left":
        x_expr = str(max(0, logo_config.offset_x))
        y_expr = f"H-h-{max(0, logo_config.offset_y)}"
    elif logo_config.anchor == "Bottom Center":
        x_expr = f"(W-w)/2+{logo_config.offset_x}"
        y_expr = f"H-h-{max(0, logo_config.offset_y)}"
    elif logo_config.anchor == "Bottom Right":
        x_expr = f"W-w-{max(0, logo_config.offset_x)}"
        y_expr = f"H-h-{max(0, logo_config.offset_y)}"
    else:
        x_expr = f"W-w-{max(0, logo_config.offset_x)}"
        y_expr = str(max(0, logo_config.offset_y))

    return (
        chain
        + f";[{logo_input_index}:v]format=rgba,colorchannelmixer=aa={opacity:.3f},scale={logo_width}:-1[logo]"
        + f";[subbed][logo]overlay={x_expr}:{y_expr}:format=auto[final_video]"
    )


def _build_censor_filter_chain(
    input_label: str,
    output_label: str,
    censor_regions: list[CensorRegionConfig] | None,
    output_width: int,
    output_height: int,
) -> str:
    active_regions = [region for region in (censor_regions or []) if region.enabled]
    if not active_regions:
        return f"{input_label}copy{output_label}"

    parts: list[str] = []
    current = input_label
    for index, region in enumerate(active_regions):
        x = max(0, min(output_width - 2, int(region.x)))
        y = max(0, min(output_height - 2, int(region.y)))
        region_width = max(2, min(output_width - x, int(region.width)))
        region_height = max(2, min(output_height - y, int(region.height)))
        blur = max(2, min(40, int(region.blur)))
        start_seconds = parse_srt_time(region.start_time) / 1000
        end_seconds = max(start_seconds + 0.05, parse_srt_time(region.end_time) / 1000)
        main_label = f"censor_main_{index}"
        crop_label = f"censor_crop_{index}"
        blur_label = f"censor_blur_{index}"
        next_label = output_label.strip("[]") if index == len(active_regions) - 1 else f"censor_out_{index}"
        parts.append(
            f"{current}split[{main_label}][{crop_label}];"
            f"[{crop_label}]crop={region_width}:{region_height}:{x}:{y},boxblur={blur}:1[{blur_label}];"
            f"[{main_label}][{blur_label}]overlay={x}:{y}:enable='between(t,{start_seconds:.3f},{end_seconds:.3f})'[{next_label}]"
        )
        current = f"[{next_label}]"
    return ";".join(parts)


def render_final_video(
    input_video: Path,
    srt_path: Path,
    tts_audio_paths: list[Path],
    segments: list[SubtitleSegment],
    output_path: Path,
    original_volume: float = 0.0,
    subtitle_style: str = SUBTITLE_STYLE,
    tts_gap_seconds: float = 0.0,
    video_layout: VideoLayoutConfig | None = None,
    subtitle_fonts_dir: Path | None = None,
    logo_overlay: LogoOverlayConfig | None = None,
    source_crop: SourceCropConfig | None = None,
    censor_regions: list[CensorRegionConfig] | None = None,
) -> Path:
    """Render the final MP4 with TTS, optional original audio, and hard subtitles."""
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not srt_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
    if len(tts_audio_paths) != len(segments):
        raise ValueError("TTS audio count must match segment count")

    ensure_ffmpeg_available()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_original_audio = video_has_audio_stream(input_video)
    keep_original_audio = has_original_audio and original_volume > 0

    ffmpeg = resolve_ffmpeg_binary()
    command = [ffmpeg, "-y", "-i", str(input_video)]
    for audio_path in tts_audio_paths:
        if not audio_path.exists():
            raise FileNotFoundError(f"TTS audio file not found: {audio_path}")
        command.extend(["-i", str(audio_path)])
    logo_input_index: int | None = None
    if logo_overlay and logo_overlay.path.exists():
        command.extend(["-i", str(logo_overlay.path)])
        logo_input_index = len(tts_audio_paths) + 1

    if not keep_original_audio:
        duration = get_video_duration_seconds(input_video)
        command.extend(
            [
                "-f",
                "lavfi",
                "-t",
                f"{duration:.3f}",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=44100",
            ]
        )

    tts_filter, tts_mix_label = _build_tts_mix_filter(
        segments,
        len(tts_audio_paths),
    )
    subtitle_filter = _build_subtitle_filter(srt_path, subtitle_style, subtitle_fonts_dir)
    video_filter = _build_video_filter_chain(
        subtitle_filter,
        video_layout,
        source_crop=source_crop,
        censor_regions=censor_regions,
        logo_input_index=logo_input_index,
        logo_config=logo_overlay,
    )

    if keep_original_audio:
        filter_complex = (
            f"{tts_filter};"
            f"[0:a]volume={original_volume}[original_audio];"
            f"[original_audio][{tts_mix_label}]amix=inputs=2:duration=first:dropout_transition=0[final_audio];"
            f"{video_filter}"
        )
    else:
        silent_input_index = len(tts_audio_paths) + 1 + (1 if logo_input_index is not None else 0)
        filter_complex = (
            f"{tts_filter};"
            f"[{silent_input_index}:a][{tts_mix_label}]amix=inputs=2:duration=first:dropout_transition=0[final_audio];"
            f"{video_filter}"
        )

    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[final_video]",
            "-map",
            "[final_audio]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    )

    _run_command(command, "Final video render")
    return output_path
