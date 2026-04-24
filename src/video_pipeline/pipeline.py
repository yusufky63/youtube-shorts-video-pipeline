from dataclasses import dataclass
from pathlib import Path

from .ass_writer import write_animated_ass
from .ffmpeg_renderer import render_final_video
from .gemini_analyzer import DEFAULT_GEMINI_MODEL, analyze_video_with_gemini
from .srt_writer import write_srt
from .tts_generator import (
    DEFAULT_TTS_PITCH,
    DEFAULT_TTS_RATE,
    DEFAULT_TTS_VOICE,
    DEFAULT_TTS_VOLUME,
    generate_tts_files,
)


@dataclass(frozen=True)
class PipelineConfig:
    input_video: Path
    output_dir: Path
    output_filename: str = "final_output.mp4"
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_api_key: str | None = None
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_rate: str = DEFAULT_TTS_RATE
    tts_volume: str = DEFAULT_TTS_VOLUME
    tts_pitch: str = DEFAULT_TTS_PITCH
    original_volume: float = 0.0


def run_video_pipeline(config: PipelineConfig) -> Path:
    """Run Gemini analysis, SRT creation, TTS generation, and FFmpeg rendering."""
    try:
        input_video = config.input_video.resolve()
        output_dir = config.output_dir.resolve()
        tts_dir = output_dir / "tts"
        srt_path = output_dir / "narration.srt"
        ass_path = output_dir / "narration.ass"
        output_path = output_dir / config.output_filename

        output_dir.mkdir(parents=True, exist_ok=True)

        segments = analyze_video_with_gemini(
            video_path=input_video,
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
        )
        write_srt(segments, srt_path)
        write_animated_ass(segments, ass_path)
        tts_audio_paths = generate_tts_files(
            segments=segments,
            output_dir=tts_dir,
            voice=config.tts_voice,
            rate=config.tts_rate,
            volume=config.tts_volume,
            pitch=config.tts_pitch,
        )

        return render_final_video(
            input_video=input_video,
            srt_path=ass_path,
            tts_audio_paths=tts_audio_paths,
            segments=segments,
            output_path=output_path,
            original_volume=config.original_volume,
        )
    except Exception as exc:
        raise RuntimeError(f"Video pipeline failed: {exc}") from exc
