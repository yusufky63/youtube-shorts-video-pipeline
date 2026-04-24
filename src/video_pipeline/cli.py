import argparse
from pathlib import Path

from .gemini_analyzer import DEFAULT_GEMINI_MODEL
from .pipeline import PipelineConfig, run_video_pipeline
from .tts_generator import DEFAULT_TTS_PITCH, DEFAULT_TTS_RATE, DEFAULT_TTS_VOICE, DEFAULT_TTS_VOLUME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a video with Gemini, generate documentary TTS, and render hard subtitles."
    )
    parser.add_argument("input_video", type=Path, help="Path to the source .mp4 video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("video_pipeline_runs"),
        help="Directory for SRT, TTS clips, and final output",
    )
    parser.add_argument(
        "--output-name",
        default="final_output.mp4",
        help="Final rendered MP4 filename",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API key. If omitted, GEMINI_API_KEY is used.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_TTS_VOICE,
        help="Edge TTS voice, for example tr-TR-AhmetNeural",
    )
    parser.add_argument("--rate", default=DEFAULT_TTS_RATE, help="Edge TTS rate")
    parser.add_argument("--volume", default=DEFAULT_TTS_VOLUME, help="Edge TTS volume")
    parser.add_argument("--pitch", default=DEFAULT_TTS_PITCH, help="Edge TTS pitch")
    parser.add_argument(
        "--original-volume",
        type=float,
        default=0.0,
        help="Original audio volume multiplier before mixing with TTS. Use 0 to remove original audio.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = PipelineConfig(
            input_video=args.input_video,
            output_dir=args.output_dir,
            output_filename=args.output_name,
            gemini_model=args.model,
            gemini_api_key=args.api_key,
            tts_voice=args.voice,
            tts_rate=args.rate,
            tts_volume=args.volume,
            tts_pitch=args.pitch,
            original_volume=args.original_volume,
        )
        output_path = run_video_pipeline(config)
        print(f"Final video created: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
