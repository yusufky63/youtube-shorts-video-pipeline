from pathlib import Path

import pytest

from src.video_pipeline.ass_writer import AssStyleConfig, write_animated_ass
from src.video_pipeline.audio_sync import align_segments_to_audio, append_outro_segment
from src.video_pipeline.ffmpeg_renderer import (
    CensorRegionConfig,
    SourceCropConfig,
    VideoLayoutConfig,
    _build_subtitle_filter,
    _build_tts_mix_filter,
    _build_video_filter_chain,
)
from src.video_pipeline.json_parser import parse_gemini_segments
from src.video_pipeline.models import SubtitleSegment
from src.video_pipeline.narration_utils import fit_text_to_duration, normalize_segment_overlaps
from src.video_pipeline.timecode import format_srt_time, parse_srt_time
from src.video_pipeline.text_utils import remove_pause_punctuation, turkish_lower, turkish_upper
from src.video_pipeline.tts_generator import _extract_word_timings_from_alignment
from src.video_pipeline.youtube_metadata import generate_youtube_metadata


def test_parse_srt_time_round_trip():
    assert parse_srt_time("00:01:02,345") == 62345
    assert format_srt_time(62345) == "00:01:02,345"


def test_turkish_upper_preserves_dotted_i():
    assert turkish_upper("izlediğiniz için teşekkürler") == "İZLEDİĞİNİZ İÇİN TEŞEKKÜRLER"


def test_turkish_upper_preserves_dotted_i():
    assert turkish_upper("izledi\u011finiz i\u00e7in te\u015fekk\u00fcrler kap\u0131n\u0131n") == "\u0130ZLED\u0130\u011e\u0130N\u0130Z \u0130\u00c7\u0130N TE\u015eEKK\u00dcRLER KAPININ"


def test_turkish_lower_restores_dotless_i_for_tts():
    assert turkish_lower("KAPININ \u0130\u00c7\u0130N") == "kap\u0131n\u0131n i\u00e7in"


def test_remove_pause_punctuation_for_tts_flow():
    assert remove_pause_punctuation("KAPININ A\u00c7ILDI, SONRA DURDU.") == "KAPININ A\u00c7ILDI SONRA DURDU"


def test_generate_youtube_metadata_returns_copyable_shorts_fields():
    metadata = generate_youtube_metadata(
        [
            SubtitleSegment("00:00:00,000", "00:00:04,000", "Kahraman sahneye ciddi bir planla giriyor"),
            SubtitleSegment("00:00:04,000", "00:00:08,000", "Fakat zemin bu plani saygiyla reddediyor"),
            SubtitleSegment(
                "00:00:08,000",
                "00:00:11,000",
                "İzlediğiniz için teşekkürler. Beğenmeyi ve abone olmayı unutmayın.",
            ),
        ]
    )

    assert metadata["title"] == "Kahraman sahneye ciddi bir planla giriyor"
    assert "İzlediğiniz için teşekkürler" in metadata["description"]
    assert "shorts" in metadata["tags"]
    assert "teşekkürler" not in ",".join(metadata["tags"]).lower()


def test_parse_gemini_segments_accepts_markdown_json_block():
    segments = parse_gemini_segments(
        """```json
        [{"start_time":"00:00:00,000","end_time":"00:00:02,000","text":"Test"}]
        ```"""
    )

    assert len(segments) == 1
    assert segments[0].text == "Test"


def test_parse_gemini_segments_rejects_bad_order():
    with pytest.raises(ValueError):
        parse_gemini_segments(
            '[{"start_time":"00:00:03,000","end_time":"00:00:02,000","text":"Bad"}]'
        )


def test_animated_ass_splits_long_text_without_ellipsis():
    text = (
        "Kahramanimiz kontrolu tamamen ele aldim derken zemin fikrini degistiriyor "
        "ve belgesel ekibi sessizce olay yerinden uzaklasiyor."
    )
    output_path = write_animated_ass(
        [
            SubtitleSegment(
                start_time="00:00:00,000",
                end_time="00:00:06,000",
                text=text,
            )
        ],
        Path("video_pipeline_runs/_ass_writer_test/narration.ass"),
        AssStyleConfig(max_chars=45, line_width=24),
    )

    contents = output_path.read_text(encoding="utf-8")

    assert "..." not in contents
    assert "uzaklasiyor" in contents
    assert contents.count("Dialogue: 1") > 1


def test_animated_ass_can_emit_karaoke_tags():
    output_path = write_animated_ass(
        [
            SubtitleSegment(
                start_time="00:00:00,000",
                end_time="00:00:03,000",
                text="İzlediğiniz için teşekkürler",
            )
        ],
        Path("video_pipeline_runs/_ass_writer_test/karaoke.ass"),
        AssStyleConfig(karaoke_words=True),
    )

    contents = output_path.read_text(encoding="utf-8")

    assert r"\kf" in contents
    assert "teşekkürler" in contents


def test_animated_ass_uses_display_words_with_tts_timings():
    output_path = write_animated_ass(
        [
            SubtitleSegment(
                start_time="00:00:00,000",
                end_time="00:00:03,000",
                text="KAPININ A\u00c7ILMASI",
            )
        ],
        Path("video_pipeline_runs/_ass_writer_test/karaoke_display.ass"),
        AssStyleConfig(karaoke_words=True),
        word_timings_by_segment=[
            [
                {"text": "kap\u0131n\u0131n", "start_ms": 0, "end_ms": 600},
                {"text": "a\u00e7\u0131lmas\u0131", "start_ms": 600, "end_ms": 1300},
            ]
        ],
    )

    contents = output_path.read_text(encoding="utf-8")

    assert "KAPININ" in contents
    assert "kap\u0131n\u0131n" not in contents


def test_full_width_subtitle_box_is_not_written_even_when_requested():
    output_path = write_animated_ass(
        [
            SubtitleSegment(
                start_time="00:00:00,000",
                end_time="00:00:03,000",
                text="Arka plan kutusu yaziyla ayni hizayi kullanir",
            )
        ],
        Path("video_pipeline_runs/_ass_writer_test/full_width_box.ass"),
        AssStyleConfig(boxed_background=True, box_mode="full_width", alignment=2, margin_v=150),
    )

    contents = output_path.read_text(encoding="utf-8")

    assert r"\an2\pos(540,1770)" in contents
    assert "Dialogue: 0" not in contents
    assert r"\p1" not in contents


def test_full_width_subtitle_box_is_not_written_when_background_disabled():
    output_path = write_animated_ass(
        [
            SubtitleSegment(
                start_time="00:00:00,000",
                end_time="00:00:03,000",
                text="Arka plan kapaliysa finalde kutu cikmamali",
            )
        ],
        Path("video_pipeline_runs/_ass_writer_test/no_background_box.ass"),
        AssStyleConfig(boxed_background=False, box_mode="full_width", alignment=2, margin_v=150),
    )

    contents = output_path.read_text(encoding="utf-8")

    assert "Dialogue: 0" not in contents
    assert "Dialogue: 1" in contents


def test_animated_ass_can_emit_3d_shadow_layer():
    output_path = write_animated_ass(
        [
            SubtitleSegment(
                start_time="00:00:00,000",
                end_time="00:00:03,000",
                text="SIMDI ILE SERINLEMEK",
            )
        ],
        Path("video_pipeline_runs/_ass_writer_test/shadow_3d.ass"),
        AssStyleConfig(shadow_3d=True, shadow_3d_depth=14, outline=8),
    )

    contents = output_path.read_text(encoding="utf-8")

    assert "Dialogue: 0" in contents
    assert r"\pos(554,1784)" in contents
    assert "Dialogue: 1" in contents


def test_ass_header_never_uses_text_background_box():
    output_path = write_animated_ass(
        [
            SubtitleSegment(
                start_time="00:00:00,000",
                end_time="00:00:03,000",
                text="Yazi arkasi kutu tamamen kapali",
            )
        ],
        Path("video_pipeline_runs/_ass_writer_test/no_text_box.ass"),
        AssStyleConfig(boxed_background=True, box_mode="text", back_alpha=40),
    )

    contents = output_path.read_text(encoding="utf-8")
    style_line = next(line for line in contents.splitlines() if line.startswith("Style: AnimatedShorts"))

    assert ",1," in style_line
    assert "&HFF000000" in style_line


def test_fit_text_to_duration_shortens_long_line():
    fitted = fit_text_to_duration(
        "Kahramanimiz beklenenden uzun uzun anlatmaya devam ediyor ve tempo dusuyor.",
        duration_seconds=2.0,
        speed_multiplier=1.0,
        chars_per_second=12.0,
    )

    assert len(fitted) < len("Kahramanimiz beklenenden uzun uzun anlatmaya devam ediyor ve tempo dusuyor.")
    assert fitted


def test_normalize_segment_overlaps_pushes_second_segment_forward():
    segments = normalize_segment_overlaps(
        [
            SubtitleSegment("00:00:00,000", "00:00:02,000", "Ilk"),
            SubtitleSegment("00:00:01,500", "00:00:03,000", "Ikinci"),
        ],
        min_gap_ms=50,
    )

    assert segments[1].start_time == "00:00:02,050"


def test_align_segments_to_audio_uses_real_audio_length(monkeypatch):
    monkeypatch.setattr(
        "src.video_pipeline.audio_sync.read_media_duration_seconds",
        lambda _: 2.4,
    )

    segments = align_segments_to_audio(
        [
            SubtitleSegment("00:00:00,000", "00:00:01,000", "Ilk"),
            SubtitleSegment("00:00:01,050", "00:00:02,000", "Ikinci"),
        ],
        [Path("a.mp3"), Path("b.mp3")],
        min_gap_ms=100,
        gap_seconds=0.5,
        tail_padding_ms=0,
    )

    assert segments[0].end_time == "00:00:02,400"
    assert segments[1].start_time == "00:00:03,000"


def test_align_segments_to_audio_can_run_without_added_gap(monkeypatch):
    monkeypatch.setattr(
        "src.video_pipeline.audio_sync.read_media_duration_seconds",
        lambda _: 1.0,
    )

    segments = align_segments_to_audio(
        [
            SubtitleSegment("00:00:00,000", "00:00:01,000", "Ilk"),
            SubtitleSegment("00:00:01,000", "00:00:02,000", "Ikinci"),
        ],
        [Path("a.mp3"), Path("b.mp3")],
        min_gap_ms=0,
        gap_seconds=0.0,
        tail_padding_ms=0,
    )

    assert segments[0].end_time == "00:00:01,000"
    assert segments[1].start_time == "00:00:01,000"


def test_build_tts_mix_filter_uses_segment_starts_without_extra_gap():
    filter_text, label = _build_tts_mix_filter(
        [
            SubtitleSegment("00:00:00,500", "00:00:02,000", "Ilk"),
            SubtitleSegment("00:00:03,000", "00:00:04,000", "Ikinci"),
        ],
        2,
    )

    assert "adelay=500|500" in filter_text
    assert "adelay=3000|3000" in filter_text
    assert label == "tts_mix"


def test_video_filter_chain_can_crop_and_censor_before_subtitles():
    filter_text = _build_video_filter_chain(
        "subtitles='sample.ass'",
        VideoLayoutConfig(mode="fill_crop", crop_offset_y=-120),
        source_crop=SourceCropConfig(top=100, bottom=50, left=10, right=20),
        censor_regions=[
            CensorRegionConfig(
                start_time="00:00:01,000",
                end_time="00:00:03,000",
                x=800,
                y=60,
                width=200,
                height=90,
                blur=18,
            )
        ],
    )

    assert "crop=w='max(2,iw-10-20)'" in filter_text
    assert "crop=1080:1920:(iw-ow)/2:(ih-oh)/2+-120" in filter_text
    assert "boxblur=18:1" in filter_text
    assert "enable='between(t,1.000,3.000)'" in filter_text
    assert "[censored]subtitles='sample.ass'[subbed]" in filter_text


def test_align_segments_to_audio_moves_last_segment_earlier_to_fit_video(monkeypatch):
    durations = iter([2.4, 3.0])
    monkeypatch.setattr(
        "src.video_pipeline.audio_sync.read_media_duration_seconds",
        lambda _: next(durations),
    )

    segments = align_segments_to_audio(
        [
            SubtitleSegment("00:00:00,000", "00:00:01,000", "Ilk"),
            SubtitleSegment("00:00:05,500", "00:00:06,000", "Final"),
        ],
        [Path("a.mp3"), Path("b.mp3")],
        min_gap_ms=100,
        gap_seconds=0.0,
        tail_padding_ms=0,
        max_end_ms=6000,
    )

    assert segments[1].start_time == "00:00:03,000"
    assert segments[1].end_time == "00:00:06,000"


def test_append_outro_segment_uses_requested_duration():
    segments = append_outro_segment(
        [SubtitleSegment("00:00:00,000", "00:00:03,000", "Ana bolum")],
        video_duration_seconds=12.0,
        text="İzlediğiniz için teşekkürler. Beğenmeyi ve abone olmayı unutmayın.",
        duration_seconds=5.0,
    )

    assert segments[-1].start_time == "00:00:07,000"
    assert segments[-1].end_time == "00:00:12,000"


def test_extract_word_timings_from_alignment_groups_words_and_scales_speed():
    timings = _extract_word_timings_from_alignment(
        {
            "characters": list("Merhaba dunya"),
            "character_start_times_seconds": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            "character_end_times_seconds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
        },
        speed_multiplier=2.0,
    )

    assert timings[0]["text"] == "Merhaba"
    assert timings[0]["start_ms"] == 0
    assert timings[0]["end_ms"] == 400
    assert timings[1]["text"] == "dunya"


def test_build_subtitle_filter_can_include_fontsdir():
    fonts_dir = Path("video_pipeline_runs/fonts")
    fonts_dir.mkdir(parents=True, exist_ok=True)
    filter_text = _build_subtitle_filter(
        Path("video_pipeline_runs/sample.ass"),
        "FontName=Arial",
        fonts_dir=fonts_dir,
    )

    assert "fontsdir=" in filter_text
