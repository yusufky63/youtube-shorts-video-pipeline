import json
import os
import re
import shutil
import sys
import base64
import subprocess
import requests
from textwrap import wrap
from html import escape
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.video_pipeline.ass_writer import AssStyleConfig, style_for_position, write_animated_ass
from src.video_pipeline.audio_sync import align_segments_to_audio, append_outro_segment
from src.video_pipeline.ffmpeg_renderer import (
    CensorRegionConfig,
    LogoOverlayConfig,
    SourceCropConfig,
    SUBTITLE_STYLE,
    VideoLayoutConfig,
    get_video_duration_seconds,
    render_final_video,
    resolve_ffmpeg_binary,
)
from src.video_pipeline.gemini_analyzer import (
    ANALYSIS_PROMPT,
    DEFAULT_GEMINI_MODEL,
    analyze_video_with_gemini,
    detect_censor_regions_with_gemini,
    list_generate_content_models,
    normalize_model_name,
)
from src.video_pipeline.json_parser import validate_segments
from src.video_pipeline.models import SubtitleSegment
from src.video_pipeline.narration_utils import (
    estimate_speech_seconds,
    fit_text_to_duration,
    has_overlaps,
    normalize_segment_overlaps,
)
from src.video_pipeline.srt_writer import write_srt
from src.video_pipeline.timecode import parse_srt_time
from src.video_pipeline.text_utils import remove_pause_punctuation, turkish_upper
from src.video_pipeline.tts_generator import (
    DEFAULT_TTS_PITCH,
    DEFAULT_ELEVENLABS_MODEL,
    DEFAULT_TTS_RATE,
    DEFAULT_TTS_VOICE,
    DEFAULT_TTS_VOLUME,
    generate_tts_files,
    generate_tts_package,
    list_elevenlabs_voices,
)
from src.video_pipeline.youtube_metadata import generate_youtube_metadata, write_youtube_metadata


RUNS_DIR = ROOT_DIR / "video_pipeline_runs"
PREVIEW_DIR = RUNS_DIR / "_voice_previews"
SETTINGS_PATH = RUNS_DIR / "_saved_ui_settings.json"
API_KEYS_PATH = RUNS_DIR / "_saved_api_keys.json"
VOICE_PREVIEW_TEXT = "Kahramanimiz sahneye giriyor. Doganin dengesi, bir anligina kararsiz kaliyor."
DEFAULT_ELEVENLABS_VOICE_NAME = "Arman Yılmazkurt - Upbeat and Casual"
DEFAULT_ELEVENLABS_VOICE_ID = "qUGjOGoUuQBTUqwY8n0e"
FONT_FAMILY_OPTIONS = [
    "Inter",
    "Luckiest Guy",
    "Anton",
    "Bebas Neue",
    "Bangers",
    "Titan One",
    "Lilita One",
    "Burbank Big Condensed Black",
    "Komika Axis",
    "Proxima Nova",
    "SF Pro Display",
    "Roboto",
    "Arial",
]
FREE_FONT_DOWNLOADS = {
    "Luckiest Guy": {
        "family": "Luckiest Guy",
        "file": "LuckiestGuy-Regular.ttf",
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/luckiestguy/LuckiestGuy-Regular.ttf",
    },
    "Anton": {
        "family": "Anton",
        "file": "Anton-Regular.ttf",
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    },
    "Bebas Neue": {
        "family": "Bebas Neue",
        "file": "BebasNeue-Regular.ttf",
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    },
    "Bangers": {
        "family": "Bangers",
        "file": "Bangers-Regular.ttf",
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/bangers/Bangers-Regular.ttf",
    },
    "Titan One": {
        "family": "Titan One",
        "file": "TitanOne-Regular.ttf",
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/titanone/TitanOne-Regular.ttf",
    },
    "Lilita One": {
        "family": "Lilita One",
        "file": "LilitaOne-Regular.ttf",
        "url": "https://raw.githubusercontent.com/google/fonts/main/ofl/lilitaone/LilitaOne-Regular.ttf",
    },
}

RECOMMENDED_VOICES = [
    "tr-TR-AhmetNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-US-ChristopherNeural",
    "en-US-GuyNeural",
    "en-US-SteffanNeural",
    "en-GB-RyanNeural",
    "en-GB-ThomasNeural",
]

SUBTITLE_PRESETS = {
    "Documentary": SUBTITLE_STYLE,
    "Shorts Bold": (
        "FontName=Arial,"
        "FontSize=30,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=3,"
        "Shadow=1,"
        "Alignment=2,"
        "MarginV=68"
    ),
    "Minimal": (
        "FontName=Arial,"
        "FontSize=22,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=1,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=44"
    ),
    "Pycaps Hype": (
        "FontName=Luckiest Guy,"
        "FontSize=34,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=5,"
        "Shadow=2,"
        "Alignment=2,"
        "MarginV=80"
    ),
    "3D Block": (
        "FontName=Anton,"
        "FontSize=34,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=6,"
        "Shadow=3,"
        "Alignment=2,"
        "MarginV=82"
    ),
}

SUBTITLE_UI_PRESETS = {
    "Documentary": {
        "subtitle_font_family": "Inter",
        "subtitle_font_weight": 900,
        "subtitle_font_size": 64,
        "subtitle_color": "#FFFFFF",
        "subtitle_accent": "#FFE45C",
        "outline_color": "#000000",
        "boxed_background": False,
        "box_mode": "full_width",
        "box_alpha": 148,
        "box_padding_px": 38,
        "glow_subtitle": True,
        "karaoke_words": True,
        "subtitle_max_chars": 84,
        "line_width": 24,
    },
    "Shorts Bold": {
        "subtitle_font_family": "Inter",
        "subtitle_font_weight": 900,
        "subtitle_font_size": 72,
        "subtitle_color": "#FFFFFF",
        "subtitle_accent": "#FFD84D",
        "outline_color": "#000000",
        "boxed_background": False,
        "box_mode": "full_width",
        "box_alpha": 135,
        "box_padding_px": 44,
        "glow_subtitle": True,
        "karaoke_words": True,
        "subtitle_max_chars": 70,
        "line_width": 20,
    },
    "Minimal": {
        "subtitle_font_family": "Arial",
        "subtitle_font_weight": 800,
        "subtitle_font_size": 54,
        "subtitle_color": "#FFFFFF",
        "subtitle_accent": "#FFFFFF",
        "outline_color": "#000000",
        "boxed_background": False,
        "box_mode": "text",
        "box_alpha": 170,
        "box_padding_px": 24,
        "glow_subtitle": False,
        "karaoke_words": False,
        "subtitle_max_chars": 92,
        "line_width": 28,
    },
    "Pycaps Hype": {
        "subtitle_motion": "Animated pulse",
        "subtitle_font_family": "Luckiest Guy",
        "subtitle_font_weight": 900,
        "subtitle_font_size": 78,
        "subtitle_color": "#FFFFFF",
        "subtitle_accent": "#FFE600",
        "outline_color": "#000000",
        "boxed_background": False,
        "box_mode": "text",
        "box_alpha": 0,
        "box_padding_px": 0,
        "glow_subtitle": True,
        "karaoke_words": True,
        "subtitle_max_chars": 48,
        "line_width": 18,
    },
    "3D Block": {
        "subtitle_motion": "Animated pulse",
        "subtitle_font_family": "Anton",
        "subtitle_font_weight": 900,
        "subtitle_font_size": 78,
        "subtitle_color": "#F4F7FA",
        "subtitle_accent": "#00BDEB",
        "outline_color": "#050505",
        "boxed_background": False,
        "box_mode": "text",
        "box_alpha": 0,
        "box_padding_px": 0,
        "glow_subtitle": True,
        "karaoke_words": True,
        "subtitle_3d_shadow": True,
        "subtitle_3d_depth": 14,
        "subtitle_max_chars": 44,
        "line_width": 18,
    },
}

PROMPT_PRESETS = {
    "Deadpan Documentary": ANALYSIS_PROMPT,
    "Shorts Comedy": """
Bu videoyu analiz et ve Turkce, kisa, ritimli bir Shorts seslendirmesi yaz.
Ton: ciddi belgesel anlatimi, absurt ama kontrollu mizah.
Sadece JSON don:
[{"start_time": "00:00:00,000", "end_time": "00:00:05,000", "text": "Cumle..."}, ...]

Kurallar:
- Ekranda olmayan hicbir seyi uydurma.
- Segmentler 2.5-6 saniye araliginda olsun.
- Her text en fazla 95 karakter olsun.
- Tek kisa cumle yaz; aciklama uzatma.
- Cumleler dogal Turkiye Turkcesiyle yazilsin; bozuk ceviri gibi duran kaliplar kullanma.
- "hesaplama hatasi payi yok" gibi anlamsiz ifadeler yazma.
- Ozne, fiil ve nesne uyumlu olsun; cumle sesli okununca normal dursun.
- Metinler tek bir akici hikaye gibi ilerlesin; segmentler birbirinden kopuk durmasin.
- Ayni ana ozneyi takip et ve onceki cumlenin sonucunu sonraki cumleye tasi.
- Gecisleri dogal kur: "bu sirada", "ardindan", "fakat", "derken" gibi baglaclari gerektikce kullan.
- Her segmentte yeni saka kurma; mizah olaylar biriktikce ortaya ciksin.
- Emoji, hashtag, kufur, kaba hakaret ve abartili internet argosu kullanma.
- Metin sesli okundugunda dogal dursun.
""".strip(),
    "Dry Roast": """
Videoyu izle ve Turkce, kuru mizahli bir anlatim yaz.
Ton: sakin belgesel anlatani, olaylara hafif alayci ama temiz yorum yapar.
Sadece JSON don:
[{"start_time": "00:00:00,000", "end_time": "00:00:05,000", "text": "Cumle..."}, ...]

Kurallar:
- Kisiye saldirma; komedi davranisin absurtlugunden gelsin.
- Ekranda olmayan niyet, isim, marka veya hikaye uydurma.
- Segmentler 3-6 saniye olsun.
- Her text en fazla 95 karakter olsun.
- Tek kisa cumle kullan.
- Cumleler dogal Turkiye Turkcesiyle yazilsin; yapay ceviri gibi duran kaliplar kullanma.
- "hesaplama hatasi payi yok" gibi anlamsiz veya bozuk ifadeler yazma.
- Ozne, fiil ve nesne uyumlu olsun; cumle sesli okundugunda normal dursun.
- Metinler tek bir kuru mizahli belgesel akisi gibi baglansin.
- Ayni ana ozneyi ve olay cizgisini koru; her segment oncekinin devami gibi hissedilsin.
- Baglaclari gerektikce kullan ama ayni kalibi surekli tekrar etme.
- Her segmenti ayri punchline gibi yazma; final hissi sonlara dogru gelsin.
""".strip(),
}


SETTINGS_PROFILE_VERSION = 6
RECOMMENDED_UI_SETTINGS = {
    "settings_profile_version": SETTINGS_PROFILE_VERSION,
    "analysis_auto_fit": False,
    "timing_density": 17.0,
    "prevent_overlap": True,
    "min_gap_ms": 0,
    "playback_speed": 1.05,
    "auto_fit_tts_to_segment": True,
    "max_auto_tts_speed": 1.24,
    "original_volume": 0.0,
    "tts_gap_seconds": 0.0,
    "subtitle_motion": "Single layer",
    "subtitle_preset": "Documentary",
    "subtitle_position": "Bottom",
    "subtitle_vertical_offset": 0,
    "subtitle_font_size": 62,
    "subtitle_font_family": "Inter",
    "subtitle_font_weight": 900,
    "custom_font_family_name": "",
    "subtitle_bold": True,
    "subtitle_color": "#FFFFFF",
    "subtitle_accent": "#FFE45C",
    "outline_color": "#000000",
    "subtitle_3d_shadow": False,
    "subtitle_3d_depth": 0,
    "boxed_background": False,
    "box_mode": "full_width",
    "box_alpha": 150,
    "box_padding_px": 34,
    "glow_subtitle": True,
    "karaoke_words": True,
    "subtitle_max_chars": 84,
    "line_width": 24,
    "layout_label": "Fit 9:16 with blurred background",
    "blur_sigma": 24,
    "crop_top": 0,
    "crop_bottom": 0,
    "crop_left": 0,
    "crop_right": 0,
    "crop_offset_y": 0,
    "censor_regions": [],
    "logo_enabled": False,
    "logo_scale_percent": 12,
    "logo_opacity": 0.95,
    "logo_anchor": "Top Right",
    "logo_offset_x": 36,
    "logo_offset_y": 36,
    "elevenlabs_stability": 0.68,
    "elevenlabs_similarity_boost": 0.86,
    "elevenlabs_style": 0.08,
    "elevenlabs_speaker_boost": True,
    "selected_elevenlabs_voice_id": DEFAULT_ELEVENLABS_VOICE_ID,
}


def init_state() -> None:
    saved_settings = load_saved_settings()
    saved_api_keys = load_saved_api_keys()
    defaults = {
        "job_dir": None,
        "input_video": None,
        "segments": [],
        "srt_path": None,
        "ass_path": None,
        "tts_audio_paths": [],
        "word_timings": [],
        "final_output": None,
        "publish_metadata": None,
        "logs": [],
        "available_models": [],
        "voice_preview_path": None,
        "prompt_preset": "Deadpan Documentary",
        "edge_voice_names": RECOMMENDED_VOICES,
        "selected_voice": DEFAULT_TTS_VOICE,
        "elevenlabs_voices": [],
        "selected_elevenlabs_voice_id": DEFAULT_ELEVENLABS_VOICE_ID,
        "last_uploaded_signature": "",
        "editor_version": 0,
        "ui_language": "TR",
        "ui_theme": "Auto",
        "preview_segment_index": 0,
        "custom_prompt": ANALYSIS_PROMPT,
        "gemini_api_key": saved_api_keys.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", ""),
        "elevenlabs_api_key": saved_api_keys.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY", ""),
        "cta_enabled": True,
        "cta_text": "İzlediğiniz için teşekkürler. Beğenmeyi ve abone olmayı unutmayın.",
        "karaoke_words": True,
        "subtitle_font_family": "Inter",
        "subtitle_font_weight": 900,
        "custom_font_path": "",
        "custom_font_family_name": "",
        "subtitle_3d_shadow": False,
        "subtitle_3d_depth": 0,
        "logo_enabled": False,
        "logo_path": "",
        "censor_regions": [],
        "logo_scale_percent": 12,
        "logo_opacity": 0.95,
        "logo_anchor": "Top Right",
        "logo_offset_x": 36,
        "logo_offset_y": 36,
        "elevenlabs_model_id": DEFAULT_ELEVENLABS_MODEL,
        "elevenlabs_stability": 0.68,
        "elevenlabs_similarity_boost": 0.86,
        "elevenlabs_style": 0.08,
        "elevenlabs_speaker_boost": True,
    }
    defaults.update(RECOMMENDED_UI_SETTINGS)
    defaults.update(saved_settings)
    if saved_settings.get("settings_profile_version") != SETTINGS_PROFILE_VERSION:
        defaults.update(RECOMMENDED_UI_SETTINGS)
        defaults["custom_prompt"] = ANALYSIS_PROMPT
    defaults["box_blur"] = 0
    defaults["boxed_background"] = False
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_saved_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_saved_api_keys() -> dict:
    if not API_KEYS_PATH.exists():
        return {}
    try:
        payload = json.loads(API_KEYS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "gemini_api_key": str(payload.get("gemini_api_key", "") or "").strip(),
        "elevenlabs_api_key": str(payload.get("elevenlabs_api_key", "") or "").strip(),
    }


def save_api_keys(gemini_api_key: str | None, elevenlabs_api_key: str | None) -> bool:
    payload = {}
    if gemini_api_key and gemini_api_key.strip():
        payload["gemini_api_key"] = gemini_api_key.strip()
    if elevenlabs_api_key and elevenlabs_api_key.strip():
        payload["elevenlabs_api_key"] = elevenlabs_api_key.strip()
    if not payload:
        return False
    API_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    API_KEYS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    st.session_state.gemini_api_key = payload.get("gemini_api_key", "")
    st.session_state.elevenlabs_api_key = payload.get("elevenlabs_api_key", "")
    return True


def clear_saved_api_keys() -> None:
    if API_KEYS_PATH.exists():
        API_KEYS_PATH.unlink()
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")


def mask_secret(value: str | None) -> str:
    value = (value or "").strip()
    if not value:
        return "Kayitli degil"
    if len(value) <= 8:
        return "********"
    return f"{value[:4]}...{value[-4:]}"


def save_ui_settings(settings: dict) -> None:
    allowed_keys = [
        "settings_profile_version",
        "ui_theme",
        "prompt_preset",
        "custom_prompt",
        "playback_speed",
        "auto_fit_tts_to_segment",
        "max_auto_tts_speed",
        "original_volume",
        "analysis_auto_fit",
        "timing_density",
        "prevent_overlap",
        "min_gap_ms",
        "subtitle_motion",
        "subtitle_preset",
        "subtitle_position",
        "subtitle_vertical_offset",
        "subtitle_font_size",
        "subtitle_font_family",
        "subtitle_font_weight",
        "custom_font_family_name",
        "subtitle_bold",
        "subtitle_color",
        "subtitle_accent",
        "outline_color",
        "subtitle_3d_shadow",
        "subtitle_3d_depth",
        "boxed_background",
        "box_mode",
        "box_alpha",
        "box_padding_px",
        "glow_subtitle",
        "karaoke_words",
        "subtitle_max_chars",
        "line_width",
        "layout_label",
        "blur_sigma",
        "crop_top",
        "crop_bottom",
        "crop_left",
        "crop_right",
        "crop_offset_y",
        "censor_regions",
        "tts_gap_seconds",
        "output_name",
        "cta_enabled",
        "cta_text",
        "custom_font_path",
        "logo_enabled",
        "logo_path",
        "logo_scale_percent",
        "logo_opacity",
        "logo_anchor",
        "logo_offset_x",
        "logo_offset_y",
        "elevenlabs_model_id",
        "selected_elevenlabs_voice_id",
        "elevenlabs_stability",
        "elevenlabs_similarity_boost",
        "elevenlabs_style",
        "elevenlabs_speaker_boost",
    ]
    payload = {key: settings[key] for key in allowed_keys if key in settings}
    payload["settings_profile_version"] = SETTINGS_PROFILE_VERSION
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_recommended_settings() -> None:
    for key, value in RECOMMENDED_UI_SETTINGS.items():
        st.session_state[key] = value
    st.session_state["box_blur"] = 0
    st.session_state["boxed_background"] = False
    save_ui_settings(RECOMMENDED_UI_SETTINGS)
    append_log("Recommended settings applied.")


def append_log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")


def slugify_filename(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "video"


def create_job_dir(original_name: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = RUNS_DIR / f"{stamp}-{slugify_filename(original_name)}"
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "tts").mkdir(exist_ok=True)
    (job_dir / "assets" / "fonts").mkdir(parents=True, exist_ok=True)
    (job_dir / "assets" / "logos").mkdir(parents=True, exist_ok=True)
    return job_dir


def save_job_asset(uploaded_file, target_dir: Path, filename: str | None = None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(uploaded_file.name).suffix
    safe_name = filename or f"{slugify_filename(Path(uploaded_file.name).stem)}{extension}"
    asset_path = target_dir / safe_name
    uploaded_file.seek(0)
    with asset_path.open("wb") as target:
        shutil.copyfileobj(uploaded_file, target)
    return asset_path


def download_free_font(font_label: str, job_dir: Path) -> tuple[Path, str]:
    font_info = FREE_FONT_DOWNLOADS[font_label]
    target_dir = job_dir / "assets" / "fonts"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / font_info["file"]
    response = requests.get(font_info["url"], timeout=30)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path, font_info["family"]


def metadata_path(job_dir: Path) -> Path:
    return job_dir / "metadata.json"


def write_metadata(job_dir: Path, **updates) -> None:
    current = {}
    path = metadata_path(job_dir)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    current["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def list_jobs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    jobs = [path for path in RUNS_DIR.iterdir() if path.is_dir()]
    return sorted(jobs, key=lambda item: item.name, reverse=True)


def load_job(job_dir: Path) -> None:
    st.session_state.job_dir = job_dir
    input_candidates = list(job_dir.glob("input.*"))
    st.session_state.input_video = input_candidates[0] if input_candidates else None
    st.session_state.srt_path = job_dir / "narration.srt" if (job_dir / "narration.srt").exists() else None
    st.session_state.ass_path = job_dir / "narration.ass" if (job_dir / "narration.ass").exists() else None
    st.session_state.tts_audio_paths = sorted((job_dir / "tts").glob("*.mp3"))

    metadata = {}
    if metadata_path(job_dir).exists():
        metadata = json.loads(metadata_path(job_dir).read_text(encoding="utf-8"))
    final_output = Path(metadata["final_output"]) if metadata.get("final_output") else job_dir / "final_output.mp4"
    st.session_state.final_output = final_output if final_output.exists() else None
    publish_metadata_path = job_dir / "youtube_metadata.json"
    if publish_metadata_path.exists():
        st.session_state.publish_metadata = json.loads(publish_metadata_path.read_text(encoding="utf-8"))
    else:
        st.session_state.publish_metadata = None

    narration_json = job_dir / "narration.json"
    if narration_json.exists():
        rows = json.loads(narration_json.read_text(encoding="utf-8"))
        st.session_state.segments = rows
    else:
        st.session_state.segments = []
    word_timing_path = job_dir / "narration_word_timings.json"
    if word_timing_path.exists():
        st.session_state.word_timings = json.loads(word_timing_path.read_text(encoding="utf-8"))
    else:
        st.session_state.word_timings = []

    font_files = sorted((job_dir / "assets" / "fonts").glob("*"))
    st.session_state.custom_font_path = str(font_files[0]) if font_files else ""
    if font_files and not st.session_state.get("custom_font_family_name"):
        st.session_state.custom_font_family_name = font_files[0].stem
    logo_files = sorted((job_dir / "assets" / "logos").glob("*"))
    st.session_state.logo_path = str(logo_files[0]) if logo_files else ""

    st.session_state.logs = [f"Loaded job: {job_dir.name}"]
    st.session_state.editor_version += 1


def save_uploaded_video(uploaded_file) -> Path:
    job_dir = create_job_dir(uploaded_file.name)
    extension = Path(uploaded_file.name).suffix or ".mp4"
    input_path = job_dir / f"input{extension.lower()}"

    with input_path.open("wb") as target:
        shutil.copyfileobj(uploaded_file, target)

    st.session_state.job_dir = job_dir
    st.session_state.input_video = input_path
    st.session_state.segments = []
    st.session_state.srt_path = None
    st.session_state.ass_path = None
    st.session_state.tts_audio_paths = []
    st.session_state.word_timings = []
    st.session_state.final_output = None
    st.session_state.publish_metadata = None
    st.session_state.logs = []
    st.session_state.editor_version += 1

    write_metadata(
        job_dir,
        original_name=uploaded_file.name,
        status="uploaded",
        input_video=str(input_path),
    )
    append_log("Video uploaded.")
    return input_path


def segments_to_rows(segments: list[SubtitleSegment]) -> list[dict]:
    rows = []
    for segment in segments:
        row = asdict(segment)
        row["text"] = turkish_upper(remove_pause_punctuation(row["text"]))
        rows.append(row)
    return rows


def rows_to_segments(
    rows: list[dict],
    normalize_overlaps: bool = False,
    min_gap_ms: int = 0,
) -> list[SubtitleSegment]:
    def clean_cell(value) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    clean_rows = []
    for row in rows:
        if not any(clean_cell(value) for value in row.values()):
            continue
        clean_rows.append(
            {
                "start_time": clean_cell(row.get("start_time", "")),
                "end_time": clean_cell(row.get("end_time", "")),
                "text": turkish_upper(remove_pause_punctuation(clean_cell(row.get("text", "")))),
            }
        )
    segments = validate_segments(clean_rows)
    if normalize_overlaps:
        segments = normalize_segment_overlaps(segments, min_gap_ms=min_gap_ms)
    return segments


def clean_narration_text(text: str) -> str:
    cleaned = re.sub(r"[#@][\w-]+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"!{2,}", "!", cleaned)
    cleaned = re.sub(r"\?{2,}", "?", cleaned)
    cleaned = cleaned.strip(" -")
    return turkish_upper(remove_pause_punctuation(cleaned[:180].strip()))


def quality_warnings(
    rows: list[dict],
    speed_multiplier: float = 1.0,
) -> list[str]:
    warnings: list[str] = []
    seen_texts: set[str] = set()
    previous_end_ms = -1

    for index, row in enumerate(rows, start=1):
        text = str(row.get("text", "") or "").strip()
        start_time = str(row.get("start_time", "") or "").strip()
        end_time = str(row.get("end_time", "") or "").strip()

        if not text:
            warnings.append(f"Segment {index}: text is empty.")
            continue

        compact = text.lower()
        if compact in seen_texts:
            warnings.append(f"Segment {index}: repeated text.")
        seen_texts.add(compact)

        if len(text) > 90:
            warnings.append(f"Segment {index}: text is long ({len(text)} chars). Aim for 90 or less.")
        if any(token in text for token in ["#", "@"]):
            warnings.append(f"Segment {index}: hashtags or handles usually sound bad in narration.")
        if text.count("!") > 1:
            warnings.append(f"Segment {index}: too many exclamation marks for documentary tone.")

        try:
            start_ms = parse_srt_time(start_time)
            end_ms = parse_srt_time(end_time)
            duration = (end_ms - start_ms) / 1000
            spoken_estimate = estimate_speech_seconds(text, speed_multiplier=speed_multiplier)
            if duration < 2:
                warnings.append(f"Segment {index}: duration is very short ({duration:.1f}s).")
            if spoken_estimate > duration + 0.8:
                warnings.append(
                    f"Segment {index}: text may not fit the timing ({spoken_estimate:.1f}s speech for {duration:.1f}s segment)."
                )
            if previous_end_ms >= 0 and start_ms < previous_end_ms:
                warnings.append(f"Segment {index}: overlaps with the previous subtitle.")
            previous_end_ms = end_ms
        except Exception:
            warnings.append(f"Segment {index}: timestamp format must be HH:MM:SS,mmm.")

    return warnings


def generate_voice_preview(settings: dict) -> Path:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_segment = SubtitleSegment(
        start_time="00:00:00,000",
        end_time="00:00:05,000",
        text=settings.get("preview_text") or VOICE_PREVIEW_TEXT,
    )
    audio_paths = generate_tts_files(
        segments=[preview_segment],
        output_dir=PREVIEW_DIR,
        provider=settings["tts_provider"],
        voice=settings["voice"],
        rate=settings["rate"],
        volume=settings["volume"],
        pitch=settings["pitch"],
        elevenlabs_api_key=settings.get("elevenlabs_api_key"),
        elevenlabs_voice_id=settings.get("elevenlabs_voice_id"),
        elevenlabs_model_id=settings.get("elevenlabs_model_id", DEFAULT_ELEVENLABS_MODEL),
        elevenlabs_language_code=settings.get("elevenlabs_language_code", "tr"),
        elevenlabs_stability=settings.get("elevenlabs_stability", 0.68),
        elevenlabs_similarity_boost=settings.get("elevenlabs_similarity_boost", 0.86),
        elevenlabs_style=settings.get("elevenlabs_style", 0.08),
        elevenlabs_speaker_boost=settings.get("elevenlabs_speaker_boost", True),
        playback_speed=settings.get("playback_speed", 1.0),
    )
    safe_rate = settings["rate"].replace("%", "pct").replace("+", "plus")
    safe_pitch = settings["pitch"].replace("+", "plus")
    preview_path = PREVIEW_DIR / f"preview_{settings['voice']}_{safe_rate}_{safe_pitch}.mp3"
    audio_paths[0].replace(preview_path)
    return preview_path


def clean_preview_text(value: str) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip()


def rate_to_speed_multiplier(rate: str) -> float:
    match = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)%\s*", str(rate or ""))
    if not match:
        return 1.0
    return max(0.6, min(1.8, 1.0 + (float(match.group(1)) / 100.0)))


def effective_speech_speed(settings: dict) -> float:
    base = settings.get("playback_speed", 1.0)
    if settings.get("tts_provider") == "edge":
        base *= rate_to_speed_multiplier(settings.get("rate", DEFAULT_TTS_RATE))
    return max(0.6, min(1.8, base))


def current_preview_text(max_chars: int) -> str:
    fallback = "Kahramanimiz sahneye giriyor; doga bu karari sessizce sorguluyor."
    source_text = fallback
    rows = st.session_state.get("segments", [])
    if rows:
        selected_index = min(max(st.session_state.get("preview_segment_index", 0), 0), len(rows) - 1)
        preferred = rows[selected_index]
        candidate = clean_preview_text(preferred.get("text", "") if isinstance(preferred, dict) else "")
        if candidate:
            source_text = candidate
        else:
            for row in rows:
                candidate = clean_preview_text(row.get("text", "") if isinstance(row, dict) else "")
                if candidate:
                    source_text = candidate
                    break

    words: list[str] = []
    for word in source_text.split():
        candidate = " ".join([*words, word])
        if words and len(candidate) > max_chars:
            break
        words.append(word)
    return " ".join(words) or source_text[:max_chars].strip() or fallback


def preview_text_html(text: str, line_width: int, karaoke_words: bool = False) -> str:
    cleaned = clean_preview_text(text)
    lines = wrap(cleaned, width=max(12, line_width), break_long_words=False)
    if len(lines) > 2:
        midpoint = max(1, len(text.split()) // 2)
        words = text.split()
        lines = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
    rendered: list[str] = []
    for line in lines[:2]:
        if karaoke_words and line.strip():
            words = line.split()
            active_index = min(len(words) - 1, max(0, len(words) // 2))
            tokens = []
            for index, word in enumerate(words):
                token = escape(word)
                if index == active_index:
                    token = f'<span class="subtitle-preview-word-active">{token}</span>'
                tokens.append(token)
            rendered.append(" ".join(tokens))
        else:
            rendered.append(escape(line))
    return "<br>".join(rendered)


def tighten_segments_to_timing(
    segments: list[SubtitleSegment],
    speed_multiplier: float,
    chars_per_second: float,
) -> list[SubtitleSegment]:
    tightened: list[SubtitleSegment] = []
    for segment in segments:
        duration_seconds = max(
            0.5,
            (parse_srt_time(segment.end_time) - parse_srt_time(segment.start_time)) / 1000,
        )
        tightened.append(
            SubtitleSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=fit_text_to_duration(
                    segment.text,
                    duration_seconds=duration_seconds,
                    speed_multiplier=speed_multiplier,
                    chars_per_second=chars_per_second,
                ),
            )
        )
    return tightened


def apply_cta_segment(
    segments: list[SubtitleSegment],
    settings: dict,
    video_path: Path | None,
) -> list[SubtitleSegment]:
    if not settings.get("cta_enabled") or not video_path or not Path(video_path).exists():
        return segments
    cta_text = turkish_upper(remove_pause_punctuation(clean_preview_text(settings.get("cta_text", ""))))
    if not cta_text:
        return segments
    estimated_seconds = estimate_speech_seconds(
        cta_text,
        speed_multiplier=effective_speech_speed(settings),
        chars_per_second=max(13.0, float(settings.get("timing_density", 17.0))),
    )
    outro_duration_seconds = max(3.6, min(8.5, estimated_seconds + 0.9))
    return append_outro_segment(
        segments,
        video_duration_seconds=get_video_duration_seconds(Path(video_path)),
        text=cta_text,
        duration_seconds=outro_duration_seconds,
    )


def persist_runtime_settings(settings: dict) -> None:
    for key, value in settings.items():
        if key in {"api_key", "elevenlabs_api_key", "source_crop", "censor_regions"}:
            continue
        st.session_state[key] = value


def default_censor_rows() -> list[dict]:
    return [
        {
            "enabled": True,
            "start_time": "00:00:00,000",
            "end_time": "00:00:05,000",
            "x": 820,
            "y": 60,
            "width": 220,
            "height": 90,
            "blur": 18,
        }
    ]


def rows_to_censor_regions(rows: list[dict]) -> list[CensorRegionConfig]:
    regions: list[CensorRegionConfig] = []
    for row in rows or []:
        try:
            if not bool(row.get("enabled", True)):
                continue
            start_time = str(row.get("start_time", "00:00:00,000")).strip()
            end_time = str(row.get("end_time", "00:00:05,000")).strip()
            parse_srt_time(start_time)
            parse_srt_time(end_time)
            regions.append(
                CensorRegionConfig(
                    start_time=start_time,
                    end_time=end_time,
                    x=int(row.get("x", 0) or 0),
                    y=int(row.get("y", 0) or 0),
                    width=int(row.get("width", 200) or 200),
                    height=int(row.get("height", 100) or 100),
                    blur=int(row.get("blur", 18) or 18),
                    enabled=True,
                )
            )
        except Exception:
            continue
    return regions


def probe_video_dimensions(video_path: str, modified_at: float) -> tuple[int, int]:
    del modified_at
    try:
        result = subprocess.run(
            [resolve_ffmpeg_binary(), "-hide_banner", "-i", video_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        info = f"{result.stdout}\n{result.stderr}"
        match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return 1080, 1920


def inline_font_face_css(font_path: str | None, family_alias: str = "PreviewCustomFont") -> tuple[str, str]:
    if not font_path:
        return "", ""
    resolved = Path(font_path)
    if not resolved.exists():
        return "", ""
    suffix = resolved.suffix.lower()
    if suffix not in {".ttf", ".otf", ".woff", ".woff2"}:
        return "", ""
    mime = {
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
    }[suffix]
    font_data = base64.b64encode(resolved.read_bytes()).decode("ascii")
    css = (
        "@font-face {"
        f"font-family: '{family_alias}';"
        f"src: url(data:{mime};base64,{font_data}) format('{suffix.lstrip('.')}');"
        "font-weight: 100 900;"
        "font-style: normal;"
        "}"
    )
    return css, family_alias


def subtitle_preview_scale(stage_width: int, source_dimensions: tuple[int, int], video_layout: VideoLayoutConfig) -> float:
    source_width = max(1, int(source_dimensions[0]))
    render_width = video_layout.width if video_layout.mode in {"fit_blur", "fit_black", "fill_crop"} else source_width
    return max(0.05, stage_width / max(1, render_width))


def scaled_preview_px(value: int | float, scale: float, minimum: int = 0) -> int:
    return max(minimum, int(round(float(value) * scale)))


def render_file_download(label: str, path: Path, mime: str, key_prefix: str) -> None:
    path = Path(path)
    stat = path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    st.caption(f"{path.name} - {size_mb:.1f} MB")
    with path.open("rb") as file:
        st.download_button(
            label,
            data=file,
            file_name=path.name,
            mime=mime,
            use_container_width=True,
            key=f"{key_prefix}_{stat.st_mtime_ns}_{stat.st_size}",
        )
    st.caption("Indirme calismazsa dosya yolu:")
    st.code(str(path), language="text")


def render_subtitle_preview(
    text_color: str,
    accent_color: str,
    outline_color: str,
    font_size: int,
    font_family: str,
    font_weight: int,
    custom_font_path: str | None,
    bold: bool,
    shadow_3d: bool,
    shadow_3d_depth: int,
    boxed_background: bool,
    box_mode: str,
    box_alpha: int,
    box_padding_px: int,
    box_blur: int,
    position: str,
    glow: bool,
    karaoke_words: bool,
    max_chars: int,
    vertical_offset_px: int,
    video_layout: VideoLayoutConfig,
    background_image_data: str | None = None,
    source_dimensions: tuple[int, int] = (1080, 1920),
    logo_path: str | None = None,
    logo_scale_percent: int = 12,
    logo_opacity: float = 1.0,
    logo_anchor: str = "Top Right",
    logo_offset_x: int = 36,
    logo_offset_y: int = 36,
    censor_regions: list[CensorRegionConfig] | None = None,
) -> None:
    sample = current_preview_text(max_chars)
    sample_html = preview_text_html(sample, max(18, min(34, max_chars // 3)), karaoke_words=karaoke_words)
    justify = {
        "Top": "flex-start",
        "Middle": "center",
        "Bottom": "flex-end",
    }.get(position, "flex-end")
    source_width, source_height = source_dimensions
    source_width = max(1, source_width)
    source_height = max(1, source_height)
    is_phone_layout = video_layout.mode in {"fit_blur", "fit_black", "fill_crop"}
    aspect_ratio = "9 / 16" if is_phone_layout else f"{source_width} / {source_height}"
    stage_width = 310 if is_phone_layout else min(520, max(260, int(360 * source_width / source_height)))
    preview_scale = subtitle_preview_scale(stage_width, (source_width, source_height), video_layout)
    preview_font_px = scaled_preview_px(font_size, preview_scale, 10)
    preview_padding_y = scaled_preview_px(box_padding_px, preview_scale, 2)
    preview_padding_x = scaled_preview_px(box_padding_px * 1.4, preview_scale, 4)
    preview_offset = scaled_preview_px(vertical_offset_px, preview_scale, 0)
    preview_stage_padding_y = scaled_preview_px(72, preview_scale, 12)
    preview_stage_padding_x = scaled_preview_px(36, preview_scale, 8)
    outline_px = scaled_preview_px(max(4, font_size * 0.105), preview_scale, 1)
    glow_px = scaled_preview_px(18, preview_scale, 3)
    shadow_3d_px = scaled_preview_px(shadow_3d_depth, preview_scale, 0) if shadow_3d else 0
    shadow_steps = sorted(
        {max(1, shadow_3d_px // 3), max(1, shadow_3d_px // 2), max(1, shadow_3d_px)},
        reverse=True,
    ) if shadow_3d_px > 0 else []
    shadow_3d_css = "".join(f"{step}px {step}px 0 {outline_color}, " for step in shadow_steps)
    logo_render_width = (video_layout.width if is_phone_layout else source_width) * max(1, logo_scale_percent) / 100
    logo_preview_width = scaled_preview_px(logo_render_width, preview_scale, 24)
    logo_half_width = max(1, int(round(logo_preview_width / 2)))
    logo_offset_x_preview = scaled_preview_px(logo_offset_x, preview_scale, 0)
    logo_offset_y_preview = scaled_preview_px(logo_offset_y, preview_scale, 0)
    if position == "Bottom":
        translate_y = -preview_offset
    elif position == "Top":
        translate_y = preview_offset
    else:
        translate_y = preview_offset
    embedded_font_css, embedded_font_family = inline_font_face_css(custom_font_path)
    resolved_font_family = embedded_font_family or font_family or "Arial"
    weight = font_weight if bold else max(500, font_weight - 200)
    background_alpha = max(0, min(240, box_alpha)) / 255
    media_fit = "cover" if video_layout.mode in {"fill_crop", "source"} else "contain"
    stage_bg = "#050505" if video_layout.mode == "fit_black" else "#111318"
    has_frame = bool(background_image_data)
    image_url = f"url(data:image/jpeg;base64,{background_image_data})" if has_frame else "none"
    media_position = f"center calc(50% + {-scaled_preview_px(video_layout.crop_offset_y, preview_scale, 0)}px)"
    box_style = ""
    if boxed_background and box_mode == "text":
        box_style = (
            f"background: rgba(0,0,0,{background_alpha:.2f}); "
            f"padding: {preview_padding_y}px {preview_padding_x}px; "
            f"border-radius: {scaled_preview_px(28, preview_scale, 4)}px;"
        )
    band_html = ""
    if boxed_background and box_mode == "full_width":
        preview_line_count = sample_html.count("<br>") + 1
        band_height = max(
            scaled_preview_px(160, preview_scale, 34),
            int(preview_font_px * 1.18 * preview_line_count) + preview_padding_y * 2,
        )
        band_offset = translate_y
        band_position = (
            f"top: {max(0, preview_stage_padding_y + band_offset - preview_padding_y)}px;"
            if position == "Top"
            else f"top: calc(50% + {band_offset}px); transform: translateY(-50%);"
            if position == "Middle"
            else f"bottom: {max(0, preview_stage_padding_y - band_offset - preview_padding_y)}px;"
        )
        band_html = (
            f'<div class="subtitle-preview-band" style="height:{band_height}px;'
            f'background:rgba(0,0,0,{background_alpha:.2f});'
            f'{band_position}"></div>'
        )
    glow_class = " subtitle-preview-text-glow" if glow else ""
    blur_layer = (
        '<div class="subtitle-preview-blur"></div>'
        if video_layout.mode == "fit_blur" and has_frame
        else ""
    )
    media_layer = (
        '<div class="subtitle-preview-media"></div>'
        if has_frame
        else '<div class="subtitle-preview-empty"></div>'
    )
    logo_html = ""
    if logo_path and Path(logo_path).exists():
        logo_data = base64.b64encode(Path(logo_path).read_bytes()).decode("ascii")
        logo_ext = Path(logo_path).suffix.lower()
        logo_mime = "image/png" if logo_ext == ".png" else "image/jpeg"
        if logo_anchor == "Top Left":
            position_style = f"left:{logo_offset_x_preview}px; top:{logo_offset_y_preview}px;"
        elif logo_anchor == "Top Center":
            position_style = f"left:calc(50% - {logo_half_width}px + {logo_offset_x_preview}px); top:{logo_offset_y_preview}px;"
        elif logo_anchor == "Bottom Left":
            position_style = f"left:{logo_offset_x_preview}px; bottom:{logo_offset_y_preview}px;"
        elif logo_anchor == "Bottom Center":
            position_style = f"left:calc(50% - {logo_half_width}px + {logo_offset_x_preview}px); bottom:{logo_offset_y_preview}px;"
        elif logo_anchor == "Bottom Right":
            position_style = f"right:{logo_offset_x_preview}px; bottom:{logo_offset_y_preview}px;"
        else:
            position_style = f"right:{logo_offset_x_preview}px; top:{logo_offset_y_preview}px;"
        logo_html = (
            f'<img class="subtitle-preview-logo" style="{position_style}; width:{logo_preview_width}px; opacity:{max(0.1, min(1.0, logo_opacity)):.2f};" '
            f'src="data:{logo_mime};base64,{logo_data}" alt="logo" />'
        )
    censor_html = ""
    for index, region in enumerate(censor_regions or []):
        if not region.enabled:
            continue
        censor_x = scaled_preview_px(region.x, preview_scale, 0)
        censor_y = scaled_preview_px(region.y, preview_scale, 0)
        censor_w = scaled_preview_px(region.width, preview_scale, 4)
        censor_h = scaled_preview_px(region.height, preview_scale, 4)
        censor_html += (
            f'<div class="subtitle-preview-censor" title="Sansur {index + 1}" '
            f'style="left:{censor_x}px;top:{censor_y}px;width:{censor_w}px;height:{censor_h}px;"></div>'
        )

    st.markdown(
        f"""
        <style>
        {embedded_font_css}
        @keyframes subtitlePreviewPulse {{
            0% {{ transform: scale(.96); color: {accent_color}; text-shadow: 0 0 {glow_px}px {accent_color}, 0 {outline_px}px 0 {outline_color}; }}
            45% {{ transform: scale(1.04); color: {text_color}; text-shadow: 0 0 {max(2, glow_px - 2)}px {accent_color}, 0 {outline_px}px 0 {outline_color}; }}
            100% {{ transform: scale(1); color: {text_color}; text-shadow: 0 {outline_px}px 0 {outline_color}, 0 0 {max(2, glow_px // 2)}px rgba(0,0,0,.55); }}
        }}
        .subtitle-preview-stage {{
            position: relative;
            width: min(100%, {stage_width}px);
            aspect-ratio: {aspect_ratio};
            max-height: 520px;
            margin: 8px auto 0 auto;
            border-radius: 8px;
            overflow: hidden;
            background: {stage_bg};
            display: flex;
            align-items: {justify};
            justify-content: center;
            padding: {preview_stage_padding_y}px {preview_stage_padding_x}px;
            box-sizing: border-box;
            box-shadow: 0 18px 48px rgba(15, 23, 42, .22);
        }}
        .subtitle-preview-media,
        .subtitle-preview-blur,
        .subtitle-preview-empty {{
            position: absolute;
            inset: 0;
        }}
        .subtitle-preview-media {{
            z-index: 0;
            background:
                linear-gradient(180deg, rgba(0,0,0,.08), rgba(0,0,0,.42)),
                {image_url} {media_position} / {media_fit} no-repeat;
        }}
        .subtitle-preview-blur {{
            z-index: 0;
            background: {image_url} center / cover no-repeat;
            filter: blur({max(8, int(video_layout.blur_sigma * 0.6))}px);
            transform: scale(1.08);
            opacity: .82;
        }}
        .subtitle-preview-empty {{
            z-index: 0;
            background:
                linear-gradient(180deg, rgba(0,0,0,.15), rgba(0,0,0,.45)),
                radial-gradient(circle at 48% 24%, #7a8190, #252936 58%, #111318);
        }}
        .subtitle-preview-band {{
            position: absolute;
            left: 0;
            right: 0;
            z-index: 1;
        }}
        .subtitle-preview-censor {{
            position: absolute;
            z-index: 2;
            border-radius: 6px;
            background: rgba(18, 18, 18, .56);
            border: 1px solid rgba(255,255,255,.58);
            backdrop-filter: blur(7px);
            box-shadow: inset 0 0 0 1px rgba(0,0,0,.22), 0 8px 20px rgba(0,0,0,.22);
        }}
        .subtitle-preview-text {{
            position: relative;
            z-index: 3;
            {box_style}
            color: {text_color};
            font-size: {preview_font_px}px;
            line-height: 1.08;
            font-weight: {weight};
            font-family: '{resolved_font_family}', Arial, sans-serif;
            text-align: center;
            max-width: 96%;
            text-shadow:
                {shadow_3d_css}
                -{outline_px}px -{outline_px}px 0 {outline_color},
                {outline_px}px -{outline_px}px 0 {outline_color},
                -{outline_px}px {outline_px}px 0 {outline_color},
                {outline_px}px {outline_px}px 0 {outline_color},
                0 0 {glow_px}px rgba(0,0,0,.55),
                0 0 {max(2, glow_px // 2)}px rgba(0,0,0,.65);
            animation: subtitlePreviewPulse 1.35s ease-in-out infinite;
            transform-origin: center;
            margin-top: {translate_y}px;
            margin-bottom: {-translate_y}px;
        }}
        .subtitle-preview-logo {{
            position: absolute;
            z-index: 3;
            height: auto;
            object-fit: contain;
            filter: drop-shadow(0 8px 18px rgba(0,0,0,.28));
        }}
        .subtitle-preview-text:not(.subtitle-preview-text-glow) {{
            animation: none;
        }}
        .subtitle-preview-word-active {{
            color: {accent_color};
            text-shadow:
                0 0 {glow_px}px {accent_color},
                {shadow_3d_css}
                -{outline_px}px -{outline_px}px 0 {outline_color},
                {outline_px}px -{outline_px}px 0 {outline_color},
                -{outline_px}px {outline_px}px 0 {outline_color},
                {outline_px}px {outline_px}px 0 {outline_color};
        }}
        </style>
        <div class="subtitle-preview-stage">
          {blur_layer}
          {media_layer}
          {censor_html}
          {logo_html}
          {band_html}
          <div class="subtitle-preview-text{glow_class}">{sample_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_video_subtitle_preview(
    video_path: Path | None,
    text_color: str,
    accent_color: str,
    outline_color: str,
    font_size: int,
    font_family: str,
    font_weight: int,
    custom_font_path: str | None,
    bold: bool,
    shadow_3d: bool,
    shadow_3d_depth: int,
    boxed_background: bool,
    box_mode: str,
    box_alpha: int,
    box_padding_px: int,
    box_blur: int,
    position: str,
    glow: bool,
    karaoke_words: bool,
    max_chars: int,
    vertical_offset_px: int,
    video_layout: VideoLayoutConfig,
    logo_path: str | None = None,
    logo_scale_percent: int = 12,
    logo_opacity: float = 1.0,
    logo_anchor: str = "Top Right",
    logo_offset_x: int = 36,
    logo_offset_y: int = 36,
    source_crop: SourceCropConfig | None = None,
    censor_regions: list[CensorRegionConfig] | None = None,
) -> None:
    frame_data = None
    source_dimensions = (video_layout.width, video_layout.height)
    if not video_path or not Path(video_path).exists():
        render_subtitle_preview(
            text_color,
            accent_color,
            outline_color,
            font_size,
            font_family,
            font_weight,
            custom_font_path,
            bold,
            shadow_3d,
            shadow_3d_depth,
            boxed_background,
            box_mode,
            box_alpha,
            box_padding_px,
            box_blur,
            position,
            glow,
            karaoke_words,
            max_chars,
            vertical_offset_px,
            video_layout,
            logo_path=logo_path,
            logo_scale_percent=logo_scale_percent,
            logo_opacity=logo_opacity,
            logo_anchor=logo_anchor,
            logo_offset_x=logo_offset_x,
            logo_offset_y=logo_offset_y,
            censor_regions=censor_regions,
        )
        return

    try:
        preview_dir = Path(video_path).parent / "_preview"
        preview_dir.mkdir(exist_ok=True)
        source_dimensions = probe_video_dimensions(str(video_path), Path(video_path).stat().st_mtime)
        crop_config = source_crop or SourceCropConfig()
        if any(value > 0 for value in (crop_config.top, crop_config.bottom, crop_config.left, crop_config.right)):
            source_dimensions = (
                max(2, source_dimensions[0] - crop_config.left - crop_config.right),
                max(2, source_dimensions[1] - crop_config.top - crop_config.bottom),
            )
        crop_key = f"{crop_config.top}_{crop_config.bottom}_{crop_config.left}_{crop_config.right}"
        frame_path = preview_dir / f"subtitle_preview_{crop_key}.jpg"
        if not frame_path.exists() or frame_path.stat().st_mtime < Path(video_path).stat().st_mtime:
            command = [
                resolve_ffmpeg_binary(),
                "-y",
                "-ss",
                "00:00:01",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
            ]
            if any(value > 0 for value in (crop_config.top, crop_config.bottom, crop_config.left, crop_config.right)):
                command.extend(
                    [
                        "-vf",
                        (
                            f"crop=w='max(2,iw-{max(0, crop_config.left)}-{max(0, crop_config.right)})':"
                            f"h='max(2,ih-{max(0, crop_config.top)}-{max(0, crop_config.bottom)})':"
                            f"x={max(0, crop_config.left)}:y={max(0, crop_config.top)}"
                        ),
                    ]
                )
            command.append(str(frame_path))
            subprocess.run(command, check=True, capture_output=True)
        frame_data = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    except Exception:
        frame_data = None

    render_subtitle_preview(
        text_color,
        accent_color,
        outline_color,
        font_size,
        font_family,
        font_weight,
        custom_font_path,
        bold,
        shadow_3d,
        shadow_3d_depth,
        boxed_background,
        box_mode,
        box_alpha,
        box_padding_px,
        box_blur,
        position,
        glow,
        karaoke_words,
        max_chars,
        vertical_offset_px,
        video_layout,
        frame_data,
        source_dimensions,
        logo_path=logo_path,
        logo_scale_percent=logo_scale_percent,
        logo_opacity=logo_opacity,
        logo_anchor=logo_anchor,
        logo_offset_x=logo_offset_x,
        logo_offset_y=logo_offset_y,
        censor_regions=censor_regions,
    )


def save_segments(
    job_dir: Path,
    segments: list[SubtitleSegment],
    ass_config: AssStyleConfig | None = None,
    word_timings: list[list[dict]] | None = None,
) -> Path:
    rows = segments_to_rows(segments)
    json_path = job_dir / "narration.json"
    youtube_metadata_path = job_dir / "youtube_metadata.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    word_timing_path = job_dir / "narration_word_timings.json"
    if word_timings:
        word_timing_path.write_text(json.dumps(word_timings, indent=2, ensure_ascii=False), encoding="utf-8")
    elif word_timing_path.exists():
        word_timing_path.unlink()
    srt_path = write_srt(segments, job_dir / "narration.srt")
    ass_path = write_animated_ass(
        segments,
        job_dir / "narration.ass",
        ass_config,
        word_timings_by_segment=word_timings,
    )

    st.session_state.segments = rows
    st.session_state.word_timings = word_timings or []
    st.session_state.srt_path = srt_path
    st.session_state.ass_path = ass_path
    write_youtube_metadata(
        segments,
        youtube_metadata_path,
        source_name=Path(st.session_state.input_video).name if st.session_state.input_video else None,
    )
    st.session_state.publish_metadata = json.loads(youtube_metadata_path.read_text(encoding="utf-8"))
    write_metadata(
        job_dir,
        status="subtitles_ready",
        narration_json=str(json_path),
        youtube_metadata_path=str(youtube_metadata_path),
        word_timing_path=str(word_timing_path) if word_timings else None,
        srt_path=str(srt_path),
        ass_path=str(ass_path),
    )
    append_log("Narration JSON, SRT, animated ASS, and YouTube metadata saved.")
    return srt_path


def render_header() -> None:
    st.set_page_config(page_title="Video Pipeline Studio", layout="wide")
    theme = st.session_state.get("ui_theme", "Auto")
    light_vars = """
            --vp-bg: #f6f7f9;
            --vp-bg-top: #fbfcfd;
            --vp-bg-bottom: #eef1f5;
            --vp-surface: #ffffff;
            --vp-surface-soft: rgba(255, 255, 255, .86);
            --vp-surface-muted: rgba(255, 255, 255, .74);
            --vp-field: rgba(255,255,255,.72);
            --vp-ink: #16181d;
            --vp-muted: #68707d;
            --vp-line: #dde1e7;
            --vp-button-line: #cfd5dd;
            --vp-pill: rgba(255,255,255,.78);
            --vp-pill-ink: #29313d;
            --vp-log-bg: #111827;
            --vp-log-ink: #f8fafc;
            --vp-log-line: #1f2937;
            --vp-accent: #0f766e;
            --vp-accent-strong: #0b5f59;
            --vp-accent-shadow: rgba(15, 118, 110, .11);
    """
    dark_vars = """
            --vp-bg: #0e1117;
            --vp-bg-top: #121722;
            --vp-bg-bottom: #090b10;
            --vp-surface: #161b22;
            --vp-surface-soft: rgba(22, 27, 34, .92);
            --vp-surface-muted: rgba(22, 27, 34, .82);
            --vp-field: rgba(15, 23, 42, .78);
            --vp-ink: #edf2f7;
            --vp-muted: #a6b0bf;
            --vp-line: #2b3545;
            --vp-button-line: #334155;
            --vp-pill: rgba(21, 29, 40, .9);
            --vp-pill-ink: #d6dee8;
            --vp-log-bg: #05070b;
            --vp-log-ink: #e5edf7;
            --vp-log-line: #263244;
            --vp-accent: #14b8a6;
            --vp-accent-strong: #2dd4bf;
            --vp-accent-shadow: rgba(20, 184, 166, .18);
    """
    if theme == "Dark":
        root_vars = dark_vars
        auto_dark_block = ""
    elif theme == "Light":
        root_vars = light_vars
        auto_dark_block = ""
    else:
        root_vars = light_vars
        auto_dark_block = f"""
        @media (prefers-color-scheme: dark) {{
            :root {{
{dark_vars}
            }}
        }}
        """
    css = """
        <style>
        :root {
__ROOT_VARS__
        }
        __AUTO_DARK_BLOCK__
        .stApp {
            background:
                linear-gradient(180deg, var(--vp-bg-top) 0%, var(--vp-bg) 42%, var(--vp-bg-bottom) 100%);
            color: var(--vp-ink);
        }
        .stApp,
        .stMarkdown,
        .stMarkdown p,
        label,
        h1,
        h2,
        h3 {
            color: var(--vp-ink);
        }
        [data-testid="stCaptionContainer"],
        .stCaptionContainer,
        small {
            color: var(--vp-muted) !important;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        section[data-testid="stSidebar"] {
            background: var(--vp-surface);
            border-right: 1px solid var(--vp-line);
        }
        section[data-testid="stSidebar"] {
            min-width: 430px !important;
            max-width: 430px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: .72rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--vp-line);
            border-radius: 8px;
            background: var(--vp-surface-soft);
        }
        div[data-testid="stFileUploader"] {
            border: 1px dashed var(--vp-button-line);
            border-radius: 8px;
            padding: .35rem;
            background: var(--vp-field);
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] {
            background: var(--vp-field) !important;
            color: var(--vp-ink) !important;
            border-color: var(--vp-line) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder {
            color: var(--vp-muted) !important;
        }
        div[data-testid="stCodeBlock"],
        pre,
        code {
            background: var(--vp-log-bg) !important;
            color: var(--vp-log-ink) !important;
            border-color: var(--vp-log-line) !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            border-color: var(--vp-button-line);
            color: var(--vp-ink);
            background: var(--vp-surface-muted);
            transition: transform .14s ease, border-color .14s ease, box-shadow .14s ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--vp-accent);
            box-shadow: 0 8px 22px var(--vp-accent-shadow);
            transform: translateY(-1px);
        }
        .stButton > button[kind="primary"] {
            background: var(--vp-accent);
            border-color: var(--vp-accent);
            color: #ffffff;
        }
        .stButton > button[kind="primary"]:hover {
            background: var(--vp-accent-strong);
            border-color: var(--vp-accent-strong);
        }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        [data-testid="stMetric"] {
            background: var(--vp-surface-muted);
            border: 1px solid var(--vp-line);
            border-radius: 8px;
            padding: 10px 12px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px; }
        h1 {
            letter-spacing: 0;
            font-size: 2.05rem;
            margin-bottom: .1rem;
        }
        h2, h3 {
            letter-spacing: 0;
        }
        .pipeline-log {
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.82rem;
            line-height: 1.45;
            white-space: pre-wrap;
            background: var(--vp-log-bg);
            color: var(--vp-log-ink);
            padding: 14px;
            border-radius: 8px;
            min-height: 120px;
            border: 1px solid var(--vp-log-line);
        }
        .studio-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 10px 0 0 0;
        }
        .studio-chip {
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid var(--vp-line);
            background: var(--vp-pill);
            color: var(--vp-pill-ink);
            font-size: 0.78rem;
        }
        .preview-shell {
            border: 1px solid var(--vp-line);
            border-radius: 8px;
            background: var(--vp-surface-soft);
            padding: 12px;
        }
        </style>
        """.replace("__ROOT_VARS__", root_vars).replace("__AUTO_DARK_BLOCK__", auto_dark_block)
    st.markdown(css, unsafe_allow_html=True)
    if st.session_state.get("ui_language") == "TR":
        st.title("Video Seslendirme Studyosu")
        st.caption("Videoyu yukle, anlatimi uret, sesi olustur, altyaziyi kontrol et ve final render al.")
    else:
        st.title("Video Pipeline Studio")
        st.caption("Upload, analyze, edit, voice, subtitle, and render short-form videos from one workspace.")
    


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Kontrol Merkezi")
        st.session_state.ui_language = "TR"
        ui_language = "TR"
        is_tr = True
        theme_options = ["Auto", "Light", "Dark"]
        ui_theme = st.radio(
            "Tema",
            theme_options,
            index=theme_options.index(st.session_state.ui_theme),
            horizontal=True,
            help="Auto sistem temasini takip eder. Dark, uygulamanin ozel UI renklerini koyu moda zorlar.",
        )
        st.session_state.ui_theme = ui_theme
        st.caption(
            "Ayarlar sekmelere ayrildi; solda hizli kontrol, sagda monitor."
        )
        if st.button("Onerilen ayarlari yukle", use_container_width=True):
            apply_recommended_settings()
            st.rerun()
        advanced_settings = st.checkbox(
            "Gelismis ayarlari goster",
            value=False,
            help="Kapaliyken onerilen varsayilanlar kullanilir ve sadece temel kontroller gosterilir.",
        )

        api_key = st.session_state.get("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
        model = DEFAULT_GEMINI_MODEL
        prompt_preset = st.session_state.prompt_preset
        analysis_auto_fit = bool(st.session_state.get("analysis_auto_fit", False))
        timing_density = float(st.session_state.get("timing_density", 17.0))
        prevent_overlap = bool(st.session_state.get("prevent_overlap", True))
        min_gap_ms = int(st.session_state.get("min_gap_ms", RECOMMENDED_UI_SETTINGS["min_gap_ms"]))

        tts_provider = "elevenlabs"
        voice = st.session_state.selected_voice
        rate = DEFAULT_TTS_RATE
        volume = DEFAULT_TTS_VOLUME
        pitch = DEFAULT_TTS_PITCH
        elevenlabs_api_key = st.session_state.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY", "")
        elevenlabs_voice_id = st.session_state.get("selected_elevenlabs_voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
        elevenlabs_model_id = st.session_state.get("elevenlabs_model_id", DEFAULT_ELEVENLABS_MODEL)
        elevenlabs_language_code = "tr"
        elevenlabs_stability = float(st.session_state.get("elevenlabs_stability", 0.68))
        elevenlabs_similarity_boost = float(st.session_state.get("elevenlabs_similarity_boost", 0.86))
        elevenlabs_style = float(st.session_state.get("elevenlabs_style", 0.08))
        elevenlabs_speaker_boost = bool(st.session_state.get("elevenlabs_speaker_boost", True))
        playback_speed = float(st.session_state.get("playback_speed", RECOMMENDED_UI_SETTINGS["playback_speed"]))
        auto_fit_tts_to_segment = bool(st.session_state.get("auto_fit_tts_to_segment", True))
        max_auto_tts_speed = float(st.session_state.get("max_auto_tts_speed", 1.32))
        original_volume = float(st.session_state.get("original_volume", 0.0))

        layout_label = st.session_state.get("layout_label", "Fit 9:16 with blurred background")
        blur_sigma = int(st.session_state.get("blur_sigma", 24))
        crop_top = int(st.session_state.get("crop_top", 0))
        crop_bottom = int(st.session_state.get("crop_bottom", 0))
        crop_left = int(st.session_state.get("crop_left", 0))
        crop_right = int(st.session_state.get("crop_right", 0))
        crop_offset_y = int(st.session_state.get("crop_offset_y", 0))
        censor_region_rows = st.session_state.get("censor_regions", [])
        tts_gap_seconds = float(st.session_state.get("tts_gap_seconds", RECOMMENDED_UI_SETTINGS["tts_gap_seconds"]))

        subtitle_motion = st.session_state.get("subtitle_motion", "Single layer")
        subtitle_preset = st.session_state.get("subtitle_preset", "Documentary")
        subtitle_position = st.session_state.get("subtitle_position", "Bottom")
        subtitle_vertical_offset = int(st.session_state.get("subtitle_vertical_offset", 0))
        subtitle_font_size = int(st.session_state.get("subtitle_font_size", 62))
        subtitle_font_family = st.session_state.get("subtitle_font_family", "Inter")
        subtitle_font_weight = int(st.session_state.get("subtitle_font_weight", 900))
        custom_font_path = st.session_state.get("custom_font_path", "")
        custom_font_family_name = st.session_state.get("custom_font_family_name", "")
        subtitle_bold = bool(st.session_state.get("subtitle_bold", True))
        subtitle_color = st.session_state.get("subtitle_color", "#FFFFFF")
        subtitle_accent = st.session_state.get("subtitle_accent", "#FFE45C")
        outline_color = st.session_state.get("outline_color", "#000000")
        subtitle_3d_shadow = bool(st.session_state.get("subtitle_3d_shadow", False))
        subtitle_3d_depth = int(st.session_state.get("subtitle_3d_depth", 0))
        boxed_background = False
        st.session_state.boxed_background = False
        box_mode = st.session_state.get("box_mode", "text")
        box_alpha = int(st.session_state.get("box_alpha", 150))
        box_padding_px = int(st.session_state.get("box_padding_px", 28))
        box_blur = 0
        st.session_state.box_blur = 0
        glow_subtitle = bool(st.session_state.get("glow_subtitle", True))
        subtitle_max_chars = int(st.session_state.get("subtitle_max_chars", 84))
        line_width = int(st.session_state.get("line_width", 24))
        karaoke_words = bool(st.session_state.get("karaoke_words", True))

        output_name = st.session_state.get("output_name", "final_output.mp4")
        logo_enabled = bool(st.session_state.get("logo_enabled", False))
        logo_path = st.session_state.get("logo_path", "")
        logo_scale_percent = int(st.session_state.get("logo_scale_percent", 12))
        logo_opacity = float(st.session_state.get("logo_opacity", 0.95))
        logo_anchor = st.session_state.get("logo_anchor", "Top Right")
        logo_offset_x = int(st.session_state.get("logo_offset_x", 36))
        logo_offset_y = int(st.session_state.get("logo_offset_y", 36))
        cta_enabled = bool(st.session_state.get("cta_enabled", True))
        cta_text = st.session_state.get(
            "cta_text",
            "İzlediğiniz için teşekkürler. Beğenmeyi ve abone olmayı unutmayın.",
        )

        ai_tab, voice_tab, subtitle_tab, video_tab, session_tab = st.tabs(
            ["Yapay Zeka", "Ses", "Altyazi", "Video", "Oturum"]
        )

        with ai_tab:
            api_key = st.text_input(
                "Gemini API anahtari",
                value=api_key,
                type="password",
                help="Sadece video analizi icin kullanilir. Kaydedersen sonraki acilislarda otomatik dolar.",
            ).strip()
            st.session_state.gemini_api_key = api_key
            prompt_preset = st.selectbox(
                "Anlatim stili",
                list(PROMPT_PRESETS.keys()),
                index=list(PROMPT_PRESETS.keys()).index(st.session_state.prompt_preset),
                help="Gemini prompt stilini belirler. Deadpan Documentary komik videolarda daha dengeli kalir.",
            )
            st.session_state.prompt_preset = prompt_preset
            cta_enabled = st.checkbox("Final tesekkur mesaji ekle", value=cta_enabled)
            cta_text = st.text_area("Final mesaji", value=cta_text, height=90)
            if advanced_settings:
                if st.button(
                    "Gemini modellerini yenile",
                    use_container_width=True,
                    help="Bu anahtar ile kullanilabilen ve generateContent destekleyen modelleri listeler.",
                ):
                    try:
                        st.session_state.available_models = list_generate_content_models(api_key or None)
                        append_log(f"Loaded {len(st.session_state.available_models)} Gemini models.")
                    except Exception as exc:
                        append_log(f"Model list failed: {exc}")
                        st.error(str(exc))

                if st.session_state.available_models:
                    default_index = 0
                    if DEFAULT_GEMINI_MODEL in st.session_state.available_models:
                        default_index = st.session_state.available_models.index(DEFAULT_GEMINI_MODEL)
                    model = st.selectbox(
                        "Gemini modeli",
                        st.session_state.available_models,
                        index=default_index,
                        help="Sadece generateContent destekleyen modeller gosterilir.",
                    )
                else:
                    model = st.text_input(
                        "Gemini modeli",
                        value=DEFAULT_GEMINI_MODEL,
                        help="Varsayilan gemini-2.5-flash. Gerekirse model listesini yenileyebilirsin.",
                    )
                    normalized_model = normalize_model_name(model)
                    if normalized_model != model:
                        st.warning(f"{model} is deprecated here; using {normalized_model}.")
                        model = normalized_model

                if st.button("Hazir promptu yukle", use_container_width=True):
                    st.session_state.custom_prompt = PROMPT_PRESETS[prompt_preset]
                    st.rerun()
                prompt_text = st.text_area(
                    "Gemini prompt",
                    value=st.session_state.get("custom_prompt", PROMPT_PRESETS[prompt_preset]),
                    height=180,
                    help="Gemini analiz promptunu burada duzenleyebilirsin.",
                )
                st.session_state.custom_prompt = prompt_text
                analysis_auto_fit = st.checkbox(
                    "Gemini metnini sureye gore kisalt",
                    value=analysis_auto_fit,
                    help="Analizden sonra fazla uzun satirlari sureye gore otomatik kisaltir.",
                )
                timing_density = st.slider(
                    "Metin yogunlugu",
                    12.0,
                    18.0,
                    timing_density,
                    0.5,
                    help="Dusuk deger daha kisa metin ister; yuksek deger daha yogun anlatima izin verir.",
                )
                prevent_overlap = st.checkbox(
                    "Segment cakismalarini duzelt",
                    value=prevent_overlap,
                )
                min_gap_ms = st.slider(
                    "Min segment boslugu ms",
                    0,
                    300,
                    min_gap_ms,
                    10,
                )

        with voice_tab:
            st.caption("Sadece ElevenLabs kullaniliyor. Dil sabit olarak Turkce.")
            tts_provider = "elevenlabs"
            elevenlabs_api_key = st.text_input(
                "ElevenLabs API anahtari",
                value=elevenlabs_api_key,
                type="password",
                help="Kaydedersen sonraki acilislarda otomatik dolar.",
            ).strip()
            st.session_state.elevenlabs_api_key = elevenlabs_api_key
            if st.button("ElevenLabs seslerini getir", use_container_width=True):
                try:
                    with st.spinner("ElevenLabs sesleri getiriliyor..."):
                        st.session_state.elevenlabs_voices = list_elevenlabs_voices(elevenlabs_api_key or None)
                    append_log(f"Loaded {len(st.session_state.elevenlabs_voices)} ElevenLabs voices.")
                except Exception as exc:
                    append_log(f"ElevenLabs voices failed: {exc}")
                    st.error(str(exc))
            if st.session_state.elevenlabs_voices:
                voice_labels = [f"{item.get('name', 'Isimsiz')} | {item.get('voice_id', '')}" for item in st.session_state.elevenlabs_voices]
                selected_voice_id = st.session_state.get("selected_elevenlabs_voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
                default_voice_index = next(
                    (
                        index
                        for index, item in enumerate(st.session_state.elevenlabs_voices)
                        if item.get("voice_id") == selected_voice_id
                    ),
                    0,
                )
                selected_label = st.selectbox("ElevenLabs sesi", voice_labels, index=default_voice_index)
                elevenlabs_voice_id = selected_label.split("|")[-1].strip()
            else:
                st.caption(f"Varsayilan ses: {DEFAULT_ELEVENLABS_VOICE_NAME}")
                elevenlabs_voice_id = st.text_input(
                    "ElevenLabs voice_id",
                    value=st.session_state.get("selected_elevenlabs_voice_id", DEFAULT_ELEVENLABS_VOICE_ID),
                )
            st.session_state.selected_elevenlabs_voice_id = elevenlabs_voice_id or ""
            voice = elevenlabs_voice_id or "elevenlabs"
            rate = DEFAULT_TTS_RATE
            volume = DEFAULT_TTS_VOLUME
            pitch = DEFAULT_TTS_PITCH

            elevenlabs_language_code = "tr"
            if advanced_settings:
                elevenlabs_model_id = st.text_input("Model ID", value=elevenlabs_model_id)
                st.session_state.elevenlabs_model_id = elevenlabs_model_id
                st.text_input("Dil", value="tr", disabled=True)
                elevenlabs_stability = st.slider("Stability", 0.0, 1.0, elevenlabs_stability, 0.05)
                elevenlabs_similarity_boost = st.slider("Similarity boost", 0.0, 1.0, elevenlabs_similarity_boost, 0.05)
                elevenlabs_style = st.slider("Style", 0.0, 1.0, elevenlabs_style, 0.05)
                elevenlabs_speaker_boost = st.checkbox("Speaker boost", value=elevenlabs_speaker_boost)

                playback_speed = st.slider(
                    "Gercek okuma hizi",
                    0.85,
                    1.45,
                    playback_speed,
                    0.01,
                )
                auto_fit_tts_to_segment = st.checkbox(
                    "Sesleri segmente sigdir",
                    value=auto_fit_tts_to_segment,
                )
                max_auto_tts_speed = st.slider(
                    "Maks otomatik hiz",
                    1.0,
                    1.6,
                    max_auto_tts_speed,
                    0.01,
                )
                original_volume = st.slider(
                    "Orijinal video sesi",
                    0,
                    100,
                    int(round(original_volume * 100)),
                    help="0 ise orijinal ses tamamen kaldirilir. Yeni ses tek basina kullanilacaksa 0 birak.",
                ) / 100

            preview_settings = {
                "tts_provider": tts_provider,
                "voice": voice,
                "rate": rate,
                "volume": volume,
                "pitch": pitch,
                "elevenlabs_api_key": elevenlabs_api_key,
                "elevenlabs_voice_id": elevenlabs_voice_id,
                "elevenlabs_model_id": elevenlabs_model_id,
                "elevenlabs_language_code": elevenlabs_language_code,
                "elevenlabs_stability": elevenlabs_stability,
                "elevenlabs_similarity_boost": elevenlabs_similarity_boost,
                "elevenlabs_style": elevenlabs_style,
                "elevenlabs_speaker_boost": elevenlabs_speaker_boost,
                "playback_speed": playback_speed,
                "preview_text": current_preview_text(88),
            }
            if st.button("Secili sesi onizle", use_container_width=True):
                try:
                    with st.spinner("Ses onizlemesi olusturuluyor..."):
                        st.session_state.voice_preview_path = generate_voice_preview(preview_settings)
                    append_log(f"Voice preview generated: {voice}")
                except Exception as exc:
                    append_log(f"Voice preview failed: {exc}")
                    st.error(str(exc))
            if st.session_state.voice_preview_path and Path(st.session_state.voice_preview_path).exists():
                st.audio(str(st.session_state.voice_preview_path))

        with subtitle_tab:
            subtitle_motion = st.selectbox(
                "Altyazi giris efekti",
                ["Single layer", "Animated pulse"],
                index=["Single layer", "Animated pulse"].index(
                    subtitle_motion if subtitle_motion in {"Single layer", "Animated pulse"} else "Single layer"
                ),
                format_func=lambda value: "Sabit / parlama yok" if value == "Single layer" else "Giris parlamasi",
                help="Bu ayar sadece altyazi ekrana geldigindeki genel parlama/buyume efektini kontrol eder.",
            )
            karaoke_words = st.checkbox(
                "Sesle sari kelime takibi",
                value=karaoke_words,
                help="Aktif kelimeyi sesle beraber renklendirir. Tamamen sabit altyazi istiyorsan bunu kapat.",
            )
            if subtitle_motion == "Single layer":
                st.caption("Sabit mod giris parlamasini kapatir. Sari kelime hareketi icin yukaridaki kelime takibi ayri calisir.")
            subtitle_preset = st.selectbox(
                "Hazir stil",
                list(SUBTITLE_PRESETS.keys()),
                index=list(SUBTITLE_PRESETS.keys()).index(subtitle_preset if subtitle_preset in SUBTITLE_PRESETS else "Documentary"),
            )
            if st.button("Stili uygula", use_container_width=True):
                for key, value in SUBTITLE_UI_PRESETS[subtitle_preset].items():
                    st.session_state[key] = value
                st.session_state.subtitle_preset = subtitle_preset
                append_log(f"Subtitle preset applied: {subtitle_preset}")
                st.rerun()
            preset_values = SUBTITLE_UI_PRESETS.get(subtitle_preset, {})
            if st.session_state.get("subtitle_preset") != subtitle_preset:
                subtitle_motion = preset_values.get("subtitle_motion", subtitle_motion)
                subtitle_font_family = preset_values.get("subtitle_font_family", subtitle_font_family)
                subtitle_font_weight = preset_values.get("subtitle_font_weight", subtitle_font_weight)
                subtitle_font_size = preset_values.get("subtitle_font_size", subtitle_font_size)
                subtitle_color = preset_values.get("subtitle_color", subtitle_color)
                subtitle_accent = preset_values.get("subtitle_accent", subtitle_accent)
                outline_color = preset_values.get("outline_color", outline_color)
                subtitle_3d_shadow = preset_values.get("subtitle_3d_shadow", subtitle_3d_shadow)
                subtitle_3d_depth = preset_values.get("subtitle_3d_depth", subtitle_3d_depth)
                boxed_background = False
                box_mode = preset_values.get("box_mode", box_mode)
                box_alpha = preset_values.get("box_alpha", box_alpha)
                box_padding_px = preset_values.get("box_padding_px", box_padding_px)
                glow_subtitle = preset_values.get("glow_subtitle", glow_subtitle)
                karaoke_words = preset_values.get("karaoke_words", karaoke_words)
                subtitle_max_chars = preset_values.get("subtitle_max_chars", subtitle_max_chars)
                line_width = preset_values.get("line_width", line_width)
            subtitle_font_family = st.selectbox(
                "Font ailesi",
                FONT_FAMILY_OPTIONS,
                index=FONT_FAMILY_OPTIONS.index(subtitle_font_family if subtitle_font_family in FONT_FAMILY_OPTIONS else "Inter"),
                help="Final renderda fontun sistemde kurulu olmasi veya asagidan TTF/OTF olarak yuklenmesi gerekir.",
            )
            subtitle_font_weight = st.select_slider(
                "Font agirligi",
                options=[700, 800, 900],
                value=subtitle_font_weight,
                format_func=lambda value: {700: "Bold", 800: "Extra Bold", 900: "Black"}[value],
            )
            uploaded_font = st.file_uploader(
                "Custom font yukle",
                type=["ttf", "otf", "woff", "woff2"],
                help="Inter/Proxima/Roboto dosyasi yuklersen preview ve final render ayni fontu kullanir.",
                key="custom_font_uploader",
            )
            free_font_label = st.selectbox(
                "Ucretsiz font indir",
                list(FREE_FONT_DOWNLOADS.keys()),
                index=list(FREE_FONT_DOWNLOADS.keys()).index(
                    subtitle_font_family if subtitle_font_family in FREE_FONT_DOWNLOADS else "Luckiest Guy"
                ),
                help="Bu fontlar Google Fonts uzerinden job klasorune TTF olarak indirilir.",
            )
            if st.button(
                "Secili fontu indir ve kullan",
                use_container_width=True,
                disabled=not st.session_state.job_dir,
                help="Font dosyasi indirilmeden final render bu stili uygulayamaz.",
            ):
                try:
                    saved_font, downloaded_family = download_free_font(free_font_label, st.session_state.job_dir)
                    custom_font_path = str(saved_font)
                    custom_font_family_name = downloaded_family
                    subtitle_font_family = downloaded_family
                    st.session_state.custom_font_path = custom_font_path
                    st.session_state.custom_font_family_name = custom_font_family_name
                    st.session_state.subtitle_font_family = subtitle_font_family
                    append_log(f"Downloaded font: {downloaded_family}")
                    st.success(f"{downloaded_family} indirildi ve secildi.")
                    st.rerun()
                except Exception as exc:
                    append_log(f"Font download failed: {exc}")
                    st.error(f"Font indirilemedi: {exc}")
            if uploaded_font and st.session_state.job_dir:
                saved_font = save_job_asset(uploaded_font, st.session_state.job_dir / "assets" / "fonts", "subtitle_font" + Path(uploaded_font.name).suffix.lower())
                custom_font_path = str(saved_font)
                st.session_state.custom_font_path = custom_font_path
                custom_font_family_name = Path(uploaded_font.name).stem
                st.session_state.custom_font_family_name = custom_font_family_name
                append_log(f"Custom font saved: {saved_font.name}")
            elif uploaded_font and not st.session_state.job_dir:
                st.info("Custom fontu kalici kaydetmek icin once bir is olustur.")
            if custom_font_path:
                custom_font_family_name = st.text_input(
                    "Custom font adi",
                    value=custom_font_family_name or Path(custom_font_path).stem,
                    help="TTF/OTF icindeki gercek font family adi. Ornek: Luckiest Guy, Burbank Big Condensed Black, Komika Axis.",
                ).strip()
                st.session_state.custom_font_family_name = custom_font_family_name
                st.caption("Final render bu adi ASS font adi olarak kullanir; dosya adi farkliysa bu alan onemlidir.")
            if not custom_font_path and subtitle_font_family not in {"Inter", "Roboto", "Arial", "SF Pro Display"}:
                st.warning("Bu font secili ama dosyasi yuklu degil. Fontu indir veya TTF/OTF yukle; yoksa final render fallback font kullanir.")
            subtitle_position = st.selectbox(
                "Konum",
                ["Bottom", "Middle", "Top"],
                index=["Bottom", "Middle", "Top"].index(subtitle_position if subtitle_position in {"Bottom", "Middle", "Top"} else "Bottom"),
                format_func=lambda value: {
                    "Bottom": "Alt",
                    "Middle": "Orta",
                    "Top": "Ust",
                }[value],
            )
            subtitle_vertical_offset = st.slider("Dikey offset px", -220, 420, subtitle_vertical_offset, 10)
            subtitle_font_size = st.slider("Yazi boyutu", 38, 88, subtitle_font_size)
            subtitle_max_chars = st.slider("Tek ekranda karakter", 45, 140, subtitle_max_chars)
            if advanced_settings:
                subtitle_bold = st.checkbox("Kalin yazi", value=subtitle_bold)
                subtitle_color = st.color_picker("Yazi rengi", subtitle_color)
                subtitle_accent = st.color_picker("Parlama rengi", subtitle_accent)
                outline_color = st.color_picker("Dis cizgi rengi", outline_color)
                subtitle_3d_shadow = st.checkbox(
                    "3D siyah golge",
                    value=subtitle_3d_shadow,
                    help="Yazinin arkasina asagi/saga kaydirilmis kalin katman ekler. Gorseldeki blok golge etkisi budur.",
                )
                subtitle_3d_depth = st.slider(
                    "3D golge derinligi px",
                    0,
                    34,
                    subtitle_3d_depth if subtitle_3d_shadow else 0,
                    1,
                    disabled=not subtitle_3d_shadow,
                )
                glow_subtitle = st.checkbox(
                    "Giris parlamasinda renk degisimi",
                    value=glow_subtitle,
                    disabled=subtitle_motion == "Single layer",
                    help="Sadece 'Giris parlamasi' seciliyken etkilidir. Kelime takibini kontrol etmez.",
                )

        with video_tab:
            layout_label = st.selectbox(
                "Video yerlesimi",
                [
                    "Original",
                    "Fit 9:16 with blurred background",
                    "Fit 9:16 with black bars",
                    "Fill 9:16 crop",
                ],
                index=[
                    "Original",
                    "Fit 9:16 with blurred background",
                    "Fit 9:16 with black bars",
                    "Fill 9:16 crop",
                ].index(layout_label if layout_label in {
                    "Original",
                    "Fit 9:16 with blurred background",
                    "Fit 9:16 with black bars",
                    "Fill 9:16 crop",
                } else "Fit 9:16 with blurred background"),
                format_func=lambda value: {
                    "Original": "Orijinal",
                    "Fit 9:16 with blurred background": "9:16 blur arka plan",
                    "Fit 9:16 with black bars": "9:16 siyah bosluk",
                    "Fill 9:16 crop": "9:16 kirparak doldur",
                }[value],
            )
            with st.expander("Kadraj / kırpma", expanded=False):
                st.caption("Kaynak videodan gereksiz üst/alt/yan alanları keser. Değerler kaynak video pikselidir.")
                crop_col_a, crop_col_b = st.columns(2)
                with crop_col_a:
                    crop_top = st.number_input("Üstten kırp px", 0, 2000, crop_top, 10)
                    crop_left = st.number_input("Soldan kırp px", 0, 2000, crop_left, 10)
                with crop_col_b:
                    crop_bottom = st.number_input("Alttan kırp px", 0, 2000, crop_bottom, 10)
                    crop_right = st.number_input("Sağdan kırp px", 0, 2000, crop_right, 10)
                crop_offset_y = st.slider(
                    "9:16 kırpma kadrajı yukarı/aşağı",
                    -600,
                    600,
                    crop_offset_y,
                    10,
                    help="Sadece '9:16 kırparak doldur' modunda dikey kadrajı kaydırır. Negatif yukarı, pozitif aşağı kaydırır.",
                )
            with st.expander("Sansür / blur bölgeleri", expanded=False):
                st.caption("Koordinatlar final çıktı üzerindedir. Varsayılan Shorts çıktı 1080x1920 kabul edilir.")
                censor_col_a, censor_col_b = st.columns(2)
                with censor_col_a:
                    if st.button("Örnek sansür kutusu ekle", use_container_width=True):
                        censor_region_rows = [*list(censor_region_rows or []), *default_censor_rows()]
                        st.session_state.censor_regions = censor_region_rows
                        st.rerun()
                with censor_col_b:
                    if st.button(
                        "Gemini ile sansür öner",
                        use_container_width=True,
                        disabled=not st.session_state.input_video,
                    ):
                        try:
                            with st.spinner("Gemini sansür bölgelerini öneriyor..."):
                                suggested_rows = detect_censor_regions_with_gemini(
                                    Path(st.session_state.input_video),
                                    api_key=api_key or None,
                                    model_name=model,
                                )
                            censor_region_rows = suggested_rows
                            st.session_state.censor_regions = censor_region_rows
                            append_log(f"Gemini suggested {len(suggested_rows)} censor regions.")
                            st.rerun()
                        except Exception as exc:
                            append_log(f"Gemini censor detection failed: {exc}")
                            st.error(str(exc))
                censor_frame = pd.DataFrame(
                    censor_region_rows or [],
                    columns=["enabled", "start_time", "end_time", "x", "y", "width", "height", "blur"],
                )
                edited_censors = st.data_editor(
                    censor_frame,
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    height=180,
                    key="censor_region_editor",
                    column_config={
                        "enabled": st.column_config.CheckboxColumn("Aktif"),
                        "start_time": st.column_config.TextColumn("Başlangıç", help="HH:MM:SS,mmm"),
                        "end_time": st.column_config.TextColumn("Bitiş", help="HH:MM:SS,mmm"),
                        "x": st.column_config.NumberColumn("X", min_value=0, step=4),
                        "y": st.column_config.NumberColumn("Y", min_value=0, step=4),
                        "width": st.column_config.NumberColumn("Genişlik", min_value=2, step=4),
                        "height": st.column_config.NumberColumn("Yükseklik", min_value=2, step=4),
                        "blur": st.column_config.NumberColumn("Blur", min_value=2, max_value=40, step=1),
                    },
                )
                censor_region_rows = edited_censors.to_dict("records")
                st.session_state.censor_regions = censor_region_rows
            logo_enabled = st.checkbox("Logo ekle", value=logo_enabled)
            uploaded_logo = st.file_uploader(
                "Logo yukle",
                type=["png", "jpg", "jpeg", "webp"],
                key="logo_uploader",
                help="Tercihen seffaf PNG kullan.",
            )
            if uploaded_logo and st.session_state.job_dir:
                saved_logo = save_job_asset(uploaded_logo, st.session_state.job_dir / "assets" / "logos", "overlay_logo" + Path(uploaded_logo.name).suffix.lower())
                logo_path = str(saved_logo)
                st.session_state.logo_path = logo_path
                append_log(f"Logo saved: {saved_logo.name}")
            elif uploaded_logo and not st.session_state.job_dir:
                st.info("Logoyu kalici kaydetmek icin once bir is olustur.")
            if advanced_settings:
                blur_sigma = st.slider("Arka plan blur gucu", 4, 48, blur_sigma)
                tts_gap_seconds = st.slider("Sesler arasi bosluk (sn)", 0.0, 2.0, tts_gap_seconds, 0.25)
                logo_anchor = st.selectbox(
                    "Logo konumu",
                    ["Top Right", "Top Left", "Top Center", "Bottom Right", "Bottom Left", "Bottom Center"],
                    index=["Top Right", "Top Left", "Top Center", "Bottom Right", "Bottom Left", "Bottom Center"].index(
                        logo_anchor if logo_anchor in {"Top Right", "Top Left", "Top Center", "Bottom Right", "Bottom Left", "Bottom Center"} else "Top Right"
                    ),
                )
                logo_scale_percent = st.slider("Logo genisligi %", 4, 24, logo_scale_percent)
                logo_opacity = st.slider("Logo opakligi", 0.1, 1.0, float(logo_opacity), 0.05)
                logo_offset_x = st.slider("Logo yatay bosluk px", 0, 240, logo_offset_x, 4)
                logo_offset_y = st.slider("Logo dikey bosluk px", 0, 240, logo_offset_y, 4)

            layout_map = {
                "Original": "source",
                "Fit 9:16 with blurred background": "fit_blur",
                "Fit 9:16 with black bars": "fit_black",
                "Fill 9:16 crop": "fill_crop",
            }
            video_layout = VideoLayoutConfig(mode=layout_map[layout_label], blur_sigma=blur_sigma, crop_offset_y=crop_offset_y)
            source_crop = SourceCropConfig(
                top=crop_top,
                bottom=crop_bottom,
                left=crop_left,
                right=crop_right,
            )
            censor_regions = rows_to_censor_regions(censor_region_rows)

            if st.session_state.segments:
                preview_labels = [
                    f"{index + 1}. {clean_preview_text(row.get('text', ''))[:56]}"
                    for index, row in enumerate(st.session_state.segments)
                ]
                selected_preview = st.selectbox(
                    "Onizleme segmenti",
                    range(len(preview_labels)),
                    index=min(st.session_state.preview_segment_index, len(preview_labels) - 1),
                    format_func=lambda item: preview_labels[item],
                )
                st.session_state.preview_segment_index = int(selected_preview)
            st.caption("Canli onizleme")
            render_video_subtitle_preview(
                video_path=st.session_state.input_video,
                text_color=subtitle_color,
                accent_color=subtitle_accent,
                outline_color=outline_color,
                font_size=subtitle_font_size,
                font_family=subtitle_font_family,
                font_weight=subtitle_font_weight,
                custom_font_path=custom_font_path or None,
                bold=subtitle_bold,
                shadow_3d=subtitle_3d_shadow,
                shadow_3d_depth=subtitle_3d_depth,
                boxed_background=boxed_background,
                box_mode=box_mode,
                box_alpha=box_alpha,
                box_padding_px=box_padding_px,
                box_blur=box_blur,
                position=subtitle_position,
                glow=glow_subtitle and subtitle_motion == "Animated pulse",
                karaoke_words=karaoke_words,
                max_chars=subtitle_max_chars,
                vertical_offset_px=subtitle_vertical_offset,
                video_layout=video_layout,
                logo_path=logo_path if logo_enabled else None,
                logo_scale_percent=logo_scale_percent,
                logo_opacity=logo_opacity,
                logo_anchor=logo_anchor,
                logo_offset_x=logo_offset_x,
                logo_offset_y=logo_offset_y,
                source_crop=source_crop,
                censor_regions=censor_regions,
            )

        with session_tab:
            try:
                ffmpeg_path = resolve_ffmpeg_binary()
                st.success("FFmpeg ready")
                with st.expander("FFmpeg path", expanded=False):
                    st.code(ffmpeg_path, language="text")
            except Exception as exc:
                st.error("FFmpeg unavailable")
                with st.expander("FFmpeg error", expanded=False):
                    st.code(str(exc), language="text")

            output_name = st.text_input(
                "Cikti dosya adi",
                value=output_name,
                help="Render edilen MP4 dosyasinin job klasoru icindeki adi.",
            )
            st.markdown("#### API anahtarları")
            saved_api_keys = load_saved_api_keys()
            st.caption(f"Gemini: {mask_secret(saved_api_keys.get('gemini_api_key'))}")
            st.caption(f"ElevenLabs: {mask_secret(saved_api_keys.get('elevenlabs_api_key'))}")
            key_col_a, key_col_b = st.columns(2)
            with key_col_a:
                if st.button(
                    "API anahtarlarini kaydet",
                    use_container_width=True,
                    help="Gemini ve ElevenLabs anahtarlarini bu cihaza kaydeder.",
                ):
                    if save_api_keys(api_key, elevenlabs_api_key):
                        st.success("API anahtarlari kaydedildi.")
                    else:
                        st.warning("Kaydedilecek API anahtari yok.")
            with key_col_b:
                if st.button(
                    "Kayitli anahtarlari sil",
                    use_container_width=True,
                    help="Bu cihazda tutulan Gemini ve ElevenLabs anahtarlarini siler.",
                ):
                    clear_saved_api_keys()
                    st.success("Kayitli API anahtarlari silindi.")
                    st.rerun()
            st.caption(f"Yerel dosya: {API_KEYS_PATH}")

            if st.button(
                "Ayarlari kaydet",
                use_container_width=True,
                help="Mevcut UI ayarlari sonraki acilislarda otomatik yuklenir.",
            ):
                save_ui_settings(
                    {
                        "ui_theme": ui_theme,
                        "prompt_preset": prompt_preset,
                        "custom_prompt": st.session_state.get("custom_prompt", PROMPT_PRESETS[prompt_preset]),
                        "playback_speed": playback_speed,
                        "auto_fit_tts_to_segment": auto_fit_tts_to_segment,
                        "max_auto_tts_speed": max_auto_tts_speed,
                        "original_volume": original_volume,
                        "analysis_auto_fit": analysis_auto_fit,
                        "timing_density": timing_density,
                        "prevent_overlap": prevent_overlap,
                        "min_gap_ms": min_gap_ms,
                        "subtitle_motion": subtitle_motion,
                        "subtitle_preset": subtitle_preset,
                        "subtitle_position": subtitle_position,
                        "subtitle_vertical_offset": subtitle_vertical_offset,
                        "subtitle_font_size": subtitle_font_size,
                        "subtitle_font_family": subtitle_font_family,
                        "subtitle_font_weight": subtitle_font_weight,
                        "custom_font_family_name": custom_font_family_name,
                        "subtitle_bold": subtitle_bold,
                        "subtitle_color": subtitle_color,
                        "subtitle_accent": subtitle_accent,
                        "outline_color": outline_color,
                        "subtitle_3d_shadow": subtitle_3d_shadow,
                        "subtitle_3d_depth": subtitle_3d_depth,
                        "boxed_background": False,
                        "box_mode": box_mode,
                        "box_alpha": box_alpha,
                        "box_padding_px": box_padding_px,
                        "glow_subtitle": glow_subtitle,
                        "subtitle_max_chars": subtitle_max_chars,
                        "line_width": line_width,
                        "layout_label": layout_label,
                        "blur_sigma": blur_sigma,
                        "crop_top": crop_top,
                        "crop_bottom": crop_bottom,
                        "crop_left": crop_left,
                        "crop_right": crop_right,
                        "crop_offset_y": crop_offset_y,
                        "censor_regions": censor_region_rows,
                        "tts_gap_seconds": tts_gap_seconds,
                        "output_name": output_name,
                        "cta_enabled": cta_enabled,
                        "cta_text": cta_text,
                        "custom_font_path": custom_font_path,
                        "logo_enabled": logo_enabled,
                        "logo_path": logo_path,
                        "logo_scale_percent": logo_scale_percent,
                        "logo_opacity": logo_opacity,
                        "logo_anchor": logo_anchor,
                        "logo_offset_x": logo_offset_x,
                        "logo_offset_y": logo_offset_y,
                        "karaoke_words": karaoke_words,
                        "elevenlabs_model_id": elevenlabs_model_id,
                        "selected_elevenlabs_voice_id": elevenlabs_voice_id,
                        "elevenlabs_stability": elevenlabs_stability,
                        "elevenlabs_similarity_boost": elevenlabs_similarity_boost,
                        "elevenlabs_style": elevenlabs_style,
                        "elevenlabs_speaker_boost": elevenlabs_speaker_boost,
                    }
                )
                st.success("Ayarlar kaydedildi.")
            jobs = list_jobs()
            if jobs:
                labels = ["Eski isi yukle"] + [job.name for job in jobs]
                selected = st.selectbox("Is gecmisi", labels)
                if selected != "Eski isi yukle" and st.button("Isi yukle", use_container_width=True):
                    load_job(RUNS_DIR / selected)
                    st.rerun()

        ass_font_name = (
            custom_font_family_name.strip()
            if custom_font_path and custom_font_family_name.strip()
            else (Path(custom_font_path).stem if custom_font_path else subtitle_font_family)
        )
        position_config = style_for_position(
            AssStyleConfig(
                font_name=ass_font_name,
                font_size=subtitle_font_size,
                primary_color=subtitle_color,
                accent_color=subtitle_accent,
                outline_color=outline_color,
                outline=max(5, int(round(subtitle_font_size * 0.11))),
                shadow=max(2, int(round(subtitle_font_size * 0.04))),
                shadow_3d=subtitle_3d_shadow,
                shadow_3d_depth=subtitle_3d_depth,
                bold=subtitle_bold,
                boxed_background=boxed_background,
                box_mode=box_mode,
                box_padding_px=box_padding_px,
                box_blur=0,
                back_alpha=255 - box_alpha,
                glow=glow_subtitle and subtitle_motion == "Animated pulse",
                animate=subtitle_motion == "Animated pulse",
                karaoke_words=karaoke_words,
                max_chars=subtitle_max_chars,
                line_width=line_width,
            ),
            subtitle_position,
            subtitle_vertical_offset,
        )

    return {
        "api_key": api_key or None,
        "model": model,
        "tts_provider": tts_provider,
        "voice": voice,
        "rate": rate,
        "volume": volume,
        "pitch": pitch,
        "elevenlabs_api_key": elevenlabs_api_key,
        "elevenlabs_voice_id": elevenlabs_voice_id,
        "elevenlabs_model_id": elevenlabs_model_id,
        "elevenlabs_language_code": elevenlabs_language_code,
        "elevenlabs_stability": elevenlabs_stability,
        "elevenlabs_similarity_boost": elevenlabs_similarity_boost,
        "elevenlabs_style": elevenlabs_style,
        "elevenlabs_speaker_boost": elevenlabs_speaker_boost,
        "playback_speed": playback_speed,
        "auto_fit_tts_to_segment": auto_fit_tts_to_segment,
        "max_auto_tts_speed": max_auto_tts_speed,
        "original_volume": original_volume,
        "subtitle_style": SUBTITLE_PRESETS[subtitle_preset],
        "ass_config": position_config,
        "video_layout": video_layout,
        "source_crop": source_crop,
        "censor_regions": censor_regions,
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "crop_left": crop_left,
        "crop_right": crop_right,
        "crop_offset_y": crop_offset_y,
        "custom_font_path": custom_font_path or None,
        "custom_font_family_name": custom_font_family_name,
        "logo_enabled": logo_enabled,
        "logo_path": logo_path or None,
        "logo_scale_percent": logo_scale_percent,
        "logo_opacity": logo_opacity,
        "logo_anchor": logo_anchor,
        "logo_offset_x": logo_offset_x,
        "logo_offset_y": logo_offset_y,
        "tts_gap_seconds": tts_gap_seconds,
        "analysis_auto_fit": analysis_auto_fit,
        "timing_density": timing_density,
        "prevent_overlap": prevent_overlap,
        "min_gap_ms": min_gap_ms,
        "output_name": output_name,
        "prompt": st.session_state.get("custom_prompt", PROMPT_PRESETS[prompt_preset]),
        "cta_enabled": cta_enabled,
        "cta_text": cta_text,
        "karaoke_words": karaoke_words,
        "ui_language": ui_language,
    }


def render_upload_step() -> None:
    st.subheader("1. Video Yukle")
    uploaded_file = st.file_uploader(
        "MP4 video dosyasi",
        type=["mp4"],
        accept_multiple_files=False,
        help="Kaynak videoyu yukle. Dosya secilince is otomatik olusturulur.",
    )
    if uploaded_file:
        uploaded_signature = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("last_uploaded_signature") == uploaded_signature:
            st.info("Bu video icin is zaten olusturuldu.")
            return
        input_path = save_uploaded_video(uploaded_file)
        st.session_state.last_uploaded_signature = uploaded_signature
        st.success(f"Kaydedildi: {input_path}")
        st.rerun()


def render_analysis_step(settings: dict) -> None:
    st.subheader("2. Gemini Analizi")
    prompt = settings["prompt"]
    st.caption("Prompt duzenleme soldaki Yapay Zeka sekmesindedir.")
    disabled = not st.session_state.input_video

    if st.button(
        "Videoyu analiz et",
        disabled=disabled,
        use_container_width=True,
        help="MP4 dosyasini Gemini'ye gonderir ve zaman damgali anlatim JSON'u ister.",
    ):
        try:
            append_log("Gemini analysis started.")
            write_metadata(st.session_state.job_dir, status="analyzing")
            with st.spinner("Gemini videoyu analiz ediyor..."):
                segments = analyze_video_with_gemini(
                    video_path=st.session_state.input_video,
                    api_key=settings["api_key"],
                    model_name=settings["model"],
                    prompt=prompt,
                )
                if settings["prevent_overlap"] and has_overlaps(segments):
                    segments = normalize_segment_overlaps(
                        segments,
                        min_gap_ms=settings["min_gap_ms"],
                    )
                    append_log("Overlapping Gemini segments were normalized.")
                if settings["analysis_auto_fit"]:
                    before_texts = [segment.text for segment in segments]
                    segments = tighten_segments_to_timing(
                        segments,
                        speed_multiplier=effective_speech_speed(settings),
                        chars_per_second=settings["timing_density"],
                    )
                    if before_texts != [segment.text for segment in segments]:
                        append_log("Long Gemini lines were tightened to fit segment timing.")
                segments = apply_cta_segment(segments, settings, st.session_state.input_video)
                save_segments(st.session_state.job_dir, segments, settings["ass_config"])
            write_metadata(st.session_state.job_dir, status="analysis_ready")
            append_log(f"Gemini returned {len(segments)} segments.")
            st.success("Analiz tamamlandi.")
            st.rerun()
        except Exception as exc:
            append_log(f"Gemini analysis failed: {exc}")
            write_metadata(st.session_state.job_dir, status="analysis_failed", error=str(exc))
            st.error(str(exc))


def render_editor_step(settings: dict) -> None:
    st.subheader("3. Senaryo / Altyazi Editoru")
    rows = st.session_state.segments or [
        {"start_time": "00:00:00,000", "end_time": "00:00:05,000", "text": ""}
    ]
    dataframe = pd.DataFrame(rows, columns=["start_time", "end_time", "text"])

    edited = st.data_editor(
        dataframe,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        height=260,
        key=f"narration_editor_{st.session_state.editor_version}",
        column_config={
            "start_time": st.column_config.TextColumn(
                "Start",
                width="small",
                help="Segment baslangic zamani. Format: HH:MM:SS,mmm",
            ),
            "end_time": st.column_config.TextColumn(
                "End",
                width="small",
                help="Segment bitis zamani. Format: HH:MM:SS,mmm",
            ),
            "text": st.column_config.TextColumn(
                "Metin",
                width="medium",
                help="Okunabilir tut. Uzun metinler kesilmek yerine altyazi kartlarina bolunur.",
            ),
        },
    )

    editor_rows = edited.to_dict("records")
    warnings = quality_warnings(editor_rows, speed_multiplier=effective_speech_speed(settings))
    if warnings:
        with st.expander(f"Kalite kontrolleri ({len(warnings)})", expanded=False):
            st.dataframe(
                pd.DataFrame({"Oneri": warnings[:10]}),
                use_container_width=True,
                hide_index=True,
                height=min(280, 42 + len(warnings[:10]) * 35),
            )
            if len(warnings) > 10:
                st.caption(f"{len(warnings) - 10} ek madde gizlendi.")
    else:
        st.success("Zamanlama ve metin kullanilabilir gorunuyor.")

    col_clean, col_fit, col_save = st.columns(3)
    with col_clean:
        if st.button(
            "Metni temizle",
            disabled=not st.session_state.job_dir,
            use_container_width=True,
            help="Hashtag, fazla bosluk ve gereksiz noktalama temizler.",
        ):
            cleaned_rows = []
            for row in editor_rows:
                cleaned = dict(row)
                cleaned["text"] = clean_narration_text(str(cleaned.get("text", "") or ""))
                cleaned_rows.append(cleaned)
            st.session_state.segments = cleaned_rows
            st.session_state.editor_version += 1
            append_log("Narration text cleanup applied.")
            st.rerun()

    with col_fit:
        if st.button(
            "Sureye sigdir",
            disabled=not st.session_state.job_dir,
            use_container_width=True,
            help="Segment surelerine ve mevcut ses hizina gore fazla uzun satirlari kisaltir.",
        ):
            try:
                segments = rows_to_segments(
                    editor_rows,
                    normalize_overlaps=settings["prevent_overlap"],
                    min_gap_ms=settings["min_gap_ms"],
                )
                tightened = tighten_segments_to_timing(
                    segments,
                    speed_multiplier=effective_speech_speed(settings),
                    chars_per_second=settings["timing_density"],
                )
                tightened = apply_cta_segment(tightened, settings, st.session_state.input_video)
                st.session_state.segments = segments_to_rows(tightened)
                st.session_state.editor_version += 1
                append_log("Narration lines were tightened to fit segment timing.")
                st.rerun()
            except Exception as exc:
                append_log(f"Fit-to-timing failed: {exc}")
                st.error(str(exc))

    with col_save:
        save_clicked = st.button(
            "Altyazilari kaydet",
            disabled=not st.session_state.job_dir,
            use_container_width=True,
            help="Satirlari dogrular ve narration.json, SRT ve ASS dosyalarini yazar.",
        )

    if save_clicked:
        try:
            segments = rows_to_segments(
                editor_rows,
                normalize_overlaps=settings["prevent_overlap"],
                min_gap_ms=settings["min_gap_ms"],
            )
            segments = apply_cta_segment(segments, settings, st.session_state.input_video)
            srt_path = save_segments(st.session_state.job_dir, segments, settings["ass_config"])
            st.success(f"SRT kaydedildi: {srt_path}")
            st.rerun()
        except Exception as exc:
            append_log(f"Editor validation failed: {exc}")
            st.error(str(exc))

    if st.session_state.srt_path and Path(st.session_state.srt_path).exists():
        col_srt, col_ass = st.columns(2)
        with col_srt:
            render_file_download(
                "SRT indir",
                Path(st.session_state.srt_path),
                "application/x-subrip",
                "download_srt",
            )
        if st.session_state.ass_path and Path(st.session_state.ass_path).exists():
            with col_ass:
                render_file_download(
                    "ASS indir",
                    Path(st.session_state.ass_path),
                    "text/plain",
                    "download_ass",
                )


def render_tts_step(settings: dict) -> None:
    st.subheader("4. Seslendirme")
    disabled = not st.session_state.segments or not st.session_state.job_dir

    if st.button(
        "Ses dosyalarini olustur",
        disabled=disabled,
        use_container_width=True,
        help="Her anlatim segmenti icin bir ses dosyasi olusturur.",
    ):
        try:
            append_log("Seslendirme uretimi basladi.")
            write_metadata(st.session_state.job_dir, status="tts_generating")
            segments = rows_to_segments(
                st.session_state.segments,
                normalize_overlaps=settings["prevent_overlap"],
                min_gap_ms=settings["min_gap_ms"],
            )
            segments = apply_cta_segment(segments, settings, st.session_state.input_video)
            with st.spinner("Ses dosyalari olusturuluyor..."):
                tts_audio_paths, word_timings = generate_tts_package(
                    segments=segments,
                    output_dir=st.session_state.job_dir / "tts",
                    provider=settings["tts_provider"],
                    voice=settings["voice"],
                    rate=settings["rate"],
                    volume=settings["volume"],
                    pitch=settings["pitch"],
                    elevenlabs_api_key=settings["elevenlabs_api_key"],
                    elevenlabs_voice_id=settings["elevenlabs_voice_id"],
                    elevenlabs_model_id=settings["elevenlabs_model_id"],
                    elevenlabs_language_code=settings["elevenlabs_language_code"],
                    elevenlabs_stability=settings["elevenlabs_stability"],
                    elevenlabs_similarity_boost=settings["elevenlabs_similarity_boost"],
                    elevenlabs_style=settings["elevenlabs_style"],
                    elevenlabs_speaker_boost=settings["elevenlabs_speaker_boost"],
                    playback_speed=settings["playback_speed"],
                    auto_fit_to_segment=settings["auto_fit_tts_to_segment"],
                    max_auto_speed=settings["max_auto_tts_speed"],
                )
                video_duration_seconds = get_video_duration_seconds(Path(st.session_state.input_video))
                aligned_segments = align_segments_to_audio(
                    segments,
                    tts_audio_paths,
                    min_gap_ms=settings["min_gap_ms"],
                    gap_seconds=settings["tts_gap_seconds"],
                    max_end_ms=int(video_duration_seconds * 1000),
                )
                save_segments(st.session_state.job_dir, aligned_segments, settings["ass_config"], word_timings=word_timings)
            st.session_state.tts_audio_paths = tts_audio_paths
            st.session_state.word_timings = word_timings
            write_metadata(
                st.session_state.job_dir,
                status="tts_ready",
                tts_audio_paths=[str(path) for path in tts_audio_paths],
            )
            append_log(f"{len(tts_audio_paths)} ses dosyasi olusturuldu ve segment sureleri sese gore guncellendi.")
            st.success("Ses dosyalari olusturuldu.")
            st.rerun()
        except Exception as exc:
            append_log(f"TTS failed: {exc}")
            write_metadata(st.session_state.job_dir, status="tts_failed", error=str(exc))
            st.error(str(exc))

    if st.session_state.tts_audio_paths:
        with st.expander("Ses onizleme", expanded=False):
            for index, audio_path in enumerate(st.session_state.tts_audio_paths, start=1):
                st.caption(f"Segment {index}: {Path(audio_path).name}")
                st.audio(str(audio_path))


def render_render_step(settings: dict) -> None:
    st.subheader("5. Final Render")
    disabled = (
        not st.session_state.input_video
        or not (st.session_state.ass_path or st.session_state.srt_path)
        or not st.session_state.tts_audio_paths
    )

    if st.button(
        "Final MP4 olustur",
        disabled=disabled,
        type="primary",
        use_container_width=True,
        help="Orijinal sesi ve yeni sesi karistirir, altyaziyi videoya gomerek final MP4 olusturur.",
    ):
        try:
            append_log("FFmpeg render started.")
            write_metadata(st.session_state.job_dir, status="rendering")
            segments = rows_to_segments(
                st.session_state.segments,
                normalize_overlaps=settings["prevent_overlap"],
                min_gap_ms=settings["min_gap_ms"],
            )
            output_path = st.session_state.job_dir / settings["output_name"]
            subtitle_path = write_animated_ass(
                segments,
                st.session_state.job_dir / "narration.ass",
                settings["ass_config"],
                word_timings_by_segment=st.session_state.get("word_timings", []),
            )
            st.session_state.ass_path = subtitle_path

            with st.spinner("FFmpeg ile final video olusturuluyor..."):
                final_output = render_final_video(
                    input_video=st.session_state.input_video,
                    srt_path=subtitle_path,
                    tts_audio_paths=[Path(path) for path in st.session_state.tts_audio_paths],
                    segments=segments,
                    output_path=output_path,
                    original_volume=settings["original_volume"],
                    subtitle_style=settings["subtitle_style"],
                    tts_gap_seconds=settings["tts_gap_seconds"],
                    video_layout=settings["video_layout"],
                    subtitle_fonts_dir=(Path(settings["custom_font_path"]).parent if settings.get("custom_font_path") else None),
                    source_crop=settings["source_crop"],
                    censor_regions=settings["censor_regions"],
                    logo_overlay=(
                        LogoOverlayConfig(
                            path=Path(settings["logo_path"]),
                            scale_percent=settings["logo_scale_percent"],
                            opacity=settings["logo_opacity"],
                            anchor=settings["logo_anchor"],
                            offset_x=settings["logo_offset_x"],
                            offset_y=settings["logo_offset_y"],
                        )
                        if settings.get("logo_enabled") and settings.get("logo_path")
                        else None
                    ),
                )

            st.session_state.final_output = final_output
            write_metadata(st.session_state.job_dir, status="done", final_output=str(final_output))
            append_log(f"Final render ready: {final_output}")
            st.success("Final video olusturuldu.")
            st.rerun()
        except Exception as exc:
            append_log(f"Render failed: {exc}")
            write_metadata(st.session_state.job_dir, status="render_failed", error=str(exc))
            st.error(str(exc))

    if st.session_state.final_output and Path(st.session_state.final_output).exists():
        final_path = Path(st.session_state.final_output)
        st.video(str(final_path))
        render_file_download("Final MP4 indir", final_path, "video/mp4", "download_final_mp4")


def collect_output_files() -> list[tuple[str, Path]]:
    job_dir = st.session_state.get("job_dir")
    if not job_dir:
        return []

    candidates = [
        ("Final MP4", st.session_state.get("final_output")),
        ("Narration JSON", job_dir / "narration.json"),
        ("YouTube Metadata", job_dir / "youtube_metadata.json"),
        ("Plain SRT", st.session_state.get("srt_path")),
        ("Animated ASS", st.session_state.get("ass_path")),
    ]

    outputs: list[tuple[str, Path]] = []
    for label, path in candidates:
        if path and Path(path).exists():
            outputs.append((label, Path(path)))
    return outputs


def render_youtube_metadata_panel(job_dir: Path) -> None:
    metadata = st.session_state.get("publish_metadata")
    metadata_path = job_dir / "youtube_metadata.json"

    if metadata is None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        st.session_state.publish_metadata = metadata

    if metadata is None and st.session_state.segments:
        segments = rows_to_segments(st.session_state.segments)
        metadata = generate_youtube_metadata(
            segments,
            source_name=Path(st.session_state.input_video).name if st.session_state.input_video else None,
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        st.session_state.publish_metadata = metadata

    with st.expander("YouTube Shorts yayın metni", expanded=True):
        if not metadata:
            st.caption("Başlık, açıklama ve etiket üretmek için önce Gemini analizi veya altyazı kaydı yap.")
            return

        if st.button("Yayın metnini yeniden üret", use_container_width=True):
            segments = rows_to_segments(st.session_state.segments)
            metadata = generate_youtube_metadata(
                segments,
                source_name=Path(st.session_state.input_video).name if st.session_state.input_video else None,
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            st.session_state.publish_metadata = metadata
            append_log("YouTube metadata regenerated.")
            st.rerun()

        title = metadata.get("title", "")
        description = metadata.get("description", "")
        tags = metadata.get("tags", [])
        tags_text = ", ".join(tags)
        combined = f"Başlık:\n{title}\n\nAçıklama:\n{description}\n\nEtiketler:\n{tags_text}"

        st.text_area("Başlık", value=title, height=78, key="copy_title")
        st.text_area("Açıklama", value=description, height=170, key="copy_description")
        st.text_area("Etiketler", value=tags_text, height=90, key="copy_tags")
        st.text_area("Hepsi birlikte", value=combined, height=260, key="copy_all_metadata")


def render_outputs_panel() -> None:
    st.subheader("Ciktilar")
    job_dir = st.session_state.job_dir
    if not job_dir:
        st.caption("Cikti yollarini gormek icin once bir is olustur.")
        return

    st.caption("Job klasoru")
    st.code(str(job_dir), language="text")

    render_youtube_metadata_panel(job_dir)

    outputs = collect_output_files()
    if outputs:
        for label, path in outputs:
            with st.expander(label, expanded=(label == "Final MP4")):
                mime = "video/mp4" if path.suffix.lower() == ".mp4" else "text/plain"
                render_file_download(f"{label} indir", path, mime, f"download_output_{label.lower().replace(' ', '_')}")
    else:
        st.caption("Henuz cikti dosyasi yok.")

    tts_dir = job_dir / "tts"
    if tts_dir.exists():
        tts_files = sorted(tts_dir.glob("*.mp3"))
        with st.expander(f"Ses klasoru ({len(tts_files)} dosya)", expanded=False):
            st.code(str(tts_dir), language="text")
            for audio_path in tts_files[:8]:
                st.caption(audio_path.name)
            if len(tts_files) > 8:
                st.caption(f"{len(tts_files) - 8} dosya daha gizlendi.")


def render_status_panel() -> None:
    st.subheader("Durum")
    job_dir = st.session_state.job_dir
    segments_count = len(st.session_state.segments)
    tts_count = len(st.session_state.tts_audio_paths)
    final_ready = bool(st.session_state.final_output and Path(st.session_state.final_output).exists())

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Segment", segments_count)
    col_b.metric("Ses dosyasi", tts_count)
    col_c.metric("Final", "Hazir" if final_ready else "Bekliyor")

    logs = escape("\n".join(st.session_state.logs[-30:]) or "Henuz olay yok.")
    st.markdown(f'<div class="pipeline-log">{logs}</div>', unsafe_allow_html=True)


def render_monitor_panel(settings: dict) -> None:
    st.subheader("Monitor")
    tabs = st.tabs(["Onizleme", "Ciktilar", "Loglar"])

    with tabs[0]:
        st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
        chip_values = [
            f"Model: {settings['model']}",
            f"Ses: {settings['voice']}",
            f"Speed x{settings['playback_speed']:.2f}",
            f"Layout: {settings['video_layout'].mode}",
        ]
        st.markdown(
            '<div class="studio-chip-row">' + "".join(
                f'<span class="studio-chip">{escape(value)}</span>' for value in chip_values
            ) + "</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.final_output and Path(st.session_state.final_output).exists():
            st.caption("Final onizleme")
            st.video(str(st.session_state.final_output))
        elif st.session_state.input_video:
            st.caption("Kaynak onizleme")
            st.video(str(st.session_state.input_video))
        else:
            st.caption("Canli onizleme alanini gormek icin once bir video yukle.")

        st.caption("Altyazi stili onizlemesi")
        render_video_subtitle_preview(
            video_path=st.session_state.input_video,
            text_color=settings["ass_config"].primary_color,
            accent_color=settings["ass_config"].accent_color,
            outline_color=settings["ass_config"].outline_color,
            font_size=settings["ass_config"].font_size,
            font_family=settings["ass_config"].font_name,
            font_weight=max(700, int(round(settings["ass_config"].outline * 120))),
            custom_font_path=settings.get("custom_font_path"),
            bold=settings["ass_config"].bold,
            shadow_3d=settings["ass_config"].shadow_3d,
            shadow_3d_depth=settings["ass_config"].shadow_3d_depth,
            boxed_background=settings["ass_config"].boxed_background,
            box_mode=settings["ass_config"].box_mode,
            box_alpha=255 - settings["ass_config"].back_alpha,
            box_padding_px=settings["ass_config"].box_padding_px,
            box_blur=settings["ass_config"].box_blur,
            position={
                2: "Bottom",
                5: "Middle",
                8: "Top",
            }.get(settings["ass_config"].alignment, "Bottom"),
            glow=settings["ass_config"].glow,
            karaoke_words=settings.get("karaoke_words", False),
            max_chars=settings["ass_config"].max_chars,
            vertical_offset_px=int(st.session_state.get("subtitle_vertical_offset", 0)),
            video_layout=settings["video_layout"],
            logo_path=settings.get("logo_path") if settings.get("logo_enabled") else None,
            logo_scale_percent=settings.get("logo_scale_percent", 12),
            logo_opacity=settings.get("logo_opacity", 1.0),
            logo_anchor=settings.get("logo_anchor", "Top Right"),
            logo_offset_x=settings.get("logo_offset_x", 36),
            logo_offset_y=settings.get("logo_offset_y", 36),
            source_crop=settings.get("source_crop"),
            censor_regions=settings.get("censor_regions"),
        )

        if st.session_state.segments:
            rows = st.session_state.segments
            active_index = min(st.session_state.preview_segment_index, len(rows) - 1)
            active_segment = rows[active_index]
            st.caption("Secili segment")
            st.write(clean_preview_text(active_segment.get("text", "")))
            st.caption(f"{active_segment.get('start_time', '')} -> {active_segment.get('end_time', '')}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        render_outputs_panel()

    with tabs[2]:
        render_status_panel()


def main() -> None:
    init_state()
    render_header()
    settings = render_sidebar()
    persist_runtime_settings(settings)

    left, right = st.columns([0.64, 0.36], gap="large")
    with left:
        workflow_tabs = st.tabs(["Kaynak", "Senaryo", "Ses", "Render"])
        with workflow_tabs[0]:
            render_upload_step()
            st.divider()
            render_analysis_step(settings)
        with workflow_tabs[1]:
            render_editor_step(settings)
        with workflow_tabs[2]:
            render_tts_step(settings)
        with workflow_tabs[3]:
            render_render_step(settings)

    with right:
        render_monitor_panel(settings)


if __name__ == "__main__":
    main()

