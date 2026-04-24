import asyncio
import base64
import os
import re
import subprocess
from pathlib import Path

import edge_tts
import requests

from .ffmpeg_renderer import resolve_ffmpeg_binary
from .models import SubtitleSegment
from .text_utils import remove_pause_punctuation, turkish_lower
from .timecode import parse_srt_time


DEFAULT_TTS_VOICE = "tr-TR-AhmetNeural"
DEFAULT_TTS_RATE = "+8%"
DEFAULT_TTS_VOLUME = "+12%"
DEFAULT_TTS_PITCH = "-12Hz"
DEFAULT_TTS_PROVIDER = "edge"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
CTA_TEXT_RE = re.compile(
    r"izledi[gğ]iniz|be[gğ]enmeyi|abone\s+olmay[iı]|te[sş]ekk[uü]rler",
    re.IGNORECASE,
)


async def _list_voices() -> list[dict]:
    return await edge_tts.list_voices()


def list_edge_voices(locale_prefix: str | None = None, gender: str | None = None) -> list[dict]:
    """Fetch Edge TTS voices, optionally filtered by locale prefix and gender."""
    try:
        voices = asyncio.run(_list_voices())
        if locale_prefix:
            voices = [
                voice for voice in voices
                if str(voice.get("Locale", "")).lower().startswith(locale_prefix.lower())
            ]
        if gender:
            voices = [
                voice for voice in voices
                if str(voice.get("Gender", "")).lower() == gender.lower()
            ]
        return sorted(voices, key=lambda voice: str(voice.get("ShortName", "")))
    except Exception as exc:
        raise RuntimeError(f"Edge TTS voice list failed: {exc}") from exc


def list_elevenlabs_voices(api_key: str | None = None) -> list[dict]:
    """Fetch ElevenLabs voices available for the API key."""
    resolved_api_key = (api_key or os.getenv("ELEVENLABS_API_KEY") or "").strip()
    if not resolved_api_key:
        raise RuntimeError("Set ELEVENLABS_API_KEY or enter an ElevenLabs API key")

    try:
        response = requests.get(
            "https://api.elevenlabs.io/v2/voices",
            headers={"xi-api-key": resolved_api_key},
            timeout=45,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "ElevenLabs rejected the API key with 401 Unauthorized. "
                "Check that the key is real, active, and copied without spaces."
            )
        response.raise_for_status()
        voices = response.json().get("voices", [])
        return sorted(voices, key=lambda voice: str(voice.get("name", "")))
    except requests.RequestException as exc:
        raise RuntimeError(f"ElevenLabs voice list failed: {exc}") from exc


def _extract_word_timings_from_alignment(
    alignment: dict | None,
    speed_multiplier: float = 1.0,
) -> list[dict]:
    if not alignment:
        return []

    characters = alignment.get("characters") or []
    start_times = alignment.get("character_start_times_seconds") or []
    end_times = alignment.get("character_end_times_seconds") or []
    limit = min(len(characters), len(start_times), len(end_times))
    if limit <= 0:
        return []

    safe_speed = max(0.5, speed_multiplier)
    words: list[dict] = []
    current_chars: list[str] = []
    current_start_ms: int | None = None
    current_end_ms: int | None = None

    for index in range(limit):
        character = str(characters[index])
        start_ms = int(float(start_times[index]) * 1000 / safe_speed)
        end_ms = int(float(end_times[index]) * 1000 / safe_speed)
        if character.isspace():
            if current_chars and current_start_ms is not None and current_end_ms is not None:
                words.append(
                    {
                        "text": "".join(current_chars),
                        "start_ms": current_start_ms,
                        "end_ms": current_end_ms,
                    }
                )
            current_chars = []
            current_start_ms = None
            current_end_ms = None
            continue

        if current_start_ms is None:
            current_start_ms = start_ms
        current_end_ms = end_ms
        current_chars.append(character)

    if current_chars and current_start_ms is not None and current_end_ms is not None:
        words.append(
            {
                "text": "".join(current_chars),
                "start_ms": current_start_ms,
                "end_ms": current_end_ms,
            }
        )
    return words


def _save_elevenlabs_audio(
    text: str,
    output_path: Path,
    api_key: str,
    voice_id: str,
    model_id: str = DEFAULT_ELEVENLABS_MODEL,
    language_code: str = "tr",
    stability: float = 0.68,
    similarity_boost: float = 0.86,
    style: float = 0.08,
    use_speaker_boost: bool = True,
    output_format: str = DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    previous_text: str | None = None,
    next_text: str | None = None,
    with_timestamps: bool = False,
) -> tuple[Path, list[dict]]:
    api_key = api_key.strip()
    voice_id = voice_id.strip()
    if not api_key:
        raise RuntimeError("ElevenLabs API key is required")
    if not voice_id:
        raise RuntimeError("ElevenLabs voice_id is required")

    endpoint = "with-timestamps" if with_timestamps else ""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    if endpoint:
        url = f"{url}/{endpoint}"
    payload = {
        "text": text,
        "model_id": model_id,
        "language_code": language_code or None,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        },
    }
    if previous_text:
        payload["previous_text"] = previous_text
    if next_text:
        payload["next_text"] = next_text
    response = requests.post(
        url,
        params={"output_format": output_format},
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json" if with_timestamps else "audio/mpeg",
        },
        json=payload,
        timeout=120,
    )
    try:
        if response.status_code == 401:
            raise RuntimeError(
                "ElevenLabs rejected the API key with 401 Unauthorized. "
                "Check that the key is real, active, and copied without spaces."
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = response.text[:500] if response.text else str(exc)
        raise RuntimeError(f"ElevenLabs TTS failed: {detail}") from exc

    if with_timestamps:
        payload = response.json()
        audio_base64 = payload.get("audio_base64", "")
        output_path.write_bytes(base64.b64decode(audio_base64))
        return output_path, _extract_word_timings_from_alignment(payload.get("alignment"))

    output_path.write_bytes(response.content)
    return output_path, []


async def _save_segment_audio(
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    volume: str,
    pitch: str,
) -> Path:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
    )
    await communicate.save(str(output_path))
    return output_path


def read_media_duration_seconds(media_path: Path) -> float:
    result = subprocess.run(
        [resolve_ffmpeg_binary(), "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    info = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", info)
    if not match:
        return 0.0
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


def _atempo_chain(speed: float) -> str:
    safe_speed = max(0.5, min(2.0, speed))
    filters: list[str] = []
    while safe_speed > 2.0:
        filters.append("atempo=2.0")
        safe_speed /= 2.0
    while safe_speed < 0.5:
        filters.append("atempo=0.5")
        safe_speed /= 0.5
    filters.append(f"atempo={safe_speed:.4f}")
    return ",".join(filters)


def _apply_audio_speed(audio_path: Path, speed: float) -> None:
    if abs(speed - 1.0) < 0.01:
        return
    temp_path = audio_path.with_name(f"{audio_path.stem}.speed.mp3")
    subprocess.run(
        [
            resolve_ffmpeg_binary(),
            "-y",
            "-i",
            str(audio_path),
            "-filter:a",
            _atempo_chain(speed),
            str(temp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    temp_path.replace(audio_path)


def _apply_audio_gain(audio_path: Path, gain: float) -> None:
    if abs(gain - 1.0) < 0.01:
        return
    temp_path = audio_path.with_name(f"{audio_path.stem}.gain.mp3")
    subprocess.run(
        [
            resolve_ffmpeg_binary(),
            "-y",
            "-i",
            str(audio_path),
            "-filter:a",
            f"volume={max(0.2, min(3.0, gain)):.3f}",
            str(temp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    temp_path.replace(audio_path)


def _postprocess_tts_audio(audio_path: Path, gain: float = 1.15) -> None:
    """Make separately generated TTS clips sit at a consistent perceived loudness."""
    temp_path = audio_path.with_name(f"{audio_path.stem}.post.mp3")
    filter_chain = ",".join(
        [
            "loudnorm=I=-15.5:TP=-1.5:LRA=7",
            "acompressor=threshold=-20dB:ratio=2.4:attack=3:release=90:makeup=1.8",
            f"volume={max(0.2, min(2.5, gain)):.3f}",
            "alimiter=limit=0.95",
        ]
    )
    subprocess.run(
        [
            resolve_ffmpeg_binary(),
            "-y",
            "-i",
            str(audio_path),
            "-filter:a",
            filter_chain,
            str(temp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    temp_path.replace(audio_path)


def _segment_output_gain(text: str, default_gain: float) -> float:
    if CTA_TEXT_RE.search(text or ""):
        return min(default_gain, 0.98)
    return default_gain


def generate_tts_files(
    segments: list[SubtitleSegment],
    output_dir: Path,
    provider: str = DEFAULT_TTS_PROVIDER,
    voice: str = DEFAULT_TTS_VOICE,
    rate: str = DEFAULT_TTS_RATE,
    volume: str = DEFAULT_TTS_VOLUME,
    pitch: str = DEFAULT_TTS_PITCH,
    elevenlabs_api_key: str | None = None,
    elevenlabs_voice_id: str | None = None,
    elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL,
    elevenlabs_language_code: str = "tr",
    elevenlabs_stability: float = 0.68,
    elevenlabs_similarity_boost: float = 0.86,
    elevenlabs_style: float = 0.08,
    elevenlabs_speaker_boost: bool = True,
    playback_speed: float = 1.0,
    auto_fit_to_segment: bool = False,
    max_auto_speed: float = 1.35,
    fit_padding_seconds: float = 0.12,
    output_gain: float = 1.18,
) -> list[Path]:
    audio_paths, _ = generate_tts_package(
        segments=segments,
        output_dir=output_dir,
        provider=provider,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        elevenlabs_api_key=elevenlabs_api_key,
        elevenlabs_voice_id=elevenlabs_voice_id,
        elevenlabs_model_id=elevenlabs_model_id,
        elevenlabs_language_code=elevenlabs_language_code,
        elevenlabs_stability=elevenlabs_stability,
        elevenlabs_similarity_boost=elevenlabs_similarity_boost,
        elevenlabs_style=elevenlabs_style,
        elevenlabs_speaker_boost=elevenlabs_speaker_boost,
        playback_speed=playback_speed,
        auto_fit_to_segment=auto_fit_to_segment,
        max_auto_speed=max_auto_speed,
        fit_padding_seconds=fit_padding_seconds,
        output_gain=output_gain,
    )
    return audio_paths


def generate_tts_package(
    segments: list[SubtitleSegment],
    output_dir: Path,
    provider: str = DEFAULT_TTS_PROVIDER,
    voice: str = DEFAULT_TTS_VOICE,
    rate: str = DEFAULT_TTS_RATE,
    volume: str = DEFAULT_TTS_VOLUME,
    pitch: str = DEFAULT_TTS_PITCH,
    elevenlabs_api_key: str | None = None,
    elevenlabs_voice_id: str | None = None,
    elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL,
    elevenlabs_language_code: str = "tr",
    elevenlabs_stability: float = 0.68,
    elevenlabs_similarity_boost: float = 0.86,
    elevenlabs_style: float = 0.08,
    elevenlabs_speaker_boost: bool = True,
    playback_speed: float = 1.0,
    auto_fit_to_segment: bool = False,
    max_auto_speed: float = 1.35,
    fit_padding_seconds: float = 0.12,
    output_gain: float = 1.18,
) -> tuple[list[Path], list[list[dict]]]:
    """Generate one MP3 narration file per segment with Edge TTS."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_paths: list[Path] = []
        word_timings_by_segment: list[list[dict]] = []

        for index, segment in enumerate(segments, start=1):
            audio_path = output_dir / f"tts_{index:03}.mp3"
            effective_speed = max(0.5, playback_speed)
            clip_word_timings: list[dict] = []
            speech_text = remove_pause_punctuation(turkish_lower(segment.text))
            if provider == "elevenlabs":
                previous_text = remove_pause_punctuation(turkish_lower(segments[index - 2].text)) if index > 1 else None
                next_text = remove_pause_punctuation(turkish_lower(segments[index].text)) if index < len(segments) else None
                _, clip_word_timings = _save_elevenlabs_audio(
                    text=speech_text,
                    output_path=audio_path,
                    api_key=elevenlabs_api_key or "",
                    voice_id=elevenlabs_voice_id or voice,
                    model_id=elevenlabs_model_id,
                    language_code=elevenlabs_language_code,
                    stability=elevenlabs_stability,
                    similarity_boost=elevenlabs_similarity_boost,
                    style=elevenlabs_style,
                    use_speaker_boost=elevenlabs_speaker_boost,
                    previous_text=previous_text,
                    next_text=next_text,
                    with_timestamps=True,
                )
            else:
                asyncio.run(
                    _save_segment_audio(
                        text=speech_text,
                        output_path=audio_path,
                        voice=voice,
                        rate=rate,
                        volume=volume,
                        pitch=pitch,
                    )
                )
            if auto_fit_to_segment:
                segment_duration = max(
                    0.25,
                    (parse_srt_time(segment.end_time) - parse_srt_time(segment.start_time)) / 1000 - fit_padding_seconds,
                )
                audio_duration = read_media_duration_seconds(audio_path)
                if audio_duration > segment_duration and segment_duration > 0:
                    effective_speed = max(
                        effective_speed,
                        min(max_auto_speed, audio_duration / segment_duration),
                    )
            _apply_audio_speed(audio_path, effective_speed)
            _postprocess_tts_audio(audio_path, _segment_output_gain(segment.text, output_gain))
            if clip_word_timings:
                clip_word_timings = [
                    {
                        **word_timing,
                        "start_ms": int(word_timing["start_ms"] / effective_speed),
                        "end_ms": int(word_timing["end_ms"] / effective_speed),
                    }
                    for word_timing in clip_word_timings
                ]
            audio_paths.append(audio_path)
            word_timings_by_segment.append(clip_word_timings)

        return audio_paths, word_timings_by_segment
    except Exception as exc:
        raise RuntimeError(f"Edge TTS generation failed: {exc}") from exc
