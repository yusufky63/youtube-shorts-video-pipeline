import os
import json
import time
from pathlib import Path

import google.generativeai as genai

from .json_parser import parse_gemini_segments
from .models import SubtitleSegment


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

CENSOR_DETECTION_PROMPT = """
Bu videoda sansurlenmesi gereken logo, marka, kullanici adi, isim etiketi, plaka veya hassas metin bolgelerini bul.
Koordinatlari final YouTube Shorts ciktisi 1080x1920 kabul ederek tahmini ver.
Bana sadece gecerli JSON array don:
[{"start_time":"00:00:00,000","end_time":"00:00:05,000","x":820,"y":60,"width":220,"height":90,"blur":18,"reason":"logo"}]

Kurallar:
- Sadece JSON don; markdown, aciklama veya not ekleme.
- start_time ve end_time formati HH:MM:SS,mmm olsun.
- x, y, width, height 1080x1920 final kadraj koordinatlari olsun.
- Emin olmadigin bolgeleri ekleme.
- Video boyunca sabit duran logo varsa start_time 00:00:00,000 kullan ve end_time'i videonun sonuna yakin ver.
- En fazla 8 bolge don.
""".strip()

DEPRECATED_MODEL_ALIASES = {
    "gemini-1.5-flash": DEFAULT_GEMINI_MODEL,
    "models/gemini-1.5-flash": DEFAULT_GEMINI_MODEL,
}

ANALYSIS_PROMPT = """
Bu videoyu analiz et ve ekranda gercekten gorunen olaylara dayali dogal Turkce bir seslendirme yaz.
Ton: ciddi belgesel anlatimi + kuru/absurt mizah. Video komikse mizah, olayin kendisinden gelsin.
Bana sadece su formatta gecici aciklama olmadan JSON don:
[{"start_time": "00:00:00,000", "end_time": "00:00:05,000", "text": "Cumle 1..."}, ...]

Kurallar:
- Sadece gecerli JSON dondur; markdown, onsoz, not veya aciklama ekleme.
- start_time ve end_time formati kesinlikle HH:MM:SS,mmm olsun.
- Her segment 3-6 saniye araliginda olsun; cok hizli kesme yapma.
- Her text en fazla 95 karakter olsun.
- Her text tek kisa cumle olsun; uzun aciklama yazma.
- Her cumle Turkiye Turkcesine uygun, dogal ve sesli okunabilir olsun.
- Bozuk, yapay veya kelime kelime ceviri gibi duran kaliplar kullanma.
- "hesaplama hatasi payi yok", "durumun matematiksel sonucu", "operasyonel denge" gibi anlamsiz kaliplar yazma.
- Turkcede dogal olmayan isim tamlamalari, eksik fiiller ve yarim cumleler kullanma.
- Cumleyi yazdiktan sonra icinden oku; kulağa garip geliyorsa daha sade bir Turkceyle yeniden kur.
- Ozne, fiil ve nesne uyumlu olsun; anlatim mantikli bir sebep-sonuc iliskisi kursun.
- Absurt mizah yaparken bile olayla baglantisiz teknik veya akademik kelimeler uydurma.
- Anlasilir ve sade kelimeler kullan: "adam", "kedi", "araba", "zemin", "hamle", "denge" gibi ekranda gorunen seylerden bahset.
- Metinler ayri ayri caption gibi degil, tek bir belgesel anlatiminin parcasi gibi aksin.
- Once videonun genel akisini zihninde planla; JSON'da sadece bu akisin bolunmus cumlelerini ver.
- Ayni ana ozneyi ve olay cizgisini koru; her segment onceki cumlenin devami gibi hissedilsin.
- Her yeni segmentte konuyu sifirdan acma; ozneyi ve zamani durduk yere degistirme.
- Giris, gelisme ve final hissi tek akisin parcasi olsun; birbirinden kopuk tek satirlar yazma.
- Gecisleri dogal kur: "bu sirada", "ardindan", "fakat", "derken" gibi baglaclari gerektikce kullan.
- Her segmenti yeni bir saka gibi baslatma; mizah biriken anlatinin sonucundan gelsin.
- Ekranda olmayan kisileri, markalari, niyetleri veya olaylari uydurma.
- Kaba hakaret, kufur, nefret, cinsel icerik ve politik yorum ekleme.
- Mizah "cringe" olmasin: emoji, hashtag, internet argosu, bagiran unlem zinciri kullanma.
- Olayi kisa tarif et, sonra gerekiyorsa tek kuru yorum ekle; baglamdan kopma.
- Video sessiz veya karisiksa bile metin akici ve seslendirilebilir olsun.
- Zaman damgalari videodaki olay akisina yakin dursun.
""".strip()


def _wait_until_file_active(uploaded_file, timeout_seconds: int = 300):
    """Poll the Gemini File API until the uploaded video is ready."""
    deadline = time.time() + timeout_seconds
    current_file = uploaded_file

    while getattr(current_file, "state", None) and current_file.state.name == "PROCESSING":
        if time.time() > deadline:
            raise TimeoutError("Gemini video processing timed out")
        time.sleep(5)
        current_file = genai.get_file(current_file.name)

    if getattr(current_file, "state", None) and current_file.state.name == "FAILED":
        raise RuntimeError("Gemini could not process the uploaded video")

    return current_file


def normalize_model_name(model_name: str) -> str:
    """Accept both 'gemini-x' and 'models/gemini-x' values from ListModels."""
    cleaned = model_name.strip()
    if cleaned in DEPRECATED_MODEL_ALIASES:
        return DEPRECATED_MODEL_ALIASES[cleaned]
    return cleaned.removeprefix("models/")


def list_generate_content_models(api_key: str | None = None) -> list[str]:
    """List Gemini models that support generateContent for the active API key."""
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("Set GEMINI_API_KEY or enter a Gemini API key first")

    try:
        genai.configure(api_key=resolved_api_key)
        models = []
        for model in genai.list_models():
            methods = getattr(model, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                models.append(normalize_model_name(model.name))
        return sorted(models)
    except Exception as exc:
        raise RuntimeError(f"Could not list Gemini models: {exc}") from exc


def analyze_video_with_gemini(
    video_path: Path,
    api_key: str | None = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    prompt: str = ANALYSIS_PROMPT,
) -> list[SubtitleSegment]:
    """Upload a video to Gemini and return validated timed narration segments."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("Set GEMINI_API_KEY or pass --api-key before running")

    uploaded_file = None
    try:
        genai.configure(api_key=resolved_api_key)
        uploaded_file = genai.upload_file(path=str(video_path), mime_type="video/mp4")
        active_file = _wait_until_file_active(uploaded_file)

        model = genai.GenerativeModel(normalize_model_name(model_name))
        response = model.generate_content(
            [prompt, active_file],
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 600},
        )

        response_text = getattr(response, "text", "")
        if not response_text:
            raise RuntimeError("Gemini returned an empty response")

        return parse_gemini_segments(response_text)
    except Exception as exc:
        raise RuntimeError(f"Gemini analysis failed: {exc}") from exc
    finally:
        # Uploaded videos count against File API storage; cleanup is best-effort.
        if uploaded_file is not None:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass


def detect_censor_regions_with_gemini(
    video_path: Path,
    api_key: str | None = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    prompt: str = CENSOR_DETECTION_PROMPT,
) -> list[dict]:
    """Upload a video to Gemini and return suggested censor boxes."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("Set GEMINI_API_KEY or enter a Gemini API key first")

    uploaded_file = None
    try:
        genai.configure(api_key=resolved_api_key)
        uploaded_file = genai.upload_file(path=str(video_path), mime_type="video/mp4")
        active_file = _wait_until_file_active(uploaded_file)
        model = genai.GenerativeModel(normalize_model_name(model_name))
        response = model.generate_content(
            [prompt, active_file],
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 600},
        )
        response_text = getattr(response, "text", "")
        if not response_text:
            return []
        payload = json.loads(response_text)
        if not isinstance(payload, list):
            return []
        rows: list[dict] = []
        for item in payload[:8]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "enabled": True,
                    "start_time": str(item.get("start_time", "00:00:00,000")),
                    "end_time": str(item.get("end_time", "00:00:05,000")),
                    "x": int(item.get("x", 0) or 0),
                    "y": int(item.get("y", 0) or 0),
                    "width": int(item.get("width", 200) or 200),
                    "height": int(item.get("height", 100) or 100),
                    "blur": int(item.get("blur", 18) or 18),
                }
            )
        return rows
    except Exception as exc:
        raise RuntimeError(f"Gemini censor detection failed: {exc}") from exc
    finally:
        if uploaded_file is not None:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass
