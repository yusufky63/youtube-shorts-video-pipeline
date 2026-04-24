import json
import re
from collections import Counter
from pathlib import Path

from .models import SubtitleSegment


CTA_RE = re.compile(
    r"izledi[gğ]iniz\s+i[cç]in\s+te[sş]ekk[uü]rler|be[gğ]enmeyi|abone\s+olmay[iı]",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]{3,}")

STOPWORDS = {
    "ama",
    "artık",
    "aslında",
    "bile",
    "bir",
    "biraz",
    "böyle",
    "burada",
    "bunu",
    "daha",
    "değil",
    "diye",
    "fakat",
    "gibi",
    "halen",
    "için",
    "ile",
    "ise",
    "kadar",
    "kendi",
    "sonra",
    "şimdi",
    "tam",
    "veya",
    "yani",
}

DEFAULT_TAGS = [
    "shorts",
    "youtube shorts",
    "komik video",
    "belgesel anlatım",
    "absürt mizah",
    "türkçe shorts",
    "viral shorts",
]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace("...", "").strip(" -")
    return text


def _trim_to_words(text: str, limit: int) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    trimmed = text[: limit + 1].rsplit(" ", 1)[0].strip(" ,.-")
    return trimmed or text[:limit].strip()


def _narration_text(segments: list[SubtitleSegment]) -> str:
    usable = [_clean_text(segment.text) for segment in segments if not CTA_RE.search(segment.text)]
    return " ".join(text for text in usable if text)


def extract_keywords(segments: list[SubtitleSegment], limit: int = 10) -> list[str]:
    words = []
    for word in WORD_RE.findall(_narration_text(segments).lower()):
        if word in STOPWORDS:
            continue
        if word.isdigit():
            continue
        words.append(word)

    counts = Counter(words)
    keywords = [word for word, _ in counts.most_common(limit)]
    return keywords


def generate_youtube_metadata(segments: list[SubtitleSegment], source_name: str | None = None) -> dict:
    narration = _narration_text(segments)
    first_line = _clean_text(next((segment.text for segment in segments if not CTA_RE.search(segment.text)), ""))
    source_stem = Path(source_name).stem if source_name else ""

    title_base = first_line or source_stem or "Komik Belgesel Anı"
    title = _trim_to_words(title_base, 68)
    if title and not title.endswith(("?", "!")):
        title = title.rstrip(".")

    summary = _trim_to_words(narration, 260)
    if not summary:
        summary = "Kısa video için hazırlanmış komik belgesel anlatımı."

    keywords = extract_keywords(segments, limit=10)
    keyword_tags = [word for word in keywords if word not in DEFAULT_TAGS]
    tags = (keyword_tags + DEFAULT_TAGS)[:15]
    hashtags = ["#shorts", "#komik", "#belgesel", "#türkçe"]

    description = "\n".join(
        [
            summary,
            "",
            "İzlediğiniz için teşekkürler. Beğenmeyi ve abone olmayı unutmayın.",
            " ".join(hashtags),
        ]
    )

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "hashtags": hashtags,
    }


def write_youtube_metadata(
    segments: list[SubtitleSegment],
    output_path: Path,
    source_name: str | None = None,
) -> Path:
    metadata = generate_youtube_metadata(segments, source_name=source_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
