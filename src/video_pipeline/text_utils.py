import re


TURKISH_UPPER_MAP = str.maketrans(
    {
        "i": "\u0130",
        "\u0131": "I",
        "\u011f": "\u011e",
        "\u00fc": "\u00dc",
        "\u015f": "\u015e",
        "\u00f6": "\u00d6",
        "\u00e7": "\u00c7",
    }
)

TURKISH_LOWER_MAP = str.maketrans(
    {
        "I": "\u0131",
        "\u0130": "i",
        "\u011e": "\u011f",
        "\u00dc": "\u00fc",
        "\u015e": "\u015f",
        "\u00d6": "\u00f6",
        "\u00c7": "\u00e7",
    }
)

PAUSE_PUNCTUATION_RE = re.compile(r"[.,;:!?…]+")


def turkish_upper(text: str) -> str:
    return str(text or "").translate(TURKISH_UPPER_MAP).upper()


def turkish_lower(text: str) -> str:
    return str(text or "").translate(TURKISH_LOWER_MAP).lower()


def remove_pause_punctuation(text: str) -> str:
    cleaned = PAUSE_PUNCTUATION_RE.sub(" ", str(text or ""))
    return " ".join(cleaned.split())
