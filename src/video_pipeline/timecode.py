import re


SRT_TIME_RE = re.compile(
    r"^(?P<hours>\d{2}):(?P<minutes>\d{2}):(?P<seconds>\d{2}),(?P<millis>\d{3})$"
)


def parse_srt_time(value: str) -> int:
    """Convert an SRT timestamp to milliseconds."""
    match = SRT_TIME_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    millis = int(match.group("millis"))

    if minutes > 59 or seconds > 59:
        raise ValueError(f"Invalid SRT timestamp range: {value!r}")

    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + millis


def format_srt_time(milliseconds: int) -> str:
    """Convert milliseconds to an SRT timestamp."""
    if milliseconds < 0:
        raise ValueError("Timestamp cannot be negative")

    seconds_total, millis = divmod(milliseconds, 1000)
    minutes_total, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes_total, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

