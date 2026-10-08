"""Interval and command input parser for recurring loop execution (FR-Loop, P4)."""

from __future__ import annotations

import re
from typing import Literal

LoopParseReason = Literal["usage", "missing_prompt", "interval_too_short", "no_interval"]
"""Why a `/loop` command could not be read, for a caller that words the answer itself."""


class LoopParseError(ValueError):
    """A `/loop` command that could not be read, with the reason as a value.

    The message is for the CLI and the log. A conversation does not show it: it says
    `reason` in the reader's language instead (`room/notices.py`), so the sentence is never
    matched on or relayed.
    """

    def __init__(self, message: str, *, reason: LoopParseReason) -> None:
        super().__init__(message)
        self.reason: LoopParseReason = reason


# Minimum allowed interval to protect against tight spin loops and API abuse (P4)
MIN_INTERVAL_SECONDS: float = 1.0

# Multipliers to convert units to seconds
UNIT_MULTIPLIERS: dict[str, float] = {
    # Seconds
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "초": 1.0,
    # Minutes
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "분": 60.0,
    # Hours
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "시간": 3600.0,
    # Days
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "일": 86400.0,
}

# Regex for structured tokens like "10s", "5m", "1.5h", "2d"
STRUCTURED_TOKEN_PATTERN = re.compile(r"^(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+|[가-힣]+)$")

# Regex patterns for natural language interval expressions within sentences
NATURAL_LANGUAGE_PATTERNS = [
    # Korean: "5분마다", "30초 간격으로", "1시간 주기로", "10초 마다"
    re.compile(
        r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>초|분|시간|일)\s*(?:마다|간격으로|주기로)",
        re.IGNORECASE,
    ),
    # English: "every 5m", "every 10 minutes", "every 1 hour", "every 30 seconds"
    re.compile(
        r"\bevery\s+(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)\b",
        re.IGNORECASE,
    ),
    # English: "with an interval of 5m" / "with interval 30s"
    re.compile(
        r"\bwith\s+(?:an?\s+)?interval\s+of\s+(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)\b",
        re.IGNORECASE,
    ),
]


def parse_interval_string(val: str) -> float:
    """Parse a single interval token string into seconds.

    Examples:
        '5s' -> 5.0
        '10m' -> 600.0
        '1.5h' -> 5400.0
        '2d' -> 172800.0
        '5분' -> 300.0

    Raises:
        ValueError: If the format is invalid or interval is less than MIN_INTERVAL_SECONDS.
    """
    clean_val = val.strip()
    match = STRUCTURED_TOKEN_PATTERN.match(clean_val)
    if not match:
        raise ValueError(
            f"Invalid interval format: '{val}'. Expected format like '30s', '5m', '1h', '2d'."
        )

    num_str = match.group("val")
    unit_str = match.group("unit").lower()

    multiplier = UNIT_MULTIPLIERS.get(unit_str)
    if multiplier is None:
        valid_units = ", ".join(sorted(set(UNIT_MULTIPLIERS.keys())))
        raise ValueError(f"Unknown time unit '{unit_str}'. Supported units: {valid_units}")

    try:
        seconds = float(num_str) * multiplier
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid numeric value in interval: '{num_str}'") from exc

    if seconds < MIN_INTERVAL_SECONDS:
        raise LoopParseError(
            f"Interval must be at least {MIN_INTERVAL_SECONDS:.1f}s to prevent spin loops; got {seconds}s.",
            reason="interval_too_short",
        )

    return seconds


def parse_loop_command_input(text: str) -> tuple[float, str]:
    """Parse user arguments supplied to `/loop` into (interval_seconds, clean_prompt).

    Safety Guarantee:
        This function should only be invoked within explicit loop command contexts
        ('/loop' REPL command or 'ucx loop' CLI), never on arbitrary conversational turns.

    Supported syntax:
        1. Structured prefix:
           '/loop 5m ./ucx test check' -> (300.0, './ucx test check')
           '/loop 30s status' -> (30.0, 'status')
        2. Natural language within sentence:
           '/loop 5분마다 테스트 돌리고 보고해줘' -> (300.0, '테스트 돌리고 보고해줘')
           '/loop every 10 minutes check git log' -> (600.0, 'check git log')

    Raises:
        LoopParseError: If no interval can be recognized or prompt is empty; `reason` says
            which.
    """
    clean_text = text.strip()
    if not clean_text:
        raise LoopParseError(
            "Usage: /loop <interval> <prompt> (e.g. /loop 5m ./ucx test check)", reason="usage"
        )

    # 1. If only a single token was provided and it looks like an interval, raise missing prompt error
    parts = clean_text.split(maxsplit=1)
    if len(parts) == 1:
        try:
            parse_interval_string(parts[0])
            raise LoopParseError(
                f"Missing prompt after interval '{parts[0]}'. Usage: /loop {parts[0]} <prompt>",
                reason="missing_prompt",
            )
        except LoopParseError:
            raise
        except ValueError:
            # Not a valid interval either, continue to fallback error
            pass

    # 2. Try scanning for natural language interval patterns across the full sentence first
    for pattern in NATURAL_LANGUAGE_PATTERNS:
        match = pattern.search(clean_text)
        if match:
            num_str = match.group("val")
            unit_str = match.group("unit").lower()
            multiplier = UNIT_MULTIPLIERS.get(unit_str)
            if multiplier is not None:
                seconds = float(num_str) * multiplier
                if seconds < MIN_INTERVAL_SECONDS:
                    raise LoopParseError(
                        f"Interval must be at least {MIN_INTERVAL_SECONDS:.1f}s; got {seconds}s.",
                        reason="interval_too_short",
                    )
                # Remove matched interval expression to form clean prompt
                prompt = (clean_text[: match.start()] + clean_text[match.end() :]).strip()
                # A sentence's own full stop after the interval ("10분 마다. <prompt>") is
                # not part of the prompt; a leading "./path" is, so only a lone "." goes.
                prompt = re.sub(r"^(?:간격으로|마다|주기로|[,;\s]|\.(?=\s|$))+", "", prompt).strip()
                prompt = re.sub(r"[,;\s]+$", "", prompt).strip()
                if not prompt:
                    raise LoopParseError(
                        "Prompt is empty after extracting interval.", reason="missing_prompt"
                    )
                return seconds, prompt

    # 3. Try first whitespace-delimited token as a structured interval (e.g. '5m', '30s')
    first_token = parts[0]
    try:
        seconds = parse_interval_string(first_token)
        prompt = parts[1].strip() if len(parts) > 1 else ""
        prompt = re.sub(r"^(?:간격으로|마다|주기로|[,;\s])+", "", prompt).strip()
        if not prompt:
            raise LoopParseError(
                f"Missing prompt after interval '{first_token}'. "
                f"Usage: /loop {first_token} <prompt>",
                reason="missing_prompt",
            )
        return seconds, prompt
    except LoopParseError:
        raise
    except ValueError:
        pass

    raise LoopParseError(
        "Could not recognize interval in command. "
        "Expected e.g. '/loop 5m <prompt>' or '/loop 5분마다 <prompt>'.",
        reason="no_interval",
    )
