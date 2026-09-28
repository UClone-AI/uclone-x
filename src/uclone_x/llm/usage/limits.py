"""The user's limits on paid-model tokens, and the one check against them.

See the token-gateway design §4.2. Three rolling windows, counted back from now:
the last 10 minutes, the last 5 hours and the last 7 days. Every window with a limit must
pass; a window with no limit (`None`) is skipped. A fresh install has no limits (§4.4): the
budget is the user's to set.

`check` is a pure function of the limits, the rows and the time, so a test needs no clock
and no file.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import UsageLimitsUnreadableError
from uclone_x.llm.usage.store import UsageEntry, UsageStore

__all__ = [
    "LIMITS_SETTINGS_KEY",
    "USAGE_LIMIT_ENV_VARS",
    "UsageLimits",
    "UsageStatus",
    "UsageWindow",
    "WindowStatus",
    "check",
    "env_limits",
    "limit_reached_message",
    "load_limits",
    "save_limits",
    "saved_limits",
]


class UsageWindow(StrEnum):
    """A rolling window. The value is the matching `UsageLimits` field name."""

    PER_10_MINUTES = "per_10_minutes"
    PER_5_HOURS = "per_5_hours"
    PER_WEEK = "per_week"

    @property
    def duration(self) -> timedelta:
        return _DURATIONS[self]

    @property
    def phrase(self) -> str:
        """How the window is named to a person: "10-minute", "5-hour", "weekly"."""
        return _PHRASES[self]


_DURATIONS: dict[UsageWindow, timedelta] = {
    UsageWindow.PER_10_MINUTES: timedelta(minutes=10),
    UsageWindow.PER_5_HOURS: timedelta(hours=5),
    UsageWindow.PER_WEEK: timedelta(days=7),
}

_PHRASES: dict[UsageWindow, str] = {
    UsageWindow.PER_10_MINUTES: "10-minute",
    UsageWindow.PER_5_HOURS: "5-hour",
    UsageWindow.PER_WEEK: "weekly",
}

#: The key under which `settings.json` keeps the limits.
LIMITS_SETTINGS_KEY = "usage_limits"

#: A variable per window. A set variable outranks the saved limit, as variables outrank
#: `settings.json` everywhere else. `none` means no limit for that window.
USAGE_LIMIT_ENV_VARS: dict[UsageWindow, str] = {
    UsageWindow.PER_10_MINUTES: "UCLONE_USAGE_LIMIT_10_MINUTES",
    UsageWindow.PER_5_HOURS: "UCLONE_USAGE_LIMIT_5_HOURS",
    UsageWindow.PER_WEEK: "UCLONE_USAGE_LIMIT_WEEK",
}


class UsageLimits(BaseModel):
    """Tokens (input and output together) allowed per window. `None` is no limit."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    per_10_minutes: int | None = Field(default=None, ge=1)
    per_5_hours: int | None = Field(default=None, ge=1)
    per_week: int | None = Field(default=None, ge=1)

    def limit_for(self, window: UsageWindow) -> int | None:
        return cast(int | None, getattr(self, window.value))


class WindowStatus(BaseModel):
    """One window's usage against its limit."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    window: UsageWindow
    used: int
    limit: int | None
    available_again_at: datetime | None = Field(
        default=None,
        description="Set when the limit is reached: the moment enough of the window's "
        "usage ages out that it falls back under the limit.",
    )

    @property
    def reached(self) -> bool:
        return self.limit is not None and self.used >= self.limit


class UsageStatus(BaseModel):
    """Every window's status at one moment."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    checked_at: datetime
    windows: tuple[WindowStatus, ...]

    @property
    def blocking(self) -> WindowStatus | None:
        """The reached window that lifts last, or `None` when every window has room.

        The last to lift, because paid models stay stopped until every window has room,
        and naming an earlier time would promise something untrue.
        """
        reached = [w for w in self.windows if w.reached]
        if not reached:
            return None
        return max(reached, key=lambda w: w.available_again_at or self.checked_at)


def _available_again_at(
    entries: Sequence[UsageEntry], window: UsageWindow, limit: int, now: datetime
) -> datetime:
    """When the window's usage, as rows age out oldest first, falls under `limit`."""
    remaining = sum(e.tokens for e in entries)
    for entry in entries:
        remaining -= entry.tokens
        if remaining < limit:
            return entry.at + window.duration
    return now  # pragma: no cover - the loop ends below any positive limit


def check(limits: UsageLimits, store: UsageStore, now: datetime) -> UsageStatus:
    """Each window's usage, limit and, when reached, the time it lifts.

    Reads the store once, for the longest window, and derives the shorter ones from it.
    A window is reached when its usage is **at or above** its limit.
    """
    longest = max(w.duration for w in UsageWindow)
    rows = store.entries_since(now - longest)
    windows: list[WindowStatus] = []
    for window in UsageWindow:
        start = now - window.duration
        inside = [e for e in rows if e.at > start]
        used = sum(e.tokens for e in inside)
        limit = limits.limit_for(window)
        again = (
            _available_again_at(inside, window, limit, now)
            if limit is not None and used >= limit
            else None
        )
        windows.append(
            WindowStatus(window=window, used=used, limit=limit, available_again_at=again)
        )
    return UsageStatus(checked_at=now, windows=tuple(windows))


def _clock(moment: datetime) -> str:
    """A local clock time a person reads: "3:40 PM"."""
    return moment.astimezone().strftime("%I:%M %p").lstrip("0")


def limit_reached_message(window: UsageWindow, available_again_at: datetime, now: datetime) -> str:
    """The stop message (§4.3): what stopped, when it lifts, and what to do now.

    No token figure, status code or class name. The 10-minute window leads with the wait,
    because it is short.
    """
    if window is UsageWindow.PER_10_MINUTES:
        minutes = max(1, math.ceil((available_again_at - now).total_seconds() / 60))
        unit = "minute" if minutes == 1 else "minutes"
        when = f"You can use them again in {minutes} {unit} ({_clock(available_again_at)})."
    else:
        when = f"You can use them again at {_clock(available_again_at)}."
    return (
        f"You've reached your {window.phrase} limit for paid models. {when} "
        "To keep going now, raise the limit in Settings → Usage, "
        "or switch this clone to a local model."
    )


def _parse_limit(raw: object, source: str) -> int | None:
    """A limit value from settings or the environment; refused when it is not one."""
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "").replace("_", "")
        if text.lower() in ("", "none"):
            return None
        if text.isdigit() and int(text) > 0:
            return int(text)
    elif isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    raise UsageLimitsUnreadableError(
        f"The usage limit in {source} is {raw!r}, which is not a number of tokens. "
        "Set a whole number above zero, or `none` for no limit."
    )


def _read_settings_limits(path: Path) -> Mapping[str, object]:
    """The `usage_limits` object saved in `path`, or an empty mapping when none is saved."""
    import json

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise UsageLimitsUnreadableError(
            f"The saved settings at {path} could not be read, so the usage limits in them "
            f"are unknown ({exc.strerror})."
        ) from exc
    try:
        data: object = json.loads(text)
    except ValueError as exc:
        raise UsageLimitsUnreadableError(
            f"The saved settings at {path} could not be read, so the usage limits in them "
            "are unknown."
        ) from exc
    if not isinstance(data, dict):
        raise UsageLimitsUnreadableError(
            f"The saved settings at {path} could not be read, so the usage limits in them "
            "are unknown."
        )
    saved = cast(dict[str, Any], data).get(LIMITS_SETTINGS_KEY)
    if saved is None:
        return {}
    if not isinstance(saved, dict):
        raise UsageLimitsUnreadableError(
            f'"{LIMITS_SETTINGS_KEY}" in {path} is not a set of limits. '
            "Set the limits again in Settings → Usage."
        )
    return cast(dict[str, object], saved)


def _settings_path(path: Path | None) -> Path:
    if path is not None:
        return path
    from uclone_x.llm.connectors.saved_choice import settings_file

    return settings_file()


def saved_limits(path: Path | None = None) -> UsageLimits:
    """The limits saved in `settings.json`, ignoring the environment.

    What the preset picker edits. A saved value that is not a limit raises
    `UsageLimitsUnreadableError`, as it does for `load_limits`.
    """
    target = _settings_path(path)
    saved = _read_settings_limits(target)
    return UsageLimits(
        **{
            window.value: _parse_limit(
                saved.get(window.value), f'"{LIMITS_SETTINGS_KEY}.{window.value}" in {target}'
            )
            for window in UsageWindow
        }
    )


def env_limits() -> dict[UsageWindow, int | None]:
    """The windows whose variable is set, each with the limit it sets.

    A blank variable is not set. A value that is not a limit raises
    `UsageLimitsUnreadableError`.
    """
    limits: dict[UsageWindow, int | None] = {}
    for window, variable in USAGE_LIMIT_ENV_VARS.items():
        env = os.environ.get(variable)
        if env is not None and env.strip():
            limits[window] = _parse_limit(env, variable)
    return limits


def load_limits(path: Path | None = None) -> UsageLimits:
    """The limits in effect: each window's variable if set, else the saved limit.

    Read on every check, so a limit saved in Settings applies to every running process
    from its next call. A value that is not a limit raises `UsageLimitsUnreadableError`
    rather than being treated as no limit.
    """
    saved = saved_limits(path)
    overrides = env_limits()
    return UsageLimits(
        **{
            window.value: overrides[window] if window in overrides else saved.limit_for(window)
            for window in UsageWindow
        }
    )


def save_limits(limits: UsageLimits, path: Path | None = None) -> None:
    """Save `limits` to `settings.json`, keeping every other key in the file."""
    from uclone_x.llm.connectors.saved_choice import update_settings_file

    update_settings_file({LIMITS_SETTINGS_KEY: limits.model_dump()}, path=path)
