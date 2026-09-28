"""The head's paid-model usage surface: `GET /api/usage` and `PUT /api/usage/limits`.

See the token-gateway design §4.5. The Core owns the counting (`llm/usage/`);
this module only reads it for Settings → Usage and saves the limits the user picks.

`windows` report the limits **in effect** (a `UCLONE_USAGE_LIMIT_*` variable outranks the
saved value), while `limits` report what is **saved**, which is what the preset picker
edits. `env_overrides` names the windows where the two can differ, so the surface can say
that a saved value is not the one being applied.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from uclone_x.errors import UsageLimitsUnreadableError
from uclone_x.llm.usage.gate import shared_store
from uclone_x.llm.usage.limits import (
    USAGE_LIMIT_ENV_VARS,
    UsageLimits,
    UsageWindow,
    check,
    env_limits,
    load_limits,
    save_limits,
    saved_limits,
)
from uclone_x.llm.usage.store import UsageEntry, UsageStore

__all__ = ["register_usage_routes", "usage_report"]

#: How far back the per-provider breakdown reaches: the longest window.
PROVIDER_BREAKDOWN_SPAN = timedelta(days=7)

INVALID_LIMITS_DETAIL = (
    "Each limit must be a whole number of tokens above zero, or empty for no limit."
)
UNKNOWN_LIMIT_DETAIL = "Only the 10-minute, 5-hour and weekly limits can be set."
MISSING_LIMIT_DETAIL = (
    "Send all three limits, the 10-minute, 5-hour and weekly one; an empty one is no limit."
)
NOT_AN_OBJECT_DETAIL = "The limits could not be read from this request."
SETTINGS_UNREADABLE_DETAIL = (
    "The saved settings could not be read, so the limits were not saved. "
    "Fix or remove the settings file, then try again."
)
SETTINGS_UNWRITABLE_DETAIL = "The limits could not be saved: the settings file is not writable."


def _env_overrides() -> list[str]:
    """The windows whose variable is set, and so outranks the saved limit."""
    return [
        window.value
        for window, variable in USAGE_LIMIT_ENV_VARS.items()
        if (os.environ.get(variable) or "").strip()
    ]


def _iso(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat() if moment is not None else None


def _providers(entries: Sequence[UsageEntry]) -> list[dict[str, Any]]:
    """Tokens per provider, and per model within it, most used first."""
    totals: dict[str, dict[str | None, int]] = {}
    for entry in entries:
        models = totals.setdefault(entry.provider, {})
        models[entry.model] = models.get(entry.model, 0) + entry.tokens
    rows: list[dict[str, Any]] = [
        {
            "provider": provider,
            "tokens": sum(models.values()),
            "models": [
                {"model": model, "tokens": tokens}
                for model, tokens in sorted(
                    models.items(), key=lambda item: (-item[1], item[0] or "")
                )
            ],
        }
        for provider, models in totals.items()
    ]
    rows.sort(key=lambda row: (-cast(int, row["tokens"]), cast(str, row["provider"])))
    return rows


def usage_report(
    settings_file: Path, store: UsageStore, now: datetime | None = None
) -> dict[str, Any]:
    """The `GET /api/usage` body. Raises `UsageLimitsUnreadableError` for a bad limit."""
    moment = now if now is not None else datetime.now(UTC)
    effective = load_limits(settings_file)
    status = check(effective, store, moment)
    return {
        "checked_at": _iso(status.checked_at),
        "windows": [
            {
                "window": window.window.value,
                "used": window.used,
                "limit": window.limit,
                "available_again_at": _iso(window.available_again_at),
            }
            for window in status.windows
        ],
        "limits": saved_limits(settings_file).model_dump(mode="json"),
        "env_overrides": _env_overrides(),
        "providers": _providers(store.entries_since(moment - PROVIDER_BREAKDOWN_SPAN)),
    }


def _limits_from_body(body: object) -> UsageLimits:
    """The limits a `PUT` asked for, or a 400 in a plain sentence (no field dump)."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail=NOT_AN_OBJECT_DETAIL)
    fields = cast(dict[str, object], body)
    known = {window.value for window in UsageWindow}
    if any(key not in known for key in fields):
        raise HTTPException(status_code=400, detail=UNKNOWN_LIMIT_DETAIL)
    # A missing key is refused rather than read as "no limit": `{}` must not lift every limit.
    if any(key not in fields for key in known):
        raise HTTPException(status_code=400, detail=MISSING_LIMIT_DETAIL)
    try:
        return UsageLimits.model_validate(fields)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=INVALID_LIMITS_DETAIL) from exc


def register_usage_routes(
    app: FastAPI,
    *,
    settings_file: Path,
    usage_file: Path,
    refuse_cross_origin: Callable[[Request], None],
) -> None:
    """Mount the usage routes, reading and writing the dashboard's own settings and store."""

    def _report() -> dict[str, Any]:
        try:
            return usage_report(settings_file, shared_store(usage_file))
        except UsageLimitsUnreadableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/usage")
    async def get_usage(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Paid-model usage per window, the saved limits, and a per-provider breakdown."""
        refuse_cross_origin(request)
        return _report()

    @app.put("/api/usage/limits")
    async def put_usage_limits(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Save the user's limits; answer with the `GET /api/usage` body."""
        refuse_cross_origin(request)  # another tab must not lift the user's spending limit
        try:
            body: object = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=NOT_AN_OBJECT_DETAIL) from exc
        limits = _limits_from_body(body)
        try:
            # A malformed variable would make the answer below a 409 for a save that
            # landed, so it is refused first and nothing is written.
            env_limits()
        except UsageLimitsUnreadableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            save_limits(limits, path=settings_file)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=SETTINGS_UNREADABLE_DETAIL) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=SETTINGS_UNWRITABLE_DETAIL) from exc
        return _report()
