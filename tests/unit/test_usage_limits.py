"""The user's usage limits: the pure `check`, the SQLite store, and loading the limits.

the token-gateway design §5. Unit tests (P8): a fixed `now`, `MemoryUsageStore`
where the store is not the subject, and a `tmp_path` SQLite file where it is.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from uclone_x.errors import UsageLimitsUnreadableError
from uclone_x.llm.models import TokenCountSource
from uclone_x.llm.usage.limits import (
    USAGE_LIMIT_ENV_VARS,
    UsageLimits,
    UsageStatus,
    UsageWindow,
    WindowStatus,
    check,
    load_limits,
    save_limits,
)
from uclone_x.llm.usage.store import MemoryUsageStore, SqliteUsageStore, UsageEntry

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def _entry(at: datetime, tokens: int, provider: str = "openai") -> UsageEntry:
    return UsageEntry(at=at, tokens=tokens, provider=provider, model="m")


def _store(*rows: tuple[timedelta, int]) -> MemoryUsageStore:
    """A store holding `tokens` recorded `ago` before `NOW`, for each `(ago, tokens)`."""
    store = MemoryUsageStore()
    for ago, tokens in rows:
        store.add(_entry(NOW - ago, tokens))
    return store


def _window(status: UsageStatus, window: UsageWindow) -> WindowStatus:
    return next(w for w in status.windows if w.window is window)


# --- check (pure) -------------------------------------------------------------------------


def test_check_reports_each_window_used_and_limit() -> None:
    store = _store(
        (timedelta(minutes=3), 100),
        (timedelta(hours=2), 200),
        (timedelta(days=3), 300),
        (timedelta(days=9), 5_000),  # older than every window
    )
    limits = UsageLimits(per_10_minutes=1_000, per_5_hours=2_000, per_week=3_000)

    status = check(limits, store, NOW)

    assert status.checked_at == NOW
    got = {w.window: (w.used, w.limit, w.available_again_at) for w in status.windows}
    assert got == {
        UsageWindow.PER_10_MINUTES: (100, 1_000, None),
        UsageWindow.PER_5_HOURS: (300, 2_000, None),
        UsageWindow.PER_WEEK: (600, 3_000, None),
    }
    assert status.blocking is None


def test_check_skips_a_window_with_no_limit() -> None:
    store = _store((timedelta(minutes=1), 10_000))
    status = check(UsageLimits(per_week=50_000), store, NOW)

    ten = _window(status, UsageWindow.PER_10_MINUTES)
    assert (ten.used, ten.limit, ten.reached, ten.available_again_at) == (
        10_000,
        None,
        False,
        None,
    )
    assert status.blocking is None


def test_check_usage_exactly_at_the_limit_is_reached() -> None:
    store = _store((timedelta(minutes=4), 60), (timedelta(minutes=2), 40))

    at_limit = _window(
        check(UsageLimits(per_10_minutes=100), store, NOW), UsageWindow.PER_10_MINUTES
    )
    one_above = _window(
        check(UsageLimits(per_10_minutes=101), store, NOW), UsageWindow.PER_10_MINUTES
    )

    assert at_limit.reached is True
    assert at_limit.available_again_at is not None
    assert one_above.reached is False
    assert one_above.available_again_at is None


@pytest.mark.parametrize(
    ("limit", "lifts_after"),
    [
        # 60+50+40 = 150. Limit 100: the oldest row (60) ageing out leaves 90 < 100.
        (100, timedelta(minutes=2)),
        # Limit 50: 60 out leaves 90, 50 out leaves 40 < 50 — the second row decides.
        (50, timedelta(minutes=5)),
    ],
)
def test_available_again_at_is_when_enough_oldest_rows_age_out(
    limit: int, lifts_after: timedelta
) -> None:
    store = _store(
        (timedelta(minutes=5), 50),
        (timedelta(minutes=8), 60),  # added out of order: oldest first is check's job
        (timedelta(minutes=1), 40),
    )
    ten = _window(check(UsageLimits(per_10_minutes=limit), store, NOW), UsageWindow.PER_10_MINUTES)

    assert ten.reached
    assert ten.available_again_at == NOW + lifts_after


def test_blocking_names_the_reached_window_that_lifts_last() -> None:
    store = _store((timedelta(days=2), 500), (timedelta(minutes=1), 100))
    limits = UsageLimits(per_10_minutes=100, per_5_hours=10_000, per_week=600)

    status = check(limits, store, NOW)

    ten = _window(status, UsageWindow.PER_10_MINUTES)
    week = _window(status, UsageWindow.PER_WEEK)
    assert ten.reached and week.reached
    assert not _window(status, UsageWindow.PER_5_HOURS).reached
    assert ten.available_again_at == NOW + timedelta(minutes=9)
    # Week: 600 at limit; the day-2 row ageing out leaves 100 < 600.
    assert week.available_again_at == NOW - timedelta(days=2) + timedelta(days=7)
    assert status.blocking == week


# --- SqliteUsageStore ---------------------------------------------------------------------


def test_two_sqlite_stores_on_one_file_see_one_total(tmp_path: Path) -> None:
    path = tmp_path / "usage.sqlite3"
    first = SqliteUsageStore(path, now=NOW)
    second = SqliteUsageStore(path, now=NOW)

    first.add(_entry(NOW - timedelta(minutes=2), 700, provider="openai"))
    second.add(_entry(NOW - timedelta(minutes=1), 300, provider="anthropic"))

    limits = UsageLimits(per_5_hours=1_000)
    for store in (first, second):
        five = _window(check(limits, store, NOW), UsageWindow.PER_5_HOURS)
        assert (five.used, five.reached) == (1_000, True)


def test_sqlite_store_prunes_rows_older_than_eight_days_on_open(tmp_path: Path) -> None:
    path = tmp_path / "usage.sqlite3"
    writer = SqliteUsageStore(path, now=NOW)
    writer.add(_entry(NOW - timedelta(days=9), 11))
    writer.add(_entry(NOW - timedelta(days=7, hours=23), 22))

    reopened = SqliteUsageStore(path, now=NOW)

    assert [e.tokens for e in reopened.entries_since(NOW - timedelta(days=30))] == [22]


def test_sqlite_entries_since_is_oldest_first_exclusive_and_round_trips(tmp_path: Path) -> None:
    store = SqliteUsageStore(tmp_path / "usage.sqlite3", now=NOW)
    since = NOW - timedelta(hours=1)
    store.add(_entry(NOW - timedelta(minutes=5), 3))
    store.add(_entry(since, 99))  # exactly at `since`: excluded
    store.add(
        UsageEntry(
            at=NOW - timedelta(minutes=30),
            tokens=1,
            provider="gemini",
            model=None,
            count_source=TokenCountSource.ESTIMATE,
        )
    )
    store.add(_entry(NOW - timedelta(minutes=10), 2))

    rows = store.entries_since(since)

    assert [e.tokens for e in rows] == [1, 2, 3]
    assert rows[0] == UsageEntry(
        at=NOW - timedelta(minutes=30),
        tokens=1,
        provider="gemini",
        model=None,
        count_source=TokenCountSource.ESTIMATE,
    )


# --- load_limits / save_limits ------------------------------------------------------------


@pytest.fixture
def no_limit_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for variable in USAGE_LIMIT_ENV_VARS.values():
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


def _settings(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_missing_settings_file_means_no_limits(
    tmp_path: Path, no_limit_env: pytest.MonkeyPatch
) -> None:
    assert load_limits(tmp_path / "absent.json") == UsageLimits()


def test_settings_without_usage_limits_means_no_limits(
    tmp_path: Path, no_limit_env: pytest.MonkeyPatch
) -> None:
    assert load_limits(_settings(tmp_path, {"llm_provider": "openai"})) == UsageLimits()


def test_env_outranks_saved_limit_per_window(
    tmp_path: Path, no_limit_env: pytest.MonkeyPatch
) -> None:
    path = _settings(tmp_path, {"usage_limits": {"per_5_hours": 2_000, "per_week": 9_000}})
    no_limit_env.setenv("UCLONE_USAGE_LIMIT_5_HOURS", "500")
    no_limit_env.setenv("UCLONE_USAGE_LIMIT_10_MINUTES", "70")

    assert load_limits(path) == UsageLimits(per_10_minutes=70, per_5_hours=500, per_week=9_000)


def test_env_none_clears_a_saved_limit(tmp_path: Path, no_limit_env: pytest.MonkeyPatch) -> None:
    path = _settings(tmp_path, {"usage_limits": {"per_week": 9_000}})
    no_limit_env.setenv("UCLONE_USAGE_LIMIT_WEEK", "none")

    assert load_limits(path) == UsageLimits()


@pytest.mark.parametrize("value", ["abc", "-5", "0", "1.5"])
def test_malformed_env_value_is_refused_naming_the_variable(
    tmp_path: Path, no_limit_env: pytest.MonkeyPatch, value: str
) -> None:
    no_limit_env.setenv("UCLONE_USAGE_LIMIT_10_MINUTES", value)

    with pytest.raises(UsageLimitsUnreadableError) as caught:
        load_limits(tmp_path / "absent.json")
    assert "UCLONE_USAGE_LIMIT_10_MINUTES" in str(caught.value)


@pytest.mark.parametrize("value", ["lots", -5, 0, True, 2.5])
def test_malformed_saved_value_is_refused_naming_the_key_and_file(
    tmp_path: Path, no_limit_env: pytest.MonkeyPatch, value: object
) -> None:
    path = _settings(tmp_path, {"usage_limits": {"per_week": value}})

    with pytest.raises(UsageLimitsUnreadableError) as caught:
        load_limits(path)
    assert "usage_limits.per_week" in str(caught.value)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize("value", [[1, 2], "2000", 5])
def test_usage_limits_that_is_not_an_object_is_refused(
    tmp_path: Path, no_limit_env: pytest.MonkeyPatch, value: object
) -> None:
    path = _settings(tmp_path, {"usage_limits": value})

    with pytest.raises(UsageLimitsUnreadableError) as caught:
        load_limits(path)
    assert "usage_limits" in str(caught.value)
    assert str(path) in str(caught.value)


def test_save_limits_keeps_other_settings_and_round_trips(
    tmp_path: Path, no_limit_env: pytest.MonkeyPatch
) -> None:
    path = _settings(tmp_path, {"llm_provider": "ollama", "read_roots": ["/x"]})
    limits = UsageLimits(per_5_hours=2_000_000, per_week=None)

    save_limits(limits, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["llm_provider"] == "ollama"
    assert saved["read_roots"] == ["/x"]
    assert saved["usage_limits"] == {
        "per_10_minutes": None,
        "per_5_hours": 2_000_000,
        "per_week": None,
    }
    assert load_limits(path) == limits
