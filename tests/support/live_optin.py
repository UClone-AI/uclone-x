"""Tier 3 opt-in: live tests are collected but skipped unless the run asks for them.

The mechanism is `uclone2`'s (`conftest.py:53`), adopted because it makes the safe case
the default: a Tier 3 test is *visible* in a default collection but never executes, so
adding a file to `tests/live/` cannot silently widen `./ucx test check` into something
that spends tokens. The alternative — filtering by directory — fails the moment someone
puts a live test somewhere else.

The logic lives here rather than inline in `tests/conftest.py` so it can be unit-tested
against stub items. A hook body that is only reachable through pytest's own collection
machinery is a hook body nobody has watched fail.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import pytest

LIVE_MARKER = "live"
LIVE_SKIP_REASON = "Tier 3: requires --live"

PRE_RELEASE_MARKER = "pre_release"
PRE_RELEASE_SKIP_REASON = (
    "Pre-release qualification scenario: requires --pre-release or -m pre_release"
)

__all__ = [
    "LIVE_MARKER",
    "LIVE_SKIP_REASON",
    "PRE_RELEASE_MARKER",
    "PRE_RELEASE_SKIP_REASON",
    "ItemLike",
    "apply_live_skip",
    "apply_pre_release_skip",
]


class ItemLike(Protocol):
    """The slice of `pytest.Item` this module touches."""

    @property
    def nodeid(self) -> str: ...

    def get_closest_marker(self, name: str) -> object | None: ...

    def add_marker(self, marker: pytest.MarkDecorator) -> None: ...


def apply_live_skip(
    *,
    live_enabled: bool,
    items: Iterable[ItemLike],
    skip_marker: pytest.MarkDecorator,
) -> tuple[str, ...]:
    """Attach `skip_marker` to every `live`-marked item unless live mode was requested.

    Returns the node ids that were marked, so a caller (and a test) can verify the
    effect rather than trusting that the loop ran — the same reason AGENTS.md refuses to
    accept an exit code as proof of an effect.
    """
    if live_enabled:
        return ()

    skipped: list[str] = []
    for item in items:
        # `get_closest_marker`, never `item.keywords`. `keywords` is a bag holding the
        # module name, the class name, the test name and every parametrize id as well as
        # the markers, so `"live" in item.keywords` skips any test merely *named* live —
        # measured on this repository's own suite, where it silently skipped
        # `test_partial_scopes_disable_coverage[live]`. A guard that skips tests nobody
        # marked is worse than no guard: the run stays green while covering less.
        if item.get_closest_marker(LIVE_MARKER) is not None:
            item.add_marker(skip_marker)
            skipped.append(item.nodeid)
    return tuple(skipped)


def apply_pre_release_skip(
    *,
    pre_release_enabled: bool,
    items: Iterable[ItemLike],
    skip_marker: pytest.MarkDecorator,
) -> tuple[str, ...]:
    """Attach `skip_marker` to every `pre_release`-marked item unless pre_release mode was requested."""
    if pre_release_enabled:
        return ()

    skipped: list[str] = []
    for item in items:
        if item.get_closest_marker(PRE_RELEASE_MARKER) is not None:
            item.add_marker(skip_marker)
            skipped.append(item.nodeid)
    return tuple(skipped)
