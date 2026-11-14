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

CLUSTER_LIVE_MARKER = "cluster_live"
CLUSTER_LIVE_SKIP_REASON = (
    "needs the docker compose cluster: run `ucx cluster up`, then `ucx test cluster` "
    "(or pytest tests/integration/cluster_live --cluster-live)"
)

PRE_RELEASE_MARKER = "pre_release"
PRE_RELEASE_SKIP_REASON = (
    "Pre-release qualification scenario: requires --pre-release or -m pre_release"
)

__all__ = [
    "CLUSTER_LIVE_MARKER",
    "CLUSTER_LIVE_SKIP_REASON",
    "LIVE_MARKER",
    "LIVE_SKIP_REASON",
    "PRE_RELEASE_MARKER",
    "PRE_RELEASE_SKIP_REASON",
    "ItemLike",
    "apply_cluster_live_skip",
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


def apply_cluster_live_skip(
    *,
    cluster_live_enabled: bool,
    items: Iterable[ItemLike],
    skip_marker: pytest.MarkDecorator,
) -> tuple[str, ...]:
    """Skip every `cluster_live` item unless `--cluster-live` was passed; return the skipped ids.

    A separate marker from `live`: `ucx test live` spends LLM tokens and must not start
    needing docker, and the cluster suite needs docker but no LLM.
    """
    if cluster_live_enabled:
        return ()
    skipped: list[str] = []
    for item in items:
        if item.get_closest_marker(CLUSTER_LIVE_MARKER) is not None:
            item.add_marker(skip_marker)
            skipped.append(item.nodeid)
    return tuple(skipped)
