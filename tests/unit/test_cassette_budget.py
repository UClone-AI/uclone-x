"""Tests for the Tier 2 cassette budget guard (#377).

The guard exists because `uclone2` accumulated 18.3 MB of cassettes across three tests
with nothing measuring them at commit time. These tests check the guard fires, and the
last one enforces the budget against the real cassette tree — which is empty today, and
is exactly when the rule is cheapest to establish.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.cassette_budget import (
    MAX_CASSETTE_BYTES,
    MAX_INTERACTIONS,
    check_cassette,
    check_cassette_dir,
    count_interactions,
    format_violations,
    iter_cassettes,
)

_INTERACTION = """- request:
    body: '{"messages": []}'
    method: POST
    uri: https://api.openai.com/v1/chat/completions
  response:
    body: {string: '{"choices": []}'}
    status: {code: 200, message: OK}
"""


def _cassette(tmp_path: Path, *, interactions: int = 1, filler: str = "") -> Path:
    path = tmp_path / "test_scenario.yaml"
    path.write_text(
        "interactions:\n" + _INTERACTION * interactions + filler,
        encoding="utf-8",
    )
    return path


def test_a_small_cassette_passes(tmp_path: Path) -> None:
    assert check_cassette(_cassette(tmp_path)) == ()


def test_interaction_counting_matches_vcrpy_layout(tmp_path: Path) -> None:
    """One `- request:` entry per interaction; counted without a YAML dependency."""
    assert count_interactions(_cassette(tmp_path, interactions=7).read_text()) == 7


def test_oversized_cassette_is_reported(tmp_path: Path) -> None:
    path = _cassette(tmp_path, filler="# " + "x" * (MAX_CASSETTE_BYTES + 1))

    violations = check_cassette(path)

    assert [v.rule for v in violations] == ["size"]
    assert str(MAX_CASSETTE_BYTES) in violations[0].detail


def test_too_many_interactions_is_reported(tmp_path: Path) -> None:
    """The uclone2 failure shape: 397 interactions in one file, most of them not LLM calls."""
    path = _cassette(tmp_path, interactions=MAX_INTERACTIONS + 1)

    violations = check_cassette(path)

    assert [v.rule for v in violations] == ["interactions"]
    assert f"{MAX_INTERACTIONS + 1} interactions" in violations[0].detail


def test_interaction_count_at_the_limit_passes(tmp_path: Path) -> None:
    """The budget is a ceiling, not an exclusive bound."""
    assert check_cassette(_cassette(tmp_path, interactions=MAX_INTERACTIONS)) == ()


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnopqrstuvwxyz012345",
        "AIzaSyA1234567890abcdefghijklmnopqrstu",
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "Bearer eyJhbGciOiJIUzI1NiwidHlwIjoiSldUIn0",
    ],
)
def test_leaked_credentials_are_reported(tmp_path: Path, secret: str) -> None:
    path = _cassette(tmp_path, filler=f"    authorization: {secret}\n")

    violations = check_cassette(path)

    assert "secret" in [v.rule for v in violations], f"{secret!r} slipped past the scan"


def test_redaction_placeholder_is_not_mistaken_for_a_secret(tmp_path: Path) -> None:
    """The Tier 2 filter_headers config writes REDACTED; the guard must not fire on it."""
    path = _cassette(tmp_path, filler="    authorization: Bearer REDACTED\n")

    assert check_cassette(path) == ()


def test_all_violations_are_reported_together(tmp_path: Path) -> None:
    """Reporting one rule at a time turns a re-record into several round trips."""
    path = _cassette(
        tmp_path,
        interactions=MAX_INTERACTIONS + 1,
        filler="    authorization: sk-abcdefghijklmnopqrstuvwxyz012345\n"
        + "# "
        + "x" * MAX_CASSETTE_BYTES,
    )

    assert {v.rule for v in check_cassette(path)} == {"size", "interactions", "secret"}


def test_absent_directory_yields_nothing(tmp_path: Path) -> None:
    """The tree does not exist until roadmap step 6; that is not a violation."""
    assert list(iter_cassettes(tmp_path / "nope")) == []
    assert check_cassette_dir(tmp_path / "nope") == ()


def test_directory_scan_walks_nested_provider_partitions(tmp_path: Path) -> None:
    """Cassettes live under cassettes/<provider>/<model-slug>/<module>/<test>.yaml."""
    nested = tmp_path / "ollama" / "qwen2.5-coder-7b" / "test_mod"
    nested.mkdir(parents=True)
    (nested / "test_a.yaml").write_text("interactions:\n" + _INTERACTION, encoding="utf-8")
    (nested / "test_b.yaml").write_text(
        "interactions:\n" + _INTERACTION * (MAX_INTERACTIONS + 1), encoding="utf-8"
    )

    violations = check_cassette_dir(tmp_path)

    assert [v.path.name for v in violations] == ["test_b.yaml"]
    assert "test_b.yaml" in format_violations(violations)


def test_committed_cassettes_are_within_budget() -> None:
    """The budget applied to the real tree — the assertion that actually guards the repo."""
    root = Path(__file__).resolve().parents[1] / "recorded" / "cassettes"

    violations = check_cassette_dir(root)

    assert violations == (), "cassette budget exceeded:\n" + format_violations(violations)
