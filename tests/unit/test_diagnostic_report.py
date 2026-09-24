"""Rendering and the pre-filled issue URL, including where it refuses.

The URL is the transmission path, and it is the one place where a silent
truncation would lose exactly the part of a report that mattered -- the stack.
`issue_url` returns `None` rather than a URL nobody can post, and the CLI says
so instead of opening a browser onto a half-report.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import pytest

from uclone_x.core.diagnostic_report import (
    ISSUE_REPO,
    MAX_URL_CHARS,
    issue_url,
    render_report,
    report_title,
    search_url,
    summarise,
)
from uclone_x.core.failure_journal import (
    FailureEntry,
    JournalRead,
    build_entry,
    journal_path,
    read_journal,
    set_consent,
)


@pytest.fixture(autouse=True)
def isolated_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))


def _entry(message: str, error: type[Exception] = ValueError) -> FailureEntry:
    try:
        raise error(message)
    except Exception as exc:  # noqa: BLE001
        return build_entry(exc)


def test_empty_report_says_so_rather_than_rendering_a_blank_section() -> None:
    body = render_report(JournalRead())

    assert "No failures recorded." in body
    assert "## Environment" in body


def test_summary_counts_repeats_of_one_fingerprint() -> None:
    """The count is the finding: one configuration error is an afternoon, forty
    of them is a defect in how the product explains itself."""
    set_consent(True)
    entries = [_entry("boom") for _ in range(3)] + [_entry("other", TypeError)]

    summary = summarise(entries)

    assert summary[0][1] == "ValueError"
    assert summary[0][2] == 3
    assert summary[1][2] == 1


def test_title_carries_the_fingerprint_so_duplicates_are_searchable() -> None:
    entries = [_entry("boom")]

    title = report_title(entries)

    assert entries[0].fingerprint in title
    assert title.startswith("[ucx ")


def test_issue_url_targets_the_public_repository_and_encodes_the_body() -> None:
    url = issue_url("body with spaces & symbols", "title")

    assert url is not None
    assert url.startswith(f"https://github.com/{ISSUE_REPO}/issues/new?")
    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert parsed["environment"] == ["body with spaces & symbols"]
    assert parsed["template"] == ["bug.yml"]


def test_issue_url_refuses_a_body_that_will_not_fit() -> None:
    """Mutation: return the URL anyway. GitHub or the browser then truncates
    it, and the report arrives missing its stack with nothing to indicate it."""
    assert issue_url("x" * MAX_URL_CHARS, "title") is None


def test_search_url_scopes_to_the_repository() -> None:
    url = search_url("a1b2c3d4")

    assert "a1b2c3d4" in urllib.parse.unquote(url)
    assert ISSUE_REPO in urllib.parse.unquote(url)


def test_the_report_body_carries_a_stopped_recording_not_only_the_screen() -> None:
    """The body is the only part a maintainer ever receives.

    Taking a list of entries meant the renderer could not know that recording
    had stopped, so the banner appeared on screen while the saved, submitted or
    copied text said nothing about it — this module's recurring substitution,
    displaced one surface outward onto the artifact that actually travels.

    Killed by: src/uclone_x/core/diagnostic_report.py :: def render_report(read
    """
    read = JournalRead(
        entries=(_entry("boom"),),
        recording_blocked="failures cannot be recorded: /x/failures.jsonl is not writable",
    )

    body = render_report(read)

    assert "Recording is not working" in body
    assert "is not writable" in body
    # The entries are still there: what stopped is the keeping of new ones.
    assert "ValueError" in body


def test_a_read_failure_and_a_blocked_recording_are_both_stated() -> None:
    """From a real state, not a hand-built pair.

    The first version of this test constructed a `JournalRead` with two
    different strings — a combination `read_journal()` never produced — so it
    passed over an unreachable state while the reachable ones were either
    duplicated or missing the second fact entirely.
    """
    set_consent(True)
    journal = journal_path()
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("{}\n", encoding="utf-8")
    journal.chmod(0o000)

    try:
        read = read_journal()
    finally:
        journal.chmod(0o644)

    assert read.error is not None and "could not be read" in read.error
    assert read.recording_blocked is not None and "not writable" in read.recording_blocked

    body = render_report(read)

    assert "could not be read" in body
    assert "Recording is not working" in body


def test_one_cause_is_not_reported_twice_under_two_headings() -> None:
    """A dangling symlink fails the inspection that both facts are read from.

    `_recording_blocked()` then re-runs it and returns the same sentence, so
    setting both printed it twice — one cause, two headings, and a section
    promising entries "kept before that started" above a section saying nothing
    could be read at all.
    """
    set_consent(True)
    journal = journal_path()
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.symlink_to(journal.parent / "nowhere")

    read = read_journal()

    assert read.error is not None
    assert read.recording_blocked is None
    assert render_report(read).count("is a symbolic link whose target does not exist") == 1
