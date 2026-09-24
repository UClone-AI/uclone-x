"""The journal must collect nothing before consent, and only what it declares.

Two properties carry the whole design, and both are invisible in ordinary use:
a journal that records before the question is answered, and a field that
reaches the file because nobody noticed it was being passed. Neither shows up
as a failing run -- they show up as a privacy incident -- so they are pinned
here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import TracebackType

import pytest

from uclone_x.core.failure_journal import (
    CONSENT_DENIED,
    CONSENT_GRANTED,
    CONSENT_UNASKED,
    JOURNAL_FILENAME,
    MAX_ENTRIES,
    MAX_MESSAGE_CHARS,
    ClearOutcome,
    build_entry,
    clear_journal,
    consent_path,
    consent_state,
    fingerprint_of,
    install_excepthook,
    journal_path,
    read_consent,
    read_entries,
    read_journal,
    record_failure,
    set_consent,
)


@pytest.fixture(autouse=True)
def isolated_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the invoking user's home directory."""
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))


def _raise(error: BaseException) -> BaseException:
    """Return `error` with a real traceback attached."""
    try:
        raise error
    except BaseException as exc:  # noqa: BLE001 - the point is to capture it
        return exc


def test_nothing_is_recorded_before_consent() -> None:
    """Silence until asked, and `unasked` is not `no`.

    Mutation this exists to catch: treat a missing consent file as permission,
    or buffer the entry "pending" an answer that may never come.
    """
    assert consent_state() == CONSENT_UNASKED

    assert record_failure(_raise(ValueError("boom"))) is None
    assert not journal_path().exists()
    assert read_entries() == []


def test_denial_is_honoured_and_distinguishable_from_silence() -> None:
    set_consent(False)

    assert consent_state() == CONSENT_DENIED
    assert record_failure(_raise(ValueError("boom"))) is None
    assert read_entries() == []


def test_consent_makes_it_record() -> None:
    set_consent(True)
    assert consent_state() == CONSENT_GRANTED

    entry = record_failure(_raise(ValueError("boom")))

    assert entry is not None
    assert entry.code == "ValueError"
    assert read_entries()[-1].fingerprint == entry.fingerprint


def test_only_allow_listed_context_survives() -> None:
    """A caller passing something unexpected must not widen what is collected.

    Mutation: copy `context` through verbatim. This fails, because the object
    that is not a str/int/bool would then reach the file.
    """
    set_consent(True)

    entry = build_entry(
        _raise(ValueError("boom")),
        context={
            "provider": "ollama",
            "attempts": 2,
            "degraded": True,
            # Deliberately the wrong type: a caller on a failure path passing a
            # `Path` by mistake is the case the allow-list exists for.
            "prompt": Path("/secret/prompt.txt"),
        },
    )

    assert entry.context == {"provider": "ollama", "attempts": 2, "degraded": True}
    assert "prompt" not in entry.context


def test_credentials_and_home_directory_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one free-text field passes through both filters, not either one."""

    def fake_home(cls: type[Path]) -> Path:
        return Path("/Users/someone")

    monkeypatch.setattr(Path, "home", classmethod(fake_home))

    entry = build_entry(
        _raise(RuntimeError("token sk-abcdefghijklmnopqrstuvwxyz012345 at /Users/someone/app.py"))
    )

    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in entry.message
    assert "/Users/someone" not in entry.message
    assert "~/app.py" in entry.message


def test_message_is_truncated() -> None:
    entry = build_entry(_raise(RuntimeError("x" * (MAX_MESSAGE_CHARS * 3))))

    assert len(entry.message) == MAX_MESSAGE_CHARS


def test_frames_are_package_relative_so_fingerprints_match_across_machines() -> None:
    """An absolute path carries the account name and differs per install.

    Both reasons point the same way, and the second is what makes grouping
    work at all: two users hitting one bug must produce one fingerprint.
    """
    entry = build_entry(_raise(ValueError("boom")))

    assert entry.frames
    for frame in entry.frames:
        # Either package-relative (`uclone_x/...`) or a bare filename for a
        # frame outside the package. Never an absolute path either way.
        assert not frame.startswith("/")
        assert ".." not in frame
        path_part = frame.rsplit(":", 1)[0]
        assert path_part.startswith("uclone_x/") or "/" not in path_part


def test_fingerprint_ignores_line_numbers_but_not_the_path() -> None:
    """Mutation: fold line numbers in, and one edit splits a bug into two."""
    first = fingerprint_of("ValueError", ("uclone_x/a.py:run", "uclone_x/b.py:call"))
    same = fingerprint_of("ValueError", ("uclone_x/a.py:run", "uclone_x/b.py:call"))
    other_path = fingerprint_of("ValueError", ("uclone_x/a.py:run", "uclone_x/c.py:call"))
    other_code = fingerprint_of("TypeError", ("uclone_x/a.py:run", "uclone_x/b.py:call"))

    assert first == same
    assert first != other_path
    assert first != other_code


def test_journal_is_bounded() -> None:
    """An unbounded file in someone's home directory is a defect of its own."""
    set_consent(True)
    for index in range(MAX_ENTRIES + 25):
        record_failure(_raise(ValueError(f"boom {index}")))

    assert len(read_entries()) == MAX_ENTRIES
    assert read_entries()[-1].message.endswith(str(MAX_ENTRIES + 24))


def test_recording_never_raises_into_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """This runs on the failure path of whatever the user was doing.

    A diagnostics writer that turns a handled error into a crash has made the
    product worse in the name of observing it. Nothing is substituted: the
    return is `None` and `ucx report` shows an empty journal.

    Mutation: drop the guard in `record_failure`. This test then raises OSError.
    """
    set_consent(True)

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", explode)

    assert record_failure(_raise(ValueError("boom"))) is None


def test_unreadable_lines_are_skipped_not_fatal() -> None:
    set_consent(True)
    record_failure(_raise(ValueError("good")))
    with journal_path().open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps({"ts": "t", "code": "LaterError"}) + "\n")

    entries = read_entries()

    assert [entry.code for entry in entries] == ["ValueError", "LaterError"]


def test_clear_removes_the_journal_but_not_the_consent() -> None:
    set_consent(True)
    record_failure(_raise(ValueError("boom")))

    assert clear_journal() == ClearOutcome(deleted=True)
    assert read_entries() == []
    assert consent_state() == CONSENT_GRANTED
    # Nothing to delete is not a failure to delete, and both used to be `False`.
    assert clear_journal() == ClearOutcome(deleted=False, error=None)


def test_an_unreadable_journal_is_not_reported_as_an_empty_one() -> None:
    """The defect this type exists for, reproduced and then pinned.

    The first version swallowed the read error and returned `[]`, so `chmod 000`
    on the journal produced "No failures recorded." byte-identical to a
    genuinely empty one. Each half was defensible -- guard the write, tolerate
    the read -- and together they are the substituted empty result P6 forbids.

    Mutation this exists to catch: return `JournalRead()` from the `OSError`
    branch of `read_journal`.
    """
    set_consent(True)
    record_failure(_raise(ValueError("boom")))
    journal_path().chmod(0o000)

    try:
        read = read_journal()
    finally:
        journal_path().chmod(0o644)

    assert read.is_available is False
    assert read.error is not None
    assert "could not be read" in read.error
    assert str(journal_path()) in read.error


def test_an_empty_journal_is_available_and_says_nothing_else() -> None:
    """The other side of the same distinction: absent is not broken."""
    set_consent(True)

    read = read_journal()

    assert read.is_available is True
    assert read.entries == ()
    assert read.unreadable_lines == 0


def test_unparsable_lines_are_counted_rather_than_silently_dropped() -> None:
    set_consent(True)
    record_failure(_raise(ValueError("good")))
    with journal_path().open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write("[]\n")

    read = read_journal()

    assert len(read.entries) == 1
    assert read.unreadable_lines == 2
    assert read.is_available is True


def test_context_strings_are_redacted_and_capped_like_the_message() -> None:
    """Review found context going in raw: a key verbatim, a 5000-char value whole.

    No caller passes either today, which is why it would not have been noticed
    when one started.
    """
    entry = build_entry(
        _raise(ValueError("boom")),
        context={
            "token": "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789",
            "long": "x" * (MAX_MESSAGE_CHARS * 4),
        },
    )

    assert "sk-ant-api03" not in str(entry.context["token"])
    assert len(str(entry.context["long"])) == MAX_MESSAGE_CHARS


def test_home_masking_covers_other_accounts_and_windows_paths() -> None:
    """A message can name an account that is not the one running the process.

    The literal-prefix version masked only `Path.home()`, so `/Users/alice/...`
    -- someone else's name on a shared machine -- went through untouched.
    """
    entry = build_entry(
        _raise(RuntimeError(r"/Users/alice/app.py, C:\Users\bob\x.py, /home/carol/y.py"))
    )

    assert "alice" not in entry.message
    assert "bob" not in entry.message
    assert "carol" not in entry.message
    assert entry.message.count("~") == 3


def test_the_excepthook_records_and_still_reaches_the_previous_hook() -> None:
    """Mutation: drop the `previous(...)` call, and a crash prints nothing.

    The hook is a diagnostics concern layered onto Python's; swallowing the
    traceback in order to record it would be the observer breaking the thing
    observed.
    """
    set_consent(True)
    seen: list[str] = []

    def previous(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        seen.append(type(exc).__name__)

    original = sys.excepthook
    sys.excepthook = previous
    try:
        install_excepthook()
        error = _raise(ValueError("unhandled"))
        sys.excepthook(type(error), error, error.__traceback__)
    finally:
        sys.excepthook = original

    assert seen == ["ValueError"]
    recorded = read_journal().entries
    assert recorded and recorded[-1].code == "ValueError"
    assert recorded[-1].context.get("unhandled") is True


def test_the_excepthook_records_nothing_without_consent() -> None:
    def silent(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        return None

    original = sys.excepthook
    sys.excepthook = silent
    try:
        install_excepthook()
        error = _raise(ValueError("unhandled"))
        sys.excepthook(type(error), error, error.__traceback__)
    finally:
        sys.excepthook = original

    assert read_journal().entries == ()
    assert not journal_path().exists()


def test_a_journal_that_is_a_directory_is_reported_not_treated_as_empty() -> None:
    """The fixed defect, one branch later.

    `is_file()` is False for a directory, so the first repair returned an empty
    read for it — the same "No failures recorded." over a journal that cannot
    be read. Found by re-review, which is the argument for re-reviewing a fix
    rather than the code it fixed.
    """
    set_consent(True)
    journal_path().parent.mkdir(parents=True, exist_ok=True)
    journal_path().mkdir()

    read = read_journal()

    assert read.is_available is False
    assert read.error is not None
    assert "not a regular file" in read.error


def test_a_journal_of_invalid_bytes_is_reported_rather_than_raised() -> None:
    """`UnicodeDecodeError` is a `ValueError`, so the `OSError` guard missed it.

    It escaped as a traceback from the CLI and a 500 from the API — from the
    function whose docstring says it reports rather than raises. A write
    interrupted mid-multibyte character reaches this more easily than the
    permission error that prompted the guard.
    """
    set_consent(True)
    journal_path().parent.mkdir(parents=True, exist_ok=True)
    journal_path().write_bytes(b"\xff\xfe not utf-8\n")

    read = read_journal()

    assert read.is_available is False
    assert read.error is not None
    assert "not valid UTF-8" in read.error
    assert "--clear" in read.error


def test_an_unreadable_diagnostics_directory_is_reported_rather_than_raised() -> None:
    """`Path.is_file()` itself raises on a directory with no execute bit.

    Which is why the guard wraps `exists()` and `is_file()` too, not just the
    read: the check that was supposed to protect the read was outside it.
    """
    set_consent(True)
    journal_path().parent.mkdir(parents=True, exist_ok=True)
    journal_path().write_text("{}\n", encoding="utf-8")
    journal_path().parent.chmod(0o000)

    try:
        read = read_journal()
    finally:
        journal_path().parent.chmod(0o755)

    assert read.is_available is False
    assert read.error is not None


@pytest.mark.parametrize(
    ("case", "matches"),
    [
        ("symlink_loop", "could not be read"),
        ("dangling_symlink", "target does not exist"),
        ("parent_is_a_file", "could not be read"),
    ],
)
def test_paths_that_exists_lies_about_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, matches: str
) -> None:
    """`Path.exists()` returns False for errors it swallows, so it is not a guard.

    CPython ignores `ENOENT`, `ENOTDIR`, `ELOOP` and `EBADF` inside
    `exists()`. Wrapping it in `try` therefore looks like a guard and is not
    one: each of these arrived as "no journal" — available, empty, nothing
    wrong — and in the `parent_is_a_file` case `record_failure` is
    simultaneously failing to write, so failures were dropped *and* reported as
    none. `stat()` raises for all three, which is why the decision is made from
    it.

    Killed by: src/uclone_x/core/failure_journal.py :: return path.stat().st_mode, None
    """
    root = tmp_path / case
    if case == "parent_is_a_file":
        (tmp_path / "not-a-dir").write_text("x", encoding="utf-8")
        root = tmp_path / "not-a-dir" / "sub"
    else:
        root.mkdir()
        journal = root / JOURNAL_FILENAME
        if case == "symlink_loop":
            journal.symlink_to(journal)
        else:
            journal.symlink_to(root / "nowhere")

    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(root))

    read = read_journal()

    assert read.is_available is False
    assert read.error is not None and matches in read.error


def test_a_genuinely_absent_journal_is_still_just_absent(tmp_path: Path) -> None:
    """The other side of the same change: `stat()` must not turn "no file yet"
    into an error, which would make every first run look broken."""
    read = read_journal()

    assert read.is_available is True
    assert read.entries == ()


def test_a_delete_that_fails_is_not_reported_as_nothing_to_delete() -> None:
    """`False` meant both, and every surface rendered it as success.

    `ucx report --clear` printed "Nothing to delete." at exit 0 with the file
    still on disk, and the API answered `200 {"cleared": false}` — which a
    client checking the status code cannot see.

    Killed by: src/uclone_x/core/failure_journal.py :: class ClearOutcome
    """
    set_consent(True)
    record_failure(_raise(ValueError("boom")))
    journal_path().parent.chmod(0o500)

    try:
        outcome = clear_journal()
    finally:
        journal_path().parent.chmod(0o755)

    assert outcome.deleted is False
    assert outcome.error is not None and "could not be deleted" in outcome.error


def test_an_unreadable_consent_record_is_reported_while_failing_closed() -> None:
    """Two decisions, and they are not the same one.

    The state falls back to `unasked`, which fails closed: nothing is collected
    and the question is asked again. The error exists because telling someone
    they have never been asked, when their answer is on disk and unreadable, is
    a false statement about their own choice.

    Killed by: src/uclone_x/core/failure_journal.py :: class ConsentRead
    """
    set_consent(True)
    consent_path().write_text("{not json", encoding="utf-8")

    consent = read_consent()

    assert consent.state == CONSENT_UNASKED
    assert consent.error is not None and "readable JSON" in consent.error


def test_recording_that_cannot_write_is_not_read_back_as_nothing_failed() -> None:
    """The substitution arrived at from the write side instead of the read.

    With consent granted and the directory unwritable, `record_failure` drops
    every failure and the journal is absent — so an empty read reported "no
    failures" about a run in which failures were being discarded.

    Killed by: src/uclone_x/core/failure_journal.py :: f"failures cannot be recorded: {directory} is not writable, so nothing has been "
    Becomes: f"failures cannot be recorded: {directory} is read-only, so nothing has been "
    """
    set_consent(True)
    journal_path().parent.chmod(0o500)

    try:
        recorded = record_failure(_raise(ValueError("boom")))
        read = read_journal()
    finally:
        journal_path().parent.chmod(0o755)

    assert recorded is None
    # Reported through `recording_blocked`, not `error`: the journal really is
    # readable, and it really is empty. What is false is the conclusion a
    # reader would draw from that alone, which is why the two facts are
    # separate fields rather than one overloaded one.
    assert read.is_available is True
    assert read.recording_blocked is not None and "not writable" in read.recording_blocked


def test_a_journal_that_cannot_be_appended_to_is_reported_while_still_readable() -> None:
    """The fifth instance: the check was on a branch, not on the question.

    A journal that exists but is not appendable — mode 444, or root-owned after
    a single `sudo ucx ...` — had `record_failure` dropping every failure while
    the read reported an available journal with its old entries intact. The
    reader concludes nothing else has gone wrong.

    Note what this is *not*: the entries are still returned and `is_available`
    stays true, because the file really is readable. "Cannot be read" and
    "nothing more is being kept" are separate facts and the type carries both.

    Killed by: src/uclone_x/core/failure_journal.py :: def _recording_blocked
    """
    set_consent(True)
    record_failure(_raise(ValueError("kept before it broke")))
    journal_path().chmod(0o444)

    try:
        read = read_journal()
    finally:
        journal_path().chmod(0o644)

    assert read.is_available is True
    assert len(read.entries) == 1
    assert read.recording_blocked is not None
    assert "not writable" in read.recording_blocked


def test_nothing_is_reported_as_blocked_when_recording_is_off() -> None:
    """Mutation: drop the consent check in `_recording_blocked`.

    Every user who has not turned recording on would then be told their
    failures are not being kept, which is true and useless — it is the state
    they chose.
    """
    set_consent(False)
    journal_path().parent.chmod(0o500)

    try:
        read = read_journal()
    finally:
        journal_path().parent.chmod(0o755)

    assert read.recording_blocked is None


def test_a_path_that_stat_and_lstat_both_refuse_is_not_called_absent() -> None:
    """`lstat` failing for a reason other than absence is not absence.

    The fallback swallowed every `OSError` into "genuinely absent", which is
    the substitution the helper exists to prevent, inside the helper.
    """
    set_consent(True)
    journal_path().parent.mkdir(parents=True, exist_ok=True)
    journal_path().parent.chmod(0o000)

    try:
        read = read_journal()
    finally:
        journal_path().parent.chmod(0o755)

    assert read.is_available is False
    assert read.error is not None
