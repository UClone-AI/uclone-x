"""Unit and regression tests for session log format versioning and fail-closed reader (#568)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from uclone_x.engine.event_bus import EventType
from uclone_x.errors import LogHeaderError, UnknownLogEventError
from uclone_x.log.reader import (
    CURRENT_LOG_FORMAT_VERSION,
    CURRENT_LOG_SCHEMA,
    SUPPORTED_LOG_FORMAT_VERSIONS,
    TURN_LOOP_EVENT_TYPES,
    read_log_header,
    read_session_log,
    validate_log_header,
)


def test_valid_log_header_parsing(tmp_path: Path) -> None:
    """Valid header with current version parses successfully.

    Killed by: src/uclone_x/log/reader.py :: schema != CURRENT_LOG_SCHEMA
    """
    log_file = tmp_path / "session.jsonl"
    header_data = {
        "schema": CURRENT_LOG_SCHEMA,
        "version": CURRENT_LOG_FORMAT_VERSION,
        "created_at": "2026-09-11T12:00:00Z",
        "session_id": "sess-123",
    }
    log_file.write_text(json.dumps(header_data) + "\n", encoding="utf-8")

    hdr = read_log_header(log_file)
    assert hdr.schema == CURRENT_LOG_SCHEMA
    assert hdr.version == CURRENT_LOG_FORMAT_VERSION
    assert hdr.metadata["session_id"] == "sess-123"
    assert hdr.to_dict()["session_id"] == "sess-123"


def test_empty_log_file_raises_log_header_error(tmp_path: Path) -> None:
    """Empty log file is rejected with LogHeaderError rather than failing silently.

    Killed by: src/uclone_x/log/reader.py :: raise LogHeaderError("Log artefact is empty or contains no header line", path=path)
    """
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(LogHeaderError, match="empty or contains no header line"):
        read_log_header(empty_file)


def test_unsupported_log_version_refused(tmp_path: Path) -> None:
    """Unsupported future version is refused immediately before reading records.

    Killed by: src/uclone_x/log/reader.py :: version not in supported_versions
    """
    log_file = tmp_path / "future.jsonl"
    header_data = {
        "schema": CURRENT_LOG_SCHEMA,
        "version": "99.0.0",
    }
    log_file.write_text(json.dumps(header_data) + "\n", encoding="utf-8")

    with pytest.raises(LogHeaderError, match="Unsupported log format version '99.0.0'"):
        read_log_header(log_file)


def test_invalid_json_header_refused(tmp_path: Path) -> None:
    """Non-JSON first line fails closed with LogHeaderError.

    Killed by: src/uclone_x/log/reader.py :: raw_parsed = json.loads(first_line)
    Becomes: raw_parsed = {"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION}
    """
    log_file = tmp_path / "corrupt.jsonl"
    log_file.write_text("NOT_VALID_JSON\n", encoding="utf-8")

    with pytest.raises(LogHeaderError, match="not valid JSON"):
        read_log_header(log_file)


def test_missing_schema_or_version_refused() -> None:
    """Missing required schema or version fields in header dictionary raises LogHeaderError.

    Killed by: src/uclone_x/log/reader.py :: "Log header missing required 'schema' field"
    """
    with pytest.raises(LogHeaderError, match="missing required 'schema' field"):
        validate_log_header({"version": "0.1.0-dev"})

    with pytest.raises(LogHeaderError, match="missing required 'version' field"):
        validate_log_header({"schema": CURRENT_LOG_SCHEMA})

    with pytest.raises(LogHeaderError, match="Unrecognized log schema"):
        validate_log_header({"schema": "unknown.v0", "version": "0.1.0-dev"})


def test_fail_closed_on_unknown_event_type(tmp_path: Path) -> None:
    """Unknown event type without ignorable flag raises UnknownLogEventError naming type and path.

    Killed by: src/uclone_x/log/reader.py :: event_type not in known_event_types
    """
    log_file = tmp_path / "unknown_event.jsonl"
    header = {"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION}
    event1 = {"type": "TURN_START", "turn_index": 0}
    bad_event = {"type": "MYSTERY_FUTURE_OP", "data": 42}

    lines = [json.dumps(header), json.dumps(event1), json.dumps(bad_event)]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reader = read_session_log(log_file)
    ev1 = next(reader)
    assert ev1["type"] == "TURN_START"

    with pytest.raises(UnknownLogEventError) as exc_info:
        next(reader)

    err = exc_info.value
    assert err.event_type == "MYSTERY_FUTURE_OP"
    assert err.path == log_file
    assert "MYSTERY_FUTURE_OP" in str(err)
    assert "ignorable: True" in str(err)


def test_ignorable_unknown_event_is_skipped(tmp_path: Path) -> None:
    """Unknown event marked ignorable: True is skipped without raising error.

    Killed by: src/uclone_x/log/reader.py :: if parsed is not None:
    Becomes: if parsed is not None or True:
    """
    log_file = tmp_path / "ignorable.jsonl"
    header = {"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION}
    event1 = {"type": "TURN_START", "turn_index": 0}
    ignorable_event = {"type": "TELEMETRY_PING", "ignorable": True, "seq": 1}
    event2 = {"type": "TURN_END", "turn_index": 0}

    lines = [
        json.dumps(header),
        json.dumps(event1),
        json.dumps(ignorable_event),
        json.dumps(event2),
    ]
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    events = list(read_session_log(log_file))
    assert len(events) == 2
    assert events[0]["type"] == "TURN_START"
    assert events[1]["type"] == "TURN_END"


def test_known_event_types_parsed_correctly(tmp_path: Path) -> None:
    """All standard EventType and turn loop event types parse without error.

    The turn-loop half is read from `TURN_LOOP_EVENT_TYPES` rather than restated. It used
    to be a hand-maintained literal, which made this the *third* copy of the vocabulary —
    and when `EVIDENCE_NUDGE` was added in #702 it was the copy nobody updated, in the one
    file whose job is to notice that class of drift (#716, #728). There is no historical
    vocabulary to pin here: `SUPPORTED_LOG_FORMAT_VERSIONS` holds exactly one version, so
    "the types this reader accepts" and "the types v0.1.0-dev accepts" are the same set.
    That precondition is asserted below rather than only described, because the day it
    stops holding nothing else in the repository would say so.
    The registry-versus-emission binding lives in
    `tests/fitness/test_turn_loop_event_vocabulary.py`; this test asks the narrower
    question of whether each registered type survives a round trip through the reader.

    Killed by: src/uclone_x/log/reader.py :: {str(e.value) for e in EventType} | {e.name for e in EventType} | TURN_LOOP_EVENT_TYPES
    Becomes: {str(e.value) for e in EventType} | {e.name for e in EventType}
    """
    log_file = tmp_path / "all_known.jsonl"
    header = {"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION}
    records: list[dict[str, Any]] = [header]

    for ev_type in EventType:
        records.append({"type": str(ev_type.value), "payload": {}})

    # Deriving the turn-loop half from the registry is only equivalent to pinning the
    # v0.1.0-dev vocabulary while the reader supports exactly one version. Nothing else in
    # the repository asserts that: outside `src/`, `SUPPORTED_LOG_FORMAT_VERSIONS` is named
    # only in this file. Without this line, adding a second version degrades the question
    # silently, from "every type a v0.1.0-dev log can contain round-trips" to "every type
    # the current reader accepts round-trips against the current reader".
    assert len(SUPPORTED_LOG_FORMAT_VERSIONS) == 1, (
        f"the reader now supports {list(SUPPORTED_LOG_FORMAT_VERSIONS)}, so 'the types this "
        f"reader accepts' and 'the types v0.1.0-dev accepts' have come apart. Deriving "
        f"turn_events from TURN_LOOP_EVENT_TYPES no longer asks the older question. Re-derive "
        f"which vocabulary each supported version admits and pin the older set explicitly "
        f"here, rather than widening this assertion."
    )

    turn_events = sorted(TURN_LOOP_EVENT_TYPES)
    # A derived list makes a vacuous pass possible in a way a literal did not: an empty
    # registry would satisfy every assertion below. Named here so the count cannot become
    # its own evidence.
    assert turn_events, "TURN_LOOP_EVENT_TYPES is empty; this test would pass vacuously"
    for te in turn_events:
        records.append({"type": te, "turn_index": 1})

    log_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    parsed = list(read_session_log(log_file))
    assert len(parsed) == len(EventType) + len(turn_events)


def test_missing_type_field_fails_closed(tmp_path: Path) -> None:
    """An event line lacking a 'type' field fails closed unless ignorable.

    Killed by: src/uclone_x/log/reader.py :: if not event_type or not isinstance(event_type, str):
    """
    log_file = tmp_path / "missing_type.jsonl"
    header = {"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION}
    bad_record = {"payload": "no type specified"}

    log_file.write_text(
        "\n".join([json.dumps(header), json.dumps(bad_record)]) + "\n", encoding="utf-8"
    )

    with pytest.raises(UnknownLogEventError, match="missing or invalid 'type' field"):
        list(read_session_log(log_file))


def test_read_from_sequence_and_textio() -> None:
    """Reading header and events from sequence of strings or TextIO stream works seamlessly.

    Killed by: src/uclone_x/log/reader.py :: stripped = str(item).strip()
    Becomes: stripped = ""
    """
    import io

    header = {"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION}
    event = {"type": "TURN_START", "turn_index": 0}

    # Sequence of strings
    seq = [json.dumps(header), json.dumps(event)]
    hdr = read_log_header(seq)
    assert hdr.version == CURRENT_LOG_FORMAT_VERSION
    events = list(read_session_log(seq))
    assert len(events) == 1
    assert events[0]["type"] == "TURN_START"

    # TextIO stream
    buf = io.StringIO("\n".join(seq) + "\n")
    hdr_stream = read_log_header(buf)
    assert hdr_stream.version == CURRENT_LOG_FORMAT_VERSION

    buf.seek(0)
    events_stream = list(read_session_log(buf))
    assert len(events_stream) == 1
    assert events_stream[0]["type"] == "TURN_START"


def test_corrupted_json_event_line_raises_log_header_error() -> None:
    """A corrupted (non-JSON) event line raises LogHeaderError.

    Killed by: src/uclone_x/log/reader.py :: raise LogHeaderError(f"Corrupted log line (invalid JSON): {exc}", path=path) from exc
    """
    seq = [
        json.dumps({"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION}),
        "NOT_VALID_JSON_EVENT",
    ]
    with pytest.raises(LogHeaderError, match="Corrupted log line"):
        list(read_session_log(seq))
