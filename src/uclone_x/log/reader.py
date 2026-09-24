"""Session log reading and validation with explicit versioning and fail-closed semantics (#568).

Principle 6 (Fail-Fast & Zero Silent Fallbacks):
A session log is the authoritative record adopted by decision #526. A record outlives
the schema under which it was written. Without format versioning decided before the
first line is written, subsequent schema iterations either strand legacy sessions or force
the reader to guess — and guessing at a record it cannot parse reconstructs a session
that never happened while reporting success doing it.

Version Disambiguation:
-----------------------
This repository carries two version indicators that must never be conflated:

1. `CURRENT_LOG_FORMAT_VERSION` (Log Artefact Version, e.g. "0.1.0-dev"):
   - Governs the *file container format* of the on-disk session log JSONL artefact.
   - Stamped as the very first line of the log artefact in a header object:
     `{"schema": "uclone_x.log.v0", "version": "0.1.0-dev", ...}`.
   - Validated *before* any event record is parsed.
   - An unreadable, corrupted, or unsupported log version fails closed immediately
     with `LogHeaderError`.

2. `AgentEvent.schema_version` (In-Memory Event Contract, e.g. "1.0.0"):
   - Governs the Pydantic data contract of an individual in-memory `AgentEvent`
     envelope dispatched across the reactive `EventBus`.
   - Does NOT govern the on-disk log container format.
   - Orthogonal lifecycles: changes to in-memory bus envelope schema do not dictate
     log artefact container versioning, and vice-versa.

Format Freeze Policy:
---------------------
DeepSeek Harness demonstrates frozen decoders per released generation (`session-format-v0-to-v1`,
`v1-to-v2`). This repository is currently at `v0.1.0-dev`:
**NOTHING IS FROZEN YET.**
The first frozen generation will be declared when UClone-X cuts its first stable release
or when persistent production logs exist to migrate. Building a multi-stage migration
chain before there is anything to migrate is speculative cost with no return. What cannot
be deferred is placing the version header on the artefact *now*, so that future readers
never need to guess whether a log has a version or what format it adheres to.

Fail-Closed Event Reading & `ignorable` Semantics:
-------------------------------------------------
By default, any unknown event type encountered in a session log raises `UnknownLogEventError`
naming the offending event type and the log artefact path.
Skipping is permitted ONLY when the event payload explicitly declares `ignorable: True`
(or `ignorable=True` in JSON boolean notation).
Silently skipping an unknown non-ignorable event is strictly forbidden by Principle 6.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from typing import Any, TextIO, cast

from uclone_x.engine.event_bus import EventType
from uclone_x.errors import LogHeaderError, UnknownLogEventError

__all__ = [
    "CURRENT_LOG_FORMAT_VERSION",
    "CURRENT_LOG_SCHEMA",
    "KNOWN_LOG_EVENT_TYPES",
    "SUPPORTED_LOG_FORMAT_VERSIONS",
    "LogHeader",
    "parse_log_entry",
    "read_log_header",
    "read_session_log",
    "validate_log_header",
]

CURRENT_LOG_FORMAT_VERSION: str = "0.1.0-dev"
CURRENT_LOG_SCHEMA: str = "uclone_x.log.v0"
SUPPORTED_LOG_FORMAT_VERSIONS: tuple[str, ...] = (CURRENT_LOG_FORMAT_VERSION,)

# Turn loop durable event types (emitted by BaseAgent.execute_turn)
TURN_LOOP_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "TURN_START",
        "USER_MESSAGE",
        "ASSISTANT_ATTEMPT",
        "REQUEST_CONTEXT",
        "MODEL_RESPONSE",
        "ASSISTANT_MESSAGE",
        "TOOL_CALL",
        "TOOL_RESULT",
        "EVIDENCE_NUDGE",
        "EVIDENCE_NUDGE_DECLINED",
        "GROUNDING_NUDGE",
        "ARTIFACT_NUDGE",
        "TURN_END",
        # A turn's unanswered tool step, dropped at its end, and a turn a caller undid
        # (#1423). Both say what left the conversation, so the log still accounts for it.
        "TOOL_STEP_DROPPED",
        "TURN_ROLLED_BACK",
    }
)

# Known event types recognized by the v0.1.0-dev reader
KNOWN_LOG_EVENT_TYPES: frozenset[str] = frozenset(
    {str(e.value) for e in EventType} | {e.name for e in EventType} | TURN_LOOP_EVENT_TYPES
)


@dataclass(frozen=True)
class LogHeader:
    """Parsed format header from the first line of a session log artefact."""

    schema: str
    version: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert header to serializable dictionary."""
        return {
            "schema": self.schema,
            "version": self.version,
            **self.metadata,
        }


def validate_log_header(
    raw_header: dict[str, Any],
    *,
    path: Path | None = None,
    supported_versions: tuple[str, ...] = SUPPORTED_LOG_FORMAT_VERSIONS,
) -> LogHeader:
    """Validate a raw header mapping against supported log format versions.

    Raises:
        LogHeaderError: If the header is missing required fields, schema does not match,
            or the version is unsupported.
    """
    schema = raw_header.get("schema")
    if not schema or not isinstance(schema, str):
        raise LogHeaderError(
            "Log header missing required 'schema' field",
            path=path,
        )

    if schema != CURRENT_LOG_SCHEMA:
        raise LogHeaderError(
            f"Unrecognized log schema {schema!r}; expected {CURRENT_LOG_SCHEMA!r}",
            path=path,
        )

    version = raw_header.get("version")
    if not version or not isinstance(version, str):
        raise LogHeaderError(
            "Log header missing required 'version' field",
            path=path,
        )

    if version not in supported_versions:
        raise LogHeaderError(
            f"Unsupported log format version {version!r}. Supported versions: {supported_versions}",
            path=path,
            found_version=version,
            supported_versions=supported_versions,
        )

    metadata: dict[str, Any] = {
        str(k): v for k, v in raw_header.items() if k not in ("schema", "version")
    }

    return LogHeader(
        schema=schema,
        version=version,
        metadata=metadata,
    )


def read_log_header(
    source: Path | TextIO | Sequence[str],
    *,
    supported_versions: tuple[str, ...] = SUPPORTED_LOG_FORMAT_VERSIONS,
) -> LogHeader:
    """Read and validate the format header from the first line of a log artefact.

    Raises:
        LogHeaderError: If file is empty, first line is invalid JSON, or version is unsupported.
    """
    first_line: str | None = None
    path: Path | None = None

    if isinstance(source, Path):
        path = source
        if not source.is_file():
            raise LogHeaderError(f"Log file not found: {source}", path=path)
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    first_line = stripped
                    break
    elif isinstance(source, TextIOBase):
        line = source.readline()
        if line:
            first_line = line.strip()
    elif isinstance(source, (list, tuple)):
        for item in source:
            stripped = str(item).strip()
            if stripped:
                first_line = stripped
                break

    if not first_line:
        raise LogHeaderError("Log artefact is empty or contains no header line", path=path)

    try:
        raw_parsed = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise LogHeaderError(
            f"Log header line is not valid JSON: {exc}",
            path=path,
        ) from exc

    if not isinstance(raw_parsed, dict):
        raise LogHeaderError(
            f"Log header must be a JSON object, got {type(raw_parsed).__name__}",
            path=path,
        )

    return validate_log_header(
        cast(dict[str, Any], raw_parsed),
        path=path,
        supported_versions=supported_versions,
    )


def parse_log_entry(
    raw_record: dict[str, Any],
    *,
    path: Path | None = None,
    known_event_types: frozenset[str] = KNOWN_LOG_EVENT_TYPES,
) -> dict[str, Any] | None:
    """Validate and parse a single event record from a session log.

    Returns:
        The validated event dict, or None if the event was an unknown type marked `ignorable: True`.

    Raises:
        UnknownLogEventError: If the event type is unrecognised and NOT marked `ignorable: True`.
    """
    event_type = raw_record.get("type")
    if not event_type or not isinstance(event_type, str):
        # Even untyped records fail closed unless explicitly ignorable
        is_ignorable = bool(raw_record.get("ignorable", False))
        if is_ignorable:
            return None
        raise UnknownLogEventError(
            f"Log entry has missing or invalid 'type' field in {path or 'stream'}",
            event_type="<missing>",
            path=path,
            event_data=raw_record,
        )

    if event_type not in known_event_types:
        is_ignorable = bool(raw_record.get("ignorable", False))
        if is_ignorable:
            return None
        raise UnknownLogEventError(
            f"Unknown event type {event_type!r} encountered in log artefact {path or 'stream'}. "
            "Refusing to parse to prevent incorrect session reconstruction. "
            "If this event is safe to omit, writer must set 'ignorable: True'.",
            event_type=event_type,
            path=path,
            event_data=raw_record,
        )

    return raw_record


def read_session_log(
    source: Path | TextIO | Sequence[str],
    *,
    supported_versions: tuple[str, ...] = SUPPORTED_LOG_FORMAT_VERSIONS,
    known_event_types: frozenset[str] = KNOWN_LOG_EVENT_TYPES,
) -> Iterator[dict[str, Any]]:
    """Iterate through session log events, verifying the header first and failing closed on unknown events.

    Yields:
        Parsed event dictionaries (header is consumed and verified, but not yielded).

    Raises:
        LogHeaderError: If the header is absent, malformed, or unsupported.
        UnknownLogEventError: If an unknown event type is encountered without `ignorable: True`.
    """
    path = source if isinstance(source, Path) else None

    # Helper iterator over non-empty lines
    def _iter_lines() -> Iterator[str]:
        if isinstance(source, Path):
            if not source.is_file():
                raise LogHeaderError(f"Log file not found: {source}", path=path)
            with source.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        yield s
        elif isinstance(source, TextIOBase):
            for line in source:
                s = line.strip()
                if s:
                    yield s
        elif isinstance(source, (list, tuple)):
            for item in source:
                s = str(item).strip()
                if s:
                    yield s

    line_iter = _iter_lines()

    # Step 1: Read and validate header from the very first line
    try:
        first_line = next(line_iter)
    except StopIteration:
        raise LogHeaderError("Log artefact is empty; missing version header", path=path) from None

    try:
        raw_header = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise LogHeaderError(f"Log header line is not valid JSON: {exc}", path=path) from exc

    if not isinstance(raw_header, dict):
        raise LogHeaderError(
            f"Log header must be a JSON object, got {type(raw_header).__name__}", path=path
        )

    validate_log_header(
        cast(dict[str, Any], raw_header),
        path=path,
        supported_versions=supported_versions,
    )

    # Step 2: Iterate over subsequent event records, failing closed on unknown events
    for line in line_iter:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogHeaderError(f"Corrupted log line (invalid JSON): {exc}", path=path) from exc

        if not isinstance(record, dict):
            raise LogHeaderError(
                f"Log entry must be a JSON object, got {type(record).__name__}", path=path
            )

        parsed = parse_log_entry(
            cast(dict[str, Any], record),
            path=path,
            known_event_types=known_event_types,
        )
        if parsed is not None:
            yield parsed
