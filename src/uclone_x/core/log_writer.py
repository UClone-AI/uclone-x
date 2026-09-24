"""Append-only log writing with write-time credential redaction (#526, #569).

Option B adopts an append-only event log as a session's record. This module provides
the log writing abstraction, enforcing write-time credential redaction (Option A from #569)
so that sensitive credential shapes never reach persistent storage or log streams.

Known Limitations of Pattern-Based Redaction (Option A):
Pattern-based redaction is a heuristic mitigation, not a complete security guarantee.
It recognizes known credential shapes (e.g. OpenAI `sk-...`, GitHub `ghp_...`, Anthropic
`sk-ant-...`, AWS access keys, Bearer tokens, and explicit secret assignments), but cannot
detect arbitrary high-entropy strings, bespoke tokens without prefixes, or obfuscated
secrets without high false-positive rates. Per Principle 3 (P3) and threat model T3,
redaction on write reduces retention risk, but does not replace process-level isolation
or host egress boundaries.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from uclone_x.core.secrets import REDACTED_PLACEHOLDER, redact_credentials

__all__ = [
    "LogWriterProtocol",
    "RedactingLogWriter",
    "redact_log_payload",
]


def redact_log_payload(payload: Any, placeholder: str = REDACTED_PLACEHOLDER) -> Any:
    """Recursively redact credential shapes from strings within mappings and sequences."""
    if isinstance(payload, str):
        return redact_credentials(payload, placeholder=placeholder)
    if isinstance(payload, Mapping):
        return {
            str(k): redact_log_payload(v, placeholder=placeholder)
            for k, v in cast(Mapping[Any, Any], payload).items()
        }
    if isinstance(payload, (list, tuple)):
        seq = cast("Sequence[object]", payload)
        return [redact_log_payload(item, placeholder=placeholder) for item in seq]
    return payload


class LogWriterProtocol(Protocol):
    """Protocol for writing log entries with write-time credential redaction."""

    def write_entry(self, entry: str | Mapping[str, Any]) -> str:
        """Write an entry (string or mapping), returning the redacted serialized entry written."""
        ...

    def write_line(self, line: str) -> str:
        """Write a single raw line, redacting any credentials before persisting."""
        ...


class RedactingLogWriter:
    """An append-only log writer that enforces credential redaction on write.

    May write to a file path, a TextIO stream, or operate in-memory.
    All written content is guaranteed to pass through `redact_credentials`
    before touching disk or underlying streams.
    """

    def __init__(
        self,
        dest: Path | TextIO | None = None,
        *,
        flush_immediately: bool = True,
    ) -> None:
        self._dest = dest
        self._flush_immediately = flush_immediately

    def write_line(self, line: str) -> str:
        """Redact credentials and write a single line (appending newline if needed)."""
        clean = redact_credentials(line)
        if not clean.endswith("\n"):
            clean_line = clean + "\n"
        else:
            clean_line = clean

        if isinstance(self._dest, Path):
            self._dest.parent.mkdir(parents=True, exist_ok=True)
            with self._dest.open("a", encoding="utf-8") as f:
                f.write(clean_line)
                if self._flush_immediately:
                    f.flush()
        elif self._dest is not None:
            self._dest.write(clean_line)
            if self._flush_immediately and hasattr(self._dest, "flush"):
                self._dest.flush()

        return clean

    def write_entry(self, entry: str | Mapping[str, Any]) -> str:
        """Serialize entry if needed, redact credentials, and write."""
        if isinstance(entry, Mapping):
            sanitized = redact_log_payload(entry)
            line = json.dumps(sanitized)
        else:
            line = str(entry)
        return self.write_line(line)
