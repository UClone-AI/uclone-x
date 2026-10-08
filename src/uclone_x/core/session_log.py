"""The session log: every message that entered a session's history, typed and append-only (#1443).

`SessionState.messages` is what the next request carries. It shrinks: a retry replaces a
rejected answer, a refused step is withheld, a rollback restores a checkpoint, compaction
folds turns into a ledger. The log is the record none of those touch. Each entry names
one message that was in the history at some point, by the SHA-256 of its canonical JSON,
and the message itself is stored once as a body in the session's context body store
(#1421) under that digest -- so the record grows by a few fields per message, not by the
message.

A tool result too long to keep whole in the history (#1422) is logged twice: as the
full redacted output, an entry whose body is the text itself (`logged_text`), and as the
form the history shows it in -- the message with its `form` and what it was rendered
from (`ChatMessage.rendered_from`), and no text -- whose entry names the same `tr_`
handle. The handle is the first 16 hex digits of the full body's digest, so it resolves
only among the session's own entries (`stored_result_entry`). The log holds every tool
output in full, and never the text of an excerpt or stub: that is rendered from the full
body wherever it is read (#1848).

A tool result that fit the history whole and is stubbed later, by compaction, is not
logged again (#2013): the message entry that holds it already has its full text as its
body, so the stub names that entry -- its handle is the first 16 hex digits of the
message's own digest (`is_result_message`, `kept_result_text`).

Kinds (#1849). A tool result is logged by what produced it: `subagent` for a delegated
sub-agent's result, `retrieval` for a search tool's hits (`SUBAGENT_TOOLS`,
`RETRIEVAL_TOOLS`), `tool_result` for any other. The memory facts a turn recalled are
not a message: they are sent in that turn's `[Turn Context]` tail, never in the history.
They are logged once per turn as a `memory` entry whose body is the section's text
(`logged_text`), with the `tr_` handle `tool_result_read` reads it by. Logging them
changes no request.

A handle that names no such entry -- one from before #1848, whose body was a file under
the workspace -- resolves to nothing, and reading it says the result is no longer
available.

The writer is `_LiveSession` in `agent/session_lifecycle.py`, the one place a session's
history is held (`_LiveSession.append` and its siblings for messages, `log_entry` for the rest); a loaded record
is reconciled against its messages there too.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uclone_x.core.secrets import redact_credentials
from uclone_x.llm.models import ChatMessage, ImagePart, MessageRole, image_digest

__all__ = [
    "RETRIEVAL_TOOLS",
    "SUBAGENT_TOOLS",
    "LoggedMessage",
    "SessionLogEntry",
    "SessionLogKind",
    "SessionLogProvenance",
    "history_entry_ids",
    "is_kept_text",
    "is_result_message",
    "kept_result_text",
    "log_kind",
    "logged_message",
    "logged_text",
    "result_handle_of",
    "new_entry",
    "stored_result_entry",
    "tool_result_kind",
    "with_image_data",
]


class SessionLogKind(StrEnum):
    """What a log entry records.

    `RETRIEVAL` and `SUBAGENT` are tool results, told apart by the tool that produced them
    (`log_kind`). `MEMORY` is the memory section a turn recalled, which is not a message
    (`logged_text`, #1849).
    """

    SYSTEM = "system"
    UTTERANCE = "utterance"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SUMMARY = "summary"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    SUBAGENT = "subagent"


class SessionLogProvenance(StrEnum):
    """How an entry came to be in the log.

    `RECORDED`: written when the message entered the history.
    """

    RECORDED = "recorded"


class SessionLogEntry(BaseModel):
    """One message that entered a session's history."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str = Field(description="`e<n>`, the entry's position in the log.")
    kind: SessionLogKind
    turn: int = Field(
        description="The session's turn counter when the message entered the history."
    )
    digest: str = Field(
        description="SHA-256 of the message's canonical JSON, the name its body is stored "
        "under in the session's context body store."
    )
    size: int = Field(description="Bytes of the stored body (UTF-8).")
    provenance: SessionLogProvenance
    blob: str | None = Field(
        default=None,
        description="The `tr_` handle of a full tool result or memory section: on the "
        "entry whose body is that text, and on an excerpt or stub that names it.",
    )

    @field_validator("digest")
    @classmethod
    def _is_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("A log entry's digest is 64 lowercase hex characters (SHA-256).")
        return value


#: The tool whose result is a delegated sub-agent's answer (`tools/builtin/subagent.py`).
#: Its entry's handle, when the result was over the cap, reads the whole answer back.
SUBAGENT_TOOLS: Final = frozenset({"delegate_subagent"})

#: The tools whose results are retrieval hits: ranked or matched items of a store the
#: clone searched (`memory/tools.py`, `tools/builtin/web.py`, `tools/builtin/filesystem.py`).
#: Named here rather than read from the tools so this kernel module depends on none of
#: them; `tests/unit/test_context_state.py` checks the names against the tools.
RETRIEVAL_TOOLS: Final = frozenset({"query_memory_facts", "web_search", "file_search"})


def tool_result_kind(tool_name: str | None) -> SessionLogKind:
    """The kind of entry a result of the tool `tool_name` is logged as (#1849)."""
    if tool_name in SUBAGENT_TOOLS:
        return SessionLogKind.SUBAGENT
    if tool_name in RETRIEVAL_TOOLS:
        return SessionLogKind.RETRIEVAL
    return SessionLogKind.TOOL_RESULT


def log_kind(message: ChatMessage) -> SessionLogKind:
    """The kind of entry `message` is logged as.

    A tool result is `subagent` or `retrieval` by the tool that produced it (#1849), and
    `tool_result` otherwise. The kind is a label on the entry: a request renders every
    tool result the same way, whatever its kind.
    """
    if message.role == MessageRole.SYSTEM:
        return SessionLogKind.SUMMARY if message.compaction_ledger else SessionLogKind.SYSTEM
    if message.role == MessageRole.TOOL:
        return tool_result_kind(message.name)
    if message.role == MessageRole.ASSISTANT and message.tool_calls:
        return SessionLogKind.TOOL_CALL
    return SessionLogKind.UTTERANCE


@dataclass(frozen=True, slots=True)
class LoggedMessage:
    """A message rendered for the log: its body, the body's digest, and its entry fields."""

    body: str
    digest: str
    kind: SessionLogKind
    blob: str | None
    #: The message's image bytes, as (digest, base64) for each image that has them (#2107).
    #: Never part of `body`: each is kept as its own context body, named by its digest.
    images: tuple[tuple[str, str], ...] = ()


def logged_message(message: ChatMessage) -> LoggedMessage:
    """Render `message` as the body the log stores and the digest that names it.

    The body is the message's canonical JSON -- every field, keys sorted -- so it is the
    message, not a rendering of it, and `ChatMessage.model_validate_json(body)` gives it
    back. Callers pass messages that are already redacted, as everything in a history is.

    A tool result shown in a smaller form is logged as its form, not its text (#1848): the
    message with `form` and `rendered_from` and no `content`, and the entry's `blob` is
    the handle of the full result it is rendered from. Any other message's entry names
    no blob, whatever its text says: a handle is never read out of a message's text. Its text is a function of that
    body and the recorded parameters (`core/context_state.render_form`), so the log holds
    each result once, in full, and a record of how it is shown -- never the text of a
    form. The digest is the same whether or not the message given carries the text.

    Raises:
        ValueError: `message` has a form but records nothing it was rendered from, so the
            log could keep it only as its text.
    """
    if message.form is not None:
        if message.rendered_from is None:
            raise ValueError(
                f"a tool result in the {message.form} form records no result it was "
                "rendered from, so the log cannot keep it as a form (#1848)"
            )
        message = message.model_copy(update={"content": None})
    body = json.dumps(
        message.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    # The handle a form records, never one read from a message's text: output that only
    # quotes a stored-result header is its own full text, and names no kept result.
    blob: str | None = message.rendered_from.handle if message.rendered_from is not None else None
    return LoggedMessage(
        body=body,
        digest=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        kind=log_kind(message),
        blob=blob,
        images=tuple((part.digest, part.data) for part in message.images if part.data is not None),
    )


def with_image_data(message: ChatMessage, load: Callable[[str], str | None]) -> ChatMessage:
    """`message` with each image's bytes put back from the body `load` gives its digest.

    A logged body and a saved record keep an image's digest only (`ImagePart.data` is
    never dumped), so a message read back has no bytes until they are put back here. A
    body that is missing, will not read, or does not hash to the digest leaves that image
    without bytes, and a connector says an image was there (`IMAGE_UNAVAILABLE_NOTE`): a
    lost picture is not a reason to refuse the conversation, as a lost message is (#2107).
    """
    if all(part.data is not None for part in message.images):
        return message
    parts: list[ImagePart] = []
    for part in message.images:
        if part.data is None:
            try:
                data = load(part.digest)
            except (OSError, UnicodeDecodeError):
                data = None
            if data is None or image_digest(data) != part.digest:
                parts.append(part)
                continue
            part = part.model_copy(update={"data": data})
        parts.append(part)
    return message.model_copy(update={"images": tuple(parts)})


def result_handle_of(message: ChatMessage) -> str:
    """The `tr_` handle the full text of the tool result `message` is read by (#2013).

    A form is rendered from a kept result and names its handle. A result shown whole is
    its own message entry, named by the first 16 hex digits of that entry's digest
    (`stored_result_entry`), so a record that points at the result -- the log's
    `TOOL_RESULT` event -- holds the handle, not a second copy of the text.
    """
    if message.rendered_from is not None:
        return message.rendered_from.handle
    return "tr_" + logged_message(message).digest[:16]


def logged_text(kind: SessionLogKind, text: str, *, blob: str | None) -> LoggedMessage:
    """Render `text`, which is not a message, as the body the log stores (#1849).

    For what a request sent outside the history, such as the recalled memory section of
    a turn's `[Turn Context]`. The body is the text itself, redacted, so its digest is
    never that of a message's canonical JSON and `history_entry_ids` never matches it to
    a history message. `blob` is the `tr_` handle `tool_result_read` reads the text by,
    the first 16 hex digits of this body's digest.
    """
    body = redact_credentials(text)
    return LoggedMessage(
        body=body,
        digest=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        kind=kind,
        blob=blob,
    )


def new_entry(
    position: int, rendered: LoggedMessage, *, turn: int, provenance: SessionLogProvenance
) -> SessionLogEntry:
    """The entry at `position` for `rendered`."""
    return SessionLogEntry(
        id=f"e{position}",
        kind=rendered.kind,
        turn=turn,
        digest=rendered.digest,
        size=len(rendered.body.encode("utf-8")),
        provenance=provenance,
        blob=rendered.blob,
    )


def stored_result_entry(log: Sequence[SessionLogEntry], handle: str) -> SessionLogEntry | None:
    """The entry whose body is the full text `handle` names, the latest if several (#1848).

    That entry names `handle` and its digest begins with the handle's 16 hex digits: an
    excerpt or stub that names the handle is a message's JSON, whose digest does not.
    `None` when the log holds no such entry.

    A tool result message logged whole (`is_result_message`) is such an entry too: a
    stub compaction renders from it names it by its own digest rather than logging the
    text a second time (#2013).
    """
    prefix = handle.removeprefix("tr_")
    if prefix == handle or not prefix:
        return None
    for entry in reversed(log):
        if entry.digest.startswith(prefix) and (entry.blob == handle or is_result_message(entry)):
            return entry
    return None


_RESULT_KINDS: Final = frozenset(
    {SessionLogKind.TOOL_RESULT, SessionLogKind.RETRIEVAL, SessionLogKind.SUBAGENT}
)


def is_result_message(entry: SessionLogEntry) -> bool:
    """Whether `entry` is a tool result message logged whole, which a handle can name (#2013).

    Its body is the message's JSON and it names no blob: the history showed the result
    in full. A form names the handle it is rendered from, and a kept text names its own.
    """
    return entry.blob is None and entry.kind in _RESULT_KINDS


def kept_result_text(entry: SessionLogEntry, body: str) -> str | None:
    """The full text a handle to `entry` reads, given the entry's stored `body` (#2013).

    A kept text's body is the text. A tool result message's body is its JSON, and the
    text is its content; `None` when that body is not a whole tool result message.
    """
    if not is_result_message(entry):
        return body
    try:
        message = ChatMessage.model_validate_json(body)
    except ValueError:
        return None
    if message.role is not MessageRole.TOOL or message.form is not None:
        return None
    return message.content if isinstance(message.content, str) else None


def is_kept_text(entry: SessionLogEntry) -> bool:
    """Whether `entry` is a kept text -- a full tool result or a recalled memory section
    -- rather than a history message (#1848).

    A kept text is logged as a record of what a handle names, never as a message: its
    digest begins with its own handle's hex digits, which a message's JSON body does not.
    """
    return entry.blob is not None and stored_result_entry((entry,), entry.blob) is entry


def history_entry_ids(log: Sequence[SessionLogEntry], digests: Sequence[str]) -> list[str | None]:
    """The log entry each of a history's messages is, by digest, or `None` for one it lacks.

    For a history read back with its log: the `n` occurrences of a digest in the history
    are the last `n` entries of that digest in the log, in order, since the history holds
    what entered it most recently. An occurrence beyond what the log has is `None` -- it
    has not been logged. A multiset match: two identical messages are two entries.
    """
    by_digest: dict[str, list[str]] = {}
    for entry in log:
        by_digest.setdefault(entry.digest, []).append(entry.id)
    wanted = Counter(digests)
    pools = {digest: by_digest.get(digest, [])[-count:][::-1] for digest, count in wanted.items()}
    return [pools[digest].pop() if pools[digest] else None for digest in digests]
