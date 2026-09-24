"""Tool results as they enter the conversation: one canonical text, and a size cap (#1422).

A tool result is normalised once, where it enters the history, and never again:

* **Canonical text.** A `str` result stays exactly that string. Anything else is written
  as JSON with sorted keys, compact separators and `ensure_ascii=False`, so a dict result
  round-trips through `json.loads` and is byte-identical on every run. Before this the
  history held `str(output)` -- a Python `repr`, which is not JSON, quotes with `'`, and
  depends on dict insertion order.
* **The cap.** A result above `TOOL_RESULT_CAP_BYTES` is stored in full in the session's
  artifact directory, and the history holds an *excerpt*: a header line naming a handle,
  the start of the result, a marker for the part not shown, and its end. The excerpt is
  decided here, once, and is the same every time the history is rendered.
* **Reading it back.** `read_tool_result_page` returns a window of a stored result that
  itself fits under the cap, so a page is never shortened again on its way in.

The handle is content-addressed -- `tr_` and the first 16 hex digits of the SHA-256 of the
stored text -- so storing the same result twice writes one file, and the handle names what
it stores rather than when it was stored. It resolves only inside the reader's own
session directory. The blob is `<artifacts_dir>/<session_id>/<handle>.txt`, the directory
`cleanup_session_artifacts` removes when a session is deleted or reset.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path, PurePath
from typing import Final, cast

from pydantic import BaseModel

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.secrets import redact_credentials

logger = logging.getLogger(__name__)

__all__ = [
    "ARTIFACT_SUBDIR",
    "STEP_EXCERPT_MIN_BYTES",
    "STEP_NO_ROOM_MESSAGE",
    "STEP_OVER_WINDOW_MESSAGE",
    "STEP_REPLY_RESERVE_TOKENS",
    "STORED_RESULT_PREFIX",
    "TOOL_RESULT_CAP_BYTES",
    "TOOL_RESULT_CAP_TOKENS",
    "TOOL_RESULT_READ_TOOL",
    "StoredResultNotFoundError",
    "artifacts_dir_for",
    "canonical_tool_text",
    "excerpt_tool_result",
    "handle_in",
    "ingest_tool_text",
    "load_tool_result",
    "read_tool_result_page",
    "result_handle",
    "store_tool_result",
    "step_result_caps",
    "stub_tool_result",
]

TOOL_RESULT_CAP_TOKENS: Final = 2_000
"""The most one tool result may cost in the history, in estimated tokens.

2,000 tokens is a quarter of an 8K window, the smallest served window this project
targets, and so leaves room for the system turn, the tool schemas and several results
below the 70% compaction trigger (5,734 tokens at 8K). It is also above what almost every
ordinary result costs -- a directory listing, a search hit list, a short file -- so the
cap bites on the outliers (a 1 MB page fetch, a whole log) and leaves the common case
whole. Not a configuration knob, for the reason `max_ledgers` is bounded rather than
free: a cap large enough to fill the window defeats itself, and nobody tunes it
knowingly.
"""

TOOL_RESULT_CAP_BYTES: Final = TOOL_RESULT_CAP_TOKENS * 4
"""`TOOL_RESULT_CAP_TOKENS` in UTF-8 bytes, at the shared estimator's four bytes a token."""

TOOL_RESULT_READ_TOOL: Final = "tool_result_read"
"""The name of the read-only tool that pages through a stored result."""

STORED_RESULT_PREFIX: Final = "[Stored tool result "
"""How every excerpt, page and stub this module writes begins.

Recognising the prefix alone is never enough to drop text: `stub_tool_result` is applied
only when the handle it names resolves to a stored blob, so text a tool happened to
begin with these words is never mistaken for something that can be read back.
"""

# Room reserved for the header line and the omitted-part marker. Both are a few hundred
# bytes at most -- two numbers, a handle and fixed words -- so this bounds every excerpt
# and page at the cap without computing the header before the body it describes.
_FRAMING_BYTES: Final = 480

STEP_EXCERPT_MIN_BYTES: Final = _FRAMING_BYTES + 120
"""The smallest share a result may get when one step's results are fitted to the window.

The framing plus 120 bytes of the result itself: enough for the header that names the
handle, and a line or so of the start and the end. Below this an excerpt would be a
handle with nothing beside it, so a step whose share is smaller is refused instead (#1480).
"""

STEP_OVER_WINDOW_MESSAGE: Final = (
    "The results of the tools called in this step do not fit in what the model can still "
    "read at once, even shortened, so they were not sent to it. Try asking for fewer "
    "things at a time, or use a model with a larger context window."
)
"""What a turn says when one step's results cannot fit the window even as excerpts (#1480).

Worded about the room that is left, not about how much the tools returned: a step of many
small results can fail to fit as surely as one of a few large ones (#1509).
"""

STEP_NO_ROOM_MESSAGE: Final = (
    "This conversation already takes up all the room the model has to read and reply, so "
    "the results of the tools called in this step were not sent to it. Start a new "
    "conversation, or use a model with a larger context window."
)
"""What a turn says when the request leaves no room for a step's results at all (#1509).

The conversation before the step, with the system turn, the tool schemas, the turn context
and the room kept for the reply, already reaches the window, so any result would be over
it. Asking for fewer things would not help, and the step message would blame the tools.
"""

STEP_REPLY_RESERVE_TOKENS: Final = 1_024
"""Room kept for the reply when the agent sets no `max_tokens` (#1509).

Every server this project sends to counts the reply against the same window as the
request: Ollama's `num_ctx`, vLLM's `max_model_len`, and the hosted providers' published
windows. A request fitted to the window's last token leaves the model no room to answer.
With `max_tokens` set, that is the reserve. Without it the reply's length is the server's
choice, and this keeps room for a few paragraphs or a round of tool calls.
"""

_HANDLE_RE: Final = re.compile(r"tr_[0-9a-f]{16}")
_FORBIDDEN_IN_SESSION_ID: Final = ("..", "/", "\\", "\x00")


ARTIFACT_SUBDIR: Final = ".sandbox/tool_artifacts"
"""Where a workspace keeps tool artifacts: the directory `SessionStore` reaps and cleans."""


def artifacts_dir_for(workspace_root: Path) -> Path:
    """The artifact directory of `workspace_root`."""
    return workspace_root / ARTIFACT_SUBDIR


class StoredResultNotFoundError(LookupError):
    """A handle that does not name a stored result in this session. The message is plain."""


def _json_default(value: object) -> object:
    """Map the non-JSON types tools actually return onto JSON; refuse the rest."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return unwrap_immutable(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, _dt.datetime | _dt.date | _dt.time):
        return value.isoformat()
    if isinstance(value, set | frozenset):
        # Sorted by canonical form, so a set renders the same on every run.
        members: list[object] = [unwrap_immutable(v) for v in cast("set[object]", value)]
        return sorted(members, key=canonical_tool_text)
    raise TypeError(f"a tool result holds a {type(value).__name__}, which has no JSON form")


def canonical_tool_text(value: object) -> str:
    """The one text form of a tool result: the string itself, or canonical JSON.

    A `str` is returned unchanged -- not JSON-quoted -- because a tool that answers in
    prose means that prose. `None` becomes `null`. A value with no JSON form raises
    `TypeError` rather than falling back to `repr`, which is the silent substitution this
    function exists to remove (P6). So do `bytes`, which have no text form without a
    guess at their encoding, and a NaN or infinite float, which JSON cannot hold:
    `json.dumps` would write `NaN`, which no JSON reader accepts.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            unwrap_immutable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    except ValueError as exc:
        if "Out of range float" not in str(exc):
            raise
        raise TypeError(
            "a tool result holds a NaN or infinite number, which has no JSON form"
        ) from exc


def result_handle(body: str) -> str:
    """The content-addressed handle of a stored body."""
    return "tr_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def handle_in(content: str | None) -> str | None:
    """The handle a stored-result header names, if `content` begins with one."""
    if not content or not content.startswith(STORED_RESULT_PREFIX):
        return None
    match = _HANDLE_RE.match(content, len(STORED_RESULT_PREFIX))
    return match.group(0) if match else None


def _session_dir(artifacts_dir: Path, session_id: str) -> Path:
    """The session's artifact directory, contained lexically.

    A session id with no separator, no `..` and no NUL is one path component, and a
    handle matches `tr_[0-9a-f]{16}`, so `<artifacts_dir>/<session_id>/<handle>.txt`
    cannot name anything outside `artifacts_dir`. Checked on the characters rather than
    by resolving, which keeps the one resolving guard in `resolve_session_path`.
    """
    if (
        not session_id
        or session_id in (".", "..")
        or any(bad in session_id for bad in _FORBIDDEN_IN_SESSION_ID)
    ):
        raise ValueError(f"session id {session_id!r} cannot name a directory")
    return artifacts_dir / session_id


def _blob_path(artifacts_dir: Path, session_id: str, handle: str) -> Path:
    if not _HANDLE_RE.fullmatch(handle):
        raise StoredResultNotFoundError(
            f"'{handle}' is not a stored tool result name. Names look like "
            "tr_ followed by 16 letters and digits, as shown in the shortened result."
        )
    return _session_dir(artifacts_dir, session_id) / f"{handle}.txt"


def store_tool_result(artifacts_dir: Path, session_id: str, body: str) -> str:
    """Store `body` in full under the session and return its handle.

    Credentials are redacted before the body is hashed or written, so neither the file
    nor the handle derives from a secret (#569). The write is atomic -- a temporary file
    and `os.replace` -- and skipped when the blob exists, since the name is its content.
    """
    redacted = redact_credentials(body)
    handle = result_handle(redacted)
    path = _blob_path(artifacts_dir, session_id, handle)
    if path.is_file():
        return handle
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(redacted, encoding="utf-8")
    os.replace(tmp, path)
    return handle


def load_tool_result(artifacts_dir: Path, session_id: str, handle: str) -> str:
    """The full stored body `handle` names in this session, or a plain refusal."""
    path = _blob_path(artifacts_dir, session_id, handle)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise StoredResultNotFoundError(
            f"No stored tool result named '{handle}' exists in this conversation. It may "
            "have been cleared with the conversation, or the name was mistyped."
        ) from None


def _head_within(text: str, max_bytes: int) -> str:
    """The longest prefix of `text` whose UTF-8 encoding fits in `max_bytes`."""
    return text.encode("utf-8")[: max(max_bytes, 0)].decode("utf-8", errors="ignore")


def _tail_within(text: str, max_bytes: int) -> str:
    """The longest suffix of `text` whose UTF-8 encoding fits in `max_bytes`."""
    if max_bytes <= 0:
        return ""
    return text.encode("utf-8")[-max_bytes:].decode("utf-8", errors="ignore")


def _fits(text: str, cap_bytes: int) -> bool:
    return len(text.encode("utf-8")) <= cap_bytes


def excerpt_tool_result(
    body: str,
    handle: str | None,
    *,
    cap_bytes: int = TOOL_RESULT_CAP_BYTES,
    readable: bool = True,
) -> str:
    """The history's form of an over-cap result: a header, its start, its end.

    The start gets two thirds of the room and the end one third: a result's head says
    what it is, and its tail is where a command's error or a log's latest lines are. With
    no `handle` -- nothing could be stored -- or `readable=False` -- this agent cannot
    call the reader -- the header says the rest cannot be read back, rather than naming a
    way to read it that does not work (P6).
    """
    room = max(cap_bytes - _FRAMING_BYTES, 0)
    head = _head_within(body, room * 2 // 3)
    tail = _tail_within(body[len(head) :], room // 3)
    omitted_from = len(head)
    omitted_to = len(body) - len(tail)
    total = len(body)
    if handle is not None and readable:
        header = (
            f"{STORED_RESULT_PREFIX}{handle}: {total:,} characters, too long to show in "
            f"full. Its start and end are below. Read the rest with {TOOL_RESULT_READ_TOOL}"
            f'(handle="{handle}", offset={omitted_from}).]'
        )
    elif handle is not None:
        header = (
            f"{STORED_RESULT_PREFIX}{handle}: {total:,} characters, too long to show in "
            "full. Its start and end are below. The rest was kept, but this agent has no "
            "tool to read it.]"
        )
    else:
        header = (
            f"[Tool result shortened: {total:,} characters, too long to show in full. Its "
            "start and end are below. The rest could not be kept, so it cannot be read "
            "back.]"
        )
    marker = f"[... characters {omitted_from:,} to {omitted_to:,} not shown ...]"
    return f"{header}\n{head}\n{marker}\n{tail}"


def ingest_tool_text(
    text: str,
    *,
    artifacts_dir: Path | None,
    session_id: str,
    readable: bool = True,
    cap_bytes: int = TOOL_RESULT_CAP_BYTES,
) -> str:
    """`text` as the history should hold it: unchanged under the cap, else an excerpt.

    With no `artifacts_dir`, or when the write fails, the full text has nowhere to go:
    the excerpt says so in band and the failure is logged, rather than the turn failing
    over a result the tool did produce.
    """
    if _fits(text, cap_bytes):
        return text
    handle: str | None = None
    body = redact_credentials(text)
    if artifacts_dir is not None:
        try:
            handle = store_tool_result(artifacts_dir, session_id, text)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not store an over-cap tool result for session %r; the history "
                "keeps an excerpt only: %s",
                session_id,
                exc,
            )
    else:
        logger.warning(
            "No artifact directory for session %r; an over-cap tool result is kept as an "
            "excerpt only",
            session_id,
        )
    return excerpt_tool_result(body, handle, cap_bytes=cap_bytes, readable=readable)


def read_tool_result_page(
    artifacts_dir: Path,
    session_id: str,
    handle: str,
    offset: int = 0,
    length: int | None = None,
    *,
    cap_bytes: int = TOOL_RESULT_CAP_BYTES,
) -> str:
    """One window of a stored result, starting at character `offset`, that fits the cap.

    `length` asks for at most that many characters; the window is shorter when the cap
    requires it, and the header says exactly which characters it holds and where the
    next window starts, so paging needs no arithmetic from the reader.
    """
    body = load_tool_result(artifacts_dir, session_id, handle)
    total = len(body)
    if offset < 0:
        raise ValueError(f"The offset must be 0 or more; {offset} was given.")
    if length is not None and length < 1:
        raise ValueError(f"The length must be at least 1; {length} was given.")
    if offset >= total and total > 0:
        raise ValueError(
            f"Offset {offset:,} is past the end of this result, which has {total:,} characters."
        )
    wanted = body[offset : offset + length] if length is not None else body[offset:]
    page = _head_within(wanted, max(cap_bytes - _FRAMING_BYTES, 0))
    end = offset + len(page)
    if end < total:
        where_next = f'Next: {TOOL_RESULT_READ_TOOL}(handle="{handle}", offset={end}).'
    else:
        where_next = "This is the end of the result."
    header = (
        f"{STORED_RESULT_PREFIX}{handle}: characters {offset:,} to {end:,} of {total:,}. "
        f"{where_next}]"
    )
    return f"{header}\n{page}"


def stub_tool_result(
    content: str,
    artifacts_dir: Path,
    session_id: str,
    *,
    keep_chars: int,
) -> str | None:
    """The smaller form compaction gives an excerpt or a page, or `None` to leave it be.

    `None` unless `content` names a handle whose blob exists in this session: the stub
    drops text, and text may be dropped only when it can still be read back. The stub
    keeps the first `keep_chars` characters of the stored body -- what an offloaded
    result keeps -- so the model still sees what the result was about.
    """
    handle = handle_in(content)
    if handle is None:
        return None
    try:
        body = load_tool_result(artifacts_dir, session_id, handle)
    except (StoredResultNotFoundError, ValueError):
        return None
    header = (
        f"{STORED_RESULT_PREFIX}{handle}: {len(body):,} characters, not shown here since "
        f"the conversation was compacted. Read it with {TOOL_RESULT_READ_TOOL}"
        f'(handle="{handle}", offset=0).]'
    )
    return f"{header}\n{body[: max(keep_chars, 0)]}"


def step_result_caps(
    sizes: Sequence[int], budget_bytes: int, *, floor: int = STEP_EXCERPT_MIN_BYTES
) -> list[int] | None:
    """A byte cap for each result of one step, so that together they fit `budget_bytes`.

    One step's results are each under the result cap, but together they can exceed the
    window (#1480). The budget is shared out smallest first: a result that fits in an
    equal share of what is left keeps its size, and the rest split what remains equally,
    each to be cut to an excerpt of its share. So small results stay whole and no large
    one is starved by another. `None` when an excerpt's share would fall below `floor`:
    the step cannot fit even as excerpts, and the caller refuses it.
    """
    caps = [0] * len(sizes)
    remaining = budget_bytes
    order = sorted(range(len(sizes)), key=lambda index: sizes[index])
    for position, index in enumerate(order):
        share = remaining // (len(sizes) - position)
        if sizes[index] <= share:
            caps[index] = sizes[index]
        elif share < floor:
            return None
        else:
            caps[index] = share
        remaining -= caps[index]
    return caps
