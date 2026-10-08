"""Tool results as they enter the conversation: one canonical text, and a size cap (#1422).

A tool result is normalised once, where it enters the history, and never again:

* **Canonical text.** A `str` result stays exactly that string. Anything else is written
  as JSON with sorted keys, compact separators and `ensure_ascii=False`, so a dict result
  round-trips through `json.loads` and is byte-identical on every run. Before this the
  history held `str(output)` -- a Python `repr`, which is not JSON, quotes with `'`, and
  depends on dict insertion order.
* **The cap.** A result above `TOOL_RESULT_CAP_BYTES` is kept in full as a body of the
  session's own log (`ResultBodies`), and the history holds an *excerpt*: a header line
  naming a handle, the start of the result, a marker for the part not shown, and its end.
  The excerpt is rendered from that body, here, once, and is the same every time the
  history is rendered.
* **Reading it back.** `read_tool_result_page` returns a window of a kept body that
  itself fits under the cap, so a page is never shortened again on its way in.

The handle is content-addressed -- `tr_` and the first 16 hex digits of the SHA-256 of the
redacted text -- so keeping the same result twice keeps one body, and the handle names
what it keeps rather than when. It is the first 16 hex digits of the body's name in the
session's context body store too, so it resolves only among the session's own log
entries: one store for every body a session holds, removed with the session (#1848).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Final, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.secrets import redact_credentials
from uclone_x.errors import PlainRefusalError

logger = logging.getLogger(__name__)

__all__ = [
    "EXCERPT_NOTE",
    "STEP_EXCERPT_MIN_BYTES",
    "STEP_NO_ROOM_MESSAGE",
    "STEP_NO_ROOM_NO_COMPACTION_MESSAGE",
    "STEP_NO_ROOM_SETUP_MESSAGE",
    "STEP_NO_ROOM_SETUP_REPLY_MESSAGE",
    "STEP_OVER_WINDOW_MESSAGE",
    "STEP_REFUSAL_TEXT",
    "STEP_REPLY_RESERVE_TOKENS",
    "STORED_RESULT_PREFIX",
    "STUB_NOTE",
    "TOOL_RESULT_CAP_BYTES",
    "TOOL_RESULT_CAP_TOKENS",
    "TOOL_RESULT_READ_TOOL",
    "UNAVAILABLE_RESULT_MESSAGE",
    "UNSTORED_EXCERPT_PREFIX",
    "ResultBodies",
    "StepRefusalCode",
    "StoredResultNotFoundError",
    "canonical_tool_text",
    "excerpt_tool_result",
    "handle_in",
    "ingest_tool_result",
    "ingest_tool_text",
    "is_result_handle",
    "load_stored_result",
    "read_tool_result_page",
    "result_handle",
    "step_result_caps",
    "stored_result_stub",
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

The prefix is never what makes text droppable: compaction shrinks a result to a stub only
when the message records the kept result it was cut from (`ChatMessage.rendered_from`, #1848),
so text a tool happened to begin with these words is never mistaken for something that can be
read back.
"""

UNSTORED_EXCERPT_PREFIX: Final = "[Tool result shortened: "
"""How an excerpt of a result that could not be stored begins."""

EXCERPT_NOTE: Final = "too long to show in full"
"""What every excerpt's header says, so the model can tell it is not the whole result."""

STUB_NOTE: Final = "not shown here since the conversation was compacted"
"""What a stub's header says, so the model can tell to read the rest by its handle."""

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
    "the results of the tools called in this step were not sent to it. Your next message "
    "can carry on: older parts of the conversation are shortened first to make room."
)
"""What a turn says when the request leaves no room for a step's results at all (#1509).

The conversation before the step, with the system turn, the tool schemas, the turn context
and the room kept for the reply, already reaches the window, so any result would be over
it. Asking for fewer things would not help, and the step message would blame the tools.

It does not send the person to a new conversation (#1854). No compaction runs between the
steps of a turn (design §5.8), so this is usually the turn's own growth, and the next
turn's start-of-turn compaction shortens the conversation before that turn's first request.
When the agent does not compact on its own, `STEP_NO_ROOM_NO_COMPACTION_MESSAGE` is said
instead, since nothing would shorten it.
"""

STEP_NO_ROOM_NO_COMPACTION_MESSAGE: Final = (
    "This conversation already takes up all the room the model has to read and reply, so "
    "the results of the tools called in this step were not sent to it. Shorten this "
    "conversation, or use a model with a larger context window."
)
"""`STEP_NO_ROOM_MESSAGE` for an agent with automatic shortening turned off (#1854).

Nothing shortens the conversation before the next turn, so the person has to: the room
head offers "Shorten this conversation", and the CLI heads `/compact`.
"""

STEP_NO_ROOM_SETUP_MESSAGE: Final = (
    "The instructions and tools this assistant starts every request with already take up "
    "all the room the model has to read and reply, so the results of the tools called in "
    "this step were not sent to it. Use a model with a larger context window."
)
"""What a turn says when the system turn and the tool schemas alone fill the window (#1866).

Shortening the conversation cannot help here: without any of it, the request still leaves
no room for the reply. So it neither promises that the next message can carry on, as
`STEP_NO_ROOM_MESSAGE` does, nor asks the person to shorten the conversation, as
`STEP_NO_ROOM_NO_COMPACTION_MESSAGE` does. A head that writes in the person's language
says it from its catalog by the refusal's code (`StepRefusalCode`, #1862).
"""

STEP_NO_ROOM_SETUP_REPLY_MESSAGE: Final = (
    "The instructions and tools this assistant starts every request with, together with "
    "the room it keeps for its reply, already take up all the room the model has, so the "
    "results of the tools called in this step were not sent to it. Allow this assistant "
    "shorter replies, or use a model with a larger context window."
)
"""`STEP_NO_ROOM_SETUP_MESSAGE` when a shorter reply length would make room (#1875).

The agent sets a reply length (`max_tokens`) above the default reserve, and the system
turn and the tool schemas leave room for a reply of the default size. So lowering the
reply length is a second fix besides a larger model, and the refusal names both. Plain
words, like the other step refusals, and translated by code like them (#1862).
"""

StepRefusalCode = Literal[
    "step.over_window",
    "step.no_room",
    "step.no_room_no_compaction",
    "step.no_room_setup",
    "step.no_room_setup_reply",
]
"""Why a turn refused a step, as a key a head translates (#1862).

A turn that refuses a step carries this beside its sentence (`TurnResult.error_code`), so a
head writing in Korean says the refusal from its own catalog instead of printing Core's
English. Closed: every code has a sentence in `STEP_REFUSAL_TEXT` and in each language's
`refusals` catalog, which `tests/unit/test_refusal_i18n.py` holds to.
"""

STEP_REFUSAL_TEXT: Final[Mapping[StepRefusalCode, str]] = {
    "step.over_window": STEP_OVER_WINDOW_MESSAGE,
    "step.no_room": STEP_NO_ROOM_MESSAGE,
    "step.no_room_no_compaction": STEP_NO_ROOM_NO_COMPACTION_MESSAGE,
    "step.no_room_setup": STEP_NO_ROOM_SETUP_MESSAGE,
    "step.no_room_setup_reply": STEP_NO_ROOM_SETUP_REPLY_MESSAGE,
}
"""Each step refusal's sentence in English: `TurnResult.error`, and what a head without a
catalog for the person's language shows."""

STEP_REPLY_RESERVE_TOKENS: Final = 1_024
"""Room kept for the reply when the agent sets no `max_tokens` (#1509).

Every server this project sends to counts the reply against the same window as the
request: Ollama's `num_ctx`, vLLM's `max_model_len`, and the hosted providers' published
windows. A request fitted to the window's last token leaves the model no room to answer.
With `max_tokens` set, that is the reserve. Without it the reply's length is the server's
choice, and this keeps room for a few paragraphs or a round of tool calls.
"""

_HANDLE_RE: Final = re.compile(r"tr_[0-9a-f]{16}")


class StoredResultNotFoundError(PlainRefusalError, LookupError):
    """A handle that does not name a stored result in this session. The message is plain.

    A `PlainRefusalError`, so `tool_result_read` fails with exactly this sentence rather
    than with the class name and a prefix around it (#1848).
    """


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


UNAVAILABLE_RESULT_MESSAGE: Final = (
    "That stored result is no longer available. Run the tool again to see its output."
)
"""What reading a handle this conversation no longer holds says (#1848).

A handle from before tool results were kept in the session, a conversation that was
cleared, or a mistyped name: none of them can be read, and the remedy is the same. It
names no file, error code or store, since the model and the person see it as it is.
"""


@runtime_checkable
class ResultBodies(Protocol):
    """A session's full tool-result bodies, kept as entries of its own log (#1848).

    `keep` logs `text` as an entry whose body is the whole redacted text and returns its
    handle; `read` returns the body a handle names in this session, or `None`. The one
    implementation is the live session's (`agent/session_lifecycle.py`); this module only
    renders what it keeps.
    """

    def keep(self, text: str, *, tool_name: str | None) -> str:
        """Keep `text` in full under the session and return its handle."""
        ...

    def read(self, handle: str) -> str | None:
        """The body `handle` names in this session, or `None` when it holds none."""
        ...


def is_result_handle(handle: str) -> bool:
    """Whether `handle` has the form of a stored result's handle."""
    return _HANDLE_RE.fullmatch(handle) is not None


def load_stored_result(bodies: ResultBodies | None, handle: str) -> str:
    """The full body `handle` names in this session, or a plain refusal.

    Raises:
        StoredResultNotFoundError: `handle` is not a handle's form, or the session holds
            no body under it. The message is plain words only.
    """
    if not is_result_handle(handle):
        raise StoredResultNotFoundError(
            f"'{handle}' is not a stored tool result name. Names look like "
            "tr_ followed by 16 letters and digits, as shown in the shortened result."
        )
    body = bodies.read(handle) if bodies is not None else None
    if body is None:
        raise StoredResultNotFoundError(UNAVAILABLE_RESULT_MESSAGE)
    return body


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
            f"{STORED_RESULT_PREFIX}{handle}: {total:,} characters, {EXCERPT_NOTE}. "
            f"Its start and end are below. Read the rest with {TOOL_RESULT_READ_TOOL}"
            f'(handle="{handle}", offset={omitted_from}).]'
        )
    elif handle is not None:
        header = (
            f"{STORED_RESULT_PREFIX}{handle}: {total:,} characters, {EXCERPT_NOTE}. "
            "Its start and end are below. The rest was kept, but this agent has no "
            "tool to read it.]"
        )
    else:
        header = (
            f"{UNSTORED_EXCERPT_PREFIX}{total:,} characters, {EXCERPT_NOTE}. Its "
            "start and end are below. The rest could not be kept, so it cannot be read "
            "back.]"
        )
    marker = f"[... characters {omitted_from:,} to {omitted_to:,} not shown ...]"
    return f"{header}\n{head}\n{marker}\n{tail}"


def ingest_tool_text(
    text: str,
    *,
    bodies: ResultBodies | None,
    tool_name: str | None = None,
    readable: bool = True,
    cap_bytes: int = TOOL_RESULT_CAP_BYTES,
) -> str:
    """`text` as the history should hold it: unchanged under the cap, else an excerpt.

    Over the cap, the whole redacted text is kept in the session (`bodies.keep`) and the
    excerpt is rendered from that same body, so what the model sees and what
    `tool_result_read` returns are one text. With no `bodies` -- nothing holds this
    session -- the full text has nowhere to go: the excerpt says so in band and it is
    logged, rather than the turn failing over a result the tool did produce.
    """
    return ingest_tool_result(
        text, bodies=bodies, tool_name=tool_name, readable=readable, cap_bytes=cap_bytes
    )[0]


def ingest_tool_result(
    text: str,
    *,
    bodies: ResultBodies | None,
    tool_name: str | None = None,
    readable: bool = True,
    cap_bytes: int = TOOL_RESULT_CAP_BYTES,
) -> tuple[str, str | None]:
    """`ingest_tool_text`, with the handle the full text was kept under (#1974).

    The handle is `bodies.keep`'s own, so a caller recording the form never reads it back
    out of the excerpt's text. `None` when the text fits the cap, or when no `bodies`
    holds it and the excerpt says the rest cannot be read.
    """
    if _fits(text, cap_bytes):
        return text, None
    body = redact_credentials(text)
    handle: str | None = None
    if bodies is not None:
        handle = bodies.keep(body, tool_name=tool_name)
    else:
        logger.warning("No session holds an over-cap tool result; it is kept as an excerpt only")
    return excerpt_tool_result(body, handle, cap_bytes=cap_bytes, readable=readable), handle


def read_tool_result_page(
    bodies: ResultBodies | None,
    handle: str,
    offset: int = 0,
    length: int | None = None,
    *,
    cap_bytes: int = TOOL_RESULT_CAP_BYTES,
) -> str:
    """One window of a kept result, starting at character `offset`, that fits the cap.

    `length` asks for at most that many characters; the window is shorter when the cap
    requires it, and the header says exactly which characters it holds and where the
    next window starts, so paging needs no arithmetic from the reader.
    """
    body = load_stored_result(bodies, handle)
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


def stored_result_stub(handle: str, body: str, *, keep_chars: int, readable: bool = True) -> str:
    """The compacted form of the stored `body` named `handle`: a header and its start.

    It keeps the first `keep_chars` characters, so the model still sees what the result
    was about. With `readable=False` -- this agent cannot call the reader -- the header
    says the rest was kept but cannot be read, rather than naming a tool that is not
    offered, as `excerpt_tool_result` does (P6).
    """
    if readable:
        where = f'Read it with {TOOL_RESULT_READ_TOOL}(handle="{handle}", offset=0).]'
    else:
        where = "It was kept, but this agent has no tool to read it.]"
    header = f"{STORED_RESULT_PREFIX}{handle}: {len(body):,} characters, {STUB_NOTE}. {where}"
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
