"""The context state: what each request of a session showed, per epoch, in which form (#1443).

The session log (`core/session_log.py`) is the record: every message that entered the
history. The context state is what the model was shown of it. For each epoch it holds an
ordered list of `(entry id, form)`, one per conversation message of the latest request,
and it grows only by appending while the epoch lasts:

* **Rule 1 -- within an epoch the context only appends.** `advance` extends the current
  epoch when a request's conversation starts with everything the epoch has shown, and
  opens a new epoch otherwise. What opened it is recorded (`opened_by`): a compaction at
  a turn boundary, or one of the divergences the design declares -- a rollback (§5.6), a
  nudged retry (§5.5), a history a caller replaced. A new epoch nothing declared says
  `undeclared`, so a test can find it.
* **Rule 2 -- an entry already shown is not sent again.** `render_conversation` replaces a
  tool result whose text a result earlier in the same conversation already carries with a
  one-line back-reference to it. The conversation is the epoch's rendering -- a compaction
  starts a new one -- so "earlier in the conversation" is "already shown this epoch".
  Whether a message is a back-reference depends only on the messages before it, so a
  request never changes how an earlier one rendered an entry.
* **Rule 3 -- forms drop only at a compaction.** A form is part of an epoch's list, so it
  cannot change inside one. At a compaction an entry keeps its id and drops a form: the
  new epoch lists the same entry in the smaller form, and names the log entry whose body
  is that rendering (`ContextEntry.rendering`, #1848).
* **Rule 4 -- a stub is the index.** A stub names the `tr_` handle `tool_result_read`
  reads (`core/tool_results.py`). Its form is recorded on the message where compaction
  made it (`ChatMessage.form`), not read back from its header (#1854).

A request's conversation is rendered from its entries (`render_entries`, #1848): each
entry in its form, read from the session log -- the entry's own body, or the body of the
rendering its form names -- and a back-reference where the entry records one.
`SessionState.messages` names which log bodies the history holds and in what order: each
message as the log holds it, a form with its source and no text, so a saved record is
never a rendering (#1848). A request's entries are read from the epoch in force alone:
the last epoch's entries, forms and renderings, or the opening state a turn-boundary
compaction or a rollback set for the next one, plus what was appended since
(`opening_entries`, `shown_entries`). No earlier epoch is read.

At a compaction the new epoch's entries are derived from the entries the history showed
before it and the compactor's account of where each message came from
(`derive_compacted_forms`): a kept entry keeps its form, the ledger is `summary`, and a
pruned message is the entry it replaces, shown in the smaller form the compactor gave it,
which may not rank above the form it had (`FORM_ORDER`). A smaller form is recorded, not
written: the log keeps the message with its `form` and what it was cut from
(`ChatMessage.rendered_from`) and no text, and every reader -- the live request and the
log-only rebuild alike -- renders the text from the kept full body (`render_form`,
`rendered_message`, #1848). A form that records nothing to render from is refused, never
shown as some text of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from uclone_x.core.tool_results import excerpt_tool_result, stored_result_stub
from uclone_x.llm.models import ChatMessage, MessageRole, RenderedFrom

__all__ = [
    "BACK_REFERENCE_MIN_CHARS",
    "EPOCH_PERSONA_EDITED",
    "EPOCH_RESTORED",
    "EPOCH_TOOLS_MODULE_CHANGED",
    "EPOCH_UNDECLARED",
    "FORM_ORDER",
    "CompactedForms",
    "ContextEntry",
    "ContextEpoch",
    "ContextForm",
    "advance",
    "back_reference_text",
    "back_references",
    "compacted_entries",
    "derive_compacted_forms",
    "message_form",
    "opening_entries",
    "render_conversation",
    "render_entries",
    "render_form",
    "rendered_message",
    "shown_entries",
]

#: A tool result shorter than this is sent again rather than back-referenced: the
#: back-reference line itself is a couple of hundred characters, so below this it would
#: save nothing and cost the model a lookup.
BACK_REFERENCE_MIN_CHARS: Final = 512

#: What `advance` records as having opened an epoch when nothing declared a divergence.
EPOCH_UNDECLARED: Final = "undeclared"

#: The cause a turn declares when a saved persona edit took effect at its start (#1899).
#: The history can still extend the epoch -- the edit changes the identity layer, not the
#: conversation -- so this cause opens an epoch even then: the log shows the boundary the
#: identity changed at, which an extended epoch would hide.
EPOCH_PERSONA_EDITED: Final = "persona_edited"

#: The cause loading a session declares. A record does not save it: every load declares
#: it again, while a cause declared before the save, such as `compaction`, is saved with
#: the record and read back beside it (#1848).
EPOCH_RESTORED: Final = "restored"

#: The cause a request declares when its tools module is not the one the session last sent
#: under (#2188): a seat rebuilt with another module, after a restart or in a new build of
#: the clone. Like `persona_edited`, it opens an epoch even when the history only grew,
#: because the tools layer changed there.
EPOCH_TOOLS_MODULE_CHANGED: Final = "tools_module_changed"

#: Declared causes that open an epoch even when the request only appended to the last one.
_EPOCH_FORCING: Final = frozenset({EPOCH_PERSONA_EDITED, EPOCH_TOOLS_MODULE_CHANGED})


class ContextForm(StrEnum):
    """How an entry is shown, from most to least detailed (design §5.8, *Forms*).

    `HIDDEN` is not listed in an epoch: an entry of the log that an epoch does not list is
    not shown in it.
    """

    FULL = "full"
    EXCERPT = "excerpt"
    STUB = "stub"
    SUMMARY = "summary"
    HIDDEN = "hidden"


#: The forms an epoch lists, from most to least detailed (design §5.8, Rule 3). At a
#: compaction an entry's form may stay or move right, never left.
FORM_ORDER: Final = (ContextForm.FULL, ContextForm.EXCERPT, ContextForm.STUB, ContextForm.SUMMARY)


class ContextEntry(BaseModel):
    """One conversation message of a request: the log entry it is, and its form."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    entry: str = Field(description="The session log entry's id, `e<n>`.")
    form: ContextForm
    same_as: str | None = Field(
        default=None,
        description="When the message was sent as a back-reference (Rule 2): the entry "
        "shown earlier in the epoch that carries the same text.",
    )
    rendering: str | None = Field(
        default=None,
        description="When a compaction dropped the entry to a smaller form (Rule 3, #1848): "
        "the log entry whose body is the entry in `form`. The entry keeps its own body; "
        "this epoch shows it through the rendering. `None` when the entry's own body is "
        "what is shown.",
    )

    @model_serializer(mode="wrap")
    def _omit_no_rendering(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Leave `rendering` out when it is unset, so an entry without one is written in
        the shape a build from before the field reads (#1844)."""
        data: dict[str, object] = handler(self)
        if data.get("rendering") is None:
            data.pop("rendering", None)
        return data

    @property
    def body(self) -> str:
        """The log entry whose body this entry is shown as: its rendering, or itself."""
        return self.rendering if self.rendering is not None else self.entry


class ContextEpoch(BaseModel):
    """What the requests of one epoch showed, in order; it only grows (Rule 1)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    number: int = Field(description="The epoch's position among the session's epochs.")
    turn: int = Field(description="The session's turn counter at the request that opened it.")
    step: int = Field(description="The step of that turn whose request opened it.")
    opened_by: tuple[str, ...] = Field(
        description="What opened it: `start` for a session's first epoch, `compaction`, "
        "`rollback`, `retry`, `history_replaced`, `restored`, `persona_edited`, "
        "`tools_module_changed`, or `undeclared`."
    )
    entries: tuple[ContextEntry, ...]


def message_form(message: ChatMessage) -> ContextForm:
    """The form `message` shows its entry in.

    A compaction ledger is the `summary` form. A tool result is the form recorded on it
    where it was shortened (`ChatMessage.form`): `excerpt` where ingest, the step budget or
    compaction cut it to a head and tail, `stub` where compaction stored it and kept a
    handle. Everything else is `full` -- including a tool result whose own text happens to
    begin like a stub's or an excerpt's header, such as a file that quotes one (#1854). The
    text is never parsed to decide the form.
    """
    if message.role == MessageRole.SYSTEM and message.compaction_ledger:
        return ContextForm.SUMMARY
    if message.role == MessageRole.TOOL and message.form is not None:
        return ContextForm(message.form)
    return ContextForm.FULL


def opening_entries(
    epochs: Sequence[ContextEpoch], pending: Mapping[str, ContextEntry]
) -> dict[str, ContextEntry]:
    """Per log body of the history, the entry it shows in the epoch in force (#1848).

    Read from the last epoch alone: each rendering it lists, keyed by the rendering's own
    log entry -- the body the history holds -- with the entry and form it renders.
    `pending` is the opening state a turn-boundary compaction or a rollback set for the
    request that opens the next epoch (`compacted_entries`), and it wins. No earlier
    epoch is read: an epoch opens with every entry it shows, so what one before it
    recorded is either carried in it or no longer shown.
    """
    renders: dict[str, ContextEntry] = {}
    if epochs:
        for shown in epochs[-1].entries:
            if shown.rendering is not None:
                renders.setdefault(shown.rendering, shown.model_copy(update={"same_as": None}))
    return {**renders, **pending}


def shown_entries(
    logged: Sequence[tuple[str, ChatMessage]],
    known: Mapping[str, ContextEntry],
) -> tuple[ContextEntry, ...]:
    """What a history shows, per message: the entry, its form, and its back-reference.

    `logged` is the history as the session log holds it, each message rendered, with the
    log entry whose body it is. A body `known` names -- a rendering an epoch recorded
    (`opening_entries`), or what a compaction derived (`compacted_entries`) -- shows
    the entry `known` gives, in that form. Any other body is its own entry, in the form
    recorded on its message (`message_form`). A tool result that repeats an earlier one's
    text refers to the entry that one shows (Rule 2, `back_references`).
    """
    bases = [
        known.get(body) or ContextEntry(entry=body, form=message_form(message))
        for body, message in logged
    ]
    refers = back_references([message for _body, message in logged])
    return tuple(
        base if earlier is None else base.model_copy(update={"same_as": bases[earlier].entry})
        for base, earlier in zip(bases, refers, strict=True)
    )


@dataclass(frozen=True, slots=True)
class CompactedForms:
    """What a compacted history shows, per message, and the rewrites refused.

    `forms` gives each message's form. `shows` gives the entry a message shows, when it
    is the entry of a message from before the compaction -- kept, or pruned to a smaller
    form -- and `None` for a message that is its own entry, such as the ledger the pass
    wrote. `rising` lists the positions whose pruned message would show its entry in a
    form ranked above the one it had; those positions keep that form and the caller keeps
    the message from before instead of the rewrite.
    """

    forms: tuple[ContextForm, ...]
    shows: tuple[str | None, ...]
    rising: tuple[int, ...]


def derive_compacted_forms(
    before: Sequence[tuple[ContextEntry, ChatMessage]],
    after: Sequence[ChatMessage],
    origins: Sequence[int | None],
) -> CompactedForms:
    """The new epoch's entry and form of each message a compaction left (Rule 3, #1848).

    `before` is the history the compaction was given, each message with the entry the
    history showed it as (`shown_entries`: the previous epoch's entries, read from the
    log). `after` is what the compaction returned, and `origins` gives, per message of
    `after`, the index in `before` of the message it renders, or `None` for a ledger the
    pass wrote.

    * A ledger the pass wrote is `summary`, and its own entry.
    * A message the compaction kept as it was is its entry, carried over in its form.
    * A pruned message is the entry it replaces, dropped to the form the compactor
      recorded on it where it shortened it (#1854); its text is that entry's rendering in
      the form. The form may not rank above the one the entry had (`FORM_ORDER`): such a
      rewrite is listed in `rising`, and its position keeps the entry's form, for the
      caller to keep that message. A rewrite that is still `full` is not a smaller form
      of anything, so it is its own entry.

    Raises:
        ValueError: `origins` does not match `after`, names no message of `before`, or
            names none for a message that is not a compaction ledger.
    """
    if len(origins) != len(after):
        raise ValueError(f"{len(origins)} origins for {len(after)} compacted messages")
    forms: list[ContextForm] = []
    shows: list[str | None] = []
    rising: list[int] = []
    for position, (message, origin) in enumerate(zip(after, origins, strict=True)):
        if origin is None:
            if message_form(message) is not ContextForm.SUMMARY:
                raise ValueError(f"compacted message {position} has no origin and is no ledger")
            forms.append(ContextForm.SUMMARY)
            shows.append(None)
            continue
        if not 0 <= origin < len(before):
            raise ValueError(f"compacted message {position} names input {origin}")
        previous, source = before[origin]
        if message == source:
            forms.append(previous.form)
            shows.append(previous.entry)
            continue
        pruned = message_form(message)
        if FORM_ORDER.index(pruned) < FORM_ORDER.index(previous.form):
            rising.append(position)
            forms.append(previous.form)
            shows.append(previous.entry)
        else:
            forms.append(pruned)
            shows.append(None if pruned is ContextForm.FULL else previous.entry)
    return CompactedForms(forms=tuple(forms), shows=tuple(shows), rising=tuple(rising))


def compacted_entries(bodies: Sequence[str], derived: CompactedForms) -> dict[str, ContextEntry]:
    """Per log entry of a compacted history, the entry it shows and in which form (#1848).

    `bodies` is the log entry each message of the compacted history is, once logged; a
    message that shows another entry is that entry's rendering.
    """
    if len(bodies) != len(derived.forms):
        raise ValueError(f"{len(bodies)} logged messages for {len(derived.forms)} forms")
    return {
        body: ContextEntry(entry=body, form=form)
        if shows is None or shows == body
        else ContextEntry(entry=shows, form=form, rendering=body)
        for body, form, shows in zip(bodies, derived.forms, derived.shows, strict=True)
    }


def back_reference_text(earlier: ChatMessage) -> str:
    """The line sent in place of a tool result `earlier` already carried (Rule 2)."""
    name = earlier.name or "tool"
    call = earlier.tool_call_id or ""
    return f"[Same output as the earlier `{name}` result above (call `{call}`); not repeated here.]"


def render_conversation(
    messages: Sequence[ChatMessage],
) -> tuple[list[ChatMessage], list[int | None]]:
    """The conversation as a request sends it, and what each message back-references.

    A tool result of at least `BACK_REFERENCE_MIN_CHARS` whose text an earlier tool result
    in `messages` carries is sent as `back_reference_text` of the first one; its call id
    and name stay, so every call is still answered. The second list gives, per message,
    the index of the message it refers to, or `None`. Deterministic: the same messages
    give the same rendering.
    """
    refers = back_references(messages)
    rendered = [
        message
        if earlier is None
        else message.model_copy(update={"content": back_reference_text(messages[earlier])})
        for message, earlier in zip(messages, refers, strict=True)
    ]
    return rendered, refers


def back_references(messages: Sequence[ChatMessage]) -> list[int | None]:
    """For each message, the earlier one it is sent as a back-reference to, or `None`.

    Rule 2's decision on its own: a tool result of at least `BACK_REFERENCE_MIN_CHARS`
    whose text an earlier tool result carries refers to the first that did.
    """
    first_with: dict[str, int] = {}
    refers: list[int | None] = []
    for index, message in enumerate(messages):
        content = message.content
        earlier: int | None = None
        if (
            message.role == MessageRole.TOOL
            and content is not None
            and len(content) >= BACK_REFERENCE_MIN_CHARS
        ):
            earlier = first_with.get(content)
            if earlier is None:
                first_with[content] = index
        refers.append(earlier)
    return refers


def render_form(form: ContextForm, body: str, source: RenderedFrom) -> str:
    """The text of `form` rendered from the kept result `body`, as `source` says (#1848).

    The same functions that cut the result where its form was decided: an `excerpt` is
    `excerpt_tool_result` at the cap `source.limit`, a `stub` is `stored_result_stub`
    keeping `source.limit` characters of the start. So the text depends only on the body
    and the recorded parameters.

    Raises:
        ValueError: `form` is not `excerpt` or `stub`.
    """
    if form is ContextForm.EXCERPT:
        return excerpt_tool_result(
            body, source.handle, cap_bytes=source.limit, readable=source.readable
        )
    if form is ContextForm.STUB:
        return stored_result_stub(
            source.handle, body, keep_chars=source.limit, readable=source.readable
        )
    raise ValueError(f"a {form.value} is not rendered from a kept result")


def rendered_message(message: ChatMessage, result_of: Callable[[str], str]) -> ChatMessage:
    """`message` with its text, rendered from the kept result its form names (#1848).

    The log keeps an `excerpt` or `stub` as its form and what it was cut from
    (`ChatMessage.rendered_from`), not as text; this renders the text from the full body
    `result_of` gives for that handle (`render_form`). Any other message is its own text
    and is returned as it is.

    Raises:
        ValueError: `message` has a form but records nothing it was rendered from, so
            there is no text it could be shown as.
    """
    source = message.rendered_from
    if source is None:
        if message.form is not None:
            raise ValueError(
                f"a tool result in the {message.form} form records no result it was rendered from"
            )
        return message
    text = render_form(message_form(message), result_of(source.handle), source)
    return message.model_copy(update={"content": text})


def render_entries(
    entries: Sequence[ContextEntry],
    message_of: Callable[[str], ChatMessage],
    result_of: Callable[[str], str],
) -> list[ChatMessage]:
    """The conversation an epoch's `entries` render to, reading each entry in its form.

    `message_of` gives the message a log entry's body decodes to. An entry is read from
    the body of its `rendering` when a compaction dropped it to a smaller form, and from
    its own body otherwise (`ContextEntry.body`); nothing is read from the history. An
    entry recorded as a back-reference (`same_as`) renders as the line that points at the
    entry it repeats. The live request and a rebuild from the record both render through
    this (#1848), so the request is what its entries say.

    An `excerpt` or `stub` is rendered from the full body `result_of` gives for the handle
    it records (`rendered_message`): the log holds no text for it.

    Raises:
        ValueError: A message's form is not the form its entry records, or it records
            nothing it was rendered from.
    """
    rendered: list[ChatMessage] = []
    read: dict[str, ChatMessage] = {}
    for shown in entries:
        message = message_of(shown.body)
        if message_form(message) is not shown.form:
            raise ValueError(
                f"entry {shown.entry} is recorded as {shown.form.value} but its message "
                f"is {message_form(message).value}"
            )
        message = rendered_message(message, result_of)
        read.setdefault(shown.entry, message)
        if shown.same_as is not None:
            earlier = read.get(shown.same_as) or rendered_message(
                message_of(shown.same_as), result_of
            )
            message = message.model_copy(update={"content": back_reference_text(earlier)})
        rendered.append(message)
    return rendered


def advance(
    epochs: Sequence[ContextEpoch],
    shown: Sequence[ContextEntry],
    *,
    turn: int,
    step: int,
    opened_by: Sequence[str] = (),
) -> tuple[ContextEpoch, ...]:
    """`epochs` after a request that showed `shown`.

    When `shown` begins with everything the current epoch lists, the epoch is extended
    by the rest (Rule 1). Otherwise a new epoch opens with `shown`, recording `opened_by`,
    or `start` for a session's first, or `EPOCH_UNDECLARED` when nothing was declared.
    A declared `persona_edited` or `tools_module_changed` opens a new epoch even when
    `shown` extends the current one, so the identity or tools change has a boundary in
    the log.
    """
    shown = tuple(shown)
    if epochs and _EPOCH_FORCING.isdisjoint(opened_by):
        current = epochs[-1]
        if shown[: len(current.entries)] == current.entries:
            if len(shown) == len(current.entries):
                return tuple(epochs)
            return (*epochs[:-1], current.model_copy(update={"entries": shown}))
    reasons = tuple(opened_by) or (("start",) if not epochs else (EPOCH_UNDECLARED,))
    opened = ContextEpoch(
        number=len(epochs), turn=turn, step=step, opened_by=reasons, entries=shown
    )
    return (*epochs, opened)
