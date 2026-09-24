"""The session log's ordering: a durable, per-session, gap-free offset.

`AgentEvent.sequence` is not this, and the difference is the reason this module exists.
`EventBus._next_sequence` increments one counter from zero at bus construction, so it is
**bus-global** (one counter across every session), **not durable** (it restarts at zero, so
offset 1 in a second run collides with offset 1 in the first), and **not authoritative** —
the bus stamps only when `event.sequence == 0`, so a publisher may supply its own value and
the bus keeps it. Its own field description says what it is for: deterministic FIFO
tie-breaking. That is a useful number and it is not a log offset.

An append-only log needs the opposite properties on all three counts, because everything the
record promises is a statement about ordering that survives a restart: that nothing is
deleted, that a surface can be recomputed, and that a shadowed range names seqs a later reader
can still resolve.

**Allocated at the persistence boundary, never supplied by a caller.** A number a writer hands
in is a number the store cannot vouch for, and the store is what a reader trusts. This mirrors
the reasoning already in `SessionStore.save`, which re-validates the payload it is given
rather than trusting the object it was handed.

**Contiguity is checked, not assumed.** A batch must continue the stored log from its cursor.
A gap is refused rather than accepted, because a hole in an append-only log is
indistinguishable, later, from an event that was never written — and #526 adopted this record
model precisely so that "what happened" is answerable.

**Identity is checked before ordering.** A batch naming a different session is refused as an
id mismatch, not as an ordering conflict. UClone-X learned this in #256, where the reverse
order reported a stale-revision error naming a writer and a revision for a session the user
had never heard of.

**Relationship to `SessionState.revision`, stated rather than left to be inferred.** They are
different mechanisms protecting different things and neither replaces the other:

* `revision` is **per record**. It guards a whole-document overwrite: `SessionStore.save`
  compares the caller's revision with the stored one and refuses on any difference — ahead as
  well as behind — and that file documents the bound on what the precondition buys, including
  that an exactly-forged value is accepted.
* `LogOffset` is **per event**. It guards append ordering: a batch must continue the log at
  its cursor, and the cursor must belong to the session named.

A design that introduces the second without saying this reintroduces the divergence #219 was
filed for. Which of the two owns durability once the log is the record is #567's decision;
this module does not pre-empt it, and keeps the cursor beside the log rather than inside
`SessionState` so that the log's ordering and the snapshot's revision stay separable for that
decision to be made.

> An earlier draft justified that placement by claiming `extra="forbid"` would make carrying
> the offset in `SessionState` a schema change to every record on disk. That is false, and the
> repository contains its own counter-example: `extra="forbid"` rejects unknown keys in input,
> while a new field with a default reads legacy records fine — which is exactly what
> `revision` does, where `0` "is also what a legacy record with no `revision` key reads back
> as". The placement may still be right; that argument for it was not.

Design reference: the core/shell architecture note; decision #526 (Option B).
"""

from __future__ import annotations

from typing import NewType, Protocol

__all__ = ["FIRST_LOG_OFFSET", "LogOffset", "LogOffsetAllocatorProtocol", "MAX_LOG_OFFSET"]

LogOffset = NewType("LogOffset", int)
"""A position in one session's log.

A `NewType` over `int` rather than a bare `int`, so a bus sequence cannot be passed where a
log offset is expected without the type checker objecting. The two are both integers and mean
different things, and conflating them is the specific error this module was written against.
"""

FIRST_LOG_OFFSET: LogOffset = LogOffset(1)
"""The offset of a session's first event.

One rather than zero, so that `0` remains distinguishable as "unset" in the same way
`SessionState.revision` uses it for "never persisted".
"""

MAX_LOG_OFFSET: LogOffset = LogOffset(2**53)
"""The largest offset every JSON consumer can represent exactly.

It lives in the kernel rather than in a backend because it constrains the *wire format*, not
the medium: any backend that serialises an offset to JSON inherits it, so a backend-local
bound would be a different limit per backend for one shared format.

The reason is a consumer that parses JSON numbers as IEEE-754 doubles, as every JavaScript
client does — above 2**53 two distinct offsets arrive as the same number, and an offset that is
not distinct is not an offset.

Two things this is **not**. It is not a Python limitation: `json.loads(json.dumps(2**70))` is
exact, and an earlier version of this bound was justified by claiming otherwise. And there is
no such consumer today — an earlier version named this repository's TypeScript UI, which does
not reference offsets at all. This is forward-looking design for the format #526 will expose,
and it stands without the invented evidence.

A log that reaches 9e15 events has a different problem; refusing the value is how that problem
is found rather than silently mis-rendered.
"""


class LogOffsetAllocatorProtocol(Protocol):
    """Allocates and validates offsets for one session's log.

    All three methods are on the protocol. An earlier draft declared two and left
    `record_appended` off, so a backend written against the protocol would have omitted the
    only method that persists anything — the shape review finding 2026-09-02-035,
    "protocols unbound to implementations", records. (The Markdown findings register that
    held it is retired; the snapshot survives as an eval fixture.)
    """

    def next_offset(self, session_id: str) -> LogOffset:
        """The offset the next event appended to `session_id` will occupy.

        Returns `FIRST_LOG_OFFSET` for a session with no log. Must survive a process
        restart: a second run continues where the first stopped rather than starting over.

        **Not safe against a concurrent allocator on its own.** Two allocators over one store
        both read the same value, and the read-modify-write to `record_appended` has an
        unbounded window between them. A backend that admits concurrent writers owes an
        exclusion mechanism; the shipped file backend does not admit them and says so.
        """
        ...

    def validate_batch(
        self, session_id: str, offsets: tuple[LogOffset, ...], cursor: LogOffset
    ) -> None:
        """Refuse a batch that does not continue `session_id`'s log at `cursor`.

        Two distinct refusals, and the distinction is the contract rather than a nicety:

        * **Identity** — the stored log does not belong to `session_id`. Raised from a
          comparison of *identities*, never inferred from a number agreeing. A cursor that
          happens to line up is not evidence the session matches.
        * **Contiguity** — the session is right and the offsets do not continue its log.

        Identity is checked first (#256): an id mismatch reported as an ordering conflict
        names a position in a log the caller never meant.
        """
        ...

    def record_appended(self, session_id: str, offsets: tuple[LogOffset, ...]) -> None:
        """Advance the cursor after a batch has been durably written.

        Called after the write, never before. Must refuse a batch that does not continue the
        stored log, for the same reason `validate_batch` does — a cursor that moves backward
        makes the next batch re-issue offsets the log already holds, which is duplicate
        positions in an append-only log and is not recoverable by reading it back.
        """
        ...
