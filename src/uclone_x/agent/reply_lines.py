"""Lines the code appends to a turn's final reply, rather than asking the model to (#1808).

A small local model does not reliably repeat what a tool asked it to tell the person: in
the 2026-09-28 Writer eval on qwen3:8b it named the seed of an idea draw in 0 of 8 replies,
and told the person what it kept of the story in 0 of 15. So the line is written by code.

A tool asks for one by putting `REPLY_NOTE_KEY` in its output: a string, or a mapping with
an `en` and a `ko` string, chosen by the language of the person's message (replies follow
the language they are addressed in). A turn aid (`TurnAidProtocol.reply_lines`) can add
lines too. Pure functions; none reads the agent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.tools.models import REPLY_NOTE_KEY, ToolResultStatus

__all__ = ["REPLY_NOTE_KEY", "is_korean", "lines_to_add", "reply_notes", "with_lines"]

_HANGUL = re.compile(r"[가-힣]")


def is_korean(text: str) -> bool:
    """Whether `text` is written in Korean: it holds a Hangul syllable."""
    return bool(_HANGUL.search(text))


def _note_of(output: object, *, korean: bool) -> str | None:
    if not isinstance(output, Mapping):
        return None
    note = cast(Mapping[str, object], output).get(REPLY_NOTE_KEY)
    if isinstance(note, Mapping):
        note = cast(Mapping[str, object], note).get("ko" if korean else "en")
    return note.strip() if isinstance(note, str) and note.strip() else None


def reply_notes(records: Iterable[ToolExecutionRecord], *, korean: bool) -> list[str]:
    """The reply notes of the turn's successful tool calls, in the order they ran."""
    notes: list[str] = []
    for record in records:
        if record.status is not ToolResultStatus.SUCCESS:
            continue
        note = _note_of(record.output, korean=korean)
        if note is not None:
            notes.append(note)
    return notes


def lines_to_add(reply: str, lines: Sequence[str]) -> list[str]:
    """`lines` less the empty ones, the repeats, and those the reply already contains."""
    added: list[str] = []
    for line in lines:
        text = line.strip()
        if text and text not in added and text not in reply:
            added.append(text)
    return added


def with_lines(reply: str, lines: Sequence[str]) -> str:
    """`reply` with each of `lines` after it on a line of its own, a blank line between.

    `reply` is kept as it is, as the start of the result: what streamed of it stays true,
    and the rest streams as one more token.
    """
    if not lines:
        return reply
    appended = "\n".join(lines)
    if not reply.strip():
        return reply + appended
    return reply + "\n" * max(0, 2 - (len(reply) - len(reply.rstrip("\n")))) + appended
