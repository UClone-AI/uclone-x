"""Lines the code appends to a turn's final reply, rather than asking the model to (#1808).

A small local model does not reliably repeat what a tool asked it to tell the person: in
the 2026-09-28 Writer eval on qwen3:8b it named the seed of an idea draw in 0 of 8 replies,
and told the person what it kept of the story in 0 of 15. So the line is written by code.

A tool asks for one by putting `REPLY_NOTE_KEY` in its output: a string, or a mapping with
an `en` and a `ko` string, chosen by the language of the person's message (replies follow
the language they are addressed in). A turn aid (`TurnAidProtocol.reply_lines`) can add
lines too. Pure functions; none reads the agent.

A note that lists things the tool made (three premises, by title) names them under
`NOTE_LISTS_KEY`. Its list comes first and its question after: a reply that already names
every one of them has shown the list in its own words, so only the lines after the list are
added (#2301). The names decide, not the wording: a model that copies a list renumbers it,
bolds it and changes its dashes, and no comparison of lines saw the two as one. When the names are
worded per language (genre labels), `NOTE_LISTS_KEY` holds a list per language instead.

A note may also name its question lines under `NOTE_ASKS_KEY`: the line a stop ends on,
which a model copies into its own reply with other marks around the same words. A marked
line counts as asked when the reply holds its words -- its letters and digits, in order --
and only a line the tool marked is compared that way (#2305: the model's "'네'라고만
하셔도" and the note's "‘네’라고만 하셔도" were added twice).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.i18n.scripts import sentences
from uclone_x.tools.models import REPLY_NOTE_KEY, ToolResultStatus

__all__ = [
    "NOTE_ASKS_KEY",
    "NOTE_LISTS_KEY",
    "REPLY_NOTE_KEY",
    "is_korean",
    "lines_to_add",
    "reply_notes",
    "with_lines",
]

_HANGUL = re.compile(r"[가-힣]")


def is_korean(text: str) -> bool:
    """Whether `text` is written in Korean: it holds a Hangul syllable."""
    return bool(_HANGUL.search(text))


#: In a reply note's mapping, the names of what the note lists, in the note's own words.
NOTE_LISTS_KEY = "lists"
#: In a reply note's mapping, its question lines, as the note writes them.
NOTE_ASKS_KEY = "asks"


def _note_of(output: object, *, korean: bool, reply: str) -> str | None:
    if not isinstance(output, Mapping):
        return None
    note = cast(Mapping[str, object], output).get(REPLY_NOTE_KEY)
    listed: list[str] = []
    asks: list[str] = []
    if isinstance(note, Mapping):
        found = cast(Mapping[str, object], note)
        listed = _listed(found.get(NOTE_LISTS_KEY), korean=korean)
        asks = _strings(found.get(NOTE_ASKS_KEY))
        note = found.get("ko" if korean else "en")
    if not isinstance(note, str) or not note.strip():
        return None
    return _unasked(_after_list(note, listed, reply), asks, reply).strip() or None


def _listed(value: object, *, korean: bool) -> list[str]:
    """A note's listed names: one list for every language, or a mapping of a list per
    language (`ko`, `en`) when the names are worded per language (#2360: genre labels)."""
    if isinstance(value, Mapping):
        return _strings(cast(Mapping[str, object], value).get("ko" if korean else "en"))
    return _strings(value)


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [n for n in cast(list[object], value) if isinstance(n, str) and n.strip()]


def _words(text: str) -> str:
    """`text`'s letters and digits, in order, case folded: the words without their marks."""
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _unasked(note: str, asks: Sequence[str], reply: str) -> str:
    """`note` less each of its marked question lines whose words `reply` already holds."""
    if not asks or not reply.strip():
        return note
    held = [w for w in (_words(said) for said in sentences(reply)) if w]
    marked = {line.strip() for line in asks}
    return "\n".join(
        line
        for line in note.splitlines()
        if not (line.strip() in marked and _words(line) and _asked(line, held))
    )


def _asked(line: str, held: Sequence[str]) -> bool:
    """Whether each sentence of `line` is, word for word, inside one sentence of the reply
    (`held`, each sentence's words). Compared sentence by sentence, so a short line is not
    found across the end of one of the reply's sentences and the start of the next (#2317
    item 5). The sentences are cut by `i18n.scripts.sentences`, the same in every script."""
    wanted = [w for w in (_words(own) for own in sentences(line)) if w]
    return all(any(own in said for said in held) for own in wanted)


def _after_list(note: str, listed: Sequence[str], reply: str) -> str:
    """`note`, or only its lines after its list when `reply` names every listed thing.

    The list is the note's lines up to the last one that names a listed thing; what is
    before it introduces the list, and goes with it.
    """
    if not listed or not reply.strip():
        return note
    plain_reply = _plain(reply)
    names = [_plain(n) for n in listed]
    if not all(name and name in plain_reply for name in names):
        return note
    lines = note.splitlines()
    naming = [i for i, line in enumerate(lines) if any(name in _plain(line) for name in names)]
    return "\n".join(lines[naming[-1] + 1 :]) if naming else note


def reply_notes(
    records: Iterable[ToolExecutionRecord], *, korean: bool, reply: str = ""
) -> list[str]:
    """The reply notes of the turn's successful tool calls, in the order they ran.

    Given the turn's `reply`, a note whose listed things it already names gives only what
    follows its list (`NOTE_LISTS_KEY`).
    """
    notes: list[str] = []
    for record in records:
        if record.status is not ToolResultStatus.SUCCESS:
            continue
        note = _note_of(record.output, korean=korean, reply=reply)
        if note is not None:
            notes.append(note)
    return notes


#: Markdown emphasis or code marks wrapped around words (`**SF**`, `_Fantasy_`, `~~x~~`,
#: `` `x` ``): how a model that copied a line into its reply most often reworded it (#2263).
#: A mark inside a token wraps nothing and is kept, so `my_story.md` never reads as
#: `mystory.md`, nor `~/notes` as `/notes` (#2272). The token edges are ASCII only: `\w`
#: matches Hangul, and bold before a particle (`**미스터리**로`) is how a Korean reply bolds
#: a word, so a Unicode edge kept those marks and the note was added again (#2279 review;
#: the same edge as #2261's `\b` before a particle).
_WRAPPED = re.compile(
    r"(?<![A-Za-z0-9*_`~])([*_`~]+)(?=[^\s*_`~])(.+?)(?<=[^\s*_`~])\1(?![A-Za-z0-9*_`~])"
)


def _plain(text: str) -> str:
    """`text` as it reads: no emphasis or code marks around words, each line's spacing as
    single spaces, and no blank lines. Lines stay lines: a hard break ("  \n") is a line
    end, and two lines never read as one (#2272)."""
    lines = (" ".join(line.split()) for line in _WRAPPED.sub(r"\2", text).splitlines())
    return "\n".join(line for line in lines if line)


def lines_to_add(reply: str, lines: Sequence[str]) -> list[str]:
    """`lines` less the empty ones, the repeats, and those the reply already contains.

    A reply contains a line when it holds the line's words in order, line by line, whatever
    markdown emphasis or spacing either puts around them (`_plain`).
    """
    plain_reply = _plain(reply)
    added: list[str] = []
    for line in lines:
        text = line.strip()
        if text and text not in added and _plain(text) not in plain_reply:
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
