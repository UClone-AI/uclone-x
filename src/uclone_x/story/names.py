"""Codex names, ids and aliases, to the entries they name (#1557, #1808).

A model names a character the way the prose does -- `도윤`, `Lord Vane` -- more often than
by its entry id: qwen3:8b passed a display name to `story_codex propose` in 8 of 8 tries
(#1808). An id, a name and an alias are one key here, compared as `quotes.folded` reads
text: ignoring case, Unicode normalisation form and runs of white space.

`resolve` gives the first entry a name was seen on, as the audit reads a fact's subject.
`matches` gives every entry, so a caller that must not guess -- a codex change -- can
refuse a name two entries share. `named_in` finds the entries a free text names, for
handing a character's appearance to another persona. `id_for_name` makes the id of a new
entry from its name, as the codex's ids are written (`라온` is `raon`, `Lord Vane` is
`lord_vane`).

This module is pure.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from uclone_x.story.context import CodexIndex, CodexItem
from uclone_x.story.quotes import folded

__all__ = ["MIN_NAME_CHARACTERS", "CodexNames", "id_for_name"]

#: A name shorter than this is not looked for in a free text: one letter matches everything.
MIN_NAME_CHARACTERS = 2


def _keys(item: CodexItem) -> list[str]:
    return [folded(name) for name in (item.entry.id, item.entry.name, *item.entry.aliases)]


class CodexNames:
    """Every codex entry by its folded id, name and aliases."""

    def __init__(self, codex: CodexIndex) -> None:
        self._order = codex.items
        self._items: dict[str, list[CodexItem]] = {}
        for item in codex.items:
            for key in _keys(item):
                if not key:
                    continue
                named = self._items.setdefault(key, [])
                if item not in named:
                    named.append(item)

    def resolve(self, name: str) -> str | None:
        """The id of the first entry `name` names, or `None`."""
        found = self._items.get(folded(name))
        return found[0].entry.id if found else None

    def matches(self, name: str) -> list[CodexItem]:
        """Every entry whose id, name or an alias is `name`, in codex order."""
        return list(self._items.get(folded(name), ()))

    def named_in(self, texts: Iterable[str]) -> list[CodexItem]:
        """The entries whose name, id or an alias occurs in any of `texts`, in codex order.

        Matched inside words, as `context` finds a name in a scene: a Korean name carries
        its particle (`도윤이`), so a word edge would miss it. Names shorter than
        `MIN_NAME_CHARACTERS` are not looked for.
        """
        haystack = "\n".join(folded(text) for text in texts)
        named = {
            n
            for key, items in self._items.items()
            if len(key) >= MIN_NAME_CHARACTERS and key in haystack
            for n, item in enumerate(self._order)
            if item in items
        }
        return [self._order[n] for n in sorted(named)]


# Revised Romanization of Korean, letter by letter: the ids of a Korean story's codex are
# its names written this way (`예린` is `yerin`, #1808). The sound changes between
# syllables are not applied; an id only has to be readable and stable.
_INITIALS = "g kk n d tt r m b pp s ss _ j jj ch k t p h".split()
_VOWELS = "a ae ya yae eo e yeo ye o wa wae oe yo u wo we wi yu eu ui i".split()
_FINALS = [""] + "k k k n n n t l k m l l l p l m p p t t ng t t k t p t".split()
_HANGUL_FIRST, _HANGUL_LAST = 0xAC00, 0xD7A3
_ID_LENGTH = 80


def _romanized(text: str) -> str:
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if not _HANGUL_FIRST <= code <= _HANGUL_LAST:
            out.append(ch)
            continue
        index = code - _HANGUL_FIRST
        initial = _INITIALS[index // 588]
        out.append(("" if initial == "_" else initial) + _VOWELS[index % 588 // 28])
        out.append(_FINALS[index % 28])
    return "".join(out)


def id_for_name(name: str) -> str | None:
    """The codex id for a new entry called `name`, or `None` when none can be made.

    Korean is romanized; accents are dropped from Latin letters; every other run of
    characters becomes one `_`. A name in a script with no such spelling (Chinese,
    Japanese) gives `None`, and the caller asks for an id.
    """
    decomposed = unicodedata.normalize("NFKD", _romanized(unicodedata.normalize("NFC", name)))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
    words = re.findall(r"[a-z0-9]+", ascii_only)
    made = "_".join(words)[:_ID_LENGTH].strip("_")
    return made or None
