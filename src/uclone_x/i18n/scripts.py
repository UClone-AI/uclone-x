"""Which writing systems a text's letters are in, and the words written outside some (#2301).

A small local model slips into another script inside generated text: a premise of a
Korean story said "죄孽" and another was written whole in English (the 2026-10-12 Dell run,
#2299). The check is the same for every language: a letter's script is a property of the
character (its Unicode name), and a text is held to the scripts its own story is in. No
rule here names a language.

The script of a letter is the first word of its Unicode name ("HANGUL SYLLABLE GA" is
Hangul, "CJK UNIFIED IDEOGRAPH-5B7D" is Han, "LATIN SMALL LETTER A" is Latin), with the
width forms ("FULLWIDTH LATIN ...") read as their script. Only letters have a script here:
digits, punctuation, spaces and symbols are in every script's text. Pure functions.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Collection, Iterable

__all__ = ["letter_scripts", "script_of", "scripts_of", "sentences", "words_outside"]

#: A name's leading word that says how the letter is drawn, not which script it is in.
_FORMS = frozenset({"FULLWIDTH", "HALFWIDTH", "SMALL", "CIRCLED", "PARENTHESIZED"})
#: Leading words that name a block of a script by another word: the CJK ideographs are Han.
_ALIASES = {"CJK": "HAN"}


def script_of(char: str) -> str | None:
    """The script of a letter, upper case ("HANGUL", "HAN", "LATIN"); `None` for anything
    that is not a letter, or a letter whose name says no script ("MODIFIER LETTER ...")."""
    if not unicodedata.category(char).startswith("L"):
        return None
    words = unicodedata.name(char, "").split()
    while words and words[0] in _FORMS:
        words = words[1:]
    if not words or words[0] == "MODIFIER":
        return None
    return _ALIASES.get(words[0], words[0])


def letter_scripts(text: str) -> set[str]:
    """Every script a letter of `text` is in."""
    return {s for s in (script_of(ch) for ch in text) if s is not None}


def words_outside(text: str, scripts: Collection[str]) -> list[str]:
    """The words of `text`, once each and in order, that hold a letter in none of `scripts`.

    A word is what whitespace separates; it is given as written, punctuation and all.
    """
    allowed = {s.upper() for s in scripts}
    found: list[str] = []
    for word in text.split():
        if word in found:
            continue
        if any((s := script_of(ch)) is not None and s not in allowed for ch in word):
            found.append(word)
    return found


def scripts_of(texts: Iterable[str]) -> set[str]:
    """Every script a letter of any of `texts` is in."""
    found: set[str] = set()
    for text in texts:
        found |= letter_scripts(text)
    return found


#: The words in the Unicode name of a punctuation mark that ends a sentence, in any script:
#: ".", "。", "？", "!", "…", "।" and the rest are named by one of these.
_SENTENCE_END_NAMES = ("FULL STOP", "QUESTION MARK", "EXCLAMATION MARK", "ELLIPSIS", "DANDA")


def _ends_sentence(char: str) -> bool:
    if unicodedata.category(char) != "Po":
        return False
    name = unicodedata.name(char, "")
    return any(word in name for word in _SENTENCE_END_NAMES)


def _spaceless(char: str) -> bool:
    """Whether a mark is written with no space after it: the ideographic and full-width
    forms ("。", "！")."""
    name = unicodedata.name(char, "")
    return "IDEOGRAPHIC" in name or "FULLWIDTH" in name


def sentences(text: str) -> list[str]:
    """`text` cut into sentences, in order, each stripped: at a line break, and after a run
    of sentence-ending marks (with any closing quote or bracket after it) that whitespace
    or the end follows, or that ends in a mark written with no space after it ("。").
    "3.5" and "e.g.x" are not cut. The marks are found by their Unicode
    names, so the cut is the same for every script (#2303)."""
    found: list[str] = []
    for line in text.splitlines():
        start, index, size = 0, 0, len(line)
        while index < size:
            if not _ends_sentence(line[index]):
                index += 1
                continue
            end = index
            while end < size and (
                _ends_sentence(line[end])
                or unicodedata.category(line[end]) in ("Pe", "Pf")
                or line[end] in "\"'"
            ):
                end += 1
            if end == size or line[end].isspace() or _spaceless(line[end - 1]):
                found.append(line[start:end].strip())
                start = end
            index = end
        found.append(line[start:].strip())
    return [s for s in found if s]
