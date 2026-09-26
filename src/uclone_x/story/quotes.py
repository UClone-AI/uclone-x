"""Whether a quote is in a scene's text (#1557).

A fact the model reads from a scene, and a change it proposes to the codex, each carry the
words they rest on. The words must be in the scene: matched ignoring case, Unicode
normalisation form and runs of white space -- a model that re-wraps a line has still quoted
it -- and nothing looser. A paraphrase is not a quote.

A quote is also whole words and long enough to show something (#1584): at least
`MIN_QUOTE_CHARACTERS` letters or digits, and a match may not begin or end inside a word of
the scene -- `a` is in almost every scene, and `live` is in `delivered`. Scripts written
without spaces between words (Chinese, Japanese kana, Thai, Lao, Khmer, Myanmar) have no
word edge to find, so there only the length is required. That includes the marks written
among their characters: `々`, `〇` and halfwidth katakana (#1601).

Whether a proposal's passage is still in its scene is a looser question (`passage_in`): the
passage was checked as a quote when it was proposed, and a proposal made before a quote
had to be whole words is not "gone" because its passage is short (#1601).

This module is pure.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["MIN_QUOTE_CHARACTERS", "folded", "passage_in", "quote_found", "quote_too_short"]

_SPACE = re.compile(r"\s+")

#: The fewest letters or digits a quote may have.
MIN_QUOTE_CHARACTERS = 3

_UNSPACED_SCRIPTS = (
    "CJK UNIFIED IDEOGRAPH",
    "CJK COMPATIBILITY IDEOGRAPH",
    "HIRAGANA",
    "KATAKANA",
    "THAI",
    "LAO",
    "KHMER",
    "MYANMAR",
    # `々` (IDEOGRAPHIC ITERATION MARK), `〇` (IDEOGRAPHIC NUMBER ZERO), `〆`, `〻`.
    "IDEOGRAPHIC",
    "VERTICAL IDEOGRAPHIC",
    "HALFWIDTH KATAKANA",
)


def folded(text: str) -> str:
    """`text` in NFC, with runs of white space as one space, trimmed, and case-folded."""
    return _SPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def quote_too_short(quote: str) -> bool:
    """Whether `quote` has fewer than `MIN_QUOTE_CHARACTERS` letters or digits."""
    return sum(1 for ch in folded(quote) if ch.isalnum()) < MIN_QUOTE_CHARACTERS


def _in_word(ch: str) -> bool:
    """Whether `ch` is part of a word that has edges: a letter or digit of a spaced script."""
    return ch.isalnum() and not unicodedata.name(ch, "").startswith(_UNSPACED_SCRIPTS)


def _cuts_a_word(outside: str, inside: str) -> bool:
    """Whether a match edge between `outside` and `inside` falls inside a word."""
    return bool(outside) and _in_word(outside) and _in_word(inside)


def quote_found(quote: str, text: str) -> bool:
    """Whether `quote` is in `text`, as the module docstring says.

    An empty or too short quote is not, and neither is one found only inside a longer word.
    """
    if quote_too_short(quote):
        return False
    needle = folded(quote)
    haystack = folded(text)
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        before = haystack[start - 1] if start else ""
        after = haystack[end] if end < len(haystack) else ""
        if not _cuts_a_word(before, needle[0]) and not _cuts_a_word(after, needle[-1]):
            return True
        start = haystack.find(needle, start + 1)
    return False


def passage_in(passage: str, text: str) -> bool:
    """Whether `passage` is still in `text`, matched as `folded` reads both.

    Neither its length nor word edges are checked: they were checked when the passage was
    quoted, under the rule of that day, and this asks only whether the words are still
    there.
    """
    needle = folded(passage)
    return bool(needle) and needle in folded(text)
