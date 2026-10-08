"""One way to compare names and texts, and the character-pair measure built on it (uGraph step 1).

`fold` is NFC, runs of white space as one space, trimmed, casefolded. It is what
`memory.models.fold_name` and `story.quotes.folded` were, as two copies of one rule; both
now call this. `story/audit.py` reads a predicate or a class name with a stricter rule
(`_` and `-` are separators, and the capitalised spelling is shown to the author), so it
is a layer built on `fold`, not a replacement for it.

This module is pure.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["bigram_overlap", "bigrams", "fold"]

_NO_SPACE = re.compile(r"\s+")


def fold(text: str) -> str:
    """`text` in NFC, with runs of white space as one space, trimmed, and case-folded.

    NFC first, so a name stored decomposed ("José" as `e` and a combining accent) matches
    the composed form a model writes, and the reverse.
    """
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def bigrams(text: str) -> set[str]:
    """The character pairs of `text`, in NFC and casefolded, with white space removed."""
    squeezed = _NO_SPACE.sub("", unicodedata.normalize("NFC", text).casefold())
    return {squeezed[i : i + 2] for i in range(len(squeezed) - 1)}


def bigram_overlap(a: str, b: str) -> float:
    """How much two texts share, 0 to 1: the Jaccard index of their character pairs.

    The model-free measure behind `story.start.premise_similarity` and the entity
    resolver's near-match stage. Composed and decomposed spellings of one text are equal.
    """
    left, right = bigrams(a), bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
