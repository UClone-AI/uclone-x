"""Detect a retrieval query that names a term unique to the document it targets.

Lives in `tests/support/` rather than inside the test that uses it so the property it
enforces is itself mutation-checkable: a declaration whose target is the file the
declaration is written in cannot have a unique needle, because the docstring quotes the
line it names.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

#: Function words: present for grammar, not for meaning. Dropped because they carry no
#: retrieval signal, and a stopword that happens to fall in exactly one document would
#: read as leakage without being any.
STOPWORDS = frozenset(
    """the and for that with from are was were its not but who can one two than own how what
    why when which does into out before after per has have had all any our their there this
    these those you your they them been will would more most only every each some such because
    rather without within over under long much many new old now get got put take run runs go
    goes make made use used uses see seen say said time times way thing""".split()
)

_MIN_STEM_LENGTH = 2


def content_stems(text: str) -> set[str]:
    """Words of `text` worth matching on, crudely singularised."""
    return {
        word.rstrip("s")
        for word in re.findall(r"[a-z]+", text.lower())
        if len(word) > _MIN_STEM_LENGTH and word not in STOPWORDS
    }


def leaked_stems(query: str, target: set[str], document_frequency: Mapping[str, int]) -> list[str]:
    """Stems of `query` that occur in exactly one document of the corpus, and it is `target`.

    The property is about *distinctive* stems, not long ones. Scoping this to stems above
    some length is the weakening that let five English queries each share a document-unique
    term — `tokens`, `answer`, `code`, `merge`, `kept`, all six characters or fewer — with
    the document they were meant to retrieve.
    """
    return sorted(
        stem
        for stem in content_stems(query)
        if document_frequency.get(stem, 0) == 1 and stem in target
    )
