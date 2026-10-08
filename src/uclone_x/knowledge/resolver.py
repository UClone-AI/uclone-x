"""Entity resolution, stages 1-3 (the uGraph design §3.4): normalise, exact, near.

Stage 4 (a model asks whether a near match is the same thing) and the policy that decides
what to do with it are later steps. Nothing here guesses: an exact name two entities share
is reported as ambiguous, and a near match is only ever a candidate.

This module is pure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from uclone_x.knowledge.fold import bigram_overlap, fold
from uclone_x.knowledge.models import Entity, EntityRef, mint_entity_ref

__all__ = [
    "CONTAINMENT_MIN_RATIO",
    "JACCARD_THRESHOLD",
    "MIN_NEAR_CHARACTERS",
    "EntityResolver",
    "name_key",
]

#: Near-match threshold on `bigram_overlap`. Chosen from the measure's own tests (the same
#: text is 1.0, unrelated texts are under 0.1) and the shortest case it must catch:
#: `지현` against `김지현` is 1 shared pair of 2 in the union, exactly 0.5.
JACCARD_THRESHOLD = 0.5
#: A name inside another counts when it is at least this share of the longer one's
#: characters (`지현` in `김지현` is 2/3; `an` in `daniel` is 1/3 and does not count), or
#: when it is a whole run of the longer name's words (`vane` in `lord vane`).
CONTAINMENT_MIN_RATIO = 0.5
#: A name with fewer characters than this is never a near match: one letter is in everything.
MIN_NEAR_CHARACTERS = 2

_PARENS = re.compile(r"[(（][^()（）]*[)）]")
_QUOTES = re.compile(r"[\"'“”‘’「」『』«»`]")
_HONORIFIC = re.compile(r"^(?P<name>.+?)\s+(?:씨|님|군|양)$")


def name_key(name: str) -> str:
    """`name` as names are compared: folded, without a parenthetical note or quotation marks."""
    return fold(_QUOTES.sub("", _PARENS.sub("", name)))


def _contains(short: str, long: str) -> bool:
    if len(short) < MIN_NEAR_CHARACTERS or short == long or short not in long:
        return False
    words, wanted = long.split(), short.split()
    if any(words[i : i + len(wanted)] == wanted for i in range(len(words) - len(wanted) + 1)):
        return True
    return len(short.replace(" ", "")) / len(long.replace(" ", "")) >= CONTAINMENT_MIN_RATIO


class EntityResolver:
    """Resolve a name against one scope's entities."""

    def __init__(self, scope: str, entities: Iterable[Entity]) -> None:
        self.scope = scope
        self._entities = [e for e in entities if e.scope == scope]
        self._index: dict[str, list[str]] = {}
        for entity in self._entities:
            for key in {name_key(n) for n in (entity.id, entity.name, *entity.aliases)}:
                if key:
                    self._index.setdefault(key, []).append(entity.id)

    def resolve(self, name: str) -> EntityRef:
        key = name_key(name)  # stage 1
        if not key:
            raise ValueError("a name is not empty")
        keys = [key]
        if (honorific := _HONORIFIC.match(key)) is not None:
            keys.append(honorific.group("name"))  # an alias candidate only; the full name first
        for stage_key in keys:  # stage 2
            if ids := self._index.get(stage_key):
                if len(ids) == 1:
                    return mint_entity_ref(kind="existing", name=name, entity_id=ids[0], stage=2)
                return mint_entity_ref(
                    kind="ambiguous", name=name, ambiguous_ids=tuple(ids), stage=2
                )
        near = self._near(keys[-1])  # stage 3
        return mint_entity_ref(kind="new", name=name, candidates=near, stage=3 if near else 0)

    def _near(self, key: str) -> tuple[tuple[str, float], ...]:
        if len(key.replace(" ", "")) < MIN_NEAR_CHARACTERS:
            return ()
        best: dict[str, float] = {}
        for entity in self._entities:
            for other in {name_key(n) for n in (entity.id, entity.name, *entity.aliases)}:
                score = bigram_overlap(key, other)
                if _contains(key, other) or _contains(other, key):
                    score = max(score, JACCARD_THRESHOLD)
                elif score < JACCARD_THRESHOLD:
                    continue
                best[entity.id] = max(best.get(entity.id, 0.0), score)
        return tuple(sorted(best.items(), key=lambda item: (-item[1], item[0])))
