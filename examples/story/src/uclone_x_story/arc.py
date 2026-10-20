"""The rise of a story outline: what each chapter does, and where the twist is revealed.

`story_start` outlines a story one chapter per act of a structure template. Told only to
"raise the stakes, land the twist in a middle chapter", qwen3:8b revealed a fantasy twist
in the first beat and settled a drama's conflict in its turn chapter, then restated it in
the last (2026-09-29, one run per request). Checked after the fact and asked for the whole
outline again, it fixed one of four missing twists (#1991). So the shape is decided here,
by code, and the outline is written a chapter at a time inside it:

- `arc_for` gives every chapter a **function** (`FUNCTIONS`) from the structure alone: the
  first chapter sets the conflict going, the twist chapter reverses it, the climax decides
  it. The twist chapter is never before the middle: the first act at or after the middle
  that holds a turning beat (`TWIST_BEATS`: a midpoint, the kishotenketsu `ten`, the hero's
  ordeal), else the middle itself, and always before the last chapter. Two chapters in a
  row never have the same function.
- `Spine` is what the story turns on -- its conflict, twist, a clue to the twist, and how
  the conflict is decided -- held by code. `Spine.told` gives each chapter only its part:
  a chapter before the twist chapter gets the clue and never the twist; only the climax
  and after get the resolution.
- `chapter_problems` checks one chapter against its function as soon as it is written, so
  only that chapter is asked for again. The twist question is yes/no (`CHECK_MARKER`,
  temperature 0) and asks whether the chapter states the twist's core fact: asked instead
  whether "the reader learns this twist", qwen3:8b said no for most chapters that plainly
  told it, so a told twist was reported missing. Whether a chapter before the climax already settles the conflict is not
  asked as yes/no: asked "is this conflict settled", qwen3:8b said no for all 15 outlines
  of #1991, three of which a person read as settled. It is a choice between two named
  endings of the chapter (`_SETTLED_CHOICE`) instead, asked only from the twist chapter
  up to the climax: on those outlines the choice picked "settled" for 15 of 25 chapters a
  person read as open, most of them before the twist.

Nothing here writes a file; the questions go to the model `story_start` hands in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

__all__ = [
    "CHECK_MARKER",
    "FUNCTIONS",
    "TWIST_BEATS",
    "Arc",
    "Spine",
    "arc_for",
    "chapter_problems",
    "chapter_text",
    "problem_reasons",
]

#: The first line of every outline check question, so a scripted model can tell it apart.
CHECK_MARKER = "[outline check]"

#: What a chapter does for the story's rise, in the order a story goes through them.
FUNCTIONS = ("setup", "complication", "escalation", "reversal", "climax", "resolution")

#: Beat ids of the bundled structures that turn the story: its twist belongs there.
TWIST_BEATS = frozenset({"midpoint", "twist", "ordeal"})

#: What each function asks of a chapter, as the outline prompt says it.
_DOES: Mapping[str, str] = {
    "setup": "set the conflict going: who wants what, and what stands in the way; end on the "
    "first real problem",
    "complication": "make the problem worse with a new obstacle, a failed attempt or a cost",
    "escalation": "raise the stakes again, beyond the chapter before, with a new event",
    "reversal": "reveal the twist to the reader for the first time; it changes what the "
    "conflict is about, and things look worse, not better",
    "climax": "the protagonist, changed by the twist, faces the central conflict directly "
    "and decides it",
    "resolution": "show what the decision cost or gave, and the new normal, without "
    "retelling earlier events",
}


class _Model(Protocol):
    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str: ...


@dataclass(frozen=True)
class Arc:
    """Each chapter's functions (1-based chapter n is `functions[n - 1]`), and the turns."""

    functions: tuple[tuple[str, ...], ...]
    twist_chapter: int
    climax_chapter: int

    def does(self, number: int) -> str:
        """What chapter `number` must do, and must not, as the outline prompt says it."""
        parts = [_DOES[f] for f in self.functions[number - 1]]
        if number < self.twist_chapter:
            parts.append("the twist stays hidden: clues may be planted, but no one learns it")
        if number < self.climax_chapter:
            parts.append("the conflict stays open")
        return "; ".join(parts)

    def record(self) -> dict[str, Any]:
        return {
            "functions": [list(f) for f in self.functions],
            "twist_chapter": self.twist_chapter,
            "climax_chapter": self.climax_chapter,
        }


def arc_for(acts: Sequence[Sequence[str]]) -> Arc | None:
    """The arc of a structure whose acts hold these beat ids; `None` under three acts."""
    count = len(acts)
    if count < 3:
        return None
    middle = count // 2 + 1
    marked = next(
        (
            number
            for number, beats in enumerate(acts, start=1)
            if middle <= number < count and TWIST_BEATS & set(beats)
        ),
        None,
    )
    twist = marked or min(middle, count - 1)
    climax = count if count == twist + 1 else count - 1
    functions: list[tuple[str, ...]] = [("setup",)]
    for number in range(2, count + 1):
        if number < twist:
            before = functions[-1][-1]
            functions.append(("escalation",) if before == "complication" else ("complication",))
        elif number == twist:
            functions.append(("complication", "reversal") if number == 2 else ("reversal",))
        elif number < climax:
            functions.append(("escalation",))
        elif number == climax:
            functions.append(("climax", "resolution") if climax == count else ("climax",))
        else:
            functions.append(("resolution",))
    return Arc(functions=tuple(functions), twist_chapter=twist, climax_chapter=climax)


def chapter_text(chapter: Mapping[str, Any]) -> str:
    """One chapter of the model's outline reply as the check reads it."""
    lines = [str(chapter.get("title") or "")]
    if chapter.get("stakes"):
        lines.append(f"Stakes: {chapter['stakes']}")
    raw = chapter.get("scenes")
    for scene in cast(list[object], raw) if isinstance(raw, list) else []:
        if not isinstance(scene, dict):
            continue
        sc = cast(dict[str, Any], scene)
        beats = sc.get("beats")
        beat_text = (
            "; ".join(str(b) for b in cast(list[object], beats)) if isinstance(beats, list) else ""
        )
        lines.append(f"- {sc.get('title') or ''}: {sc.get('summary') or ''} {beat_text}".strip())
    return "\n".join(line for line in lines if line)


async def _yes(model: _Model, question: str) -> bool:
    reply = await model.complete(
        f"{CHECK_MARKER}\n{question}\nAnswer with one word, yes or no.",
        system="You check story outlines. Answer yes or no.",
        temperature=0.0,
        max_tokens=10,
    )
    return reply.strip().casefold().startswith(("yes", "예", "네"))


#: How a chapter before the climax is asked whether it already settles the conflict: a
#: choice between two endings, answered with a letter. B is settled.
_SETTLED_CHOICE = (
    "The central conflict of a story is: {conflict}\n\nOne chapter of its outline:\n"
    "{text}\n\nHow does this chapter end?\n"
    "A) the conflict is still unsolved, or worse than before\n"
    "B) the conflict is solved, forgiven, accepted or decided\n"
    "Answer with one letter, A or B."
)


async def _settled(model: _Model, conflict: str, text: str) -> bool:
    reply = await model.complete(
        f"{CHECK_MARKER}\n" + _SETTLED_CHOICE.format(conflict=conflict, text=text),
        system="You read story outlines. Answer with one letter.",
        temperature=0.0,
        max_tokens=5,
    )
    return reply.strip()[:1].upper() == "B"


@dataclass(frozen=True)
class Spine:
    """What the story turns on, decided before any chapter and held by code."""

    conflict: str
    twist: str
    #: A hint a chapter before the twist may plant, which does not state it.
    clue: str = ""
    #: How the climax decides the conflict.
    resolution: str = ""

    def told(self, arc: Arc, number: int) -> str:
        """What chapter `number` is told of the spine: the twist only from its chapter on."""
        lines = [f"Central conflict: {self.conflict}"]
        if number < arc.twist_chapter:
            if self.clue:
                lines.append(f"A clue this chapter may plant, without explaining it: {self.clue}")
            lines.append(
                f"The twist is kept from this chapter; the reader learns it in chapter "
                f"{arc.twist_chapter}."
            )
        elif self.twist:
            when = (
                "the reader learns it in THIS chapter"
                if number == arc.twist_chapter
                else f"revealed in chapter {arc.twist_chapter}"
            )
            lines.append(f"The twist ({when}): {self.twist}")
        if number >= arc.climax_chapter and self.resolution:
            when = "in THIS chapter" if number == arc.climax_chapter else "before this chapter"
            lines.append(f"How the conflict is decided ({when}): {self.resolution}")
        return "\n".join(lines)

    def record(self) -> dict[str, str]:
        return {"clue": self.clue, "resolution": self.resolution}


async def chapter_problems(
    model: _Model,
    arc: Arc,
    number: int,
    text: str,
    *,
    spine: Spine,
    previous: str | None = None,
) -> list[dict[str, Any]]:
    """Where chapter `number` (its outline as `chapter_text`) breaks the arc.

    Each problem is `{"kind", "chapter"}`. Kinds: ``twist_early`` (a chapter before the
    twist chapter reveals it), ``twist_missing`` (the twist chapter does not),
    ``resolved_early`` (a chapter from the twist up to the climax settles the conflict),
    ``repeats``
    (the chapter only repeats `previous`).
    """
    problems: list[dict[str, Any]] = []
    twist = spine.twist.strip()
    if twist and number <= arc.twist_chapter:
        shown = await _yes(
            model,
            f"The twist of a story is: {twist}\n\nOne chapter of its outline:\n{text}\n\n"
            "Does this chapter state the core fact of the twist, even in different words "
            "or as a character's discovery?",
        )
        if shown and number < arc.twist_chapter:
            problems.append({"kind": "twist_early", "chapter": number})
        elif not shown and number == arc.twist_chapter:
            problems.append({"kind": "twist_missing", "chapter": number})
    if spine.conflict.strip() and arc.twist_chapter <= number < arc.climax_chapter:
        if await _settled(model, spine.conflict, text):
            problems.append({"kind": "resolved_early", "chapter": number})
    if previous is not None:
        again = await _yes(
            model,
            f"Chapter A of a story outline:\n{previous}\n\nChapter B, the next "
            f"one:\n{text}\n\nDoes chapter B only repeat or restate what happens in "
            "chapter A, with no new event of its own?",
        )
        if again:
            problems.append({"kind": "repeats", "chapter": number})
    return problems


def problem_reasons(arc: Arc, problems: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each problem as the outline prompt is told it when the outline is asked for again."""
    reasons: list[str] = []
    for problem in problems:
        number = int(problem["chapter"])
        kind = problem["kind"]
        if kind == "twist_early":
            reasons.append(
                f"Chapter {number} already reveals the twist. The reader must not learn it "
                f"before chapter {arc.twist_chapter}: in chapter {number}, show only what "
                "the characters believe, and at most a clue."
            )
        elif kind == "twist_missing":
            reasons.append(f"Chapter {number} must reveal the twist, and it does not.")
        elif kind == "resolved_early":
            reasons.append(
                f"Chapter {number} already settles the central conflict. It must stay open "
                f"until chapter {arc.climax_chapter}; end chapter {number} worse off instead."
            )
        elif kind == "repeats":
            reasons.append(
                f"Chapter {number} repeats chapter {number - 1}. Give it a new event that "
                "raises the stakes."
            )
        elif kind == "no_twist":
            reasons.append("The reply named no twist.")
    return reasons
