# pyright: reportPrivateUsage=false
"""Stage 0 (Brief direction) and Stage 1 (Creative Premise generation with Muse cards and Engines)."""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from typing import Any, cast

from uclone_x.i18n.scripts import words_outside
from uclone_x_story.library import StoryError
from uclone_x_story.muse import MAX_SEED, draw_cards
from uclone_x_story.skill_data import MuseTable
from uclone_x_story.start_state import (
    _FALLBACK_TABLE,
    ENGINES,
    LANGUAGE_NAMES,
    PREMISE_COUNT,
    Brief,
    Premise,
    PremiseEmbedder,
    StartState,
    StoryModel,
    StoryStartParams,
    _ask_json,
    _brief_lines,
    _genre_turned_down,
    _pairs,
    _text,
    _texts,
    brief_from,
    genre_label,
    muse_genre,
    premise_similarity,
    story_scripts,
    turned_down_genres,
)


async def _direction(model: StoryModel, request: str) -> Brief:
    """Stage 0: only what the request says; code fills the rest (`brief_from`)."""
    prompt = (
        "Read the person's request for a story and write down ONLY what it says. Leave a "
        "field null (or an empty list) when the request does not say it; do not invent.\n"
        'Reply as JSON: {"title": a short working title in the request\'s language, '
        '"genre": ..., "tone": ..., "audience": ..., "length": ..., '
        '"must_have": [up to 4 plain strings of specific beings, places, or objects the '
        "request wants in the story; keep a noun phrase and its proper name or modifier "
        "together as one item rather than splitting them], "
        '"avoid": [plain strings of things the request does not want]}\n\n'
        f"Request:\n{request}"
    )
    stated = await _ask_json(model, prompt, temperature=0.0, max_tokens=600)
    return brief_from(stated, request)


_PREMISE_REPLY = (
    'Reply as JSON: {"title": a short title, "premise": the premise in two or three sentences}'
)


def _premise_prompt(state: StartState, engine: str, cards: str, fix: str = "") -> str:
    return (
        f"The person asked for:\n{state.request}\n\nDirection:\n{_brief_lines(state.brief)}"
        f"{_turned_down_lines(state)}"
        "\n\nIdea cards to spark from (adapt one or two naturally into the story's own "
        "world and everyday words; ignore any that break the genre or tone, and never "
        f"translate card terms stiffly):\n{cards or '- (none)'}\n\n"
        f"Write one premise for this story whose engine is {engine}.\n"
        "Keep every 'must include' element and the genre and tone, but do not retell the "
        "request or copy its phrasing: the person already knows it. Open with the concrete "
        "event that sets this story going, not with who the protagonist is. Always give the "
        "protagonist a proper name in the premise (never leave them as an unnamed role or "
        "occupation), invent and name the other person at the "
        "centre of it, say exactly what happened, what the protagonist wants now, and what "
        f"stands in the way.\n{_in_language(state)}{fix}\n{_PREMISE_REPLY}"
    )


def _in_language(state: StartState) -> str:
    """The line that asks for the story's text in its language, by name (#2301)."""
    return f"Write the title and the premise in {LANGUAGE_NAMES[state.brief.language]} only."


def _said(state: StartState) -> list[str]:
    """The person's own words in the start: the request and what they said at the premises."""
    return [state.request, *state.premise_feedback]


def _premise_words(title: str, text: str) -> str:
    return f"{title}\n{text}"


def _title_key(title: str) -> str:
    """A title as two titles are compared: letters and digits only, case folded."""
    return "".join(ch for ch in title.casefold() if ch.isalnum())


def _same_title(premises: Sequence[Premise]) -> int | None:
    """The index (in `_pairs` order) of the first pair of premises with one title, or `None`."""
    for index, (i, j) in enumerate(_pairs(len(premises))):
        key = _title_key(premises[i].title)
        if key and key == _title_key(premises[j].title):
            return index
    return None


def same_title_count(premises: Sequence[Premise]) -> int:
    """How many pairs of premises share one title."""
    return sum(
        1
        for i, j in _pairs(len(premises))
        if (key := _title_key(premises[i].title)) and key == _title_key(premises[j].title)
    )


def _turned_down_lines(state: StartState) -> str:
    """What the person said when they turned premises down, for the next draw."""
    if not state.premise_feedback:
        return ""
    said = "".join(f"- {words}\n" for words in state.premise_feedback)
    titles = "; ".join(state.premises_turned_down)
    return (
        f"\n\nThe person turned down the premises offered before ({titles}) and said:\n"
        f"{said}Follow what they said, and do not offer those premises again."
    )


async def _holds(model: StoryModel, element: str, text: str) -> bool:
    prompt = (
        f"Does this story premise contain this element: '{element}'? It counts when every "
        "part of the element is there in substance, in any wording. A person or thing without "
        "the trait or role the element gives it does not count. Answer with one word, yes "
        f"or no.\n\nPremise:\n{text}"
    )
    reply = await model.complete(
        prompt, system="You check stories. Answer yes or no.", temperature=0.0, max_tokens=10
    )
    return reply.strip().casefold().startswith(("yes", "예", "네"))


async def _missing(model: StoryModel, must_have: Sequence[str], text: str) -> list[str]:
    return [m for m in must_have if not await _holds(model, m, text)]


async def _drafted(model: StoryModel, prompt: str) -> tuple[str, str]:
    found = await _ask_json(model, prompt, temperature=0.9, max_tokens=500) or {}
    return _text(found.get("title")), _text(found.get("premise"))


async def _one_premise(
    model: StoryModel, state: StartState, engine: str, cards: str, fix: str = ""
) -> tuple[str, str, list[str], list[str]]:
    """A premise around `engine`: its title, text, what a rewrite was for, what is missing.

    A premise that drops a must-have is rewritten once, told every must-have and not only
    the dropped ones: told only the dropped one, qwen3:8b rewrote a premise around it and
    lost another. The rewrite is kept only when it misses fewer.
    """
    title, text = await _drafted(model, _premise_prompt(state, engine, cards, fix))
    must = state.brief.must_have
    if not text or not must:
        return title, text, [], []
    lacking = await _missing(model, must, text)
    if not lacking:
        return title, text, [], []
    listed = "".join(f"- {m}\n" for m in must)
    revise = (
        f"Story premise:\n{text}\n\nThe person asked the story to have all of these:\n"
        f"{listed}This premise leaves out: {'; '.join(lacking)}.\nRewrite the premise so it "
        "has every one of them, keeping its own events, people and conflict. Do not "
        f"restate the request.\n{_in_language(state)}{fix}\n{_PREMISE_REPLY}"
    )
    new_title, new_text = await _drafted(model, revise)
    if not new_text:
        return title, text, lacking, lacking
    still = await _missing(model, must, new_text)
    if len(still) < len(lacking):
        return new_title or title, new_text, lacking, still
    return title, text, lacking, lacking


async def _premises(
    model: StoryModel,
    state: StartState,
    tables: Mapping[str, MuseTable],
    embedder: PremiseEmbedder | None = None,
) -> StartState:
    """Stage 1: three premises, each around its own engine, with its own cards.

    A premise with a word outside the story's scripts (`story_scripts`) is drafted once
    more, told the words, and the draft with fewer such words is kept (#2301). The three
    are measured with their titles (`premise_similarity`); a pair with one title, or a
    pair too alike, gets its later premise drawn once more around an engine not used yet,
    told the titles taken.
    """
    notes = list(state.notes)
    scripts = story_scripts(state.brief.language, _said(state))
    table_key = muse_genre(state.brief.genre, sorted(tables))
    if table_key is None and tables:
        table_key = _FALLBACK_TABLE if _FALLBACK_TABLE in tables else sorted(tables)[0]
        notes.append(
            f"No idea table fits the genre '{state.brief.genre}', so the cards were drawn "
            f"from '{table_key}'."
        )
    engines = secrets.SystemRandom().sample(ENGINES, len(ENGINES))
    korean = state.brief.language == "ko"
    seen_titles: list[str] = []

    async def draft(number: int, engine: str, taken: Sequence[str] = ()) -> Premise:
        seed = secrets.randbelow(MAX_SEED + 1)
        card = draw_cards(tables[table_key], seed, 1)[0] if table_key else {}
        cards = "\n".join(f"- {slot}: {entry}" for slot, entry in card.items())
        active_taken = list(taken) if taken else list(seen_titles)
        fix = (
            f"\nThe other premises are titled {'; '.join(active_taken)}: give this one another title."
            if active_taken
            else ""
        )
        title, text, rewritten_for, missing = await _one_premise(model, state, engine, cards, fix)
        foreign = words_outside(_premise_words(title, text), scripts)
        if text and foreign:
            again = (
                f"{fix}\nA draft of this premise had words outside "
                f"{LANGUAGE_NAMES[state.brief.language]}: {', '.join(foreign)}. Write every "
                "word in it."
            )
            redone = await _one_premise(model, state, engine, cards, again)
            still = words_outside(_premise_words(redone[0], redone[1]), scripts)
            if redone[1] and len(still) < len(foreign):
                title, text, rewritten_for, missing = redone
                foreign = still
            if foreign:
                notes.append(
                    f"Premise {number} still has words outside the story's language: "
                    f"{', '.join(foreign)}."
                )
        if not text:
            notes.append(f"Premise {number} came back without text.")
        if not taken and title and not words_outside(title, scripts) and title not in seen_titles:
            seen_titles.append(title)
        return Premise(
            number=number,
            title=title or (f"발상 {number}" if korean else f"Premise {number}"),
            premise=text or ("(내용 없음)" if korean else "(empty)"),
            muse_genre=table_key,
            seed=seed if table_key else None,
            card=card,
            engine=engine,
            rewritten_for=rewritten_for,
            missing=missing,
        )

    premises = [await draft(n, engines[n - 1]) for n in range(1, PREMISE_COUNT + 1)]
    similarity = await premise_similarity(_premise_texts(premises), embedder)
    same = _same_title(premises)
    if (same is not None or not similarity.distinct) and len(engines) > PREMISE_COUNT:
        # Once: the later premise of the pair with one title, else of the closest pair,
        # again around an engine not used yet.
        worst = same if same is not None else similarity.pairs.index(similarity.closest)
        number = _pairs(len(premises))[worst][1] + 1
        taken = [p.title for p in premises if p.number != number]
        again = await draft(number, engines[PREMISE_COUNT], taken)
        tried = [again if p.number == number else p for p in premises]
        retried = await premise_similarity(_premise_texts(tried), embedder)
        before = (same_title_count(premises), similarity.closest)
        if (same_title_count(tried), retried.closest) < before:
            premises, similarity = tried, retried
    if _same_title(premises) is not None:
        notes.append("Two premises have the same title.")
    if not similarity.distinct:
        notes.append(
            f"Two premises read alike ({similarity.measure} {similarity.closest:.2f}, over "
            f"{similarity.limit})."
        )
    return state.model_copy(
        update={
            "premises": premises,
            "premise_similarity": similarity.record(),
            "notes": notes,
        }
    )


def _premise_texts(premises: Sequence[Premise]) -> list[str]:
    """Each premise as it is measured against the others: its title, then its text (#2301:
    two premises titled "은시계의 유산" read as distinct on their prose alone)."""
    return [_premise_words(p.title, p.premise) for p in premises]


def _genre_choice(
    state: StartState,
    params: StoryStartParams,
    wanted: Mapping[str, Sequence[str]],
) -> str:
    """The genre the person's answer to the proposal takes, without a model call.

    A number picks one of the genres offered; `go_on=True` takes the recommended one;
    `genre` names another, skipping any genre the brief or `avoid` turned down (#2301).
    """
    options = state.genre_options
    if params.choice is not None and params.choice <= len(options):
        return options[params.choice - 1]
    if params.genre and params.genre.strip():
        genre = genre_label(params.genre.strip(), state.brief.language)
        refused = turned_down_genres([*state.brief.avoid, *wanted.get("avoid", ())])
        if not _genre_turned_down(genre, refused):
            return genre
    if params.go_on:
        return options[0]
    raise StoryError(
        "The person has not taken or named a genre yet: show them the genres proposed and "
        "call story_start 'continue' with 'choice', 'genre', or 'go_on' when they decide. "
        "Nothing was changed."
    )


def _wanted_from(params: StoryStartParams) -> dict[str, list[str]]:
    """What the structured tool parameters ask to keep in the story or keep out of it."""
    return {"must_have": _texts(params.must_have), "avoid": _texts(params.avoid)}


def _with_wanted(brief: Brief, wanted: Mapping[str, Any]) -> Brief:
    """`brief` with what the person asked for since the request: their must-haves and
    what to avoid, each kept once after the request's own, and counted as stated."""
    must = [*brief.must_have, *cast("list[str]", wanted.get("must_have") or [])]
    avoid = [*brief.avoid, *cast("list[str]", wanted.get("avoid") or [])]
    stated = [k for k in ("must_have", "avoid") if wanted.get(k)]
    return brief.model_copy(
        update={
            "must_have": list(dict.fromkeys(must)),
            "avoid": list(dict.fromkeys(avoid)),
            "from_request": list(dict.fromkeys([*brief.from_request, *stated])),
        }
    )


def _without_avoided_cast(
    cast_lines: Sequence[Mapping[str, Any]], avoid: Sequence[str]
) -> list[dict[str, Any]]:
    """`cast_lines` without any character or place whose name matches an avoided term,
    unless dropping it would leave the story with no character at all."""
    terms = [a.strip().casefold() for a in avoid if len(a.strip()) >= 2]
    if not terms:
        return [dict(c) for c in cast_lines]
    kept = [
        dict(c)
        for c in cast_lines
        if not (
            (name := str(c.get("name", "")).strip().casefold())
            and len(name) >= 2
            and any(name in t or t in name for t in terms)
        )
    ]
    if any(c.get("kind") == "characters" for c in kept):
        return kept
    return [dict(c) for c in cast_lines]


def _feedback_with_brief(feedback: str, brief: Brief) -> str:
    """`feedback` with `brief.must_have` and `brief.avoid` appended so revisions whose
    specific items were passed in `must_have`/`avoid` still receive them explicitly."""
    lines = [feedback] if feedback else []
    if brief.must_have:
        lines.append(f"Must include: {', '.join(brief.must_have)}")
    if brief.avoid:
        lines.append(f"Must NOT include (remove any mention of these): {', '.join(brief.avoid)}")
    return "\n".join(lines)


__all__ = [
    "_PREMISE_REPLY",
    "_direction",
    "_drafted",
    "_feedback_with_brief",
    "_genre_choice",
    "_holds",
    "_in_language",
    "_missing",
    "_one_premise",
    "_premise_prompt",
    "_premise_texts",
    "_premise_words",
    "_premises",
    "_said",
    "_same_title",
    "_title_key",
    "_turned_down_lines",
    "_wanted_from",
    "_with_wanted",
    "_without_avoided_cast",
    "same_title_count",
]
