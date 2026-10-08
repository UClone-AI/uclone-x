"""`story_start`: a person's request made into a story, one fixed stage after another.

The Writer started a story by improvising a chain of tool calls -- `story_library create`,
`muse_spark`, `story_outline init`, `story_codex create`, `story_manuscript write` -- and
a small model skipped steps, left the outline as the template's placeholders, and wrote
the first scene before anyone chose what the story was about. Here the order is code:

0. direction -- genre, tone, audience, length, what must be in it and what must not,
   read from the request; only what the request leaves open is filled in, and the
   brief says which fields the request set. A genre the request leaves open is not
   filled silently: code draws one from the muse tables at random, with alternatives,
   and the flow stops for the person to take it or name another;
1. premises -- three, each from its own model call around its own story engine (a
   secret, a rival, a hard choice...) drawn by code, with its own muse cards; each is
   checked for the request's must-haves and rewritten once when it drops one, and the
   three are measured against each other (`premise_similarity`); the flow stops for the
   person to choose one by number;
2. cast -- the characters and places of the chosen premise, proposed as new codex entries
   (the `story_codex create` path), which the person approves as any other proposal; each
   character starts with the status the premise gives them (alive, missing, dead, unknown);
3. outline -- chapters and scenes with real beats, from the premise and the cast; each
   chapter has as many scenes as its weight in the arc gives it (`scenes_for`: two for a
   chapter that sets up or winds down, four for the twist or the climax, three otherwise).
   Agreeing settles the plot and proposes its key events -- happenings, and the physical
   objects the outline names, with abstract nouns ("기억", "진실") dropped by code and
   counted (`abstract_item`) -- for the codex;
4. chapter -- once the proposed cast is decided or the person says to go on without
   deciding it, the first chapter's plan is shown (4b), and on "네" its scenes are written
   through `StoryWork.write_scene`, then checked against the codex. Each scene saved (and each scene rewritten after the check)
   is read for what it adds to the codex, as `story_manuscript write` reads one
   (`uclone_x.story.enrich.grow_scene`): the additions wait as proposals for the person,
   and a reading that fails leaves the chapter as written;
4b. chapter plan -- before each chapter is written, the first included, its scenes and
   beats are planned from the locked outline, what the arc says that chapter must do, the
   cast, and how the chapter before it ended; the flow stops to show the plan. Words of
   feedback revise it (one model call) and it is shown again; "네" or "써 줘"
   (`writes_chapter`) writes that chapter the same way and plans the next, until the
   outline's last chapter is written. A new character the accepted plan introduces with
   a stated relation to one of the cast is proposed as a new entry, with the kinship kept
   on both sides as the cast's is (a progression on an approved relative).

The flow stops after stages 1, 2 and 3 for the person, unless the request asks for the
story at once ("바로 써줘"), which runs every stage in the same order with the first
premise and writes the chapter over the cast as proposed -- and says so to the person.
Even then one call writes at most `CHAPTERS_PER_CALL` chapters, and the reply says how to
go on.
At every stop a reply that only agrees ("네", "좋아요", "ok": `says_yes`) takes what the
flow recommends -- the genre proposed, the recommended premise, the cast as proposed --
so "yes" alone carries a bare request through each chapter's plan to a written chapter; a number or any other words
are read as what they say. Every call returns the stage it reached and what it needs next. What the flow
knows is kept in `start.yaml` in the story's folder, not in memory, so a later call --
in the same conversation or after a restart -- continues where the last one stopped.

Each stage asks the model through `StoryModel`, a provider-neutral seam: the tool's
default goes through `invoke_auxiliary_model` on the calling clone's own connector and
budget (P5), and a test passes a fake.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uclone_x.knowledge.fold import bigram_overlap
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole
from uclone_x.story import OPEN_STORY_KEY
from uclone_x.story.arc import (
    Arc,
    Spine,
    arc_for,
    chapter_problems,
    chapter_text,
    problem_reasons,
)
from uclone_x.story.context import CodexIndex, CodexItem, scene_context
from uclone_x.story.continuity import (
    INVERSE_ROLE,
    ExtractedFact,
    Finding,
    came_back,
    codex_with,
    continuity_prompt,
    family_role,
    findings_from,
    gender_of,
    parse_extraction,
    salvage_extraction,
)
from uclone_x.story.enrich import grow_scene, growth_line
from uclone_x.story.library import StoryError, StoryLibrary
from uclone_x.story.muse import MAX_SEED, draw_cards, normalize_genre
from uclone_x.story.names import id_for_name
from uclone_x.story.plan_check import absent_cast, asks_model, plan_findings, plan_prompt
from uclone_x.story.proposals import new_entry_clash
from uclone_x.story.prose import prose_refusal
from uclone_x.story.schemas import Chapter, Outline, Proposal, Scene, StoryFileError
from uclone_x.story.skill_data import (
    BUNDLED_DATA_ROOT,
    MuseTable,
    SkillDataError,
    StructureTemplate,
    data_roots,
    load_muse_tables_sourced,
    load_structure_templates_sourced,
)
from uclone_x.story.tools import NewEntryParams, new_entry_draft
from uclone_x.story.work import StoryWork
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import REPLY_NOTE_KEY, ToolContext

logger = logging.getLogger(__name__)

__all__ = [
    "CHAPTER_PROMPT_MARK",
    "KG_MARK",
    "MAX_CAST_TRIES",
    "MAX_PREMISE_BIGRAM",
    "MAX_PREMISE_COSINE",
    "REVISE_MARK",
    "ABSTRACT_ITEM_WORDS",
    "SPINE_MARK",
    "START_FILE",
    "Brief",
    "PremiseEmbedder",
    "Similarity",
    "StartState",
    "abstract_item",
    "asks_rewrite",
    "StoryModel",
    "StoryStartParams",
    "StoryStartTool",
    "bigram_overlap",
    "brief_from",
    "cosine",
    "json_object",
    "premise_similarity",
    "says_go_on",
    "names_genre",
    "says_yes",
    "settles_plot",
    "scenes_for",
    "structure_for",
    "wants_it_now",
]

#: The flow's own file in the story's folder: where it stopped and what it has made.
START_FILE = "start.yaml"

Stage = Literal[
    "choose_genre", "choose_premise", "review_cast", "review_outline", "review_chapter", "done"
]

#: What the brief fills a field with when the request leaves it open. Code, not the model,
#: decides this: a field the request set is never replaced (#1723).
DEFAULTS: Mapping[str, str] = {
    "genre": "판타지",
    "tone": "진지하면서도 따뜻한",
    "audience": "성인 일반 독자",
    "length": "단편, 4장 안팎",
}
_BRIEF_FIELDS = ("genre", "tone", "audience", "length")

#: A genre as a person writes it, to the bundled muse table it draws from.
_GENRE_WORDS: tuple[tuple[str, str], ...] = (
    ("판타지", "fantasy"),
    ("fantasy", "fantasy"),
    ("호러", "horror"),
    ("공포", "horror"),
    ("horror", "horror"),
    ("미스터리", "mystery"),
    ("추리", "mystery"),
    ("mystery", "mystery"),
    ("로맨스", "romance"),
    ("연애", "romance"),
    ("romance", "romance"),
    ("sf", "science-fiction"),
    ("과학", "science-fiction"),
    ("우주", "science-fiction"),
    ("science", "science-fiction"),
)
_FALLBACK_TABLE = "fantasy"
#: The bundled muse tables a genre is proposed from, as the person reads each one. When
#: the request names no genre, code draws the proposal from these at random rather than
#: fixing `DEFAULTS["genre"]`, and the person takes it or names another (§1.1 principle 1).
GENRE_LABELS: Mapping[str, str] = {
    "fantasy": "판타지",
    "horror": "호러",
    "mystery": "미스터리",
    "romance": "로맨스",
    "science-fiction": "SF",
}
#: How many genres are proposed: the recommended one first, then alternatives.
GENRE_OPTIONS = 3

#: What drives each premise, three drawn by code without repeats. Three angles on one
#: request (a want, a place, an event) gave qwen3:8b the same story three times -- mean
#: premise cosine up to 0.99 -- because each call restated the request; a different engine
#: per call, and a prompt that forbids restating the request, brought it well under
#: `MAX_PREMISE_COSINE` (the eval ledger, 2026-09-29). The engines are genre-neutral,
#: so a realistic drama is not given a ghost.
ENGINES: tuple[str, ...] = (
    "a secret someone has kept for years, and who is hurt when it comes out",
    "a rival who wants the very thing the protagonist wants",
    "a choice the protagonist must make between two people, or two duties",
    "a mistake from the protagonist's past that comes back now",
    "a search that takes the protagonist far from where they began",
    "a stranger who arrives and upsets the balance of everyone's lives",
    "a promise that can only be kept at a real cost",
)
PREMISE_COUNT = 3

#: Two premises closer than this, by embedding cosine, read as one idea told twice. The
#: eval holds the premises to the same bar (`evals/suites/writer_story.py`).
MAX_PREMISE_COSINE = 0.92
#: The same bar for the character-pair fallback, used when no embedder is configured. On
#: qwen3:8b premises measured both ways, every pair above `MAX_PREMISE_COSINE` shared at
#: least 0.26 of its character pairs, and no premise from the engines more than 0.17.
MAX_PREMISE_BIGRAM = 0.25

_QUICK = re.compile(
    r"바로\s*(?:써|작성|시작)|알아서\s*(?:써|해)|그냥\s*써|묻지\s*말고|"
    r"\bjust write\b|\bwrite it now\b|\bdon'?t ask\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<!\d)([1-3])(?!\d)")
#: Words that tell the flow to write the chapter over the cast as proposed, without
#: waiting for the person to approve or reject each entry in the story's view.
_GO_ON = re.compile(
    r"그대로\s*(?:진행|써|쓰)|제안(?:한|된)?\s*대로|승인(?:하지\s*않|\s*안\s*하|\s*없이)|"
    r"그냥\s*(?:진행|계속)|확인\s*(?:없이|안\s*해도)|"
    r"\bgo ahead\b|\bproceed\b|\bas proposed\b|\bwithout approv",
    re.IGNORECASE,
)
#: A reply made only of these words agrees: it takes the option the flow recommends at the
#: stop it answers (§1.1 principle 2). A reply with any other word in it is a request of
#: its own and is read as one; so is a number.
_YES_TOKEN = re.compile(
    r"네|넵|넹|예|응|웅|어|ㅇㅇ|ㅇㅋ|그래|그래요|그러자|그럼|그럼요|좋아|좋아요|좋네요|좋습니다|"
    r"좋군요|오케이|알겠어|알겠어요|알겠습니다|진행|진행해|진행해요|진행해줘|진행하자|"
    r"진행합시다|진행하세요|계속|계속해|계속해요|계속해줘|그렇게|그걸로|그걸로요|추천대로|"
    r"추천대로요|해|해요|해줘|해주세요|하자|합시다|주세요|줘|yes|yeah|yep|yup|ok|okay|sure|"
    r"go|y"
)
#: Words at the outline stop that settle the plot as it is, beside a bare yes and "go on"
#: words: "확정해 주세요", "이대로", "첫 장 써 주세요". Any other words there are feedback
#: on the outline, which revises it (§4.3).
_LOCK = re.compile(
    r"확정|이대로|이걸로|(?:첫|다음)\s*(?:장|챕터)\S*\s*(?:을|를)?\s*(?:써|쓰|작성|시작)|"
    r"\block (?:it|the plot)\b|\bwrite the (?:first )?chapter\b|\blooks good\b",
    re.IGNORECASE,
)
_YES_SPLIT = re.compile(r"[\s,.!?~…'\"]+")
#: Words at a chapter's stop that ask for the chapter to be written as planned, beside a
#: bare yes, "go on" words and the words that settle the plot: "써 줘", "이어서 써",
#: "다음 장 써 주세요". Any other words there are feedback on the chapter's plan.
_WRITE_ON = re.compile(
    r"^\s*(?:(?:그럼|이제|네|좋아요?)\s*,?\s*)?(?:써|쓰자|써\s*줘|써\s*주세요|작성해\s*(?:줘|주세요)?)"
    r"\s*[.!~]*\s*$|(?:이어|계속|이어서)\s*(?:써|쓰|작성)|"
    r"(?:다음|\d+)\s*(?:장|챕터)\S*\s*(?:을|를)?\s*(?:써|쓰|작성|시작|진행)|"
    r"\bwrite (?:it|on|the next)\b|\bcontinue\b|\bkeep going\b",
    re.IGNORECASE,
)
#: Words that ask for a written scene to be written again, after the check found a line
#: that disagrees with the codex.
_REWRITE = re.compile(
    r"다시\s*(?:써|쓰|작성)|고쳐\s*(?:써|쓰|줘|주)|\brewrite\b|\bwrite (?:it|them) again\b",
    re.IGNORECASE,
)
#: The statuses a character may start the story with, and the words that give each.
STATUSES = ("alive", "missing", "dead", "unknown")
_STATUS_WORDS: tuple[tuple[str, str], ...] = (
    ("행방불명", "missing"),  # before "불명", which it contains
    ("unknown", "unknown"),
    ("불명", "unknown"),
    ("알 수 없", "unknown"),
    ("missing", "missing"),
    ("실종", "missing"),
    ("dead", "dead"),
    ("사망", "dead"),
    ("죽", "dead"),
    ("alive", "alive"),
    ("생존", "alive"),
    ("살아", "alive"),
)

#: A structure the person names, in the words they may use, to the bundled template's id.
#: A skill's own template is also found by its id or title (`structure_for`).
_NAMED_STRUCTURES: tuple[tuple[str, str], ...] = (
    (r"3\s*막|삼\s*막|three[\s-]*act", "three-act"),
    (r"기승전결|kish[oō]tenketsu", "kishotenketsu"),
    (r"영웅의\s*(?:여정|여행)|hero'?s\s*journey", "heros-journey"),
    (r"세이브\s*더\s*캣|save\s*the\s*cat", "save-the-cat"),
)
#: The structure a genre is given when the request names none (#1723: a skill's default
#: fills only what the person left open). A mystery or a horror story is built on turns
#: and a reveal; an adventure on a journey out and back; an everyday drama on a quiet
#: turn rather than a fight.
_GENRE_STRUCTURES: tuple[tuple[str, str], ...] = (
    ("미스터리", "three-act"),
    ("추리", "three-act"),
    ("스릴러", "three-act"),
    ("호러", "three-act"),
    ("공포", "three-act"),
    ("mystery", "three-act"),
    ("thriller", "three-act"),
    ("horror", "three-act"),
    ("판타지", "heros-journey"),
    ("모험", "heros-journey"),
    ("fantasy", "heros-journey"),
    ("adventure", "heros-journey"),
    ("sf", "heros-journey"),
    ("드라마", "kishotenketsu"),
    ("일상", "kishotenketsu"),
    ("가족", "kishotenketsu"),
    ("성장", "kishotenketsu"),
    ("로맨스", "kishotenketsu"),
    ("연애", "kishotenketsu"),
    ("drama", "kishotenketsu"),
    ("romance", "kishotenketsu"),
)
DEFAULT_STRUCTURE = "three-act"

MAX_CHARACTERS = 5
MAX_PLACES = 3
#: Tries at the cast: the first, then retries told why the last reply could not be used.
MAX_CAST_TRIES = 3
MAX_CHAPTERS = 6
#: A chapter's scenes: how many its weight in the arc gives it (`scenes_for`), from
#: `MIN_SCENES_PER_CHAPTER` for a chapter that sets up or winds down to
#: `MAX_SCENES_PER_CHAPTER` for the twist or the climax.
MIN_SCENES_PER_CHAPTER = 2
SCENES_PER_CHAPTER = 3
MAX_SCENES_PER_CHAPTER = 4
#: How many of a chapter's scenes stage 4 writes in one call.
MAX_SCENES_WRITTEN = MAX_SCENES_PER_CHAPTER
#: The functions (`uclone_x.story.arc.FUNCTIONS`) that make a chapter heavy or light.
_HEAVY = frozenset({"reversal", "climax"})
_LIGHT = frozenset({"setup", "resolution"})
#: How many chapters one call writes, even when the story was asked for at once: each
#: chapter is several model calls, so a call that wrote every chapter could run for as
#: long as the outline is. The next call writes the next one.
CHAPTERS_PER_CALL = 1
# Drafts per scene: the first, then rewrites told what the prose check refused.
MAX_DRAFTS = 3


# -- the model seam ----------------------------------------------------------------------


class StoryModel(Protocol):
    """The one thing the flow needs from a model: text for a prompt (P5)."""

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str: ...


class _Auxiliary(Protocol):
    async def invoke_auxiliary_model(self, request: LLMRequest) -> Any: ...


class AuxiliaryStoryModel:
    """`StoryModel` over the calling clone's own connector and budget."""

    def __init__(self, seat: _Auxiliary) -> None:
        self._seat = seat

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str:
        request = LLMRequest(
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content=system),
                ChatMessage(role=MessageRole.USER, content=prompt),
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=False,
            auto_compact=False,
        )
        response = await self._seat.invoke_auxiliary_model(request)
        return str(getattr(response, "content", None) or "")


def _default_model(context: ToolContext) -> StoryModel:
    seat = context.agent_delegate
    if seat is None or not callable(getattr(seat, "invoke_auxiliary_model", None)):
        raise StoryError(
            "This clone has no model to draft the story with here, so nothing was started."
        )
    return AuxiliaryStoryModel(cast(_Auxiliary, seat))


ModelFactory = Callable[[ToolContext], StoryModel]


class PremiseEmbedder(Protocol):
    """What the premise check needs from an embedder: vectors for texts (P5)."""

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def _default_embedder(context: ToolContext) -> PremiseEmbedder | None:
    """The calling clone's host embedder, the one its tool binder ranks with; else `None`."""
    found = getattr(context.agent_delegate, "embedder", None)
    return cast(PremiseEmbedder, found) if callable(getattr(found, "embed", None)) else None


EmbedderFactory = Callable[[ToolContext], PremiseEmbedder | None]


# -- pure helpers ------------------------------------------------------------------------


def wants_it_now(request: str) -> bool:
    """Whether the request asks for the story without stopping ("바로 써줘")."""
    return bool(_QUICK.search(request))


def says_go_on(words: str) -> bool:
    """Whether the person says to write on over the cast as proposed ("그대로 진행해")."""
    return bool(_GO_ON.search(words)) or wants_it_now(words)


def says_yes(words: str) -> bool:
    """Whether the reply only agrees ("네", "좋아요", "ok"): it takes the recommended option."""
    tokens = [t for t in _YES_SPLIT.split(words.strip().casefold()) if t]
    return bool(tokens) and all(_YES_TOKEN.fullmatch(t) for t in tokens)


def settles_plot(words: str) -> bool:
    """Whether the person's words at the outline stop settle the plot rather than change it."""
    return bool(_LOCK.search(words)) or says_yes(words) or says_go_on(words)


def genre_options(tables: Sequence[str], rng: random.Random) -> list[str]:
    """Genres to propose, drawn at random: the recommended one first, then alternatives.

    Drawn from the muse tables loaded that have a label (`GENRE_LABELS`); from every
    labelled table when none of those loaded.
    """
    keys = sorted(k for k in tables if k in GENRE_LABELS) or sorted(GENRE_LABELS)
    return [GENRE_LABELS[k] for k in rng.sample(keys, min(GENRE_OPTIONS, len(keys)))]


def named_genre(words: str) -> str | None:
    """A bundled genre the person's words name ("로맨스로 해 줘"), as its label."""
    lowered = words.casefold()
    for word, table in _GENRE_WORDS:
        if word in lowered:
            return GENRE_LABELS.get(table)
    return None


def names_genre(request: str, genre: str) -> bool:
    """Whether the request's own words name a genre: `genre` itself, or one code knows."""
    lowered = request.casefold()
    words = [genre.strip().casefold()] if genre.strip() else []
    words += [word for word, _ in (*_GENRE_WORDS, *_GENRE_STRUCTURES)]
    # An English word stands alone ("sf", not the "sf" in "transfer").
    return any(
        re.search(rf"\b{re.escape(word)}\b", lowered) if word.isascii() else word in lowered
        for word in words
    )


def writes_chapter(words: str) -> bool:
    """Whether the person's words at a chapter's stop ask for it to be written as planned."""
    return bool(_WRITE_ON.search(words)) or settles_plot(words)


def asks_rewrite(words: str) -> bool:
    """Whether the person asks for the flagged scenes to be written again ("다시 써 줘")."""
    return bool(_REWRITE.search(words))


def status_of(value: object) -> str:
    """A character's starting status from the model's word for it; `unknown` for none."""
    word = _text(value).casefold()
    for found, status in _STATUS_WORDS:
        if found in word:
            return status
    return "unknown"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


# A stray quotation mark opening an object's key: `{ " "name": ...` or `{ ""name": ...`.
# qwen3:8b wrote it in 5 of 8 cast replies for a premise that quotes a name ('용의 눈',
# 2026-09-29), and the whole reply -- its characters too -- was lost to it.
_STRAY_QUOTE = re.compile(r'([{,]\s*)"\s*(?="[^"\n]+"\s*:)')


def json_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's reply, fenced or not; `None` when there is none.

    A reply that does not parse is read once more with stray quotation marks before its
    keys taken out (`_STRAY_QUOTE`); nothing else is repaired.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        body = candidate[start : end + 1]
        for attempt in (body, _STRAY_QUOTE.sub(r"\1", body)):
            try:
                value: object = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return {str(k): v for k, v in cast(dict[object, Any], value).items()}
            break
    return None


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _texts(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [t for t in (_text(v) for v in cast(list[object], value)) if t]
    return []


class Brief(BaseModel):
    """Stage 0: the story's direction, and which of it the request itself set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    genre: str
    tone: str
    audience: str
    length: str
    must_have: list[str] = Field(default_factory=list[str])
    avoid: list[str] = Field(default_factory=list[str])
    #: The fields the request set; the rest are `DEFAULTS`.
    from_request: list[str] = Field(default_factory=list[str])


def brief_from(stated: Mapping[str, Any] | None, request: str) -> Brief:
    """The brief: what the request stated, and `DEFAULTS` for only what it left open."""
    data = stated or {}
    fields: dict[str, str] = {}
    from_request: list[str] = []
    for key in _BRIEF_FIELDS:
        value = _text(data.get(key))
        if key == "genre" and value and not names_genre(request, value):
            # qwen3:8b read "소설 하나 써줘" as genre "일반" and called it stated
            # (2026-09-29), so the genre was never proposed: a genre counts as the
            # request's only when the request's own words name it.
            value = ""
        if value:
            fields[key] = value
            from_request.append(key)
        else:
            fields[key] = DEFAULTS[key]
    must_have = _texts(data.get("must_have"))
    avoid = _texts(data.get("avoid"))
    if must_have:
        from_request.append("must_have")
    if avoid:
        from_request.append("avoid")
    title = _text(data.get("title")) or _title_of(request)
    return Brief(
        title=title[:60],
        must_have=must_have,
        avoid=avoid,
        from_request=from_request,
        **fields,
    )


def _title_of(request: str) -> str:
    words = request.strip().split()
    return " ".join(words[:6]) or "새 이야기"


def muse_genre(genre: str, tables: Sequence[str]) -> str | None:
    """The muse table for a genre as a person wrote it; `None` when none fits."""
    key = normalize_genre(genre)
    if key in tables:
        return key
    lowered = genre.casefold()
    for word, table in _GENRE_WORDS:
        if word in lowered and table in tables:
            return table
    return None


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """The cosine of two vectors; 0 when either is all zeros."""
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm > 0.0 else 0.0


@dataclass(frozen=True)
class Similarity:
    """How alike the premises are, pair by pair, and by which measure.

    One measure for the tool and the eval: embedding cosine when an embedder answers, the
    character-pair Jaccard index (`bigram_overlap`) when none is configured or it fails.
    """

    measure: Literal["embedding_cosine", "bigram_jaccard"]
    #: One figure per pair, in the order (1, 2), (1, 3), (2, 3), ...
    pairs: tuple[float, ...]

    @property
    def limit(self) -> float:
        return MAX_PREMISE_COSINE if self.measure == "embedding_cosine" else MAX_PREMISE_BIGRAM

    @property
    def mean(self) -> float:
        return sum(self.pairs) / len(self.pairs) if self.pairs else 0.0

    @property
    def closest(self) -> float:
        return max(self.pairs, default=0.0)

    @property
    def distinct(self) -> bool:
        return self.closest <= self.limit

    def record(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "pairs": [round(p, 4) for p in self.pairs],
            "mean": round(self.mean, 4),
            "closest": round(self.closest, 4),
            "limit": self.limit,
        }


def _pairs(count: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(count) for j in range(i + 1, count)]


async def premise_similarity(texts: Sequence[str], embedder: PremiseEmbedder | None) -> Similarity:
    """How alike `texts` are: by `embedder` when it answers, else by character pairs."""
    if embedder is not None and texts:
        try:
            vectors = await embedder.embed(list(texts))
        except Exception as exc:  # noqa: BLE001 - any embedder failure falls back
            # By type only: an embedder's error text can carry a URL or a server's words.
            logger.warning("story_start: the embedder failed (%s)", type(exc).__name__)
        else:
            if len(vectors) == len(texts):
                return Similarity(
                    "embedding_cosine",
                    tuple(cosine(vectors[i], vectors[j]) for i, j in _pairs(len(texts))),
                )
    return Similarity(
        "bigram_jaccard",
        tuple(bigram_overlap(texts[i], texts[j]) for i, j in _pairs(len(texts))),
    )


StructureSource = Literal["request", "genre", "default"]


def structure_for(
    words: str, genre: str, known: Mapping[str, StructureTemplate]
) -> tuple[str, StructureSource] | None:
    """The structure to outline on, and why: the one `words` name, else the genre's.

    The person's words win; a genre, and then `DEFAULT_STRUCTURE`, fill in only when they
    name none. `None` when no structure is loaded at all.
    """
    lowered = words.casefold()
    for pattern, structure_id in _NAMED_STRUCTURES:
        if structure_id in known and re.search(pattern, lowered):
            return structure_id, "request"
    for structure_id, template in known.items():
        if structure_id in lowered or template.title.casefold() in lowered:
            return structure_id, "request"
    genre_words = genre.casefold()
    for word, structure_id in _GENRE_STRUCTURES:
        if word in genre_words and structure_id in known:
            return structure_id, "genre"
    if DEFAULT_STRUCTURE in known:
        return DEFAULT_STRUCTURE, "default"
    return (sorted(known)[0], "default") if known else None


def scenes_for(arc: Arc | None, number: int) -> int:
    """How many scenes chapter `number` has, by its weight in the arc, decided by code.

    A chapter that reveals the twist or decides the conflict (`reversal`, `climax`) gets
    `MAX_SCENES_PER_CHAPTER`, even when it also does a lighter job; one that only sets the
    story going or winds it down (`setup`, `resolution`) gets `MIN_SCENES_PER_CHAPTER`; a
    chapter that complicates or escalates, or any chapter with no arc, gets
    `SCENES_PER_CHAPTER`. The model is told the number and never asked for it.
    """
    if arc is None or not 1 <= number <= len(arc.functions):
        return SCENES_PER_CHAPTER
    functions = set(arc.functions[number - 1])
    if functions & _HEAVY:
        return MAX_SCENES_PER_CHAPTER
    if functions and functions <= _LIGHT:
        return MIN_SCENES_PER_CHAPTER
    return SCENES_PER_CHAPTER


def choice_in(text: str | None) -> int | None:
    """The premise number (1 to 3) a person's words give, when they give exactly one."""
    if not text:
        return None
    found = {int(m) for m in _NUMBER.findall(text)}
    return found.pop() if len(found) == 1 else None


# -- the flow's file ---------------------------------------------------------------------


class Premise(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    premise: str
    muse_genre: str | None = None
    seed: int | None = None
    card: dict[str, str] = Field(default_factory=dict[str, str])
    #: The story engine (`ENGINES`) the premise was built around.
    engine: str | None = None
    #: Must-haves the check found missing in the first draft, which a rewrite was asked for.
    rewritten_for: list[str] = Field(default_factory=list[str])
    #: Must-haves the check still found missing in the premise offered.
    missing: list[str] = Field(default_factory=list[str])


class StartState(BaseModel):
    """`start.yaml`: where the flow stopped, and what each stage made."""

    model_config = ConfigDict(extra="forbid")

    request: str
    quick: bool = False
    stage: Stage = "choose_premise"
    brief: Brief
    #: The genres proposed because the request named none, the recommended one first;
    #: empty when the request named the genre. The brief's genre is what was taken.
    genre_options: list[str] = Field(default_factory=list[str])
    #: The person answered the proposal (took it or named another), rather than the
    #: flow using the recommended genre on its own because the story was asked for at once.
    genre_answered: bool = False
    premises: list[Premise] = Field(default_factory=list[Premise])
    #: The premise a bare "yes" takes at the premise stop.
    recommended: int = 1
    #: `Similarity.record()` of the premises offered.
    premise_similarity: dict[str, Any] | None = None
    chosen: int | None = None
    cast: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    cast_proposals: list[str] = Field(default_factory=list[str])
    #: Why the last try at the cast proposed no character (`_CAST_FAILED`); the flow
    #: stays before the outline and says so, rather than writing a story with no one in it.
    cast_failed: str | None = None
    #: The structure template the outline is built on, and why it is that one: the
    #: person named it, the genre gave it, or nothing did (`structure_for`).
    structure: str | None = None
    structure_source: StructureSource | None = None
    #: What the outline stage decided the story turns on, before filling the beats.
    central_conflict: str | None = None
    twist: str | None = None
    #: The rest of what the story turns on (`uclone_x.story.arc.Spine.record`): the clue a
    #: chapter before the twist may plant, and how the climax decides the conflict.
    spine: dict[str, str] | None = None
    #: What each chapter does (`uclone_x.story.arc.Arc.record`), decided by code from the
    #: structure; the stakes each chapter's outline gave; and what the check of the
    #: outline against that arc found, on the first ask and on the outline kept.
    arc: dict[str, Any] | None = None
    stakes: list[str] = Field(default_factory=list[str])
    outline_check: dict[str, Any] | None = None
    #: What the person said at the outline stop that asked for a change, in turn. Each
    #: revised the outline once, and the flow stopped at the outline again.
    outline_feedback: list[str] = Field(default_factory=list[str])
    #: The last revision came back unreadable, so the outline was kept as it was.
    revision_failed: bool = False
    #: The person settled the plot at the outline stop: a bare yes, words that settle it
    #: (`settles_plot`), or asking for the story at once. The flow no longer revises it.
    plot_locked: bool = False
    #: What settling the plot proposed for the codex -- threads (the central conflict, the
    #: twist, key events and relationships) and items -- as `{"proposal", "kind",
    #: "entry_id", "name"}`. Each waits for the person's approval, like the cast.
    plot_kg: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
    #: How many items the plot lock's reading named that were ideas, not objects
    #: (`abstract_item`), and so were not proposed.
    plot_kg_dropped: int = 0
    #: Cast proposals still pending when the chapter was asked for without the person
    #: saying to go on: the chapter waits for them.
    cast_waiting: list[str] = Field(default_factory=list[str])
    #: Cast proposals still pending when the chapter was written over them, because the
    #: person asked for the story at once or said to go on; the reply says so.
    cast_assumed: list[str] = Field(default_factory=list[str])
    outline_saved: bool = False
    scenes_written: list[str] = Field(default_factory=list[str])
    # The last chapter attempt saved no scene; the stage stays at review_chapter.
    chapter_failed: bool = False
    #: What the check after writing found (`Finding.record()`): shown to the person, who
    #: may ask for the scene to be rewritten. Nothing is rewritten without that.
    continuity: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
    #: Scenes the check read, and scenes it could not read (the reply had no facts).
    continuity_checked: list[str] = Field(default_factory=list[str])
    continuity_unread: list[str] = Field(default_factory=list[str])
    #: Scenes whose reading against their plan and the twist (`plan_check`) could not be
    #: read: their plan's beats and the twist were not checked, never called kept.
    plan_unread: list[str] = Field(default_factory=list[str])
    #: Characters a written scene left dead, by id: that scene's id and title and the quote.
    #: A death reaches the codex only when a person approves it, so this is what holds a
    #: later scene to it (`continuity.came_back`).
    written_deaths: dict[str, dict[str, str]] = Field(default_factory=dict[str, dict[str, str]])
    #: Scenes rewritten because the person asked, after a finding.
    rewritten: list[str] = Field(default_factory=list[str])
    #: Codex proposals read from the scenes this start wrote (`grow_scene`), every one so
    #: far, and those of the last call that wrote a scene: pending until a person decides.
    codex_growth: list[str] = Field(default_factory=list[str])
    growth_now: list[str] = Field(default_factory=list[str])
    #: The last chapter written (its number in the outline), and the scenes that call
    #: wrote: the check's notes and a rewrite the person asks for are about those.
    chapter_written: int = 0
    chapter_scenes: list[str] = Field(default_factory=list[str])
    #: The chapter whose plan is shown at the chapter stop (`review_chapter`), and the
    #: plan: its scenes as `{"title", "summary", "beats", "characters", "places"}`, the
    #: cast by name. Agreeing writes the chapter from it; other words revise it.
    details_chapter: int | None = None
    chapter_details: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    #: What the person said at this chapter's stop that asked for a change, in turn.
    details_feedback: list[str] = Field(default_factory=list[str])
    #: The last revision of the plan came back unreadable, so the plan was kept.
    details_revision_failed: bool = False
    #: The plan could not be read, so the outline's scenes for the chapter are shown.
    details_from_outline: bool = False
    #: New characters the plan brings in with a stated family relation to a character
    #: the story has, as `{"name", "profile", "gender", "family": [{"relative", "is",
    #: "relative_id", "role"}]}` ("is" what the relative is to the new character, "role"
    #: the kinship `family_role` reads from it). Agreeing to the plan proposes them
    #: (`_introduce`).
    details_new: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    #: The new characters a chapter plan introduced and the kinship proposals made for
    #: them, as `{"proposal", "kind", "entry_id", "name", "chapter", "change"}` ("change"
    #: is ``new_entry`` for the character, ``relation`` for the relative's side). Each is
    #: pending until the person decides it; none is ever approved here.
    introduced: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    notes: list[str] = Field(default_factory=list[str])

    def premise(self) -> Premise:
        number = self.chosen or 1
        return next((p for p in self.premises if p.number == number), self.premises[0])


def _load_state(work: StoryWork) -> tuple[StartState, str]:
    found = work.read(START_FILE)
    if found is None:
        raise StoryError(
            "This story was not started with story_start, so there is no start to continue. "
            "Use story_start 'start' with the person's request to begin a new one."
        )
    try:
        raw: object = yaml.safe_load(found.text)
        return StartState.model_validate(raw), found.digest
    except (yaml.YAMLError, ValidationError) as exc:
        raise StoryError(
            "The record of how this story was started could not be read, so nothing was done."
        ) from exc


def _dump(state: StartState) -> str:
    return yaml.safe_dump(
        state.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100
    )


# -- the tool ----------------------------------------------------------------------------


class StoryStartParams(BaseModel):
    """Where to take the story start."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["start", "continue", "status"] = Field(
        description="'start' a new story from the person's request; 'continue' the story "
        "this conversation started, to the next stage; 'status' says where it stands."
    )
    request: str | None = Field(
        default=None,
        description="For 'start': the person's request, word for word. For 'continue': "
        "what the person just said, word for word.",
    )
    choice: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="For 'continue': the number the person chose, of the genres or the "
        "premises offered. Leave it out when they gave no number.",
    )


class StoryStartTool(BaseTool[StoryStartParams]):
    """A new story from a person's request, through fixed stages, stopping for the person."""

    name = "story_start"
    description = (
        "Start a new story from what the person asks for, and take it forward one stage at "
        "a time: direction (a genre is proposed when the request names none), three "
        "premises for the person to choose from, cast and places (proposed for approval), "
        "outline (the person's changes revise it; agreeing settles the plot and proposes "
        "its key events for the codex), then each chapter in turn: its scene plan is shown "
        "for the person to change or accept, and accepting writes it. 'start' with the request word for word; after "
        "the person answers, 'continue' with what they said word for word, even a bare "
        "'yes' (and 'choice' when they gave a number). Each call says the stage it reached "
        "and what the person is asked. If the person asks to have it written at once, the "
        "whole start runs in one call, one chapter per call."
    )
    params_type = StoryStartParams
    writes_files: ClassVar[bool] = True
    opens_story: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"status"})
    needs_room: ClassVar[bool] = True
    not_run_note: ClassVar[str] = "No story was started or changed."

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        data_roots: Sequence[Path] = (BUNDLED_DATA_ROOT,),
        embedder_factory: EmbedderFactory | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._model_factory = model_factory or _default_model
        self._embedder_factory = embedder_factory or _default_embedder
        # Draws the genre proposed when the request names none; a test passes a seeded one.
        self._rng = rng or secrets.SystemRandom()
        self._data_roots = tuple(data_roots)
        super().__init__(name=self.name, description=self.description, params_type=StoryStartParams)

    async def run(self, params: StoryStartParams, context: ToolContext) -> dict[str, Any]:
        if params.action == "start":
            return await self._start(params, context)
        if params.action == "status":
            work = StoryWork.open_in(context)
            state, _ = _load_state(work)
            return self._result(work, state, done=[])
        story_id, room_id = StoryLibrary.writer_of(context)
        work = StoryWork(StoryLibrary(context.require_workspace()), story_id)
        state, digest = _load_state(work)
        choice = params.choice or choice_in(params.request)
        if params.request and wants_it_now(params.request):
            state = state.model_copy(update={"quick": True})
        yes = bool(params.request and says_yes(params.request))
        go_on = bool(params.request and says_go_on(params.request)) or yes
        structures = self._structures(context)
        model = self._model_factory(context)
        if state.stage == "choose_genre":
            genre = await _genre_answer(model, state, params.request, choice, yes=yes)
            run = _Run(work, room_id, context, model, state, digest, structures)
            await run.take_genre(genre, self._tables(context), self._embedder_factory(context))
            done = ["genre", "premises"]
            if run.state.quick:
                done += await run.forward(choice=None, go_on=True)
            return self._result(work, run.state, done=done)
        if params.request and not state.outline_saved:
            # A structure the person names before the outline is built wins over the
            # genre's; words that name none change nothing.
            named = structure_for(params.request, "", structures)
            if named is not None and named[1] == "request":
                state = state.model_copy(
                    update={"structure": named[0], "structure_source": "request"}
                )
        rewrite = bool(params.request and asks_rewrite(params.request))
        run = _Run(work, room_id, context, model, state, digest, structures)
        done = await run.forward(
            choice=choice, go_on=go_on, rewrite=rewrite, yes=yes, words=params.request
        )
        return self._result(work, run.state, done=done)

    async def _start(self, params: StoryStartParams, context: ToolContext) -> dict[str, Any]:
        request = (params.request or "").strip()
        if not request:
            raise StoryError(
                "Pass the person's request word for word in 'request', so nothing was started."
            )
        room_id = context.room_id
        if room_id is None:
            raise StoryError("A story is started inside a conversation, so nothing was started.")
        library = StoryLibrary(context.require_workspace())
        model = self._model_factory(context)
        brief = await _direction(model, request)
        tables = self._tables(context)
        quick = wants_it_now(request)
        # A genre the request names is kept (#1723); otherwise one is proposed, drawn by
        # code, and the flow stops for the person to take it or name another -- unless
        # they asked for the story at once, when the one drawn is used and they are told.
        options = [] if "genre" in brief.from_request else genre_options(sorted(tables), self._rng)
        if options:
            brief = brief.model_copy(update={"genre": options[0]})
        record = library.create(
            brief.title,
            room_id,
            genre=brief.genre,
            style_notes=_style_notes(brief),
        )
        released: dict[str, Any] = {}
        if context.story_id is not None and context.story_id != record.story_id:
            try:
                if library.release(context.story_id, room_id):
                    released["released"] = context.story_id
            except StoryError:
                released["previous_story_not_released"] = context.story_id
        work = StoryWork(library, record.story_id)
        structures = self._structures(context)
        picked = structure_for(request, brief.genre, structures)
        state = StartState(
            request=request,
            quick=quick,
            stage="choose_genre" if options and not quick else "choose_premise",
            brief=brief,
            genre_options=options,
            structure=picked[0] if picked else None,
            structure_source=picked[1] if picked else None,
        )
        done = ["direction"]
        if state.stage == "choose_premise":
            state = await _premises(model, state, tables, self._embedder_factory(context))
            done.append("premises")
        digest = work.write(START_FILE, _dump(state), room_id=room_id, expected_digest=None)
        try:
            work.note_session(room_id)
        except (StoryError, StoryFileError):
            logger.warning("story_start: the session of %s was not recorded", record.story_id)
        run = _Run(work, room_id, context, model, state, digest, structures)
        if state.quick:
            done.extend(await run.forward(choice=None, go_on=True))
        result = self._result(work, run.state, done=done)
        return {OPEN_STORY_KEY: record.story_id, **released, **result}

    def _tables(self, context: ToolContext) -> dict[str, MuseTable]:
        try:
            found = load_muse_tables_sourced(data_roots(context.skill_dirs, self._data_roots))
        except SkillDataError as exc:
            logger.warning("story_start could not load the muse tables: %s", exc)
            return {}
        return {genre: sourced.item for genre, sourced in found.items()}

    def _structures(self, context: ToolContext) -> dict[str, StructureTemplate]:
        """The bundled structure templates and those of the clone's active skills."""
        try:
            found = load_structure_templates_sourced(
                data_roots(context.skill_dirs, self._data_roots)
            )
        except SkillDataError as exc:
            logger.warning("story_start could not load the structures: %s", exc)
            return {}
        return {structure_id: sourced.item for structure_id, sourced in found.items()}

    @staticmethod
    def _result(work: StoryWork, state: StartState, *, done: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "story_id": work.story_id,
            "stage": state.stage,
            "done_now": done,
            "brief": state.brief.model_dump(mode="json"),
            "premises": [
                {"number": p.number, "title": p.title, "premise": p.premise} for p in state.premises
            ],
        }
        if state.genre_options:
            result["genre_options"] = state.genre_options
        if state.premises:
            result["recommended"] = state.recommended
        if state.chosen is not None:
            result["chosen"] = state.chosen
        if state.cast_failed is not None:
            result["cast_failed"] = state.cast_failed
        if state.structure is not None:
            result["structure"] = {"id": state.structure, "source": state.structure_source}
        if state.central_conflict or state.twist:
            result["central_conflict"] = state.central_conflict
            result["twist"] = state.twist
        if state.spine is not None:
            result["spine"] = state.spine
        if state.arc is not None:
            result["arc"] = {**state.arc, "stakes": state.stakes}
        if state.outline_check is not None:
            result["outline_check"] = state.outline_check
        if state.outline_feedback:
            result["outline_feedback"] = state.outline_feedback
        if state.plot_locked:
            result["plot_locked"] = True
            result["plot_kg"] = state.plot_kg
        if state.plot_kg_dropped:
            result["plot_kg_dropped"] = state.plot_kg_dropped
        if state.cast_proposals:
            result["cast_proposals"] = state.cast_proposals
        if state.premise_similarity is not None:
            result["premise_similarity"] = state.premise_similarity
        if state.scenes_written:
            result["scenes_written"] = state.scenes_written
        if state.cast_waiting:
            result["cast_waiting"] = state.cast_waiting
        if state.cast_assumed:
            result["cast_assumed"] = state.cast_assumed
        if state.introduced:
            result["introduced"] = state.introduced
        if state.continuity_checked or state.continuity_unread:
            result["continuity"] = state.continuity
            result["continuity_unread"] = state.continuity_unread
            result["plan_unread"] = state.plan_unread
        if state.rewritten:
            result["rewritten"] = state.rewritten
        if state.codex_growth:
            result["codex_growth"] = state.codex_growth
        if state.chapter_written:
            result["chapter_written"] = state.chapter_written
            result["chapter_scenes"] = state.chapter_scenes
        if state.details_chapter is not None:
            result["details_chapter"] = state.details_chapter
            result["chapter_details"] = state.chapter_details
            if state.details_new:
                result["details_new"] = state.details_new
            if state.details_feedback:
                result["details_feedback"] = state.details_feedback
        if state.notes:
            result["notes"] = state.notes
        result["next"] = (
            _WAITING_NEXT
            if state.cast_waiting
            else _CAST_FAILED_NEXT
            if state.cast_failed
            else _CONTINUITY_NEXT
            if state.stage in ("review_chapter", "done") and _shown_findings(state)
            else _NEXT[state.stage]
        )
        result[REPLY_NOTE_KEY] = _reply_note(state, done)
        return result


_NEXT: Mapping[Stage, str] = {
    "choose_genre": "The request named no genre, so genres are proposed, the recommended "
    "one first. Show them to the person and ask; then call story_start 'continue' with "
    "what they said, word for word, as 'request' (and 'choice' if they gave a number). A "
    "bare yes takes the recommended genre.",
    "choose_premise": "Show the person the three premises and ask which one. Then call "
    "story_start 'continue' with what they said, word for word, as 'request' (and "
    "'choice' if they gave a number). A bare yes takes the recommended premise.",
    "review_cast": "The cast and places are proposed; the person approves or rejects them "
    "in the story's view. When they want to go on, call story_start 'continue'. The first "
    "chapter's plan waits until the proposals are decided or the person says to go on anyway.",
    "review_outline": "The outline is saved; ask the person whether it works. If they ask "
    "for a change, call story_start 'continue' with their words, word for word, as "
    "'request': the outline is revised and shown again. When they agree (a bare yes is "
    "enough), call 'continue': the plot is settled, its key events are proposed for the "
    "codex, and the first chapter's scene plan is shown.",
    "review_chapter": "A chapter's scene plan is shown before the chapter is written (after "
    "the chapter before it, if any, was written). Ask the person whether it works. If they ask for a change, call "
    "story_start 'continue' with their words, word for word, as 'request': the plan is "
    "revised and shown again. When they agree (a bare yes is enough), call 'continue': "
    "that chapter is written, checked, and the plan of the one after it is shown.",
    "done": "Every chapter of the outline is written. Go on with story_context and "
    "story_manuscript if the person wants more.",
}
_CAST_FAILED_NEXT = (
    "No character could be proposed from the chosen premise, so the flow stopped before "
    "the outline. Tell the person so; when they want to try again, call story_start "
    "'continue' (with 'choice' if they choose another premise)."
)
_CONTINUITY_NEXT = (
    "A chapter is written, and the check after writing found lines that disagree with the "
    "codex, or with the scene's plan or the settled plot. Show the person the notes as they are, and ask whether to rewrite those scenes. "
    "Only if they ask for it, call story_start 'continue' with their words. Anything else "
    "they say goes on as for the plan shown, if there is one."
)
_WAITING_NEXT = (
    "The first chapter was not planned: proposed cast entries are still waiting for the "
    "person's approval. Ask them to approve or reject them in the story's view, then call "
    "story_start 'continue'; or, if they say to go on as proposed, call 'continue' with "
    "their words."
)


#: Why no character was proposed, as the person is told it.
_CAST_FAILED: Mapping[str, tuple[str, str]] = {
    "unreadable": (
        f"인물 목록을 {MAX_CAST_TRIES}번 만들었지만 모두 형식이 깨져 읽을 수 없었습니다.",
        f"the cast came back {MAX_CAST_TRIES} times in a form that could not be read",
    ),
    "empty": (
        f"{MAX_CAST_TRIES}번 시도했지만 이름이 있는 인물이 한 명도 나오지 않았습니다.",
        f"{MAX_CAST_TRIES} tries gave no character with a name",
    ),
    "taken": (
        "제안한 인물이 모두 이 이야기에 이미 있는 인물과 겹쳤습니다.",
        "every character proposed is already in the story",
    ),
}


def _reply_note(state: StartState, done: Sequence[str] = ()) -> dict[str, str]:
    """What code tells the person, whatever the model writes: the stage and the question.

    The call that settled the plot says so first, and what it proposed for the codex.
    """
    note = _stage_note(state, done)
    if "introduced" in done:
        names = [
            str(k["name"])
            for k in state.introduced
            if k.get("change") == "new_entry" and k.get("chapter") == state.chapter_written
        ]
        if names:
            listed = ", ".join(names)
            note = {
                "ko": f"계획에 나온 새 인물({listed})과 그 가족 관계를 설정집에 제안했습니다. "
                f"이야기 보기에서 승인하시면 설정집에 들어갑니다. {note['ko']}",
                "en": f"The new characters the plan brought in ({listed}) and their family "
                f"ties were proposed for the codex; they join it once you approve them. "
                f"{note['en']}",
            }
    if "plot" not in done:
        return note
    ko, en = _plot_note(state)
    return {"ko": f"{ko} {note['ko']}", "en": f"{en} {note['en']}"}


def _plot_note(state: StartState) -> tuple[str, str]:
    if not state.plot_kg:
        return (
            "줄거리를 확정했습니다. 설정집에 새로 제안할 항목은 없었습니다.",
            "The plot is settled. Nothing new was proposed for the codex.",
        )
    names = ", ".join(k["name"] for k in state.plot_kg)
    count = len(state.plot_kg)
    return (
        f"줄거리를 확정했습니다. 확정한 줄거리에서 설정집에 넣을 항목 {count}개({names})를 "
        "제안했습니다. 이야기 보기에서 승인하시면 설정집에 들어갑니다.",
        f"The plot is settled. {count} entries from it were proposed for the codex "
        f"({names}); they join it once you approve them in the story's view.",
    )


def _stage_note(state: StartState, done: Sequence[str] = ()) -> dict[str, str]:
    if state.cast_failed is not None:
        why_ko, why_en = _CAST_FAILED.get(state.cast_failed, _CAST_FAILED["empty"])
        return {
            "ko": f"고르신 {state.chosen or 1}번 발상에서 인물을 뽑지 못했습니다. {why_ko} "
            "인물 없이 개요나 첫 장을 만들지는 않았습니다. 다시 해 보라고 하시거나, "
            "다른 발상의 번호를 골라 주세요.",
            "en": f"No character could be drawn from premise {state.chosen or 1}: {why_en}. "
            "Nothing was outlined or written without a cast. Ask me to try again, or choose "
            "another premise by number.",
        }
    if state.stage == "choose_genre":
        first = state.genre_options[0]
        lines = [
            f"{n}. {g}" + (" (추천)" if n == 1 else "")
            for n, g in enumerate(state.genre_options, start=1)
        ]
        lines_en = [
            f"{n}. {g}" + (" (recommended)" if n == 1 else "")
            for n, g in enumerate(state.genre_options, start=1)
        ]
        return {
            "ko": "요청에 장르가 없어 장르부터 제안합니다.\n" + "\n".join(lines) + "\n"
            f"‘네’라고 하시면 {first}로 발상을 만들겠습니다. 다른 장르를 원하시면 번호나 "
            "장르 이름을 말씀해 주세요.",
            "en": "The request named no genre, so here are some to choose from:\n"
            + "\n".join(lines_en)
            + f"\nSay yes and I will build the premises as {first}; or name another genre "
            "or number.",
        }
    if state.stage == "choose_premise":
        lines = [f"{p.number}. {p.title} — {p.premise}" for p in state.premises]
        genre = f"{state.brief.genre} " if state.genre_options else ""
        ko = f"{genre}이야기의 핵심 발상 세 가지입니다.\n" + "\n".join(lines)
        ko += (
            f"\n마음에 드는 발상의 번호를 골라 주세요. 추천은 {state.recommended}번입니다. "
            f"‘네’라고만 하셔도 {state.recommended}번으로 진행합니다."
        )
        en = "Three premises for the story:\n" + "\n".join(lines)
        en += f"\nPlease choose one by number. I recommend {state.recommended}; say yes to take it."
        return {"ko": ko, "en": en}
    if state.stage == "review_cast":
        names = ", ".join(str(c.get("name", "")) for c in state.cast)
        return {
            "ko": f"인물과 장소를 제안했습니다: {names}. 이야기 보기의 파일에서 승인하거나 "
            "고칠 수 있습니다. ‘네’라고 하시면 개요를 만들겠습니다. 첫 장은 제안을 모두 "
            "정하신 뒤에 계획합니다. 다만 개요에서 ‘네’라고만 하시면 정하지 않은 제안도 "
            "제안대로 보고 첫 장을 계획하고 쓰며, 그 제안은 승인하실 때까지 승인되지 않은 채로 "
            "남습니다.",
            "en": f"The cast and places are proposed: {names}. You can approve or change "
            "them in the story's view. Say yes and I will make the outline. The first "
            "chapter waits until every proposal is decided; but if you just say yes at the "
            "outline, it is planned and written with the undecided ones as proposed, and they stay "
            "unapproved until you approve them.",
        }
    if state.stage == "review_outline" and state.cast_waiting:
        count = len(state.cast_waiting)
        return {
            "ko": f"제안한 인물과 장소 중 {count}개가 아직 승인되지 않아 첫 장을 계획하지 "
            "않았습니다. 이야기 보기에서 승인하거나 거절하신 뒤 이어 달라고 말씀해 주세요. "
            "‘네’라고 하시거나 제안대로 진행하라고 하시면 첫 장의 계획을 보여 드리겠습니다.",
            "en": f"{count} proposed cast entries are not approved yet, so the first chapter "
            "is not planned. Approve or reject them in the story's view and ask me to go "
            "on, or tell me to go ahead as proposed.",
        }
    if state.stage == "review_outline":
        ko_shape, en_shape = _arc_note(state)
        if state.revision_failed:
            ko_head = "말씀하신 수정을 반영하지 못해 개요를 그대로 두었습니다."
            en_head = "Your change could not be made, so the outline is as it was."
        elif state.outline_feedback:
            ko_head = "말씀하신 대로 개요를 고쳐 저장했습니다."
            en_head = "The outline is revised as you asked, and saved."
        else:
            ko_head = "개요를 저장했습니다."
            en_head = "The outline is saved."
        return {
            "ko": f"{ko_head}{ko_shape} 이야기 보기에서 확인할 수 있습니다. 바꾸고 "
            "싶은 점이 있으면 말씀해 주세요. ‘네’라고 하시면 이 줄거리로 확정하고 "
            "첫 장의 장면 계획을 보여 드리겠습니다.",
            "en": f"{en_head}{en_shape} You can see it in the story's view. Tell me what to "
            "change, or say yes to settle the plot and see the first chapter's scene plan.",
        }
    if state.stage == "review_chapter" and state.chapter_failed:
        number = state.details_chapter or state.chapter_written + 1
        return {
            "ko": f"{number}장의 초안이 저장 기준을 통과하지 못해 저장하지 못했습니다. 다시 써 "
            "보려면 ‘네’라고 말씀해 주세요.",
            "en": f"Chapter {number}'s draft did not pass the checks, so nothing was saved. "
            "Say yes and I will try again.",
        }
    wrote = state.stage == "done" or "chapter" in done or "rewrite" in done
    ko, en = _written_note(state, done) if wrote else ("", "")
    if state.stage == "review_chapter":
        ko_plan, en_plan = _plan_note(state, done)
        ko = f"{ko}\n\n{ko_plan}" if ko else ko_plan
        en = f"{en}\n\n{en_plan}" if en else en_plan
    return {"ko": ko, "en": en}


def _written_note(state: StartState, done: Sequence[str]) -> tuple[str, str]:
    """What the call wrote: the chapter, or the scenes rewritten, and what the check found."""
    number = state.chapter_written or 1
    count = len(state.chapter_scenes) or len(state.scenes_written)
    ko_label = "첫 장" if number == 1 else f"{number}장"
    en_label = "The first chapter" if number == 1 else f"Chapter {number}"
    ko = f"{ko_label}을 썼습니다. 장면 {count}개를 저장했습니다."
    en = f"{en_label} is written: {count} scene(s) saved."
    first = number == 1
    if first and state.quick and state.genre_options and not state.genre_answered:
        ko = f"요청에 장르가 없어 {state.brief.genre}로 정해 썼습니다. " + ko
        en = f"The request named no genre, so it was written as {state.brief.genre}. " + en
    if state.stage == "done" and "chapter" in done:
        ko += " 개요의 마지막 장까지 모두 썼습니다."
        en += " Every chapter of the outline is written."
    if state.rewritten and ("rewrite" in done or state.stage == "done" and "chapter" not in done):
        shown = [
            s for s in state.rewritten if not state.chapter_scenes or s in state.chapter_scenes
        ]
        ko = f"말씀하신 대로 장면 {len(shown)}개를 다시 써서 저장했습니다."
        en = f"{len(shown)} scene(s) rewritten as you asked, and saved."
    grown = growth_line(len(state.growth_now), scenes=True)
    if grown is not None and ("chapter" in done or "rewrite" in done):
        ko += f" {grown['ko']}"
        en += f" {grown['en']}"
    if first and state.cast_assumed and "rewrite" not in done:
        ko += (
            " 제안한 인물과 장소를 승인된 것으로 보고 썼습니다. 제안은 아직 이야기 보기에서 "
            "승인을 기다리고 있으니, 확인하고 승인하거나 고쳐 주세요."
        )
        en += (
            " It was written treating the proposed cast and places as approved; the "
            "proposals still wait for your approval in the story's view."
        )
    ko_check, en_check = _continuity_note(state)
    return ko + ko_check, en + en_check


def _plan_note(state: StartState, done: Sequence[str]) -> tuple[str, str]:
    """The next chapter's plan, shown before it is written, and what the person may say."""
    number = state.details_chapter or state.chapter_written + 1
    if "details_revised" in done and state.details_revision_failed:
        ko = f"말씀하신 수정을 반영하지 못해 {number}장의 장면 계획을 그대로 두었습니다."
        en = f"Your change could not be made, so chapter {number}'s plan is as it was."
    elif "details_revised" in done:
        ko = f"말씀하신 대로 {number}장의 장면 계획을 고쳤습니다."
        en = f"Chapter {number}'s plan is revised as you asked."
    elif state.details_from_outline:
        ko = f"{number}장의 장면 계획을 새로 짜지 못해 개요에 있는 장면을 그대로 보여 드립니다."
        en = f"Chapter {number}'s plan could not be made, so here are its scenes from the outline."
    else:
        ko = f"다음은 {number}장의 장면 계획입니다."
        en = f"Here is the plan of chapter {number}."
    lines = "\n".join(
        f"{n}. {s.get('title', '')}"
        + (f" — {s['summary']}" if s.get("summary") and s["summary"] != s.get("title") else "")
        for n, s in enumerate(state.chapter_details, start=1)
    )
    ko += f"\n{lines}\n"
    en += f"\n{lines}\n"
    if state.details_new:
        names = ", ".join(str(c["name"]) for c in state.details_new)
        ko += (
            f"이 장에서 새로 나오는 인물은 {names}입니다. 이대로 쓰면 이 인물과 가족 관계를 "
            "설정집에 제안합니다.\n"
        )
        en += (
            f"New in this chapter: {names}. Written as planned, they and their family ties "
            "are proposed for the codex.\n"
        )
    if not state.quick:
        ko += (
            "바꾸고 싶은 곳이 있으면 말씀해 주세요. ‘네’라고 하시면 이대로 "
            f"{number}장을 쓰겠습니다."
        )
        en += f"Tell me what to change, or say yes and I will write chapter {number} as planned."
        return ko, en
    # Asked for at once: still one chapter per call, and the person is told how to go on.
    ko += (
        f"한 번에 한 장씩 씁니다. ‘계속’이라고 하시면 {number}장을 이어 쓰겠습니다. "
        "바꾸고 싶은 곳을 말씀하시면 고쳐서 바로 쓰겠습니다."
    )
    en += (
        f"One chapter is written per request. Say 'continue' and I will write chapter "
        f"{number}; tell me what to change and I will change it and write it."
    )
    return ko, en


#: A problem the outline check left, as the person reads it (`uclone_x.story.arc`).
_PROBLEM_KO: Mapping[str, str] = {
    "twist_early": "{n}장에서 반전이 너무 일찍 드러납니다",
    "twist_missing": "{n}장에서 반전이 드러나지 않습니다",
    "resolved_early": "{n}장에서 갈등이 너무 일찍 풀립니다",
    "repeats": "{n}장이 앞 장의 사건을 되풀이합니다",
    "no_twist": "반전이 정해지지 않았습니다",
    "missing_chapters": "{n}장부터 개요가 비어 있습니다",
}


def _arc_note(state: StartState) -> tuple[str, str]:
    """Where the outline turns, and what its check still found, in a sentence or two."""
    if state.arc is None:
        return "", ""
    twist, climax = state.arc["twist_chapter"], state.arc["climax_chapter"]
    ko = f" 반전은 {twist}장에서 처음 드러나고, 갈등은 {climax}장에서 결판나도록 짰습니다."
    en = f" The twist is revealed in chapter {twist} and the conflict decided in chapter {climax}."
    raw = (state.outline_check or {}).get("final")
    left = [cast(dict[str, Any], p) for p in cast(list[object], raw or []) if isinstance(p, dict)]
    if left:
        said = ", ".join(
            _PROBLEM_KO.get(str(p["kind"]), "{n}장에 고칠 곳이 있습니다").format(n=p["chapter"])
            for p in left
        )
        ko += f" 다만 개요를 점검해 보니 아직 고칠 곳이 있습니다: {said}."
        en += (
            " The outline check still found: "
            + ", ".join(f"{p['kind']} (chapter {p['chapter']})" for p in left)
            + "."
        )
    return ko, en


def _continuity_note(state: StartState) -> tuple[str, str]:
    """What the check after writing found, with each quote, and the offer to rewrite.

    Only the chapter written last is reported: earlier chapters' findings were shown when
    they were written.
    """
    ko = en = ""
    scenes = set(state.chapter_scenes)
    shown = _shown_findings(state)
    checked = [s for s in state.continuity_checked if not scenes or s in scenes]
    unread = [s for s in state.continuity_unread if not scenes or s in scenes]
    if shown:
        lines = "\n".join(f"- {f['note']}" for f in shown)
        count = len(shown)
        ko += (
            f"\n\n쓴 장면을 설정집과 앞 장면, 장면 계획에 대조해 보니 맞지 않는 곳이 {count}곳 "
            f"있습니다.\n{lines}\n"
            "이 장면을 다시 쓸까요? 다시 써 달라고 하시면 그 장면만 고쳐 쓰겠습니다. "
            "이야기에서 일부러 그렇게 쓴 것이라면 그대로 두셔도 됩니다."
        )
        en += (
            f"\n\nChecked against the codex and the scenes' plan, {count} line(s) "
            f"disagree:\n{lines}\n"
            "Shall I rewrite those scenes? Ask me to and I will rewrite only them; if it is "
            "meant that way, leave it."
        )
    elif checked and unread:
        # Some scenes were never read, so "nothing disagreed" holds only for the rest.
        ko += (
            f"\n\n대조한 장면 {len(checked)}개에서는 인물의 생사, 성별, 가족 관계가 설정집과 "
            "어긋나는 곳을 찾지 못했습니다."
        )
        en += f"\n\nIn the {len(checked)} scene(s) checked against the codex, nothing disagreed."
    elif checked:
        ko += (
            "\n\n쓴 장면의 인물 생사, 성별, 가족 관계를 설정집과 대조했고, 어긋나는 곳은 "
            "찾지 못했습니다."
        )
        en += "\n\nThe scenes' characters were checked against the codex; nothing disagreed."
    if unread:
        ko += f" 장면 {len(unread)}개는 설정집과 대조하지 못했습니다."
        en += f" {len(unread)} scene(s) could not be checked."
    plan_unread = [s for s in state.plan_unread if not scenes or s in scenes]
    if not shown and scenes and len(plan_unread) < len(scenes):
        ko += " 장면 계획에 있던 인물과 사건, 반전을 드러낼 때와도 어긋나는 곳을 찾지 못했습니다."
        en += " Nothing disagreed with the scenes' plan or the twist's timing either."
    if plan_unread:
        ko += f" 장면 {len(plan_unread)}개는 장면 계획과 대조하지 못했습니다."
        en += f" {len(plan_unread)} scene(s) could not be checked against their plan."
    return ko, en


def _style_notes(brief: Brief) -> str:
    parts = [f"톤: {brief.tone}", f"독자: {brief.audience}", f"분량: {brief.length}"]
    if brief.avoid:
        parts.append("피할 것: " + ", ".join(brief.avoid))
    return "; ".join(parts)


# -- the stages --------------------------------------------------------------------------

_SYSTEM = (
    "You help plan and write fiction. Write every piece of story text in the language of "
    "the person's request. When asked for JSON, reply with one JSON object and nothing else. "
    "No sexual content, and nothing that harms minors."
)


async def _ask_json(
    model: StoryModel, prompt: str, *, temperature: float, max_tokens: int
) -> dict[str, Any] | None:
    """A JSON object from the model: asked twice at most, `None` when neither reply had one."""
    for _ in range(2):
        reply = await model.complete(
            prompt, system=_SYSTEM, temperature=temperature, max_tokens=max_tokens
        )
        found = json_object(reply)
        if found is not None:
            return found
    return None


def _brief_lines(brief: Brief) -> str:
    lines = [
        f"Genre: {brief.genre}",
        f"Tone: {brief.tone}",
        f"Audience: {brief.audience}",
        f"Length: {brief.length}",
    ]
    if brief.must_have:
        lines.append("Must include: " + "; ".join(brief.must_have))
    if brief.avoid:
        lines.append("Must avoid: " + "; ".join(brief.avoid))
    return "\n".join(lines)


async def _direction(model: StoryModel, request: str) -> Brief:
    """Stage 0: only what the request says; code fills the rest (`brief_from`)."""
    prompt = (
        "Read the person's request for a story and write down ONLY what it says. Leave a "
        "field null (or an empty list) when the request does not say it; do not invent.\n"
        'Reply as JSON: {"title": a short working title in the request\'s language, '
        '"genre": ..., "tone": ..., "audience": ..., "length": ..., '
        '"must_have": [elements the request wants in the story], '
        '"avoid": [things the request does not want]}\n\n'
        f"Request:\n{request}"
    )
    stated = await _ask_json(model, prompt, temperature=0.0, max_tokens=600)
    return brief_from(stated, request)


_PREMISE_REPLY = (
    'Reply as JSON: {"title": a short title, "premise": the premise in two or three sentences}'
)


def _premise_prompt(state: StartState, engine: str, cards: str) -> str:
    return (
        f"The person asked for:\n{state.request}\n\nDirection:\n{_brief_lines(state.brief)}"
        "\n\nIdea cards to spark from (use or bend them; ignore any that break the genre "
        f"or tone):\n{cards or '- (none)'}\n\n"
        f"Write one premise for this story whose engine is {engine}.\n"
        "Keep every 'must include' element and the genre and tone, but do not retell the "
        "request: the person already knows it. Open with the event that sets this story "
        "going, not with who the protagonist is. Invent and name the other person at the "
        "centre of it, say exactly what happened, what the protagonist wants now, and what "
        f"stands in the way.\n{_PREMISE_REPLY}"
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
    model: StoryModel, state: StartState, engine: str, cards: str
) -> tuple[str, str, list[str], list[str]]:
    """A premise around `engine`: its title, text, what a rewrite was for, what is missing.

    A premise that drops a must-have is rewritten once, told every must-have and not only
    the dropped ones: told only the dropped one, qwen3:8b rewrote a premise around it and
    lost another. The rewrite is kept only when it misses fewer.
    """
    title, text = await _drafted(model, _premise_prompt(state, engine, cards))
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
        f"restate the request.\n{_PREMISE_REPLY}"
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
    """Stage 1: three premises, each around its own engine, with its own cards."""
    notes = list(state.notes)
    table_key = muse_genre(state.brief.genre, sorted(tables))
    if table_key is None and tables:
        table_key = _FALLBACK_TABLE if _FALLBACK_TABLE in tables else sorted(tables)[0]
        notes.append(
            f"No idea table fits the genre '{state.brief.genre}', so the cards were drawn "
            f"from '{table_key}'."
        )
    engines = secrets.SystemRandom().sample(ENGINES, len(ENGINES))

    async def draft(number: int, engine: str) -> Premise:
        seed = secrets.randbelow(MAX_SEED + 1)
        card = draw_cards(tables[table_key], seed, 1)[0] if table_key else {}
        cards = "\n".join(f"- {slot}: {entry}" for slot, entry in card.items())
        title, text, rewritten_for, missing = await _one_premise(model, state, engine, cards)
        if not text:
            notes.append(f"Premise {number} came back without text.")
        return Premise(
            number=number,
            title=title or f"발상 {number}",
            premise=text or "(내용 없음)",
            muse_genre=table_key,
            seed=seed if table_key else None,
            card=card,
            engine=engine,
            rewritten_for=rewritten_for,
            missing=missing,
        )

    premises = [await draft(n, engines[n - 1]) for n in range(1, PREMISE_COUNT + 1)]
    similarity = await premise_similarity([p.premise for p in premises], embedder)
    if not similarity.distinct and len(engines) > PREMISE_COUNT:
        # Once: the later premise of the closest pair again, around an engine not used yet.
        worst = similarity.pairs.index(similarity.closest)
        number = _pairs(len(premises))[worst][1] + 1
        again = await draft(number, engines[PREMISE_COUNT])
        tried = [again if p.number == number else p for p in premises]
        retried = await premise_similarity([p.premise for p in tried], embedder)
        if retried.closest < similarity.closest:
            premises, similarity = tried, retried
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


async def _genre_answer(
    model: StoryModel, state: StartState, words: str | None, choice: int | None, *, yes: bool
) -> str:
    """The genre the person's answer to the proposal takes.

    A number picks one of the genres offered; a genre the words name replaces the proposal;
    a bare yes takes the recommended one. Other words are read for a genre as a request
    is (`_direction`), and words that name none are refused, so nothing is guessed.
    """
    options = state.genre_options
    if choice is not None and choice <= len(options):
        return options[choice - 1]
    if words:
        named = named_genre(words)
        if named is not None:
            return named
        if yes:
            return options[0]
        stated = await _direction(model, words)
        if "genre" in stated.from_request:
            return stated.genre
    raise StoryError(
        "The person has not taken or named a genre yet: show them the genres proposed and "
        "pass what they say, word for word, as 'request' (a bare yes takes the first). "
        "Nothing was changed."
    )


#: The reason a cast reply could not be used, as the model is told it when asked again.
_CAST_RETRY: Mapping[str, str] = {
    "unreadable": "it was not one valid JSON object -- a key had an extra quotation mark "
    'before it (write {"name": ...}, never { " "name": ...}), a value held a double quote, '
    "or it was cut off. Keep every profile and appearance to one short sentence, and write "
    "quotation marks inside a value as ‘ ’",
    "empty": "it listed no character with a name. The story needs its main characters",
    "taken": "every character it listed is already in the story. Propose the characters "
    "this premise still needs, under other names",
}


class _Run:
    """One call's walk through the stages after the premises, saving after each."""

    def __init__(
        self,
        work: StoryWork,
        room_id: str,
        context: ToolContext,
        model: StoryModel,
        state: StartState,
        digest: str,
        structures: Mapping[str, StructureTemplate] | None = None,
    ) -> None:
        self.work = work
        self.room_id = room_id
        self.context = context
        self.model = model
        self.state = state
        self.digest = digest
        self.structures = dict(structures or {})
        #: Codex proposals read from the scenes this call wrote (`_grow`); `None` until
        #: this call writes a scene, so a call that writes none keeps the last call's.
        self.grown: list[str] | None = None

    def _save(self, **update: Any) -> None:
        if self.grown is not None:
            update.setdefault("growth_now", list(self.grown))
            update.setdefault(
                "codex_growth", list(dict.fromkeys([*self.state.codex_growth, *self.grown]))
            )
        self.state = self.state.model_copy(update=update)
        self.digest = self.work.write(
            START_FILE, _dump(self.state), room_id=self.room_id, expected_digest=self.digest
        )

    async def take_genre(
        self,
        genre: str,
        tables: Mapping[str, MuseTable],
        embedder: PremiseEmbedder | None,
    ) -> None:
        """The genre the person took: the brief, the story and the structure follow it,
        and then the premises are drawn in it."""
        state = self.state
        brief = state.brief.model_copy(update={"genre": genre})
        update: dict[str, Any] = {
            "brief": brief,
            "stage": "choose_premise",
            "genre_answered": True,
        }
        if state.structure_source != "request":
            picked = structure_for(state.request, genre, self.structures)
            update["structure"] = picked[0] if picked else None
            update["structure_source"] = picked[1] if picked else None
        library = StoryLibrary(self.context.require_workspace())
        library.set_genre(self.work.story_id, self.room_id, genre)
        self.state = await _premises(self.model, state.model_copy(update=update), tables, embedder)
        self._save()

    def _pending_cast(self) -> list[str]:
        status = {p.id: p.status for p, _ in self.work.proposals()[0]}
        return [i for i in self.state.cast_proposals if status.get(i) == "pending"]

    async def forward(
        self,
        *,
        choice: int | None,
        go_on: bool = False,
        rewrite: bool = False,
        yes: bool = False,
        words: str | None = None,
    ) -> list[str]:
        """Run stages until one stops for the person, or all are done; the stages run.

        The chapter waits while a proposed cast entry is still pending, unless the person
        asked for the story at once or said to go on (`go_on`): then it is written over
        the cast as proposed, and the entries it assumed are kept for the reply. A cast
        stage that proposes no character stops the flow before the outline, even when the
        story was asked for at once. Once the chapter is written, the scenes the check
        flagged are rewritten only when the person asks for it (`rewrite`). A bare yes
        (`yes`) takes the recommended premise; a number the person gave wins over it.

        At the outline stop, the person's `words` either settle the plot (`settles_plot`,
        or no words at all) or are feedback: feedback revises the outline and the flow
        stops there again. Settling the plot proposes its key events for the codex
        (`_lock_plot`), then plans the first chapter and shows the plan.

        Before each chapter, the first included, its plan waits (`review_chapter`): words that ask
        for it to be written (`writes_chapter`, or none) write it; other words revise the
        plan and the flow stops again, except in quick mode, where the revised plan is
        written in the same call. No call writes more than `CHAPTERS_PER_CALL` chapters.
        """
        done: list[str] = []
        while True:
            stage = self.state.stage
            if stage == "choose_premise":
                retry = self.state.chosen if self.state.cast_failed else None
                number = choice if choice is not None else retry
                if number is None and (self.state.quick or yes):
                    number = self.state.recommended
                if number is None:
                    raise StoryError(
                        "The person has not chosen a premise yet: ask which of the three, by "
                        "number, and pass it as 'choice'. Nothing was changed."
                    )
                self._save(chosen=number)
                await self._cast()
                if self.state.cast_failed:
                    return done
                done.append("cast")
                if not self.state.quick:
                    return done
            elif stage == "review_cast":
                await self._outline()
                done.append("outline")
                if not self.state.quick:
                    return done
            elif stage == "review_outline":
                if not self.state.plot_locked:
                    feedback = (words or "").strip()
                    if feedback and not (self.state.quick or settles_plot(feedback)):
                        await self._revise_outline(feedback)
                        done.append("outline_revised")
                        return done
                    await self._lock_plot()
                    done.append("plot")
                pending = self._pending_cast()
                if pending and not (go_on or self.state.quick):
                    self._save(cast_waiting=pending)
                    return done
                self._save(cast_waiting=[], cast_assumed=pending)
                # The first chapter is planned and shown before it is written, as every
                # later one is (§1.1 4). A story asked for at once writes it in this call.
                await self._propose_details(1)
                done.append("details")
                if not self.state.quick:
                    return done
                words = None  # spent on the outline: not feedback on the plan
            elif stage == "review_chapter":
                if "chapter" not in done:
                    flagged = _shown_findings(self.state)
                    if rewrite and flagged:
                        await self._rewrite()
                        done.append("rewrite")
                        return done
                    said = (words or "").strip()
                    if said and not writes_chapter(said):
                        await self._revise_details(said)
                        done.append("details_revised")
                        if not self.state.quick:  # the revised plan is shown first
                            return done
                number = self.state.details_chapter or self.state.chapter_written + 1
                await self._apply_details(number)
                if self._introduce(number):
                    done.append("introduced")
                await self._write_chapter(number, done)
                if not self._writes_on(done):
                    return done
            else:
                if rewrite and _shown_findings(self.state):
                    await self._rewrite()
                    done.append("rewrite")
                return done

    def _writes_on(self, done: Sequence[str]) -> bool:
        """Whether this call goes on to the next chapter without stopping for the person.

        Only a story asked for at once does, and only up to `CHAPTERS_PER_CALL` chapters
        in one call; the next call writes the next one.
        """
        return (
            self.state.quick
            and self.state.stage == "review_chapter"
            and not self.state.chapter_failed
            and done.count("chapter") < CHAPTERS_PER_CALL
        )

    async def _write_chapter(self, number: int, done: list[str]) -> None:
        """Chapter `number` written and checked; then the next chapter's plan, or done.

        A chapter that saved no scene leaves the stage where it was, for the person to
        try again. Otherwise the scenes are checked against the codex (the notes are
        shown, never acted on), and the next chapter of the outline is planned and shown
        (`review_chapter`); after the outline's last chapter the start is done.
        """
        await self._chapter(number)
        if self.state.chapter_failed:
            return
        done.append("chapter")
        await self._check(self.state.chapter_scenes)
        done.append("continuity")
        outline, _ = self.work.require_outline()
        if number < len(outline.chapters):
            await self._propose_details(number + 1)
            done.append("details")
        else:
            self._save(
                stage="done",
                details_chapter=None,
                chapter_details=[],
                details_new=[],
                details_feedback=[],
                details_revision_failed=False,
                details_from_outline=False,
            )

    # -- stage 2: the cast ----------------------------------------------------------------

    async def _cast(self) -> None:
        """Stage 2: characters and places, each proposed as a new codex entry.

        A reply that yields no character is asked again, told why (`_CAST_RETRY`): no
        readable JSON (`json_object` already reads past a stray quotation mark before a
        key, which lost qwen3:8b's whole cast for a premise that quotes a name), no named
        character, or only characters the story has already. After
        `MAX_CAST_TRIES` the flow stops and says why, rather than outlining and writing a
        story with no one in it.
        """
        state = self.state
        premise = state.premise()
        prompt = (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\n"
            f"List the main characters (at most {MAX_CHARACTERS}) and places (at most "
            f"{MAX_PLACES}) this story needs.\n"
            'Reply as JSON: {"characters": [{"name": ..., "profile": who they are and what '
            'they want, in one sentence, "appearance": how they look, in a few words, '
            '"gender": "female", "male" or "other", "status": where they stand when the '
            'story begins, from the premise: "alive", "missing", "dead" or "unknown", '
            '"family": [{"relative": another character\'s name, "is": what that relative is '
            'to this character, as in "my father": "father", "mother", "son", "daughter", '
            '"brother", "sister", "husband", "wife", "grandfather", "grandmother", '
            '"grandson" or "granddaughter"}]}], '
            '"places": [{"name": ..., "profile": what it is, in a sentence}]}'
            "\nGive family only where the premise implies it, and [] for none. A daughter "
            'lists her father as {"relative": his name, "is": "father"}.'
            "\nA character the premise says has vanished is missing, one it says has died is "
            "dead, and one whose fate the story sets out to learn is unknown. Inside a value, "
            "write quotation marks as ‘ ’, never as a double quote."
        )
        notes = list(state.notes)
        saved: list[Proposal] = []
        cast_lines: list[dict[str, Any]] = []
        asked = prompt
        reason: str | None = "empty"
        tries = 0
        for _ in range(MAX_CAST_TRIES):
            tries += 1
            reply = await self.model.complete(
                asked, system=_SYSTEM, temperature=0.7, max_tokens=2000
            )
            found = json_object(reply)
            if found is None:
                reason = "unreadable"
            else:
                wanted = _cast_params(found, notes)
                reason = await self._propose(wanted, saved, cast_lines, notes)
            if reason is None:
                break
            asked = (
                f"{prompt}\n\nYour last reply could not be used: {_CAST_RETRY[reason]}. "
                "Reply again with the JSON object only."
            )
        if tries > 1:
            notes.append(f"The cast was asked for {tries} times.")
        if reason is not None:
            notes.append(f"No character was proposed after {tries} tries ({reason}).")
            self._save(
                cast=cast_lines,
                cast_proposals=[p.id for p in saved],
                cast_failed=reason,
                notes=notes,
            )
            return
        self._save(
            stage="review_cast",
            cast=cast_lines,
            cast_proposals=[p.id for p in saved],
            cast_failed=None,
            notes=notes,
        )

    async def _propose(
        self,
        wanted: Sequence[NewEntryParams],
        saved: list[Proposal],
        cast_lines: list[dict[str, Any]],
        notes: list[str],
    ) -> str | None:
        """Propose `wanted`; `None` once the story has a proposed character, else why not."""
        pending = [p for p, _ in self.work.proposals()[0] if p.status == "pending"]
        taken = False
        for entry_params in wanted:
            try:
                draft = new_entry_draft(entry_params, room_id=self.room_id, context=self.context)
            except StoryError as exc:
                notes.append(str(exc))
                continue
            clash = new_entry_clash(
                self.work, draft.kind, draft.new_entry(), pending=[*pending, *saved]
            )
            if clash is not None:
                if entry_params.kind == "characters":
                    taken = True
                notes.append(f"'{entry_params.name}' was not proposed: {clash}.")
                continue
            proposal = self.work.add_proposal(draft, room_id=self.room_id)
            saved.append(proposal)
            cast_lines.append(
                {
                    "id": draft.entry_id,
                    "kind": draft.kind,
                    "name": entry_params.name,
                    "profile": entry_params.profile,
                    **({"status": entry_params.state["status"]} if entry_params.state else {}),
                }
            )
        if any(c["kind"] == "characters" for c in cast_lines):
            return None
        return "taken" if taken else "empty"

    # -- stage 3: the outline -------------------------------------------------------------

    def _template(self) -> StructureTemplate | None:
        state = self.state
        if state.structure and state.structure in self.structures:
            return self.structures[state.structure]
        picked = structure_for(state.request, state.brief.genre, self.structures)
        if picked is None:
            return None
        self._save(structure=picked[0], structure_source=picked[1])
        return self.structures[picked[0]]

    async def _outline(self) -> None:
        """Stage 3: the chosen structure's beats, filled with this premise and cast.

        One chapter per act of the structure (`structure_for`: the person's named one, else
        the genre's). Code decides what each chapter does and where the twist is revealed
        (`arc_for`). The model first says what the story turns on (`Spine`), and then
        outlines one chapter per call, in order (`_outline_by_chapter`). Asked for the
        whole outline in one reply and then again with its problems, qwen3:8b fixed one of
        four missing twists and settled conflicts early (#1991).
        """
        state = self.state
        premise = state.premise()
        cast_text = "\n".join(
            f"- {c['name']} ({c['kind']}): {c.get('profile', '')}" for c in state.cast
        )
        template = self._template()
        head = (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\nCast and places:\n{cast_text}\n\n"
        )
        kept_acts = template.acts[:MAX_CHAPTERS] if template is not None else ()
        acts = [act.id for act in kept_acts]
        arc = arc_for([[beat.id for beat in act.beats] for act in kept_acts])
        check: dict[str, Any] | None = None
        spine: Spine | None = None
        if template is None or arc is None:
            found = await self._outline_at_once(head, template)
        else:
            found, spine, check = await self._outline_by_chapter(head, template, arc)
        outline = outline_from(found, state.cast, acts=acts)
        notes = list(state.notes)
        if not outline.chapters:
            notes.append("The outline came back empty; one chapter was made from the premise.")
            outline = outline_from(
                {
                    "chapters": [
                        {
                            "title": premise.title,
                            "scenes": [{"title": premise.title, "summary": premise.premise}],
                        }
                    ]
                },
                state.cast,
                acts=acts,
            )
        elif acts and len(outline.chapters) < len(acts):
            notes.append(
                f"The outline covers {len(outline.chapters)} of the structure's {len(acts)} acts."
            )
        current = self.work.outline()
        self.work.save_outline(
            outline, room_id=self.room_id, expected_digest=current[1] if current else None
        )
        self._save(
            stage="review_outline",
            outline_saved=True,
            central_conflict=_text(found.get("central_conflict")) or None,
            twist=_text(found.get("twist")) or None,
            spine=spine.record() if spine else None,
            arc=arc.record() if arc else None,
            stakes=[_text(c.get("stakes")) for c in _chapters(found)][: len(acts) or None],
            outline_check=check,
            notes=notes,
        )

    async def _outline_at_once(
        self, head: str, template: StructureTemplate | None
    ) -> dict[str, Any]:
        """The whole outline in one reply: with no structure, or one under three acts."""
        plan = ""
        if template is not None:
            plan = f" on the structure '{template.title}', one chapter per act, in order:\n" + (
                "\n".join(
                    f"Chapter {number} (act '{act.title}'):\n"
                    + "\n".join(f"  - {beat.title}: {beat.purpose}" for beat in act.beats)
                    for number, act in enumerate(template.acts[:MAX_CHAPTERS], start=1)
                )
            )
        prompt = (
            f"{head}Outline the story in 3 to {MAX_CHAPTERS} chapters{plan}\n"
            f"Each chapter has {SCENES_PER_CHAPTER} scenes. {_SCENE_SHAPE}\n"
            'Reply as JSON: {"chapters": [{"title": ..., "scenes": [{"title": ..., '
            '"summary": ..., "beats": [...], "characters": [names], "places": [names]}]}]}'
        )
        return await _ask_json(self.model, prompt, temperature=0.6, max_tokens=3500) or {}

    async def _outline_by_chapter(
        self, head: str, template: StructureTemplate, arc: Arc
    ) -> tuple[dict[str, Any], Spine, dict[str, Any]]:
        """The outline one chapter per call, each checked and asked for once more alone.

        First the spine (conflict, twist, clue, resolution), held by code; then, act by
        act, a call told the chapters so far, its act's beats, its function, what it must
        not do, and only its part of the spine (`Spine.told`: no twist before the twist
        chapter). Each chapter is checked as it comes (`chapter_problems`); with a problem,
        that chapter alone is asked for again with the reasons, and the draft with fewer
        problems is kept. A chapter that comes back with no scene ends the outline there.
        """
        kept_acts = template.acts[:MAX_CHAPTERS]
        found = (
            await _ask_json(
                self.model,
                head + _spine_prompt(template.title, len(kept_acts), arc),
                temperature=0.6,
                max_tokens=800,
            )
            or {}
        )
        spine = Spine(
            conflict=_text(found.get("central_conflict")),
            twist=_text(found.get("twist")),
            clue=_text(found.get("clue")),
            resolution=_text(found.get("resolution")),
        )
        chapters: list[dict[str, Any]] = []
        record: list[dict[str, Any]] = []
        asked = 0
        for number, act in enumerate(kept_acts, start=1):
            so_far = "\n\n".join(
                f"Chapter {n}:\n{chapter_text(c)}" for n, c in enumerate(chapters, start=1)
            )
            must_not: list[str] = []
            if number < arc.twist_chapter:
                must_not.append(
                    "reveal the twist: no character learns, says, rightly guesses or confirms it"
                )
            if number < arc.climax_chapter:
                must_not.append(
                    "settle the conflict: no win, loss, forgiveness, reconciliation, "
                    "acceptance or final decision; the main character ends it worse off "
                    "than at its start"
                )
            prompt = (
                f"{head}The story is built on the structure '{template.title}', one chapter "
                f"per act, {len(kept_acts)} chapters. What it turns on (the writer's plan, "
                f"which the reader does not know yet):\n{spine.told(arc, number)}\n\n"
                + (f"The chapters so far:\n{so_far}\n\n" if so_far else "")
                + f"{CHAPTER_PROMPT_MARK} {number} of {len(kept_acts)} (act '{act.title}'). "
                "It covers these beats, in order, with events of THIS premise and cast:\n"
                + "\n".join(f"  - {beat.title}: {beat.purpose}" for beat in act.beats)
                + f"\nThis chapter must {arc.does(number)}.\n"
                + (f"It must not {'; nor '.join(must_not)}.\n" if must_not else "")
                + "It does not restate the premise or an earlier chapter's events: it starts "
                "where the chapter before ended.\n"
                f"It has {scenes_for(arc, number)} scenes. {_SCENE_SHAPE}\n"
                'Reply as JSON: {"title": ..., "stakes": what can be lost at its end, '
                '"scenes": [{"title": ..., "summary": ..., "beats": [...], '
                '"characters": [names], "places": [names]}]}'
            )
            previous = chapter_text(chapters[-1]) if chapters else None
            reply = await _ask_json(self.model, prompt, temperature=0.6, max_tokens=1500)
            asked += 1
            if not _has_scenes(reply):
                break
            chapter = cast(dict[str, Any], reply)
            first = await chapter_problems(
                self.model, arc, number, chapter_text(chapter), spine=spine, previous=previous
            )
            entry: dict[str, Any] = {"chapter": number, "first": first, "asked": 1, "final": first}
            if first:
                again = (
                    f"{prompt}\n\nYour last draft of this chapter broke the story's shape:\n"
                    + "\n".join(f"- {r}" for r in problem_reasons(arc, first))
                    + "\nWrite this chapter again, fixing these."
                )
                second = await _ask_json(self.model, again, temperature=0.6, max_tokens=1500)
                asked += 1
                entry["asked"] = 2
                if _has_scenes(second):
                    retried = cast(dict[str, Any], second)
                    later = await chapter_problems(
                        self.model,
                        arc,
                        number,
                        chapter_text(retried),
                        spine=spine,
                        previous=previous,
                    )
                    entry["second"] = later
                    if len(later) < len(first):
                        chapter = retried
                        entry["final"] = later
            # The chapter's weight decides its scenes, not the reply's length.
            chapter = {
                **chapter,
                "scenes": _items(chapter.get("scenes"))[: scenes_for(arc, number)],
            }
            chapters.append(chapter)
            record.append(entry)
        head_problems: list[dict[str, Any]] = []
        if not spine.twist:
            head_problems.append({"kind": "no_twist", "chapter": arc.twist_chapter})
        if len(chapters) < len(kept_acts):
            head_problems.append({"kind": "missing_chapters", "chapter": len(chapters) + 1})
        check = {
            "first": head_problems + [p for e in record for p in e["first"]],
            "final": head_problems + [p for e in record for p in e["final"]],
            "asked": asked,
            "chapters": record,
        }
        outline = {
            "central_conflict": spine.conflict,
            "twist": spine.twist,
            "chapters": chapters,
        }
        return outline, spine, check

    # -- stage 3b: feedback on the outline, and settling the plot --------------------------

    def _acts(self) -> list[str]:
        template = self._template()
        return [act.id for act in template.acts[:MAX_CHAPTERS]] if template is not None else []

    async def _revise_outline(self, feedback: str) -> None:
        """The outline revised by the person's `feedback`: one call, then the code's checks.

        The model is given the outline as it stands and the feedback, and asked to change
        only what the feedback asks. The reply goes through `outline_from` as the first
        outline did, and each chapter is checked against the arc (`chapter_problems`);
        what the check finds is shown, not fixed behind the person's back. A chapter the
        person asked to remove is not reported missing. An unreadable reply keeps the
        outline as it was, and the reply says so.
        """
        state = self.state
        current = self.work.outline()
        names = {str(c.get("id")): str(c.get("name", "")) for c in state.cast}
        shown = _outline_json(current[0] if current else Outline(), names, state.stakes)
        premise = state.premise()
        cast_text = "\n".join(
            f"- {c['name']} ({c['kind']}): {c.get('profile', '')}" for c in state.cast
        )
        prompt = (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\nCast and places:\n{cast_text}\n\n"
            f"Central conflict: {state.central_conflict or ''}\n"
            f"Twist: {state.twist or ''}\n\n"
            f"{REVISE_MARK}. The outline as it stands, as JSON:\n{shown}\n\n"
            f"The person's feedback, word for word:\n{feedback}\n\n"
            "Change what the feedback asks for and keep everything else as it is. If it "
            "asks to remove a chapter, leave that chapter out. If it changes the conflict "
            "or the twist, give the new one.\n"
            "Keep each chapter's number of scenes unless the feedback asks otherwise, and "
            f"at most {MAX_SCENES_PER_CHAPTER}. {_SCENE_SHAPE}\n"
            'Reply as JSON: {"central_conflict": ..., "twist": ..., "chapters": '
            '[{"title": ..., "stakes": ..., "scenes": [{"title": ..., "summary": ..., '
            '"beats": [...], "characters": [names], "places": [names]}]}]}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.4, max_tokens=3500) or {}
        said = [*state.outline_feedback, feedback]
        notes = list(state.notes)
        acts = self._acts()
        raw = _chapters(found)
        outline = outline_from(found, state.cast, acts=acts if len(raw) == len(acts) else ())
        if not outline.chapters:
            notes.append("The revised outline could not be read; the outline was kept.")
            self._save(outline_feedback=said, revision_failed=True, notes=notes)
            return
        if acts and len(outline.chapters) != len(acts):
            notes.append(
                f"After the feedback the outline has {len(outline.chapters)} chapters for "
                f"the structure's {len(acts)} acts, so chapters are not tied to acts."
            )
        conflict = _text(found.get("central_conflict")) or state.central_conflict or ""
        twist = _text(found.get("twist")) or state.twist or ""
        stakes = [_text(c.get("stakes")) for c in raw][: len(outline.chapters)]
        check = state.outline_check
        arc = _arc_of(state.arc)
        if arc is not None:
            spine = Spine(
                conflict=conflict,
                twist=twist,
                clue=(state.spine or {}).get("clue", ""),
                resolution=(state.spine or {}).get("resolution", ""),
            )
            problems: list[dict[str, Any]] = (
                [] if twist else [{"kind": "no_twist", "chapter": arc.twist_chapter}]
            )
            previous: str | None = None
            for number, chapter in enumerate(_chapter_dicts(outline, stakes), start=1):
                text = chapter_text(chapter)
                problems += await chapter_problems(
                    self.model, arc, number, text, spine=spine, previous=previous
                )
                previous = text
            check = {**(state.outline_check or {}), "final": problems, "revised": len(said)}
        current = self.work.outline()
        self.work.save_outline(
            outline, room_id=self.room_id, expected_digest=current[1] if current else None
        )
        self._save(
            outline_feedback=said,
            revision_failed=False,
            central_conflict=conflict or None,
            twist=twist or None,
            stakes=stakes,
            outline_check=check,
            notes=notes,
        )

    async def _lock_plot(self) -> None:
        """Settle the plot: record it, and propose its key events for the codex.

        Code proposes the central conflict and the twist as threads, planted and paid off
        at the scenes the arc gives them. One call reads the settled outline for the
        rest: key events, secrets and relationships between characters that are not
        family (threads, with the characters they involve) and the objects the plot
        turns on (items). Each is a new-entry proposal, pending like the cast: nothing is
        approved here (§5.3). An unreadable reply leaves the conflict and the twist.
        """
        state = self.state
        current = self.work.outline()
        outline = current[0] if current else Outline()
        notes = list(state.notes)
        firsts = [c.scenes[0].id for c in outline.chapters if c.scenes]

        def scene_of(number: object) -> str | None:
            if isinstance(number, bool) or not isinstance(number, int):
                return None
            return firsts[number - 1] if 1 <= number <= len(firsts) else None

        arc = _arc_of(state.arc)
        spine = state.spine or {}
        wanted: list[tuple[NewEntryParams, dict[str, str | None]]] = []
        if state.central_conflict:
            climax = arc.climax_chapter if arc else len(firsts)
            wanted.append(
                (
                    NewEntryParams(
                        kind="threads",
                        id="central_conflict",
                        name="중심 갈등",
                        profile=state.central_conflict,
                    ),
                    {
                        "planted_in": scene_of(1),
                        "pay_off_by": scene_of(min(climax, len(firsts))),
                        "notes": spine.get("resolution") or None,
                    },
                )
            )
        if state.twist:
            reveal = arc.twist_chapter if arc else len(firsts)
            wanted.append(
                (
                    NewEntryParams(kind="threads", id="twist", name="반전", profile=state.twist),
                    {
                        "pay_off_by": scene_of(min(reveal, len(firsts))),
                        "notes": spine.get("clue") or None,
                    },
                )
            )
        dropped = 0
        found = await _ask_json(
            self.model, _kg_prompt(state, outline), temperature=0.3, max_tokens=1500
        )
        if found is None:
            notes.append("The settled outline's key events could not be read.")
        else:
            read, abstract = _kg_params(found, state.cast, scene_of)
            wanted += read
            dropped = len(abstract)
            if abstract:
                notes.append(
                    f"{dropped} item(s) named an idea, not an object, and were not proposed: "
                    + ", ".join(abstract)
                    + "."
                )
        pending = [p for p, _ in self.work.proposals()[0] if p.status == "pending"]
        saved: list[Proposal] = []
        added: list[dict[str, str]] = []
        for params, fields in wanted:
            try:
                draft = _with_entry_fields(
                    new_entry_draft(params, room_id=self.room_id, context=self.context), fields
                )
            except StoryError as exc:
                notes.append(str(exc))
                continue
            clash = new_entry_clash(
                self.work, draft.kind, draft.new_entry(), pending=[*pending, *saved]
            )
            if clash is not None:
                notes.append(f"'{params.name}' was not proposed: {clash}.")
                continue
            proposal = self.work.add_proposal(draft, room_id=self.room_id)
            saved.append(proposal)
            added.append(
                {
                    "proposal": proposal.id,
                    "kind": proposal.kind,
                    "entry_id": proposal.entry_id,
                    "name": params.name,
                }
            )
        self._save(plot_locked=True, plot_kg=added, plot_kg_dropped=dropped, notes=notes)

    # -- stage 4: the chapter -------------------------------------------------------------

    def _cast_text(self) -> str:
        """The cast the chapter is written over: approved, or assumed; never rejected."""
        status = {p.id: p.status for p, _ in self.work.proposals()[0]}
        rejected = {i for i, s in status.items() if s == "rejected"}
        return "\n".join(
            f"- {c['name']} ({c['kind']}): {c.get('profile', '')}"
            for c, proposal in zip(self.state.cast, self.state.cast_proposals, strict=False)
            if proposal not in rejected
        )

    def _scene_prompt(
        self, outline: Outline, scene: Scene, previous: str | None, cast_text: str
    ) -> str:
        state = self.state
        meta = self.work.record().model_dump(mode="json", exclude={"lease"})
        bundle = scene_context(
            outline, scene.id, self.work.codex(), previous_text=previous, story=meta, request=None
        )
        return (
            f"Direction:\n{_brief_lines(state.brief)}\n\n"
            f"Premise: {state.premise().premise}\n\nCast and places:\n"
            f"{cast_text}\n\nWhat to know before writing this scene (JSON):\n"
            f"{json.dumps(bundle, ensure_ascii=False)[:6000]}\n\n"
            f"Write scene '{scene.title}' in full as prose: {scene.summary} "
            f"Beats, in order: {'; '.join(scene.beats)}. Write only the scene's text, "
            "with no headings and no notes about the writing."
        )

    async def _draft(self, prompt: str) -> tuple[str, str | None]:
        """A scene's text and `None`, or the last draft and why the prose check refused it."""
        text = ""
        refused: str | None = "empty"
        asked = prompt
        for _ in range(MAX_DRAFTS):
            text = (
                await self.model.complete(asked, system=_SYSTEM, temperature=0.8, max_tokens=2500)
            ).strip()
            refused = prose_refusal(text) if text else "empty"
            if refused is None:
                break
            asked = (
                f"{prompt}\n\nYour last draft was not saved. Fix this and write the "
                f"whole scene again:\n{refused if text else 'The draft was empty.'}"
            )
        return text, refused

    def _last_text(self, outline: Outline, number: int) -> str | None:
        """The text of the last written scene of chapter `number`, or `None`."""
        if not 1 <= number <= len(outline.chapters):
            return None
        for scene in reversed(outline.chapters[number - 1].scenes):
            found = self.work.manuscript(scene.id)
            if found is not None:
                return found.text
        return None

    async def _grow(self, scene_id: str, text: str, notes: list[str]) -> None:
        """The scene just saved, read for what it adds to the codex (§1.1 row 7).

        The same reading `story_manuscript write` does (`grow_scene`). The additions are
        pending proposals, never applied here; a "yes" that carries the flow on leaves
        them pending. A reading that fails proposes nothing and stops nothing: the scene
        stays written, and the note says so to the model, not to the person.
        """
        so_far = self.grown or []
        self.grown = so_far
        grown = await grow_scene(
            self.work,
            model=self.model,
            scene_id=scene_id,
            text=text,
            room_id=self.room_id,
            agent_id=self.context.agent_id,
        )
        if grown is None:
            notes.append(
                f"Scene '{scene_id}' was saved, but it could not be read for what it adds "
                "to the codex."
            )
            return
        self.grown = [*so_far, *(p.id for p in grown.proposed)]

    async def _chapter(self, number: int = 1) -> None:
        """Stage 4: chapter `number`'s scenes, written through the manuscript path.

        The first scene is written after the end of the chapter before, so the story
        runs on from where it stopped.
        """
        state = self.state
        outline, _ = self.work.require_outline()
        chapter = outline.chapters[number - 1]
        notes = list(state.notes)
        written = list(state.scenes_written)
        previous: str | None = self._last_text(outline, number - 1)
        cast_text = self._cast_text()
        for scene in chapter.scenes[:MAX_SCENES_WRITTEN]:
            if scene.id in written:
                found = self.work.manuscript(scene.id)
                previous = found.text if found else previous
                continue
            text, refused = await self._draft(
                self._scene_prompt(outline, scene, previous, cast_text)
            )
            if refused is not None:
                notes.append(
                    f"Scene '{scene.id}' was not written after {MAX_DRAFTS} drafts: {refused}"
                )
                break
            self.work.write_scene(
                scene.id,
                text,
                room_id=self.room_id,
                expected_digest=None,
                agent_id=self.context.agent_id,
                turn_index=self.context.turn_index,
            )
            try:
                self.work.note_session(self.room_id, scene_written=scene.id)
            except (StoryError, StoryFileError):
                notes.append(f"Scene '{scene.id}' was written, but the session was not noted.")
            written.append(scene.id)
            await self._grow(scene.id, text, notes)
            previous = text
        this_chapter = [s.id for s in chapter.scenes if s.id in written]
        if not this_chapter:
            self._save(scenes_written=written, notes=notes, chapter_failed=True)
            return
        self._save(
            scenes_written=written,
            notes=notes,
            chapter_failed=False,
            chapter_written=number,
            chapter_scenes=this_chapter,
        )

    # -- stage 4b: each later chapter's plan, shown before it is written -----------------

    def _details_head(self, number: int, outline: Outline) -> str:
        """What a chapter's plan is made from: the premise, the cast, the story's plan for
        this chapter, the chapter as outlined, and how the chapter before it ended."""
        state = self.state
        names = {str(c.get("id")): str(c.get("name", "")) for c in state.cast}
        chapter = outline.chapters[number - 1]
        outlined = _outline_json(Outline(chapters=[chapter]), names, [])
        arc = _arc_of(state.arc)
        plan = ""
        if arc is not None and number <= len(arc.functions):
            spine = Spine(
                conflict=state.central_conflict or "",
                twist=state.twist or "",
                clue=(state.spine or {}).get("clue", ""),
                resolution=(state.spine or {}).get("resolution", ""),
            )
            plan = (
                f"What the story turns on (the writer's plan, which the reader does not know "
                f"yet):\n{spine.told(arc, number)}\nThis chapter must {arc.does(number)}.\n"
            )
        elif state.central_conflict:
            plan = f"Central conflict: {state.central_conflict}\n"
        stakes = state.stakes[number - 1] if number <= len(state.stakes) else ""
        if stakes:
            plan += f"What can be lost at its end: {stakes}\n"
        tracked = ", ".join(k["name"] for k in state.plot_kg)
        before = self._last_text(outline, number - 1)
        ending = (
            f"How chapter {number - 1} ended, in its last lines:\n{before[-1200:]}\n\n"
            if before
            else ""
        )
        premise = state.premise()
        return (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\nCast and places:\n"
            f"{self._cast_text()}\n"
            + (f"The story bible also tracks: {tracked}\n" if tracked else "")
            + f"\n{plan}\nChapter {number} of {len(outline.chapters)} as the settled outline "
            f"has it, as JSON:\n{outlined}\n\n{ending}"
        )

    async def _propose_details(self, number: int) -> None:
        """The plan of chapter `number`, one call, shown to the person before it is written.

        The plan keeps the settled outline's events for the chapter and makes its scenes
        and beats concrete, starting where the chapter before ended. A reply that cannot
        be read shows the outline's scenes for the chapter as the plan, and says so.
        """
        outline, _ = self.work.require_outline()
        count = scenes_for(_arc_of(self.state.arc), number)
        start = (
            "The first scene opens the story."
            if number == 1
            else "The first scene starts where the chapter before ended."
        )
        prompt = (
            f"{self._details_head(number, outline)}"
            f"{DETAILS_MARK} {number}, before it is written. Keep the outline's events for "
            "this chapter, in order, and make each scene concrete: who does what, where, "
            f"and what changes. {start} Do not settle or reveal anything the plan above "
            "says this chapter must not.\n"
            f"It has {count} scenes. {_SCENE_SHAPE}\n{_NEW_CHARACTERS_ASK}"
            'Reply as JSON: {"scenes": [{"title": ..., "summary": ..., "beats": [...], '
            f'"characters": [names], "places": [names]}}], {_NEW_CHARACTERS_SHAPE}}}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.6, max_tokens=1500)
        scenes = _detail_scenes(found, self.state.cast, count)
        new = _new_characters(found, self._known_characters()) if scenes else []
        notes = list(self.state.notes)
        from_outline = not scenes
        if from_outline:
            notes.append(
                f"The plan of chapter {number} could not be read; the outline's scenes are shown."
            )
            names = {str(c.get("id")): str(c.get("name", "")) for c in self.state.cast}
            shown = json.loads(
                _outline_json(Outline(chapters=[outline.chapters[number - 1]]), names, [])
            )
            scenes = cast(list[dict[str, Any]], shown["chapters"][0]["scenes"])
        self._save(
            stage="review_chapter",
            details_chapter=number,
            chapter_details=scenes,
            details_new=new,
            details_feedback=[],
            details_revision_failed=False,
            details_from_outline=from_outline,
            notes=notes,
        )

    async def _revise_details(self, feedback: str) -> None:
        """The chapter's plan revised by the person's `feedback`, in one call.

        Only what the feedback asks is changed. An unreadable reply keeps the plan as it
        was, and the reply says so.
        """
        state = self.state
        number = state.details_chapter or state.chapter_written + 1
        outline, _ = self.work.require_outline()
        shown_new = [
            {k: c[k] for k in ("name", "profile", "gender")}
            | {"family": [{"relative": f["relative"], "is": f["is"]} for f in c["family"]]}
            for c in state.details_new
        ]
        plan = json.dumps(
            {"scenes": state.chapter_details, "new_characters": shown_new},
            ensure_ascii=False,
            indent=1,
        )
        count = scenes_for(_arc_of(state.arc), number)
        prompt = (
            f"{self._details_head(number, outline)}"
            f"{DETAILS_REVISE_MARK} {number} as the person asks. The plan as it stands, as "
            f"JSON:\n{plan}\n\nThe person's feedback, word for word:\n{feedback}\n\n"
            "Change what the feedback asks for and keep everything else as it is. If it "
            "asks to remove a scene, leave that scene out.\n"
            f"It has at most {count} scenes. {_SCENE_SHAPE}\n{_NEW_CHARACTERS_ASK}"
            'Reply as JSON: {"scenes": [{"title": ..., "summary": ..., "beats": [...], '
            f'"characters": [names], "places": [names]}}], {_NEW_CHARACTERS_SHAPE}}}'
        )
        found = await _ask_json(self.model, prompt, temperature=0.4, max_tokens=1500)
        scenes = _detail_scenes(found, state.cast, count)
        said = [*state.details_feedback, feedback]
        if not scenes:
            notes = [*state.notes, f"The revised plan of chapter {number} could not be read."]
            self._save(details_feedback=said, details_revision_failed=True, notes=notes)
            return
        self._save(
            details_feedback=said,
            details_revision_failed=False,
            details_from_outline=False,
            chapter_details=scenes,
            details_new=_new_characters(found, self._known_characters()),
        )

    async def _apply_details(self, number: int) -> None:
        """The chapter's plan put into the outline, as its scenes, before it is written.

        The chapter keeps its id, title and act; its scenes get ids by code, as the
        outline's did. With no plan, the outline is left as it is.
        """
        details = self.state.chapter_details
        if not details:
            return
        outline, digest = self.work.require_outline()
        if not 1 <= number <= len(outline.chapters):
            return
        kept = outline.chapters[number - 1]
        scenes = _scene_dicts(details, kept.id, _cast_ids(self.state.cast))
        if not scenes:
            return
        chapter = Chapter.model_validate(
            {"id": kept.id, "title": kept.title, "act": kept.act, "scenes": scenes}
        )
        chapters = list(outline.chapters)
        chapters[number - 1] = chapter
        self.work.save_outline(
            Outline(chapters=chapters), room_id=self.room_id, expected_digest=digest
        )

    def _known_characters(self) -> dict[str, tuple[str, str]]:
        """Every character the story has, by folded name or alias: (name, id). The cast
        (proposed or approved), the codex's characters, and those an earlier plan
        introduced (proposed), so a later plan neither brings them in again nor loses
        them as a relative."""
        known: dict[str, tuple[str, str]] = {}
        for item in self.work.codex().items:
            if item.kind == "characters":
                for name in (item.entry.name, *item.entry.aliases):
                    if name:
                        known[name.casefold()] = (item.entry.name, item.entry.id)
        for c in self.state.cast:
            if c.get("kind") == "characters" and c.get("name"):
                known[str(c["name"]).casefold()] = (str(c["name"]), str(c.get("id")))
        for k in self.state.introduced:
            if k.get("change") == "new_entry":
                known.setdefault(str(k["name"]).casefold(), (str(k["name"]), str(k["entry_id"])))
        return known

    def _introduce(self, number: int) -> list[dict[str, Any]]:
        """The new characters of chapter `number`'s accepted plan, proposed with their family.

        Each is a new-entry proposal whose `relations:` hold the role the new character has
        to the relative (the inverse of what the relative is to it), as `_cast_relations`
        writes the cast's. A relative the codex has (approved) also gets a progression
        proposal at the chapter's first scene, a relation to the new character: the kinship
        on both sides. A relative that is itself still a pending proposal has no entry to
        progress yet, so the relation is kept on the new character's side only. Nothing
        is approved here; a name the story already has, or has pending, is not proposed.
        """
        if not self.state.details_new:
            return []
        outline, _ = self.work.require_outline()
        if not 1 <= number <= len(outline.chapters) or not outline.chapters[number - 1].scenes:
            return []
        at = outline.chapters[number - 1].scenes[0].id
        pending = [p for p, _ in self.work.proposals()[0] if p.status == "pending"]
        notes = list(self.state.notes)
        added: list[dict[str, Any]] = []
        for person in self.state.details_new:
            family = cast(list[dict[str, str]], person.get("family") or [])
            facts: dict[str, Any] = {"status": "alive"}
            relations = {f["relative_id"]: INVERSE_ROLE[f["role"]] for f in family}
            gender = str(person.get("gender") or "")
            try:
                draft = new_entry_draft(
                    NewEntryParams(
                        kind="characters",
                        name=str(person["name"]),
                        profile=str(person.get("profile") or ""),
                        state=facts,
                        relations=relations,
                        gender=cast(Any, gender) if gender in ("female", "male") else None,
                    ),
                    room_id=self.room_id,
                    context=self.context,
                )
            except (StoryError, ValidationError) as exc:
                notes.append(f"'{person['name']}' was not proposed: {exc}")
                continue
            clash = new_entry_clash(self.work, "characters", draft.new_entry(), pending=pending)
            if clash is not None:
                notes.append(f"'{person['name']}' was not proposed: {clash}.")
                continue
            proposal = self.work.add_proposal(draft, room_id=self.room_id)
            pending.append(proposal)
            added.append(
                {
                    "proposal": proposal.id,
                    "kind": "characters",
                    "entry_id": proposal.entry_id,
                    "name": str(person["name"]),
                    "chapter": number,
                    "change": "new_entry",
                }
            )
            for f in family:
                loaded = self.work.entry("characters", f["relative_id"])
                if loaded is None:  # a relative still pending: this side only
                    continue
                try:
                    back = Proposal.model_validate(
                        {
                            "id": "p000",
                            "kind": "characters",
                            "entry_id": f["relative_id"],
                            "change": {
                                "progression": {
                                    "at": at,
                                    "relations": {proposal.entry_id: f["role"]},
                                }
                            },
                            "proposed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                            "room_id": self.room_id,
                            "agent_id": self.context.agent_id,
                            "entry_digest": loaded[1],
                        }
                    )
                except (ValidationError, ValueError):
                    continue
                saved = self.work.add_proposal(back, room_id=self.room_id)
                added.append(
                    {
                        "proposal": saved.id,
                        "kind": "characters",
                        "entry_id": f["relative_id"],
                        "name": f["relative"],
                        "chapter": number,
                        "change": "relation",
                    }
                )
        self._save(introduced=[*self.state.introduced, *added], details_new=[], notes=notes)
        return added

    # -- after writing: continuity --------------------------------------------------------

    def _checked_codex(self) -> CodexIndex:
        """The codex the chapter was written over: with the pending entries it assumed.

        Those are the cast it went on with and the items this start proposed (the settled
        plot's and those written scenes added), so who holds an item is checked before a
        person approves it.
        """
        items = {k["proposal"] for k in self.state.plot_kg if k.get("kind") == "items"}
        items |= {*self.state.codex_growth, *(self.grown or ())}
        introduced = {k["proposal"] for k in self.state.introduced}
        assumed: list[CodexItem] = []
        for proposal, _ in self.work.proposals()[0]:
            wanted = (
                proposal.id in self.state.cast_assumed
                or proposal.id in introduced
                or (proposal.id in items and proposal.kind == "items")
            )
            if wanted and proposal.status == "pending":
                try:
                    assumed.append(CodexItem(kind=proposal.kind, entry=proposal.new_entry()))
                except ValueError:
                    continue
        return codex_with(self.work.codex(), assumed)

    async def _check(self, scene_ids: Sequence[str]) -> None:
        """Each scene read for facts and checked against the codex, then held to its plan
        and to the settled plot (`_held_to_plan`); nothing is rewritten.

        The path of the eval's graph grader (`uclone_x.story.continuity`): the model reads
        the facts with their quotes, and code checks them. A scene whose reading has no
        facts to read is listed as not checked, never as consistent.
        """
        outline, _ = self.work.require_outline()
        codex = self._checked_codex()
        axioms, are_defaults = self.work.axioms()
        kept = [f for f in self.state.continuity if f.get("scene_id") not in scene_ids]
        checked = [s for s in self.state.continuity_checked if s not in scene_ids]
        unread = [s for s in self.state.continuity_unread if s not in scene_ids]
        not_held = [s for s in self.state.plan_unread if s not in scene_ids]
        order = {
            scene.id: n for n, scene in enumerate(s for c in outline.chapters for s in c.scenes)
        }
        deaths = {
            who: died
            for who, died in self.state.written_deaths.items()
            if died.get("scene_id") not in scene_ids
        }
        for scene_id in scene_ids:
            found = self.work.manuscript(scene_id)
            if found is None:
                continue
            reply = await self.model.complete(
                continuity_prompt(outline, codex, scene_id, found.text),
                system="You extract facts from fiction. Reply with JSON only.",
                temperature=0.0,
                max_tokens=1500,
            )
            facts: list[ExtractedFact] | None
            try:
                facts = parse_extraction(reply)
            except ValueError:
                try:
                    facts = salvage_extraction(reply)
                except ValueError:
                    facts = None
            if facts is None:
                unread.append(scene_id)
            else:
                findings: list[Finding] = findings_from(
                    story_id=self.work.story_id,
                    outline=outline,
                    codex=codex,
                    axioms=axioms,
                    axioms_are_defaults=are_defaults,
                    scene_id=scene_id,
                    text=found.text,
                    facts=facts,
                )
                kept += [f.record() for f in findings]
                checked.append(scene_id)
                at = order.get(scene_id, len(order))
                earlier = {w: d for w, d in deaths.items() if order.get(d["scene_id"], at) < at}
                located = outline.find(scene_id)
                back, ends = came_back(
                    codex,
                    scene_id=scene_id,
                    scene_title=located[1].title if located else scene_id,
                    text=found.text,
                    facts=facts,
                    deaths=earlier,
                )
                kept += [f.record() for f in back]
                for who, died in ends.items():
                    if died is None:
                        deaths.pop(who, None)
                    else:
                        deaths[who] = died
            held, read = await self._held_to_plan(outline, codex, scene_id, found.text)
            kept += [f.record() for f in held]
            if not read:
                not_held.append(scene_id)
        self._save(
            continuity=kept,
            continuity_checked=checked,
            continuity_unread=unread,
            plan_unread=not_held,
            written_deaths=deaths,
        )

    async def _held_to_plan(
        self, outline: Outline, codex: CodexIndex, scene_id: str, text: str
    ) -> tuple[list[Finding], bool]:
        """What scene `scene_id` leaves out of its plan, adds to it, or tells too early.

        Code finds a planned character the scene never names (`absent_cast`). One call
        (temperature 0) reads the planned beats, the twist while it must stay hidden, and
        unplanned turns, and code keeps only what the reply shows (`plan_findings`). A
        scene with no beats and no twist to keep is not sent. The flag is false when the
        reply could not be read: code's findings are kept, and the scene is listed as not
        read against its plan, never as keeping it.
        """
        located = outline.find(scene_id)
        if located is None:
            return [], True
        chapter, scene = located
        number = next(n for n, c in enumerate(outline.chapters, start=1) if c.id == chapter.id)
        arc = _arc_of(self.state.arc)
        where: dict[str, Any] = {
            "chapter": number,
            "twist": self.state.twist,
            "twist_chapter": arc.twist_chapter if arc else None,
        }
        findings = absent_cast(codex, scene, text)
        if not asks_model(scene, **where):
            return findings, True
        reply = await self.model.complete(
            plan_prompt(scene, text, **where),
            system="You compare fiction with its plan. Reply with JSON only.",
            temperature=0.0,
            max_tokens=1000,
        )
        try:
            return findings + plan_findings(scene, text, reply, **where), True
        except ValueError:
            return findings, False

    async def _rewrite(self) -> None:
        """The flagged scenes written again, told what disagreed, then checked again.

        Run only when the person asks for it after reading the findings. The scene
        replaced is kept in the manuscript's history by `write_scene`.
        """
        outline, _ = self.work.require_outline()
        cast_text = self._cast_text()
        notes = list(self.state.notes)
        shown = _shown_findings(self.state)
        flagged = list(dict.fromkeys(f["scene_id"] for f in shown))
        rewritten = list(self.state.rewritten)
        done: list[str] = []
        for scene_id in flagged:
            located = outline.find(scene_id)
            current = self.work.manuscript(scene_id)
            if located is None or current is None:
                continue
            _, scene = located
            order = [s.id for _, s in outline.scenes_in_order()]
            at = order.index(scene_id)
            before = self.work.manuscript(order[at - 1]) if at > 0 else None
            disagree = "\n".join(f"- {f['note']}" for f in shown if f["scene_id"] == scene_id)
            prompt = (
                f"{self._scene_prompt(outline, scene, before.text if before else None, cast_text)}"
                f"\n\nThis scene was written before, but these lines disagree with the "
                f"story's codex or with the scene's plan:\n{disagree}\nWrite the whole scene "
                f"again so that it agrees with the codex and carries out the plan above, "
                f"keeping its events otherwise.\n\nThe scene as written:\n"
                f"{current.text}"
            )
            text, refused = await self._draft(prompt)
            if refused is not None:
                notes.append(f"Scene '{scene_id}' was not rewritten: {refused}")
                continue
            self.work.write_scene(
                scene_id,
                text,
                room_id=self.room_id,
                expected_digest=current.digest,
                agent_id=self.context.agent_id,
                turn_index=self.context.turn_index,
            )
            done.append(scene_id)
            await self._grow(scene_id, text, notes)
        self._save(rewritten=[*rewritten, *(s for s in done if s not in rewritten)], notes=notes)
        if done:
            await self._check(done)


def _cast_relations(
    found: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """The family the cast reply names: each character's codex `relations:` (the other's id
    to what this character is to them) by name, and the gender a gendered kinship word
    gives ("daughter" is female).

    A character's ``family`` lists each relative and what the relative is to it, as in "my
    father". Each relation is written on both sides (the father's relation to the child is
    ``parent``, the child's to the father ``child``), with ids made as
    `new_entry_draft` makes them. qwen3:8b sometimes wrote the word from the other side (a
    daughter listing her father as "father" in one run, as "daughter" in another), so a
    gendered word that contradicts the relative's stated gender and matches the
    character's own is read the other way round, and one that matches neither is dropped.
    A relation to a name not in the cast, to oneself, or in a word `family_role` does not
    know is dropped; the first relation of a pair wins.
    """
    raw = found.get("characters")
    people = [
        cast(dict[str, Any], c)
        for c in (cast(list[object], raw) if isinstance(raw, list) else [])
        if isinstance(c, dict)
    ][:MAX_CHARACTERS]
    ids = {
        _text(c.get("name")).casefold(): (_text(c.get("name")), id_for_name(_text(c.get("name"))))
        for c in people
        if _text(c.get("name"))
    }
    stated = {
        _text(c.get("name")).casefold(): gender_of(c.get("gender"))
        for c in people
        if _text(c.get("name"))
    }
    relations: dict[str, dict[str, str]] = {name: {} for name, _ in ids.values()}
    gender: dict[str, str] = {}
    for person in people:
        name, own = ids.get(_text(person.get("name")).casefold(), ("", None))
        family = person.get("family")
        for relation in cast(list[object], family) if isinstance(family, list) else []:
            if not isinstance(relation, dict):
                continue
            rel = cast(dict[str, Any], relation)
            relative, relative_id = ids.get(_text(rel.get("relative")).casefold(), ("", None))
            found_role = family_role(rel.get("is"))
            if not own or not relative_id or relative_id == own or found_role is None:
                continue
            role, word_gender = found_role
            holder, holder_id, other, other_id = relative, relative_id, name, own
            if word_gender is not None and stated.get(relative.casefold()) not in (
                None,
                word_gender,
            ):
                if stated.get(name.casefold()) != word_gender:
                    continue
                holder, holder_id, other, other_id = name, own, relative, relative_id
            relations[holder].setdefault(other_id, role)
            relations[other].setdefault(holder_id, INVERSE_ROLE[role])
            if word_gender is not None:
                gender.setdefault(holder, word_gender)
    return relations, gender


def _cast_params(found: Mapping[str, Any], notes: list[str]) -> list[NewEntryParams]:
    """The entries a cast reply asks for, each shaped as a new codex entry.

    A character's state holds its ``status`` and its ``gender`` (female or male; "other"
    stays in the visual block only), and its `relations:` its family (`_cast_relations`),
    which the continuity check holds each scene to.
    """
    family, family_gender = _cast_relations(found)
    asked: list[NewEntryParams] = []
    for kind, cap in (("characters", MAX_CHARACTERS), ("places", MAX_PLACES)):
        raw = found.get(kind)
        items = cast(list[object], raw)[:cap] if isinstance(raw, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = cast(dict[str, Any], item)
            name = _text(entry.get("name"))
            if not name:
                continue
            gender = _text(entry.get("gender")).lower()
            if gender not in ("female", "male", "other"):
                gender = family_gender.get(name, "")
            status = status_of(entry.get("status"))
            facts: dict[str, Any] = {"status": status}
            if gender in ("female", "male"):
                facts["gender"] = gender
            try:
                asked.append(
                    NewEntryParams(
                        kind=kind,  # pyright: ignore[reportArgumentType]
                        name=name,
                        profile=_text(entry.get("profile")),
                        # Where the premise leaves each character when the story opens --
                        # the missing father is missing, not alive -- so the continuity
                        # checks hold a scene to the premise.
                        state=facts if kind == "characters" else {},
                        relations=family.get(name, {}) if kind == "characters" else {},
                        appearance=(_text(entry.get("appearance")) or None)
                        if kind == "characters"
                        else None,
                        gender=cast(Any, gender)
                        if kind == "characters" and gender in ("female", "male", "other")
                        else None,
                    )
                )
            except ValidationError:
                notes.append(f"'{name}' did not fit a codex entry and was left out.")
    return asked


#: The first words of the call that plans what a story turns on, before its chapters.
#: The first words of the call that revises the outline by the person's feedback.
REVISE_MARK = "Revise this story outline as the person asks"
#: The first words of the call that reads the settled outline for the codex.
KG_MARK = "Read this settled story outline for its story bible"
#: How many threads and items settling the plot proposes, beside the conflict and twist.
MAX_PLOT_THREADS = 4
MAX_PLOT_ITEMS = 3
#: Words that name an idea, a feeling or a motif, not a thing a character can hold: an
#: item proposed at the plot lock whose name is one of these, or ends in one, is dropped
#: (`abstract_item`). "용기" is not here: it is also a container.
ABSTRACT_ITEM_WORDS: frozenset[str] = frozenset(
    {
        "기억",
        "추억",
        "희망",
        "진실",
        "비밀",
        "사랑",
        "두려움",
        "공포",
        "과거",
        "미래",
        "약속",
        "죄책감",
        "믿음",
        "신뢰",
        "복수",
        "운명",
        "자유",
        "상실",
        "슬픔",
        "분노",
        "우정",
        "정의",
        "명예",
        "꿈",
        "욕망",
        "배신",
        "후회",
        "용서",
        "증오",
        "외로움",
        "고독",
        "트라우마",
        "그리움",
        "침묵",
        "유산",
        "memory",
        "memories",
        "hope",
        "truth",
        "secret",
        "secrets",
        "love",
        "fear",
        "past",
        "future",
        "promise",
        "guilt",
        "trust",
        "faith",
        "courage",
        "revenge",
        "destiny",
        "fate",
        "freedom",
        "loss",
        "grief",
        "anger",
        "friendship",
        "justice",
        "honor",
        "dream",
        "dreams",
        "desire",
        "betrayal",
        "regret",
        "forgiveness",
        "hatred",
        "loneliness",
        "silence",
        "legacy",
    }
)
#: The first words of the call that plans a later chapter before it is written, and of
#: the call that revises that plan by the person's feedback: "... chapter 2".
DETAILS_MARK = "Plan the scenes of chapter"
DETAILS_REVISE_MARK = "Revise the plan of chapter"


def _arc_of(record: Mapping[str, Any] | None) -> Arc | None:
    """The `Arc` a state's `arc` record holds; `None` for none."""
    if not record:
        return None
    try:
        return Arc(
            functions=tuple(tuple(str(f) for f in fs) for fs in record["functions"]),
            twist_chapter=int(record["twist_chapter"]),
            climax_chapter=int(record["climax_chapter"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _chapter_dicts(outline: Outline, stakes: Sequence[str]) -> list[dict[str, Any]]:
    """The outline's chapters as the model writes them: names, not ids."""
    return [
        {
            "title": chapter.title,
            **({"stakes": stakes[n]} if n < len(stakes) and stakes[n] else {}),
            "scenes": [
                {"title": s.title, "summary": s.summary, "beats": list(s.beats)}
                for s in chapter.scenes
            ],
        }
        for n, chapter in enumerate(outline.chapters)
    ]


def _outline_json(outline: Outline, names: Mapping[str, str], stakes: Sequence[str]) -> str:
    """The outline as JSON the model can revise, with the cast by name."""
    chapters = _chapter_dicts(outline, stakes)
    for chapter, kept in zip(chapters, outline.chapters, strict=True):
        for scene, source in zip(chapter["scenes"], kept.scenes, strict=True):
            scene["characters"] = [names.get(i, i) for i in source.characters]
            scene["places"] = [names.get(i, i) for i in source.places]
    return json.dumps({"chapters": chapters}, ensure_ascii=False, indent=1)


def _kg_prompt(state: StartState, outline: Outline) -> str:
    characters = ", ".join(str(c["name"]) for c in state.cast if c.get("kind") == "characters")
    chapters = "\n\n".join(
        f"Chapter {n}:\n{chapter_text(c)}"
        for n, c in enumerate(_chapter_dicts(outline, state.stakes), start=1)
    )
    return (
        f"{KG_MARK}. The person has settled this plot.\n\n"
        f"Characters: {characters}\n"
        f"Central conflict: {state.central_conflict or ''}\nTwist: {state.twist or ''}\n\n"
        f"{chapters}\n\n"
        "List what the story bible should track from this plot, beyond the characters, "
        "places, conflict and twist above:\n"
        f"- threads: up to {MAX_PLOT_THREADS} key events, secrets or promises that one "
        "chapter sets up and a later one pays off, and relationships between characters "
        "that are not family (a rivalry, a debt, a friendship, a love), each with the "
        "chapter that sets it up and the chapter that pays it off;\n"
        f"- items: up to {MAX_PLOT_ITEMS} physical objects the plot turns on, each named "
        "in the outline above: a thing a character can hold, carry, lose or hand over (a "
        "letter, a key, a lantern). Not an idea, feeling, memory, secret, promise or "
        "motif: those are threads or nothing.\n"
        "Write names and profiles in the language of the outline.\n"
        'Reply as JSON: {"threads": [{"name": a short name, "profile": one sentence, '
        '"characters": [names from the list above], "planted_chapter": number, '
        '"pay_off_chapter": number}], "items": [{"name": ..., "profile": one sentence}]}'
    )


def abstract_item(name: str) -> bool:
    """Whether an item's `name` names an idea or a motif ("기억", "아버지의 비밀") rather
    than a physical object.

    The name, or its last word, is one of `ABSTRACT_ITEM_WORDS`; a Korean last word also
    counts when it ends in a Korean one of two or more letters ("잃어버린기억"), while an
    English last word must be one exactly ("the locket" is kept, "a promise" is not).
    """
    words = re.findall(r"\w+", name.casefold())
    if not words:
        return False
    whole, last = " ".join(words), words[-1]
    if whole in ABSTRACT_ITEM_WORDS or last in ABSTRACT_ITEM_WORDS:
        return True
    return any(len(w) >= 2 and last.endswith(w) for w in ABSTRACT_ITEM_WORDS if not w.isascii())


def _kg_params(
    found: Mapping[str, Any],
    cast_lines: Sequence[Mapping[str, Any]],
    scene_of: Callable[[object], str | None],
) -> tuple[list[tuple[NewEntryParams, dict[str, str | None]]], list[str]]:
    """The threads and items of the settled-plot reply, as new entries to propose, and
    the names of the items dropped as abstract (`abstract_item`), which do not count
    toward `MAX_PLOT_ITEMS`."""
    ids = {
        str(c.get("name", "")).casefold(): str(c.get("id"))
        for c in cast_lines
        if c.get("kind") == "characters"
    }
    wanted: list[tuple[NewEntryParams, dict[str, str | None]]] = []
    dropped: list[str] = []
    for kind, key, most in (
        ("threads", "threads", MAX_PLOT_THREADS),
        ("items", "items", MAX_PLOT_ITEMS),
    ):
        raw = found.get(key)
        items = cast(list[object], raw) if isinstance(raw, list) else []
        taken = 0
        for item in items:
            if taken >= most:
                break
            if not isinstance(item, dict):
                continue
            it = cast(dict[str, Any], item)
            name = _text(it.get("name"))
            if not name:
                continue
            if kind == "items" and abstract_item(name):
                dropped.append(name)
                continue
            taken += 1
            fields: dict[str, str | None] = {}
            state: dict[str, Any] = {}
            if kind == "threads":
                involves = [
                    ids[n.casefold()] for n in _texts(it.get("characters")) if n.casefold() in ids
                ]
                if involves:
                    state["involves"] = involves
                fields = {
                    "planted_in": scene_of(it.get("planted_chapter")),
                    "pay_off_by": scene_of(it.get("pay_off_chapter")),
                }
            params = NewEntryParams(
                kind="threads" if kind == "threads" else "items",
                name=name,
                profile=_text(it.get("profile")),
                state=state,
            )
            wanted.append((params, fields))
    return wanted, dropped


def _with_entry_fields(draft: Proposal, fields: Mapping[str, str | None]) -> Proposal:
    """`draft` with fields `NewEntryParams` has no place for (a thread's scenes, notes)."""
    given = {k: v for k, v in fields.items() if v}
    if not given:
        return draft
    data = draft.model_dump(mode="json")
    data["change"] = {"new_entry": {**data["change"]["new_entry"], **given}}
    try:
        return Proposal.model_validate(data)
    except ValidationError as exc:
        raise StoryError(f"'{draft.entry_id}' was not proposed: it does not fit.") from exc


SPINE_MARK = "Before any chapter is outlined, decide what this story turns on"
#: The first words of the call that outlines one chapter, as in "... chapter 2 of 3".
CHAPTER_PROMPT_MARK = "Outline only chapter"

_SCENE_SHAPE = (
    "Every scene has a title, a one-sentence summary of what happens, 2 to 4 beats "
    "(concrete events, in order), and the names of the characters and places in it, "
    "exactly as listed above. Work every 'must include' element into the story."
)


def _spine_prompt(title: str, count: int, arc: Arc) -> str:
    return (
        f"The story is built on the structure '{title}', {count} chapters.\n"
        f"{SPINE_MARK}. Write each in one sentence, with THIS premise and cast:\n"
        "- central_conflict: who wants what, against whom or what.\n"
        "- twist: a turn the reader does not see coming, which changes what the conflict "
        f"is about. The reader learns it in chapter {arc.twist_chapter}.\n"
        "- clue: a small, concrete thing an earlier chapter can show that points at the "
        "twist without stating it.\n"
        f"- resolution: how chapter {arc.climax_chapter} decides the conflict, and what it "
        "costs.\n"
        'Reply as JSON: {"central_conflict": ..., "twist": ..., "clue": ..., '
        '"resolution": ...}'
    )


def _has_scenes(reply: Mapping[str, Any] | None) -> bool:
    """Whether a chapter reply holds at least one scene the outline can keep."""
    if reply is None:
        return False
    scenes = [cast(dict[str, Any], s) for s in _items(reply.get("scenes")) if isinstance(s, dict)]
    return any(_text(s.get("title")) or _text(s.get("summary")) for s in scenes)


def _items(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _chapters(found: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The chapter objects of an outline reply, in order."""
    raw = found.get("chapters")
    items = cast(list[object], raw) if isinstance(raw, list) else []
    return [cast(dict[str, Any], c) for c in items if isinstance(c, dict)]


def outline_from(
    data: Mapping[str, Any],
    cast_lines: Sequence[Mapping[str, Any]],
    *,
    acts: Sequence[str] = (),
) -> Outline:
    """An outline from the model's JSON: ids by code, names mapped to the cast's ids.

    With `acts`, the chapters are the structure's acts in order: the n-th chapter kept is
    act n, and chapters past the last act are dropped.
    """
    by_name = _cast_ids(cast_lines)
    chapters: list[dict[str, Any]] = []
    raw_chapters = data.get("chapters")
    items = cast(list[object], raw_chapters) if isinstance(raw_chapters, list) else []
    for chapter in items[: len(acts) or MAX_CHAPTERS]:
        if not isinstance(chapter, dict):
            continue
        ch = cast(dict[str, Any], chapter)
        number = len(chapters) + 1
        scenes = _scene_dicts(ch.get("scenes"), f"ch{number:02d}", by_name)
        if not scenes:
            continue
        chapters.append(
            {
                "id": f"ch{number:02d}",
                "title": _text(ch.get("title")) or f"{number}장",
                **({"act": acts[number - 1]} if number <= len(acts) else {}),
                "scenes": scenes,
            }
        )
    return Outline.model_validate({"chapters": chapters})


def _cast_ids(cast_lines: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """The cast's ids by `(kind, name)`, the name casefolded."""
    return {
        (str(c.get("kind")), str(c.get("name", "")).casefold()): str(c.get("id"))
        for c in cast_lines
    }


#: A title that ends like this is a clause cut short, not a name: a comma, a sentence's
#: end, or a Korean connective ending ("느끼며", "알았지만").
_CLAUSE_END = re.compile(r"(?:[,.!?…:;]|(?:며|고|서|지만|는데|면서|으나|니까|다가|하여))$")
#: The longest a scene title may run before it reads as a sentence.
MAX_SCENE_TITLE = 20
#: "X의 Y" -- the shape most scene titles have ("장로의 속삭임"); X is not a pronoun.
_OF_PHRASE = re.compile(r"(?<!\S)([^\s,.]+)의\s+([^\s,.]+)")
_PRONOUN_OWNERS = frozenset({"자신", "자기", "그", "그녀", "나", "너", "우리", "그들", "저"})
#: Particles taken off the end of the phrase's noun, longest first.
_PARTICLES = tuple("에서 에게 으로 을 를 은 는 이 가 에 로 와 과 도 만".split())


def _clause_title(title: str, summary: str) -> bool:
    """Whether the model's title is a clause rather than a name: cut off mid-sentence,
    too long, or the opening of the summary itself."""
    if _CLAUSE_END.search(title) or len(title) > MAX_SCENE_TITLE:
        return True
    return len(title) >= 8 and summary.startswith(title) and title != summary


def _noun_of(word: str) -> str:
    for particle in _PARTICLES:
        if word.endswith(particle) and len(word) - len(particle) >= 2:
            return word[: -len(particle)]
    return word


def scene_title(title: str, summary: str) -> str:
    """A short title for a planned scene. The model's own title is kept unless it is a
    clause (`_clause_title`) or missing; then a "X의 Y" phrase from the summary or the
    clause is taken, or failing that the clause's first words, cut at the first break.
    Empty only when both title and summary are."""
    if title and not _clause_title(title, summary):
        return title
    source = summary or title
    for text in (summary, title):
        for m in _OF_PHRASE.finditer(text):
            if m.group(1) not in _PRONOUN_OWNERS:
                return f"{m.group(1)}의 {_noun_of(m.group(2))}"
    head = re.split(r"[,.!?…:;]", source, maxsplit=1)[0].split()
    if len(head) > 1 and re.search(r"(?:은|는)$", head[0]):
        head = head[1:]  # the subject ("이진호는") names the scene no better than its cast
    short = " ".join(head[:3])
    return _CLAUSE_END.sub("", short)[:MAX_SCENE_TITLE].strip()


def _scene_dicts(
    value: object, chapter_id: str, by_name: Mapping[tuple[str, str], str]
) -> list[dict[str, Any]]:
    """A chapter's scenes from the model's JSON: ids by code (`<chapter_id>.s01`), and
    the names of the cast as its ids; a scene with neither title nor summary is dropped."""
    scenes: list[dict[str, Any]] = []
    for scene in _items(value)[:MAX_SCENES_PER_CHAPTER]:
        if not isinstance(scene, dict):
            continue
        sc = cast(dict[str, Any], scene)
        title = scene_title(_text(sc.get("title")), _text(sc.get("summary")))
        if not title:
            continue
        scenes.append(
            {
                "id": f"{chapter_id}.s{len(scenes) + 1:02d}",
                "title": title,
                "summary": _text(sc.get("summary")),
                "beats": _texts(sc.get("beats")),
                "characters": _ids(sc.get("characters"), "characters", by_name),
                "places": _ids(sc.get("places"), "places", by_name),
            }
        )
    return scenes


def _detail_scenes(
    found: Mapping[str, Any] | None,
    cast_lines: Sequence[Mapping[str, Any]],
    limit: int = MAX_SCENES_PER_CHAPTER,
) -> list[dict[str, Any]]:
    """A chapter plan's scenes from the model's JSON, the cast by name as the person reads
    it; a name not in the cast is left out, and scenes past `limit` (the chapter's count,
    `scenes_for`) are too. Empty when the reply holds no scene."""
    if found is None:
        return []
    by_name = _cast_ids(cast_lines)
    names = {str(c.get("id")): str(c.get("name", "")) for c in cast_lines}
    return [
        {
            "title": scene["title"],
            "summary": scene["summary"],
            "beats": scene["beats"],
            "characters": [names[i] for i in scene["characters"]],
            "places": [names[i] for i in scene["places"]],
        }
        for scene in _scene_dicts(found.get("scenes"), "plan", by_name)[:limit]
    ]


#: What a chapter plan is asked about the characters it brings in (`_new_characters`).
_NEW_CHARACTERS_ASK = (
    "If this chapter brings in a character who is not in the cast above and is family of "
    "someone in it, list them under new_characters, with what that cast member is to them "
    '(as in "my father": father, mother, son, daughter, brother, sister, husband, wife, '
    "grandfather, grandmother, grandson or granddaughter). [] for none.\n"
)
_NEW_CHARACTERS_SHAPE = (
    '"new_characters": [{"name": ..., "profile": one sentence, "gender": "female" or '
    '"male", "family": [{"relative": a name from the cast above, "is": what that '
    "relative is to this new character}]}]"
)


def _new_characters(
    found: Mapping[str, Any] | None, known: Mapping[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    """The new characters a chapter plan brings in with family among the known characters.

    A name the story already has is not new; a relation to a name it does not have, or in
    a word `family_role` does not know, is dropped, and a new character left with no relation
    is not kept (a scene that brings them in proposes them when it is written, through
    `grow_scene`). A gender the new character's reply leaves out is not guessed.
    """
    if found is None:
        return []
    raw = found.get("new_characters")
    people = cast(list[object], raw) if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for person in people[:MAX_CHARACTERS]:
        if not isinstance(person, dict):
            continue
        p = cast(dict[str, Any], person)
        name = _text(p.get("name"))
        if not name or name.casefold() in known or name.casefold() in seen:
            continue
        given = p.get("family")
        family: list[dict[str, str]] = []
        for relation in cast(list[object], given) if isinstance(given, list) else []:
            if not isinstance(relation, dict):
                continue
            rel = cast(dict[str, Any], relation)
            relative = known.get(_text(rel.get("relative")).casefold())
            word = _text(rel.get("is"))
            role = family_role(word)
            if relative is None or role is None:
                continue
            if any(f["relative_id"] == relative[1] for f in family):
                continue
            family.append(
                {"relative": relative[0], "is": word, "relative_id": relative[1], "role": role[0]}
            )
        if not family:
            continue
        seen.add(name.casefold())
        gender = gender_of(p.get("gender"))
        out.append(
            {
                "name": name,
                "profile": _text(p.get("profile")),
                "gender": gender or "",
                "family": family,
            }
        )
    return out


def _shown_findings(state: StartState) -> list[dict[str, str]]:
    """The check's notes on the chapter written last: what the person is shown, and what
    a rewrite they ask for rewrites. Notes on earlier chapters were shown with them."""
    if not state.chapter_scenes:
        return list(state.continuity)
    return [f for f in state.continuity if f.get("scene_id") in state.chapter_scenes]


def _ids(value: object, kind: str, by_name: Mapping[tuple[str, str], str]) -> list[str]:
    """The cast ids of the names given; a name not in the cast is left out."""
    ids: list[str] = []
    for name in _texts(value):
        found = by_name.get((kind, name.casefold()))
        if found and found not in ids:
            ids.append(found)
    return ids
