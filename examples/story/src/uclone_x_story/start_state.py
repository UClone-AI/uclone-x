# pyright: reportPrivateUsage=false
"""State models, brief extraction, genre/structure resolution, and similarity metrics for `story_start`."""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uclone_x.agent.reply_lines import is_korean
from uclone_x.i18n.scripts import scripts_of, words_outside
from uclone_x.knowledge.fold import bigram_overlap
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole
from uclone_x.tools.models import ToolContext
from uclone_x_story.arc import Arc
from uclone_x_story.enrich import Enrichment
from uclone_x_story.kg_plan import candidate_names
from uclone_x_story.library import StoryError
from uclone_x_story.muse import normalize_genre
from uclone_x_story.schemas import Outline
from uclone_x_story.skill_data import StructureTemplate
from uclone_x_story.work import StoryWork

if TYPE_CHECKING:
    from uclone_x.tools.protocols import ToolProtocol

logger = logging.getLogger(__name__)

#: The flow's own file in the story's folder: where it stopped and what it has made.
START_FILE = "start.yaml"

Stage = Literal[
    "choose_genre",
    "choose_premise",
    "review_cast",
    "review_outline",
    "review_chapter",
    "review_scene",
    "done",
]

#: The languages the flow's own words come in: the story's, chosen once from the request
#: by the rule the reply notes are chosen by (`story_language`).
Language = Literal["ko", "en"]

#: Each story language by name, as a prompt asks for text in it (#2301): the system prompt's
#: "the language of the person's request" let qwen3:8b write a Korean story's premise in
#: English.
LANGUAGE_NAMES: Mapping[Language, str] = {"ko": "Korean", "en": "English"}
#: The script each story language is written in (`uclone_x.i18n.scripts`). Generated text is
#: held to it and to the scripts the person's own words use (`story_scripts`); a word in any
#: other script is redrawn once, whatever the script is (#2301).
LANGUAGE_SCRIPTS: Mapping[Language, frozenset[str]] = {
    "ko": frozenset({"HANGUL"}),
    "en": frozenset({"LATIN"}),
}

#: What the brief fills a field with when the request leaves it open, in the story's
#: language. Code, not the model, decides this: a field the request set is never replaced
#: (#1723). Korean defaults in an English story's prompts gave an English speaker Korean
#: premise titles (#2263).
DEFAULTS: Mapping[Language, Mapping[str, str]] = {
    "ko": {
        "genre": "판타지",
        "tone": "진지하면서도 따뜻한",
        "audience": "성인 일반 독자",
        "length": "단편, 4장 안팎",
    },
    "en": {
        "genre": "Fantasy",
        "tone": "serious but warm",
        "audience": "general adult readers",
        "length": "a short story of about four chapters",
    },
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
    ("느와르", "mystery"),
    ("누아르", "mystery"),
    ("범죄", "mystery"),
    ("수사", "mystery"),
    ("mystery", "mystery"),
    ("noir", "mystery"),
    ("crime", "mystery"),
    ("로맨스", "romance"),
    ("연애", "romance"),
    ("romance", "romance"),
    ("sf", "science-fiction"),
    ("과학", "science-fiction"),
    ("우주", "science-fiction"),
    ("사이버펑크", "science-fiction"),
    ("디스토피아", "science-fiction"),
    ("science", "science-fiction"),
    ("cyberpunk", "science-fiction"),
    ("dystopia", "science-fiction"),
)
#: Adjacent or everyday genres that have no dedicated muse table: mapped to the closest
#: bundled table (`romance` for realistic human drama, `mystery` for thriller/historical
#: intrigue, `fantasy` for adventure) so they do not fall back to dragon/magic cards.
_MUSE_GENRE_WORDS: tuple[tuple[str, str], ...] = (
    ("드라마", "romance"),
    ("일상", "romance"),
    ("가족", "romance"),
    ("휴먼", "romance"),
    ("성장", "romance"),
    ("drama", "romance"),
    ("slice of life", "romance"),
    ("스릴러", "mystery"),
    ("역사", "mystery"),
    ("시대극", "mystery"),
    ("사극", "mystery"),
    ("thriller", "mystery"),
    ("historical", "mystery"),
    ("모험", "fantasy"),
    ("무협", "fantasy"),
    ("adventure", "fantasy"),
)
_FALLBACK_TABLE = "fantasy"
#: The bundled muse tables a genre is proposed from, as the person reads each one. When
#: the request names no genre, code draws the proposal from these at random rather than
#: fixing `DEFAULTS["genre"]`, and the person takes it or names another (§1.1 principle 1).
#: Each label in each language the flow speaks: the catalog an English speaker's proposal
#: is worded from (#2263).
GENRE_LABELS: Mapping[str, Mapping[Language, str]] = {
    "fantasy": {"ko": "판타지", "en": "Fantasy"},
    "horror": {"ko": "호러", "en": "Horror"},
    "mystery": {"ko": "미스터리", "en": "Mystery"},
    "romance": {"ko": "로맨스", "en": "Romance"},
    "science-fiction": {"ko": "SF", "en": "Science fiction"},
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
    ("느와르", "three-act"),
    ("누아르", "three-act"),
    ("범죄", "three-act"),
    ("수사", "three-act"),
    ("호러", "three-act"),
    ("공포", "three-act"),
    ("사이버펑크", "three-act"),
    ("디스토피아", "three-act"),
    ("mystery", "three-act"),
    ("thriller", "three-act"),
    ("noir", "three-act"),
    ("crime", "three-act"),
    ("horror", "three-act"),
    ("cyberpunk", "three-act"),
    ("dystopia", "three-act"),
    ("판타지", "heros-journey"),
    ("모험", "heros-journey"),
    ("fantasy", "heros-journey"),
    ("adventure", "heros-journey"),
    ("sf", "heros-journey"),
    ("science fiction", "heros-journey"),
    ("드라마", "kishotenketsu"),
    ("일상", "kishotenketsu"),
    ("가족", "kishotenketsu"),
    ("휴먼", "kishotenketsu"),
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
#: The functions (`uclone_x_story.arc.FUNCTIONS`) that make a chapter heavy or light.
_HEAVY = frozenset({"reversal", "climax"})
_LIGHT = frozenset({"setup", "resolution"})
#: How many chapters one call writes, even when the story was asked for at once: each
#: chapter is several model calls, so a call that wrote every chapter could run for as
#: long as the outline is. The next call writes the next one.
CHAPTERS_PER_CALL = 1
# Drafts per scene: the first, then rewrites told what the prose check refused.
MAX_DRAFTS = 3
#: What a scene's second draft is told when its first repeats the scene before (#2307).
REPEAT_MARK = "Your last draft was not saved: it repeats the scene before this one."


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


_INTERACTIVE = re.compile(
    r"\brpg\b|인터랙티브|선택지\s*(?:를|로|주|제시)|(?:한\s*)?(?:씬|장면)\s*씩|"
    r"(?:씬|장면)\s*마다\s*선택|\binteractive\b|\bscene[\s-]+by[\s-]+scene\b|"
    r"\bchoose\s+each\s+scene\b",
    re.IGNORECASE,
)


def wants_interactive(request: str) -> bool:
    """Whether the request asks for RPG-style scene-by-scene interactive story building."""
    return bool(_INTERACTIVE.search(request))


def story_language(request: str) -> Language:
    """The story's language: Korean when the request is, else English -- the rule the
    reply notes are chosen by (`uclone_x.agent.reply_lines.is_korean`), so the flow's
    labels and the note around them agree.

    One Hangul syllable makes it Korean ("Write a 호러 story" is `ko`), as it makes the reply
    notes Korean. Kept on purpose (#2272): a share-of-script threshold would be a second
    language rule beside the one core uses (#2204), and the story's defaults would then
    disagree with the notes the same request gets. A person who writes any Korean reads it.
    """
    return "ko" if is_korean(request) else "en"


def genre_label(label: str, language: Language) -> str:
    """A proposed genre's label in `language`; a genre that is not one of the catalog's
    (one the person named in their own words) as it is."""
    for labels in GENRE_LABELS.values():
        if label in labels.values():
            return labels[language]
    return label


def genre_options(tables: Sequence[str], rng: random.Random, language: Language) -> list[str]:
    """Genres to propose, drawn at random: the recommended one first, then alternatives.

    Drawn from the muse tables loaded that have a label (`GENRE_LABELS`); from every
    labelled table when none of those loaded.
    """
    keys = sorted(k for k in tables if k in GENRE_LABELS) or sorted(GENRE_LABELS)
    return [GENRE_LABELS[k][language] for k in rng.sample(keys, min(GENRE_OPTIONS, len(keys)))]


def turned_down_genres(avoid: Sequence[str]) -> set[str]:
    """The bundled genres among what the person turned down (a reading's `avoid`), by
    table: the structured reading says what they do not want, so code reads no negation
    (#2301, #2204)."""
    return {table for text in avoid if (table := _genre_table(text)) is not None}


def _genre_turned_down(genre: str, refused: set[str]) -> bool:
    return (_genre_table(genre) or genre.strip().casefold()) in refused


def story_scripts(language: Language, said: Sequence[str]) -> frozenset[str]:
    """The scripts a story's generated text may be in: its language's, and every script the
    person's own words use ("SF로" lets a Korean story say "SF")."""
    return LANGUAGE_SCRIPTS[language] | frozenset(scripts_of(said))


def _has_word(lowered: str, word: str) -> bool:
    # An English word stands alone ("sf", not the "sf" in "transfer"); a Korean genre word is
    # not preceded by a negating/pejorative prefix ("비일상", "흑역사").
    return (
        bool(re.search(rf"\b{re.escape(word)}\b", lowered))
        if word.isascii()
        else bool(re.search(rf"(?<![비흑]){re.escape(word)}", lowered))
    )


def _genre_table(genre: str) -> str | None:
    """The bundled genre a genre's name is one of the words for ("추리" is mystery)."""
    lowered = genre.casefold()
    return next((table for word, table in _GENRE_WORDS if _has_word(lowered, word)), None)


def names_genre(request: str, genre: str) -> bool:
    """Whether the request's own words name `genre`: the genre itself, or another word code
    knows for the same genre ("추리" for "미스터리"). A word for another genre does not
    count: in "호러는 싫어" it once let a "로맨스" the model made up through (#2270)."""
    lowered = request.casefold()
    wanted = genre.strip().casefold()
    if wanted and _has_word(lowered, wanted):
        return True
    table = _genre_table(genre)
    return table is not None and any(
        known == table and _has_word(lowered, word) for word, known in _GENRE_WORDS
    )


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


def _list_item_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        for key in ("text", "name", "value", "item", "element"):
            raw = mapping.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        if len(mapping) == 1:
            only = next(iter(mapping.values()))
            if isinstance(only, str):
                return only.strip()
    return ""


def _texts(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [t for t in (_list_item_text(v) for v in cast(list[object], value)) if t]
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
    #: The language of the request, which the flow's own words in the story follow.
    language: Language = "ko"


def brief_from(stated: Mapping[str, Any] | None, request: str) -> Brief:
    """The brief: what the request stated, and `DEFAULTS` for only what it left open."""
    data = stated or {}
    language = story_language(request)
    avoid = _texts(data.get("avoid"))
    fields: dict[str, str] = {}
    from_request: list[str] = []
    for key in _BRIEF_FIELDS:
        value = _text(data.get(key))
        if key == "genre" and value and not names_genre(request, value):
            # qwen3:8b read "소설 하나 써줘" as genre "일반" and called it stated
            # (2026-09-29), so the genre was never proposed: a genre counts as the
            # request's only when the request's own words name it.
            value = ""
        if key == "genre" and _genre_turned_down(value, turned_down_genres(avoid)):
            # The request names the genre to turn it down ("판타지 말고"): what it does
            # not want wins over the word (#2301).
            value = ""
        if value:
            fields[key] = value
            from_request.append(key)
        else:
            fields[key] = DEFAULTS[language][key]
    must_have = _texts(data.get("must_have"))
    if must_have:
        from_request.append("must_have")
    if avoid:
        from_request.append("avoid")
    title = _text(data.get("title")) or _title_of(request, language)
    return Brief(
        title=title[:60],
        language=language,
        must_have=must_have,
        avoid=avoid,
        from_request=from_request,
        **fields,
    )


def _title_of(request: str, language: Language) -> str:
    words = request.strip().split()
    return " ".join(words[:6]) or ("새 이야기" if language == "ko" else "New story")


def muse_genre(genre: str, tables: Sequence[str]) -> str | None:
    """The muse table for a genre as a person wrote it; `None` when none fits."""
    key = normalize_genre(genre)
    if key in tables:
        return key
    lowered = genre.casefold()
    for words in (_GENRE_WORDS, _MUSE_GENRE_WORDS):
        matches = [
            (lowered.find(word), table)
            for word, table in words
            if _has_word(lowered, word) and table in tables
        ]
        if matches:
            return min(matches, key=lambda m: m[0])[1]
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
    #: Whether scene-by-scene RPG interactive mode (`review_scene`) is enabled.
    interactive: bool = False
    stage: Stage = "choose_premise"
    brief: Brief
    #: The genres proposed because the request named none, the recommended one first;
    #: empty when the request named the genre. The brief's genre is what was taken.
    genre_options: list[str] = Field(default_factory=list[str])
    #: The person answered the proposal (took it or named another), rather than the
    #: flow using the recommended genre on its own because the story was asked for at once.
    genre_answered: bool = False
    premises: list[Premise] = Field(default_factory=list[Premise])
    #: The premise `go_on=True` takes at the premise stop when `choice` is omitted.
    recommended: int = 1
    #: What the person said at the premise stop that turned the premises down, in turn,
    #: and the titles of every premise turned down: the next draw follows the words and
    #: does not offer those again.
    premise_feedback: list[str] = Field(default_factory=list[str])
    premises_turned_down: list[str] = Field(default_factory=list[str])
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
    #: The rest of what the story turns on (`uclone_x_story.arc.Spine.record`): the clue a
    #: chapter before the twist may plant, and how the climax decides the conflict.
    spine: dict[str, str] | None = None
    #: What each chapter does (`uclone_x_story.arc.Arc.record`), decided by code from the
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
    #: The person settled the plot at the outline stop (`go_on=True`), or asked for the
    #: story at once. The flow no longer revises it.
    plot_locked: bool = False
    #: What settling the plot proposed for the codex -- threads (the central conflict, the
    #: twist, key events and relationships) and items -- as `{"proposal", "kind",
    #: "entry_id", "name"}`. Each waits for the person's approval, like the cast.
    plot_kg: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
    #: How many items the plot lock's reading named that were ideas, not objects
    #: (`abstract_item`), and so were not proposed.
    plot_kg_dropped: int = 0
    #: How many transitions the plot lock's reading gave that it did not place under a scene
    #: label the outline has, and so were not planned, by why (`unplaced_transitions`:
    #: `label_missing`, `label_unknown`; #2376, #2378).
    plot_kg_unplaced: dict[str, int] = Field(default_factory=dict[str, int])
    #: Planned scene-level KG state and relation transitions (`SceneTransition.record()`),
    #: validated across the outline's timeline ($G_0 \to G_1 \to \dots \to G_n$).
    planned_transitions: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    #: Summary of the simulated KG trajectory (`TrajectoryReport.record()`).
    kg_trajectory: dict[str, Any] | None = None
    #: Cast proposals still pending when the chapter was asked for without the person
    #: saying to go on: the chapter waits for them.
    cast_waiting: list[str] = Field(default_factory=list[str])
    #: Cast proposals still pending when the chapter was written over them, because the
    #: person asked for the story at once or said to go on; the reply says so. A record of
    #: that decision for the reply and the result, not a view of the codex: the check
    #: after writing reads pending proposals from the store (`continuity.checked_store`).
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
    #: The written scenes the person's words at a chapter's plan asked to change, as
    #: `{"id", "title"}`: an edit of the story so far, not feedback on the plan, which is
    #: kept (#2303).
    details_scene_edits: list[dict[str, str]] = Field(default_factory=list[dict[str, str]])
    #: Of `details_scene_edits`, the scenes the flow could not write again (#2361): the
    #: prose check refused every draft, so the scene is as it was.
    details_scene_edits_failed: list[str] = Field(default_factory=list[str])
    #: Titles the person's words named that more than one written scene has (#2365), each
    #: with where those scenes are, as the person reads it (`ko`, `en`), and the words that
    #: asked (`words`), read again with the answer: nothing was rewritten, and the person
    #: is asked which one they meant.
    details_scene_edits_ambiguous: list[dict[str, str]] = Field(
        default_factory=list[dict[str, str]]
    )
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
    #: RPG interactive scene stop (`review_scene`): the current scene's chapter, id,
    #: three RPG choices, entering world state snapshot, chronological KG transition log,
    #: and the id of the scene written in the last step.
    scene_chapter: int | None = None
    scene_id: str | None = None
    scene_choices: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    scene_entering_kg: str = ""
    scene_kg_timeline: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    scene_written_last: str | None = None
    #: Whether automatic scene-by-scene illustration generation is enabled when an image
    #: engine is ready, and the map of `scene_id -> relative_url` for generated illustrations.
    illustrate: bool = True
    scene_illustrations: dict[str, str] = Field(default_factory=dict[str, str])
    notes: list[str] = Field(default_factory=list[str])
    #: The person's words, verbatim, when they left the start to write the story themselves
    #: (`leave`): from then its outline and scenes are no longer kept for the flow, and it
    #: does not go on (#2270). Kept so anyone can see who left it and why.
    left_request: str | None = None
    #: The room turn whose call last moved the flow (`ToolContext.turn_id`): the stop it
    #: reached waits for the person's next message, so a 'continue' or 'leave' from the
    #: same turn is the model answering for them, and is refused (#2300).
    stop_turn_id: str | None = None

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

    action: Literal["start", "continue", "status", "leave"] = Field(
        description="'start' a new story from the person's request; 'continue' the story "
        "this conversation started, to the next stage; 'status' says where it stands; "
        "'leave' stops the start, only when the person asks to write the story themselves, "
        "with their words as 'request'."
    )
    request: str | None = Field(
        default=None,
        description="For 'start' or 'leave': the person's request, word for word. On "
        "'continue', the person's words, word for word, only when they ask for a change: "
        "they are taken as the change in place of 'changes'.",
    )
    choice: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="For 'continue' at 'choose_genre', 'choose_premise', or 'review_scene': "
        "the option number (1–3) the person chose. At 'choose_premise' a premise is taken "
        "only with this: for a bare yes, pass the recommended number.",
    )
    genre: str | None = Field(
        default=None,
        description="For 'continue' at 'choose_genre' or when redrawing at 'choose_premise': "
        "the genre the person named.",
    )
    changes: str | None = Field(
        default=None,
        description="For 'continue': what the person asked to change in the current stop "
        "(new direction when rejecting all three premises at 'choose_premise', outline "
        "revisions at 'review_outline', scene plan revisions at 'review_chapter', or custom "
        "RPG action / KG updates at 'review_scene'). Leave out when they accept what was shown.",
    )
    must_have: list[str] | None = Field(
        default=None,
        description="For 'continue': story elements the person wants included in this or "
        "later stages.",
    )
    avoid: list[str] | None = Field(
        default=None,
        description="For 'continue': story elements or genres the person wants kept out.",
    )
    go_on: bool = Field(
        default=False,
        description="For 'continue': true when the person accepts what was shown (taking the "
        "recommended genre if 'choice' is omitted, settling the outline, proceeding "
        "over pending cast proposals, writing the planned chapter, or taking choice 1 at "
        "'review_scene').",
    )
    rewrite: bool = Field(
        default=False,
        description="For 'continue': true when the person asks to rewrite scenes flagged by "
        "the continuity check.",
    )
    interactive: bool | None = Field(
        default=None,
        description="When true, enables RPG-style scene-by-scene interactive mode "
        "('review_scene'), offering 3 KG-grounded choices before each scene and updating the "
        "KG chronologically. When false, uses chapter-by-chapter batch mode.",
    )
    illustrate: bool | None = Field(
        default=None,
        description="When true (default), generates a key-frame illustration for each written "
        "scene when an image engine is ready; when false, skips automatic scene illustrations.",
    )
    # Two arguments the Writer adds on its own (#2360: 7 refusals in one live run, a step
    # each). Declared rather than `extra="ignore"`, which would also take a misspelt
    # 'choise' and turn a decision into nothing: the story is always this conversation's,
    # and the stage is read from it.
    story_id: str | None = Field(
        default=None,
        description="Not needed: the story is the one this conversation started. Ignored.",
    )
    stage: str | None = Field(
        default=None,
        description="Not needed: the stage is read from the story. Ignored.",
    )


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


def _also(line: Mapping[str, Any]) -> str:
    """A cast line's other names, as the scene prompt lists them after its name."""
    aliases = _texts(line.get("aliases"))
    return f" (also {', '.join(aliases)})" if aliases else ""


def _aliases(raw: object, name: str) -> list[str]:
    """The other names a cast reply gives a character or place: each a non-empty string,
    once, not its name, and within Hangul/Latin or the scripts of `name`. The names the prose
    will use (#2217), so a scene's reading finds them there."""
    allowed = LANGUAGE_SCRIPTS["ko"] | LANGUAGE_SCRIPTS["en"] | frozenset(scripts_of([name]))
    return list(
        dict.fromkeys(
            a
            for a in _texts(raw)
            if a.casefold() != name.casefold() and not words_outside(a, allowed)
        )
    )


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


def _items(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _ids(value: object, kind: str, by_name: Mapping[tuple[str, str], str]) -> list[str]:
    """The cast ids of the names given; a name not in the cast is left out."""
    ids: list[str] = []
    for name in _texts(value):
        found = next(
            (
                by_name[(kind, c.casefold())]
                for c in candidate_names(name)
                if (kind, c.casefold()) in by_name
            ),
            None,
        )
        if found and found not in ids:
            ids.append(found)
    return ids


class _RunBase:
    """Shared state and workspace helpers for a `story_start` stage walk."""

    def __init__(
        self,
        work: StoryWork,
        room_id: str,
        context: ToolContext,
        model: StoryModel,
        state: StartState,
        digest: str,
        structures: Mapping[str, StructureTemplate] | None = None,
        embedder: PremiseEmbedder | None = None,
        image_tool: ToolProtocol | None = None,
    ) -> None:
        self.work = work
        self.room_id = room_id
        self.context = context
        self.model = model
        self.state = state
        self.digest = digest
        self.structures = dict(structures or {})
        self.embedder = embedder
        self.image_tool = image_tool
        #: Codex proposals read from the scenes this call wrote (`_grow`); `None` until
        #: this call writes a scene, so a call that writes none keeps the last call's.
        self.grown: list[str] | None = None
        #: Each scene this call wrote, as its one reading left it (`grow_scene`): the facts
        #: the check after writing reads, with no second call to the model. `None` for a
        #: scene whose reading failed.
        self.readings: dict[str, Enrichment | None] = {}
        #: What the person asked for beside taking what a stop showed, in this call
        #: (`keep_wanted`): said back in the reply, so a misread change is seen (#2313).
        self.kept: dict[str, list[str]] = {"must_have": [], "avoid": []}

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

    def _pending_cast(self) -> list[str]:
        status = {p.id: p.status for p, _ in self.work.proposals()[0]}
        return [i for i in self.state.cast_proposals if status.get(i) == "pending"]

    def _cast_text(self) -> str:
        """The cast the chapter is written over: approved, or assumed; never rejected."""
        status = {p.id: p.status for p, _ in self.work.proposals()[0]}
        rejected = {i for i, s in status.items() if s == "rejected"}
        return "\n".join(
            f"- {c['name']}{_also(c)} ({c['kind']}): {c.get('profile', '')}"
            for c, proposal in zip(self.state.cast, self.state.cast_proposals, strict=False)
            if proposal not in rejected
        )

    def _last_text(self, outline: Outline, number: int) -> str | None:
        """The text of the last written scene of chapter `number`, or `None`."""
        if not 1 <= number <= len(outline.chapters):
            return None
        for scene in reversed(outline.chapters[number - 1].scenes):
            found = self.work.manuscript(scene.id)
            if found is not None:
                return found.text
        return None


__all__ = [
    "AuxiliaryStoryModel",
    "Brief",
    "CHAPTERS_PER_CALL",
    "DEFAULTS",
    "DEFAULT_STRUCTURE",
    "ENGINES",
    "EmbedderFactory",
    "GENRE_LABELS",
    "GENRE_OPTIONS",
    "LANGUAGE_NAMES",
    "LANGUAGE_SCRIPTS",
    "Language",
    "MAX_CAST_TRIES",
    "MAX_CHAPTERS",
    "MAX_CHARACTERS",
    "MAX_DRAFTS",
    "MAX_PLACES",
    "MAX_PREMISE_BIGRAM",
    "MAX_PREMISE_COSINE",
    "MAX_SCENES_PER_CHAPTER",
    "MAX_SCENES_WRITTEN",
    "MIN_SCENES_PER_CHAPTER",
    "ModelFactory",
    "PREMISE_COUNT",
    "Premise",
    "PremiseEmbedder",
    "REPEAT_MARK",
    "SCENES_PER_CHAPTER",
    "START_FILE",
    "Similarity",
    "Stage",
    "StartState",
    "StoryModel",
    "StoryStartParams",
    "StructureSource",
    "_Auxiliary",
    "_BRIEF_FIELDS",
    "_FALLBACK_TABLE",
    "_FENCE",
    "_GENRE_STRUCTURES",
    "_GENRE_WORDS",
    "_HEAVY",
    "_INTERACTIVE",
    "_LIGHT",
    "_NAMED_STRUCTURES",
    "_QUICK",
    "_RunBase",
    "_STRAY_QUOTE",
    "_SYSTEM",
    "_aliases",
    "_also",
    "_arc_of",
    "_ask_json",
    "_brief_lines",
    "_default_embedder",
    "_default_model",
    "_dump",
    "_genre_table",
    "_genre_turned_down",
    "_has_word",
    "_ids",
    "_items",
    "_load_state",
    "_pairs",
    "_text",
    "_texts",
    "_title_of",
    "brief_from",
    "cosine",
    "genre_label",
    "genre_options",
    "json_object",
    "logger",
    "muse_genre",
    "names_genre",
    "premise_similarity",
    "scenes_for",
    "story_language",
    "story_scripts",
    "structure_for",
    "turned_down_genres",
    "wants_interactive",
    "wants_it_now",
]
