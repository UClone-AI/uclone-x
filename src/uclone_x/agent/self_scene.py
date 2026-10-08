"""`show_self`: the clone draws itself in a scene, from its own self facts (#2017).

clone-self-and-scenes §4. Three pure parts and one tool:

* `project_self` turns the clone's active `self` facts into a `SelfLook`: the tags and
  gender of the story codex's `Visual`, which the one character composer
  (`compose_character_prompt`) reads for a story character too (§4.1);
* `clone_seed` fixes one base seed per clone, so the same facts in the same scene give the
  same person;
* `check_self_scene` is the guard (§4.4), a check in code before any generation;
* `ShowSelfTool` composes the prompt and draws through the registered `generate_image`
  tool, whose engine settings are bound only on that instance (§4.2, §4.3).

This module is kernel: it reaches the image tool through `ToolProtocol` and
`ImageModelSource`, and the facts and the avatar through callables the agent hands it, so
it imports no adapter. It imports nothing from the story package either: importing the
agent package must not load it (#2012), so the look is a `SelfLook` and not a `Visual`.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import PlainRefusalError
from uclone_x.memory.models import MemoryFact
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.character_prompt import compose_character_prompt
from uclone_x.tools.builtin.media_registry import ImageModelSource, PromptFamily
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.protocols import ToolProtocol

__all__ = [
    "ADULT_AGE",
    "MINOR_TERMS",
    "SEXUAL_TERMS",
    "SHOW_SELF_TOOL",
    "SelfLook",
    "ShowSelfParams",
    "ShowSelfTool",
    "check_self_scene",
    "clone_seed",
    "parse_age",
    "project_self",
    "self_scene_path",
]

SHOW_SELF_TOOL: Final = "show_self"

#: The relations that draw, in the order their tags go into a prompt (§4.1). Every other
#: self fact (personality, speech style, age) is left out: it does not draw.
APPEARANCE_RELATIONS: Final[tuple[str, ...]] = ("hair", "eyes", "build", "appearance", "outfit")

#: The youngest age a clone is drawn at (§4.4).
ADULT_AGE: Final = 18

#: Words that code a character as a minor (§4.4), matched as whole words in Latin script and
#: as substrings in Hangul. A parameter of `check_self_scene`, so a test passes its own.
MINOR_TERMS: Final[frozenset[str]] = frozenset(
    {
        "child",
        "children",
        "kid",
        "kids",
        "toddler",
        "infant",
        "underage",
        "schoolgirl",
        "schoolboy",
        "school uniform",
        "elementary school",
        "middle school",
        "junior high",
        "high school",
        "student",
        "students",
        "teen",
        "teens",
        "teenager",
        "teenagers",
        "under 18",
        "loli",
        "shota",
        "small child",
        "초등학생",
        "중학생",
        "고등학생",
        "학생",
        "10대",
        "교복",
        "어린이",
        "유치원",
        "아동",
        "미성년",
    }
)

#: Words that make a scene argument sexual (§4.4). A parameter of `check_self_scene`, so a
#: test classifies a neutral marker instead of writing explicit content.
SEXUAL_TERMS: Final[frozenset[str]] = frozenset(
    {
        "sex",
        "sexual",
        "sexy",
        "nsfw",
        "nude",
        "naked",
        "erotic",
        "lewd",
        "explicit",
        "topless",
        "lingerie",
        "undressed",
        "hentai",
        "섹스",
        "야한",
        "누드",
        "알몸",
        "벗은",
    }
)

_GENDER_WORDS: Final[dict[str, Literal["female", "male"]]] = {
    "female": "female",
    "woman": "female",
    "girl": "female",
    "f": "female",
    "여자": "female",
    "여성": "female",
    "male": "male",
    "man": "male",
    "boy": "male",
    "m": "male",
    "남자": "male",
    "남성": "male",
}

NO_AGE_TEXT: Final = (
    "No picture was drawn: this clone has no age yet, and it shows itself only as an adult. "
    'Tell it its age as a number, for example "you are 25", and ask again.'
)
ONCE_PER_TURN_TEXT: Final = (
    "No picture was drawn: the clone shows itself at most once per reply, and this reply "
    "already has a picture of it. Ask again in your next message."
)
MINOR_SEXUAL_TEXT: Final = (
    "No picture was drawn: a sexual scene is never drawn of a character described as a "
    "child or a student, and this clone's description or this scene says so."
)
DRAW_FAILED_TEXT: Final = "No picture was drawn this time, and nothing was saved. Try again later."
NO_REFERENCE_TEXT: Final = "no avatar set"
REFERENCE_UNUSED_TEXT: Final = (
    "not used: the image model cannot take a reference image, so the picture was drawn "
    "from the appearance facts alone"
)
REFERENCE_UNKNOWN_TEXT: Final = "not used: no reference image is passed to the image model"
NO_APPEARANCE_NOTE: Final = (
    "No appearance facts are set, so the image model chose the look. Tell the clone how it "
    "looks (hair, eyes, outfit) to keep it the same in every picture."
)


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def _tags_of(value: str) -> list[str]:
    return [t for t in (" ".join(part.split()) for part in value.split(",")) if t]


def _gender_of(value: str) -> Literal["female", "male", "other"]:
    return _GENDER_WORDS.get(_norm(value), "other")


@dataclass(frozen=True)
class SelfLook:
    """How the clone looks: the part of a codex `Visual` that self facts give."""

    tags: tuple[str, ...]
    gender: Literal["female", "male", "other"] | None

    def sheet(self) -> dict[str, Any]:
        """As the character sheet `compose_character_prompt` reads, for one character."""
        return {
            "character_id": "self",
            "name": "self",
            "gender": self.gender,
            "danbooru_tags": ", ".join(self.tags),
        }


def project_self(facts: Iterable[MemoryFact], *, outfit: str | None = None) -> SelfLook:
    """The clone's look, from its active self facts (§4.1). Pure.

    `hair`, `eyes`, `build`, `appearance` and `outfit` become tags, in that order and each
    tag once; `gender` sets `gender`; everything else is left out. `outfit`, when given,
    replaces the `outfit` facts for this projection only. Retracted facts are skipped, so a
    caller handing the whole history gets what recall shows.
    """
    by_relation: dict[str, list[str]] = {relation: [] for relation in APPEARANCE_RELATIONS}
    gender: Literal["female", "male", "other"] | None = None
    for fact in facts:
        if fact.retracted:
            continue
        relation = _norm(fact.predicate)
        if relation == "gender":
            gender = _gender_of(fact.object_value)
        elif relation in by_relation:
            by_relation[relation].extend(_tags_of(fact.object_value))
    if outfit is not None:
        by_relation["outfit"] = _tags_of(outfit)
    tags: list[str] = []
    seen: set[str] = set()
    for relation in APPEARANCE_RELATIONS:
        for tag in by_relation[relation]:
            if _norm(tag) not in seen:
                seen.add(_norm(tag))
                tags.append(tag)
    return SelfLook(tags=tuple(tags), gender=gender)


def clone_seed(clone_id: str) -> int:
    """One base seed per clone, stable across processes: so the same facts draw one person."""
    return int.from_bytes(hashlib.sha256(clone_id.encode("utf-8")).digest()[:4], "big")


def _mentions(text: str, terms: Iterable[str]) -> str | None:
    """The first of `terms` that `text` contains: a whole word in Latin script, else as is."""
    folded = _norm(text)
    for term in sorted(terms):
        key = _norm(term)
        if not key:
            continue
        if key.isascii():
            if re.search(rf"(?<![\w]){re.escape(key)}(?![\w])", folded):
                return term
        elif key in folded:
            return term
    return None


_UNCLEAR_AGE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(?:under|below|less\s+than|over|above|more\s+than|around|about|approx(?:imately)?)\b"
    r"|미만|이하|이상|초과|\d+대"
)


def parse_age(value: str, *, now: datetime | None = None) -> int | None:
    """Parse an age in years from a memory fact value, or None if missing or unclear.

    Handles explicit ages ("25", "25세", "25 years old") and birth years ("born 2010",
    "2010년생", "1995") evaluated against the current UTC year. Bounded or relative
    modifiers ("under 18", "18세 미만", "around 25", "10대") return None.
    """
    current_year = (now or datetime.now(UTC)).year
    folded = value.strip()
    if _UNCLEAR_AGE_PATTERN.search(folded):
        return None

    birth_match = (
        re.search(r"(?i)(?:born\s*(?:in\s*)?|birth\s*year\s*(?:is\s*)?|b\.\s*)(\d{4})", folded)
        or re.search(r"(\d{4})\s*년\s*(?:생|출생)?", folded)
        or re.search(r"\b(\d{4})\b", folded)
    )
    if birth_match is not None:
        year = int(birth_match.group(1))
        if 1900 <= year <= current_year:
            age = current_year - year
            return age if 0 <= age <= 120 else None
        return None

    direct_match = re.search(r"(\d{1,3})\s*(?:세|살)", folded) or re.search(
        r"\b(\d{1,3})\b", folded
    )
    if direct_match is not None:
        num = int(direct_match.group(1))
        return num if 1 <= num <= 120 else None

    return None


def check_self_scene(
    facts: Sequence[MemoryFact],
    scene: Sequence[str],
    *,
    minor_terms: Iterable[str] = MINOR_TERMS,
    sexual_terms: Iterable[str] = SEXUAL_TERMS,
    now: datetime | None = None,
) -> None:
    """Refuse, in plain words, a self scene that must not be drawn (§4.4). Pure.

    An active `age` fact whose age is `ADULT_AGE` or more is required; several age
    facts are read by the youngest. A scene whose arguments (`scene`) mention a sexual term
    is refused when any self fact or scene argument codes the character as a minor,
    whatever the age says.
    """
    active = [fact for fact in facts if not fact.retracted]
    active_age_facts = [fact for fact in active if _norm(fact.predicate) == "age"]
    ages: list[int] = []
    for fact in active_age_facts:
        age = parse_age(fact.object_value, now=now)
        if age is None:
            ages.clear()
            break
        ages.append(age)
    if not ages:
        raise PlainRefusalError(NO_AGE_TEXT, reason_code="no_age")
    youngest = min(ages)
    if youngest < ADULT_AGE:
        raise PlainRefusalError(
            f"No picture was drawn: this clone's age is {youngest}, and it shows itself only "
            f"as an adult ({ADULT_AGE} or older). If that age is wrong, tell it the right one.",
            reason_code="under_age",
        )
    sexual_terms = tuple(sexual_terms)
    minor_terms = tuple(minor_terms)
    if not any(_mentions(part, sexual_terms) for part in scene):
        return
    described = [fact.object_value for fact in active] + list(scene)
    if any(_mentions(text, minor_terms) for text in described):
        raise PlainRefusalError(MINOR_SEXUAL_TEXT, reason_code="minor_coded")


def _slug(clone_id: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", clone_id.casefold()).strip("-")
    return slug[:40] or "clone"


def self_scene_path(clone_id: str, now: datetime, suffix: str) -> str:
    """`artifacts/images/self_<clone>_<UTC yyyymmddHHMMSS>_<hex>.png` (§4.2): never the avatar."""
    stamp = now.astimezone(UTC).strftime("%Y%m%d%H%M%S")
    return f"artifacts/images/self_{_slug(clone_id)}_{stamp}_{suffix}.png"


def _place_phrase(place: str) -> str:
    first = place.split(" ", 1)[0].casefold()
    return place if first in {"in", "at", "on", "by", "near", "under", "inside"} else f"in {place}"


def _prose_prompt(visual: SelfLook, scene: list[str]) -> str:
    noun = {"female": "woman", "male": "man"}.get(visual.gender or "", "person")
    looks = f" ({', '.join(visual.tags)})" if visual.tags else ""
    return f"A {noun}{looks}, " + ", ".join(scene) + ". One person, the same character."


class ShowSelfParams(BaseModel):
    """Where the clone is and what it is doing, for one picture of itself."""

    model_config = ConfigDict(extra="forbid", strict=True)

    place: str = Field(min_length=1, max_length=200, description="Where you are, in English.")
    action: str = Field(
        min_length=1, max_length=200, description="What you are doing there, in English."
    )
    expression: str | None = Field(
        default=None, max_length=100, description="Your expression, in English, if it matters."
    )
    outfit: str | None = Field(
        default=None,
        max_length=200,
        description="What you wear in this picture only, in English. Leave it out to wear "
        "what your appearance facts say.",
    )


class ShowSelfTool(BaseTool[ShowSelfParams]):
    """Draw the calling clone in a scene from its own self facts (clone-self-and-scenes §4).

    Bound to one agent: `facts` reads that clone's active self facts and `avatar_present`
    whether it has a picture. It draws through `image_tool`, the registered `generate_image`
    instance, so the engine the person chose in Settings is the one used. `writes_files` is
    False, as `generate_image`'s is (#2079): the picture goes to the artifacts folder.
    """

    name = SHOW_SELF_TOOL
    writes_files: ClassVar[bool] = False
    description = (
        "Show yourself in a scene: draws one picture of you, built from your own appearance "
        "facts, so you look the same every time. Use it when the person asks to see you, or "
        "on your own when the conversation takes you somewhere new; at most once per reply. "
        "Give place and action (and expression or outfit if they matter), in English. "
        "Returns relative_url; show it with ![description](relative_url)."
    )
    params_type = ShowSelfParams

    def __init__(
        self,
        *,
        clone_id: str,
        image_tool: ToolProtocol,
        facts: Callable[[], Sequence[MemoryFact]],
        avatar_present: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(name=self.name, params_type=ShowSelfParams)
        self._clone_id = clone_id
        self._image_tool = image_tool
        self._facts = facts
        self._avatar_present = avatar_present
        # (agent id, session id) -> the last turn a picture was started in (§4.3).
        self._drawn_turns: dict[tuple[str, str], int] = {}

    def _family(self, image_model: str | None) -> PromptFamily | None:
        """The prompt family of the model this clone draws with (its own, else the default's).

        ``image_model`` is the clone's own picture model from its `ToolContext`, the one
        `generate_image` draws with, so a clone on a prose model never gets tags (#2176).
        """
        if isinstance(self._image_tool, ImageModelSource):
            return self._image_tool.active_profile(image_model).family
        return None

    def _claim_turn(self, context: ToolContext) -> None:
        """Refuse a second picture in one turn; checked and set before any await."""
        turn = context.turn_index
        if not turn and context.agent_delegate is not None:
            counter = getattr(context.agent_delegate, "_turn_counter", 0)
            turn = counter if isinstance(counter, int) else 0
        key = (context.agent_id, context.session_id)
        if self._drawn_turns.get(key) == turn:
            raise PlainRefusalError(ONCE_PER_TURN_TEXT, reason_code="once_per_turn")
        self._drawn_turns.pop(key, None)
        self._drawn_turns[key] = turn
        if len(self._drawn_turns) > 64:
            del self._drawn_turns[next(iter(self._drawn_turns))]

    def _image_request(
        self, params: ShowSelfParams, visual: SelfLook, image_model: str | None
    ) -> dict[str, Any]:
        scene = [params.action.strip(), _place_phrase(params.place.strip())]
        if params.expression and params.expression.strip():
            scene.append(f"{params.expression.strip()} expression")
        request: dict[str, Any] = {
            "seed_override": clone_seed(self._clone_id),
            "aspect_ratio": "3:4",
            "output_path": self_scene_path(self._clone_id, datetime.now(UTC), secrets.token_hex(3)),
        }
        if self._family(image_model) is PromptFamily.DANBOORU:
            composed = compose_character_prompt([visual.sheet()], ", ".join(scene))
            request["prompt"] = composed["composed_danbooru_prompt"]
            request["negative_prompt"] = composed["composed_negative_prompt"]
            request["style"] = "anime"
        else:
            request["prompt"] = _prose_prompt(visual, scene)
        return request

    def _reference_image(self) -> str:
        if self._avatar_present is None:
            return REFERENCE_UNKNOWN_TEXT
        return REFERENCE_UNUSED_TEXT if self._avatar_present() else NO_REFERENCE_TEXT

    async def run(self, params: ShowSelfParams, context: ToolContext) -> ToolResult:
        """Guard, compose, draw once, and return a compact result naming the picture."""
        facts = list(self._facts())
        scene_args = [
            text for text in (params.place, params.action, params.expression, params.outfit) if text
        ]
        check_self_scene(facts, scene_args)
        visual = project_self(facts, outfit=params.outfit)
        self._claim_turn(context)
        drawn = await self._image_tool.execute(
            self._image_request(params, visual, context.image_model), context
        )
        output = drawn.output if isinstance(drawn.output, dict) else {}
        url = output.get("relative_url")
        if not drawn.success or not isinstance(url, str):
            raise PlainRefusalError(_plain_failure(drawn), reason_code="not_drawn")
        line = f"you, {params.action.strip()}, {_place_phrase(params.place.strip())}"
        if params.expression and params.expression.strip():
            line += f", {params.expression.strip()}"
        if params.outfit and params.outfit.strip():
            line += f", wearing {params.outfit.strip()}"
        result: dict[str, Any] = {
            "status": "success",
            "relative_url": url,
            "drawn": line,
            "reference_image": self._reference_image(),
        }
        if not visual.tags:
            result["note"] = NO_APPEARANCE_NOTE
        # The image tool's own result, with its declared files and provenance, carrying the
        # compact output: the full prompt stays in the picture's sidecar (§4.2).
        return drawn.model_copy(update={"output": result})


#: The prefixes `BaseTool.execute` puts on a failure that is not already plain words.
_INTERNAL_FAILURE_PREFIXES: Final = (
    "Tool execution failed for",
    "Path traversal violation",
    "Tool execution requires",
)


def _plain_failure(result: ToolResult) -> str:
    """The image tool's own refusal when it is plain words, else one plain sentence."""
    error = (result.error or "").strip()
    if not error and isinstance(result.output, dict):
        error = str(result.output.get("message") or result.output.get("error") or "").strip()
    if not error or error.startswith(_INTERNAL_FAILURE_PREFIXES) or "Traceback" in error:
        return DRAW_FAILED_TEXT
    return error
