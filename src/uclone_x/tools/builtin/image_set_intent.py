"""Does a message ask for several images that differ from one another?

Read by two parties that must agree: the turn-level image-set planner
(`uclone_x.agent.image_set_planner`), which plans one prompt per image when the answer is
yes, and `generate_image`, which refuses `count` -- one prompt rendered again with new
seeds -- for such a request. One reading, so the planner and the tool cannot disagree on
the same words ("같은 포즈로 5장 더" is seeds, for both).
"""

from __future__ import annotations

import re

__all__ = ["MAX_IMAGES", "asks_for_variety", "detect_image_set", "wants_varied_locations"]

MAX_IMAGES = 10
"""The most images one plan covers: `generate_image` takes at most ten prompts."""


# --------------------------------------------------------------------------- detection

_KO_NUMBERS = {
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
}
_EN_NUMBERS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_KO_NUM = r"(\d+|" + "|".join(sorted(_KO_NUMBERS, key=len, reverse=True)) + r")"
_EN_NUM = r"\b(\d+|" + "|".join(_EN_NUMBERS) + r")"

# "5장", "다섯 컷": counts pictures directly.
_KO_PICTURES = re.compile(_KO_NUM + r"\s*(?:장|컷)")
# "포즈 5개", "5가지 표정": counts things, which are pictures only beside a variety noun.
_KO_THINGS = re.compile(_KO_NUM + r"\s*(?:개|가지)")
# "다섯 장면", "3개의 포즈": a number directly on a variety noun.
_KO_COUNTED_NOUN = re.compile(_KO_NUM + r"\s*(?:개의\s*)?(?:장면|포즈|표정|의상|구도)")
# "캐릭터 4명": counts people, which are pictures only when each is a different one.
_KO_PEOPLE = re.compile(_KO_NUM + r"\s*명")

_KO_VARIETY_WORD = re.compile(r"다양한|다양하게|여러|각각|각기|서로\s*다른")
_KO_VARIETY_NOUN = re.compile(r"장면|포즈|표정|의상|옷|캐릭터|구도|각도")

_EN_VARIETY_WORD = re.compile(r"\b(?:different|various|varied|distinct|diverse|unique)\b", re.I)
_EN_VARIETY_NOUNS = r"scenes?|poses?|expressions?|outfits?|angles?"
_EN_COUNTED = re.compile(
    _EN_NUM
    + r"\s+(?:[a-z-]+\s+){0,3}?("
    + r"images?|pictures?|pics|illustrations?|drawings?|shots?|panels?|characters?|"
    + _EN_VARIETY_NOUNS
    + r")\b",
    re.I,
)
_EN_VARIETY_NOUN = re.compile(r"^(?:" + _EN_VARIETY_NOUNS + r")$", re.I)

# One picture repeated, or seeds varied: `count` is the right call for these.
# "이 장면 3장 더", "이 의상 그대로": more of a picture already on screen, not new ones.
# Bare "그대로" is not enough: "배경은 그대로 두고, 다른 캐릭터 4명" is a set.
# "이 캐릭터로 다양한 포즈" stays a set -- a known character is what a set is usually of.
_SAME_PICTURE = re.compile(
    r"(?:같은|동일한)\s*(?:그림|구도|장면|이미지|프롬프트|포즈|사진|컷)|똑같|시드|반복"
    r"|(?:^|[\s,])(?:이|그|저)\s+(?:그림|구도|장면|이미지|포즈|사진|컷|의상|옷)"
    r"|\bseeds?\b|\bvariations?\b|\bidentical\b|\bmore\s+of\s+(?:this|these|that|it)\b"
    r"|\b(?:same|this|that|these)\s+(?:image|picture|prompt|composition|pose|scene|shot|outfit)s?\b",
    re.I,
)


def _number(token: str) -> int:
    if token.isdigit():
        return int(token)
    return _KO_NUMBERS.get(token) or _EN_NUMBERS[token.lower()]


def detect_image_set(message: str) -> int | None:
    """How many varied images `message` asks for, or `None` when it asks for no set.

    A set is at least two images that should differ from each other -- scenes, poses,
    expressions, outfits, characters. It is not one picture repeated ("같은 그림 5장"),
    seed variations ("시드만 바꿔서", "variations"), or a single image. Capped at
    `MAX_IMAGES`.
    """
    if _SAME_PICTURE.search(message):
        return None
    count: int | None = None

    for match in _KO_PICTURES.finditer(message):
        if _KO_VARIETY_WORD.search(message) or _KO_VARIETY_NOUN.search(message):
            count = _number(match.group(1))
            break
    if count is None and _KO_VARIETY_NOUN.search(message):
        match = _KO_THINGS.search(message)
        if match is not None:
            count = _number(match.group(1))
    if count is None:
        match = _KO_COUNTED_NOUN.search(message)
        if match is not None:
            count = _number(match.group(1))
    if count is None and _KO_VARIETY_WORD.search(message) and "캐릭터" in message:
        match = _KO_PEOPLE.search(message)
        if match is not None:
            count = _number(match.group(1))
    if count is None and re.search(r"다른\s*캐릭터", message):
        match = _KO_PEOPLE.search(message) or _KO_PICTURES.search(message)
        if match is not None:
            count = _number(match.group(1))
    if count is None:
        for match in _EN_COUNTED.finditer(message):
            if _EN_VARIETY_WORD.search(message) or _EN_VARIETY_NOUN.match(match.group(2)):
                count = _number(match.group(1))
                break

    if count is None or count < 2:
        return None
    return min(count, MAX_IMAGES)


# --------------------------------------------------------------------------- the plan


# Without a count, "여러" and "각각" are too loose: "여러 장 뽑아줘" is a batch of one idea.
_KO_VARIETY_STRICT = re.compile(
    r"다양한|다양하게|여러\s*가지|(?:각각|각기|서로)\s*다른|" + _KO_NUM + r"\s*가지"
)


def asks_for_variety(message: str) -> bool:
    """Whether `message` asks for images that differ, with or without a count.

    `detect_image_set` needs a number; the tool already has one (`count`), so a variety
    word alone ("다양한 포즈로 그려줘") is enough here. Bare nouns (포즈, 표정) are not:
    "같은 포즈로 5장" and "4 versions, each looking at viewer" are seed requests.
    """
    if _SAME_PICTURE.search(message) is not None:
        return False
    if detect_image_set(message) is not None:
        return True
    return bool(_KO_VARIETY_STRICT.search(message) or _EN_VARIETY_WORD.search(message))


# A set of *scenes* (or places) moves between locations; a pose or expression set does not.
_SCENE_WORD = re.compile(r"장면|장소|scenes?\b|places?\b|locations?\b", re.I)
# The user pinned the place: "배경은 그대로", "배경 고정", "같은 장소에서", "same background".
_FIXED_LOCATION = re.compile(
    r"(?:배경|장소)\s*(?:은|는|을|를)?\s*(?:그대로|고정|하나|같게|동일)"
    r"|(?:같은|동일한|한)\s*(?:배경|장소)|에서만"
    r"|\b(?:same|one|single|fixed)\s+(?:background|location|place|setting)\b"
    r"|\b(?:background|location)\s+(?:fixed|stays)\b",
    re.I,
)


def wants_varied_locations(message: str) -> bool:
    """Whether each image of the set should be somewhere else.

    "다양한 장면 5장" is five scenes, and a scene is a place as much as a pose: a plan
    that puts one forest in every image answers a pose study instead. Only when the
    message speaks of scenes or places, and does not pin the background itself.
    """
    return bool(_SCENE_WORD.search(message)) and not _FIXED_LOCATION.search(message)
