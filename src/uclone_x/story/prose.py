"""What a scene's text is refused for when it is written (#1808).

Four faults a small model leaves in prose and a person would not, each measured on
qwen3:8b (#1808):

- a Chinese character stuck to a Korean word, `잔烬`, `둔钝`: the model slipped between
  scripts in the middle of a word;
- a lowercase Latin word stuck to a Korean syllable, `왼blick`, `왼linkplain`: the same
  slip into another script;
- a loop: the same sentences over and over, in 4 of 10 long drafts;
- a note about the writing put after the story, in brackets.

A person's own text is saved through the same tool, so every rule is narrow and a doubt
goes to the text: a rule here misses a fault before it refuses good prose. What each one
lets through is said where it is defined.

A refusal is for a person to read: it quotes the words it is about and says what to do,
in the language of the scene, and names no rule, pattern or field.

This module is pure.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

__all__ = ["prose_refusal"]

_HANGUL = re.compile(r"[가-힣]")

#: A Korean syllable with a Chinese character right after it, in one word. Only this order:
#: hanja written before a particle (`韓國은`) is how mixed-script Korean is written, and
#: `잔烬` and `둔钝` are the slip the model makes. Brackets are taken out first, so
#: `한자(漢字)` is never here.
_GLUED_HAN = re.compile(r"[가-힣][㐀-䶿一-鿿豈-﫿]")
#: A Korean syllable with three or more lowercase Latin letters right after it, in one word.
#: Only this order and only lowercase: Latin before a particle (`AI를`, `iPhone을`) is how
#: Korean writes a foreign name, and two letters (`한국vs일본`) can be meant.
_GLUED_LATIN = re.compile(r"[가-힣][a-z]{3,}")
_BRACKETED = re.compile(r"\([^()]*\)|（[^（）]*）|\[[^\[\]]*\]|［[^［］]*］|【[^【】]*】")
_WORD_EDGE = re.compile(r"^[^\w]+|[^\w]+$")

#: Where a sentence ends: after terminal punctuation, a closing quote after it, then space;
#: or at a line break.
_SENTENCE_END = re.compile(r"(?<=[.!?。！？…])[\"'”’」』]?\s+|\n+")

#: A sentence shorter than this, in letters and digits, is not counted for a loop: short
#: lines repeat in ordinary dialogue (`응.` `응.`, `알았어.`).
_MIN_COUNTED = 8
#: One sentence this many times is a loop. Three is a refrain -- a line said three times
#: is a device prose uses on purpose -- so the rule starts at four.
_SAME_SENTENCE = 4
#: A text with at least this many counted sentences, of which fewer than this share are
#: distinct, is a loop: most of it is said twice or more. At eight sentences the share
#: allows a refrain of three and a pasted-in repeat of half the text; a model's loop of a
#: few sentences over and over falls well under it.
_MIN_SENTENCES = 8
_DISTINCT_SHARE = 0.5

#: A closing note needs this many letters inside its brackets: a short stage direction
#: that ends a scene, `(암전)`, is prose.
_MIN_NOTE = 20
_PAIRS = {"(": ")", "（": "）", "[": "]", "［": "］", "【": "】"}

#: How much of a sentence a refusal quotes.
_QUOTE_CHARS = 60


def _glued(text: str, pattern: re.Pattern[str]) -> list[str]:
    """The words of `text`, outside brackets, where `pattern` finds another script stuck to
    a Korean syllable."""
    words: list[str] = []
    for raw in _BRACKETED.sub(" ", text).split():
        if pattern.search(raw):
            word = _WORD_EDGE.sub("", raw) or raw
            if word not in words:
                words.append(word)
    return words


def _key(sentence: str) -> str:
    """A sentence as a loop is counted: letters and digits only, case folded."""
    return "".join(ch for ch in unicodedata.normalize("NFC", sentence).casefold() if ch.isalnum())


def _quoted(sentence: str) -> str:
    sentence = " ".join(sentence.split())
    if len(sentence) > _QUOTE_CHARS:
        sentence = sentence[: _QUOTE_CHARS - 1] + "…"
    return f'"{sentence}"'


def _loop(text: str) -> tuple[str, int] | None:
    """The sentence a loop repeats most and how many times, or `None` for no loop."""
    counted = [s for s in _SENTENCE_END.split(text) if len(_key(s)) >= _MIN_COUNTED]
    if not counted:
        return None
    times = Counter(_key(s) for s in counted)
    key, most = times.most_common(1)[0]
    looping = most >= _SAME_SENTENCE or (
        len(counted) >= _MIN_SENTENCES and len(times) / len(counted) < _DISTINCT_SHARE
    )
    if not looping:
        return None
    return next(s for s in counted if _key(s) == key), most


def _whole_note(line: str) -> bool:
    """Whether `line` is one bracketed note from its first character to its last."""
    closer = _PAIRS.get(line[:1])
    if closer is None or not line.endswith(closer):
        return False
    depth = 0
    for index, ch in enumerate(line):
        if ch == line[0]:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0 and index != len(line) - 1:
                return False  # `(웃음) 그래 (한숨)`: two notes with prose between
    return True


def _closing_note(text: str) -> str | None:
    """A note in brackets the text ends with, after its prose, or `None`.

    Not when the text is only the note, and not when an earlier line is a whole
    bracketed line too: a script's stage directions are the text's own form.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not _whole_note(lines[-1]):
        return None
    if any(_whole_note(line) for line in lines[:-1]):
        return None
    if sum(1 for ch in lines[-1] if ch.isalnum()) < _MIN_NOTE:
        return None
    return lines[-1]


def prose_refusal(text: str) -> str | None:
    """Why `text` is not saved as a scene, for a person to read; `None` when it is saved.

    Each fault found is one sentence quoting the words it is about; the last sentence
    says that nothing was written and what to do.
    """
    korean = bool(_HANGUL.search(text))
    faults: list[str] = []
    glued = _glued(text, _GLUED_HAN)
    if glued:
        words = ", ".join(f'"{w}"' for w in glued)
        faults.append(
            f"{words}: 한글 낱말 안에 한자가 붙어 있습니다. 한글로 고치거나, 한자를 꼭 "
            "쓰려면 한자(漢字)처럼 괄호 안에 쓰세요."
            if korean
            else f"{words}: a Chinese character is stuck inside a Korean word. Write the word "
            "in Hangul, or put the Chinese characters in brackets after it."
        )
    latin = _glued(text, _GLUED_LATIN)
    if latin:
        words = ", ".join(f'"{w}"' for w in latin)
        faults.append(
            f"{words}: 한글 낱말 안에 영문이 붙어 있습니다. 한글로 고치거나, 영문을 꼭 "
            "쓰려면 띄어 쓰세요."
            if korean
            else f"{words}: Latin letters are stuck inside a Korean word. Write the word in "
            "Hangul, or put a space before the Latin."
        )
    loop = _loop(text)
    if loop is not None:
        sentence, times = loop
        faults.append(
            f"{_quoted(sentence)} 같은 문장이 되풀이됩니다(이 문장만 {times}번). 되풀이한 "
            "부분을 빼고 이야기를 앞으로 나아가게 쓰세요."
            if korean
            else f"The same sentences repeat, {_quoted(sentence)} {times} times. Take out the "
            "repeats and let the scene move on."
        )
    note = _closing_note(text)
    if note is not None:
        faults.append(
            f"장면이 이야기 뒤에 괄호로 된 메모로 끝납니다: {_quoted(note)}. 글에 대한 "
            "설명이라면 빼고, 하고 싶은 말은 답장에 쓰세요."
            if korean
            else f"The scene ends with a note in brackets after the story: {_quoted(note)}. "
            "If it is about the writing, leave it out and say it in your reply instead."
        )
    if not faults:
        return None
    if korean:
        return " ".join(faults) + " 장면은 저장되지 않았습니다. 고친 글로 다시 저장하세요."
    return " ".join(faults) + " The scene was not saved. Save it again once it is fixed."
