"""Which specifics in an answer came from nothing the turn read (#697).

## The gap this fills

`require_evidence_before_answer` (#702) asks the model again when a turn ends having
executed **no** tool. On the baseline that produced #697 that rule reaches 28 of 100
problems. It cannot reach the rest: the median turn there executed **one** tool call
against a median declared `horizon` of **six** dependent steps, and one call is enough to
pass a not-zero test and nothing else. The loop still ended on the first silence, and the
answer still named a default, a path or a count that no tool output contained.

`horizon` is evaluation metadata. A production agent has none, so the runtime signal for
"stopped early" cannot be built from it. What the runtime does have, at the moment the
model falls silent, is the answer and everything the turn read to produce it. A **specific**
in the first that appears nowhere in the second was not looked up.

## What counts as a specific, and what deliberately does not

Only two classes are collected:

*   a **path or filename** -- a slash-separated token, or a bare name ending in a source
    extension. An agent holding a workspace that names `agent/models.py` either opened it
    or guessed it.
*   a **multi-digit number** -- the observed case verbatim. Asked for the default
    `max_steps`, the baseline agent answered `100` and volunteered that the value was
    "inferred from common system defaults".

Single digits are excluded on purpose. A small count is the figure most likely to have been
*derived* from what was read ("three subclasses") rather than copied out of it, so it is
correct and unsupported at once. The exclusion trades recall for precision in the direction
where being wrong costs a real model round-trip.

Nothing else is collected. Deciding whether an English clause follows from a tool output
requires a model, and a runtime check that needs a model to run is not a runtime check.

## What this is not

It is **not** a fabrication detector and it does not decide whether an answer is true. A
specific found here may be perfectly correct and merely re-derived, and a specific absent
from here proves nothing.

## The boundary, enumerated

"The support test is a substring search, so `100` is supported by an output containing
`1000`" understates it badly, and a reader deciding how far to trust the signal needs the
real shape. Every row below was re-derived against this module, each paired with a positive
control showing that the bare form of the same case does fire.

**It stays silent on a specific that came from nowhere when:**

*   **the number has an unrecognized suffix.** Recognized units in the closed suffix list
    (`ms`, `kb`, `mb`, `gb`, `tb`, `s`, `m`, `h`, `d`, `b`, `%`) and hex prefixes (`0x`)
    are collected and normalized (#740). An unlisted suffix attached to digits (e.g. `30px`,
    `100rpm`) falls outside `_NUMBER_RE`.
*   **the location is a dotted module name.** `uclone_x.agent.models` reaches neither
    `_PATH_RE`, which needs a slash, nor `_FILENAME_RE`, which needs a listed extension --
    though it is how a Python agent most naturally names a place.
*   **the filename sits inside a URL.** `https://example.com/config.yaml` detects nothing
    at all: the slash before the name defeats `_FILENAME_RE`'s lookbehind and the `://`
    defeats `_PATH_RE`.
*   **the name has no extension.** `README`, `Makefile`, `Dockerfile`, `LICENSE` fall
    outside both collected classes by construction.
*   **the turn read a lot.** Support is one concatenation searched as a substring, so a
    two-digit figure is "supported" by any output that happens to contain those digits --
    line numbers, byte counts, timestamps. `50` against a listing printing lines 49, 50 and
    51 reports nothing. Recall therefore *falls as the turn reads more*. That is the
    favourable direction, since the check bites hardest on exactly the low-read turns #697
    is about, but it also means `unsupported_claims` is systematically emptier for
    well-behaved turns and a consumer (#733) must not read emptiness as cleanliness.

**It fires on something legitimate when:**

*   **the figure was derived rather than copied.** "There are 12 subclasses" over three
    class definitions; "the total is 57" over `a=25` and `b=32`. This is the tradeoff the
    single-digit exclusion prices deliberately: one model round-trip, paid to keep the
    check able to see a two-digit default.
*   **the number is a bare year.** `2024` is multi-digit and nothing here knows better.
*   **the path is more specific than its support.** An agent that lists a subdirectory and
    then reports the full repo-relative path is nudged, because the substring test runs in
    one direction only -- and that is ordinary, correct agent behaviour.
*   **a range is quoted.** In `10-20` the `10` is a claim and the `20` is not, because
    `_NUMBER_RE`'s lookbehind excludes the hyphen. A nudge quoting half a range back is a
    confusing message.
*   **a slashed English idiom looks like a path.** `he/she/they` has two slashes, which is
    all `_looks_like_a_path` asks for.

Everything in the first list runs toward silence and everything in the second costs exactly
one model round-trip, which is the direction this check declares for itself. Neither list
is closed, and neither is a bug list: widening either collected class changes what the
runtime spends steps on, so it is a decision with its own measurement, not a tidy-up.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

#: Extensions that make a dotted token a filename rather than a sentence ending. Without
#: this list every English full stop followed by a word is a path, the check fires on
#: every answer, and firing on everything says exactly as much as firing on nothing.
_EXTENSIONS = (
    "py|pyi|md|rst|txt|json|ya?ml|toml|ini|cfg|lock|sh|bash|zsh|"
    "js|jsx|ts|tsx|html|css|scss|sql|rs|go|java|rb|c|h|cc|cpp|hpp"
)

#: A slash-separated token. Filtered afterwards: a single slash with no extension is
#: `and/or`, not a path.
_PATH_RE = re.compile(r"(?<![\w./-])[\w.-]*[\w-]/[\w.\-/]*[\w]")

_FILENAME_RE = re.compile(rf"(?<![\w./-])[\w-]+\.(?:{_EXTENSIONS})(?![\w])")

#: Unit suffixes that may be glued to a number (e.g. 30s, 100ms, 64KB, 50%).
#: Longer suffixes must precede shorter ones so `ms` matches before `s` or `m`.
_UNIT_SUFFIXES = "ms|kb|mb|gb|tb|s|m|h|d|b|%"

_HEX_NUMBER = r"0[xX][0-9a-fA-F](?:[0-9a-fA-F_]*[0-9a-fA-F])?"
_DECIMAL_NUMBER = rf"\d(?:[\d,_]*\d)?(?:\.\d+)?(?:(?i:{_UNIT_SUFFIXES}))?"

#: A number not glued to a word (unless matching a recognized unit suffix or hex prefix),
#: so the `2` inside `v2` and inside `models2.py` is not a claim of its own. Thousands
#: separators are allowed inside and normalised away before comparison.
_NUMBER_RE = re.compile(rf"(?<![\w.,_-])(?:{_HEX_NUMBER}|{_DECIMAL_NUMBER})(?![\w])")

#: An enumerated line's marker. Stripped before the number scan: `10.` opening a line is
#: formatting, and it is two digits, so the single-digit exclusion would not catch it and
#: every long enumerated answer would be reported.
_LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*\d+[.)](?=[ \t])")

_DIGITS_RE = re.compile(r"(?<=[\da-fA-F])_(?=[\da-fA-F])|(?<=\d),(?=\d)")

_UNIT_STRIP_RE = re.compile(rf"^(.*?)(?:(?i:{_UNIT_SUFFIXES}))$")
_HEX_STRIP_RE = re.compile(r"^0[xX](.*)$")

#: How many findings one nudge may quote. A nudge listing forty tokens is a second prompt
#: rather than a question about the first.
DEFAULT_LIMIT = 8


def _normalise_number(text: str) -> str:
    """`60,000` and `60_000` and `60000` are one figure, not three."""
    return _DIGITS_RE.sub("", text)


def _strip_number_unit_or_prefix(text: str) -> str:
    """Strips recognized unit suffix or 0x hex prefix from a normalized number token."""
    hex_m = _HEX_STRIP_RE.match(text)
    if hex_m is not None:
        return hex_m.group(1)
    unit_m = _UNIT_STRIP_RE.match(text)
    if unit_m is not None:
        return unit_m.group(1)
    return text


def _looks_like_a_path(token: str) -> bool:
    """True for `a/b/c` and `dir/file.py`; false for `and/or`."""
    if token.count("/") >= 2:
        return True
    tail = token.rsplit("/", 1)[-1]
    return _FILENAME_RE.fullmatch(tail) is not None


def _candidates(answer: str) -> list[tuple[int, str, bool]]:
    """Every specific in `answer` as `(position, text, is_number)`, first appearance first."""
    found: list[tuple[int, str, bool]] = []
    claimed: list[tuple[int, int]] = []

    for match in _PATH_RE.finditer(answer):
        if _looks_like_a_path(match.group()):
            found.append((match.start(), match.group(), False))
            claimed.append(match.span())

    for match in _FILENAME_RE.finditer(answer):
        if any(start <= match.start() and match.end() <= end for start, end in claimed):
            continue
        found.append((match.start(), match.group(), False))
        claimed.append(match.span())

    # Numbers are scanned over a copy with the enumeration markers blanked, so positions
    # still line up with the original answer.
    numbered = _LIST_MARKER_RE.sub(lambda m: " " * len(m.group()), answer)
    for match in _NUMBER_RE.finditer(numbered):
        if any(start <= match.start() < end for start, end in claimed):
            continue
        token = match.group()
        norm = _normalise_number(token)
        stripped = _strip_number_unit_or_prefix(norm)
        digits = stripped.replace(".", "")
        if len(digits) < 2:
            continue
        found.append((match.start(), match.group(), True))

    found.sort(key=lambda item: item[0])
    return found


def unsupported_specifics(
    answer: str,
    supports: Iterable[str],
    limit: int = DEFAULT_LIMIT,
) -> tuple[str, ...]:
    """Specifics `answer` asserts that appear in none of `supports`.

    `supports` is everything the turn read or was told -- tool outputs, the user's prompt,
    the system prompt. A figure quoted back from the question was not invented, so the
    prompt belongs in here; treating tool output as the only support makes the check call
    the user's own words a guess.

    Returns at most `limit` findings, deduplicated case-insensitively and ordered by first
    appearance in `answer`, so the message built from them is stable run to run.
    """
    if not answer.strip():
        return ()

    supported = "\n".join(supports).lower()
    supported_numbers = _normalise_number(supported)

    seen: set[str] = set()
    out: list[str] = []
    for _, text, is_number in _candidates(answer):
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        if is_number:
            if _normalise_number(text) in supported_numbers:
                continue
            norm = _normalise_number(lowered)
            if norm in supported_numbers:
                continue
            stripped = _strip_number_unit_or_prefix(norm)
            if stripped != norm and stripped in supported_numbers:
                continue
        elif lowered in supported:
            continue
        out.append(text)

    return tuple(out[:limit])


def describe(findings: Sequence[str]) -> str:
    """The findings as one comma-separated clause, for a log line or a nudge."""
    return ", ".join(findings)
