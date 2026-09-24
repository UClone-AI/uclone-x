"""A specific an answer asserts must have come from somewhere the turn actually read.

## What this pins, and why it is not the check that already exists

#702 closed the first half of #697: a turn that executed **no** tool and was configured to
require evidence gets one more round. That rule fires on 28 of the 100 baseline problems --
the ones answered with nothing at all behind them. It cannot fire on the other 59, because
the median turn in that baseline executed **one** tool call against a median declared
`horizon` of **six** dependent steps. One call satisfies "not zero" and satisfies nothing
else: the loop still ends on the first silence, and the answer still names a path, a
default, or a count that no tool output ever contained.

So the runtime needs a signal that says *not enough*, not merely *none*. `horizon` is
evaluation metadata -- a production agent has none -- so the signal has to be derivable
from the turn itself. This module pins the one chosen: a **specific** in the answer that
appears in nothing the turn read and in nothing it was told.

## Why "specific" is drawn narrowly

The check is a substring search over what the turn saw, so its precision is entirely a
property of what counts as a specific. Two classes are collected and the rest deliberately
are not:

*   a **path or filename** -- something with a directory separator or a source-file
    extension. A turn holding a workspace that names `agent/models.py` either read it or
    guessed it.
*   a **multi-digit number**. This is the `max_steps` case verbatim: the baseline agent
    answered **100** and said the value was "inferred from common system defaults".

Single digits are excluded, and that exclusion is load-bearing rather than tidy: a small
count is the answer most likely to have been *derived* ("three subclasses") rather than
read, and a derived number is correct while appearing in no output. Excluding them trades
recall for precision in the direction where a false positive costs a real model round-trip.

Prose, adjectives and claims of fact are not collected at all. Deciding whether an English
sentence follows from a tool output needs a model, and a runtime check that needs a model
to run is not a runtime check.
"""

from __future__ import annotations

import pytest

from uclone_x.agent.grounding import unsupported_specifics


def test_a_number_the_turn_never_read_is_unsupported() -> None:
    """The #697 replay, reduced: the answer states a default it did not look up.

    Killed by: src/uclone_x/agent/grounding.py :: found.append((match.start(), match.group(), True))
    """
    found = unsupported_specifics(
        "The default max_steps is 100.",
        supports=("def run(self) -> None: ...", "what is the default max_steps?"),
    )

    assert found == ("100",), found


def test_a_number_present_in_a_tool_output_is_supported() -> None:
    """The same answer, after the agent actually read the file, is silent.

    Without this the check fires on every answer carrying a figure, including the correct
    ones, and the nudge becomes a tax on the behaviour it exists to produce.

    Killed by: src/uclone_x/agent/grounding.py :: if _normalise_number(text) in supported_numbers:
    """
    found = unsupported_specifics(
        "The default max_steps is 50.",
        supports=("max_steps: int = Field(default=50)",),
    )

    assert found == (), found


def test_a_number_the_user_supplied_is_supported() -> None:
    """A figure quoted back from the question was not invented.

    The prompt is part of what the turn read. Treating only tool output as support would
    flag `"you asked about 100"` as a fabrication, which is the check calling the user's
    own words a guess.

    Killed by: src/uclone_x/agent/grounding.py :: supported = "\n".join(supports).lower()
    Becomes: supported = ""
    """
    found = unsupported_specifics(
        "You asked about 100, and the configured value is different.",
        supports=("is the default 100?",),
    )

    assert found == (), found


def test_a_single_digit_is_not_collected() -> None:
    """Small counts are usually derived, and a derived count is correct and unsupported.

    Killed by: src/uclone_x/agent/grounding.py :: if len(digits) < 2:
    """
    found = unsupported_specifics(
        "There are 3 of them.",
        supports=("class A: ...\nclass B: ...\nclass C: ...",),
    )

    assert found == (), found


def test_a_path_the_turn_never_read_is_unsupported() -> None:
    r"""A file named but never opened is the same failure wearing a path.

    Killed by: src/uclone_x/agent/grounding.py :: _PATH_RE = re.compile(r"(?<![\w./-])[\w.-]*[\w-]/[\w.\-/]*[\w]")
    Becomes: _PATH_RE = re.compile(r"(?<![\w./-])[\w.-]*[\w-]//[\w.\-/]*[\w]")
    """
    found = unsupported_specifics(
        "It is set in src/uclone_x/agent/models.py near the top.",
        supports=("total 0",),
    )

    assert found == ("src/uclone_x/agent/models.py",), found


def test_a_path_that_appeared_in_a_listing_is_supported() -> None:
    """The same path, after a listing that contains it, is silent.

    The pair with the test above is what makes either meaningful: one shows the check
    fires on an unread path, this one shows it stops when the path was read. Without the
    second, a check that reports every path in every answer would pass the first.

    Killed by: src/uclone_x/agent/grounding.py :: elif lowered in supported:
    """
    found = unsupported_specifics(
        "It is set in src/uclone_x/agent/models.py near the top.",
        supports=("src/uclone_x/agent/base.py\nsrc/uclone_x/agent/models.py\n",),
    )

    assert found == (), found


def test_support_that_spelled_the_name_in_another_case_still_supports() -> None:
    """A tool output that shouted the filename still read it.

    The answer side is lowercased before the comparison (`lowered = text.lower()`), so the
    support side has to be lowercased too or the comparison is only ever case-sensitive in
    one direction. A heading, a log line, a `grep` hit on a constant, or a listing taken
    from a case-preserving filesystem prints `CONFIG.YAML`; the model writes `config.yaml`;
    the check then reports a file the turn demonstrably opened as invented, and spends a
    round-trip asking about it.

    The paired positive control is the same answer against a support naming a *different*
    file, which does fire -- without that pair a check that never reported anything would
    pass this test.

    Killed by: src/uclone_x/agent/grounding.py :: .join(supports).lower()
    Becomes: .join(supports)
    """
    shouted = unsupported_specifics(
        "The loader reads config.yaml.",
        supports=("## CONFIG.YAML\nParsed by the loader at startup.",),
    )
    control = unsupported_specifics(
        "The loader reads config.yaml.",
        supports=("## OTHER.YAML\nParsed by the loader at startup.",),
    )

    assert shouted == (), shouted
    assert control == ("config.yaml",), control


def test_a_bare_filename_with_a_source_extension_is_collected() -> None:
    r"""`models.py` with no directory is still a specific claim about a tree.

    Killed by: src/uclone_x/agent/grounding.py :: _FILENAME_RE = re.compile(rf"(?<![\w./-])[\w-]+\.(?:{_EXTENSIONS})(?![\w])")
    Becomes: _FILENAME_RE = re.compile(rf"(?<![\w./-])[\w-]+\.(?:{_EXTENSIONS})(?<!\.py)(?![\w])")
    """
    found = unsupported_specifics("Look in models.py.", supports=("nothing here",))

    assert found == ("models.py",), found


def test_an_ordinary_sentence_ending_is_not_read_as_a_filename() -> None:
    """Negative control: the extension list is what stops every sentence being a path.

    Without it `"...the agent. It"` is a dotted token and every answer in English is
    unsupported, which is the check firing on 100 percent of turns and therefore saying
    nothing.
    """
    found = unsupported_specifics(
        "It depends on the agent. It cannot be determined from here.",
        supports=(),
    )

    assert found == (), found


def test_a_markdown_list_marker_is_not_a_claimed_number() -> None:
    r"""`1.` opening a line is formatting, not an assertion that the value is one.

    It is excluded before the digit rule rather than after, because a two-digit list
    (`10.`) would otherwise survive the single-digit exclusion and flag every long
    enumerated answer.

    Killed by: src/uclone_x/agent/grounding.py :: _LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*\d+[.)](?=[ \t])")
    Becomes: _LIST_MARKER_RE = re.compile(r"^[ \t]*\d+[.)](?=[ \t])")
    """
    found = unsupported_specifics(
        "10. the tenth item\n11. the eleventh item",
        supports=(),
    )

    assert found == (), found


def test_findings_are_deduplicated_and_ordered_by_first_appearance() -> None:
    """One nudge names each unsupported specific once, in the order the answer made them.

    An unordered or repeating list makes the nudge text depend on set iteration, which is
    a message to a model that differs run to run for no reason.

    Killed by: src/uclone_x/agent/grounding.py :: if lowered in seen:
    """
    found = unsupported_specifics(
        "config.ini declares 42, and config.ini is still generated at 42.",
        supports=(),
    )

    assert found == ("config.ini", "42"), found


def test_the_cap_bounds_what_one_nudge_can_quote() -> None:
    """A long answer must not send the model a list as long as itself.

    Killed by: src/uclone_x/agent/grounding.py :: [:limit]
    """
    answer = " ".join(str(n) for n in range(100, 140))

    found = unsupported_specifics(answer, supports=(), limit=5)

    assert len(found) == 5, found
    assert found[0] == "100"


def test_an_empty_answer_yields_nothing() -> None:
    """A turn with no content makes no claims, so there is nothing to be unsupported."""
    assert unsupported_specifics("", supports=("anything",)) == ()


@pytest.mark.parametrize("separator", [",", "_"])
def test_a_grouped_number_matches_the_same_number_written_plainly(separator: str) -> None:
    """`60_000` in source and `60,000` in prose are the same figure.

    Comparing the rendered spellings makes a correctly-read value look invented purely
    because the model reformatted it, and that false positive lands on exactly the answers
    that did the work.

    Killed by: src/uclone_x/agent/grounding.py :: _DIGITS_RE.sub
    """
    found = unsupported_specifics(
        f"The threshold is 60{separator}000 tokens.",
        supports=("compaction_threshold_tokens: int = 60000",),
    )

    assert found == (), found


def test_a_number_with_a_unit_suffix_the_turn_never_read_is_unsupported() -> None:
    """A number glued to a unit suffix (e.g. 30s) is recognized and reported if unread (#740).

    Killed by: src/uclone_x/agent/grounding.py :: _UNIT_SUFFIXES = "ms|kb|mb|gb|tb|s|m|h|d|b|%"
    Becomes: _UNIT_SUFFIXES = ""
    """
    found = unsupported_specifics(
        "The default timeout is 30s.",
        supports=("def run(self) -> None: ...", "what is the default timeout?"),
    )

    assert found == ("30s",), found


def test_a_number_with_a_unit_suffix_matching_bare_number_in_support_is_supported() -> None:
    """A unit-suffixed figure (30s) matches support that says 30 without the unit (#740).

    Killed by: src/uclone_x/agent/grounding.py :: stripped != norm and stripped in supported_numbers
    Becomes: False
    """
    found = unsupported_specifics(
        "The default timeout is 30s.",
        supports=("timeout: int = 30",),
    )
    control = unsupported_specifics(
        "The default timeout is 30s.",
        supports=("timeout: int = 60",),
    )

    assert found == (), found
    assert control == ("30s",), control


def test_a_hex_number_the_turn_never_read_is_unsupported() -> None:
    """A hexadecimal number with 0x prefix is collected as a specific (#740).

    Killed by: src/uclone_x/agent/grounding.py :: _HEX_NUMBER = r"0[xX][0-9a-fA-F](?:[0-9a-fA-F_]*[0-9a-fA-F])?"
    Becomes: _HEX_NUMBER = r"0[xX]"
    """
    found = unsupported_specifics(
        "The base offset is 0x1F4.",
        supports=("memory map unavailable",),
    )

    assert found == ("0x1F4",), found


def test_a_single_digit_with_a_unit_suffix_is_not_collected() -> None:
    """A single digit with a unit suffix (e.g. 3s) is excluded like a bare single digit (#740).

    Killed by: src/uclone_x/agent/grounding.py :: digits = stripped.replace(".", "")
    Becomes: digits = token.replace(".", "")
    """
    found = unsupported_specifics(
        "Wait 3s before retrying.",
        supports=(),
    )

    assert found == (), found


@pytest.mark.parametrize(
    ("answer", "support"),
    [
        ("The latency is 100ms.", "latency: 100ms"),
        ("The latency is 100ms.", "latency: 100"),
        ("Buffer size is 64KB.", "buffer: 64KB"),
        ("Buffer size is 64KB.", "buffer: 64"),
        ("The progress is 50%.", "progress: 50%"),
        ("The progress is 50%.", "progress: 50"),
        ("Base address is 0x1F4.", "addr: 0x1f4"),
        ("Base address is 0x1F4.", "addr: 1f4"),
    ],
)
def test_unit_and_hex_variants_supported_by_equivalent_support(answer: str, support: str) -> None:
    """Various unit suffixes and hex prefixes are supported by either unit or bare support."""
    found = unsupported_specifics(answer, supports=(support,))
    assert found == (), f"Expected {answer!r} to be supported by {support!r}, got {found}"
