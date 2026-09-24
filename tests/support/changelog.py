"""The grammar of a `CHANGELOG.md` entry heading, and the parse of one line into an entry.

**Why this is a module and not a constant beside the tests that assert it.** A
`Killed by:` declaration is executed by replacing its needle in the file it names, and
`swarm.core.mutation` refuses a needle that occurs more than once. A declaration whose
target is its *own* file always duplicates its needle -- the declaration line spells the
needle too -- so a grammar living beside the tests that assert it can carry anchors and
never an executed mutation. Moving it here is what makes those claims runnable. It is the
same shape as `tests/support/session_leak_guard.py`, which
`tests/unit/test_session_storage_isolation.py` mutates.

**What changed in #1057: the version token admits PEP 440 pre-releases.** It was
`\\d+\\.\\d+\\.\\d+`, and the consequence was not a stricter changelog but an
unsatisfiable gate. Measured on `a00e3b9e` by bumping all three version declarations to
`0.1.3rc1` and writing an honest entry for it:

*   heading spelled `## [0.1.3rc1] - 2026-09-15` -- `test_every_entry_heading_parses`
    fails naming the line, and because the strict pattern drops the heading,
    `test_newest_entry_names_the_declared_version` fails too ('0.1.2' != '0.1.3rc1');
*   heading spelled `## [0.1.3] - 2026-09-15` -- it parses, and
    `test_newest_entry_names_the_declared_version` fails ('0.1.3' != '0.1.3rc1').

There is no third spelling: the old token accepted exactly the strings of three dotted
numbers, and none of them equals `0.1.3rc1`. So the first step of the documented
publication path -- an RC to TestPyPI before PyPI -- could not pass the gate, whatever the
author wrote. `tests/unit/test_smoke.py::test_version_is_declared_once_in_effect` accepted
the same rc without complaint, so the version was legal everywhere except the changelog.

Widening the token was chosen over exempting a pre-release from the newest-entry rule
because an exemption opens a window in which a declared version carries no changelog
obligation at all, and nothing then distinguishes "pre-release, entry deferred" from
"final release, entry forgotten". Widening keeps the binding total: every declared
version, pre-release or final, must be the changelog's topmost entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

#: A version this project can publish: three dotted numbers, optionally one PEP 440
#: pre-release segment (`0.1.3`, `0.1.3rc1`, `0.1.3b2`). Deliberately a subset of PEP 440:
#: post-releases, dev-releases, epochs and local versions are not things this project
#: ships, and a token admitting them would stop reading as "the versions we release"
#: while quietly accepting headings no release will ever match.
VERSION_PATTERN: Final = r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?"

#: A complete entry heading: `## [0.1.2] - 2026-09-12`, or `## [0.1.3rc1] - 2026-09-15`.
ENTRY_HEADING: Final = re.compile(
    rf"^## \[(?P<version>{VERSION_PATTERN})\] - (?P<date>\d{{4}}-\d{{2}}-\d{{2}})$"
)

#: Anything that *looks* like an entry heading because it opens the brackets. Matched
#: separately so that a heading which is nearly right -- a missing date, an `Unreleased`
#: section -- is a failure naming the line, rather than a heading `ENTRY_HEADING`
#: silently skips.
_BRACKETED_CANDIDATE: Final = re.compile(r"^## \[")

#: A `##` heading reaching for an entry and missing its brackets (#1057 item 3). Before
#: this, candidacy was `^## \[` alone, so `## 0.1.3 - 2026-09-15` and `## Unreleased` were
#: not malformed entries -- they were not entries at all, invisible to every check, and a
#: dropped bracket downgraded an entry to nothing without failing anything.
#:
#: Scoped to headings that are *trying* to be entries rather than to every `##` line: a
#: changelog may want an ordinary section heading one day, and failing that would be a
#: rule nobody asked for.
_BRACKETLESS_CANDIDATE: Final = re.compile(rf"^## (?:{VERSION_PATTERN}\b|[Uu]nreleased\b)")

#: The same version split into the parts an ordering needs.
_VERSION_PARTS: Final = re.compile(
    r"^(?P<release>\d+\.\d+\.\d+)(?:(?P<stage>a|b|rc)(?P<serial>\d+))?$"
)

#: PEP 440 pre-release stages in ascending order: alpha, then beta, then release
#: candidate.
_STAGE_RANK: Final = {"a": 0, "b": 1, "rc": 2}

#: The rank a final release takes, above every pre-release stage. `0.1.3` is newer than
#: `0.1.3rc1`, which is newer than `0.1.3b2`.
_FINAL_RANK: Final = 3


@dataclass(frozen=True)
class Entry:
    """One `## [version] - date` section of the changelog."""

    version: str
    date: str
    line_number: int

    @property
    def sort_key(self) -> tuple[int, ...]:
        """The version as integers, so `0.1.10` sorts above `0.1.9` rather than below it.

        The two trailing components order pre-releases: a final release outranks every
        pre-release of the same three numbers, and within a stage the serial decides.
        """
        parts = _VERSION_PARTS.match(self.version)
        assert parts is not None, f"not a version this module admits: {self.version!r}"
        release = tuple(int(part) for part in parts["release"].split("."))
        stage = parts["stage"]
        if stage is None:
            return (*release, _FINAL_RANK, 0)
        return (*release, _STAGE_RANK[stage], int(parts["serial"]))


def parse_entries(text: str) -> list[Entry]:
    """Every entry heading in `text`, in the order the text lists them."""
    entries: list[Entry] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = ENTRY_HEADING.match(line)
        if match is not None:
            entries.append(Entry(version=match["version"], date=match["date"], line_number=number))
    return entries


def malformed_headings(text: str) -> list[str]:
    """Headings in `text` that are reaching for an entry and do not parse as one.

    Returned as `line <n>: <line>` so a failure names the line rather than the rule.
    """
    return [
        f"line {number}: {line}"
        for number, line in enumerate(text.splitlines(), start=1)
        if (_BRACKETED_CANDIDATE.match(line) or _BRACKETLESS_CANDIDATE.match(line))
        and not ENTRY_HEADING.match(line)
    ]
