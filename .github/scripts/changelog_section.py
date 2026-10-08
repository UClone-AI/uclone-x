"""Print one version's section of `CHANGELOG.md`, for the notes of its GitHub Release.

    python .github/scripts/changelog_section.py 0.3.1 [CHANGELOG.md]

The section runs from the version's `## [X.Y.Z] - date` heading up to the next `## [`
heading, or the end of the file. The heading line itself is left out: the Release
title already names the version. Leading and trailing blank lines are trimmed.

The version is matched whole, so `0.3.1` never selects the `0.3.10` entry. A version
with no entry, or an entry with nothing under its heading, exits 1 and prints nothing
to stdout: a Release is never published with empty notes.

Standard library only, so the release workflow runs it with the runner's `python3`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Any entry heading; the boundary between one version's section and the next.
_ENTRY: re.Pattern[str] = re.compile(r"^## \[(?P<version>[^\]]*)\]", re.MULTILINE)


def changelog_section(changelog: str, version: str) -> str | None:
    """The body of `version`'s entry, or None when it has no entry or an empty one."""
    headings = list(_ENTRY.finditer(changelog))
    for index, heading in enumerate(headings):
        if heading["version"] != version:
            continue
        line_end = changelog.find("\n", heading.end())
        start = len(changelog) if line_end == -1 else line_end + 1
        end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
        body = changelog[start:end].strip("\n").rstrip()
        return body or None
    return None


def main(argv: list[str] | None = None) -> int:
    """Print the section for the version in `argv`, or report why there is none."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (1, 2):
        print("usage: changelog_section.py VERSION [CHANGELOG]", file=sys.stderr)
        return 2
    version = args[0].removeprefix("v")
    path = Path(args[1] if len(args) == 2 else "CHANGELOG.md")
    section = changelog_section(path.read_text(encoding="utf-8"), version)
    if section is None:
        print(f"::error::{path} has no written entry for {version}", file=sys.stderr)
        return 1
    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
