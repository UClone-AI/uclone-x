"""Which exports of a frontend module nothing outside a test calls (#1285).

`frontend/src/lib/budget.ts` accumulated fifteen dead exports before anyone counted them.
Nine were invisible because a thorough-looking `budget.test.ts` imported and exercised them:
a suite that is the sole caller of the thing it tests can only ever pass, and it reads from
the outside exactly like coverage. The other six had never been imported anywhere at all.

The rule lives here rather than in the test file for the reason `frontend/src/lib/copyGuard.ts`
gives for the same move: a `Killed by:` declaration must name a line that occurs exactly once
in its target, and an anchor inside the declaring file is quoted by its own declaration.
"""

from __future__ import annotations

import re
from pathlib import Path

#: `export const X`, `export function X`, `export type X`, `export interface X`, `export class X`.
#: Deliberately anchored at column zero: a nested `export` is not a module's public surface, and
#: a mention inside a doc comment is indented under a ` *`.
_EXPORT = re.compile(
    r"^export\s+(?:const|function|interface|type|class|enum)\s+([A-Za-z_$][\w$]*)",
    re.MULTILINE,
)

#: A file whose name says vitest owns it. The scan's whole point is that these do not count as
#: callers, so the suffix list must match the one `vitest` collects.
_TEST_SUFFIXES = (".test.ts", ".test.tsx")

_SOURCE_SUFFIXES = (".ts", ".tsx")


def exported_names(source: str) -> list[str]:
    """Every name `source` exports at the top level, in the order it declares them."""
    return _EXPORT.findall(source)


def is_test_file(path: Path) -> bool:
    """Whether `path` is a vitest file, and so may not count as a caller."""
    return path.name.endswith(_TEST_SUFFIXES)


def _mentions(source: str, name: str) -> bool:
    """Whether `source` uses `name` as a whole word.

    A substring match would let `deriveRunSteps` be kept alive by `deriveRunStepsLegacy`, and
    a real parse is not worth its weight for a check whose false negatives are the safe
    direction: a name that is mentioned but unused survives the scan, which costs nothing
    beyond a dead export the scan did not catch.
    """
    return re.search(rf"\b{re.escape(name)}\b", source) is not None


def uncalled_exports(module: Path, tree: Path) -> list[str]:
    """Names `module` exports that no non-test source under `tree` mentions.

    `module` itself is excluded: a module referring to its own export is not a caller of it,
    and every declaration would otherwise count as its own proof.
    """
    names = exported_names(module.read_text(encoding="utf-8"))
    if not names:
        return []
    resolved = module.resolve()

    def may_be_a_caller(path: Path) -> bool:
        if path.suffix not in _SOURCE_SUFFIXES:
            return False
        if is_test_file(path):
            return False
        return path.resolve() != resolved

    callers = [
        path.read_text(encoding="utf-8")
        for path in sorted(tree.rglob("*"))
        if may_be_a_caller(path)
    ]
    return [name for name in names if not any(_mentions(source, name) for source in callers)]


#: A `/* ... */` or `/** ... */` block. This repository's module prose lives in these, and it
#: names the module's own exports constantly -- `rooms.ts` mentions `serviceRefLabel` twice in
#: doc comments alone. Counting those as use would let a module keep a dead export alive by
#: documenting it.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

#: A `//` comment that *starts* its line. Deliberately not every `//`: a `//` inside a string
#: literal (`'http://localhost'`) is code, and stripping it would swallow the rest of a real
#: line. Over-stripping is the dangerous direction here -- it erases evidence that a name is
#: used and so accuses a live export -- and a trailing `//` comment that happens to name an
#: export is the mild direction, one missed report.
_OWN_LINE_COMMENT = re.compile(r"^[ \t]*//[^\n]*", re.M)


def without_comments(source: str) -> str:
    """`source` with its comments blanked out, line numbering and length preserved.

    Blanked rather than deleted so a caller may still iterate lines and have them line up
    with the file it read. The TypeScript side of this repository makes the same move for the
    same reason in `frontend/src/lib/kitBoundary.ts`.
    """

    def blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return _OWN_LINE_COMMENT.sub(blank, _BLOCK_COMMENT.sub(blank, source))


def _declares(line: str, name: str) -> bool:
    """Whether `line` is the top-level declaration of `name`."""
    return (
        re.match(
            rf"export\s+(?:const|function|interface|type|class|enum)\s+{re.escape(name)}\b", line
        )
        is not None
    )


def used_inside_module(source: str, name: str) -> bool:
    """Whether `source` uses `name` in its own code, away from the line that declares it.

    This is the difference between *dead* and merely *over-exported*, and it is the reason
    `uncalled_exports` on its own does not generalise past one module. `emptyStates.ts`
    exports three sentences no other file names; all three are returned by `emptyCause`,
    which a production component imports, so the strings reach the screen. Deleting them
    because nothing outside mentions them would break the module. What is actually surplus
    there is the `export` keyword, which is a different and much smaller defect.

    Comments do not count, or a module could keep a dead export alive by naming it in prose.
    """
    return any(
        not _declares(line.lstrip(), name) and re.search(rf"\b{re.escape(name)}\b", line)
        for line in without_comments(source).splitlines()
    )


def dead_exports(module: Path, tree: Path) -> list[str]:
    """Names `module` exports that nothing reaches: no outside caller, and no use of its own.

    The stricter half of this file, and the one a repository-wide ratchet can stand on.
    `uncalled_exports` answers "is this export's public surface read anywhere", which is a
    question about the `export` keyword; this answers "does anything run this code".
    """
    source = module.read_text(encoding="utf-8")
    return [name for name in uncalled_exports(module, tree) if not used_inside_module(source, name)]


def is_reached_only_by_tests(module: Path, tree: Path) -> bool:
    """Whether *nothing* outside a test reads any of `module`'s exports.

    True of a lint library -- `copyGuard.ts`, `kitBoundary.ts` -- whose whole job is to be
    run by a vitest case, and true of a module that has genuinely fallen out of the app.
    The two are indistinguishable from here, which is exactly why the caller needs a table
    saying which it is rather than a rule that guesses.
    """
    names = exported_names(module.read_text(encoding="utf-8"))
    return bool(names) and uncalled_exports(module, tree) == names
