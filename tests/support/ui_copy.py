"""The rule that keeps principle numbers out of the strings the head renders (#1027).

This is the Python half of `frontend/src/lib/copyGuard.ts`, and it is shaped the same way:
the rule and its exemptions live here, and a fitness check asserts them. Keeping the logic out
of the test file is what lets a `Killed by:` declaration name one of these lines -- an anchor
in the declaring file is quoted by its own declaration, so it never occurs exactly once, which
the suite's declaration checker rejects.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

#: The same spelling `frontend/src/lib/copyGuard.ts` uses, so the two halves agree.
PRINCIPLE_NUMBER = re.compile(r"\bP[0-9]\b")

#: The package whose strings the head renders, relative to the repository root.
UI_PACKAGE = "src/uclone_x/ui"

#: The one legitimate `P<digit>`: the priority stamped on an SSE envelope, keyed by file.
#:
#: `app.py` stamps `"P1"` on `SYSTEM_CONNECTED`, `"P2"` on `AGENT_EVENT` and `"P3"` on
#: `HEARTBEAT`, and reads `AgentEvent.priority` for none of them. The ledger spells the three
#: back, so they are data and not principle numbers. Keyed by file for the reason the frontend
#: list is: an exemption is a site, not a string, and an exempt value pasted into another
#: module is still an offence.
EXEMPT_STRINGS: dict[str, frozenset[str]] = {
    "src/uclone_x/ui/app.py": frozenset({"P1", "P2", "P3"}),
}


def _docstring_ids(tree: ast.Module) -> set[int]:
    """The `id()` of every docstring constant in `tree`.

    A docstring is developer text. FastAPI puts endpoint docstrings in its own `/docs`, which
    is not a surface `docs/PRD.md` §1.3 gives a user, and no dashboard screen renders one.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def principle_numbers_in_strings(source: str) -> list[tuple[int, str]]:
    """Every `(line, value)` string literal in `source` naming a principle number.

    Comments never reach `ast`, so they are out by construction; docstrings are excluded
    deliberately by `_docstring_ids`.
    """
    tree = ast.parse(source)
    docstrings = _docstring_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and PRINCIPLE_NUMBER.search(node.value)
    ]


def offences_in(package: Path, root: Path, exempt: dict[str, frozenset[str]]) -> list[str]:
    """Every `<path>:<line>: <value>` in `package` that `exempt` does not allow *there*.

    `root` is what the reported paths -- and so the keys of `exempt` -- are relative to. It is
    a parameter so the file-keying can be asserted against a planted tree rather than against
    `src/uclone_x/ui`, whose contents change under the test.
    """
    found: list[str] = []
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        allowed = exempt.get(rel, frozenset())
        for line, value in principle_numbers_in_strings(path.read_text(encoding="utf-8")):
            if value not in allowed:
                found.append(f"{rel}:{line}: {value!r}")
    return found


def stale_exemptions(root: Path, exempt: dict[str, frozenset[str]]) -> list[str]:
    """Every exempt value its own file no longer holds -- an exemption outliving its site."""
    stale: list[str] = []
    for rel, values in exempt.items():
        source = (root / rel).read_text(encoding="utf-8")
        present = {value for _, value in principle_numbers_in_strings(source)}
        stale.extend(f"{rel}: {value!r}" for value in sorted(values - present))
    return stale
