"""Name the modules one `import` statement imports, with a relative import resolved.

Lives in `tests/support/` rather than inside the import-direction checks that use it so
the resolution is itself mutation-checkable: a declaration whose target is the file the
declaration is written in cannot have a unique needle, because the docstring quotes the
line it names.

A checker that reads only `ImportFrom.module` sees `from ..agent.session import X` as an
import of `agent.session` -- a name outside `uclone_x`, so an import-direction check skips
it in silence. `ImportFrom.level` says how many packages up the name starts, and the
package is the importing module's own: the directory it sits in, or itself for an
`__init__.py`.
"""

from __future__ import annotations

import ast
from pathlib import PurePosixPath


def package_of(relative_path: str, root_package: str = "uclone_x") -> list[str]:
    """The package a module under `root_package` belongs to, as dotted parts.

    `relative_path` is relative to the root package's directory, `/`-separated. A module's
    package is the directory it sits in; an `__init__.py` is its package's own module, so
    the directory is the answer for it too.
    """
    return [root_package, *PurePosixPath(relative_path).parent.parts]


def imported_modules(
    node: ast.Import | ast.ImportFrom, package: list[str], *, with_names: bool = True
) -> list[str]:
    """Every dotted module name `node` imports, absolute.

    For `from M import a, b` the answer is `M` and, when `with_names` is true, `M.a` and
    `M.b` too, because `from uclone_x import agent` names the package through its alias.
    `package` is the importing module's package (see `package_of`).
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    if node.level:
        # `.` is the package itself; each further dot is one package up.
        anchor = package[: len(package) - (node.level - 1)]
        base = ".".join([*anchor, base] if base else anchor)
    if not base:
        return []
    if not with_names:
        return [base]
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]
