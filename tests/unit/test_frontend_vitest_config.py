"""The frontend vitest suite the gate runs must not be shrinkable by a committed `.only`.

vitest defaults `allowOnly` to `!isCI`. The gate is local and runs without `CI` set (P8),
so before #930 a committed `it.only` or `describe.only` made stage 6 run only the focused
tests and still exit 0. The suite then looked green but ran only a fraction of its tests.
`test.allowOnly: false` in `frontend/vite.config.ts` makes vitest reject any `.only`,
with or without `CI`.

**What these tests guard:** the setting is in live code in every config vitest could
load, and nothing on the path the gate takes turns it back on. That covers three things.
A `vitest.config.*` shadows `vite.config.ts`, so vitest would read it instead. A CLI flag
in an npm script overrides the config, and vitest accepts both `--allowOnly` and
`--allow-only`. So do extra arguments on the gate's own `npm test` call.
**What they do not guard:** whether vitest honours the key, which is vitest's contract.
That was demonstrated on #930: with `CI` unset, a temporary `it.only` made `npm test`
exit 1 with this setting and exit 0 without it.

The files are read as text, not evaluated, because evaluating them needs `node_modules`,
and unit tests must run without them. Loading the config inside a vitest test does not
work either: vite's esbuild fails its invariant under jsdom, and `tsc` refuses the
cross-project import.
"""

import json
import re
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"(?m)^\s*//.*$|\s+//.*$")
#: Every spelling vitest's CLI accepts for re-enabling `.only`, and the config key itself.
_ALLOW_ONLY = re.compile(r"allow-?only", re.IGNORECASE)


def _code(path: Path) -> str:
    """A config file with comments removed, so a commented-out setting does not count.

    The line-comment pattern only matches `//` at the start of a line or after
    whitespace, so `'http://…'` string literals are left intact.
    """
    text = path.read_text(encoding="utf-8")
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def _test_block(code: str) -> str | None:
    """The body of the `test: { ... }` object, matched by brace depth.

    A `[^{}]*` pattern is not enough, because the block contains the glob
    `'src/**/*.test.{ts,tsx}'`, which has its own braces.
    """
    opening = re.search(r"\btest\s*:\s*\{", code)
    if opening is None:
        return None
    depth = 0
    for index in range(opening.end() - 1, len(code)):
        depth += {"{": 1, "}": -1}.get(code[index], 0)
        if depth == 0:
            return code[opening.end() : index]
    return None


def _allow_only_problems(frontend: Path) -> list[str]:
    """Every way the vitest run under `frontend` could accept a committed `.only`."""
    problems: list[str] = []
    configs = [frontend / "vite.config.ts", *sorted(frontend.glob("vitest.config.*"))]
    if not configs[0].is_file():
        problems.append(f"{configs[0]} is missing")
        configs = configs[1:]
    for config in configs:
        code = _code(config)
        settings = [v.strip() for v in re.findall(r"\ballowOnly\s*:\s*([^,\n}]+)", code)]
        if settings != ["false"]:
            problems.append(
                f"{config.name} must set `allowOnly: false` exactly once, found {settings}"
            )
            continue
        block = _test_block(code)
        if block is None or "allowOnly" not in block:
            problems.append(f"{config.name} sets `allowOnly` outside its `test:` block")
    for workspace in sorted(frontend.glob("vitest.workspace.*")):
        problems.append(
            f"{workspace.name} exists: each workspace project loads its own config, which "
            "this guard does not read. Extend the guard before adding one."
        )
    package = cast(
        "dict[str, dict[str, str]]",
        json.loads((frontend / "package.json").read_text(encoding="utf-8")),
    )
    for name, command in package.get("scripts", {}).items():
        if _ALLOW_ONLY.search(command):
            problems.append(f"npm script {name!r} re-enables `.only`: {command!r}")
    return problems


def test_the_frontend_rejects_a_committed_only() -> None:
    """The config sets `allowOnly: false` once, inside `test:`, and no npm script undoes it.

    `--allow-only` passes vitest's CLI and overrides the config (measured on #940).

    Killed by: frontend/vite.config.ts :: allowOnly: false,
    Becomes: allowOnly: true,
    Killed by: frontend/package.json :: "test": "vitest run",
    Becomes: "test": "vitest run --allow-only",
    """
    assert _allow_only_problems(FRONTEND) == []


def test_the_gate_passes_no_allow_only_argument_to_npm_test() -> None:
    """Stage 6's own `npm test` call must not carry the flag either.

    Killed by: src/uclone_x/cli/quality_gate.py :: ["npm", "test", "--silent"]
    Becomes: ["npm", "test", "--silent", "--", "--allow-only"]
    """
    gate = (REPO_ROOT / "src" / "uclone_x" / "cli" / "quality_gate.py").read_text(encoding="utf-8")
    assert _ALLOW_ONLY.findall(gate) == []


def _frontend(tmp_path: Path, *, script: str = "vitest run") -> Path:
    frontend = tmp_path / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "vite.config.ts").write_text(
        (FRONTEND / "vite.config.ts").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (frontend / "package.json").write_text(json.dumps({"scripts": {"test": script}}))
    return frontend


def test_a_vitest_config_without_the_setting_is_a_problem(tmp_path: Path) -> None:
    """vitest reads `vitest.config.*` instead of `vite.config.ts` when both exist."""
    frontend = _frontend(tmp_path)
    (frontend / "vitest.config.ts").write_text("export default { test: {} };\n")

    assert _allow_only_problems(frontend) == [
        "vitest.config.ts must set `allowOnly: false` exactly once, found []"
    ]

    (frontend / "vitest.config.ts").write_text("export default { test: { allowOnly: false } };\n")
    assert _allow_only_problems(frontend) == []


def test_every_cli_spelling_of_the_flag_is_a_problem(tmp_path: Path) -> None:
    # Indexed directories: `allowOnly` and `allowonly` collide on a case-insensitive disk.
    for index, flag in enumerate(("--allowOnly", "--allow-only", "--allow-only=true")):
        frontend = _frontend(tmp_path / str(index), script=f"vitest run {flag}")
        assert _allow_only_problems(frontend) == [
            f"npm script 'test' re-enables `.only`: 'vitest run {flag}'"
        ], flag


def test_a_commented_out_or_misplaced_setting_is_a_problem(tmp_path: Path) -> None:
    frontend = _frontend(tmp_path)
    config = frontend / "vite.config.ts"
    original = config.read_text(encoding="utf-8")

    config.write_text(original.replace("allowOnly: false,", "// allowOnly: false,"))
    assert _allow_only_problems(frontend) == [
        "vite.config.ts must set `allowOnly: false` exactly once, found []"
    ]

    moved = original.replace("    allowOnly: false,\n", "").replace(
        "port: 5173,", "port: 5173, allowOnly: false,"
    )
    config.write_text(moved)
    assert _allow_only_problems(frontend) == [
        "vite.config.ts sets `allowOnly` outside its `test:` block"
    ]
