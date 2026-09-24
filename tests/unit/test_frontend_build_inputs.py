# pyright: reportPrivateUsage=false
"""The rebuild trigger asks a content question and must answer it with content (#1075).

`_should_rebuild_frontend` decides whether `ucx ui` may serve the committed bundle. It used
to compare mtimes under `frontend/src` against `ui_static/index.html`, which was wrong in
both directions: a `git checkout` restamps every file, so a pristine tree read as stale and
rebuilt — rewriting the tracked bundle, because `vite.config.ts` builds into it with
`emptyOutDir: true` — while a `package-lock.json` or `tailwind.config.js` change, which
really does change the bundle's bytes, read as fresh.

Each test below pins one of the issue's acceptance criteria, and each names the mutation
that kills it. The scratch trees are built here rather than pointed at the real repository
so that the cases hold whether or not this checkout's bundle happens to be fresh, and so
that nothing under test can rewrite a tracked file while being measured.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.support.frontend_build_guard import RealFrontendBuildInTestError
from uclone_x.errors import FrontendBuildFailedError
from uclone_x.ui import build_inputs, server

#: A `frontend/` skeleton: every declared config input, one component, one test file and
#: one snapshot. Values are the file contents.
_TREE: dict[str, str] = {
    "index.html": "<!doctype html><title>UClone-X</title>\n",
    "package.json": '{"name": "@uclone-x/frontend", "version": "0.1.2"}\n',
    "package-lock.json": '{"lockfileVersion": 3}\n',
    "postcss.config.js": "export default { plugins: {} };\n",
    "tailwind.config.js": "export default { content: [] };\n",
    "tsconfig.json": '{"compilerOptions": {}}\n',
    "tsconfig.node.json": '{"compilerOptions": {}}\n',
    "vite.config.ts": "export default {};\n",
    "src/App.tsx": "export const App = () => null;\n",
    "src/App.test.tsx": "it('renders', () => {});\n",
    "src/lib/rooms.spec.ts": "it('joins', () => {});\n",
    "src/__snapshots__/App.snap": "exports[`App`] = `null`;\n",
}


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A scratch tree whose bundle is recorded as matching its source.

    Returns `(frontend_dir, static_dir)`. The record is written where the launcher writes
    it — beside `ui_static`, not inside it.
    """
    frontend = tmp_path / "frontend"
    static = tmp_path / "src" / "uclone_x" / "ui_static"
    _write(frontend, _TREE)
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
    (static / "assets" / "index.css").write_bytes(b"x" * 94_515)
    build_inputs.write_record(frontend, static.parent)
    return frontend, static


def _restamp(root: Path, when: float) -> None:
    """Stamp every file under `root` with one mtime, as a checkout or a clone does."""
    for path in root.rglob("*"):
        os.utime(path, (when, when))


def test_a_pristine_checkout_whose_mtimes_were_all_restamped_does_not_rebuild(
    tmp_path: Path,
) -> None:
    """Criterion 1. The everyday false positive, reproduced as a checkout reproduces it.

    `git checkout`, `git stash pop`, a rebase and a fresh clone all leave every source file
    newer than the committed `index.html`, which is precisely what the old mtime walk read
    as stale. Nothing about the content changed, so the answer must be `False` — and it
    must stay `False` with the source stamped *after* the bundle, which is the ordering the
    old code failed on and the reason the stamp below is in the future.

    Killed by: src/uclone_x/ui/server.py :: return recorded != build_inputs.build_input_digest(frontend_dir)
    Becomes: return True
    """
    frontend, static = _checkout(tmp_path)
    _restamp(static, 1_000_000_000.0)
    _restamp(frontend, 2_000_000_000.0)

    assert server._should_rebuild_frontend(frontend, static) is False


@pytest.mark.parametrize(
    "changed",
    ["package-lock.json", "vite.config.ts", "tailwind.config.js", "postcss.config.js"],
)
def test_a_config_input_outside_frontend_src_changes_the_answer(
    tmp_path: Path, changed: str
) -> None:
    """Criterion 2, the half the mtime walk could not see at all.

    These four live outside `frontend/src`, so the old walk never looked at them: a
    dependency bump or a Tailwind target change left a genuinely stale bundle looking
    fresh. That is the direction that costs something — a served UI that is not the code
    the tree describes.

    The mutation drops the lockfile from the declared inputs, which fails this test on its
    `package-lock.json` case alone; the other three are here because the list is what they
    share, not because four assertions are stronger than one.

    Killed by: src/uclone_x/ui/build_inputs.py ::     "package-lock.json",
    Becomes:
    """
    frontend, static = _checkout(tmp_path)
    assert server._should_rebuild_frontend(frontend, static) is False

    (frontend / changed).write_text(_TREE[changed] + "// a real change\n", encoding="utf-8")

    assert server._should_rebuild_frontend(frontend, static) is True


def test_tsconfig_and_index_html_are_inputs_too(tmp_path: Path) -> None:
    """Criterion 2, the remaining two names the issue lists.

    `frontend/index.html` is the bundler's entry document and `tsconfig.json` decides how
    the sources compile; both are read by `vite build` and neither is under `frontend/src`.
    """
    frontend, static = _checkout(tmp_path)

    (frontend / "index.html").write_text("<!doctype html><title>x</title>\n", encoding="utf-8")
    assert server._should_rebuild_frontend(frontend, static) is True

    build_inputs.write_record(frontend, static.parent)
    assert server._should_rebuild_frontend(frontend, static) is False

    (frontend / "tsconfig.json").write_text('{"compilerOptions": {"strict": true}}\n', "utf-8")
    assert server._should_rebuild_frontend(frontend, static) is True


@pytest.mark.parametrize(
    "edited", ["src/App.test.tsx", "src/lib/rooms.spec.ts", "src/__snapshots__/App.snap"]
)
def test_an_edit_confined_to_a_test_or_a_snapshot_does_not_rebuild(
    tmp_path: Path, edited: str
) -> None:
    """Criterion 3. `vite build` emits none of these, so none of them can make it stale.

    This is #1067's case: on a branch whose only change was a `.test.tsx` file, four unit
    tests shelled out to a real production build. `frontend/vite.config.ts` confines test
    files to vitest's `test.include` glob — the bundler reads them through no path.

    Killed by: src/uclone_x/ui/build_inputs.py :: return not any(infix in relative_path.name for infix in EXCLUDED_INFIXES)
    Becomes: return True
    """
    frontend, static = _checkout(tmp_path)

    (frontend / edited).write_text(_TREE[edited] + "// edited\n", encoding="utf-8")

    assert server._should_rebuild_frontend(frontend, static) is False


def test_a_new_component_under_frontend_src_does_rebuild(tmp_path: Path) -> None:
    """The other side of the exclusion: a real source file added is a real change.

    Without this, "exclude the test files" and "exclude `frontend/src`" would pass the same
    tests, and the second is the regression that would serve stale code.
    """
    frontend, static = _checkout(tmp_path)

    (frontend / "src" / "components" / "New.tsx").parent.mkdir(parents=True, exist_ok=True)
    (frontend / "src" / "components" / "New.tsx").write_text("export const New = 1;\n", "utf-8")

    assert server._should_rebuild_frontend(frontend, static) is True


def test_a_deleted_input_is_not_read_as_an_empty_one(tmp_path: Path) -> None:
    """A missing input and an empty input must not digest alike.

    Deleting `tailwind.config.js` changes what the bundler emits exactly as writing one
    does. Were absence recorded as the digest of no bytes, an emptied config and a deleted
    config would be indistinguishable, and one of the two would be answered wrong.

    Killed by: src/uclone_x/ui/build_inputs.py :: return _ABSENT
    Becomes: return hashlib.sha256(b"").hexdigest()
    """
    frontend, _ = _checkout(tmp_path)

    (frontend / "tailwind.config.js").write_text("", encoding="utf-8")
    emptied = build_inputs.build_input_digest(frontend)
    (frontend / "tailwind.config.js").unlink()
    deleted = build_inputs.build_input_digest(frontend)

    assert emptied != deleted


def test_the_digest_is_the_same_for_two_trees_with_the_same_content(tmp_path: Path) -> None:
    """Stability is the whole point: no timestamp, no path, no order of traversal.

    Two copies of one source tree at different absolute paths, written at different times,
    must digest identically — otherwise a clone would rebuild once per machine.
    """
    first = tmp_path / "one" / "frontend"
    second = tmp_path / "two" / "frontend"
    _write(first, _TREE)
    _write(second, _TREE)
    _restamp(second, 2_000_000_000.0)

    assert build_inputs.build_input_digest(first) == build_inputs.build_input_digest(second)


def test_a_bundle_with_no_record_beside_it_is_rebuilt(tmp_path: Path) -> None:
    """No record is "cannot tell", and the only answer that cannot serve wrong code is True.

    A bundle predating this record, or one whose record was deleted, is unjudgeable. The
    launcher rebuilds once, writes the record, and converges.

    **No `Killed by:` line, deliberately, and the reason is the finding.** The obvious
    candidate — neutering `if recorded is None:` in `_should_rebuild_frontend` — was run
    and does **not** kill this test: absence is represented by `None`, which no hex digest
    can equal, so the comparison below that branch answers `True` anyway. The branch is an
    early exit that saves hashing 62 files, not a decision, and there is no single-line
    mutation of this shape that makes a missing record read as a match. Recording an
    anchor here would claim a boundary the code does not have.
    """
    frontend, static = _checkout(tmp_path)
    (static.parent / build_inputs.RECORD_NAME).unlink()

    assert server._should_rebuild_frontend(frontend, static) is True


def test_an_unreadable_record_is_not_read_as_a_match(tmp_path: Path) -> None:
    """A record that is a directory, or empty, answers `None` rather than raising.

    The launcher must not crash on a damaged record, and must not treat "I could not read
    it" as "it matched".
    """
    frontend, static = _checkout(tmp_path)
    record = static.parent / build_inputs.RECORD_NAME

    record.write_text("   \n", encoding="utf-8")
    assert build_inputs.recorded_digest(static.parent) is None
    assert server._should_rebuild_frontend(frontend, static) is True

    record.unlink()
    record.mkdir()
    assert build_inputs.recorded_digest(static.parent) is None


def test_a_bundle_missing_its_index_or_its_assets_is_rebuilt_whatever_the_record_says(
    tmp_path: Path,
) -> None:
    """The record describes the inputs, not the output, so it cannot vouch for the bundle.

    A matching record beside a directory with no `index.html`, or no `assets/`, is a record
    about a bundle that is not there.
    """
    frontend, static = _checkout(tmp_path)
    assert server._should_rebuild_frontend(frontend, static) is False

    (static / "index.html").unlink()
    assert server._should_rebuild_frontend(frontend, static) is True

    (static / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
    (static / "assets" / "index.css").unlink()
    (static / "assets").rmdir()
    assert server._should_rebuild_frontend(frontend, static) is True


def test_a_small_css_file_no_longer_forces_a_rebuild_on_every_launch(tmp_path: Path) -> None:
    """Criterion 5. The `st_size < 10000` heuristic is gone, and could not have converged.

    It asked a content question of a proxy. Had a legitimate build ever emitted CSS under
    10 KB — a Tailwind purge that got better at its job — the launcher would have rebuilt
    on every start, and every rebuild would have re-emitted the same small file. The bundle
    below is byte-for-byte what its record describes, and 512 bytes of CSS is not evidence
    against it.
    """
    frontend, static = _checkout(tmp_path)
    (static / "assets" / "index.css").write_bytes(b"x" * 512)

    assert server._should_rebuild_frontend(frontend, static) is False


def _launcher_pointed_at(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point `_ensure_frontend_built` at a scratch repository laid out as this one is.

    It derives both directories from its own `__file__`, so moving that moves both.
    """
    monkeypatch.setattr(server, "__file__", str(repo / "src" / "uclone_x" / "ui" / "server.py"))


def test_a_failed_build_raises_instead_of_serving_the_stale_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criterion 4 (P6). The bare `except Exception` served the bundle it had just failed
    to replace, behind a yellow line.

    The user got a dashboard that looked fine and was not the code they were running —
    a substituted result, which P6 forbids. The launcher only reaches a build when the
    committed bundle does *not* match the source, so there is nothing here worth falling
    back to.

    Killed by: src/uclone_x/ui/server.py :: raise FrontendBuildFailedError(frontend_dir, exc) from exc
    Becomes: console.print(f"[yellow]{exc}[/yellow]")
    """
    frontend, static = _checkout(tmp_path)
    (frontend / "src" / "App.tsx").write_text("export const App = () => 1;\n", encoding="utf-8")
    _launcher_pointed_at(tmp_path, monkeypatch)

    def failing_run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, command, stderr=b"vite: transform failed")

    monkeypatch.setattr(subprocess, "run", failing_run)

    with pytest.raises(FrontendBuildFailedError) as caught:
        server._ensure_frontend_built()

    assert "vite: transform failed" in str(caught.value)
    assert (static / "index.html").is_file()


def test_the_test_time_build_refusal_still_propagates_through_the_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trap in #1074, restated against the handler that replaced the one it named.

    `RealFrontendBuildInTestError` derives from `BaseException` *because* the old
    `except Exception` would have swallowed it: npm blocked, a warning printed into
    captured output, the test still green. Narrowing that handler is only a fix if it stays
    narrow, so the width is asserted here rather than left to the diff to imply. The
    mutation widens it to `BaseException`, which re-opens the hole while looking like a
    tidy-up.

    Killed by: src/uclone_x/ui/server.py :: except (subprocess.CalledProcessError, OSError) as exc:
    Becomes: except BaseException as exc:
    """
    frontend, static = _checkout(tmp_path)
    (frontend / "src" / "App.tsx").write_text("export const App = () => 2;\n", encoding="utf-8")
    _launcher_pointed_at(tmp_path, monkeypatch)

    with pytest.raises(RealFrontendBuildInTestError):
        server._ensure_frontend_built()

    assert build_inputs.recorded_digest(static.parent) != build_inputs.build_input_digest(frontend)


def test_a_successful_build_records_the_inputs_it_built_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convergence. Without the record, an edited tree rebuilds on every single launch.

    The record is a tracked file, and writing it here is safe precisely because the build
    that just ran rewrote the tracked bundle beside it: the two belong in one commit, and
    the second without the first would be a lie about the first.

    `subprocess.run` is replaced rather than allowed to run: the real command is
    `npm run build`, which this suite may never execute (#1074).

    Killed by: src/uclone_x/ui/server.py :: build_inputs.write_record(frontend_dir, static_dir.parent)
    Becomes:
    """
    frontend, static = _checkout(tmp_path)
    (frontend / "src" / "App.tsx").write_text("export const App = () => 3;\n", encoding="utf-8")
    _launcher_pointed_at(tmp_path, monkeypatch)
    seen: list[Any] = []

    def succeeding_run(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", succeeding_run)
    assert server._should_rebuild_frontend(frontend, static) is True

    server._ensure_frontend_built()

    assert seen == [["npm", "run", "build"]]
    assert server._should_rebuild_frontend(frontend, static) is False


def test_the_record_is_not_written_inside_the_bundle_directory(tmp_path: Path) -> None:
    """Where the record lives is load-bearing, not cosmetic.

    Inside `src/uclone_x/ui_static` it would be deleted by the next build — `emptyOutDir:
    true` — and gate stage 6b compares the built and committed bundles with **nothing
    ignored**, so a file there that `vite build` does not emit is reported as `not_built`
    and turns the gate red. The issue's phrase "recorded beside the bundle" has to mean
    beside, and this is the test that keeps it meaning that.

    Killed by: src/uclone_x/ui/server.py :: build_inputs.recorded_digest(static_dir.parent)
    Becomes: build_inputs.recorded_digest(static_dir)
    """
    frontend, static = _checkout(tmp_path)

    assert (static.parent / build_inputs.RECORD_NAME).is_file()
    assert not (static / build_inputs.RECORD_NAME).exists()
    assert build_inputs.RECORD_NAME not in {p.name for p in static.rglob("*")}
    assert server._should_rebuild_frontend(frontend, static) is False


def test_a_symlinked_input_is_not_mistaken_for_the_file_it_points_at(tmp_path: Path) -> None:
    """Stage 6b treats a symlink as a difference; so does this, and for the same reason.

    A link whose target holds the right bytes is not the same input as the file, and
    `hashlib` reading through it would say it is.
    """
    frontend, _ = _checkout(tmp_path)
    before = build_inputs.build_input_digest(frontend)
    target = tmp_path / "elsewhere.js"
    target.write_text(_TREE["tailwind.config.js"], encoding="utf-8")

    (frontend / "tailwind.config.js").unlink()
    (frontend / "tailwind.config.js").symlink_to(target)

    assert build_inputs.build_input_digest(frontend) != before
    assert build_inputs.build_inputs(frontend)["tailwind.config.js"].startswith("symlink -> ")
    assert (
        hashlib.sha256(target.read_bytes()).hexdigest()
        != (build_inputs.build_inputs(frontend)["tailwind.config.js"])
    )
