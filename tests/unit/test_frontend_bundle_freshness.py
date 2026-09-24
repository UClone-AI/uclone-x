"""The committed UI bundle must be what the frontend source builds (#878).

The comparison is tested on `tmp_path` directories standing in for a build and a committed
bundle. The stage around it is tested with `subprocess.run` replaced by a fake bundler that
writes into whatever `--outDir` it is handed, so nothing here needs `node_modules`. The real
bundler is exercised by the gate itself: stage 6b runs it on every `./ucx test check`.
"""

import io
import os
import subprocess
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console

from uclone_x.cli import bundle_freshness
from uclone_x.cli.bundle_freshness import BUNDLE_DIR, check_bundle_freshness, compare_bundles

_BUNDLE: dict[str, bytes] = {
    "index.html": b'<script src="/assets/index-AAAA.js"></script>',
    "assets/index-AAAA.js": b'const dock="artifacts-dock";',
    "assets/index-AAAA.css": b".dock{}",
}


def _write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_identical_bundles_compare_identical(tmp_path: Path) -> None:
    """The positive control: without it, a comparison that always differs passes the rest.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: return not (self.not_committed or self.not_built or self.changed)
    Becomes: return False
    """
    built = _write_tree(tmp_path / "built", _BUNDLE)
    committed = _write_tree(tmp_path / "committed", _BUNDLE)

    assert compare_bundles(built, committed).identical


def test_a_file_the_source_no_longer_builds_is_named(tmp_path: Path) -> None:
    """#878's shape: the committed bundle carries something the current source does not.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: not_built=tuple(sorted(committed_files.keys() - built_files.keys())),
    Becomes: not_built=(),
    Killed by: src/uclone_x/cli/bundle_freshness.py :: tree.rglob("*")
    Becomes: (entry for entry in tree.rglob("*") if entry.suffix != ".js")
    """
    built = _write_tree(tmp_path / "built", _BUNDLE)
    committed = _write_tree(
        tmp_path / "committed", {**_BUNDLE, "assets/InternalsDock-OLD.js": b"internals-dock"}
    )

    difference = compare_bundles(built, committed)

    assert not difference.identical
    assert difference.not_built == ("assets/InternalsDock-OLD.js",)
    assert (difference.not_committed, difference.changed) == ((), ())


def test_a_file_the_source_builds_but_nobody_committed_is_named(tmp_path: Path) -> None:
    """The common shape: a source edit renames the hashed chunk and the rebuild was not committed.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: not_committed=tuple(sorted(built_files.keys() - committed_files.keys())),
    Becomes: not_committed=(),
    """
    committed = _write_tree(tmp_path / "committed", _BUNDLE)
    built = _write_tree(tmp_path / "built", {**_BUNDLE, "assets/index-BBBB.js": b"new"})

    difference = compare_bundles(built, committed)

    assert not difference.identical
    assert difference.not_committed == ("assets/index-BBBB.js",)
    assert (difference.not_built, difference.changed) == ((), ())


def test_a_same_named_file_with_different_bytes_of_equal_length_is_named(tmp_path: Path) -> None:
    """Content, not names and not sizes: `index.html` keeps its name across every rebuild, and
    so does any chunk whose hash a source edit did not reach.

    Same length on purpose, so a comparison of sizes cannot pass this either. The file is
    nested, so a comparison that looked only at the top level cannot pass it.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if built_files[name] != committed_files[name]
    Becomes: if False
    Killed by: src/uclone_x/cli/bundle_freshness.py :: for path in tree.rglob("*")
    Becomes: for path in tree.glob("*")
    Killed by: src/uclone_x/cli/bundle_freshness.py :: hashlib.sha256(path.read_bytes()).hexdigest()
    Becomes: str(len(path.read_bytes()))
    """
    committed = _write_tree(tmp_path / "committed", _BUNDLE)
    built = _write_tree(
        tmp_path / "built", {**_BUNDLE, "assets/index-AAAA.js": b'const dock="internals-dock";'}
    )
    assert len(_BUNDLE["assets/index-AAAA.js"]) == len(b'const dock="internals-dock";')

    difference = compare_bundles(built, committed)

    assert not difference.identical
    assert difference.changed == ("assets/index-AAAA.js",)


def test_a_missing_committed_bundle_reports_every_built_file(tmp_path: Path) -> None:
    """No committed directory is a difference in every file, not an error or a pass."""
    built = _write_tree(tmp_path / "built", _BUNDLE)

    difference = compare_bundles(built, tmp_path / "never-built")

    assert difference.not_committed == tuple(sorted(_BUNDLE))


def test_a_symlink_to_a_file_with_the_right_content_is_a_difference(tmp_path: Path) -> None:
    """`is_file()` follows links, so content alone would call this identical. Vite emits none.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if path.is_symlink():
    Becomes: if False:
    """
    built = _write_tree(tmp_path / "built", _BUNDLE)
    committed = _write_tree(tmp_path / "committed", _BUNDLE)
    elsewhere = _write_tree(tmp_path / "elsewhere", {"index.js": _BUNDLE["assets/index-AAAA.js"]})
    (committed / "assets" / "index-AAAA.js").unlink()
    (committed / "assets" / "index-AAAA.js").symlink_to(elsewhere / "index.js")
    (committed / "linked-dir").symlink_to(elsewhere, target_is_directory=True)
    (committed / "dangling.js").symlink_to(tmp_path / "nowhere.js")

    difference = compare_bundles(built, committed)

    assert difference.changed == ("assets/index-AAAA.js",)
    assert difference.not_built == ("dangling.js", "linked-dir")


def test_a_non_regular_entry_in_the_committed_bundle_is_a_difference(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/cli/bundle_freshness.py :: elif not path.is_dir():
    Becomes: elif False:
    """
    built = _write_tree(tmp_path / "built", _BUNDLE)
    committed = _write_tree(tmp_path / "committed", _BUNDLE)
    os.mkfifo(committed / "assets" / "pipe")

    assert compare_bundles(built, committed).not_built == ("assets/pipe",)


# ── The stage ───────────────────────────────────────────────────────────────────────────


def _repo(root: Path, *, node_modules: bool = True, committed: dict[str, bytes] = _BUNDLE) -> Path:
    frontend = root / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "package.json").write_text('{"scripts": {"build": "tsc && vite build"}}\n')
    if node_modules:
        (frontend / "node_modules").mkdir()
    _write_tree(root / BUNDLE_DIR, committed)
    return root


def _npm_on_path(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if present else None

    monkeypatch.setattr(bundle_freshness.shutil, "which", which)


#: What vite emits instead when `NODE_ENV` is anything but `production`. React's development
#: build, under a different content hash (measured: main chunk 922 KB -> 1.55 MB).
_DEVELOPMENT_BUNDLE: dict[str, bytes] = {
    "index.html": b'<script src="/assets/index-DEV0.js"></script>',
    "assets/index-DEV0.js": b"react.development.js",
    "assets/index-AAAA.css": b".dock{}",
}

#: What vite emits instead when browserslist resolves to targets other than `defaults`:
#: autoprefixer adds vendor prefixes, so the CSS and every name hashed from it change
#: (measured: `chrome 30, safari 6, ie 10` grew the CSS from 93,080 to 100,000 bytes).
_PREFIXED_BUNDLE: dict[str, bytes] = {
    "index.html": b'<link href="/assets/index-PREF.css">',
    "assets/index-AAAA.js": _BUNDLE["assets/index-AAAA.js"],
    "assets/index-PREF.css": b".dock{-webkit-transform:none}",
}


def _browser_targets(environment: dict[str, str]) -> str:
    """Browserslist's precedence, as far as a fake can model it without a filesystem walk.

    The `BROWSERSLIST` variable is read first, then a `BROWSERSLIST_CONFIG` file, then config
    files in the build directory and its parents (not modelled), and `defaults` if none exists.
    """
    if "BROWSERSLIST" in environment:
        return environment["BROWSERSLIST"]
    if "BROWSERSLIST_CONFIG" in environment:
        return Path(environment["BROWSERSLIST_CONFIG"]).read_text().strip()
    return "defaults"


def _fake_bundler(
    monkeypatch: pytest.MonkeyPatch,
    emits: dict[str, bytes],
    returncode: int = 0,
    also_writes: Path | None = None,
    environments: list[object] | None = None,
) -> list[tuple[list[str], object]]:
    """Replace the build with one that writes `emits` into the `--outDir` it is given.

    Modelled on vite in the two environment inputs measured to matter, read from the
    environment the process is given. For a `NODE_ENV` other than `production` (unset means
    `production`, as in vite) it emits `_DEVELOPMENT_BUNDLE`. For browser targets other than
    `defaults` (`_browser_targets`) it emits `_PREFIXED_BUNDLE`. `also_writes` makes it write
    that directory as well, the way a bundler ignoring `--outDir` would. `environments`
    collects the `env` each call was given.
    """
    seen: list[tuple[list[str], object]] = []

    def fake_run(
        cmd: list[str], *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen.append((cmd, kwargs.get("cwd")))
        given = kwargs.get("env")
        if environments is not None:
            environments.append(given)
        environment = cast(dict[str, str], given) if given is not None else dict(os.environ)
        production = environment.get("NODE_ENV", "production") == "production"
        if not production:
            output = _DEVELOPMENT_BUNDLE
        elif _browser_targets(environment) != "defaults":
            output = _PREFIXED_BUNDLE
        else:
            output = emits
        _write_tree(Path(cmd[cmd.index("--outDir") + 1]), output)
        if also_writes is not None:
            _write_tree(also_writes, {"assets/index-REWRITTEN.js": b"x"})
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout="", stderr="[vite:esbuild] boom [/x]"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


def _forbid_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_run(cmd: list[str], *args: object, **kwargs: object) -> object:
        raise AssertionError(f"the stage ran {cmd} when it should have refused first")

    monkeypatch.setattr(subprocess, "run", must_not_run)


def test_a_bundle_the_source_still_builds_is_fresh_and_the_build_never_writes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built into a temporary directory from `frontend/`, and the committed one untouched.

    The build must not target the committed directory: a gate that rebuilt it would pass by
    construction, and would rewrite tracked files on every run.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: str(built),
    Becomes: str(committed),
    """
    repo = _repo(tmp_path)
    _npm_on_path(monkeypatch, present=True)
    seen = _fake_bundler(monkeypatch, emits=_BUNDLE)

    status, code, lines = check_bundle_freshness(repo)

    assert (status, code) == ("fresh", 0)
    assert lines
    ((cmd, cwd),) = seen
    assert cwd == repo / "frontend"
    assert cmd[:6] == ["npm", "exec", "--no", "--", "vite", "build"]
    out_dir = Path(cmd[cmd.index("--outDir") + 1])
    assert not out_dir.is_relative_to(repo)
    assert not out_dir.exists(), "the temporary build directory must be removed"


def test_a_bundle_the_source_no_longer_builds_fails_naming_the_file_and_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale is a non-zero code, the differing file by name, and the rebuild command.

    The command pins `NODE_ENV=production` and `BROWSERSLIST=defaults`, as the stage's own
    build does: unpinned, a shell with `NODE_ENV=development` or a browserslist config in a
    parent directory would commit a different bundle as the "fix".

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if difference.identical:
    Becomes: if True:
    Killed by: src/uclone_x/cli/bundle_freshness.py :: " ".join(f"{name}={value}" for name, value in BUILD_ENVIRONMENT.items())
    Becomes: ""
    """
    repo = _repo(tmp_path, committed={**_BUNDLE, "assets/InternalsDock-OLD.js": b"x"})
    _npm_on_path(monkeypatch, present=True)
    _fake_bundler(monkeypatch, emits=_BUNDLE)

    status, code, lines = check_bundle_freshness(repo)

    assert status == "stale"
    assert code != 0
    text = "\n".join(lines)
    assert "assets/InternalsDock-OLD.js" in text
    assert "(cd frontend && NODE_ENV=production BROWSERSLIST=defaults npm run build)" in text


@pytest.mark.parametrize("returncode", [2, 127])
def test_a_failed_build_fails_the_stage_with_its_exit_code(
    returncode: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that failed compares nothing, so it cannot be a pass — even if it emitted files.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if build.returncode != 0:
    Becomes: if build.returncode < 0:
    Killed by: src/uclone_x/cli/bundle_freshness.py :: f"[dim]{escape(output)}[/dim]"
    Becomes: f"[dim]{output}[/dim]"
    """
    repo = _repo(tmp_path)
    _npm_on_path(monkeypatch, present=True)
    _fake_bundler(monkeypatch, emits=_BUNDLE, returncode=returncode)

    status, code, lines = check_bundle_freshness(repo)

    assert (status, code) == ("build-failed", returncode)
    # Printed the way the gate prints it: bundler output carries `[plugin:name]` prefixes,
    # and unescaped they are Rich markup that raises instead of reporting the failure.
    console = Console(file=io.StringIO(), width=200)
    for line in lines:
        console.print(line, soft_wrap=True)
    assert "[vite:esbuild] boom [/x]" in cast(io.StringIO, console.file).getvalue()


def test_a_build_that_exits_zero_having_emitted_nothing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape of a bundler that ignored `--outDir`: success, and nothing where it was told.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if not _digests(built):
    Becomes: if False:
    """
    repo = _repo(tmp_path, committed={})
    _npm_on_path(monkeypatch, present=True)
    _fake_bundler(monkeypatch, emits={})

    status, code, _ = check_bundle_freshness(repo)

    assert (status, code != 0) == ("build-failed", True)


def test_missing_node_modules_fails_instead_of_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh worktree has no `node_modules`; a skip there is a check nobody runs (P6).

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if not (frontend / "node_modules").is_dir():
    Becomes: if False:
    """
    repo = _repo(tmp_path, node_modules=False)
    _npm_on_path(monkeypatch, present=True)
    _forbid_subprocess(monkeypatch)

    status, code, lines = check_bundle_freshness(repo)

    assert (status, code != 0) == ("deps-missing", True)
    assert "node_modules" in "\n".join(lines)


def test_missing_npm_fails_instead_of_skipping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/cli/bundle_freshness.py :: if shutil.which("npm") is None:
    Becomes: if False:
    """
    repo = _repo(tmp_path)
    _npm_on_path(monkeypatch, present=False)
    _forbid_subprocess(monkeypatch)

    status, code, _ = check_bundle_freshness(repo)

    assert (status, code != 0) == ("npm-missing", True)


def test_npm_vanishing_between_lookup_and_run_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/cli/bundle_freshness.py :: except OSError as exc:
    Becomes: except LookupError as exc:
    """
    repo = _repo(tmp_path)
    _npm_on_path(monkeypatch, present=True)

    def no_npm(cmd: list[str], *args: object, **kwargs: object) -> object:
        raise FileNotFoundError(2, "No such file or directory: 'npm'")

    monkeypatch.setattr(subprocess, "run", no_npm)

    status, code, _ = check_bundle_freshness(repo)

    assert (status, code != 0) == ("npm-missing", True)


def test_node_env_in_the_shell_does_not_change_what_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`NODE_ENV=development` in the shell must not turn a fresh bundle stale (review of #956).

    Unpinned, vite built React's development bundle, the stage reported `stale`, and its fix
    committed that bundle. The build is handed `NODE_ENV=production` over the rest of the
    environment, which it keeps; a process-environment value also beats `frontend/.env.local`.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: env={**os.environ, **BUILD_ENVIRONMENT},
    Becomes: env={**os.environ},
    """
    monkeypatch.setenv("NODE_ENV", "development")
    monkeypatch.setenv("UCX_BUNDLE_PROBE", "kept")
    repo = _repo(tmp_path)
    _npm_on_path(monkeypatch, present=True)
    given: list[object] = []
    _fake_bundler(monkeypatch, emits=_BUNDLE, environments=given)

    status, code, _ = check_bundle_freshness(repo)

    assert (status, code) == ("fresh", 0)
    (environment,) = given
    assert isinstance(environment, dict)
    assert cast(dict[str, str], environment)["UCX_BUNDLE_PROBE"] == "kept"


def test_browser_targets_from_outside_the_tree_do_not_change_what_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Browserslist settings in the shell must not turn a fresh bundle stale (review of #956).

    Autoprefixer's targets come from browserslist: the `BROWSERSLIST` variable, else a
    `BROWSERSLIST_CONFIG` file, else a config in `frontend/` or any parent directory. Unpinned,
    each of these made the real build write different CSS, so the stage reported `stale` and
    its fix committed that CSS. The build is handed `BROWSERSLIST=defaults`, which browserslist
    reads before any file and which is its own no-config value, so the committed bytes result.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: , "BROWSERSLIST": "defaults"
    Becomes:
    """
    old_browsers = tmp_path / "old-browsers.browserslistrc"
    old_browsers.write_text("chrome 30, safari 6, ie 10\n")
    monkeypatch.setenv("BROWSERSLIST", "chrome 30, safari 6, ie 10")
    monkeypatch.setenv("BROWSERSLIST_CONFIG", str(old_browsers))
    repo = _repo(tmp_path / "repo")
    _npm_on_path(monkeypatch, present=True)
    given: list[object] = []
    _fake_bundler(monkeypatch, emits=_BUNDLE, environments=given)

    status, code, _ = check_bundle_freshness(repo)

    assert (status, code) == ("fresh", 0)
    (environment,) = given
    assert isinstance(environment, dict)
    assert cast(dict[str, str], environment)["BROWSERSLIST"] == "defaults"


def test_a_build_that_changes_the_committed_bundle_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundler writing both directories would compare `fresh` by construction; it must not.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if _digests(committed) != committed_before:
    Becomes: if False:
    """
    repo = _repo(tmp_path)
    _npm_on_path(monkeypatch, present=True)
    _fake_bundler(
        monkeypatch,
        emits={**_BUNDLE, "assets/index-REWRITTEN.js": b"x"},
        also_writes=repo / BUNDLE_DIR,
    )

    status, code, _ = check_bundle_freshness(repo)

    assert (status, code != 0) == ("bundle-written", True)


def test_a_committed_bundle_with_no_frontend_source_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle whose source is gone cannot be checked, so it is not waved through as `absent`.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if committed.exists():
    Becomes: if False:
    """
    _write_tree(tmp_path / BUNDLE_DIR, _BUNDLE)
    _npm_on_path(monkeypatch, present=True)
    _forbid_subprocess(monkeypatch)

    status, code, lines = check_bundle_freshness(tmp_path)

    assert (status, code != 0) == ("source-missing", True)
    assert "frontend/package.json" in "\n".join(lines)


def test_a_tree_without_a_frontend_is_absent_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `frontend/package.json` never claimed a bundle; it passes, out loud.

    Killed by: src/uclone_x/cli/bundle_freshness.py :: if not (frontend / "package.json").is_file():
    Becomes: if False:
    """
    _npm_on_path(monkeypatch, present=True)
    _forbid_subprocess(monkeypatch)

    status, code, lines = check_bundle_freshness(tmp_path)

    assert (status, code) == ("absent", 0)
    assert lines
