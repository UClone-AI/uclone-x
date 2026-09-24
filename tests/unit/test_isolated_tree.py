"""`isolated_copy_of_tree`: the private tree the mutation ratchet writes its mutants into (#967)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.support.isolated_tree import isolated_copy_of_tree


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _committed_repository(root: Path) -> Path:
    root.mkdir()
    for name in ("kept.py", "edited.py", "deleted.py"):
        (root / name).write_text("committed\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return root


def test_the_copy_carries_uncommitted_and_untracked_work(tmp_path: Path) -> None:
    """The gate runs over uncommitted work, and the declarations it checks are usually in it.

    Killed by: tests/support/isolated_tree.py :: shutil.copy2(on_disk, copied)
    Becomes: pass
    """
    source = _committed_repository(tmp_path / "source")
    (source / "edited.py").write_text("uncommitted edit\n", encoding="utf-8")
    (source / "new.py").write_text("untracked\n", encoding="utf-8")

    copy = isolated_copy_of_tree(source, tmp_path / "copy")

    assert (copy / "kept.py").read_text(encoding="utf-8") == "committed\n"
    assert (copy / "edited.py").read_text(encoding="utf-8") == "uncommitted edit\n"
    assert (copy / "new.py").read_text(encoding="utf-8") == "untracked\n"


def test_a_staged_change_is_carried_too(tmp_path: Path) -> None:
    """Staged work is uncommitted work, and the gate stages before it runs.

    `./ucx test check` is run after `git add -A`, because one of its checks reads every
    git-tracked file from disk. A copy that compared the working tree to the *index*
    saw nothing after that, so the lethality ratchet applied every mutation to the source as
    it was before the change and ran it against the tests as they were before the change. A
    new test failed to resolve and was reported; an edited assertion on an existing test
    scored `KILLED` and was not.

    Killed by: tests/support/isolated_tree.py :: "diff", "--name-only", "--no-renames", "-z", "HEAD"
    Becomes: "diff", "--name-only", "--no-renames", "-z"
    """
    source = _committed_repository(tmp_path / "source")
    (source / "edited.py").write_text("staged edit\n", encoding="utf-8")
    (source / "added.py").write_text("staged addition\n", encoding="utf-8")
    _git(source, "add", "-A")

    copy = isolated_copy_of_tree(source, tmp_path / "copy")

    assert (copy / "edited.py").read_text(encoding="utf-8") == "staged edit\n"
    assert (copy / "added.py").read_text(encoding="utf-8") == "staged addition\n"


def test_a_file_deleted_on_disk_is_absent_from_the_copy(tmp_path: Path) -> None:
    """A file deleted in the working tree must not reappear in the copy from `HEAD`.

    Killed by: tests/support/isolated_tree.py :: copied.unlink(missing_ok=True)
    Becomes: pass
    """
    source = _committed_repository(tmp_path / "source")
    (source / "deleted.py").unlink()

    copy = isolated_copy_of_tree(source, tmp_path / "copy")

    assert not (copy / "deleted.py").exists()


def test_the_copy_is_a_repository_at_the_same_head_and_writes_stay_in_it(tmp_path: Path) -> None:
    """The harness reads pristine content with `git show HEAD:<path>` under the tree it is given."""
    source = _committed_repository(tmp_path / "source")
    (source / "edited.py").write_text("uncommitted edit\n", encoding="utf-8")

    copy = isolated_copy_of_tree(source, tmp_path / "copy")

    assert _git(copy, "rev-parse", "HEAD") == _git(source, "rev-parse", "HEAD")
    assert _git(copy, "show", "HEAD:edited.py") == "committed\n"
    (copy / "edited.py").write_text("mutant\n", encoding="utf-8")
    assert (source / "edited.py").read_text(encoding="utf-8") == "uncommitted edit\n"
