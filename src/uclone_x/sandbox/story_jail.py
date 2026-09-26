"""Run a model-started process so it cannot write the story library (#1589).

The file tools refuse a path in `<workspace>/stories` (`tools.base.in_story_library`), so
a story is changed only through the story tools, which check the lease, the digest and a
person's approval. A shell command (`bash_run`, `run_command`) and a local MCP server are
separate processes: nothing in them passes through that check, and what a command string
writes cannot be read off it. So the check is made by the operating system instead.

* **macOS.** The process runs under `sandbox-exec` with a profile that allows everything
  except writing in the library. Writing, creating, deleting, renaming, changing a mode,
  and making a hardlink to a library file are refused with "Operation not permitted",
  whatever path names the file (another case, a `..` detour, a symlink into the library),
  because the check is made on the file that path opens. The
  workspace folder and each folder above it cannot be renamed, which would otherwise move
  the library to a path the profile does not name. Children of the process inherit it.
* **Other systems.** There is no jail: `story_library_jail` returns no prefix, and the
  process can write the library.

`sandbox-exec` is marked DEPRECATED in its macOS manual page, which points apps to the App
Sandbox instead. It still ships and works, but a later macOS could remove it; the jail then
refuses every command rather than running it unprotected.

What this does not stop:

* a jailed process asking a process that is not jailed to write for it (the local API,
  another application);
* a hardlink to a library file that already existed before the process started;
* a local MCP server whose configured `workspace_root` is not the app's workspace. The jail
  protects the library of the root the server is given. A server loaded from `mcp.json`
  can name its own root, and without one gets the loader's root or the current folder, so
  its jail can name a folder that is not where the stories are;
* MCP servers reached over HTTP, which are not local processes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from uclone_x.errors import PlainRefusalError

__all__ = [
    "PLATFORM",
    "SANDBOX_EXEC",
    "JAIL_SETUP_REFUSAL",
    "story_library_jail",
    "jailed_shell",
]

#: Where macOS keeps `sandbox-exec`. An absolute path, so `PATH` cannot substitute another.
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

#: What the model is told when the jail could not be set up, so the command did not run.
JAIL_SETUP_REFUSAL = (
    "The command was not run: this computer could not start it in the way that keeps it "
    "from changing the story library."
)

#: The system this runs on, read at call time.
PLATFORM = sys.platform

#: Creates the file named by `$0`, then runs `$1` in a new `/bin/sh -c`.
_MARK_STARTED = ': > "$0" && exec /bin/sh -c "$1"'

_PROFILE_HEAD = "(version 1)\n(allow default)\n(deny file-write*\n"


def story_library_jail(workspace_root: Path | None) -> list[str]:
    """The arguments to put before a command so it cannot write `workspace_root`'s library.

    Empty when there is no workspace (so no library) or no jail on this system. On macOS,
    a missing `sandbox-exec` is refused rather than run without the jail.
    """
    if workspace_root is None or PLATFORM != "darwin":
        return []
    if not SANDBOX_EXEC.is_file():
        raise PlainRefusalError(JAIL_SETUP_REFUSAL)

    # Imported here: `uclone_x.story` imports `uclone_x.tools.models`, which imports
    # `uclone_x.sandbox`.
    from uclone_x.story.schemas import STORIES_DIRNAME

    root = workspace_root.resolve()
    library = root / STORIES_DIRNAME
    subpaths = [library]
    if library.is_symlink() or library.exists():
        target = library.resolve()
        if target != library:
            subpaths.append(target)
    literals: list[Path] = []
    for path in subpaths:
        for folder in path.parents:
            if folder not in literals:
                literals.append(folder)

    # Paths go in as parameters, not into the profile's text, so no path needs quoting.
    argv = [str(SANDBOX_EXEC)]
    rules: list[str] = []
    for index, path in enumerate(subpaths):
        argv += ["-D", f"LIBRARY_{index}={path}"]
        rules.append(f'(subpath (param "LIBRARY_{index}"))')
    for index, path in enumerate(literals):
        argv += ["-D", f"FOLDER_{index}={path}"]
        rules.append(f'(literal (param "FOLDER_{index}"))')
    argv += ["-p", _PROFILE_HEAD + "\n".join(rules) + ")\n"]
    return argv


def jailed_shell(jail: list[str], command: str, started: Path | None) -> list[str]:
    """The arguments that run `command` with `/bin/sh -c` inside `jail`.

    With `started`, the shell first creates that file and only then runs the command, so
    its presence afterwards tells a command that ran (whatever it printed or exited with,
    `sandbox-exec` included) from a jail that never started one. The command still runs as
    `/bin/sh -c <command>`, as `create_subprocess_shell` would run it. The file's path
    and the command are passed as arguments, not written into the script.
    """
    if started is None:
        return [*jail, "/bin/sh", "-c", command]
    return [*jail, "/bin/sh", "-c", _MARK_STARTED, str(started), command]
