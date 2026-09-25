"""Check that an installed `ucx` is a user build.

An installed distribution registers the user commands and none of the maintainer
ones (`test`, `dev`, `setup`), which drive a source checkout and are registered only
from one. CI runs this with the interpreter of an environment the built wheel was
installed into, from outside the checkout.

The command list is read from the command group itself, not from `ucx --help`.
The help is drawn by Rich, which forces styled output under `GITHUB_ACTIONS`: every
command name arrives wrapped in escape sequences inside a box, and a pattern over
that text reported `run` missing from a help screen that plainly listed it. What the
group registers is the fact the help is drawn from, so it is what is asked.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from typing import Protocol, cast

USER_COMMANDS = ("run", "ui", "llm", "status", "version")
MAINTAINER_COMMANDS = ("test", "dev", "setup")


class Command(Protocol):
    """The part of a command this check reads."""

    @property
    def hidden(self) -> bool: ...


class CommandGroup(Protocol):
    """The part of a command group this check reads.

    Structural rather than `click.Group`: Typer now builds its groups on a Click it
    vendors, so a `TyperGroup` is not an instance of the `click` package's `Group`.
    """

    @property
    def commands(self) -> Mapping[str, Command]: ...


def registered_commands(group: CommandGroup) -> dict[str, bool]:
    """Every top-level command of `group`, mapped to whether its help shows it."""
    return {name: not command.hidden for name, command in group.commands.items()}


def user_build_problems(
    commands: dict[str, bool],
    user_commands: Iterable[str] = USER_COMMANDS,
    maintainer_commands: Iterable[str] = MAINTAINER_COMMANDS,
) -> list[str]:
    """What makes `commands` something other than a user build; empty when it is one.

    A user command must be registered and shown. A maintainer command must not be
    registered at all: hiding it from the help would still let `ucx test check` run.
    """
    problems: list[str] = []
    for name in user_commands:
        if name not in commands:
            problems.append(f"user command '{name}' missing from the installed build")
        elif not commands[name]:
            problems.append(f"user command '{name}' is hidden in the installed build")
    for name in maintainer_commands:
        if name in commands:
            problems.append(f"maintainer command '{name}' is exposed to users")
    return problems


def main() -> int:
    """Inspect the installed `ucx` and report, as CI annotations, what is wrong with it."""
    import typer

    import uclone_x
    from uclone_x.cli.main import app

    group: object = typer.main.get_command(app)
    if not hasattr(group, "commands"):
        print(f"::error::ucx is a {type(group).__name__}, not a command group", file=sys.stderr)
        return 1

    commands = registered_commands(cast(CommandGroup, group))
    print(f"uclone_x from {uclone_x.__file__}")
    print("registered commands: " + ", ".join(sorted(commands)))
    problems = user_build_problems(commands)
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
