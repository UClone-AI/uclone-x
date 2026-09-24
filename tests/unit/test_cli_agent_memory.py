"""The shells' `--agent-id` is a directory name, and a bad one is a usage error (#1125)."""

from __future__ import annotations

import pytest
import typer
from rich.console import Console

from uclone_x.cli import agent_memory
from uclone_x.cli.agent_memory import memory_for_agent_id


def test_an_agent_id_no_directory_can_carry_exits_two_with_a_message(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ucx acp serve --agent-id "My Agent"` printed a framed traceback at the user.

    The id becomes the name of that agent's home directory, so the refusal is a mistake in
    the argument -- the same class as `ucx run`'s, which already exits 2 with the rule it
    broke. A traceback tells a non-expert that the product is broken (P0), and it says the
    fault is in the memory store rather than in what they typed (P6).

    Killed by: src/uclone_x/cli/agent_memory.py :: raise typer.Exit(code=2) from exc
    Becomes: raise
    """
    monkeypatch.setattr(agent_memory, "_console", Console(force_terminal=False, width=200))

    with pytest.raises(typer.Exit) as excinfo:
        memory_for_agent_id("My Agent")

    assert excinfo.value.exit_code == 2
    out = capsys.readouterr().out
    assert "Invalid --agent-id:" in out
    assert "My Agent" in out, "the person cannot fix an id the refusal withholds"


def test_a_usable_agent_id_opens_its_memory(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The helper is the normal path too, not only the refusal.

    Killed by: src/uclone_x/cli/agent_memory.py :: return default_cross_session_memory(agent_id)
    Becomes: return default_cross_session_memory("default")
    """
    memory = memory_for_agent_id("shell-agent")

    assert memory.storage_path is not None, "a shell agent with no file has nothing to remember"
    assert memory.storage_path.parent.name == "shell-agent"
    assert memory.storage_path.name == "memory.json"
