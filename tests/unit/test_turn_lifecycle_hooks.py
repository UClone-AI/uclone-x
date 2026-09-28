"""The agent imports no domain; the story moves through a composed lifecycle hook (#1732).

`BaseAgent` carries a turn's `story_id` to its tool calls but no longer decides how it
moves: `uclone_x.story.StoryLifecycleHook` does, and every head composes it in through
`agent/clone_builder.py`. These tests pin the three halves of that: the agent package
names nothing in `uclone_x.story`, the hook moves the story exactly as `story_after` says,
and both places that compose an agent for a head -- `build_clone` and the peer-call
handler -- put the hook in.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from tests.support.import_resolution import imported_modules, package_of
from uclone_x.a2a.models import TaskMessage
from uclone_x.agent.clone_builder import AppScope, build_clone, with_app_lifecycle_hooks
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import PersonaDefinition, ToolExecutionRecord
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.room import a2a_handlers
from uclone_x.room.a2a_handlers import PersonaTaskHandler
from uclone_x.story import OPEN_STORY_KEY, StoryLifecycleHook
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.builtin.a2a import A2A_DEPTH_KEY
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry

AGENT_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "uclone_x" / "agent"


def _host(tmp_path: Path) -> HostDependencies:
    return HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(default_response="Done."),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )


def _story_hooks(hooks: tuple[Any, ...]) -> list[Any]:
    return [h for h in hooks if isinstance(h, StoryLifecycleHook)]


def test_the_agent_package_imports_nothing_from_the_story_domain() -> None:
    """No module under `agent/` but the shell `clone_builder.py` imports `uclone_x.story` (#1732).

    Checked at every scope, `TYPE_CHECKING` included; a relative import is resolved
    against its module's package, and `from uclone_x import story` counts too.
    """
    offenders: list[str] = []
    for path in sorted(AGENT_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            package = package_of(f"agent/{path.relative_to(AGENT_PACKAGE).as_posix()}")
            names = imported_modules(node, package)
            hits = [n for n in names if n == "uclone_x.story" or n.startswith("uclone_x.story.")]
            if hits:
                offenders.append(f"{path.relative_to(AGENT_PACKAGE)}: {hits[0]}")
    # `clone_builder.py` is the shell that composes the hook in; it is the one allowed name.
    assert offenders == ["clone_builder.py: uclone_x.story"]


def test_the_story_hook_moves_the_story_a_declaring_call_names(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/story/__init__.py :: return context.model_copy(update={"story_id": story})
    Becomes: return context
    """
    hook = StoryLifecycleHook()
    context = ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path, story_id="old")
    opened = ToolExecutionRecord(
        tool_name="story_library",
        output={OPEN_STORY_KEY: "new"},
        status=ToolResultStatus.SUCCESS,
        opens_story=True,
    )
    silent = opened.model_copy(update={"opens_story": False})

    assert hook.after_tool_step((opened,), context).story_id == "new"
    assert hook.after_tool_step((silent,), context) is context


def test_every_built_clone_runs_with_the_story_hook(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_builder.py :: host = with_app_lifecycle_hooks(app.host)
    Becomes: host = app.host
    """
    base = _host(tmp_path)
    app = AppScope(
        host=base,
        workspace_root=tmp_path,
        persona_registry=PersonaRegistry(include_defaults=False),
    )

    built = build_clone(app, clone_id="writer", session_id="sess")

    hooks = built.agent._lifecycle_hooks  # pyright: ignore[reportPrivateUsage]
    assert len(_story_hooks(hooks)) == 1


def test_composing_the_app_hooks_twice_adds_the_story_hook_once(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_builder.py :: if any(isinstance(hook, StoryLifecycleHook) for hook in host.lifecycle_hooks):
    Becomes: if False:
    """
    once = with_app_lifecycle_hooks(_host(tmp_path))
    twice = with_app_lifecycle_hooks(once)

    assert twice is once
    assert len(_story_hooks(twice.lifecycle_hooks)) == 1


@pytest.mark.asyncio
async def test_a_peer_call_agent_runs_with_the_story_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A callee works in its caller's story, so it must be able to move it as a seat does.

    Killed by: src/uclone_x/room/a2a_handlers.py :: base = with_app_lifecycle_hooks(self._host_factory())
    Becomes: base = self._host_factory()
    """
    composed: list[HostDependencies] = []
    real = a2a_handlers.compose_agent

    def spy(*, config: Any, host: HostDependencies, context: Any) -> Any:
        composed.append(host)
        return real(config=config, host=host, context=context)

    monkeypatch.setattr(a2a_handlers, "compose_agent", spy)
    registry = PersonaRegistry(include_defaults=False)
    registry.register_persona(
        PersonaDefinition(name="artist", role="Artist", system_prompt="You draw.")
    )
    handler = PersonaTaskHandler(
        "artist",
        host_factory=lambda: _host(tmp_path),
        persona_registry=registry,
        workspace_root=tmp_path,
    )

    await handler(
        TaskMessage(
            task_id="a2a_test",
            session_id="sess_writer",
            input_data={"task": "Draw the hero."},
            sender_agent_id="writer",
            target_agent_id="artist",
            metadata={A2A_DEPTH_KEY: "1"},
        )
    )

    (host,) = composed
    assert len(_story_hooks(host.lifecycle_hooks)) == 1
