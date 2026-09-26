"""In-memory A2A handlers that answer a peer call with a one-off persona agent (#1558).

`a2a_call` sends a `TaskMessage` naming a persona; the handler here builds an agent for
that persona the way `RoomAgentResolver` builds a seat -- same host, same persona registry,
same config rule -- runs the task as one turn, and answers with what it said, the files it
wrote and the steps it took.

What makes the agent one-off, and why each part is there:

* **No memory between calls** (owner decision, 2026-09-24). A fresh session per call, kept
  in a temporary store that is removed afterwards, no cross-session memory store, and no
  memory tools in its list -- the rule `spawn_subagent` applies to a child (#1431).
* **One level deep.** The agent is given no A2A transport, and `a2a_call` is taken out of
  its persona's tools, so it cannot call anyone in turn. A message that says it is deeper
  than one level, or does not say how deep it is, is refused before anything is built.
* **The caller's budget (P4).** The agent's step ceiling is what the caller has left, sent
  as `step_budget`, and its steps are reported back for the caller to be charged. When it
  uses them all, the caller is told it ran out of steps.
* **The caller's conversation and story** (#1555). The conversation and open story the
  caller sent are the ones the agent's turn runs in, so its tool calls see them on their
  `ToolContext`: the same story folder as the caller, and story writes checked against the
  lease as the caller conversation's. Neither is sent for a caller outside a conversation,
  and then the agent has none -- a story write is refused with that reason.
* **No unapproved tool.** A call a hook would put to a person is refused inside the agent
  and the task ends as `INPUT_REQUIRED`, naming the tool. There is nobody on a peer call to
  approve it, and waiting for an answer that cannot come only delays the same refusal.

A shell, like the resolver: it composes agents.
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from uclone_x.a2a.in_memory import A2AInMemoryTransport
from uclone_x.a2a.models import TaskMessage, TaskResult, TaskStatus
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.bootstrap import agent_config_for_persona
from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.hooks import HookAction, HookContext, HookDecision, HookEvent, HookRunner
from uclone_x.agent.models import (
    BASE_MEMORY_TOOLS,
    DEFAULT_SYSTEM_PROMPT,
    AgentContext,
    AgentLLMConfig,
    PersonaDefinition,
    TurnResult,
)
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.tools.builtin.a2a import (
    A2A_CALL_TOOL_NAME,
    A2A_DEPTH_KEY,
    A2A_ROOM_KEY,
    A2A_STEP_BUDGET_KEY,
    A2A_STORY_KEY,
)
from uclone_x.tools.models import ToolResultStatus
from uclone_x.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["PersonaTaskHandler", "register_persona_handlers"]

#: Attribution for an answer this handler wrote itself -- a refusal, or a failure it can
#: name -- as opposed to one the called agent's model produced.
_HANDLER_PROVIDER = "local.a2a"

#: The `stop_reason` of a turn that used every step it was given -- here, the caller's
#: remaining budget -- so the caller hears that rather than a bare "could not finish".
_STEP_CEILING = "step_budget_exceeded"


def _handler_provenance(persona: str) -> Provenance:
    return Provenance.primary(provider=_HANDLER_PROVIDER, model=f"persona:{persona}")


class _DeferredApprovalRunner(HookRunner):
    """A hook runner that refuses, rather than asks about, a tool needing approval.

    Records each refused tool so the handler can answer `INPUT_REQUIRED` naming it.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pending: list[str] = []

    async def run_hooks(self, event_type: HookEvent, context: HookContext) -> HookDecision:
        decision = await super().run_hooks(event_type, context)
        if event_type == HookEvent.PRE_TOOL_USE and decision.action == HookAction.ASK:
            tool = str(context.payload.get("tool_name") or "a tool")
            self.pending.append(tool)
            return HookDecision(
                action=HookAction.BLOCK,
                reason=(
                    f"{tool} needs a person's approval, and nobody can give it during a "
                    "task from another persona. It did not run."
                ),
            )
        return decision


def _written_paths(result: TurnResult) -> list[str]:
    """Files the turn's own tools say they wrote, each once, in the order they were written."""
    paths: list[str] = []
    for execution in result.tool_executions:
        if not execution.writes_files or execution.status is not ToolResultStatus.SUCCESS:
            continue
        output = execution.output
        if not isinstance(output, Mapping):
            continue
        candidates: list[object] = [output.get("path")]
        listed = output.get("paths")
        if isinstance(listed, list | tuple):
            candidates.extend(listed)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate and candidate not in paths:
                paths.append(candidate)
    return paths


def _wrote_unnamed(result: TurnResult) -> bool:
    """Whether the turn may have written a file it does not name -- the room's rule (#1366).

    A call to a writing tool that did not both succeed and name a path, any call that
    started a helper, or a turn whose tool records are incomplete.
    """
    if not result.tool_executions_complete:
        return True
    for execution in result.tool_executions:
        if execution.spawns_subagents:
            return True
        if not execution.writes_files:
            continue
        output = execution.output
        named = isinstance(output, Mapping) and (
            isinstance(output.get("path"), str) or bool(output.get("paths"))
        )
        if execution.status is not ToolResultStatus.SUCCESS or not named:
            return True
    return False


def _prompt(message: TaskMessage) -> str:
    task = message.input_data.get("task")
    details = message.input_data.get("input")
    text = task if isinstance(task, str) else ""
    if isinstance(details, Mapping) and details:
        lines = "\n".join(f"- {key}: {value}" for key, value in details.items())
        text = f"{text}\n\nDetails:\n{lines}"
    return (
        f"{message.sender_agent_id} asks you to do this task. Do it with your own tools, "
        f"then say briefly what you made.\n\n{text}"
    )


def _step_budget(message: TaskMessage) -> int | None:
    raw = message.metadata.get(A2A_STEP_BUDGET_KEY)
    try:
        value = int(raw) if raw is not None else None
    except ValueError:
        return None
    return value if value is not None and value > 0 else None


def _forwarded(message: TaskMessage, key: str) -> str | None:
    """A conversation or story id the caller sent, or `None` when it sent none."""
    value = message.metadata.get(key)
    return value if isinstance(value, str) and value else None


class PersonaTaskHandler:
    """Answers `TaskMessage`s addressed to one persona, with a fresh agent per task."""

    def __init__(
        self,
        persona_name: str,
        *,
        host_factory: Callable[[], HostDependencies],
        persona_registry: PersonaRegistry,
        workspace_root: Path,
        llm_config: AgentLLMConfig | None = None,
        read_roots: Callable[[], tuple[Path, ...]] | None = None,
    ) -> None:
        self._persona_name = persona_name
        self._host_factory = host_factory
        self._registry = persona_registry
        self._workspace_root = workspace_root
        self._llm_config = llm_config
        self._read_roots: Callable[[], tuple[Path, ...]] = read_roots or (lambda: ())

    def _refuse(self, message: TaskMessage, error: str) -> TaskResult:
        return TaskResult(
            task_id=message.task_id,
            status=TaskStatus.REJECTED,
            output_data={"steps": 0, "paths": []},
            error=error,
            provenance=_handler_provenance(self._persona_name),
        )

    @staticmethod
    def _callee_persona(persona: PersonaDefinition) -> PersonaDefinition:
        """The persona as the called agent runs it: no `a2a_call`, and no one to call."""
        tools = tuple(name for name in persona.allowed_tools if name != A2A_CALL_TOOL_NAME)
        return persona.model_copy(update={"allowed_tools": tools, "a2a_peers": ()})

    @staticmethod
    def _callee_tools(persona: PersonaDefinition) -> tuple[str, ...]:
        """The operator list the called agent is held to: its persona's, less memory.

        Written into the config rather than left to the persona, because the persona's
        list always carries the memory tools (`BASE_PERSONA_TOOLS`) and this agent has no
        memory to use them on. An unrestricted persona stays unrestricted; with no store its
        memory tools are neither offered nor resolvable (#1098).
        """
        return tuple(
            name
            for name in persona.granted_tools
            if name not in BASE_MEMORY_TOOLS and name != A2A_CALL_TOOL_NAME
        )

    def _compose(
        self,
        message: TaskMessage,
        persona: PersonaDefinition,
        runner: _DeferredApprovalRunner,
        scratch: Path,
    ) -> BaseAgent:
        """Build the one-off agent for this task, under the caller's step budget."""
        base = self._host_factory()
        inherited = tuple(base.hooks or ()) + (
            base.hook_runner.hooks if base.hook_runner is not None else ()
        )
        host = dataclasses.replace(
            base,
            store=SessionStore(storage_dir=scratch),
            memory=None,
            ontology=None,
            hooks=inherited or None,
            hook_runner=runner,
            a2a_transport=None,
            persona=persona.name,
            persona_name=persona.name,
            persona_definitions=(persona,),
        )
        config = agent_config_for_persona(
            persona,
            agent_id=persona.name,
            name=persona.name,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            llm_config=self._llm_config or persona.llm_config,
            workspace_dir=self._workspace_root,
            read_roots=self._read_roots(),
        )
        update: dict[str, Any] = {"allowed_tools": self._callee_tools(persona)}
        budget = _step_budget(message)
        if budget is not None:
            update |= {"max_steps": budget, "max_turns": budget}
        config = config.model_copy(update=update)
        if persona.allowed_tools and not config.allowed_tools:
            # Permitted only memory tools: an empty list would permit everything.
            host = dataclasses.replace(host, tools=ToolRegistry(), skills=None)
        return compose_agent(
            config=config,
            host=host,
            context=AgentContext(
                session_id=f"a2a_{message.task_id}",
                agent_id=persona.name,
                workspace_root=self._workspace_root,
                parent_agent_id=message.sender_agent_id,
                depth=1,
            ),
        )

    async def __call__(self, message: TaskMessage) -> TaskResult:
        depth = message.metadata.get(A2A_DEPTH_KEY)
        if depth is None:
            # Only `a2a_call` sends here, and it always says how deep the call is. A
            # message that does not say is not taken as the shallowest one (#1570).
            return self._refuse(
                message,
                "This task did not give its call depth (how many personas passed it on), "
                "so it was not taken.",
            )
        if depth != "1":
            return self._refuse(
                message, "A task from another persona cannot be passed on to a third one."
            )
        found = self._registry.get_persona(self._persona_name)
        if found is None:
            return self._refuse(message, f"There is no persona named '{self._persona_name}'.")
        persona = self._callee_persona(found)

        runner = _DeferredApprovalRunner()
        agent: BaseAgent | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="ucx-a2a-") as scratch:
                # Built inside the `try`: a host or config that cannot be set up must
                # reach the caller as a sentence, not as the exception's name (#1570).
                agent = self._compose(message, persona, runner, Path(scratch))
                turn = await agent.execute_turn(
                    _prompt(message),
                    room_id=_forwarded(message, A2A_ROOM_KEY),
                    story_id=_forwarded(message, A2A_STORY_KEY),
                )
        except Exception as exc:  # the caller gets a sentence, the log gets the cause
            if agent is None:
                logger.warning(
                    "a2a: %s could not be set up for a task: %s", persona.name, exc, exc_info=True
                )
                return self._refuse(
                    message,
                    f"{persona.name} could not be started for this task, so nothing was done.",
                )
            logger.warning(
                "a2a: %s failed while doing a task: %s", persona.name, exc, exc_info=True
            )
            return TaskResult(
                task_id=message.task_id,
                status=TaskStatus.FAILED,
                output_data={"steps": agent.run_steps, "paths": [], "unnamed_writes": True},
                error=f"{persona.name} ran into a problem and could not finish.",
                provenance=_handler_provenance(persona.name),
            )

        steps = agent.run_steps
        paths = _written_paths(turn)
        output: dict[str, Any] = {
            "steps": steps,
            "paths": paths,
            "unnamed_writes": _wrote_unnamed(turn),
            "response": turn.content,
        }
        if runner.pending:
            waiting = ", ".join(dict.fromkeys(runner.pending))
            return TaskResult(
                task_id=message.task_id,
                status=TaskStatus.INPUT_REQUIRED,
                output_data=output,
                error=f"{persona.name} needed approval to use {waiting}, so it did not run.",
                provenance=turn.provenance or _handler_provenance(persona.name),
            )
        if turn.provenance is None or turn.error or not turn.is_completed:
            out_of_steps = turn.stop_reason == _STEP_CEILING or (
                not turn.error and not turn.is_completed
            )
            reason = "it ran out of steps" if out_of_steps else "it could not finish"
            return TaskResult(
                task_id=message.task_id,
                status=TaskStatus.FAILED,
                output_data=output,
                error=f"{persona.name} stopped before finishing: {reason}.",
                provenance=turn.provenance or _handler_provenance(persona.name),
            )
        return TaskResult(
            task_id=message.task_id,
            status=TaskStatus.COMPLETED,
            output_data=output,
            provenance=turn.provenance,
        )


def register_persona_handlers(
    transport: A2AInMemoryTransport,
    *,
    host_factory: Callable[[], HostDependencies],
    persona_registry: PersonaRegistry,
    workspace_root: Path,
    llm_config: AgentLLMConfig | None = None,
    read_roots: Callable[[], tuple[Path, ...]] | None = None,
    personas: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Register a handler for every persona some persona may call; return their names.

    `personas` names them explicitly; by default they are read from every loaded persona's
    `a2a_peers`. A peer named there that no file defines is registered anyway and answers
    that it does not exist, so the caller hears that rather than "not available".
    """
    names = (
        tuple(personas)
        if personas is not None
        else tuple(
            dict.fromkeys(
                peer for persona in persona_registry.list_personas() for peer in persona.a2a_peers
            )
        )
    )
    for name in names:
        transport.register_handler(
            name,
            PersonaTaskHandler(
                name,
                host_factory=host_factory,
                persona_registry=persona_registry,
                workspace_root=workspace_root,
                llm_config=llm_config,
                read_roots=read_roots,
            ),
        )
    return names
