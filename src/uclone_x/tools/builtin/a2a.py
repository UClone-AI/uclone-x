"""`a2a_call`: one persona asks another for a piece of work (#1558).

The Writer asks the Artist for a picture. The Artist is not seated in the conversation; it
is built for the one call, does the work with its own tools, and is gone (the owner's
decision of 2026-09-24: it keeps no memory between calls).

What the tool enforces, in the order it checks:

* **Who may be called.** Only a persona named in the caller's own `a2a_peers`. An empty
  list is no one, never everyone.
* **One level deep.** An agent built to answer a call is given no transport, so its own
  `a2a_call` has nothing to call through; a sub-agent is not given its parent's either. The
  depth in the agent's context is checked too, so the rule does not rest on one wiring.
* **The caller's budget (P4).** The peer runs on what is left of the caller's step budget,
  and the steps it took are charged to the caller, as a sub-agent's are.
* **The caller's conversation and story.** The call carries the caller's conversation
  (`ToolContext.room_id`) and the story it has open (`ToolContext.story_id`), and the peer's
  tool calls get both on their own `ToolContext` (#1555): the peer works in the same story
  folder, and a story write it makes is checked against the lease as the caller
  conversation's -- so a peer asked from a conversation that is not writing the story is
  refused as that conversation would be.
* **The story's characters as they look (#1808).** With a story open, the characters the
  task and input name are looked up in its codex and their `visual` blocks go with the
  call under `input["characters"]` -- unless the caller sent that key itself. The Artist
  never read the codex on its own (0 of 30 on qwen3:8b) and drew from the sheet when it
  was handed over (29 of 30). The lookup is given to the tool (`CharacterLookup`), so this
  module does not read a story.
* **No borrowed result (P6).** A peer that could not do the work -- refused, failed, or
  stopped at a tool that needs a person's approval -- is a failed call that says so. The
  caller is not handed a result nobody produced.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, cast

from pydantic import BaseModel, Field, JsonValue, ValidationError

from uclone_x.a2a.models import TaskMessage, TaskStatus
from uclone_x.core.agent_home import peer_handles
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.errors import MissingProvenanceError, PlainRefusalError, TaskNotFoundError
from uclone_x.tools.base import BaseTool, describe_invalid_arguments
from uclone_x.tools.models import ToolContext, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "A2A_CALLER_SESSION_KEY",
    "A2A_CALL_TOOL_NAME",
    "A2A_DEPTH_KEY",
    "A2A_ROOM_KEY",
    "A2A_STEP_BUDGET_KEY",
    "A2A_STORY_KEY",
    "A2A_CHARACTERS_KEY",
    "A2ACallParams",
    "A2ACallTool",
    "CharacterLookup",
]

A2A_CALL_TOOL_NAME = "a2a_call"
#: `TaskMessage.metadata` key: the steps the peer may take, which is what the caller has left.
A2A_STEP_BUDGET_KEY = "step_budget"
#: `TaskMessage.metadata` key: how deep this call is. A peer handler refuses anything past 1.
A2A_DEPTH_KEY = "a2a_depth"
#: `TaskMessage.metadata` key: the caller's session -- in a room, its seat in that
#: conversation. Kept for tracing which seat asked; the lease is checked against the
#: conversation (`A2A_ROOM_KEY`), not the seat.
A2A_CALLER_SESSION_KEY = "caller_session_id"
#: `TaskMessage.metadata` key: the caller's conversation (`ToolContext.room_id`), sent only
#: when the call is part of one. The peer's story writes are checked against it (#1555).
A2A_ROOM_KEY = "room_id"
#: `TaskMessage.metadata` key: the story the caller's conversation has open
#: (`ToolContext.story_id`), sent only when one is open.
A2A_STORY_KEY = "story_id"
#: `input` key: the open story's characters the call names, each with its `visual` block.
A2A_CHARACTERS_KEY = "characters"

#: Given the call's context (with a story open) and the words of its task and input, the
#: story's characters those words name, each as `{"id", "name", "visual"}`. Raises a
#: `PlainRefusalError` when the story cannot be read.
CharacterLookup = Callable[[ToolContext, Sequence[str]], list[dict[str, Any]]]


class A2ACallParams(BaseModel):
    agent: str = Field(description="The persona to ask, by name (for example 'artist').")
    task: str = Field(description="What you want it to do, in plain words.")
    input: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Anything it needs to do the work: names, descriptions, style notes.",
    )


def _refusal(message: str, context: ToolContext, start: float, tool: str) -> ToolResult:
    return ToolResult(
        success=False,
        error=message,
        execution_time_ms=(time.perf_counter() - start) * 1000.0,
        isolation_level=context.isolation.level,
        provenance=Provenance.primary(provider="local.builtin", model=tool),
    )


def _names(text: str, persona: str) -> bool:
    """Whether `text` is a sentence about `persona`: it starts with the name, or quotes it.

    As a whole word, not inside a longer name. A reason that merely uses the name as a
    word elsewhere is not about the persona -- for one named "a", "it needed a person's
    approval" -- and keeps the lead that says who stopped (#1602).
    """
    name = re.escape(persona)
    return re.match(rf"{name}(?![\w-])", text) is not None or f"'{persona}'" in text


def _words(value: JsonValue) -> list[str]:
    """Every string in a JSON value, keys included, for finding the names it holds."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [word for item in value for word in _words(item)]
    if isinstance(value, dict):
        return [word for key, item in value.items() for word in (key, *_words(item))]
    return []


def _reported_steps(output: Mapping[str, Any]) -> int:
    steps = output.get("steps")
    return steps if isinstance(steps, int) and not isinstance(steps, bool) and steps > 0 else 0


def _reported_paths(output: Mapping[str, Any]) -> list[str]:
    listed: object = output.get("paths")
    if not isinstance(listed, list | tuple):
        return []
    paths: list[str] = []
    for item in cast(Sequence[object], listed):
        if isinstance(item, str) and item and item not in paths:
            paths.append(item)
    return paths


class A2ACallTool(BaseTool[A2ACallParams]):
    name = A2A_CALL_TOOL_NAME
    #: The peer writes files with its own tools, and those files are this call's result:
    #: the room records them from this call's `paths` (#1558). Declared, so a caller whose
    #: persona may not write cannot have a peer write for it.
    writes_files: ClassVar[bool] = True
    spawns_subagents: ClassVar[bool] = False
    description = (
        "Ask another persona you work with to do a task with its own tools, and get back "
        "what it made. Only personas you are allowed to call can be asked. Use it for work "
        "that is theirs, such as asking the artist for a picture."
    )
    params_type = A2ACallParams
    not_run_note: ClassVar[str] = "Nothing was asked of another persona."

    def __init__(self, characters: CharacterLookup | None = None) -> None:
        super().__init__()
        self._characters = characters

    def _with_characters(self, call: A2ACallParams, ctx: ToolContext) -> dict[str, JsonValue]:
        """`call.input`, with the open story's characters it names added (#1808).

        Nothing is added without a story, without a lookup, when the caller sent the key
        itself, or when no character is named. A story that cannot be read sends the call
        as it was: the peer can still read the codex with its own tools.
        """
        given = call.input
        if self._characters is None or ctx.story_id is None or A2A_CHARACTERS_KEY in given:
            return given
        try:
            found = self._characters(ctx, [call.task, *_words(given)])
        except PlainRefusalError:
            logger.warning("a2a_call: the story's characters could not be read", exc_info=True)
            return given
        if not found:
            return given
        return {**given, A2A_CHARACTERS_KEY: cast(JsonValue, found)}

    async def execute(
        self,
        params: dict[str, Any] | ToolContext | None = None,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        start = time.perf_counter()
        tool = self.name
        if isinstance(params, ToolContext):
            ctx, raw = params, kwargs
        elif isinstance(context, ToolContext):
            ctx = context
            raw = {**params, **kwargs} if isinstance(params, dict) else kwargs
        else:
            return ToolResult(
                success=False,
                error="Tool execution requires a valid ToolContext",
                provenance=Provenance.primary(provider="local.builtin", model=tool),
            )
        try:
            call = A2ACallParams.model_validate(raw)
        except ValidationError as exc:
            return _refusal(
                describe_invalid_arguments(tool, exc, A2ACallParams, self.not_run_note),
                ctx,
                start,
                tool,
            )
        except Exception:
            logger.warning("a2a_call: its arguments could not be checked", exc_info=True)
            return _refusal(
                f"The arguments of the call could not be checked. {self.not_run_note}",
                ctx,
                start,
                tool,
            )

        agent = ctx.agent_delegate
        if agent is None:
            return _refusal("Another persona cannot be asked from here.", ctx, start, tool)
        transport = agent.a2a_transport
        if transport is None or agent.context.depth >= 1:
            return _refusal(
                "You cannot ask another persona from inside a task another persona gave "
                "you. Do the work yourself or say what you need.",
                ctx,
                start,
                tool,
            )
        persona = agent.persona_definition
        # Stored as clone ids, called by handle (clone-data-scopes §3.3).
        peers: tuple[str, ...] = peer_handles(persona.a2a_peers) if persona is not None else ()
        if call.agent not in peers:
            allowed = ", ".join(peers) if peers else "no one"
            return _refusal(
                f"You are not allowed to ask '{call.agent}'. You may ask: {allowed}.",
                ctx,
                start,
                tool,
            )
        remaining = agent.steps_remaining
        if remaining <= 0:
            return _refusal(
                "There are no steps left in this turn to give another persona.",
                ctx,
                start,
                tool,
            )

        metadata = {
            A2A_STEP_BUDGET_KEY: str(remaining),
            A2A_DEPTH_KEY: "1",
            A2A_CALLER_SESSION_KEY: ctx.session_id,
        }
        # The peer works for the caller's conversation, on the story it has open (#1555).
        if ctx.room_id is not None:
            metadata[A2A_ROOM_KEY] = ctx.room_id
        if ctx.story_id is not None:
            metadata[A2A_STORY_KEY] = ctx.story_id
        message = TaskMessage(
            task_id=f"a2a_{uuid.uuid4().hex[:12]}",
            session_id=ctx.session_id,
            input_data={"task": call.task, "input": self._with_characters(call, ctx)},
            sender_agent_id=ctx.agent_id,
            target_agent_id=call.agent,
            metadata=metadata,
        )
        try:
            result = await transport.send_task(call.agent, message)
        except TaskNotFoundError:
            return _refusal(f"'{call.agent}' is not available to ask right now.", ctx, start, tool)
        except MissingProvenanceError:
            logger.warning("a2a_call: %s answered without provenance", call.agent)
            return _refusal(
                f"'{call.agent}' could not do the task, and nothing it made can be used.",
                ctx,
                start,
                tool,
            )

        output = cast(dict[str, Any], unwrap_immutable(result.output_data))
        # P4: whatever the outcome, the steps the peer took are the caller's.
        agent.consume_steps(_reported_steps(output))

        elapsed = (time.perf_counter() - start) * 1000.0
        if result.status is not TaskStatus.COMPLETED:
            lead = (
                f"'{call.agent}' stopped before finishing"
                if result.status is TaskStatus.INPUT_REQUIRED
                else f"'{call.agent}' could not do the task"
            )
            if result.error:
                # A reason that already names the persona is a whole sentence about it; a lead
                # naming it again would read "'artist' could not do the task: artist stopped".
                error = (
                    result.error if _names(result.error, call.agent) else f"{lead}: {result.error}"
                )
            else:
                reason = (
                    "it needed a person's approval"
                    if result.status is TaskStatus.INPUT_REQUIRED
                    else "no reason was given"
                )
                error = f"{lead}: {reason}"
            return ToolResult(
                success=False,
                error=error,
                execution_time_ms=elapsed,
                isolation_level=ctx.isolation.level,
                provenance=result.provenance,
            )
        paths = _reported_paths(output)
        response = output.get("response")
        return ToolResult(
            success=True,
            output={
                "agent": call.agent,
                "response": response if isinstance(response, str) else "",
                "paths": list(paths),
                # Whether the peer may have written something it did not name; the room
                # counts that as a possible unnamed write, as it does for a helper.
                "unnamed_writes": output.get("unnamed_writes") is not False,
            },
            artifacts=tuple(paths),
            execution_time_ms=elapsed,
            isolation_level=ctx.isolation.level,
            provenance=result.provenance,
        )

    def run(self, params: A2ACallParams, context: ToolContext) -> Any:
        raise NotImplementedError("a2a_call runs through execute")
