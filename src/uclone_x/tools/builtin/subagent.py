import logging
import time
from collections.abc import Mapping
from typing import Any, ClassVar, cast

from pydantic import BaseModel, Field, model_validator

from uclone_x.core.provenance import Provenance, require_provenance
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class SubagentDelegationParams(BaseModel):
    role: str = Field(description="Role of the subagent, e.g., 'researcher' or 'analyst'")
    goal: str = Field(description="Target objective for the subagent")
    prompt: str = Field(description="Instruction or query to execute")
    max_steps: int | None = Field(
        default=None, description="Max agent steps (tool rounds) for the subagent to run"
    )
    max_turns: int | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
        description="[Deprecated alias for max_steps; use max_steps instead] Max steps for the subagent",
    )
    system_prompt: str | None = Field(default=None, description="System prompt for the subagent")
    share_parent_memory: bool = Field(
        default=False,
        description=(
            "Let the subagent read the facts you have saved in memory, with "
            "query_memory_facts. It cannot save or remove facts. Off by default: a subagent "
            "starts with no memory. Turn it on only when the task needs something you "
            "remember."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_step_budget(cls, data: Any) -> Any:
        """Accept either spelling from a model that learned the tool schema at either name."""
        if isinstance(data, Mapping):
            raw = dict(cast(Mapping[str, Any], data))
            steps = raw.get("max_steps")
            turns = raw.get("max_turns")
            if steps is not None and turns is not None:
                if steps != turns:  # SubagentDelegationParams budget conflict check
                    raise ValueError(
                        f"Conflicting values for max_steps and deprecated alias max_turns: {steps} != {turns}"
                    )
            elif turns is not None and steps is None:
                raw["max_steps"] = turns
            elif steps is not None and turns is None:
                raw["max_turns"] = steps
            return raw
        if hasattr(data, "max_steps") and hasattr(data, "max_turns"):
            steps = data.max_steps
            turns = data.max_turns
            if steps is not None and turns is not None and steps != turns:
                raise ValueError(
                    f"Conflicting values for max_steps and deprecated alias max_turns: {steps} != {turns}"
                )
        return data


def _extract_child_steps(subagent: Any) -> int:
    """Extract steps taken by a subagent from run_steps or turn_counter."""
    steps = 0
    for attr in ("run_steps", "turn_counter", "_turn_counter"):
        val = getattr(subagent, attr, None)
        if isinstance(val, int) and val > steps:
            steps = val
    return steps


def _deduct_child_steps(agent: Any, child_steps: int) -> None:
    """Charge child steps against parent run budget per P4."""
    if hasattr(agent, "consume_steps") and callable(agent.consume_steps):
        agent.consume_steps(child_steps)
    elif hasattr(agent, "_run_steps"):
        agent._run_steps += child_steps


class SubagentDelegationTool(BaseTool[SubagentDelegationParams]):
    name = "delegate_subagent"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    spawns_subagents: ClassVar[bool] = True  # refused by enable_subagent_tools (#1167)
    description = (
        "Allows delegating a specific task or sub-goal to an ephemeral, context-isolated subagent."
    )
    params_type = SubagentDelegationParams

    async def execute(
        self,
        params: dict[str, Any] | ToolContext | None = None,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        start_time = time.perf_counter()
        tool_identifier = self.name or self.__class__.__name__

        # Resolve context and params
        actual_context: ToolContext
        actual_params_dict: dict[str, Any]

        if isinstance(params, ToolContext):
            actual_context = params
            actual_params_dict = kwargs
        elif isinstance(context, ToolContext):
            actual_context = context
            actual_params_dict = {**params, **kwargs} if isinstance(params, dict) else kwargs
        elif "context" in kwargs and isinstance(kwargs["context"], ToolContext):
            actual_context = kwargs.pop("context")
            actual_params_dict = {**params, **kwargs} if isinstance(params, dict) else kwargs
        else:
            return ToolResult(
                success=False,
                error="Tool execution requires a valid ToolContext",
                execution_time_ms=0.0,
                isolation_level=None,
                provenance=Provenance.primary(provider="local.builtin", model=tool_identifier),
            )

        try:
            validated_params = SubagentDelegationParams.model_validate(actual_params_dict)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Parameter validation failed for tool '{tool_identifier}': {e}",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(provider="local.builtin", model=tool_identifier),
            )

        agent = actual_context.agent_delegate
        if agent is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error="agent_delegate not available in ToolContext",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(provider="local.builtin", model=tool_identifier),
            )

        # Enforce recursion depth cap
        max_depth = 3
        if agent.context.depth >= max_depth:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Recursion depth exceeded: depth {agent.context.depth} >= cap {max_depth}",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(provider="local.builtin", model=tool_identifier),
            )

        # Inherit what is left of the parent's *step* budget for the current run.
        #
        # This was `agent.config.max_turns - agent.turn_counter`, which subtracted the
        # lifetime interaction-turn counter from the per-run step ceiling — two unrelated
        # quantities. `turn_counter` only ever grows, so an ordinary conversation reaching
        # `max_steps` messages made every later delegation fail with "exhausted", while a
        # fresh session handed a subagent the parent's whole ceiling no matter how many
        # steps the parent had already spent in that run. `steps_remaining` is the
        # quantity the P4 ceiling is actually measured against.
        parent_remaining_steps = agent.steps_remaining
        if parent_remaining_steps <= 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error="Parent agent has exhausted its step budget",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(provider="local.builtin", model=tool_identifier),
            )

        sub_max_steps = validated_params.max_steps
        if sub_max_steps is None or sub_max_steps > parent_remaining_steps:
            sub_max_steps = parent_remaining_steps

        memory_note: str | None = None
        if validated_params.share_parent_memory:
            refusal = agent.memory_share_refusal()
            memory_note = (
                "The subagent could read your memory, but not change it."
                if refusal is None
                else f"The subagent was not given your memory: {refusal}."
            )
        subagent = await agent.spawn_subagent(
            role=validated_params.role,
            goal=validated_params.goal,
            system_prompt=validated_params.system_prompt,
            share_parent_memory=validated_params.share_parent_memory,
        )

        # Override the subagent's step ceiling. `model_copy` does not run validators, so
        # the deprecated alias is written alongside the canonical field rather than left
        # holding the spawn-time default.
        subagent._config = subagent._config.model_copy(
            update={"max_steps": sub_max_steps, "max_turns": sub_max_steps}
        )

        try:
            result = await agent.delegate_task(subagent, validated_params.prompt)
            # The parent's `turn_counter` is not charged for the subagent's work. It used
            # to be (`agent._turn_counter += subagent.turn_counter`), which inflated the
            # count of *human interactions* with a child's internal execution and so made
            # the session look saturated to every surface reading that counter.
            # Per P4 (Bounded Agency), steps spent by the child are charged against the
            # parent's run budget rather than pooled per child, so siblings cannot jointly
            # exceed the session ceiling.
            child_steps = _extract_child_steps(subagent)
            if child_steps > 0 and getattr(subagent, "_steps_deducted", None) is not True:
                _deduct_child_steps(agent, child_steps)
                subagent._steps_deducted = True

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if result.error:
                return ToolResult(
                    success=False,
                    error=f"Subagent failed: {result.error}"
                    + (f" {memory_note}" if memory_note else ""),
                    execution_time_ms=elapsed_ms,
                    isolation_level=actual_context.isolation.level,
                    provenance=require_provenance(result.provenance, "SubagentDelegationTool"),
                )
            output: dict[str, Any] = {
                "subagent_id": subagent.agent_id,
                "response": result.content,
            }
            if memory_note is not None:
                output["parent_memory"] = memory_note
            return ToolResult(
                success=True,
                output=output,
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=require_provenance(result.provenance, "SubagentDelegationTool"),
            )
        except Exception as e:
            child_steps = _extract_child_steps(subagent)
            if child_steps > 0 and getattr(subagent, "_steps_deducted", None) is not True:
                _deduct_child_steps(agent, child_steps)
                subagent._steps_deducted = True

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Subagent execution failed: {e}",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(provider="local.builtin", model=tool_identifier),
            )

    def run(self, params: SubagentDelegationParams, context: ToolContext) -> Any:
        pass
