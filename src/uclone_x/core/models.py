"""The persona and plan models a session record carries, below `agent/` (#1734).

`PersonaDefinition` (with the `AgentLLMConfig` and `ModelTier` it holds, and the base tool
names its `granted_tools` adds) and `PlanState`/`PlanStep` were defined in
`agent/models.py`. `SessionState` carries both -- a persona through
`AnchorProvenance.persona`, a plan through `SessionState.plan` -- so they moved here with
it, which is what lets `core/session_store.py` name `SessionState` without importing the
single-agent package. `agent/models.py` re-exports every name below under its old spelling;
each is the one class object, not a copy.

Not to be confused with the `PlanStep` in `agent/planner.py`, a different model.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.llm.models import TokenBudget

__all__ = [
    "BASE_MEMORY_TOOLS",
    "BASE_PERSONA_TOOLS",
    "RETIRED_MODEL_TIERS",
    "AgentLLMConfig",
    "ModelTier",
    "PersonaDefinition",
    "PlanState",
    "PlanStep",
]


class ModelTier(StrEnum):
    """Which of the two global models an agent's turns run on by default.

    ``inherit`` follows the deep model in Settings, as every shipped persona does. ``fast``
    was already written by the persona editor and is kept readable. The other values once
    offered (``pro``, ``flash_lite``, ``custom``) were read by nothing: a persona file that
    still names one loads as ``inherit`` (see :data:`RETIRED_MODEL_TIERS`). The models
    themselves are named by ``AgentLLMConfig.model_name`` and ``fast_model``.
    """

    INHERIT = "inherit"
    FAST = "fast"


#: Tier names persona files may still carry from before :class:`ModelTier` was cut to
#: two. Nothing ever read them, so a file naming one loads as ``inherit`` rather than
#: failing to load.
RETIRED_MODEL_TIERS: frozenset[str] = frozenset({"pro", "flash_lite", "custom"})


class AgentLLMConfig(BaseModel):
    """Per-agent LLM parameter and token budget configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model_tier: ModelTier = ModelTier.INHERIT
    #: The agent's own deep model -- the one its turns run on. ``None`` follows Settings.
    model_name: str | None = None
    #: The agent's own fast model, for its auxiliary calls (routing, summaries,
    #: compaction). ``None`` follows the fast model in Settings, which itself follows the
    #: deep model when left empty.
    fast_model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    token_budget: TokenBudget | None = None
    auto_compact: bool = True
    compaction_threshold_tokens: int = 60_000
    context_limit: int | None = Field(
        default=None,
        description="Explicit maximum context window tokens for this agent's model. "
        "When provided or resolved from model_name, auto-compaction triggers at 70% of this limit.",
    )


#: The tools every persona is given, whatever its own `allowed_tools` lists (#1402). Owner
#: ruling, 2026-09-23: an agent is given the tools it basically needs by default. Before
#: this only `clone` -- the one built-in with no list, and so no restriction -- could save
#: a memory; every other persona's call to `record_memory_fact` was refused.
#:
#: The rule for membership: a tool belongs here only if it has no effect outside the
#: agent's own memory. So the three memory tools (bound to the agent's own store, see
#: `AGENT_BOUND_TOOL_TYPES`) and the three read-only workspace tools, which
#: `writes_files = False` declares and the workspace boundary confines, plus
#: `tool_result_read` (#1422), which reads back this conversation's own shortened tool
#: results and nothing else -- without it, an excerpt would name a reader the persona
#: cannot call. Anything that writes a file, runs a command, installs, reaches the
#: network or makes an image stays in the persona's own list.
#:
#: Names, not classes: this module sits below the tool packages. A test holds each name to
#: the class that registers under it, so a rename cannot leave a dead entry here.
#:
#: Naming a tool here grants permission; it does not register anything. An agent composed
#: with no memory store still has no memory tools, and is not offered one it cannot run.
#:
#: The memory part is named on its own, `BASE_MEMORY_TOOLS`, because a sub-agent is not
#: given it (#1431): `spawn_subagent` removes these names from the list a child inherits.
BASE_MEMORY_TOOLS: tuple[str, ...] = (
    "record_memory_fact",
    "query_memory_facts",
    "retract_memory_fact",
)

BASE_PERSONA_TOOLS: tuple[str, ...] = (
    *BASE_MEMORY_TOOLS,
    "file_read",
    "file_search",
    "directory_list",
    "tool_result_read",
    "load_skill",
)


class PersonaDefinition(BaseModel):
    """Configuration schema for dynamically defined sub-agent personas."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(description="Unique name identifier for the dynamic agent persona")
    role: str = Field(description="Human-readable role (e.g. 'Security Reviewer')")
    description: str = Field(default="", description="Description of what this persona does")
    system_prompt: str = Field(description="System instructions and persona constraints")
    allowed_tools: tuple[str, ...] = Field(
        default_factory=tuple, description="Authorized tool names"
    )
    llm_config: AgentLLMConfig = Field(
        default_factory=AgentLLMConfig, description="Dedicated LLM parameters"
    )
    enable_write_tools: bool = Field(
        default=False, description="Whether persona has write permissions"
    )
    enable_subagent_tools: bool = Field(
        default=False, description="Controls recursive sub-agent creation"
    )
    a2a_peers: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Persona names this persona may call through the `a2a_call` tool. Empty means"
            " it may call no one (#1558)."
        ),
    )

    @property
    def granted_tools(self) -> tuple[str, ...]:
        """The tools this persona may use: its own `allowed_tools` plus `BASE_PERSONA_TOOLS`.

        This, not `allowed_tools`, is what every site that scopes an agent to a persona
        reads. `allowed_tools` stays the persona's own list -- what its file says and what
        its editor shows and saves -- so the base set is never written into a persona.

        An empty `allowed_tools` is no restriction at all (every registered tool), and
        stays empty here: adding the base set to it would turn "everything" into "only
        the base set".

        The persona's own order is kept and the base names follow, each once.
        """
        if not self.allowed_tools:
            return ()
        own = self.allowed_tools
        return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)


class PlanStep(BaseModel):
    """A discrete step within an interactive agent execution plan."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    index: int = Field(description="Sequential index of the plan step")
    description: str = Field(description="Description of the action/task for this step")
    completed: bool = Field(default=False, description="Whether this step has been completed")
    verification: str | None = Field(
        default=None, description="Optional verification criteria or outcome"
    )


class PlanState(BaseModel):
    """Interactive chat plan state tracking step progress."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    plan_id: str = Field(
        default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}",
        description="Unique plan identifier",
    )
    title: str = Field(description="Title or goal of the plan")
    steps: tuple[PlanStep, ...] = Field(
        default_factory=tuple, description="Ordered tuple of plan steps"
    )
    status: Literal["proposed", "in_progress", "completed", "rejected"] = Field(
        default="proposed", description="Overall lifecycle status of the plan"
    )
