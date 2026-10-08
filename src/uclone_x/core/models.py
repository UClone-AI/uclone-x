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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from uclone_x.llm.models import TokenBudget

__all__ = [
    "BASE_MEMORY_TOOLS",
    "BASE_PERSONA_TOOLS",
    "BASE_SELF_TOOLS",
    "RETIRED_MODEL_TIERS",
    "AgentLLMConfig",
    "ModelTier",
    "PersonaDefinition",
    "PlanState",
    "PlanStep",
    "effective_tool_scope",
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
    #: The agent's own deep model -- the one its turns run on. ``None`` follows the system
    #: default. In a persona it is a model ref, ``<connection id>/<model id>``
    #: (model-gateway §3.4); on a built agent it is the bare id its connector is sent.
    model_name: str | None = None
    #: The agent's own fast model, for the compaction and summaries of its *own* history.
    #: ``None`` follows the default fast model, which itself follows the deep one when
    #: empty. Room routing and room summaries never use it: they always run on the default
    #: fast model (model-gateway §3.4, decision 5).
    fast_model: str | None = None
    #: The agent's own picture model: a ref, or ``auto``. ``None`` follows the default.
    #: Stored only; pictures read it from model-gateway step 5.
    image_model: str | None = None
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
#: agent's own state. So the three memory tools (bound to the agent's own store, see
#: `AGENT_BOUND_TOOL_TYPES`), `set_avatar` (#2160), which changes only the calling clone's
#: own picture, and the three read-only workspace tools, which
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

#: The clone's own state: its memory and its picture (#2160). `set_avatar` is here because a
#: clone whose list was written before the tool existed -- an imported, edited copy of a
#: built-in -- could otherwise never change its face, and the avatar skill, which requires
#: it, was hidden from it. Like the memory tools it is not given to a sub-agent or to a peer
#: answering an `a2a_call`: neither is the clone the user is talking to, and a sub-agent has
#: no persona of its own.
BASE_SELF_TOOLS: tuple[str, ...] = (*BASE_MEMORY_TOOLS, "set_avatar")

BASE_PERSONA_TOOLS: tuple[str, ...] = (
    *BASE_SELF_TOOLS,
    "file_read",
    "file_search",
    "directory_list",
    "tool_result_read",
    "load_skill",
)


#: Tools an agent composes from its own state when it is built, so they are in no shared
#: registry when a persona file is loaded and checked. A persona may still name them in
#: `allowed_tools`: `show_self` draws from the clone's own self facts (#2017), and an agent
#: with no memory or no image tool simply does not have it. Names, not classes, for the
#: reason `BASE_PERSONA_TOOLS` gives.
AGENT_COMPOSED_TOOLS: frozenset[str] = frozenset({"show_self"})


def persona_model_problem(
    model_name: str | None, fast_model: str | None, image_model: str | None
) -> str | None:
    """Why a persona's model fields cannot be saved, as one plain sentence; ``None`` if they can.

    Each must be empty or a ref, ``<connection id>/<model id>`` (split at the first ``/``);
    the picture model may also be ``auto``. The same rule as ``uclone_x.llm.connections.
    ModelRef``, stated here because the kernel does not import the gateway.
    """
    for label, value in (
        ("model_name", model_name),
        ("fast_model", fast_model),
        ("image_model", image_model),
    ):
        if value is None:
            continue
        clean = value.strip()
        if label == "image_model" and clean == "auto":
            continue
        head, sep, tail = clean.partition("/")
        if not (sep and head.strip() and tail.strip()):
            return (
                f"{label}: the model {clean!r} does not say which connection it is on. "
                f"Name the connection first, for example gemini/{clean or 'model-name'}."
            )
    return None


class PersonaDefinition(BaseModel):
    """Configuration schema for dynamically defined sub-agent personas."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(description="Unique name identifier for the dynamic agent persona")
    display_name: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "What a person reads as the clone's name, per locale, e.g. {'en': 'Sleepyhead'}."
            " Free text; `name` stays the ASCII handle. Empty means label it by `name`."
        ),
    )
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
    tools_module: str | None = Field(
        default=None,
        description=(
            "How this clone's requests offer its tools (#2188): `native`, `pinned` or"
            " `bound`. Unset follows the provider's default, which is `native` everywhere."
            " Applied when a seat is built; a name the running version does not have is"
            " refused then, in plain words."
        ),
    )

    @model_serializer(mode="wrap")
    def _omit_unset_tools_module(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Leave `tools_module` out when unset, so a persona without one -- every persona
        before #2188, and every session record that carries one -- is written as before
        (#1844)."""
        data: dict[str, object] = handler(self)
        if data.get("tools_module") is None:
            data.pop("tools_module", None)
        return data

    @model_validator(mode="after")
    def _models_name_their_connection(self) -> PersonaDefinition:
        """Refuse a model that does not say which connection it is on (model-gateway §3.4).

        A persona written for one provider names a bare id (``gemini-3.8-pro``); run on the
        gateway it would go to whichever connection came first. It is refused when the
        persona loads, naming the field and how to write it.
        """
        refusal = persona_model_problem(
            self.llm_config.model_name, self.llm_config.fast_model, self.llm_config.image_model
        )
        if refusal is not None:
            raise ValueError(refusal)
        return self

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


def effective_tool_scope(
    operator_tools: tuple[str, ...], persona: PersonaDefinition | None
) -> tuple[str, ...]:
    """The tools an agent may run: the operator's list if it gave one, else the persona's.

    The one rule for an agent's tool scope, read by the agent itself
    (`BaseAgent._apply_persona_tool_scope`) and by what reports that scope to a person
    (`/api/skills` `hidden_from`, #1865), so the two cannot disagree. An empty result is no
    restriction: every registered tool.
    """
    if operator_tools:
        return operator_tools
    if persona is not None:
        return persona.granted_tools
    return ()


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
