"""`agent/prompt_assembler.py` on its own: identity, turn context and the request record (#1736).

The code moved out of `agent/base.py` unchanged, and the behavioural tests that drive it
through `BaseAgent` stay where they were (`test_agent_base.py`, `test_context_snapshot.py`,
`test_read_roots.py`, ...). This file pins the module directly, and pins the one property
the move itself introduced: the assembler holds no copy of agent state, so a value swapped
after construction -- as `evals/harness_ladder/runner.py` swaps `agent._tools` -- is the
value it reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    PersonaDefinition,
    PlanState,
    PlanStep,
)
from uclone_x.agent.prompt_assembler import (
    TURN_CONTEXT_HEADER,
    AnchorWriter,
    PromptAssembler,
    PromptScope,
    compose_identity_prompt,
    persisted_anchor_provenance,
    restored_anchor_provenance,
    turn_context_block,
    undone_attempt_section,
)
from uclone_x.agent.request_record import LogReader
from uclone_x.agent.session import ContextSnapshot
from uclone_x.core.context_state import ContextEntry, ContextEpoch, ContextForm, advance
from uclone_x.core.session_log import SessionLogProvenance, logged_message, new_entry
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ToolCallRequest
from uclone_x.tools.builtin.filesystem import FileReadTool
from uclone_x.tools.protocols import ToolRegistryProtocol
from uclone_x.tools.registry import ToolRegistry


@dataclass
class _Session:
    stored_bodies: set[str] = field(default_factory=set[str])
    pending_bodies: dict[str, str] = field(default_factory=dict[str, str])
    context_snapshots: list[ContextSnapshot] = field(default_factory=list[ContextSnapshot])
    last_conversation: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    last_request: int | None = None
    history: list[ChatMessage] = field(default_factory=list[ChatMessage])
    context_epochs: tuple[ContextEpoch, ...] = ()
    recalled_memory: str | None = None
    compacted_entries: dict[str, ContextEntry] = field(default_factory=dict[str, ContextEntry])

    def reading(self, history: list[ChatMessage]) -> _Session:
        """This session, answering for `history` (the scope reads both live)."""
        self.history = history
        return self

    def logged_history(self) -> list[tuple[str, ChatMessage]]:
        return [(f"e{position}", message) for position, message in enumerate(self.history)]

    def declare_new_epoch(self, cause: str) -> None:
        del cause  # these tests open no epoch by declaration

    def log_reader(self) -> LogReader:
        """A reader over a log with one entry per history message, as `logged_history`."""
        rendered = [logged_message(message) for message in self.history]
        bodies = {item.digest: item.body for item in rendered}
        log = [
            new_entry(position, item, turn=1, provenance=SessionLogProvenance.RECORDED)
            for position, item in enumerate(rendered)
        ]
        return LogReader(log, bodies.get, purpose="send")

    def record_shown(self, shown: list[ContextEntry], *, step: int) -> ContextEpoch:
        self.context_epochs = advance(self.context_epochs, shown, turn=1, step=step)
        return self.context_epochs[-1]


class _FileToolRegistry:
    """A registry holding only `file_read`, which is all the workspace section asks about."""

    def get(self, name: str) -> object | None:
        return object() if name == "file_read" else None


@dataclass
class _State:
    """Mutable agent-shaped state the scope reads through, so a test can swap it."""

    config: AgentConfig
    tools: ToolRegistryProtocol | None = None
    workspace: Path | None = None
    plan: PlanState | None = None
    history: list[ChatMessage] = field(default_factory=list[ChatMessage])
    session: _Session = field(default_factory=_Session)
    effective_prompt: str = "EFFECTIVE"


def _assembler(state: _State) -> PromptAssembler:
    return PromptAssembler(
        PromptScope(
            ontology=lambda: None,
            skills=lambda: None,
            config=lambda: state.config,
            tools=lambda: state.tools,
            memory=lambda: None,
            workspace_root=lambda: state.workspace,
            current_plan=lambda: state.plan,
            history=lambda: state.history,
            active_session=lambda: state.session.reading(state.history),
            log_reader=lambda: state.session.reading(state.history).log_reader(),
            turn_counter=lambda: 1,
            anchor_is_stale=lambda: False,
            system_prompt_base=lambda: "BASE",
            effective_system_prompt=lambda: state.effective_prompt,
        )
    )


def _config(**overrides: Any) -> AgentConfig:
    return AgentConfig(agent_id="assembler", name="Assembler", **overrides)


def test_seat_framing_puts_the_persona_instructions_under_a_labelled_header() -> None:
    """Framing plus a persona with instructions: framing first, then the labelled persona.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: if persona is not None and persona.system_prompt:
    Becomes: if False:
    """
    persona = PersonaDefinition(name="scout", role="Scout", system_prompt="You scout.")

    prompt = compose_identity_prompt(
        config_prompt="CONFIG", persona=persona, seat_framing="You sit in room R."
    )

    assert prompt == "You sit in room R.\n\n[Persona Instructions: Scout]\nYou scout."


def test_the_turn_context_block_opens_with_its_header_and_is_empty_without_sections() -> None:
    r"""The block says what it is, and is absent when there is nothing to say.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: return "\n\n".join((TURN_CONTEXT_HEADER, *sections))
    Becomes: return "\n\n".join(sections)
    """
    assert turn_context_block([]) == ""
    assert turn_context_block(["[Plan] a", "[Memory] b"]) == (
        f"{TURN_CONTEXT_HEADER}\n\n[Plan] a\n\n[Memory] b"
    )


def test_an_unrecorded_stamp_round_trips_as_unrecorded_never_as_no_persona() -> None:
    """A record with no stamp stays a gap; it is not read as "composed under no persona"."""
    assert persisted_anchor_provenance(AnchorWriter.UNRECORDED) is None
    assert restored_anchor_provenance(None) is AnchorWriter.UNRECORDED


def test_the_assembler_reads_a_registry_swapped_in_after_construction(tmp_path: Path) -> None:
    """The scope is read on every call, so a later `_tools` swap is what the section sees.

    Built with no registry, the workspace section is withheld: nothing in the allowed list
    is a file tool that exists. Swapping a registry that holds `file_read` in afterwards
    must bring the section back without rebuilding the assembler.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: return self._scope.tools()
    Becomes: return None
    """
    state = _State(config=_config(allowed_tools=("file_read",)), workspace=tmp_path)
    assembler = _assembler(state)
    assert assembler.get_workspace_prompt_section() is None

    state.tools = cast(ToolRegistryProtocol, _FileToolRegistry())

    section = assembler.get_workspace_prompt_section()
    assert section is not None
    assert str(tmp_path) in section


def test_with_no_anchor_the_turn_sends_the_effective_prompt_and_the_plan_rides_at_the_tail() -> (
    None
):
    """No `SYSTEM` turn in history: identity is the effective prompt; the plan is turn context.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: turn_context = turn_context_block(turn_sections)
    Becomes: turn_context = ""
    """
    plan = PlanState(title="Ship it", steps=(PlanStep(index=1, description="write"),))
    state = _State(
        config=_config(),
        plan=plan,
        history=[ChatMessage(role=MessageRole.USER, content="go")],
    )

    layers = _assembler(state).prepare_turn_layers(extra_sections=("[Extra] x",))

    assert layers.identity == "EFFECTIVE"
    assert layers.slow_context == ""
    assert layers.conversation == (ChatMessage(role=MessageRole.USER, content="go"),)
    assert layers.turn_context.startswith(TURN_CONTEXT_HEADER)
    assert "[Extra] x" in layers.turn_context
    assert "### Active Execution Plan: Ship it" in layers.turn_context


def test_the_forms_a_compaction_derived_are_the_forms_the_next_request_shows() -> None:
    """What a compaction derived (#1848) is what the next request shows and records.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: logged, opening_entries(session.context_epochs, session.compacted_entries)
    Becomes: logged, opening_entries(session.context_epochs, {})
    """
    ask = ChatMessage(role=MessageRole.USER, content="go")
    full = ChatMessage(role=MessageRole.TOOL, content="the result", name="t", tool_call_id="c1")
    state = _State(config=_config(), history=[ask, full])
    # The compaction derived that body e1 shows entry e0's content through a rendering.
    state.session.compacted_entries = {
        "e1": ContextEntry(entry="e0", form=ContextForm.FULL, rendering="e1")
    }

    shown = _assembler(state).prepare_turn_layers().shown
    assert [(s.entry, s.form, s.rendering) for s in shown] == [
        ("e0", ContextForm.FULL, None),
        ("e0", ContextForm.FULL, "e1"),
    ]


def test_request_context_fields_chain_each_request_to_the_one_before() -> None:
    """The second request names the first as its base and records only what it added.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: session.last_request = request_number
    Becomes: pass
    """
    state = _State(config=_config(), history=[ChatMessage(role=MessageRole.USER, content="a")])
    assembler = _assembler(state)

    def fields() -> dict[str, Any]:
        layers = assembler.prepare_turn_layers()
        messages: Sequence[ChatMessage] = layers.conversation
        return assembler.request_context_fields(1, LLMRequest(messages=tuple(messages)), layers)

    first = fields()
    state.history.append(ChatMessage(role=MessageRole.ASSISTANT, content="b"))
    second = fields()

    assert (first["request"], first["base_request"]) == (1, None)
    assert (second["request"], second["base_request"]) == (2, 1)
    assert second["kept_entry_count"] == 1
    # What it added is named by its log entry and form, not by its text (#2013).
    assert [e["entry"] for e in second["appended_entries"]] == ["e1"]
    assert "b" not in {str(v) for e in second["appended_entries"] for v in e.values()}
    assert len(state.session.context_snapshots) == 1


def test_an_agent_whose_tools_are_swapped_after_construction_is_assembled_from_the_new_ones(
    tmp_path: Path,
) -> None:
    """The agent hands the assembler a live read of `_tools`, not the registry it was built with.

    `evals/harness_ladder/runner.py` assigns `agent._tools` after construction. A scope
    that captured the registry at construction would keep describing the old one.

    Killed by: src/uclone_x/agent/base.py :: tools=lambda: self._tools,
    Becomes: tools=lambda _t=self._tools: _t,
    """
    agent = BaseAgent(
        config=_config(allowed_tools=("file_read",)),
        tools=ToolRegistry(),
        context=AgentContext(session_id="s", agent_id="assembler", workspace_root=tmp_path),
    )
    assert agent._get_workspace_prompt_section() is None  # pyright: ignore[reportPrivateUsage]

    agent._tools = ToolRegistry(tools=[FileReadTool()])  # pyright: ignore[reportPrivateUsage]

    section = agent._get_workspace_prompt_section()  # pyright: ignore[reportPrivateUsage]
    assert section is not None and section.startswith("[Workspace]")


def test_undone_attempt_section_redacts_credentials_in_tool_arguments() -> None:
    """Tool arguments in [Undone Attempt] statement have credential patterns redacted (#1503).

    Killed by: src/uclone_x/agent/prompt_assembler.py :: return redact_credentials(f"{call.name}({_clip(', '.join(parts), _UNDONE_ARGS_CHARS)})")
    Becomes: return f"{call.name}({_clip(', '.join(parts), _UNDONE_ARGS_CHARS)})"
    """
    call = ToolCallRequest(
        id="c1",
        name="fetch_secret",
        arguments={"token": "ghp_123456789012345678901234567890123456"},
    )
    summary = undone_attempt_section([call])
    assert "[Undone Attempt]" in summary
    assert "[REDACTED]" in summary
    assert "ghp_" not in summary
