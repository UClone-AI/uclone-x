"""Session bootstrapping, and the one place a persona becomes an `AgentConfig`."""

from pathlib import Path

from uclone_x.agent.models import AgentConfig, AgentLLMConfig, PersonaDefinition
from uclone_x.agent.session import SessionStore


def agent_config_for_persona(
    persona: PersonaDefinition,
    *,
    agent_id: str | None = None,
    name: str | None = None,
    system_prompt: str | None = None,
    seat_framing: str = "",
    llm_config: AgentLLMConfig | None = None,
    workspace_dir: str | Path | None = None,
    read_roots: tuple[Path, ...] = (),
) -> AgentConfig:
    """Build the `AgentConfig` for an agent that takes on `persona`.

    Every head that seats a persona builds its config here -- the chat session manager and
    the room resolver -- so the same clone is given its tools by the same rule in a 1:1
    chat and in a room (#1448). Each passed what only it knows: the id and display name
    it seats the agent under, a fallback system prompt, the room's seat framing (the
    agent puts it ahead of the persona's prompt, `compose_identity_prompt`), and an
    installation-wide model override. Anything not passed
    is the persona's own.

    `allowed_tools` is left empty on purpose. That field is the *operator's* list, and the
    agent lets an operator's list win over any persona's (`BaseAgent`,
    `_apply_persona_tool_scope`), so copying the persona's tools into it froze them for
    the agent's life: an edit handed to the agent through `define_persona` could never
    reach them. Left empty, the agent resolves the persona's own list, base set included
    (`granted_tools`, #1402), and resolves it again when an edited persona is defined on
    it (#892).
    """
    return AgentConfig(
        agent_id=agent_id if agent_id is not None else persona.name,
        name=name if name is not None else persona.name,
        role=persona.role,
        description=persona.description,
        system_prompt=system_prompt if system_prompt is not None else persona.system_prompt,
        seat_framing=seat_framing,
        allowed_tools=(),  # the persona's list, resolved by the agent
        llm_config=llm_config if llm_config is not None else persona.llm_config,
        enable_write_tools=persona.enable_write_tools,
        enable_subagent_tools=persona.enable_subagent_tools,
        persona=persona.name,
        workspace_dir=workspace_dir,
        read_roots=read_roots,
    )


def bootstrap_session(store: SessionStore, session_id: str, config: AgentConfig) -> AgentConfig:
    """Seed an empty session with `config`'s system prompt, under `config`'s own id."""
    state = store.load(session_id)
    if state is None:
        from uclone_x.agent.session import SessionState
        from uclone_x.llm.models import ChatMessage, MessageRole

        store.save(
            SessionState(
                session_id=session_id,
                agent_id=config.agent_id,
                messages=(ChatMessage(role=MessageRole.SYSTEM, content=config.system_prompt),),
            )
        )
    return config
