"""Tests for UClone-X local agent persona grounding and raw provider suppression (#848, #871).

Enforces:
- System prompt must explicitly ground agent identity in UClone-X local runtime.
- Generic cloud foundation model self-introductions (Qwen, Llama, GPT, Claude, Gemini) are suppressed.
- Standard persona definitions (Champion, Scout, Critic) inherit UClone-X grounding and behavioral invariants.

Assertions are written against the **components** in `uclone_x.agent.prompts` and against
the tokens that carry meaning (tool names, suppressed model names), not against whole
sentences lifted out of the assembled prompt. That was #871's third complaint: sentence
literals made every wording change a test failure, so the prompt could not be iterated on.
A component can now be reworded freely; what these tests pin is that it is still *in* the
composed prompt and still names the things behaviour depends on.
"""

from __future__ import annotations

from uclone_x.agent.models import (
    DEFAULT_SYSTEM_PROMPT,
    AgentConfig,
)
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import default_prompt_for
from uclone_x.agent.prompts import (
    HONEST_REPORTING,
    IDENTITY_GROUNDING,
    STEERABILITY_POLICY,
    TASK_CAPABILITIES,
)


def test_default_system_prompt_is_the_three_components() -> None:
    """The default prompt is exactly the three components #871 decomposed it into.

    Killed by: src/uclone_x/agent/prompts.py :: compose_task_capabilities(tools, writes_permitted=writes_permitted),
    Becomes: HONEST_REPORTING,
    """
    assert DEFAULT_SYSTEM_PROMPT == "\n\n".join(
        (IDENTITY_GROUNDING, TASK_CAPABILITIES, STEERABILITY_POLICY)
    )


def test_default_system_prompt_grounds_uclone_x_identity() -> None:
    """The default system prompt must ground the agent in the UClone-X local-first runtime.

    Killed by: src/uclone_x/agent/prompts.py :: You are an autonomous AI assistant powered by the UClone-X local-first runtime.
    Becomes: You are a generic cloud chatbot assistant.
    """
    assert IDENTITY_GROUNDING in DEFAULT_SYSTEM_PROMPT
    assert "UClone-X local-first runtime" in IDENTITY_GROUNDING
    assert "locally within the user's workspace" in IDENTITY_GROUNDING


def test_identity_grounding_suppresses_foundation_model_self_introduction() -> None:
    """Raw provider greetings must be suppressed by name, not merely discouraged in general.

    Killed by: src/uclone_x/agent/prompts.py :: (such as Qwen, Llama, GPT, Claude, or Gemini)
    Becomes: (such as some other model)
    """
    for provider in ("Qwen", "Llama", "GPT", "Claude", "Gemini"):
        assert provider in IDENTITY_GROUNDING


def test_task_capabilities_names_the_tools_behaviour_depends_on() -> None:
    """Artifact reporting and image generation must route to the real tool names.

    A reworded instruction is fine; an instruction that no longer names `file_write`
    or `generate_image` routes the model nowhere. (It named `write_to_file` until #1424,
    which is no tool at all.)

    Killed by: src/uclone_x/agent/prompts.py :: via file_write into the workspace
    Becomes: into the workspace
    """
    assert TASK_CAPABILITIES in DEFAULT_SYSTEM_PROMPT
    assert "file_write" in TASK_CAPABILITIES
    assert "generate_image" in TASK_CAPABILITIES
    assert "locally and privately" in TASK_CAPABILITIES


def test_personas_inherit_uclone_x_grounding_and_invariants() -> None:
    """Every built-in persona embeds the default components.

    They embed them because their files ask the loader for them, so this also pins that
    `append_default_prompt` is honoured -- a file whose key stopped being read would ship
    a persona with its preamble and nothing else. The tool guidance in between depends
    on each persona's tools (#1424), so what is pinned is the derivation for that persona.

    Killed by: src/uclone_x/agent/persona_store.py :: return _with_default_prompt(persona) if append_default else persona
    Becomes: return persona
    """
    registry = PersonaRegistry()
    for name in ("clone", "writer", "artist", "guardian", "pioneer", "scout"):
        persona = registry.get_persona(name)
        assert persona is not None, name
        prompt = persona.system_prompt
        assert "UClone-X" in prompt
        assert IDENTITY_GROUNDING in prompt
        assert HONEST_REPORTING in prompt
        assert STEERABILITY_POLICY in prompt
        assert prompt.endswith(f"\n\n{default_prompt_for(persona)}")


def test_agent_config_default_prompt_uses_uclone_x_grounded_prompt() -> None:
    """AgentConfig initialized with default arguments inherits the UClone-X grounded system prompt."""
    config = AgentConfig(agent_id="test-agent", name="Test Agent")
    assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert "UClone-X local-first runtime" in config.system_prompt
