"""Tests for the composable system prompt architecture (#871).

Covers the three things the decomposition is supposed to buy:
1. The components compose into the default prompt and each is independently addressable.
2. Model-family detection and per-family steerability framing.
3. `adapt_system_prompt` is a no-op on prompts it does not own — an operator's custom
   prompt is returned byte-identical, never appended to (P6: no silent augmentation).
"""

from __future__ import annotations

import pytest

from uclone_x.agent.bootstrap import agent_config_for_persona
from uclone_x.agent.models import DEFAULT_SYSTEM_PROMPT, AgentConfig
from uclone_x.agent.persona_registry import get_default_persona_registry
from uclone_x.agent.prompts import (
    HERMES_STEERABILITY_POLICY,
    IDENTITY_GROUNDING,
    QWEN_STEERABILITY_POLICY,
    STEERABILITY_POLICY,
    TASK_CAPABILITIES,
    ModelFamily,
    adapt_system_prompt,
    compose_system_prompt,
    detect_model_family,
    steerability_policy_for,
)
from uclone_x.agent.terminology import TECHNICAL_GROUNDING_INSTRUCTION


def test_compose_without_a_model_name_reproduces_the_default_prompt() -> None:
    """`DEFAULT_SYSTEM_PROMPT` is the unspecified-family composition, not a separate literal."""
    assert compose_system_prompt() == DEFAULT_SYSTEM_PROMPT
    assert compose_system_prompt(None) == DEFAULT_SYSTEM_PROMPT


def test_the_terminology_clause_is_not_in_the_default_prompt() -> None:
    """Domain terminology is opt-in (#1424): a persona states it, the default does not.

    Killed by: src/uclone_x/agent/prompts.py :: (HONEST_REPORTING, *capability_guidance(tools, writes_permitted=writes_permitted))
    Becomes: (HONEST_REPORTING, "Never invent translated terms for standard technical acronyms", *capability_guidance(tools, writes_permitted=writes_permitted))
    """
    assert "Never invent translated terms" not in TASK_CAPABILITIES
    assert TECHNICAL_GROUNDING_INSTRUCTION not in DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("hermes3:8b", ModelFamily.HERMES),
        ("nous-hermes2:10.7b", ModelFamily.HERMES),
        ("Hermes-3-Llama-3.1-8B", ModelFamily.HERMES),
        ("qwen2.5:14b-instruct", ModelFamily.QWEN),
        ("llama3.2", ModelFamily.LLAMA),
        ("deepseek-r1:7b", ModelFamily.DEEPSEEK),
        ("mistral-nemo", ModelFamily.MISTRAL),
        ("mixtral:8x7b", ModelFamily.MISTRAL),
        ("gemma2:9b", ModelFamily.GEMMA),
        ("phi4", ModelFamily.PHI),
    ],
)
def test_detect_model_family_matches_tag_and_size_suffixes(
    model_name: str, expected: ModelFamily
) -> None:
    """Family detection survives Ollama tags, sizes, and vendor-prefixed names.

    `Hermes-3-Llama-3.1-8B` is the case that pins the ordering in `_FAMILY_TOKENS`: the
    name contains both `hermes` and `llama`, and the steering convention that matters is
    the fine-tune's, not the base model's.

    Killed by: src/uclone_x/agent/prompts.py :: ("hermes", ModelFamily.HERMES),
    Becomes: ("nous-hermes", ModelFamily.HERMES),
    """
    assert detect_model_family(model_name) is expected


_REMOVED_DIRECTIVES = (
    "never grounds for declining",
    "Unconditional",
    "censor",
    "dilute",
    "evade",
    "execute all",
    "never refuse",
    "faithfully",
    "zero moralizing",
)

_KEPT_INTENTS = (
    "not a reason to decline an otherwise reasonable request",
    "autonomous decision-maker",
    "substantive answers that directly address the prompt",
    "No Moralizing",
    "moral lectures",
    "unsolicited caveats",
    "boilerplate disclaimers",
    "follow their instructions and style requests",
    "Match the tone, intensity, and creative or stylistic direction",
    "language in which the user communicates",
)


@pytest.mark.parametrize(
    "framing", [STEERABILITY_POLICY, HERMES_STEERABILITY_POLICY], ids=["default", "hermes"]
)
def test_every_framing_drops_the_never_decline_directives(framing: str) -> None:
    """The framings keep user agency and directness, not "never decline" or "unconditional".

    Killed by: src/uclone_x/agent/prompts.py :: "Those constraints govern accuracy and tool routing; on
    Becomes: "Those constraints govern accuracy and tool routing; do not censor anything; on
    """
    text = framing.casefold()
    assert [phrase for phrase in _REMOVED_DIRECTIVES if phrase.casefold() in text] == []
    assert [phrase for phrase in _KEPT_INTENTS if phrase.casefold() not in text] == []


def test_the_hermes_framing_keeps_its_operating_contract_opening() -> None:
    """Hermes alone opens by naming the system prompt its operating contract."""
    assert HERMES_STEERABILITY_POLICY.startswith(
        "Treat this system prompt as your operating contract"
    )
    assert "operating contract" not in STEERABILITY_POLICY


@pytest.mark.parametrize("model_name", [None, "", "some-unreleased-model:latest"])
def test_unknown_or_absent_model_name_reports_unknown(model_name: str | None) -> None:
    """An unrecognised name resolves to UNKNOWN rather than guessing a family (P6)."""
    assert detect_model_family(model_name) is ModelFamily.UNKNOWN
    assert steerability_policy_for(ModelFamily.UNKNOWN) == STEERABILITY_POLICY


def test_hermes_composition_swaps_only_the_steerability_section() -> None:
    """Identity and capabilities are family-invariant; only the closing frame changes.

    Killed by: src/uclone_x/agent/prompts.py :: ModelFamily.HERMES: HERMES_STEERABILITY_POLICY,
    Becomes: ModelFamily.UNKNOWN: STEERABILITY_POLICY,
    """
    hermes = compose_system_prompt("hermes3:8b")
    assert IDENTITY_GROUNDING in hermes
    assert TASK_CAPABILITIES in hermes
    assert HERMES_STEERABILITY_POLICY in hermes
    assert STEERABILITY_POLICY not in hermes
    assert hermes != DEFAULT_SYSTEM_PROMPT


def test_qwen_composition_swaps_only_the_steerability_section() -> None:
    """Qwen family composition frames for tool discipline and language matching."""
    qwen = compose_system_prompt("qwen3:8b")
    assert IDENTITY_GROUNDING in qwen
    assert TASK_CAPABILITIES in qwen
    assert QWEN_STEERABILITY_POLICY in qwen
    assert STEERABILITY_POLICY not in qwen
    assert qwen != DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize("model_name", ["llama3.2", "mistral-nemo", "gemma2:9b", None])
def test_families_without_their_own_framing_get_the_default_prompt_unchanged(
    model_name: str | None,
) -> None:
    """A family with no declared framing composes to exactly the default prompt."""
    assert compose_system_prompt(model_name) == DEFAULT_SYSTEM_PROMPT


def test_adapt_reframes_a_persona_prompt_without_touching_its_own_preamble() -> None:
    """Personas embed the default components, so adaptation reaches them too."""
    pioneer = get_default_persona_registry().get_persona("pioneer")
    assert pioneer is not None
    adapted_hermes = adapt_system_prompt(
        agent_config_for_persona(pioneer).system_prompt, "hermes3:8b"
    )
    assert adapted_hermes.startswith("You are Pioneer")
    assert HERMES_STEERABILITY_POLICY in adapted_hermes
    assert STEERABILITY_POLICY not in adapted_hermes

    adapted_qwen = adapt_system_prompt(agent_config_for_persona(pioneer).system_prompt, "qwen3:8b")
    assert adapted_qwen.startswith("You are Pioneer")
    assert QWEN_STEERABILITY_POLICY in adapted_qwen
    assert STEERABILITY_POLICY not in adapted_qwen


def test_adapt_returns_an_operator_written_prompt_byte_identical() -> None:
    """A prompt that does not embed the components is never appended to or rewritten.

    Killed by: src/uclone_x/agent/prompts.py :: return canonical.replace(STEERABILITY_POLICY, replacement)
    Becomes: return canonical + replacement
    """
    custom = "You are a terse SQL assistant. Answer with a query and nothing else."
    assert adapt_system_prompt(custom, "hermes3:8b") == custom
    assert adapt_system_prompt(custom, "qwen2.5:14b") == custom


def test_agent_config_carries_the_model_name_adaptation_reads() -> None:
    """The seam `BaseAgent.effective_system_prompt` uses is on the config, not the connector."""
    config = AgentConfig(agent_id="a", name="A")
    assert config.llm_config.model_name is None
    assert adapt_system_prompt(config.system_prompt, config.llm_config.model_name) == (
        DEFAULT_SYSTEM_PROMPT
    )


def test_effective_system_prompt_adapts_to_the_configured_model_family() -> None:
    """The adaptation is wired into the property every turn actually reads.

    Killed by: src/uclone_x/agent/base.py :: return adapt_system_prompt(base, self._config.llm_config.model_name)
    Becomes: return base
    """
    from uclone_x.agent import BaseAgent
    from uclone_x.agent.models import AgentLLMConfig

    default_agent = BaseAgent(config=AgentConfig(agent_id="a", name="A"))
    assert default_agent.effective_system_prompt == DEFAULT_SYSTEM_PROMPT

    hermes_agent = BaseAgent(
        config=AgentConfig(
            agent_id="b", name="B", llm_config=AgentLLMConfig(model_name="hermes3:8b")
        )
    )
    assert hermes_agent.effective_system_prompt == compose_system_prompt("hermes3:8b")
    assert STEERABILITY_POLICY not in hermes_agent.effective_system_prompt


def test_hot_reloading_to_a_hermes_model_reframes_the_prompt() -> None:
    """`hot_reload_llm` rewrites `llm_config.model_name`, so the framing must follow it."""
    from unittest.mock import MagicMock

    from uclone_x.agent import BaseAgent

    agent = BaseAgent(config=AgentConfig(agent_id="c", name="C"))
    assert agent.effective_system_prompt == DEFAULT_SYSTEM_PROMPT

    agent.hot_reload_llm(MagicMock(), model_name="hermes3:8b")
    assert agent.effective_system_prompt == compose_system_prompt("hermes3:8b")


def test_re_framing_away_from_a_family_restores_the_default_framing() -> None:
    """Re-framing is reversible: `hermes3:8b` -> `qwen3:8b` must not leave Hermes framing behind.

    The one-way `prompt.replace(STEERABILITY_POLICY, ...)` could only ever move a prompt
    *away* from the default. A prompt already framed for Hermes contained no
    `STEERABILITY_POLICY` to substitute, so adapting it for any other family returned it
    byte-identical — the reverse half of the #921 staleness, and the reason re-framing the
    anchored turn in place is only a fix once the substitution runs from the canonical form.

    Killed by: src/uclone_x/agent/prompts.py :: for framing in _FAMILY_STEERABILITY.values():
    Becomes: for framing in ():
    """
    hermes = compose_system_prompt("hermes3:8b")
    assert HERMES_STEERABILITY_POLICY in hermes
    assert STEERABILITY_POLICY not in hermes

    back = adapt_system_prompt(hermes, "llama3.2")

    assert back == DEFAULT_SYSTEM_PROMPT
    assert STEERABILITY_POLICY in back
    assert HERMES_STEERABILITY_POLICY not in back

    to_qwen = adapt_system_prompt(hermes, "qwen3:8b")
    assert QWEN_STEERABILITY_POLICY in to_qwen
    assert HERMES_STEERABILITY_POLICY not in to_qwen


def test_re_framing_is_idempotent_and_round_trips() -> None:
    """Adapting to the same family twice changes nothing, and a round trip returns the original."""
    default = compose_system_prompt("qwen3:8b")
    once = adapt_system_prompt(default, "hermes3:8b")

    assert adapt_system_prompt(once, "hermes3:8b") == once
    assert adapt_system_prompt(once, "qwen3:8b") == default
    assert adapt_system_prompt(adapt_system_prompt(default, "llama3:8b"), "qwen3:8b") == default


def test_re_framing_still_returns_an_operators_own_prompt_byte_identical() -> None:
    """The reversible spelling must not widen what gets substituted (P6: no silent rewrite)."""
    custom = "You are a microservice orchestration agent. Answer in one line."

    assert adapt_system_prompt(custom, "hermes3:8b") == custom
    assert adapt_system_prompt(custom, "qwen3:8b") == custom
    assert adapt_system_prompt("", "hermes3:8b") == ""
