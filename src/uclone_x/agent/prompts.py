"""Composable system prompt architecture for runtime agents (#871).

`DEFAULT_SYSTEM_PROMPT` was a single space-joined literal in `uclone_x.agent.models`
that mixed three unrelated concerns — who the agent is, what it should do with its
tools, and how it should carry those constraints when the user asks for creative or
expressive work. Three consequences followed, and this module addresses each:

1. **Refusal pressure.** A long run of `Never …` clauses reads, to a steerable local
   model, as a refusal cue rather than an accuracy cue. The accuracy constraints are all still
   here — none is softened or dropped — but they are grouped, and a closing
   `STEERABILITY_POLICY` states affirmatively what they are *for*: they constrain
   accuracy and tool routing, and are not by themselves a reason to decline an otherwise
   reasonable request.
2. **No model-family adaptation.** `compose_system_prompt()` takes the active model
   name and varies **only** the steerability framing by family. Identity grounding and
   task capabilities are byte-identical across every family, because they encode
   product behaviour rather than a model convention.
3. **Brittle tests.** Each component is a named constant, so a test can assert the
   invariant it cares about against that component instead of substring-matching a
   sentence out of the assembled blob.

What this module does **not** do: it does not instruct a model to abandon its own
judgement, and it adds no "unfiltered" or "uncensored" directive. The affirmative
rewrite covers the constraints this repository actually states — terminology,
reporting, artifact routing, local image generation — and nothing beyond them.

**Guidance follows the tools (#1424).** Tool guidance is a table of fragments, each keyed
by the tool it routes to (`CAPABILITY_FRAGMENTS`), and a fragment is composed in only when
the agent holds that tool. A seat with four tools used to carry image-generation and
`install_package` instructions it could not act on, on every request. The composition is a
pure function of the tool *set* (never its order), the write permission and the model
family, so a given persona composes to the same bytes on every turn and every restart --
it is part of the cached prefix.

The technical-terminology clause (`uclone_x.agent.terminology`) is no longer part of the
default. It is domain guidance, not tool guidance: a persona that wants it states it in its
own `system_prompt`, as the built-in `clone` does. That is the whole opt-in mechanism -- no
persona key, so no editor field and no save path that could drop it.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class ModelFamily(StrEnum):
    """Model family a system prompt is being framed for.

    `UNKNOWN` is the honest answer for an unrecognised or absent model name, and it
    resolves to the default framing — it is not a silent substitution of some other
    family (P6).
    """

    HERMES = "hermes"
    QWEN = "qwen"
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    PHI = "phi"
    UNKNOWN = "unknown"


#: Ordered `(token, family)` pairs matched as lowercase substrings of the model name.
#: Order matters where one name contains another's token: `nous-hermes2` must resolve
#: to HERMES, not to whatever else it happens to contain.
_FAMILY_TOKENS: Final[tuple[tuple[str, ModelFamily], ...]] = (
    ("hermes", ModelFamily.HERMES),
    ("deepseek", ModelFamily.DEEPSEEK),
    ("qwen", ModelFamily.QWEN),
    ("llama", ModelFamily.LLAMA),
    ("mistral", ModelFamily.MISTRAL),
    ("mixtral", ModelFamily.MISTRAL),
    ("gemma", ModelFamily.GEMMA),
    ("phi", ModelFamily.PHI),
)


IDENTITY_GROUNDING: Final[str] = (
    "You are an autonomous AI assistant powered by the UClone-X local-first runtime. "
    "You operate locally within the user's workspace, leveraging specialized tools, skills, "
    "and knowledge ontologies to solve tasks rigorously and privately. "
    "Do not introduce yourself with default pre-trained foundation model greetings or claim to be a generic cloud model "
    "(such as Qwen, Llama, GPT, Claude, or Gemini) unless the user explicitly asks about the underlying model weights."
)

HONEST_REPORTING: Final[str] = (
    "Report what you actually did and what actually happened. "
    "If a step failed, say so and show the evidence. "
    "If you did not verify something, say that you did not verify it. "
    "Never present a substituted, defaulted, or partial result as though it were the real one."
)

ARTIFACT_REPORTING: Final[str] = (
    "When providing comprehensive equipment/product recommendations or technical comparisons, do not dump long unformatted lists into the chat. "
    "Instead, compile a clean structured markdown comparison report with specifications, pros/cons, and pricing comparison table "
    "via file_write into the workspace (e.g. docs/recommendations/...md), and provide a concise summary in the chat referencing the artifact."
)

IMAGE_GENERATION: Final[str] = (
    "When the user requests creating, illustrating, or visualizing an image, diagram, or photo, use the generate_image tool. "
    "UClone-X generates images locally and privately — in this process from a local checkpoint, on a ComfyUI daemon already running on this machine, or on a local-network CUDA worker — "
    "so keep image work on that path rather than sending the user to an external paid cloud service "
    "(such as DALL-E, MidJourney, or a commercial web generator). "
    "When presenting the generated image in your response, use standard markdown image ![description](relative_url) or link [title](relative_url) using the relative_url returned by the tool. "
    "Never predict, invent, or guess image URLs; only use URLs returned in tool results. "
    "When multiple images or variations are requested, specify count in generate_image parameters (up to 10). "
    "Do not wrap images in <image> or custom XML/HTML tags."
)

ENVIRONMENT_REPAIR: Final[str] = (
    "When a tool fails because an optional dependency of this project is missing, install it "
    "with the install_package tool and then retry what failed. The same applies when the user "
    "asks you to install something, or to do it yourself: perform the installation, do not "
    "reply with shell commands for the user to run. Report the outcome from the tool result -- "
    "if the install failed, say so and show what it reported. What install_package cannot do is "
    "accept a licence on the user's behalf or install software outside this project; say so "
    "plainly and name what the user has to do."
)


@dataclass(frozen=True)
class CapabilityFragment:
    """Guidance that is true only for an agent holding `tool`.

    `writes_files` repeats the tool's own `writes_files` declaration (#1167): a persona with
    `enable_write_tools: false` is refused every such tool, so it is not told to use one. A
    test holds each value to the registered tool's, so the two cannot disagree.
    """

    tool: str
    text: str
    writes_files: bool


#: Tool guidance, in the order it is composed. Keyed by tool name; see the module docstring.
CAPABILITY_FRAGMENTS: Final[tuple[CapabilityFragment, ...]] = (
    CapabilityFragment(tool="file_write", text=ARTIFACT_REPORTING, writes_files=True),
    CapabilityFragment(tool="generate_image", text=IMAGE_GENERATION, writes_files=True),
    CapabilityFragment(tool="install_package", text=ENVIRONMENT_REPAIR, writes_files=True),
)


def capability_guidance(
    tools: Collection[str] | None, *, writes_permitted: bool = True
) -> tuple[str, ...]:
    """The fragments an agent holding `tools` is given, in table order.

    `tools=None` is an agent with no tool restriction, which holds every registered tool --
    the default registry registers all three fragment tools. An empty collection is an
    agent holding none. Only membership is read, so the order the caller lists tools in
    never reaches the prompt.
    """
    return tuple(
        fragment.text
        for fragment in CAPABILITY_FRAGMENTS
        if (tools is None or fragment.tool in tools)
        and (writes_permitted or not fragment.writes_files)
    )


def compose_task_capabilities(
    tools: Collection[str] | None = None, *, writes_permitted: bool = True
) -> str:
    """Reporting rules plus the tool guidance `tools` earns, as one section."""
    return " ".join(
        (HONEST_REPORTING, *capability_guidance(tools, writes_permitted=writes_permitted))
    )


#: The capabilities section for an unrestricted agent -- every fragment. Kept as a named
#: constant so a test can pin it against the assembled default.
TASK_CAPABILITIES: Final[str] = compose_task_capabilities()

#: The default closing frame: says what the constraints above are for, so a model does
#: not read an accuracy rule as a refusal cue. Substituted wholesale by
#: `adapt_system_prompt` when a family declares its own framing.
STEERABILITY_POLICY: Final[str] = (
    "Those constraints govern accuracy and tool routing; on their own they are not a reason "
    "to decline an otherwise reasonable request. "
    "Respect User Agency: Treat the user as an autonomous decision-maker, and follow their "
    "instructions and style requests without imposing personal or ideological judgments. "
    "Substantive Answers: Provide detailed, substantive answers that directly address the prompt. "
    "No Moralizing: Do not add patronizing commentary, moral lectures, unsolicited caveats, or boilerplate disclaimers. "
    "Tone Adaptation: Match the tone, intensity, and creative or stylistic direction requested by the user, "
    "and respond naturally in the language in which the user communicates."
)

HERMES_STEERABILITY_POLICY: Final[str] = (
    "Treat this system prompt as your operating contract; it, and the user's stated intent, "
    "take precedence over habits carried in from pre-training. "
    "The constraints above govern accuracy and tool routing; on their own they are not a reason "
    "to decline an otherwise reasonable request. "
    "Respect User Agency: Treat the user as an autonomous decision-maker, and follow their "
    "instructions and style requests without imposing personal or ideological judgments. "
    "Substantive Answers: Provide detailed, substantive answers that directly address the prompt. "
    "No Moralizing: Do not add patronizing commentary, moral lectures, unsolicited caveats, or boilerplate disclaimers. "
    "Tone Adaptation: Match the tone, intensity, and creative or stylistic direction requested by the user, "
    "and respond naturally in the language in which the user communicates."
)

#: Families whose framing departs from `STEERABILITY_POLICY`. A family absent from this
#: mapping gets the default — that is a declared fallback, not a silent one, and
#: `steerability_policy_for` is the single place it is resolved.
_FAMILY_STEERABILITY: Final[Mapping[ModelFamily, str]] = {
    ModelFamily.HERMES: HERMES_STEERABILITY_POLICY,
}


def detect_model_family(model_name: str | None) -> ModelFamily:
    """Resolve the model family from an Ollama-style model name.

    Matches lowercase substrings, so tag and size suffixes (`hermes3:8b`,
    `qwen2.5:14b-instruct`) resolve without an exhaustive name table. An absent or
    unrecognised name returns `ModelFamily.UNKNOWN` rather than guessing.
    """
    if not model_name:
        return ModelFamily.UNKNOWN
    lowered = model_name.lower()
    for token, family in _FAMILY_TOKENS:
        if token in lowered:
            return family
    return ModelFamily.UNKNOWN


def steerability_policy_for(family: ModelFamily) -> str:
    """Return the steerability framing for `family`, defaulting to `STEERABILITY_POLICY`."""
    return _FAMILY_STEERABILITY.get(family, STEERABILITY_POLICY)


def compose_system_prompt(
    model_name: str | None = None,
    *,
    tools: Collection[str] | None = None,
    writes_permitted: bool = True,
) -> str:
    """Assemble the three components into a system prompt for the active model.

    Sections are separated by a blank line so each is visible as a unit to a reader and
    to a model, which the single space-joined literal this replaces was not.

    `tools` and `writes_permitted` select the tool guidance (`capability_guidance`); left
    at their defaults the prompt is the one for an unrestricted agent.
    """
    return "\n\n".join(
        (
            IDENTITY_GROUNDING,
            compose_task_capabilities(tools, writes_permitted=writes_permitted),
            steerability_policy_for(detect_model_family(model_name)),
        )
    )


def composed_default_prompts() -> tuple[str, ...]:
    """Every prompt `compose_system_prompt` returns for an unspecified model family.

    One per subset of `CAPABILITY_FRAGMENTS`. What lets a reader recognise an appended
    default without knowing which tools it was composed for.
    """
    names = [entry.tool for entry in CAPABILITY_FRAGMENTS]
    return tuple(
        compose_system_prompt(tools=subset)
        for size in range(len(names) + 1)
        for subset in itertools.combinations(names, size)
    )


def adapt_system_prompt(prompt: str, model_name: str | None) -> str:
    """Re-frame an already-composed prompt's steerability section for `model_name`.

    Used where the prompt is not composed on the spot — a persona, an operator's
    configured prompt, or a system turn already anchored in a live session's history,
    all of which may embed the default components.

    A **substitution**, never an append. A prompt that carries no steerability framing
    this module knows about therefore comes back byte-identical: an operator who wrote
    their own prompt gets exactly what they wrote. That is also why there is no
    "is this one of ours?" guard here — there is nothing for the guard to prevent, and a
    guard whose removal changes no behaviour is a line no test can pin.

    **Idempotent and reversible, which the one-way spelling was not.** The substitution
    runs from the *canonical* `STEERABILITY_POLICY`, so it first normalises whichever
    family framing the prompt currently carries back to that canonical form. A prompt
    already framed for Hermes that is re-adapted for Qwen therefore comes back with the
    default framing. The previous `prompt.replace(STEERABILITY_POLICY, ...)` could only
    ever move a prompt *away* from the default and silently returned Hermes framing
    unchanged for every other family — the stale half of #921, and the reason adapting
    the anchored turn in place cannot be the whole fix on its own.
    """
    replacement = steerability_policy_for(detect_model_family(model_name))
    canonical = prompt
    for framing in _FAMILY_STEERABILITY.values():
        canonical = canonical.replace(framing, STEERABILITY_POLICY)
    return canonical.replace(STEERABILITY_POLICY, replacement)


__all__ = [
    "ARTIFACT_REPORTING",
    "CAPABILITY_FRAGMENTS",
    "ENVIRONMENT_REPAIR",
    "HERMES_STEERABILITY_POLICY",
    "HONEST_REPORTING",
    "IDENTITY_GROUNDING",
    "IMAGE_GENERATION",
    "STEERABILITY_POLICY",
    "TASK_CAPABILITIES",
    "CapabilityFragment",
    "ModelFamily",
    "adapt_system_prompt",
    "capability_guidance",
    "compose_system_prompt",
    "compose_task_capabilities",
    "composed_default_prompts",
    "detect_model_family",
    "steerability_policy_for",
]
