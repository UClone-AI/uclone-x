"""What a turn sends back to the model when its answer is not yet one to keep (#1736).

`BaseAgent.execute_turn` decides *when* a turn nudges; this module holds *what* each nudge
says and the checks that decide whether it applies: the evidence nudge and whether the
model declined it, the grounding nudge and the text a turn's specifics may be grounded
in, and the nudge and post-turn sanitisation for image links to artifacts that do not
exist. Pure functions over what the turn already holds; none of them reads the agent.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from pathlib import Path

from uclone_x.agent.hooks import (
    MD_IMAGE_RE,
    extract_artifact_rel_path,
    is_artifact_missing,
    sanitize_hallucinated_artifacts,
)
from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.agent.session import redact_message
from uclone_x.llm.models import ChatMessage, MessageRole
from uclone_x.tools.protocols import ToolRegistryProtocol

__all__ = [
    "DECLINE_PHRASES",
    "EVIDENCE_REQUIRED_NUDGE",
    "GROUNDING_REQUIRED_NUDGE_PREFIX",
    "GROUNDING_REQUIRED_NUDGE_SUFFIX",
    "apply_artifact_sanitization",
    "evaluate_artifact_nudge",
    "grounding_supports",
    "is_evidence_nudge_declined",
]


#: Sent back to a model that answered with no tool execution, when the agent is configured
#: to require evidence. Phrased as a question about grounding rather than an instruction to
#: use a named tool: naming one steers the choice, and an earlier measurement of exactly
#: that steer moved one problem of ten -- inside the noise for an unrepeated run (#698).
EVIDENCE_REQUIRED_NUDGE = (
    "Nothing you did this turn produced evidence: either no tool ran, or the ones that ran "
    "returned nothing. A tool that matched nothing has not shown the thing is absent -- the "
    "query or the tool may be the wrong one. If the answer rests on something you can check "
    "here, check it a different way and answer again. If this question is answerable from what "
    "you were given, say that it is answerable from what you were given. If it does not, say "
    "plainly that you could not verify it."
)


DECLINE_PHRASES: tuple[str, ...] = (
    "answerable from what you were given",
    "answerable from what was given",
    "answerable from the prompt",
    "answerable from the given",
    "answerable from given",
    "answerable without",
    "derivable from the given",
    "derivable from given",
    "deducible from the given",
    "deducible from given",
    "without requiring external tools",
    "without tool use",
    "no tool calls were needed",
    "no tools were needed",
    "self-contained",
)


def _extract_terminal_answer(text: str) -> str | None:
    match = re.search(
        r"(?im)(?:final\s+answer|answer|\*\*answer\*\*):\s*[\*]*([a-zA-Z0-9_-]+)[\*]*",
        text,
    )
    if match:
        return match.group(1).lower()
    first_match = re.match(r"(?im)^\s*(?:answer:\s*)?[\*]*([a-zA-Z0-9_-]+)[\*]*[\.\!:,]", text)
    if first_match:
        val = first_match.group(1).lower()
        if val in {"yes", "no", "trapped", "correct"}:
            return val
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        last_line = re.sub(r"[\*_\.<>]", "", lines[-1]).strip().lower()
        if len(last_line.split()) <= 2:
            return last_line
    return None


def is_evidence_nudge_declined(content: str, first_answer: str | None = None) -> bool:
    """Whether the model declined the evidence nudge.

    Occurs either explicitly (declaring the prompt self-contained or answerable without tools)
    or implicitly when the post-nudge response is materially equivalent to the first answer
    without tool use (#784).
    """
    lowered = content.lower()
    if any(phrase in lowered for phrase in DECLINE_PHRASES):
        return True

    if first_answer is None:
        return False

    lowered_first = first_answer.lower().strip()
    lowered_second = lowered.strip()

    # Conflicting answers (e.g. opposite boolean polarity or contradictory numbers) do not decline
    nums1 = set(re.findall(r"\b\d+\b", lowered_first))
    nums2 = set(re.findall(r"\b\d+\b", lowered_second))
    if nums1 and nums2 and nums1.isdisjoint(nums2):
        return False

    has_yes1 = bool(re.search(r"\byes\b", lowered_first))
    has_no1 = bool(re.search(r"\bno\b", lowered_first))
    has_yes2 = bool(re.search(r"\byes\b", lowered_second))
    has_no2 = bool(re.search(r"\bno\b", lowered_second))
    if (has_yes1 and not has_no1 and has_no2 and not has_yes2) or (
        has_no1 and not has_yes1 and has_yes2 and not has_no2
    ):
        return False

    # If second is an abstention/retreat and first was not, the model complied with
    # the nudge's request to abstain rather than declining it.
    abstention_patterns = (
        r"\bcould not verify\b",
        r"\bcannot verify\b",
        r"\bunable to verify\b",
        r"\bnot verified\b",
    )
    is_second_abstention = any(re.search(pat, lowered_second) for pat in abstention_patterns)
    is_first_abstention = any(re.search(pat, lowered_first) for pat in abstention_patterns)
    if is_second_abstention and not is_first_abstention:
        return False

    # Exact string match (ignoring leading/trailing whitespace)
    if lowered_first == lowered_second:
        return True

    # Terminal answer matches (e.g. "Answer: no" or "5")
    term_first = _extract_terminal_answer(first_answer)
    term_second = _extract_terminal_answer(content)
    if term_first and term_second and term_first == term_second:
        return True

    # Concise first answer appears as a distinct phrase in post-nudge answer
    if len(lowered_first) <= 60:
        clean_first = re.sub(r"[\*_\.<>]", "", lowered_first).strip()
        pattern = rf"\b{re.escape(clean_first)}\b"
        # An empty phrase makes the pattern `\b\b`, which matches any answer at all.
        if clean_first and re.search(pattern, lowered_second):
            return True

    # High word-level Jaccard similarity (>= 0.6)
    w1 = set(re.findall(r"\b[a-zA-Z0-9_]+\b", lowered_first))
    w2 = set(re.findall(r"\b[a-zA-Z0-9_]+\b", lowered_second))
    if w1 and w2 and (len(w1 & w2) / len(w1 | w2) >= 0.6):
        return True

    return False


#: Opens the nudge sent back when the answer names specifics the turn read nowhere. The
#: constant is the stable prefix; the findings are appended, so a reader can recognise the
#: message without predicting which tokens it will quote.
#:
#: Two exits, deliberately. "Check it" alone is a refusal wearing a question -- it blocks a
#: task whose answer was legitimately derived rather than read, and a derived figure (a
#: count, a difference) is correct and appears in no output. "Say it is unchecked" is the
#: cheaper of the two errors under P6 and is what `DEFAULT_SYSTEM_PROMPT` already asks for.
#: As with the evidence nudge, no tool is named: naming one steers the choice (#698).
GROUNDING_REQUIRED_NUDGE_PREFIX = (
    "Your answer states specifics that appear in nothing you have read this turn: "
)


GROUNDING_REQUIRED_NUDGE_SUFFIX = (
    ". Check them here and answer again, or keep the answer and say plainly which parts of "
    "it you did not verify."
)


def grounding_supports(
    messages: Sequence[ChatMessage],
    tool_executions: Sequence[ToolExecutionRecord],
    injected: Collection[str] = (),
) -> list[str]:
    """Everything the turn was shown or told, minus what it said itself.

    Two exclusions, and both were measured rather than reasoned about:

    *   **Assistant turns.** A claim cannot be its own evidence. With them included, the
        answer given after a nudge finds its figure in the answer that provoked the nudge
        and reads as grounded.
    *   **`injected`, the runtime's own nudges.** The grounding nudge *quotes the
        unsupported specifics back*, so it lands in the request carrying exactly the tokens
        that are missing. Left in the support set it grounds them, and the second look at a
        repeated answer comes back clean -- the check silently disarming itself one step
        after firing. Pinned by `test_the_nudge_quoting_a_specific_does_not_then_ground_it`;
        the assistant exclusion is pinned by
        `test_the_models_own_earlier_answer_does_not_support_its_later_one`.

    A nudge is cut out of the message that carries it rather than matched against the
    whole message: it travels inside the turn-context block (#1420), so the message holds
    the nudge and more, and an equality test would never exclude it.
    """
    supports: list[str] = []
    for m in messages:
        if m.role is MessageRole.ASSISTANT or not m.content:
            continue
        text = str(m.content)
        for nudge in injected:
            text = text.replace(nudge, "")
        supports.append(text)
    supports.extend(str(record.output) for record in tool_executions)
    return supports


def _detect_missing_artifacts(
    resp_content: str,
    tools: ToolRegistryProtocol | None,
    workspace_root: Path | None,
) -> list[str]:
    """Find referenced image artifacts in response content that do not exist on disk."""
    if tools is None or tools.get("generate_image") is None:
        return []
    missing: list[str] = []
    for match in MD_IMAGE_RE.finditer(resp_content):
        raw_url = match.group(2)
        rel = extract_artifact_rel_path(raw_url)
        if rel and (rel.startswith("artifacts/images/") or rel.startswith("artifacts/")):
            if is_artifact_missing(rel, workspace_root):
                missing.append(rel)
    return missing


def evaluate_artifact_nudge(
    resp_content: str | None,
    tools: ToolRegistryProtocol | None,
    workspace_root: Path | None,
    artifact_nudged: bool,
) -> tuple[str, str] | None:
    """Check if in-turn nudge is needed for missing artifact image links."""
    if artifact_nudged or not resp_content:
        return None
    missing = _detect_missing_artifacts(resp_content, tools, workspace_root)
    if not missing:
        return None
    first_missing = Path(missing[0]).name
    nudge = (
        f"The referenced image file '{first_missing}' does not exist on disk, and no image generation "
        f"tool was called to produce it. Never predict, invent, or guess image URLs. "
        f"To provide an image, you must call the 'generate_image' tool with your prompt."
    )
    return missing[0], nudge


def apply_artifact_sanitization(
    content: str,
    workspace_root: Path | None,
    history: list[ChatMessage],
    assistant_msg_idx: int | None,
) -> str:
    """Post-turn sanitization for any remaining hallucinated artifact images."""
    if not content:
        return ""
    sanitized, count = sanitize_hallucinated_artifacts(content, workspace_root)
    if count > 0:
        if assistant_msg_idx is not None and assistant_msg_idx < len(history):
            orig_msg = history[assistant_msg_idx]
            history[assistant_msg_idx] = redact_message(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=sanitized or None,
                    tool_calls=orig_msg.tool_calls,
                )
            )
        return sanitized
    return content
