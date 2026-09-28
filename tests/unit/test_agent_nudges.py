"""`agent/nudges.py` on its own: the checks a turn runs before keeping an answer (#1736).

These helpers moved out of `agent/base.py` unchanged. The behavioural tests that drive them
through `BaseAgent.execute_turn` stay where they were (`test_nudge_retry_request.py`,
`test_turn_grounding_persistence.py`, `test_artifact_nudge.py`); this file pins each helper
directly, so a defect in one is reported against the module that holds it rather than as a
turn-loop failure several frames away.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.agent.nudges import (
    EVIDENCE_REQUIRED_NUDGE,
    evaluate_artifact_nudge,
    grounding_supports,
    is_evidence_nudge_declined,
)
from uclone_x.llm.models import ChatMessage, MessageRole
from uclone_x.tools.protocols import ToolRegistryProtocol


class _ImageToolRegistry:
    """Just enough registry for the artifact check: it asks only for `generate_image`."""

    def get(self, name: str) -> object | None:
        return object() if name == "generate_image" else None


def _registry() -> ToolRegistryProtocol:
    return cast(ToolRegistryProtocol, _ImageToolRegistry())


def test_a_self_contained_declaration_declines_the_nudge_without_a_first_answer() -> None:
    """Saying the question needs no tools is an explicit decline, whatever came before.

    Killed by: src/uclone_x/agent/nudges.py :: if any(phrase in lowered for phrase in DECLINE_PHRASES):
    Becomes: if False:
    """
    assert is_evidence_nudge_declined("This is answerable without any tools: 4.") is True


def test_a_different_answer_with_no_decline_phrase_is_not_a_decline() -> None:
    """With no first answer to compare against, only a phrase can decline."""
    assert is_evidence_nudge_declined("I checked the file; it says 4.") is False


def test_tool_output_counts_as_support_and_the_models_own_words_do_not() -> None:
    """What the tools returned is support; the assistant's own messages are not.

    Killed by: src/uclone_x/agent/nudges.py :: supports.extend(str(record.output) for record in tool_executions)
    Becomes: pass
    """
    messages = [
        ChatMessage(role=MessageRole.USER, content="how many rows?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="There are 812 rows."),
    ]
    records = [ToolExecutionRecord(tool_name="count_rows", output="812")]

    supports = grounding_supports(messages, records)

    assert supports == ["how many rows?", "812"]


def test_an_injected_nudge_is_cut_out_of_the_message_that_carries_it() -> None:
    """The runtime's own nudge text must not ground the specifics it quotes back."""
    carrier = ChatMessage(role=MessageRole.USER, content=f"context {EVIDENCE_REQUIRED_NUDGE}")

    supports = grounding_supports([carrier], [], injected=(EVIDENCE_REQUIRED_NUDGE,))

    assert supports == ["context "]


def test_a_missing_image_link_produces_a_nudge_naming_the_file(tmp_path: Path) -> None:
    """An image link to nothing on disk earns one nudge, naming the file it invented."""
    content = "Here it is: ![cat](artifacts/images/cat.png)"

    found = evaluate_artifact_nudge(content, _registry(), tmp_path, artifact_nudged=False)

    assert found is not None
    rel, nudge = found
    assert rel == "artifacts/images/cat.png"
    assert "'cat.png'" in nudge
    assert "generate_image" in nudge


def test_the_artifact_nudge_is_sent_at_most_once(tmp_path: Path) -> None:
    """Once nudged, the same missing link must not be nudged again.

    Killed by: src/uclone_x/agent/nudges.py :: if artifact_nudged or not resp_content:
    Becomes: if not resp_content:
    """
    content = "Here it is: ![cat](artifacts/images/cat.png)"

    assert evaluate_artifact_nudge(content, _registry(), tmp_path, artifact_nudged=True) is None


def test_no_image_tool_means_no_artifact_nudge(tmp_path: Path) -> None:
    """A nudge telling the model to call a tool it does not have would be a dead end."""
    content = "Here it is: ![cat](artifacts/images/cat.png)"

    assert evaluate_artifact_nudge(content, None, tmp_path, artifact_nudged=False) is None
