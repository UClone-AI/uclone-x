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
    extract_produced_artifact_paths,
    grounding_supports,
    is_evidence_nudge_declined,
)
from uclone_x.llm.models import ChatMessage, MessageRole
from uclone_x.tools.models import ToolResultStatus
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


def test_host_prefixed_missing_image_triggers_artifact_nudge(tmp_path: Path) -> None:
    """A host-prefixed missing image URL still triggers an artifact nudge."""
    found = evaluate_artifact_nudge(
        "![cat](https://example.com/api/artifacts/content?path=artifacts/images/cat.png)",
        _registry(),
        tmp_path,
        artifact_nudged=False,
    )
    assert found is not None
    rel, nudge = found
    assert rel == "artifacts/images/cat.png"
    assert "'cat.png'" in nudge
    assert "does not exist on disk" in nudge


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


def test_unproduced_existing_image_triggers_artifact_nudge(tmp_path: Path) -> None:
    """An existing disk image not produced this turn and not named by user earns an artifact nudge.

    Killed by: src/uclone_x/agent/nudges.py :: if not is_user_ref and norm_rel not in norm_produced:
    Becomes: if False:
    """
    img_file = tmp_path / "artifacts" / "images" / "old_cat.png"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    img_file.write_bytes(b"data")
    content = "Here it is: ![cat](artifacts/images/old_cat.png)"

    found = evaluate_artifact_nudge(
        content,
        _registry(),
        tmp_path,
        artifact_nudged=False,
        produced_paths=set(),
        user_message="draw me a cat",
    )

    assert found is not None
    rel, nudge = found
    assert rel == "artifacts/images/old_cat.png"
    assert "'old_cat.png'" in nudge
    assert "was not made in this turn" in nudge


def test_user_referenced_existing_image_avoids_nudge(tmp_path: Path) -> None:
    """When the user message explicitly mentions the image name, referencing it does not trigger a nudge.

    Killed by: src/uclone_x/agent/nudges.py :: is_user_ref = bool(
    Becomes: is_user_ref = not bool(
    """
    img_file = tmp_path / "artifacts" / "images" / "old_cat.png"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    img_file.write_bytes(b"data")
    content = "Here it is: ![cat](artifacts/images/old_cat.png)"

    found = evaluate_artifact_nudge(
        content,
        _registry(),
        tmp_path,
        artifact_nudged=False,
        produced_paths=set(),
        user_message="what do you think of old_cat.png?",
    )

    assert found is None


def test_turn_produced_image_avoids_nudge(tmp_path: Path) -> None:
    """When the image was produced in this turn, referencing it does not trigger a nudge."""
    img_file = tmp_path / "artifacts" / "images" / "new_cat.png"
    img_file.parent.mkdir(parents=True, exist_ok=True)
    img_file.write_bytes(b"data")
    content = "Here it is: ![cat](artifacts/images/new_cat.png)"

    found = evaluate_artifact_nudge(
        content,
        _registry(),
        tmp_path,
        artifact_nudged=False,
        produced_paths={"artifacts/images/new_cat.png"},
        user_message="draw me a cat",
    )

    assert found is None


def test_extract_produced_artifact_paths_reads_the_declared_channel() -> None:
    """A tool's declared `artifacts` count whatever its output looks like (#2085).

    The image tool's result has carried only `relative_url` since #2013 and declares no
    `writes_files` since #2079; a reader that parsed the output's shape missed every picture.

    Killed by: src/uclone_x/agent/models.py :: if self.artifacts:
    Becomes: if False:
    """
    records = [
        ToolExecutionRecord(
            tool_name="generate_image",
            output={"relative_url": "/api/artifacts/content?path=artifacts/images/img_1.png"},
            status=ToolResultStatus.SUCCESS,
            artifacts=("artifacts/images/img_1.png",),
        ),
        ToolExecutionRecord(
            tool_name="generate_image",
            output={"path": "artifacts/images/failed.png"},
            status=ToolResultStatus.ERROR,
            writes_files=True,
            artifacts=("artifacts/images/failed.png",),
        ),
    ]

    assert extract_produced_artifact_paths(records) == {"artifacts/images/img_1.png"}


def test_extract_produced_artifact_paths_falls_back_only_for_writers() -> None:
    """With nothing declared, a writer's `path` counts and a reader's never does.

    Killed by: src/uclone_x/agent/models.py :: if not self.writes_files or not isinstance(self.output, dict):
    Becomes: if not isinstance(self.output, dict):
    """
    records = [
        ToolExecutionRecord(
            tool_name="story_write",
            output={"path": "artifacts/stories/s.md"},
            status=ToolResultStatus.SUCCESS,
            writes_files=True,
        ),
        ToolExecutionRecord(
            tool_name="file_read",
            output={"path": "artifacts/images/old.png"},
            status=ToolResultStatus.SUCCESS,
        ),
    ]

    assert extract_produced_artifact_paths(records) == {"artifacts/stories/s.md"}


def test_absolute_artifact_paths_are_dropped_not_stripped() -> None:
    """An absolute fallback path never becomes a bogus workspace-relative one.

    Killed by: src/uclone_x/agent/models.py :: if path and not path.startswith("/") and ":" not in path[:3] and path not in kept:
    Becomes: if path and path not in kept:
    """
    record = ToolExecutionRecord(
        tool_name="generate_image",
        status=ToolResultStatus.SUCCESS,
        artifacts=(
            "/Users/u/ws/artifacts/images/a.png",
            "C:\\ws\\a.png",
            "artifacts\\images\\b.png",
        ),
    )

    assert record.produced_paths == ("artifacts/images/b.png",)


def test_not_made_nudge_never_claims_the_tool_was_not_called(tmp_path: Path) -> None:
    """When this turn did draw, the nudge names those files instead of denying the call.

    The false "no image generation tool was called" made the model drop a correct picture.

    Killed by: src/uclone_x/agent/nudges.py :: if made:
    Becomes: if False:
    """
    old = tmp_path / "artifacts" / "images" / "old.png"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"data")

    found = evaluate_artifact_nudge(
        "![old](artifacts/images/old.png)",
        _registry(),
        tmp_path,
        artifact_nudged=False,
        produced_paths={"artifacts/images/new.png"},
        user_message="draw me a cat",
    )

    assert found is not None
    _, nudge = found
    assert "no image generation" not in nudge
    assert "This turn made: new.png" in nudge
