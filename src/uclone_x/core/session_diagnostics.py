"""Session diagnostics, inspection, and health auditing for UClone-X.

Provides pure core diagnostic models and functions for:
- Summarizing session records and turn budgets
- Detailed turn-by-turn inspection with role breakdown
- Health checking for orphaned tool results, turn budget saturation,
  unresolved tool calls, and missing offload artifacts.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from uclone_x.agent.session import (
    SessionStore,
    validate_session_id,
)
from uclone_x.llm.models import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

OFFLOAD_PATH_PATTERN = re.compile(r"saved to '([^']+)'")
# An advisory threshold on *interaction turns* — how many times a human has gone back and
# forth with the agent — used only to label a session "saturated" in diagnostics output.
# It is deliberately unrelated to `AgentConfig.max_steps`, the P4 ceiling on agent steps
# inside a single turn: nothing here may terminate a conversation, and comparing a turn
# counter against the step ceiling is the anti-pattern recorded in issue 2026-09-05-001.
# See `docs/guides/agent-runtime-terminology.md`.
DEFAULT_MAX_CONVERSATION_TURNS = 20


@dataclass(frozen=True)
class SessionSummary:
    """Concise metadata summary of a persisted session."""

    session_id: str
    agent_id: str
    turn_counter: int
    message_count: int
    created_at: str
    updated_at: str
    revision: int
    status: str
    has_compaction: bool = False
    has_plan: bool = False


@dataclass(frozen=True)
class SessionDetails:
    """In-depth diagnostic representation of a session and its contents."""

    summary: SessionSummary
    messages: tuple[ChatMessage, ...]
    role_counts: dict[str, int]
    has_plan: bool
    plan_title: str | None
    plan_steps_total: int
    plan_steps_completed: int
    artifacts_count: int
    artifacts_total_bytes: int
    artifact_files: list[str]


def _empty_details() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class SessionHealthIssue:
    """One diagnostic finding or anomaly detected during a session health check."""

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    turn_index: int | None = None
    details: dict[str, Any] = field(default_factory=_empty_details)


@dataclass(frozen=True)
class SessionHealthReport:
    """Outcome of a comprehensive session integrity and health check."""

    session_id: str
    healthy: bool
    error_count: int
    warning_count: int
    issues: list[SessionHealthIssue]


def count_active_turns(messages: Sequence[ChatMessage]) -> int:
    """Interaction turns held in the *active* context (#872, #883).

    Assistant messages only, so a compaction ledger — which is `SYSTEM` — is not counted
    as a turn that is still in the window. This is what the saturation signal reads;
    `SessionState.turn_counter` answers a different question (how many turns this session
    has ever taken) and Core keeps it across compaction on purpose (P5).
    """
    return sum(1 for msg in messages if msg.role == MessageRole.ASSISTANT)


def _resolve_session_status(
    active_turns: int,
    message_count: int,
    max_conversation_turns: int,
    has_compaction: bool,
) -> str:
    """Derive a human-readable operational status for a session."""
    if message_count == 0:
        return "empty"
    if active_turns >= max_conversation_turns:
        return "saturated"
    if has_compaction:
        return "compacted"
    return "active"


def _list_session_artifacts(
    workspace_root: Path | None, session_id: str
) -> tuple[int, int, list[str]]:
    """Enumerate offloaded tool output artifacts for a session."""
    if workspace_root is None:
        return 0, 0, []
    artifacts_dir = workspace_root / ".sandbox" / "tool_artifacts" / session_id
    if not artifacts_dir.is_dir():
        return 0, 0, []

    files: list[str] = []
    total_bytes = 0
    try:
        for entry in sorted(artifacts_dir.iterdir()):
            if entry.is_file():
                files.append(entry.name)
                total_bytes += entry.stat().st_size
    except OSError as exc:
        logger.warning("Failed to inspect tool artifacts directory %s: %s", artifacts_dir, exc)
    return len(files), total_bytes, files


def inspect_session(
    session_id: str,
    store: SessionStore | None = None,
    workspace_root: Path | None = None,
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
) -> SessionDetails | None:
    """Inspect and return detailed diagnostics for a single session.

    Returns None if the session does not exist or cannot be read.
    """
    effective_store = store if store is not None else SessionStore()
    state = effective_store.load(session_id)
    if state is None:
        return None

    role_counts: dict[str, int] = {
        MessageRole.SYSTEM.value: 0,
        MessageRole.USER.value: 0,
        MessageRole.ASSISTANT.value: 0,
        MessageRole.TOOL.value: 0,
    }
    has_compaction = False
    for msg in state.messages:
        role_counts[msg.role.value] = role_counts.get(msg.role.value, 0) + 1
        if msg.compaction_ledger:
            has_compaction = True

    active_turns = count_active_turns(state.messages)
    status = _resolve_session_status(
        active_turns=active_turns,
        message_count=len(state.messages),
        max_conversation_turns=max_conversation_turns,
        has_compaction=has_compaction,
    )

    plan_title: str | None = None
    plan_steps_total = 0
    plan_steps_completed = 0
    has_plan = state.plan is not None
    if state.plan is not None:
        plan_title = state.plan.title
        plan_steps_total = len(state.plan.steps)
        plan_steps_completed = sum(1 for s in state.plan.steps if getattr(s, "completed", False))

    art_count, art_bytes, art_files = _list_session_artifacts(workspace_root, session_id)

    summary = SessionSummary(
        session_id=state.session_id,
        agent_id=state.agent_id,
        turn_counter=state.turn_counter,
        message_count=len(state.messages),
        created_at=state.created_at,
        updated_at=state.updated_at,
        revision=state.revision,
        status=status,
        has_compaction=has_compaction,
        has_plan=has_plan,
    )

    return SessionDetails(
        summary=summary,
        messages=state.messages,
        role_counts=role_counts,
        has_plan=has_plan,
        plan_title=plan_title,
        plan_steps_total=plan_steps_total,
        plan_steps_completed=plan_steps_completed,
        artifacts_count=art_count,
        artifacts_total_bytes=art_bytes,
        artifact_files=art_files,
    )


def list_session_summaries(
    store: SessionStore | None = None,
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
    limit: int = 50,
) -> list[SessionSummary]:
    """Enumerate and summarize persisted sessions, sorted by updated_at descending."""
    effective_store = store if store is not None else SessionStore()
    summaries: list[SessionSummary] = []

    for sid in effective_store.list_session_ids():
        state = effective_store.load(sid)
        if state is None:
            continue
        has_compaction = any(m.compaction_ledger for m in state.messages)
        active_turns = count_active_turns(state.messages)
        status = _resolve_session_status(
            active_turns=active_turns,
            message_count=len(state.messages),
            max_conversation_turns=max_conversation_turns,
            has_compaction=has_compaction,
        )
        summaries.append(
            SessionSummary(
                session_id=state.session_id,
                agent_id=state.agent_id,
                turn_counter=state.turn_counter,
                message_count=len(state.messages),
                created_at=state.created_at,
                updated_at=state.updated_at,
                revision=state.revision,
                status=status,
                has_compaction=has_compaction,
                has_plan=state.plan is not None,
            )
        )

    summaries.sort(key=lambda s: s.updated_at, reverse=True)
    return summaries[:limit]


def check_session_health(
    session_id: str,
    store: SessionStore | None = None,
    workspace_root: Path | None = None,
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
) -> SessionHealthReport:
    """Execute integrity and health verification for a session.

    Checks:
    1. Valid session identifier (P3 path traversal and character check).
    2. Session existence in persistence store.
    3. Conversation-turn saturation (turn_counter >= max_conversation_turns).
    4. Conversation structure and message anomalies:
       - Orphaned tool results (TOOL message with unrecognised tool_call_id).
       - Missing tool_call_id on TOOL messages.
       - Unresolved tool calls (assistant calls tools but conversation proceeds without results).
       - Empty messages (no content and no tool calls).
    5. Offloaded tool output artifact integrity (referenced file exists on disk).
    """
    issues: list[SessionHealthIssue] = []

    # 1. Identifier validation
    try:
        validate_session_id(session_id)
    except Exception as exc:
        issues.append(
            SessionHealthIssue(
                severity="error",
                code="INVALID_SESSION_ID",
                message=f"Session ID '{session_id}' violates security constraints: {exc}",
                details={"session_id": session_id, "error": str(exc)},
            )
        )
        return SessionHealthReport(
            session_id=session_id,
            healthy=False,
            error_count=1,
            warning_count=0,
            issues=issues,
        )

    # 2. Existence check
    effective_store = store if store is not None else SessionStore()
    state = effective_store.load(session_id)
    if state is None:
        issues.append(
            SessionHealthIssue(
                severity="error",
                code="SESSION_NOT_FOUND",
                message=f"Session '{session_id}' not found in store at '{effective_store.storage_dir}'.",
                details={"session_id": session_id, "storage_dir": str(effective_store.storage_dir)},
            )
        )
        return SessionHealthReport(
            session_id=session_id,
            healthy=False,
            error_count=1,
            warning_count=0,
            issues=issues,
        )

    # 3. Turn budget saturation
    active_turns = count_active_turns(state.messages)
    if active_turns >= max_conversation_turns:
        issues.append(
            SessionHealthIssue(
                severity="warning",
                code="TURN_BUDGET_SATURATED",
                message=(
                    f"Active conversation turns ({active_turns}) have reached or exceeded the advisory "
                    f"conversation-turn threshold ({max_conversation_turns}). "
                    "Next turns may require reset or budget extension."
                ),
                details={
                    "active_turns": active_turns,
                    "turn_counter": state.turn_counter,
                    "max_conversation_turns": max_conversation_turns,
                },
            )
        )

    # 4. Message inspection
    pending_tool_calls: dict[str, int] = {}
    known_tool_calls: set[str] = set()

    for idx, msg in enumerate(state.messages):
        # Empty message check
        has_content = bool(msg.content and msg.content.strip())
        has_calls = bool(msg.tool_calls)
        if not has_content and not has_calls:
            issues.append(
                SessionHealthIssue(
                    severity="warning",
                    code="EMPTY_MESSAGE_CONTENT",
                    message=f"Turn #{idx} ({msg.role.value}) has empty content and no tool calls.",
                    turn_index=idx,
                    details={"role": msg.role.value},
                )
            )

        # Assistant tool calls
        if msg.role == MessageRole.ASSISTANT:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    pending_tool_calls[tc.id] = idx
                    known_tool_calls.add(tc.id)

        # Tool response verification
        elif msg.role == MessageRole.TOOL:
            if not msg.tool_call_id:
                issues.append(
                    SessionHealthIssue(
                        severity="error",
                        code="MISSING_TOOL_CALL_ID",
                        message=f"Turn #{idx} (tool) is missing required tool_call_id.",
                        turn_index=idx,
                        details={"role": msg.role.value},
                    )
                )
            elif msg.tool_call_id not in known_tool_calls:
                issues.append(
                    SessionHealthIssue(
                        severity="error",
                        code="ORPHANED_TOOL_RESULT",
                        message=(
                            f"Turn #{idx} (tool) has tool_call_id '{msg.tool_call_id}' which does not "
                            "correspond to any tool call requested by an assistant."
                        ),
                        turn_index=idx,
                        details={"tool_call_id": msg.tool_call_id},
                    )
                )
            else:
                pending_tool_calls.pop(msg.tool_call_id, None)

        # User message verification: check if previous assistant turn left tool calls dangling
        elif msg.role == MessageRole.USER:
            if pending_tool_calls:
                dangling_ids = list(pending_tool_calls.keys())
                issues.append(
                    SessionHealthIssue(
                        severity="warning",
                        code="UNRESOLVED_TOOL_CALLS",
                        message=(
                            f"User message at turn #{idx} arrived while tool calls "
                            f"{dangling_ids} from assistant turn remained unresolved."
                        ),
                        turn_index=idx,
                        details={"unresolved_call_ids": dangling_ids},
                    )
                )
                pending_tool_calls.clear()

        # 5. Offloaded artifact reference check
        if msg.content and "[Tool Output Offloaded" in msg.content:
            match = OFFLOAD_PATH_PATTERN.search(msg.content)
            if match:
                rel_path = match.group(1)
                target_path: Path | None = None
                if workspace_root is not None:
                    target_path = workspace_root / rel_path
                else:
                    target_path = Path(rel_path)

                if target_path and not target_path.is_file():
                    issues.append(
                        SessionHealthIssue(
                            severity="warning",
                            code="MISSING_OFFLOAD_ARTIFACT",
                            message=(
                                f"Turn #{idx} references offloaded tool output at '{rel_path}', "
                                "but the artifact file does not exist on disk."
                            ),
                            turn_index=idx,
                            details={"artifact_path": rel_path},
                        )
                    )

    if pending_tool_calls:
        issues.append(
            SessionHealthIssue(
                severity="warning",
                code="UNRESOLVED_TOOL_CALLS_AT_END",
                message=(
                    f"Session ends with {len(pending_tool_calls)} pending tool calls "
                    f"({list(pending_tool_calls.keys())}) that have no corresponding tool results."
                ),
                details={"unresolved_call_ids": list(pending_tool_calls.keys())},
            )
        )

    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")

    return SessionHealthReport(
        session_id=session_id,
        healthy=(error_count == 0),
        error_count=error_count,
        warning_count=warning_count,
        issues=issues,
    )
