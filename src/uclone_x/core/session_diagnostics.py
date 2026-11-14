"""Session diagnostics, inspection, and health auditing for UClone-X.

Provides pure core diagnostic models and functions for:
- Summarizing session records and turn budgets
- Detailed turn-by-turn inspection with role breakdown
- Health checking for orphaned tool results, turn budget saturation,
  unresolved tool calls, and missing offload artifacts.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.session import validate_session_id
from uclone_x.core.session_log import SessionLogEntry, stored_result_entry
from uclone_x.core.session_state import SessionState
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.llm.compactor import estimate_message_tokens
from uclone_x.llm.connections import (
    IMAGE_AUTO,
    ModelRef,
    is_model_ref,
    saved_connections,
    saved_default_models,
)
from uclone_x.llm.models import ChatMessage, MessageRole

logger = logging.getLogger(__name__)

# An advisory threshold on *interaction turns* — how many times a human has gone back and
# forth with the agent — used only to label a session "saturated" in diagnostics output.
# It is deliberately unrelated to `AgentConfig.max_steps`, the P4 ceiling on agent steps
# inside a single turn: nothing here may terminate a conversation, and comparing a turn
# counter against the step ceiling is the anti-pattern recorded in issue 2026-09-05-001.
# See the agent-runtime terminology guide.
DEFAULT_MAX_CONVERSATION_TURNS = 20
DEFAULT_COMPACTION_THRESHOLD_TOKENS = 60_000
SIGNATURE_BLOAT_BYTES_THRESHOLD = 65_536
SLOW_STEP_TIMEOUT_THRESHOLD_SECONDS = 55.0


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
    estimated_tokens: int = 0
    payload_bytes: int = 0
    signature_count: int = 0
    signature_bytes: int = 0


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


@dataclass(frozen=True)
class TurnForensicStep:
    """One model or tool execution step within an interaction turn."""

    step: int
    kind: Literal["model", "tool", "error"]
    name: str
    duration_seconds: float
    success: bool
    summary: str


@dataclass(frozen=True)
class TurnForensicRecord:
    """Correlated execution record for a single interaction turn."""

    turn_index: int
    user_prompt: str
    step_count: int
    total_duration_seconds: float
    max_step_duration_seconds: float
    tool_calls: tuple[str, ...]
    failed_tools: tuple[str, ...]
    stop_reason: str
    error_message: str | None = None
    steps: tuple[TurnForensicStep, ...] = ()


@dataclass(frozen=True)
class SessionForensicReport:
    """End-to-end forensic report combining session state, event timeline, and health."""

    details: SessionDetails
    health: SessionHealthReport
    turns: tuple[TurnForensicRecord, ...]


def measure_context_payload(messages: Sequence[ChatMessage]) -> tuple[int, int, int, int]:
    """Measure active context token estimate, UTF-8 wire bytes, and signature footprint.

    Returns ``(estimated_tokens, payload_bytes, signature_count, signature_bytes)``.
    """
    estimated_tokens = estimate_message_tokens(messages)
    payload_bytes = 0
    signature_count = 0
    signature_bytes = 0
    for msg in messages:
        if msg.content:
            payload_bytes += len(msg.content.encode("utf-8"))
        if msg.name:
            payload_bytes += len(msg.name.encode("utf-8"))
        if msg.tool_call_id:
            payload_bytes += len(msg.tool_call_id.encode("utf-8"))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                payload_bytes += len((tc.id + tc.name).encode("utf-8"))
                if tc.arguments:
                    args_json = json.dumps(unwrap_immutable(tc.arguments), ensure_ascii=False)
                    payload_bytes += len(args_json.encode("utf-8"))
                if tc.signature:
                    sig_len = len(tc.signature.encode("utf-8"))
                    payload_bytes += sig_len
                    signature_count += 1
                    signature_bytes += sig_len
    return estimated_tokens, payload_bytes, signature_count, signature_bytes


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
    state: SessionState, store: SessionStoreProtocol
) -> tuple[int, int, list[str]]:
    """The full tool results this session keeps: how many, their bytes, their handles.

    Each is a session-log entry whose body is the whole text a `tr_` handle names, in the
    session's own body store (#1848); one whose body the store no longer holds is not
    counted. The workspace holds none of them.
    """
    # One pass: the latest entry per handle, as `stored_result_entry` picks it.
    latest: dict[str, tuple[int, SessionLogEntry]] = {}
    for position, entry in enumerate(state.session_log):
        handle = entry.blob
        if handle is not None and stored_result_entry((entry,), handle) is entry:
            latest[handle] = (position, entry)
    handles: list[str] = []
    total_bytes = 0
    for _position, entry in sorted(latest.values(), key=lambda kept: kept[0]):
        if store.load_context_body(state.session_id, entry.digest) is None:
            continue
        handles.append(str(entry.blob))
        total_bytes += entry.size
    return len(handles), total_bytes, handles


def inspect_session(
    session_id: str,
    store: SessionStoreProtocol,
    workspace_root: Path | None = None,
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
) -> SessionDetails | None:
    """Inspect and return detailed diagnostics for a single session.

    Returns None if the session does not exist or cannot be read.
    """
    state = store.load(session_id)
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

    art_count, art_bytes, art_files = _list_session_artifacts(state, store)
    est_tokens, payload_bytes, sig_count, sig_bytes = measure_context_payload(state.messages)

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
        estimated_tokens=est_tokens,
        payload_bytes=payload_bytes,
        signature_count=sig_count,
        signature_bytes=sig_bytes,
    )


def list_session_summaries(
    store: SessionStoreProtocol,
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
    limit: int = 50,
) -> list[SessionSummary]:
    """Enumerate and summarize persisted sessions, sorted by updated_at descending."""
    summaries: list[SessionSummary] = []

    for sid in store.list_session_ids():
        state = store.load(sid)
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


def load_session_events(
    session_id: str,
    store: SessionStoreProtocol,
) -> list[dict[str, Any]]:
    """Load persisted JSONL event records for ``session_id`` from ``<storage_dir>/core/events``."""
    storage_dir = getattr(store, "storage_dir", None)
    if not isinstance(storage_dir, Path):
        return []
    candidates = (
        storage_dir / "core" / "events" / f"{session_id}.jsonl",
        storage_dir.parent / "core" / "events" / f"{session_id}.jsonl",
    )
    for event_file in candidates:
        if not event_file.is_file():
            continue
        records: list[dict[str, Any]] = []
        try:
            text = event_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(cast(dict[str, Any], parsed))
        return records
    return []


def _load_settings_from_disk(
    store: SessionStoreProtocol,
    workspace_root: Path | None,
) -> Mapping[str, Any] | None:
    storage_dir = getattr(store, "storage_dir", None)
    candidates: list[Path] = []
    if isinstance(storage_dir, Path):
        candidates.append(storage_dir / "settings.json")
        candidates.append(storage_dir.parent / "settings.json")
    if workspace_root is not None:
        candidates.append(workspace_root / ".uclone" / "settings.json")
    for path in candidates:
        if path.is_file():
            try:
                parsed: Any = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)
    return None


def _parse_minimal_clone_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data: Any = json.loads(stripped)
            return cast(dict[str, Any], data) if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    out: dict[str, Any] = {}
    for line in raw.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or ":" not in clean:
            continue
        key, _, val = clean.partition(":")
        key_str = key.strip()
        val_str = val.strip().strip("'\"")
        if key_str in ("llm_model", "fast_model", "image_model", "model", "picture_model"):
            out[key_str] = val_str
    return out


def _load_clone_configs_from_disk(
    store: SessionStoreProtocol,
    workspace_root: Path | None,
    agent_id: str,
) -> dict[str, Mapping[str, Any]]:
    storage_dir = getattr(store, "storage_dir", None)
    candidates: list[Path] = []
    if isinstance(storage_dir, Path):
        candidates.append(storage_dir / "agents" / agent_id / "clone.yaml")
        candidates.append(storage_dir.parent / "agents" / agent_id / "clone.yaml")
    if workspace_root is not None:
        candidates.append(workspace_root / ".uclone" / "agents" / agent_id / "clone.yaml")
    for path in candidates:
        if path.is_file():
            cfg = _parse_minimal_clone_config(path)
            if cfg:
                return {agent_id: cfg}
    return {}


def _event_payload(ev: Mapping[str, Any]) -> dict[str, Any]:
    raw_payload: Any = ev.get("payload")
    if isinstance(raw_payload, dict):
        return cast(dict[str, Any], raw_payload)
    return dict(ev)


def _extract_error_text(err_field: object) -> str:
    if isinstance(err_field, str):
        return err_field
    if isinstance(err_field, Mapping):
        err_map = cast(Mapping[str, Any], err_field)
        return str(err_map.get("message") or err_map.get("type") or "")
    return ""


def _segment_turns(
    events: Sequence[dict[str, Any]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    segments: list[tuple[int, list[dict[str, Any]]]] = []
    current_turn_idx: int | None = None
    current_bucket: list[dict[str, Any]] = []
    for ev in events:
        ev_type = str(ev.get("type", ""))
        ev_turn = ev.get("turn_index")
        if ev_type == "TURN_START":
            if current_bucket and current_turn_idx is not None:
                segments.append((current_turn_idx, current_bucket))
            current_turn_idx = ev_turn if isinstance(ev_turn, int) else (current_turn_idx or 0)
            current_bucket = [ev]
        elif isinstance(ev_turn, int):
            if current_turn_idx != ev_turn:
                if current_bucket and current_turn_idx is not None:
                    segments.append((current_turn_idx, current_bucket))
                current_turn_idx = ev_turn
                current_bucket = [ev]
            else:
                current_bucket.append(ev)
        elif current_turn_idx is not None:
            current_bucket.append(ev)
    if current_bucket and current_turn_idx is not None:
        segments.append((current_turn_idx, current_bucket))
    return segments


def _step_duration_seconds(payload: Mapping[str, Any]) -> float:
    explicit = payload.get("duration_seconds")
    if isinstance(explicit, (int, float)):
        return max(0.0, float(explicit))
    duration_ms = payload.get("duration_ms")
    if isinstance(duration_ms, (int, float)):
        return max(0.0, float(duration_ms) / 1000.0)
    started = payload.get("started_at")
    ended = payload.get("ended_at")
    if isinstance(started, str) and isinstance(ended, str) and started and ended:
        try:
            dt_start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            dt_end = datetime.fromisoformat(ended.replace("Z", "+00:00"))
            return max(0.0, (dt_end - dt_start).total_seconds())
        except ValueError:
            return 0.0
    return 0.0


_CONFIG_REFUSAL_MARKERS = (
    "cannot be used",
    "there is no connection called",
    "no connection",
    "Settings › Models",
)

_TIMEOUT_ERROR_MARKERS = (
    "couldn't get an answer",
    "couldn't be reached",
    "readtimeout",
    "timed out",
)

_MISSING_CONN_REF_RE = re.compile(
    r"model\s+([A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+)\s+cannot be used:\s+there is no connection called"
)


def _resolve_tool_event_error_text(
    payload: Mapping[str, Any],
    state: SessionState | None = None,
    store: SessionStoreProtocol | None = None,
) -> str:
    err_text = _extract_error_text(payload.get("error")) or str(payload.get("output") or "")
    if err_text.strip():
        return err_text.strip()
    handle = payload.get("result_handle")
    if isinstance(handle, str) and handle and state is not None and store is not None:
        entry = stored_result_entry(state.session_log, handle)
        if entry is not None:
            body = store.load_context_body(state.session_id, entry.digest)
            if isinstance(body, str) and body.strip():
                return body.strip()
    return ""


def _check_behavioral_anomalies(
    state: SessionState,
    events: Sequence[dict[str, Any]],
    settings_data: Mapping[str, Any] | None,
    clone_configs: Mapping[str, Mapping[str, Any]] | None,
    compaction_threshold_tokens: int,
    store: SessionStoreProtocol | None = None,
) -> list[SessionHealthIssue]:
    anomalies: list[SessionHealthIssue] = []
    observed_error_texts: list[str] = []

    # Rule A: CONTEXT_PAYLOAD_BLOAT
    est_tokens, payload_bytes, sig_count, sig_bytes = measure_context_payload(state.messages)
    if est_tokens >= compaction_threshold_tokens or sig_bytes >= SIGNATURE_BLOAT_BYTES_THRESHOLD:
        anomalies.append(
            SessionHealthIssue(
                severity="warning",
                code="CONTEXT_PAYLOAD_BLOAT",
                message=(
                    f"Active context payload is bloated ({est_tokens} estimated tokens, "
                    f"{payload_bytes} bytes; {sig_count} thoughtSignature(s) totaling "
                    f"{sig_bytes} bytes against {compaction_threshold_tokens}-token threshold)."
                ),
                details={
                    "estimated_tokens": est_tokens,
                    "payload_bytes": payload_bytes,
                    "signature_count": sig_count,
                    "signature_bytes": sig_bytes,
                    "compaction_threshold_tokens": compaction_threshold_tokens,
                },
            )
        )

    # Segment events sequentially by turn for Rules B & C
    for turn_idx, turn_events in _segment_turns(events):
        tool_failures: dict[str, int] = {}
        refusal_tools: list[str] = []
        called_install_package = False
        slow_steps: list[tuple[int, float, str]] = []
        turn_error_texts: list[str] = []
        active_step = 0

        for ev in turn_events:
            ev_type = str(ev.get("type", ""))
            payload = _event_payload(ev)
            if isinstance(payload.get("step"), int):
                active_step = int(payload["step"])
            if ev_type == "TOOL_CALL":
                if str(payload.get("name") or "") == "install_package":
                    called_install_package = True
            elif ev_type in ("TOOL_EXECUTION", "TOOL_RESULT"):
                t_name = str(payload.get("tool_name") or payload.get("name") or "unknown")
                if t_name == "install_package":
                    called_install_package = True
                failed = (
                    payload.get("success") is False
                    or payload.get("status") == "error"
                    or payload.get("outcome") == "errored"
                )
                if failed:
                    tool_failures[t_name] = tool_failures.get(t_name, 0) + 1
                    err_text = _resolve_tool_event_error_text(payload, state, store)
                    if err_text:
                        observed_error_texts.append(err_text)
                    if any(marker in err_text for marker in _CONFIG_REFUSAL_MARKERS) or (
                        t_name != "install_package" and bool(payload.get("result_handle"))
                    ):
                        refusal_tools.append(t_name)
            elif ev_type == "MODEL_RESPONSE":
                for tc in payload.get("tool_calls") or ():
                    if (
                        isinstance(tc, dict)
                        and cast(dict[str, Any], tc).get("name") == "install_package"
                    ):
                        called_install_package = True
                dur = _step_duration_seconds(payload)
                step_no = int(payload.get("step") or active_step)
                m_name = str(payload.get("model_name") or "unknown")
                if dur >= SLOW_STEP_TIMEOUT_THRESHOLD_SECONDS:
                    slow_steps.append((step_no, dur, m_name))
                err_text = _extract_error_text(payload.get("error"))
                if err_text:
                    turn_error_texts.append(err_text)
                    observed_error_texts.append(err_text)
            elif ev_type == "ERROR":
                err_msg = _extract_error_text(payload.get("error")) or str(
                    payload.get("message") or ""
                )
                if err_msg:
                    turn_error_texts.append(err_msg)
                    observed_error_texts.append(err_msg)
                dur = _step_duration_seconds(payload)
                if dur >= SLOW_STEP_TIMEOUT_THRESHOLD_SECONDS:
                    step_no = int(payload.get("step") or active_step)
                    m_name = str(payload.get("model_name") or "unknown")
                    slow_steps.append((step_no, dur, m_name))

        # Rule B: TOOL_RETRY_SPIRAL
        for t_name, fail_count in tool_failures.items():
            misrouted = called_install_package and t_name in refusal_tools
            if fail_count >= 2 or misrouted:
                anomalies.append(
                    SessionHealthIssue(
                        severity="warning",
                        code="TOOL_RETRY_SPIRAL",
                        turn_index=turn_idx,
                        message=(
                            f"Turn #{turn_idx} exhibited a tool retry/recovery spiral on "
                            f"'{t_name}' ({fail_count} failure(s)"
                            + (", followed by misrouted install_package)" if misrouted else ")")
                            + "."
                        ),
                        details={
                            "tool_name": t_name,
                            "failure_count": fail_count,
                            "misrouted_install_package": misrouted,
                        },
                    )
                )

        # Rule C: READ_TIMEOUT_MASKED
        matched_errors = [
            txt
            for txt in turn_error_texts
            if any(marker in txt.lower() for marker in _TIMEOUT_ERROR_MARKERS)
        ]
        if slow_steps and matched_errors:
            step_no, dur, m_name = max(slow_steps, key=lambda item: item[1])
            anomalies.append(
                SessionHealthIssue(
                    severity="warning",
                    code="READ_TIMEOUT_MASKED",
                    turn_index=turn_idx,
                    message=(
                        f"Turn #{turn_idx} step #{step_no} ran for {dur:.2f}s before failing "
                        f"with an unreachable/timeout error on model '{m_name}' "
                        "(likely a read timeout during thinking or large tool-call streaming)."
                    ),
                    details={
                        "step": step_no,
                        "duration_seconds": round(dur, 2),
                        "model_name": m_name,
                        "error": matched_errors[0],
                    },
                )
            )

    # Rule D: DANGLING_MODEL_REF
    if isinstance(settings_data, Mapping) and "connections" in settings_data:
        conn_ids = {c.id for c in saved_connections(settings_data)}
        candidate_refs: list[tuple[str, str]] = []
        defaults = saved_default_models(settings_data)
        for slot, ref_val in (
            ("deep", defaults.deep),
            ("fast", defaults.fast),
            ("image", defaults.image),
        ):
            if isinstance(ref_val, str) and ref_val.strip():
                candidate_refs.append((f"default_models.{slot}", ref_val.strip()))
        if clone_configs:
            agent_cfg = clone_configs.get(state.agent_id)
            if isinstance(agent_cfg, Mapping):
                for key in ("llm_model", "fast_model", "image_model", "model", "picture_model"):
                    val = agent_cfg.get(key)
                    if isinstance(val, str) and val.strip():
                        candidate_refs.append((f"clone.{state.agent_id}.{key}", val.strip()))
        if store is not None:
            for entry in state.session_log:
                body = store.load_context_body(state.session_id, entry.digest)
                if isinstance(body, str) and body:
                    observed_error_texts.append(body)
        for msg in state.messages:
            if msg.content:
                observed_error_texts.append(msg.content)
        for txt in observed_error_texts:
            for match in _MISSING_CONN_REF_RE.finditer(txt):
                candidate_refs.append(("session_history", match.group(1)))
        seen_refs: set[str] = set()
        for slot_label, ref_str in candidate_refs:
            if ref_str == IMAGE_AUTO or not is_model_ref(ref_str) or ref_str in seen_refs:
                continue
            seen_refs.add(ref_str)
            parsed_ref = ModelRef.parse(ref_str)
            if parsed_ref.connection_id not in conn_ids:
                anomalies.append(
                    SessionHealthIssue(
                        severity="warning",
                        code="DANGLING_MODEL_REF",
                        message=(
                            f"Model reference '{ref_str}' in '{slot_label}' points to "
                            f"missing connection '{parsed_ref.connection_id}'."
                        ),
                        details={
                            "slot": slot_label,
                            "ref": ref_str,
                            "missing_connection": parsed_ref.connection_id,
                        },
                    )
                )

    return anomalies


def check_session_health(
    session_id: str,
    store: SessionStoreProtocol,
    workspace_root: Path | None = None,
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
    *,
    compaction_threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS,
    events: Sequence[dict[str, Any]] | None = None,
    settings_data: Mapping[str, Any] | None = None,
    clone_configs: Mapping[str, Mapping[str, Any]] | None = None,
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
    5. Stored tool result integrity: the full text a `tr_` handle names is in the
       session's store.
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
    state = store.load(session_id)
    if state is None:
        # `storage_dir` is the file store's, not the protocol's: a store with no directory
        # is reported by its type rather than by a path it does not have.
        storage_dir = getattr(store, "storage_dir", type(store).__name__)
        issues.append(
            SessionHealthIssue(
                severity="error",
                code="SESSION_NOT_FOUND",
                message=f"Session '{session_id}' not found in store at '{storage_dir}'.",
                details={"session_id": session_id, "storage_dir": str(storage_dir)},
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
        # A form is recorded with no text, which is rendered when it is shown (#1848).
        if not has_content and not has_calls and msg.rendered_from is None:
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

        # 5. A stored tool result (#1422): an excerpt or compaction stub records the
        # `tr_` handle it is rendered from, whose full text is a body of the session's own
        # log (#1848). The handle is read from that record, never from the text: output
        # that quotes a stored-result header is its own full text and needs no kept result.
        handle: str | None = msg.rendered_from.handle if msg.rendered_from is not None else None
        if handle is not None:
            entry = stored_result_entry(state.session_log, handle)
            if entry is None or store.load_context_body(session_id, entry.digest) is None:
                issues.append(
                    SessionHealthIssue(
                        severity="warning",
                        code="MISSING_OFFLOAD_ARTIFACT",
                        message=(
                            f"Turn #{idx} names stored tool result '{handle}', but this "
                            "session no longer holds its full text."
                        ),
                        turn_index=idx,
                        details={"handle": handle},
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

    resolved_events = list(events) if events is not None else load_session_events(session_id, store)
    resolved_settings = (
        settings_data
        if settings_data is not None
        else _load_settings_from_disk(store, workspace_root)
    )
    resolved_clones = (
        clone_configs
        if clone_configs is not None
        else _load_clone_configs_from_disk(store, workspace_root, state.agent_id)
    )
    issues.extend(
        _check_behavioral_anomalies(
            state=state,
            events=resolved_events,
            settings_data=resolved_settings,
            clone_configs=resolved_clones,
            compaction_threshold_tokens=compaction_threshold_tokens,
            store=store,
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


def diagnose_session_forensics(
    session_id: str,
    store: SessionStoreProtocol,
    workspace_root: Path | None = None,
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
    tail_turns: int = 5,
    *,
    compaction_threshold_tokens: int = DEFAULT_COMPACTION_THRESHOLD_TOKENS,
    events: Sequence[dict[str, Any]] | None = None,
    settings_data: Mapping[str, Any] | None = None,
    clone_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> SessionForensicReport | None:
    """Correlate session messages, persisted event timeline, and health anomalies."""
    details = inspect_session(
        session_id=session_id,
        store=store,
        workspace_root=workspace_root,
        max_conversation_turns=max_conversation_turns,
    )
    if details is None:
        return None

    state = store.load(session_id)
    resolved_events = list(events) if events is not None else load_session_events(session_id, store)
    health = check_session_health(
        session_id=session_id,
        store=store,
        workspace_root=workspace_root,
        max_conversation_turns=max_conversation_turns,
        compaction_threshold_tokens=compaction_threshold_tokens,
        events=resolved_events,
        settings_data=settings_data,
        clone_configs=clone_configs,
    )

    turn_records: list[TurnForensicRecord] = []
    for t_idx, evs in _segment_turns(resolved_events):
        user_prompt = ""
        step_records: list[TurnForensicStep] = []
        tool_names: list[str] = []
        failed_tools: list[str] = []
        stop_reason = "completed"
        error_message: str | None = None
        active_step = 0

        for ev in evs:
            ev_type = str(ev.get("type", ""))
            payload = _event_payload(ev)
            if isinstance(payload.get("step"), int):
                active_step = int(payload["step"])
            if ev_type in ("USER_INPUT", "USER_MESSAGE") and not user_prompt:
                raw_content = str(payload.get("content") or "").strip()
                lines = [ln.strip() for ln in raw_content.splitlines() if ln.strip()]
                user_prompt = (lines[-1] if lines else "")[:120]
            elif ev_type == "MODEL_RESPONSE":
                dur = _step_duration_seconds(payload)
                step_no = int(payload.get("step") or active_step or (len(step_records) + 1))
                m_name = str(payload.get("model_name") or "model")
                t_calls = payload.get("tool_calls") or ()
                call_names = [
                    str(cast(dict[str, Any], tc).get("name"))
                    for tc in t_calls
                    if isinstance(tc, dict) and cast(dict[str, Any], tc).get("name")
                ]
                err_txt = _extract_error_text(payload.get("error"))
                ok = not bool(err_txt)
                if err_txt:
                    error_message = err_txt
                    stop_reason = "error"
                    summary = err_txt[:100]
                elif call_names:
                    summary = f"calls: {', '.join(call_names)}"
                else:
                    summary = str(payload.get("finish_reason") or "stop")
                step_records.append(
                    TurnForensicStep(
                        step=step_no,
                        kind="model",
                        name=m_name,
                        duration_seconds=round(dur, 2),
                        success=ok,
                        summary=summary,
                    )
                )
            elif ev_type in ("TOOL_EXECUTION", "TOOL_RESULT"):
                dur = _step_duration_seconds(payload)
                step_no = int(payload.get("step") or active_step or (len(step_records) + 1))
                t_name = str(payload.get("tool_name") or payload.get("name") or "tool")
                ok = not (
                    payload.get("success") is False
                    or payload.get("status") == "error"
                    or payload.get("outcome") == "errored"
                )
                tool_names.append(t_name)
                if not ok:
                    failed_tools.append(t_name)
                err_txt = _resolve_tool_event_error_text(payload, state, store)
                step_records.append(
                    TurnForensicStep(
                        step=step_no,
                        kind="tool",
                        name=t_name,
                        duration_seconds=round(dur, 2),
                        success=ok,
                        summary="ok" if ok else (err_txt[:100] or "failed"),
                    )
                )
            elif ev_type == "STOP_SIGNAL":
                stop_reason = str(payload.get("reason") or "stopped")
            elif ev_type == "TURN_END":
                outcome = str(payload.get("outcome") or "")
                end_reason = str(payload.get("stop_reason") or "")
                if outcome == "error" or error_message:
                    stop_reason = end_reason or "error"
                elif end_reason:
                    stop_reason = end_reason
            elif ev_type == "TURN_ROLLED_BACK":
                rb_reason = str(payload.get("reason") or "error")
                stop_reason = f"rolled_back ({rb_reason})"
            elif ev_type == "ERROR":
                dur = _step_duration_seconds(payload)
                step_no = int(payload.get("step") or active_step or (len(step_records) + 1))
                err_msg = _extract_error_text(payload.get("error")) or str(
                    payload.get("message") or "error"
                )
                error_message = err_msg
                stop_reason = "error"
                step_records.append(
                    TurnForensicStep(
                        step=step_no,
                        kind="error",
                        name=str(payload.get("error_type") or "ERROR"),
                        duration_seconds=round(dur, 2),
                        success=False,
                        summary=err_msg[:120],
                    )
                )

        total_dur = round(sum(s.duration_seconds for s in step_records), 2)
        max_dur = round(max((s.duration_seconds for s in step_records), default=0.0), 2)
        turn_records.append(
            TurnForensicRecord(
                turn_index=t_idx,
                user_prompt=user_prompt,
                step_count=len(step_records),
                total_duration_seconds=total_dur,
                max_step_duration_seconds=max_dur,
                tool_calls=tuple(tool_names),
                failed_tools=tuple(failed_tools),
                stop_reason=stop_reason,
                error_message=error_message,
                steps=tuple(step_records),
            )
        )

    if tail_turns > 0 and len(turn_records) > tail_turns:
        turn_records = turn_records[-tail_turns:]

    return SessionForensicReport(
        details=details,
        health=health,
        turns=tuple(turn_records),
    )
