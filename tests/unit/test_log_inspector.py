"""Unit tests for log inspection, filtering, and application logging setup."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from uclone_x.core.log_inspector import (
    get_default_log_dir,
    get_log_file,
    parse_log_line,
    read_logs,
)
from uclone_x.core.logging_setup import (
    setup_application_logging,
)


def test_log_file_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    with TemporaryDirectory() as tmp:
        monkeypatch.setenv("UCX_LOG_DIR", tmp)
        log_dir = get_default_log_dir()
        assert log_dir == Path(tmp).resolve()

        log_file = get_log_file()
        assert log_file == Path(tmp).resolve() / "ucx.log"


def test_parse_log_line_structured_json() -> None:
    payload = {
        "timestamp": "2026-09-06T12:00:00Z",
        "level": "INFO",
        "logger": "uclone_x.agent.core",
        "message": "Turn complete",
        "session_id": "sess_123",
        "model": "qwen3:8b",
        "latency_ms": 120,
    }
    raw = json.dumps(payload)
    entry = parse_log_line(raw)

    assert entry.timestamp == "2026-09-06T12:00:00Z"
    assert entry.level == "INFO"
    assert entry.logger == "uclone_x.agent.core"
    assert entry.message == "Turn complete"
    assert entry.session_id == "sess_123"
    assert entry.data["model"] == "qwen3:8b"
    assert entry.data["latency_ms"] == 120


def test_parse_log_line_text_format() -> None:
    text = "2026-09-06 12:00:00 [WARN] [uclone_x.tools.web] Upstream slow (session_id=sess_456)"
    entry = parse_log_line(text)

    assert entry.timestamp == "2026-09-06 12:00:00"
    assert entry.level == "WARN"
    assert entry.logger == "uclone_x.tools.web"
    assert "Upstream slow" in entry.message
    assert entry.session_id == "sess_456"


def test_parse_log_line_fallback() -> None:
    raw = "Just a raw unstructured message with session=sess_999"
    entry = parse_log_line(raw)

    assert entry.level == "INFO"
    assert entry.message == raw
    assert entry.session_id == "sess_999"


def test_read_logs_filtering_and_tail() -> None:
    with TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ucx.log"
        lines = [
            json.dumps(
                {
                    "timestamp": "2026-09-06T10:00:00Z",
                    "level": "DEBUG",
                    "logger": "t",
                    "message": "msg 1",
                    "session_id": "sess_a",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2026-09-06T10:01:00Z",
                    "level": "INFO",
                    "logger": "t",
                    "message": "msg 2",
                    "session_id": "sess_b",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2026-09-06T10:02:00Z",
                    "level": "WARN",
                    "logger": "t",
                    "message": "msg 3",
                    "session_id": "sess_a",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2026-09-06T10:03:00Z",
                    "level": "ERROR",
                    "logger": "t",
                    "message": "msg 4",
                    "session_id": "sess_a",
                }
            ),
            json.dumps(
                {
                    "timestamp": "2026-09-06T10:04:00Z",
                    "level": "INFO",
                    "logger": "t",
                    "message": "msg 5",
                    "session_id": "sess_b",
                }
            ),
        ]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Read all
        all_entries = read_logs(log_path, tail=10)
        assert len(all_entries) == 5

        # Filter by level >= WARN
        warn_entries = read_logs(log_path, min_level="WARN")
        assert len(warn_entries) == 2
        assert [e.message for e in warn_entries] == ["msg 3", "msg 4"]

        # Filter by session
        sess_b_entries = read_logs(log_path, session_id="sess_b")
        assert len(sess_b_entries) == 2
        assert [e.message for e in sess_b_entries] == ["msg 2", "msg 5"]

        # Tail limit
        tail_entries = read_logs(log_path, tail=2)
        assert len(tail_entries) == 2
        assert [e.message for e in tail_entries] == ["msg 4", "msg 5"]


def test_read_logs_nonexistent_file() -> None:
    entries = read_logs(Path("/nonexistent/ucx.log"))
    assert entries == []


def test_setup_application_logging() -> None:
    with TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        setup_application_logging(
            log_dir=log_dir, log_level="INFO", enable_console=False, enable_file=True
        )

        test_logger = logging.getLogger("test_logger")
        test_logger.info("Application started")
        test_logger.debug("Should be filtered out")

        log_file = log_dir / "ucx.log"
        assert log_file.is_file()
        content = log_file.read_text(encoding="utf-8")
        assert "Application started" in content
        assert "Should be filtered out" not in content
