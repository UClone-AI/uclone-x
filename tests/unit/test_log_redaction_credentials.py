"""Tests for write-time credential redaction in append-only log, session state, and telemetry (#569).

Option A: Redaction on write prevents permanent credential retention in durable
records, addressing security threat model T3 where an append-only log gives P0's
user no remedy for pasted or emitted secrets.

Known limitation: Pattern-based redaction catches known credential shapes (e.g.,
`sk-...`, `ghp_...`, `AKIA...`, private keys, `Bearer ...`). It cannot distinguish
an arbitrary high-entropy string from legitimate data, nor detect structured secrets
in formats without unique prefixes or recognizable assignment keys.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from uclone_x.agent.session import (
    SessionState,
    SessionStore,
    redact_message,
)
from uclone_x.core.log_writer import RedactingLogWriter, redact_log_payload
from uclone_x.core.logging_setup import JsonLogFormatter, RedactingConsoleFormatter
from uclone_x.core.secrets import (
    REDACTED_PLACEHOLDER,
    contains_credential,
    redact_credentials,
)
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest
from uclone_x.telemetry.exporter import InMemoryTelemetryExporter
from uclone_x.telemetry.models import SpanRecord

SAMPLE_CREDENTIALS = [
    # OpenAI legacy and project keys
    ("sk-1234567890123456789012345678901234567890", "[REDACTED]"),
    ("sk-proj-1234567890123456789012345678901234567890", "[REDACTED]"),
    ("sk-admin-1234567890123456789012345678901234567890", "[REDACTED]"),
    # Anthropic keys
    ("sk-ant-api03-1234567890123456789012345678901234567890", "[REDACTED]"),
    # GitHub personal access tokens
    ("ghp_123456789012345678901234567890123456", "[REDACTED]"),
    (
        "github_pat_11ABCD123_4567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        "[REDACTED]",
    ),
    # AWS Access Key ID
    ("AKIAIOSFODNN7EXAMPLE", "[REDACTED]"),
    # Bearer tokens
    (
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDcSemACt8x4iTMCda8Yhe3iZaWbvV5XKSTbuAn0M",
        "Bearer [REDACTED]",
    ),
    # Slack tokens
    ("xoxb-123456789012-1234567890123-abcdefghijklmnopqrstuvwx", "[REDACTED]"),
    # Generic secret assignment
    ("api_key: super_secret_pass_12345", "api_key: [REDACTED]"),
    ("token = secret_token_xyz_987654", "token = [REDACTED]"),
]


class TestCredentialPatternRedaction:
    """Test suite for credential regex detection and redaction functions."""

    @pytest.mark.parametrize("secret,expected", SAMPLE_CREDENTIALS)
    def test_contains_and_redact_credentials(self, secret: str, expected: str) -> None:
        """Verify credential detection and masking for various vendor credential patterns.

        Killed by: src/uclone_x/core/secrets.py :: def contains_credential(text: str) -> bool:
        """
        assert contains_credential(secret)
        assert REDACTED_PLACEHOLDER == "[REDACTED]"
        redacted = redact_credentials(f"Prefix {secret} suffix")
        assert expected in redacted
        assert secret not in redacted

    def test_redact_private_key_pem(self) -> None:
        """Verify PEM private key blocks are replaced with a dedicated redaction marker.

        Killed by: src/uclone_x/core/secrets.py :: REDACTED_PLACEHOLDER: str = "[REDACTED]"
        """
        pem = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Y123456789abcdefghijklmnopqrstuvwxyz\n"
            "-----END RSA PRIVATE KEY-----"
        )
        assert contains_credential(pem)
        redacted = redact_credentials(f"Config:\n{pem}\nEnd")
        assert "[REDACTED PRIVATE KEY]" in redacted
        assert "MIIEowIBAAKCAQEA0Y123456789" not in redacted

    def test_benign_text_unmodified(self) -> None:
        """Verify regular conversational text is preserved without false-positive redaction.

        Killed by: src/uclone_x/core/secrets.py :: def redact_credentials(text: str, placeholder: str = REDACTED_PLACEHOLDER) -> str:
        """
        text = "Hello world! This is a benign message without any tokens or keys."
        assert not contains_credential(text)
        assert redact_credentials(text) == text

    def test_known_limitation_unstructured_high_entropy_string(self) -> None:
        """Verify known limitation: unstructured high-entropy strings without vendor prefix are not redacted.

        Documented in threat model T3: pattern-based redaction catches known shapes;
        arbitrary random strings cannot be distinguished from valid data (such as commit SHAs or base64 data).
        """
        random_entropy = "c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2"
        # Not a known vendor prefix or secret key assignment:
        redacted = redact_credentials(random_entropy)
        assert redacted == random_entropy


class TestRedactingLogWriter:
    """Test suite for RedactingLogWriter write-time log sanitization."""

    def test_log_writer_write_line_redacts_credentials(self) -> None:
        """Writing lines to RedactingLogWriter replaces credential strings on write.

        Killed by: src/uclone_x/core/log_writer.py :: class RedactingLogWriter:
        """
        buf = io.StringIO()
        writer = RedactingLogWriter(dest=buf)
        writer.write_line("User pasted OpenAI key: sk-proj-123456789012345678901234567890")
        output = buf.getvalue()
        assert "sk-proj-" not in output
        assert "[REDACTED]" in output
        assert output.endswith("\n")

    def test_log_writer_write_entry_dict_payload(self) -> None:
        """Structured dictionary entries are recursively redacted on write.

        Killed by: src/uclone_x/core/log_writer.py :: def redact_log_payload(payload: Any, placeholder: str = REDACTED_PLACEHOLDER) -> Any:
        """
        buf = io.StringIO()
        writer = RedactingLogWriter(dest=buf)
        entry = {
            "event": "tool_executed",
            "details": {
                "auth": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.do_not_leak",
                "nested_list": ["ok", "ghp_0123456789abcdefghijklmnopqrstuvwxyz"],
            },
        }
        writer.write_entry(entry)
        logged = json.loads(buf.getvalue())
        assert logged["details"]["auth"] == "Bearer [REDACTED]"
        assert logged["details"]["nested_list"][1] == "[REDACTED]"
        assert "ghp_" not in buf.getvalue()

    def test_redact_log_payload_direct(self) -> None:
        """Direct verification of recursive payload redaction function."""
        data = ["secret: sk-proj-123456789012345678901234567890", 42, None]
        cleaned = redact_log_payload(data)
        assert isinstance(cleaned, list)
        assert cleaned[0] == "secret: [REDACTED]"
        assert cleaned[1] == 42
        assert cleaned[2] is None

    def test_log_writer_file_destination(self, tmp_path: Path) -> None:
        """Writing to file destination creates parent dirs and appends redacted content."""
        log_file = tmp_path / "logs" / "session.jsonl"
        writer = RedactingLogWriter(dest=log_file)
        writer.write_line("Secret: AKIAIOSFODNN7EXAMPLE")
        content = log_file.read_text(encoding="utf-8")
        assert "AKIAIOSFODNN7EXAMPLE" not in content
        assert "[REDACTED]" in content


class TestLoggingSetupFormatters:
    """Test suite for JsonLogFormatter and RedactingConsoleFormatter."""

    def test_console_formatter_redacts_credentials(self) -> None:
        """Console formatter masks credentials in record message.

        Killed by: src/uclone_x/core/logging_setup.py :: class RedactingConsoleFormatter(logging.Formatter):
        """
        formatter = RedactingConsoleFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Connected with token: ghp_123456789012345678901234567890123456",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "ghp_" not in formatted
        assert "[REDACTED]" in formatted

    def test_json_log_formatter_redacts_credentials_and_exc_info(self) -> None:
        """JsonLogFormatter redacts credential shapes from message and exception info."""
        formatter = JsonLogFormatter()
        try:
            raise ValueError("Failed connecting with sk-ant-api03-123456789012345678901234567890")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=20,
                msg="Error occurred with key sk-proj-123456789012345678901234567890",
                args=(),
                exc_info=exc_info,
            )
            formatted = formatter.format(record)
            parsed = json.loads(formatted)
            assert "sk-proj-" not in str(parsed["message"])
            assert "[REDACTED]" in str(parsed["message"])
            assert "sk-ant-api03" not in str(parsed.get("exception"))
            assert "[REDACTED]" in str(parsed.get("exception"))


class TestSessionStateCredentialRedaction:
    """Test suite for SessionState and SessionStore write-time redaction."""

    def test_redact_message_masks_content_and_arguments(self) -> None:
        """redact_message sanitizes content and tool call arguments.

        Killed by: src/uclone_x/agent/session.py :: def redact_message(message: ChatMessage) -> ChatMessage:
        """
        msg = ChatMessage(
            role=MessageRole.USER,
            content="My Anthropic key is sk-ant-api03-123456789012345678901234567890",
            tool_calls=(
                ToolCallRequest(
                    id="call_1",
                    name="set_env",
                    arguments={"api_key": "sk-proj-123456789012345678901234567890"},
                ),
            ),
        )
        redacted = redact_message(msg)
        assert "sk-ant-api03" not in str(redacted.content)
        assert "[REDACTED]" in str(redacted.content)
        assert redacted.tool_calls is not None
        assert "sk-proj-" not in str(redacted.tool_calls[0].arguments)
        assert "[REDACTED]" in str(redacted.tool_calls[0].arguments)

    def test_session_state_creation_redacts_messages(self) -> None:
        """SessionState validator strips credentials on construction.

        Killed by: src/uclone_x/agent/session.py :: def _redact_messages(cls, messages: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
        """
        state = SessionState(
            session_id="test_sess",
            agent_id="test_agent",
            messages=(
                ChatMessage(
                    role=MessageRole.USER,
                    content="Here is my AWS key: AKIAIOSFODNN7EXAMPLE",
                ),
            ),
        )
        assert "AKIAIOSFODNN7EXAMPLE" not in str(state.messages[0].content)
        assert "[REDACTED]" in str(state.messages[0].content)

    def test_session_state_append_message(self) -> None:
        """SessionState.append_message applies redaction on write."""
        state = SessionState(session_id="test_sess", agent_id="test_agent")
        new_state = state.append_message(
            ChatMessage(
                role=MessageRole.USER,
                content="Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.token",
            )
        )
        assert len(new_state.messages) == 1
        assert "Bearer [REDACTED]" in str(new_state.messages[0].content)

    def test_session_store_save_persists_redacted_json(self, tmp_path: Path) -> None:
        """SessionStore.save guarantees on-disk json files contain no unmasked credentials."""
        store = SessionStore(storage_dir=tmp_path)
        state = SessionState(
            session_id="session_with_secret",
            agent_id="agent_1",
            messages=(
                ChatMessage(
                    role=MessageRole.USER,
                    content="Use token ghp_123456789012345678901234567890123456",
                ),
            ),
        )
        saved_state = store.save(state)
        assert saved_state.revision == 1

        record_file = store.session_path("session_with_secret")
        raw_json = record_file.read_text(encoding="utf-8")
        assert "ghp_123456789012345678901234567890123456" not in raw_json
        assert "[REDACTED]" in raw_json


class TestCompactorCredentialRedaction:
    """Test suite for compaction and tool offloading credential redaction."""

    def test_compactor_prune_and_offload_redacts_credentials(self, tmp_path: Path) -> None:
        """Compactor offloading and truncation redact credentials before saving to disk.

        Killed by: src/uclone_x/llm/compactor.py :: redacted_content = redact_credentials(msg.content)
        """
        compactor = ContextCompactor(
            workspace_root=tmp_path,
            max_tool_output_chars=80,
            session_id="test_compactor_sess",
        )
        tool_content = (
            "Execution output:\n"
            + "sk-proj-1234567890123456789012345678901234567890\n"
            + "Remaining padding lines to exceed character threshold.\n" * 5
        )
        msg = ChatMessage(
            role=MessageRole.TOOL,
            name="fetch_credentials",
            tool_call_id="call_99",
            content=tool_content,
        )
        pruned = compactor.prune_tool_message(msg)
        assert pruned.content is not None
        assert "sk-proj-" not in pruned.content
        assert "[REDACTED]" in pruned.content

        # Inspect the offloaded file on disk:
        offloaded_files = list(
            (tmp_path / compactor.artifact_subdir / "test_compactor_sess").glob("*.txt")
        )
        assert len(offloaded_files) == 1
        saved_text = offloaded_files[0].read_text(encoding="utf-8")
        assert "sk-proj-" not in saved_text
        assert "[REDACTED]" in saved_text

    @pytest.mark.asyncio
    async def test_heuristic_ledger_redaction(self) -> None:
        """Compactor heuristic ledger does not expose credentials summarized from dialogue turns."""
        compactor = ContextCompactor(keep_recent_turns=1)
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content="Please use my key sk-ant-api03-123456789012345678901234567890 for setup",
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Configured with key sk-ant-api03-123456789012345678901234567890 successfully",
            ),
            ChatMessage(
                role=MessageRole.USER,
                content="Recent turn",
            ),
        ]
        outcome = await compactor.compact(messages)
        # Find the compaction ledger in output messages
        ledger_msgs = [m for m in outcome.messages if m.compaction_ledger]
        assert len(ledger_msgs) == 1
        ledger_content = ledger_msgs[0].content or ""
        assert "sk-ant-api03" not in ledger_content
        assert "[REDACTED]" in ledger_content


class TestTelemetryCredentialRedaction:
    """Test suite for telemetry span exporter credential redaction."""

    @pytest.mark.asyncio
    async def test_telemetry_exporter_redacts_credentials_in_attributes(self) -> None:
        """Telemetry exporter sanitizes string attributes containing credentials.

        Killed by: src/uclone_x/telemetry/exporter.py :: return redact_credentials(value)
        """
        exporter = InMemoryTelemetryExporter(redact_secrets=True)
        span = SpanRecord(
            trace_id="t1",
            span_id="s1",
            name="test_span",
            attributes={
                "custom_desc": "User bearer token Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWRtaW4ifQ.signature_part",
                "custom_key": "sk-1234567890123456789012345678901234567890",
            },
        )
        await exporter.export_spans((span,))
        assert exporter.span_count == 1
        exported_span = exporter._spans[0]  # pyright: ignore[reportPrivateUsage]
        assert "sk-1234567890" not in str(exported_span.attributes.get("custom_key"))
        assert "[REDACTED]" in str(exported_span.attributes.get("custom_key"))
        assert "Bearer [REDACTED]" in str(exported_span.attributes.get("custom_desc"))
