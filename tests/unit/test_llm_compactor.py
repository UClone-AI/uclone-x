"""Unit tests for TokenBudgetManager and ContextCompactor (Principle 5 & Principle 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.core.provenance import ExecutionPath, Provenance
from uclone_x.errors import BudgetExceededError, UnmappableChatMessageError
from uclone_x.llm import (
    ChatMessage,
    ContextCompactor,
    LLMRequest,
    MessageRole,
    ModelResponse,
    TokenBudgetManager,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.llm.compactor import (
    estimate_message_tokens,
    estimate_reply_tokens,
    estimate_request_tokens,
    estimate_text_tokens,
    resolve_model_context_limit,
)
from uclone_x.llm.models import CompactionOutcome, FinishReason, LedgerSource
from uclone_x.llm.protocols import ContextCompactorProtocol, LLMProviderProtocol

# ======================================================================================
# TokenBudgetManager Tests
# ======================================================================================


def test_token_budget_manager_initial_state() -> None:
    manager = TokenBudgetManager(default_max_tokens=500_000)

    # Initial check before any session usage
    decision = manager.check_budget("session-1")
    assert decision.allowed is True
    assert decision.remaining_tokens == 500_000

    budget = manager.get_budget("session-1")
    assert budget is not None
    assert budget.max_tokens == 500_000
    assert budget.used_input_tokens == 0
    assert budget.used_output_tokens == 0


def test_token_budget_manager_record_usage_and_provider_attribution() -> None:
    manager = TokenBudgetManager(default_max_tokens=100_000)

    # Turn 1: OpenAI usage
    usage_openai = TokenUsage(
        provider="openai",
        model="gpt-4o",
        input_tokens=10_000,
        output_tokens=2_000,
        total_tokens=12_000,
    )
    manager.record_usage("session-1", usage_openai)

    # Turn 2: Anthropic usage
    usage_anthropic = TokenUsage(
        provider="anthropic",
        model="claude-3-5-sonnet",
        input_tokens=20_000,
        output_tokens=5_000,
        total_tokens=25_000,
    )
    manager.record_usage("session-1", usage_anthropic)

    budget = manager.get_budget("session-1")
    assert budget is not None
    assert budget.used_input_tokens == 30_000
    assert budget.used_output_tokens == 7_000
    assert budget.per_provider["openai"] == 12_000
    assert budget.per_provider["anthropic"] == 25_000

    history = manager.get_turn_history("session-1")
    assert len(history) == 2
    assert history[0].provider == "openai"
    assert history[1].provider == "anthropic"


def test_token_budget_manager_exceed_tokens() -> None:
    manager = TokenBudgetManager(default_max_tokens=10_000)

    manager.record_usage(
        "session-1",
        TokenUsage(
            provider="ollama",
            input_tokens=8_000,
            output_tokens=3_000,
            total_tokens=11_000,
        ),
    )

    decision = manager.check_budget("session-1")
    assert decision.allowed is False
    assert decision.remaining_tokens == 0
    assert "token limit exceeded" in (decision.reason or "").lower()

    # Principle 6: Fail-fast immediate enforcement
    with pytest.raises(BudgetExceededError, match="token limit exceeded"):
        manager.enforce_budget("session-1")


def test_token_budget_manager_provider_specific_quota() -> None:
    manager = TokenBudgetManager(default_max_tokens=1_000_000)

    # Configure session with a 50k-token limit on OpenAI and 100k on Anthropic
    manager.configure_session(
        "session-1",
        max_tokens=1_000_000,
        provider_limits={"openai": 50_000, "anthropic": 100_000},
    )

    manager.record_usage(
        "session-1",
        TokenUsage(
            provider="openai",
            input_tokens=50_000,
            output_tokens=10_000,
            total_tokens=60_000,
        ),
    )
    manager.record_usage(
        "session-1",
        TokenUsage(
            provider="anthropic",
            input_tokens=80_000,
            output_tokens=10_000,
            total_tokens=90_000,
        ),
    )

    # Overall budget is allowed, but OpenAI specifically is breached
    overall_decision = manager.check_budget("session-1")
    assert overall_decision.allowed is True

    openai_decision = manager.check_budget("session-1", provider="openai")
    assert openai_decision.allowed is False
    assert openai_decision.reason == "Provider 'openai' token limit exceeded: 60000/50000"

    with pytest.raises(BudgetExceededError, match="Provider 'openai' token limit exceeded"):
        manager.enforce_budget("session-1", provider="openai")

    # Anthropic has used 90k of its own 100k, so it may still proceed
    anthropic_decision = manager.check_budget("session-1", provider="anthropic")
    assert anthropic_decision.allowed is True
    assert anthropic_decision.remaining_tokens == 1_000_000 - 150_000


def test_token_budget_manager_reset() -> None:
    manager = TokenBudgetManager()
    manager.record_usage(
        "session-1",
        TokenUsage(
            provider="ollama",
            input_tokens=100,
            output_tokens=100,
            total_tokens=200,
        ),
    )
    assert manager.get_budget("session-1") is not None
    assert len(manager.get_turn_history("session-1")) == 1

    manager.reset_session("session-1")
    assert manager.get_budget("session-1") is None
    assert len(manager.get_turn_history("session-1")) == 0


# ======================================================================================
# ContextCompactor Tests
# ======================================================================================


def test_token_budget_manager_get_summary_and_compaction() -> None:
    """Verify TokenBudgetManager.get_summary() and record_compaction() return accurate metrics."""
    manager = TokenBudgetManager(default_max_tokens=200_000)

    # Empty summary
    empty_summary = manager.get_summary()
    assert empty_summary["total_tokens"] == 0
    assert empty_summary["prompt_tokens"] == 0
    assert empty_summary["completion_tokens"] == 0
    assert empty_summary["session_budget"]["total_used_tokens"] == 0
    assert empty_summary["session_budget"]["remaining_tokens"] == 200_000
    assert empty_summary["providers"] == {}
    assert empty_summary["compaction_history"] == []

    # Record usages
    usage_1 = TokenUsage(
        provider="anthropic",
        model="claude-3-5-sonnet",
        input_tokens=1000,
        output_tokens=200,
        total_tokens=1200,
    )
    manager.record_usage("sess_1", usage_1)

    # Record compaction
    cmp_record = manager.record_compaction(
        reason="Exceeded 70% threshold",
        original_tokens=5000,
        compacted_tokens=1200,
        kept_turns=4,
    )
    assert cmp_record["saved_tokens"] == 3800
    assert cmp_record["compression_ratio_pct"] == 76.0

    summary = manager.get_summary()
    assert summary["total_tokens"] == 1200
    assert summary["prompt_tokens"] == 1000
    assert summary["completion_tokens"] == 200
    assert "anthropic" in summary["providers"]
    assert summary["providers"]["anthropic"]["input_tokens"] == 1000
    assert len(summary["compaction_history"]) == 1
    assert summary["compaction_history"][0]["saved_tokens"] == 3800


def test_token_budget_manager_get_summary_session_filtering() -> None:
    """Verify get_summary(session_id=...) filters metrics and compaction records for specific session (P5, #213)."""
    manager = TokenBudgetManager(default_max_tokens=100_000)

    # Record usage for session 1
    manager.record_usage(
        "sess_1",
        TokenUsage(
            provider="openai",
            model="gpt-4o",
            input_tokens=500,
            output_tokens=100,
            total_tokens=600,
        ),
    )
    # Record usage for session 2
    manager.record_usage(
        "sess_2",
        TokenUsage(
            provider="anthropic",
            model="claude-3-5-sonnet",
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
        ),
    )

    # Record compactions attributed to specific sessions
    manager.record_compaction(
        reason="threshold_exceeded",
        original_tokens=4000,
        compacted_tokens=1000,
        session_id="sess_1",
    )
    manager.record_compaction(
        reason="manual_prune",
        original_tokens=8000,
        compacted_tokens=2000,
        session_id="sess_2",
    )

    # Unfiltered summary contains all
    all_summary = manager.get_summary()
    assert all_summary["total_tokens"] == 1800
    assert len(all_summary["compaction_history"]) == 2

    # Session 1 summary
    s1_summary = manager.get_summary(session_id="sess_1")
    assert s1_summary["total_tokens"] == 600
    assert s1_summary["prompt_tokens"] == 500
    assert s1_summary["completion_tokens"] == 100
    assert "openai" in s1_summary["providers"]
    assert "anthropic" not in s1_summary["providers"]
    assert len(s1_summary["compaction_history"]) == 1
    assert s1_summary["compaction_history"][0]["session_id"] == "sess_1"
    assert s1_summary["compaction_history"][0]["reason"] == "threshold_exceeded"

    # Session 2 summary
    s2_summary = manager.get_summary(session_id="sess_2")
    assert s2_summary["total_tokens"] == 1200
    assert "anthropic" in s2_summary["providers"]
    assert "openai" not in s2_summary["providers"]
    assert len(s2_summary["compaction_history"]) == 1
    assert s2_summary["compaction_history"][0]["session_id"] == "sess_2"
    assert s2_summary["compaction_history"][0]["reason"] == "manual_prune"

    # Non-existent session summary
    unknown_summary = manager.get_summary(session_id="sess_unknown")
    assert unknown_summary["total_tokens"] == 0
    assert unknown_summary["compaction_history"] == []


def test_context_compactor_estimate_tokens() -> None:
    compactor = ContextCompactor()

    # Short message
    msg = ChatMessage(role=MessageRole.USER, content="Hello world")
    tokens = compactor.estimate_tokens([msg])
    assert tokens > 0

    # Message with tool calls
    tool_msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Calling tool",
        tool_calls=(
            ToolCallRequest(id="call_1", name="search", arguments={"q": "pytest coverage"}),
        ),
    )
    tool_tokens = compactor.estimate_tokens([tool_msg])
    assert tool_tokens > tokens


def test_context_compactor_should_compact() -> None:
    compactor = ContextCompactor(threshold=0.70)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
        ChatMessage(role=MessageRole.USER, content="A" * 2800),  # ~700 tokens
    ]

    # Context limit = 2000 => 70% threshold is 1400 tokens -> should not compact
    assert compactor.should_compact(messages, context_limit=2000) is False

    # Context limit = 800 => 70% threshold is 560 tokens -> should compact
    assert compactor.should_compact(messages, context_limit=800) is True

    # Edge cases
    assert compactor.should_compact([], context_limit=800) is False
    assert compactor.should_compact(messages, context_limit=0) is False


def test_a_hangul_session_compacts_where_a_latin_one_of_equal_length_does_not() -> None:
    """The estimate counts UTF-8 bytes, so Hangul is not undercounted threefold (#939).

    `(len + 3) // 4` counted characters. The byte-level BPE tokenizers the connectors talk
    to split UTF-8 bytes, and a Hangul syllable is three of them. Against `hermes3:8b`
    (design doc §6.7) a `len // 4` estimate of a Korean reply was 31 for 67 counted
    tokens, so a Korean session compacted at roughly twice the context the threshold
    names. Latin text is one byte per character and keeps the figure it had.

    Killed by: src/uclone_x/llm/compactor.py :: len(text.encode("utf-8"))
    Becomes: len(text)
    """
    compactor = ContextCompactor(threshold=0.70)
    # 1,000 characters each: 3,000 UTF-8 bytes of Hangul, 1,000 of ASCII.
    korean = [ChatMessage(role=MessageRole.USER, content="안녕하세요" * 200)]
    latin = [ChatMessage(role=MessageRole.USER, content="hello" * 200)]

    assert compactor.estimate_tokens(korean) == 4 + 750
    assert compactor.estimate_tokens(latin) == 4 + 250
    # A 1,000-token window compacts at 700: the Korean session is over it, the Latin one
    # is not, and under a character count both were at 254.
    assert compactor.should_compact(korean, context_limit=1000) is True
    assert compactor.should_compact(latin, context_limit=1000) is False


def test_a_requests_estimate_counts_the_tool_definitions_sent_with_it() -> None:
    """A connector estimating a missing input count includes the tool schemas (#939).

    Providers count the tool definitions as input, and for an agent with many tools they
    can outweigh the conversation, so an estimate of the messages alone would understate
    exactly the requests a budget most needs to see.

    Killed by: src/uclone_x/llm/compactor.py :: total += 4 + estimate_text_tokens(f"{tool.name} {tool.description} {schema}")
    Becomes: pass
    """
    messages = (ChatMessage(role=MessageRole.USER, content="What is the weather?"),)
    tool = ToolDefinition(
        name="weather",
        description="Current weather for a city",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    schema = '{"type": "object", "properties": {"city": {"type": "string"}}}'

    bare = estimate_request_tokens(LLMRequest(messages=messages))
    with_tool = estimate_request_tokens(LLMRequest(messages=messages, tools=(tool,)))

    assert bare == estimate_message_tokens(messages)
    assert with_tool - bare == 4 + estimate_text_tokens(
        f"weather Current weather for a city {schema}"
    )


def test_a_replys_estimate_counts_the_tool_calls_it_made() -> None:
    """A connector estimating a missing output count includes the reply's tool calls (#939).

    A step whose reply is a tool call often has no text at all. An estimate of the text
    alone gives that step the one-token floor, however large its arguments, so the output
    a budget books for an agent that acts rather than talks would be close to nothing.

    Killed by: src/uclone_x/llm/compactor.py :: total += sum(_tool_call_tokens(tool_call) for tool_call in tool_calls)
    Becomes: total += 0
    """
    call = ToolCallRequest(
        id="call_1", name="read_file", arguments={"path": "docs/design/overview.md"}
    )
    arguments = '{"path": "docs/design/overview.md"}'
    call_tokens = 4 + estimate_text_tokens("call_1read_file") + estimate_text_tokens(arguments)

    assert estimate_reply_tokens(None) == 1
    assert estimate_reply_tokens(None, (call,)) == call_tokens
    assert estimate_reply_tokens("Reading it.", (call,)) == (
        estimate_text_tokens("Reading it.") + call_tokens
    )


def test_context_compactor_should_compact_at() -> None:
    compactor = ContextCompactor()

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
        ChatMessage(role=MessageRole.USER, content="A" * 2800),
    ]
    estimated = compactor.estimate_tokens(messages)
    assert estimated > 0

    assert compactor.should_compact_at(messages, threshold_tokens=estimated + 100) is False
    assert compactor.should_compact_at(messages, threshold_tokens=estimated) is True
    assert compactor.should_compact_at(messages, threshold_tokens=estimated - 100) is True

    # Edge cases
    assert compactor.should_compact_at([], threshold_tokens=500) is False
    assert compactor.should_compact_at(messages, threshold_tokens=0) is False
    assert compactor.should_compact_at(messages, threshold_tokens=-10) is False


def test_resolve_model_context_limit() -> None:
    assert resolve_model_context_limit(None) is None
    assert resolve_model_context_limit("") is None
    assert resolve_model_context_limit("unknown-custom-model") is None

    # Gemini family
    assert resolve_model_context_limit("gemini-1.5-flash") == 1_000_000
    assert resolve_model_context_limit("models/gemini-2.0-flash") == 1_000_000
    assert resolve_model_context_limit("gemini-1.5-pro") == 2_000_000

    # Claude family
    assert resolve_model_context_limit("claude-3-5-sonnet") == 200_000
    assert resolve_model_context_limit("claude-3-7-sonnet") == 200_000

    # OpenAI family
    assert resolve_model_context_limit("gpt-4o") == 128_000
    assert resolve_model_context_limit("gpt-4o-mini") == 128_000

    # Qwen family
    assert resolve_model_context_limit("qwen2.5-coder:32b") == 128_000
    assert resolve_model_context_limit("qwen2.5") == 128_000


@pytest.mark.asyncio
async def test_context_compactor_heuristic_compaction() -> None:
    compactor = ContextCompactor(threshold=0.70, keep_recent_turns=2, max_tool_output_chars=200)

    long_output = "Line " + "\nLine ".join(str(i) for i in range(100))
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System Instructions Anchored"),
        ChatMessage(role=MessageRole.USER, content="Step 1: start task"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Working on step 1",
            tool_calls=(ToolCallRequest(id="c1", name="run_cmd", arguments={"cmd": "ls"}),),
        ),
        ChatMessage(role=MessageRole.TOOL, name="run_cmd", tool_call_id="c1", content=long_output),
        ChatMessage(role=MessageRole.USER, content="Step 2: next task"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Step 2 done"),
        ChatMessage(role=MessageRole.USER, content="Recent question 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Recent answer 1"),
    ]

    compacted = (await compactor.compact(messages)).messages

    # 1. System prompt preserved at the top
    assert compacted[0].role == MessageRole.SYSTEM
    assert compacted[0].content == "System Instructions Anchored"

    # 2. Session Progress Ledger inserted as second message
    assert compacted[1].role == MessageRole.SYSTEM
    assert "[Context Auto-Compacted Summary: Heuristic Session Progress Ledger]" in (
        compacted[1].content or ""
    )
    assert "run_cmd" in (compacted[1].content or "")

    # 3. Recent 2 turns preserved
    assert compacted[-2].content == "Recent question 1"
    assert compacted[-1].content == "Recent answer 1"

    # Verify total message count was reduced from 8 to 4
    assert len(compacted) == 4


@pytest.mark.asyncio
async def test_context_compactor_prunes_long_tool_outputs() -> None:
    compactor = ContextCompactor(keep_recent_turns=4, max_tool_output_chars=100)

    long_tool_content = "X" * 500
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System Prompt"),
        ChatMessage(role=MessageRole.USER, content="Do something"),
        ChatMessage(role=MessageRole.TOOL, name="fetch", content=long_tool_content),
    ]

    compacted = (await compactor.compact(messages)).messages
    assert len(compacted) == 3
    tool_msg = compacted[2]
    assert tool_msg.role == MessageRole.TOOL
    assert "[Tool Output Truncated" in (tool_msg.content or "")
    assert "path=truncate" in (tool_msg.content or "")
    assert len(tool_msg.content or "") < 300


@pytest.mark.asyncio
async def test_context_compactor_offloads_oversized_tool_output(tmp_path: Path) -> None:
    workspace_root = tmp_path / "sandbox_ws"
    workspace_root.mkdir()
    compactor = ContextCompactor(
        keep_recent_turns=4,
        max_tool_output_chars=100,
        workspace_root=workspace_root,
        session_id="sess_alpha",
    )

    long_tool_content = "Y" * 500
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System Prompt"),
        ChatMessage(role=MessageRole.USER, content="Run tool"),
        ChatMessage(
            role=MessageRole.TOOL,
            name="query_db",
            content=long_tool_content,
            tool_call_id="call_db_1",
        ),
    ]

    compacted = (await compactor.compact(messages)).messages
    assert len(compacted) == 3
    tool_msg = compacted[2]
    assert tool_msg.role == MessageRole.TOOL
    assert "[Tool Output Offloaded" in (tool_msg.content or "")
    assert "path=offload" in (tool_msg.content or "")
    assert "query_db_call_db_1.txt" in (tool_msg.content or "")

    # Assert artifact written to disk and content matches
    artifact_path = (
        workspace_root / ".sandbox" / "tool_artifacts" / "sess_alpha" / "query_db_call_db_1.txt"
    )
    assert artifact_path.is_file()
    assert artifact_path.read_text(encoding="utf-8") == long_tool_content


@pytest.mark.asyncio
async def test_context_compactor_refuses_path_traversal(tmp_path: Path) -> None:
    from uclone_x.errors import PathTraversalError

    workspace_root = tmp_path / "sandbox_ws"
    workspace_root.mkdir()

    # 1. Traversal via session_id
    compactor_bad_session = ContextCompactor(
        max_tool_output_chars=100,
        workspace_root=workspace_root,
        session_id="../../escaped_session",
    )
    msg = ChatMessage(role=MessageRole.TOOL, name="test_tool", content="Z" * 500, tool_call_id="c1")
    with pytest.raises(PathTraversalError):
        compactor_bad_session.prune_tool_message(msg)

    # 2. Traversal via tool_call_id
    compactor_bad_call = ContextCompactor(
        max_tool_output_chars=100,
        workspace_root=workspace_root,
        session_id="valid_session",
    )
    bad_msg = ChatMessage(
        role=MessageRole.TOOL,
        name="test_tool",
        content="Z" * 500,
        tool_call_id="../../escaped_call",
    )
    with pytest.raises(PathTraversalError):
        compactor_bad_call.prune_tool_message(bad_msg)


@pytest.mark.asyncio
async def test_heuristic_ledger_includes_offloaded_artifact(tmp_path: Path) -> None:
    workspace_root = tmp_path / "sandbox_ws"
    workspace_root.mkdir()
    compactor = ContextCompactor(
        keep_recent_turns=2,
        max_tool_output_chars=100,
        workspace_root=workspace_root,
        session_id="sess_ledger",
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System Prompt"),
        ChatMessage(role=MessageRole.USER, content="Q1"),
        ChatMessage(
            role=MessageRole.TOOL, name="search_logs", content="A" * 600, tool_call_id="call_s1"
        ),
        ChatMessage(role=MessageRole.USER, content="Q2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="A2"),
    ]

    outcome = await compactor.compact(messages)
    # The middle messages: Q1 and search_logs should be summarized into the ledger
    ledger_msg = next(m for m in outcome.messages if m.compaction_ledger)
    assert ledger_msg.content is not None
    assert "search_logs" in ledger_msg.content
    assert (
        "artifact: '.sandbox/tool_artifacts/sess_ledger/search_logs_call_s1.txt'"
        in ledger_msg.content
    )


class _MockSummarizer(LLMProviderProtocol):
    @property
    def provider_name(self) -> str:
        return "mock_summarizer"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content="• Action 1: User requested initialization\n• Action 2: Completed setup successfully",
            usage=TokenUsage(
                provider="mock_summarizer", input_tokens=100, output_tokens=20, total_tokens=120
            ),
            provenance=Provenance.primary("mock_summarizer"),
        )

    async def stream(self, request: LLMRequest):  # type: ignore
        raise NotImplementedError


class _MaxTokensSummarizer(LLMProviderProtocol):
    """Returns prose at the `max_tokens=500` ceiling `_generate_llm_summary` sets.

    The worst case DEFAULT_MAX_LEDGERS' rationale reasons about, made measurable: a real
    provider is free to fill its output budget, and the ledger now carries that prose on
    top of the structural record.
    """

    @property
    def provider_name(self) -> str:
        return "max_tokens_summarizer"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        # ~4 characters per token is this module's own estimate_tokens ratio.
        return ModelResponse(
            finish_reason=FinishReason.LENGTH,
            content="lorem ipsum " * 170,
            usage=TokenUsage(
                provider="max_tokens_summarizer",
                input_tokens=100,
                output_tokens=500,
                total_tokens=600,
            ),
            provenance=Provenance.primary("max_tokens_summarizer"),
        )

    async def stream(self, request: LLMRequest):  # type: ignore
        raise NotImplementedError


class _FailingSummarizer(LLMProviderProtocol):
    @property
    def provider_name(self) -> str:
        return "failing_summarizer"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        raise RuntimeError("Summarizer LLM backend connection failed")

    async def stream(self, request: LLMRequest):  # type: ignore
        raise NotImplementedError


@pytest.mark.asyncio
async def test_context_compactor_llm_assisted_summarization() -> None:
    mock_llm = _MockSummarizer()
    compactor = ContextCompactor(keep_recent_turns=2, summarizer=mock_llm)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Anchored instructions"),
        ChatMessage(role=MessageRole.USER, content="Turn 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 1 reply"),
        ChatMessage(role=MessageRole.USER, content="Turn 2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 2 reply"),
        ChatMessage(role=MessageRole.USER, content="Turn 3 recent"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 3 recent reply"),
    ]

    compacted = (await compactor.compact(messages)).messages
    assert len(compacted) == 4
    assert compacted[0].content == "Anchored instructions"
    assert "[Context Auto-Compacted Summary: LLM Session Progress Ledger]" in (
        compacted[1].content or ""
    )
    assert "Completed setup successfully" in (compacted[1].content or "")
    assert compacted[2].content == "Turn 3 recent"
    assert compacted[3].content == "Turn 3 recent reply"


HEURISTIC_LABEL = "[Context Auto-Compacted Summary: Heuristic Session Progress Ledger]"
LLM_LABEL = "[Context Auto-Compacted Summary: LLM Session Progress Ledger]"


def _dialogue(start: int, pairs: int) -> list[ChatMessage]:
    """`pairs` user/assistant exchanges, long enough that a ledger is worth emitting."""
    turns: list[ChatMessage] = []
    for i in range(start, start + pairs):
        turns.append(
            ChatMessage(
                role=MessageRole.USER,
                content=f"User turn {i}: " + ("please continue the task in detail. " * 8),
            )
        )
        turns.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"Assistant turn {i}: " + ("here is a substantial reply. " * 10),
            )
        )
    return turns


async def _run_session(
    compactor: ContextCompactor, passes: int, pairs_between: int = 6
) -> list[tuple[ChatMessage, ...]]:
    """Drive `passes` successive compactions, feeding fresh dialogue between each.

    This is the shape a single-pass test cannot see: the defect in #196 was that the
    compactor's own output, fed back in on the next pass, accumulated in the resident
    `SYSTEM` block instead of being superseded.
    """
    history: list[ChatMessage] = [
        ChatMessage(role=MessageRole.SYSTEM, content="ANCHOR: the real system prompt.")
    ]
    turn = 1
    history += _dialogue(turn, pairs_between)
    turn += pairs_between

    outputs: list[tuple[ChatMessage, ...]] = []
    for _ in range(passes):
        compacted = (await compactor.compact(history)).messages
        outputs.append(compacted)
        history = [*compacted, *_dialogue(turn, pairs_between)]
        turn += pairs_between
    return outputs


@pytest.mark.asyncio
async def test_repeated_compaction_bounds_the_resident_system_block() -> None:
    """Six successive compactions must not grow the permanently-resident block (#196).

    The resident block is unprunable by construction: `_prune_tool_message` rewrites only
    `MessageRole.TOOL`, and the `keep_recent_turns` window is computed over
    `dialog_messages`, which excludes `SYSTEM`. So an unbounded ledger count here is an
    unbounded context, which is the outcome P5 exists to prevent.
    """
    compactor = ContextCompactor(keep_recent_turns=4, max_ledgers=2)
    outputs = await _run_session(compactor, passes=6)

    ledger_counts = [sum(1 for m in out if m.compaction_ledger) for out in outputs]
    assert ledger_counts == [1, 2, 2, 2, 2, 2], (
        f"ledger count per pass grew past max_ledgers=2: {ledger_counts}"
    )

    # The anchor survives every pass, exactly once, and is never mistaken for an artifact.
    for pass_no, out in enumerate(outputs, 1):
        anchors = [m for m in out if m.role == MessageRole.SYSTEM and not m.compaction_ledger]
        assert [m.content for m in anchors] == ["ANCHOR: the real system prompt."], (
            f"pass {pass_no} lost or duplicated the anchor: {[m.content for m in anchors]}"
        )
        assert out[0].content == "ANCHOR: the real system prompt."

    # Token trajectory: flat once the ledger budget is full. Measured on the pre-fix code
    # this was strictly increasing on every pass -- 939 tokens per pass on the heuristic
    # path -- so a plateau is what distinguishes the two.
    resident_tokens = [
        compactor.estimate_tokens([m for m in out if m.role == MessageRole.SYSTEM])
        for out in outputs
    ]
    assert resident_tokens[-3] == resident_tokens[-2] == resident_tokens[-1], (
        f"resident block still growing at the last passes: {resident_tokens}"
    )


@pytest.mark.asyncio
async def test_repeated_compaction_supersession_is_observable_not_silent() -> None:
    """A superseded ledger is the only record of turns already gone, so the loss is counted."""
    compactor = ContextCompactor(keep_recent_turns=4, max_ledgers=2)
    outputs = await _run_session(compactor, passes=4)

    # Passes 1 and 2 fit inside the budget; passes 3 and 4 each supersede one.
    assert compactor.superseded_ledger_count == 2, (
        f"expected 2 superseded ledgers over 4 passes, got {compactor.superseded_ledger_count}"
    )
    assert dict(compactor.supersession_reasons) == {"ledger_cap": 2}

    # And in band, on the replacement ledger, for a caller that only sees messages.
    newest = outputs[-1][-5]
    assert newest.compaction_ledger is True
    assert "Superseded 1 earlier compaction ledger(s) at this pass" in (newest.content or "")
    assert "2 by this compactor" in (newest.content or "")

    # A pass that supersedes nothing says nothing.
    first = outputs[0][1]
    assert first.compaction_ledger is True
    assert "Superseded" not in (first.content or "")


@pytest.mark.asyncio
async def test_fresh_compactor_per_pass_still_caps_and_states_only_what_it_knows() -> None:
    """A `ContextCompactor` built per turn must still bound the block (#196 review).

    This is the wiring the message-borne flag was chosen to survive — #183 has not settled
    the compactor's lifecycle — so it is pinned rather than assumed. Two things hold and a
    third deliberately does not: the cap holds, the per-pass count in the in-band note
    stays exact, and the running total resets with the instance. Hence "by this compactor"
    rather than "this session": in-band provenance must not claim a magnitude it cannot
    know.
    """
    history: list[ChatMessage] = [ChatMessage(role=MessageRole.SYSTEM, content="ANCHOR")]
    turn = 1
    history += _dialogue(turn, 6)
    turn += 6

    per_pass_counts: list[int] = []
    for _ in range(6):
        fresh = ContextCompactor(keep_recent_turns=4, max_ledgers=2)
        compacted = (await fresh.compact(history)).messages

        # The cap holds even though this instance has no memory of earlier passes.
        assert sum(1 for m in compacted if m.compaction_ledger) <= 2
        anchors = [m for m in compacted if m.role == MessageRole.SYSTEM and not m.compaction_ledger]
        assert [m.content for m in anchors] == ["ANCHOR"]

        per_pass_counts.append(fresh.superseded_ledger_count)
        newest = [m for m in compacted if m.compaction_ledger][-1]
        content = newest.content or ""
        if fresh.superseded_ledger_count:
            # Per-pass truth, and no claim about the session.
            assert "Superseded 1 earlier compaction ledger(s) at this pass" in content
            assert "(1 by this compactor)" in content
            assert "session" not in content

        history = [*compacted, *_dialogue(turn, 6)]
        turn += 6

    # Per instance, so it never accumulates past one pass' worth — the honest reading.
    assert per_pass_counts == [0, 0, 1, 1, 1, 1], per_pass_counts


@pytest.mark.asyncio
async def test_repeated_compaction_preserves_in_band_ledger_labels() -> None:
    """The LLM/heuristic in-band labels are provenance and survive capping unchanged."""
    heuristic = ContextCompactor(keep_recent_turns=4, max_ledgers=2)
    for out in await _run_session(heuristic, passes=4):
        for ledger in (m for m in out if m.compaction_ledger):
            content = ledger.content or ""
            assert content.startswith(HEURISTIC_LABEL), (
                f"heuristic label is no longer the first line: {content[:120]!r}"
            )

    assisted = ContextCompactor(keep_recent_turns=4, max_ledgers=2, summarizer=_MockSummarizer())
    for out in await _run_session(assisted, passes=4):
        for ledger in (m for m in out if m.compaction_ledger):
            content = ledger.content or ""
            assert content.startswith(LLM_LABEL), (
                f"LLM label is no longer the first line: {content[:120]!r}"
            )


@pytest.mark.asyncio
async def test_context_compactor_max_ledgers_one_keeps_only_the_replacement() -> None:
    """`max_ledgers=1` supersedes on every pass after the first, and says so each time."""
    compactor = ContextCompactor(keep_recent_turns=4, max_ledgers=1)
    outputs = await _run_session(compactor, passes=3)

    assert [sum(1 for m in out if m.compaction_ledger) for out in outputs] == [1, 1, 1]
    assert compactor.superseded_ledger_count == 2
    assert dict(compactor.supersession_reasons) == {"ledger_cap": 2}


@pytest.mark.asyncio
async def test_context_compactor_caps_ledgers_on_the_short_dialogue_path() -> None:
    """The early return for a dialogue inside the recent window caps ledgers too.

    It emits no new ledger, so the whole budget goes to the prior ones — but it must not
    wave through a resident block that is already over budget.
    """
    compactor = ContextCompactor(keep_recent_turns=4, max_ledgers=2)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="ANCHOR"),
        ChatMessage(
            role=MessageRole.SYSTEM, content=f"{HEURISTIC_LABEL}\nold", compaction_ledger=True
        ),
        ChatMessage(
            role=MessageRole.SYSTEM, content=f"{HEURISTIC_LABEL}\nmid", compaction_ledger=True
        ),
        ChatMessage(
            role=MessageRole.SYSTEM, content=f"{HEURISTIC_LABEL}\nnew", compaction_ledger=True
        ),
        ChatMessage(role=MessageRole.USER, content="Only recent turn"),
    ]

    compacted = (await compactor.compact(messages)).messages

    assert [m.content for m in compacted] == [
        "ANCHOR",
        f"{HEURISTIC_LABEL}\nmid",
        f"{HEURISTIC_LABEL}\nnew",
        "Only recent turn",
    ]
    assert compactor.superseded_ledger_count == 1
    assert dict(compactor.supersession_reasons) == {"ledger_cap": 1}


def test_context_compactor_rejects_out_of_bounds_max_ledgers() -> None:
    """`max_ledgers` must be bounded between 1 and 5 (#196, #204)."""
    with pytest.raises(ValueError, match="max_ledgers must be between 1 and 5"):
        ContextCompactor(max_ledgers=0)
    with pytest.raises(ValueError, match="max_ledgers must be between 1 and 5"):
        ContextCompactor(max_ledgers=-1)
    with pytest.raises(ValueError, match="max_ledgers must be between 1 and 5"):
        ContextCompactor(max_ledgers=6)
    with pytest.raises(ValueError, match="max_ledgers must be between 1 and 5"):
        ContextCompactor(max_ledgers=100)


def test_chat_message_rejects_compaction_ledger_on_non_system_roles() -> None:
    """`compaction_ledger=True` is only valid on MessageRole.SYSTEM (#204)."""
    # Valid on SYSTEM
    sys_msg = ChatMessage(role=MessageRole.SYSTEM, content="Ledger", compaction_ledger=True)
    assert sys_msg.compaction_ledger is True

    # Rejected on USER, ASSISTANT, TOOL
    for invalid_role in (MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL):
        with pytest.raises(
            ValueError,
            match="compaction_ledger=True is only meaningful and permitted on MessageRole.SYSTEM",
        ):
            ChatMessage(role=invalid_role, content="Not a ledger", compaction_ledger=True)


@pytest.mark.asyncio
async def test_context_compactor_marks_its_own_ledger_and_nothing_else() -> None:
    """Only the ledger carries `compaction_ledger`; anchors and dialogue do not."""
    compactor = ContextCompactor(keep_recent_turns=2)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Anchor"),
        ChatMessage(role=MessageRole.USER, content="Turn 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 1 reply"),
        ChatMessage(role=MessageRole.USER, content="Turn 2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 2 reply"),
    ]

    compacted = (await compactor.compact(messages)).messages

    assert [m.compaction_ledger for m in compacted] == [False, True, False, False]
    assert compactor.superseded_ledger_count == 0
    assert dict(compactor.supersession_reasons) == {}


@pytest.mark.asyncio
async def test_context_compactor_failing_summarizer_propagates_fail_fast() -> None:
    """A summarizer failure propagates rather than silently substituting heuristic ledger (P6 / Issue #50)."""
    failing_llm = _FailingSummarizer()
    compactor = ContextCompactor(keep_recent_turns=2, summarizer=failing_llm)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System"),
        ChatMessage(role=MessageRole.USER, content="Turn 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 1 reply"),
        ChatMessage(role=MessageRole.USER, content="Turn 2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 2 reply"),
        ChatMessage(role=MessageRole.USER, content="Turn 3 recent"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 3 recent reply"),
    ]

    with pytest.raises(RuntimeError, match="Summarizer LLM backend connection failed"):
        await compactor.compact(messages)


# ======================================================================================
# CompactionOutcome: what a pass produced, and who produced it (#183, P6)
# ======================================================================================


class _EmptySummarizer(LLMProviderProtocol):
    """A configured summarizer that answers with no content."""

    @property
    def provider_name(self) -> str:
        return "empty_summarizer"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content="",
            usage=TokenUsage(provider="empty_summarizer"),
            provenance=Provenance.primary("empty_summarizer"),
        )

    async def stream(self, request: LLMRequest):  # type: ignore
        raise NotImplementedError


class _UnattributedSummarizer(LLMProviderProtocol):
    """A summarizer whose response carries no attribution at all."""

    @property
    def provider_name(self) -> str:
        return "unattributed_summarizer"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content="• a summary nobody attributed",
            usage=TokenUsage(provider="unattributed_summarizer"),
            provenance=None,
        )

    async def stream(self, request: LLMRequest):  # type: ignore
        raise NotImplementedError


def _long_dialogue() -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="Anchored instructions"),
        ChatMessage(role=MessageRole.USER, content="Turn 1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 1 reply"),
        ChatMessage(role=MessageRole.USER, content="Turn 2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 2 reply"),
        ChatMessage(role=MessageRole.USER, content="Turn 3 recent"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Turn 3 recent reply"),
    ]


@pytest.mark.asyncio
async def test_heuristic_pass_reports_its_source_and_attributes_itself() -> None:
    """The heuristic ledger is a genuine local execution of the declared algorithm, not
    a substitution for a failed provider call, so it is `primary` and undegraded."""
    outcome = await ContextCompactor(keep_recent_turns=2).compact(_long_dialogue())

    assert outcome.ledger_source is LedgerSource.HEURISTIC
    assert outcome.provenance is not None
    assert outcome.provenance.path is ExecutionPath.PRIMARY
    assert outcome.provenance.served_by.provider == "uclone_x.llm.compactor"
    assert outcome.provenance.served_by.model == "heuristic-ledger"
    assert outcome.provenance.degraded is False
    assert outcome.provenance.attempts == ()


@pytest.mark.asyncio
async def test_llm_pass_forwards_the_summarizer_attribution_verbatim() -> None:
    """An LLM-written ledger is a value a model produced. The compactor relays the
    summarizer's own provenance rather than naming itself as the server."""
    outcome = await ContextCompactor(keep_recent_turns=2, summarizer=_MockSummarizer()).compact(
        _long_dialogue()
    )

    assert outcome.ledger_source is LedgerSource.LLM
    assert outcome.provenance is not None
    assert outcome.provenance.served_by.provider == "mock_summarizer"
    # Not the compactor: it did not produce this text.
    assert outcome.provenance.served_by.provider != "uclone_x.llm.compactor"


@pytest.mark.asyncio
async def test_an_unattributed_summary_stays_unattributed() -> None:
    """`None` travels. Naming the compactor as the server of text a model produced would
    turn "not stated" into a positively asserted clean result (P6)."""
    outcome = await ContextCompactor(
        keep_recent_turns=2, summarizer=_UnattributedSummarizer()
    ).compact(_long_dialogue())

    assert outcome.ledger_source is LedgerSource.LLM
    assert outcome.provenance is None


@pytest.mark.asyncio
async def test_an_empty_summarizer_response_falls_back_but_is_no_longer_silent() -> None:
    """A configured summarizer returning empty content still yields the heuristic
    ledger, but the outcome now *says* so and attributes it.

    Before `CompactionOutcome`, this case was indistinguishable from "no summarizer
    configured", because `_generate_llm_summary` returned `None` for both. Whether an
    empty provider response should instead propagate as a failure is a question about
    the compaction algorithm's P6 posture and is filed separately, not decided here.
    """
    outcome = await ContextCompactor(keep_recent_turns=2, summarizer=_EmptySummarizer()).compact(
        _long_dialogue()
    )

    assert outcome.ledger_source is LedgerSource.HEURISTIC
    assert outcome.provenance is not None
    assert outcome.provenance.served_by.model == "heuristic-ledger"
    assert "Heuristic Session Progress Ledger" in (outcome.messages[1].content or "")


@pytest.mark.asyncio
async def test_the_short_dialogue_path_reports_that_no_ledger_was_written() -> None:
    outcome = await ContextCompactor(keep_recent_turns=8).compact(_long_dialogue())

    assert outcome.ledger_source is LedgerSource.NONE
    assert outcome.superseded_ledger_count == 0
    assert outcome.provenance is not None
    assert outcome.provenance.served_by.model == "tool-output-pruner"
    # Nothing was summarized away.
    assert len(outcome.messages) == len(_long_dialogue())


@pytest.mark.asyncio
async def test_an_empty_input_reports_no_ledger_and_still_attributes() -> None:
    outcome = await ContextCompactor().compact([])
    assert outcome.messages == ()
    assert outcome.ledger_source is LedgerSource.NONE
    assert outcome.provenance is not None


@pytest.mark.asyncio
async def test_the_outcome_reports_ledgers_superseded_at_this_pass() -> None:
    """Matches the figure the replacement ledger names in band (#196)."""
    compactor = ContextCompactor(keep_recent_turns=2, max_ledgers=1)
    history = _long_dialogue()
    outcome = await compactor.compact(history)
    for _ in range(2):
        history = list(outcome.messages) + [
            ChatMessage(role=MessageRole.USER, content="another turn"),
            ChatMessage(role=MessageRole.ASSISTANT, content="another reply"),
            ChatMessage(role=MessageRole.USER, content="and again"),
            ChatMessage(role=MessageRole.ASSISTANT, content="and again reply"),
        ]
        outcome = await compactor.compact(history)

    assert outcome.superseded_ledger_count == 1
    assert compactor.superseded_ledger_count >= 1
    resident = [m for m in outcome.messages if m.compaction_ledger]
    assert len(resident) == 1


@pytest.mark.asyncio
async def test_the_outcome_is_frozen_and_forbids_unknown_fields() -> None:
    outcome = await ContextCompactor(keep_recent_turns=2).compact(_long_dialogue())
    with pytest.raises(ValueError):
        outcome.ledger_source = LedgerSource.LLM  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError):
        CompactionOutcome(
            ledger_source=LedgerSource.NONE,
            provenance=None,
            surprise="x",  # pyright: ignore[reportCallIssue]
        )


def test_the_protocol_declares_the_estimator_and_the_recent_window() -> None:
    """A caller reporting how much a compaction saved must use the same estimator the
    trigger used, and must not invent a default recent window behind a `getattr`."""
    compactor: ContextCompactorProtocol = ContextCompactor(keep_recent_turns=3)
    assert compactor.keep_recent_turns == 3
    assert compactor.estimate_tokens([ChatMessage(role=MessageRole.USER, content="hi")]) > 0


# ======================================================================================
# Issue #385 — the same class outside the connector layer
#
# `_build_heuristic_ledger` is not a wire payload, but it is not prose either: it replaces
# the turns it summarises in the history the model is given, so whatever it asserts becomes
# the model's own account of what happened.
# ======================================================================================


def test_heuristic_ledger_refuses_a_nameless_tool_result_rather_than_naming_it_tool() -> None:
    """`tool_name = msg.name or "tool"` named a tool that never ran, to the model itself.

    This is `gemini.py`'s `functionResponse.name` fabrication (#380) pointed at the one
    party the compacted history exists to inform, and in the ledger text it is
    indistinguishable from a real tool called `tool`.

    Refusing is safe on every in-repo path: `agent/base.py` passes `name=tc.name` at every
    `MessageRole.TOOL` construction site, and `ui.app`'s rehydration already refuses a
    `TOOL` record whose name is missing or blank and cannot be repaired from a preceding
    tool call (#387/PR #388). A nameless `TOOL` message reaching here therefore means a
    caller built one directly, and the ledger cannot honestly describe it.

    Mutation this exists to catch:
        -   if msg.name is None or not msg.name.strip():
        -       raise UnmappableChatMessageError(...)
        -   f"• Tool Result (turn {i}, {msg.name}): completed execution"
        +   tool_name = msg.name or "tool"
        +   f"• Tool Result (turn {i}, {tool_name}): completed execution"
    """
    compactor = ContextCompactor()

    for nameless in (None, "", "   "):
        with pytest.raises(UnmappableChatMessageError) as excinfo:
            compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
                (
                    ChatMessage(
                        role=MessageRole.TOOL,
                        content="42",
                        name=nameless,
                        tool_call_id="c1",
                    ),
                )
            )
        text = str(excinfo.value)
        assert "c1" in text, text
        assert "never ran" in text, text

    # The accepting case, so the guard cannot pass by refusing everything (§6.9 case 2).
    ledger = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (ChatMessage(role=MessageRole.TOOL, content="42", name="read_file", tool_call_id="c1"),)
    )
    assert "read_file" in ledger
    assert "tool)" not in ledger


def test_heuristic_ledger_distinguishes_absent_user_content_from_the_empty_string() -> None:
    """A `USER` turn holding `None` and one holding `""` no longer render identically.

    `msg.content or ""` produced the same empty bullet for both, so the model read one
    fact where the history held two. The marker is a *statement about the record* rather
    than a stand-in for text that was never there.

    It is distinguishable, not escaped: a user whose message is literally the marker
    string yields the same bullet. That residual ambiguity is stated rather than claimed
    away, and it is strictly narrower than the previous one, which collapsed every absent
    content onto the empty string.

    Mutation this exists to catch:
        -   if msg.content is None:
        -       content_preview = _NO_CONTENT_MARKER
        +   content_preview = msg.content or ""
    """
    compactor = ContextCompactor()

    absent = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (ChatMessage(role=MessageRole.USER, content=None),)
    )
    empty = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (ChatMessage(role=MessageRole.USER, content=""),)
    )

    assert "[no content recorded]" in absent
    assert "[no content recorded]" not in empty
    assert absent != empty

    # Long content still truncates, and a short non-empty turn is unchanged, so the new
    # branch cannot have swallowed the ordinary paths.
    long_text = "x" * 200
    long_ledger = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (ChatMessage(role=MessageRole.USER, content=long_text),)
    )
    assert long_ledger.endswith("...")
    assert "x" * 150 in long_ledger
    short = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (ChatMessage(role=MessageRole.USER, content="hello"),)
    )
    assert short.endswith("hello")


def test_heuristic_ledger_renders_an_empty_assistant_turn_rather_than_dropping_it() -> None:
    """`elif msg.content:` on the `ASSISTANT` branch emitted no bullet at all for `""`.

    So an assistant turn that said the empty string and one that recorded nothing both
    vanished from the model's own history, which is the same two-inputs-one-output filter
    as the `SYSTEM` site PR #384's review caught, reached by omission rather than by
    substitution. Both now render, symmetrically with the `USER` branch.

    Mutation this exists to catch:
        -   else:
        -       if msg.content is None:
        -           content_preview = _NO_CONTENT_MARKER
        +   elif msg.content:
    """
    compactor = ContextCompactor()

    empty = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (ChatMessage(role=MessageRole.ASSISTANT, content=""),)
    )
    absent = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (ChatMessage(role=MessageRole.ASSISTANT, content=None),)
    )

    assert "Agent Reasoning (turn 1):" in empty
    assert "Agent Reasoning (turn 1):" in absent
    assert "[no content recorded]" in absent
    assert "[no content recorded]" not in empty
    assert empty != absent

    # A turn with tool calls still reports the action rather than its (absent) prose, so
    # the new `else` cannot have swallowed the tool-call branch.
    with_calls = compactor._build_heuristic_ledger(  # pyright: ignore[reportPrivateUsage]
        (
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=(ToolCallRequest(id="t1", name="read_file", arguments={}),),
            ),
        )
    )
    assert "Agent Action (turn 1): Invoked tool(s) [read_file]" in with_calls
    assert "Agent Reasoning" not in with_calls


# ======================================================================================
# Needles Survive Successive Compactions Without Help From The Summarizer (#584)
# ======================================================================================


class _UselessProseSummarizer(LLMProviderProtocol):
    """A summarizer whose answer is fluent, non-empty, and says nothing about the turns.

    This is the shape of every offline summarizer stub, `MockLLMConnector` included: it
    returns content, so `_generate_llm_summary` accepts it and the empty-content
    fallback never fires. Pre-#584 that reply *replaced* the compactor's own record of
    the discarded turns, and every concrete value in them was lost at the first pass.
    """

    @property
    def provider_name(self) -> str:
        return "useless_prose_summarizer"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content="The session progressed and the assistant completed its work.",
            usage=TokenUsage(provider="useless_prose_summarizer"),
            provenance=Provenance.primary("useless_prose_summarizer"),
        )

    async def stream(self, request: LLMRequest):  # type: ignore
        raise NotImplementedError


_NEEDLE = "f47ac10b-58cc-4372-a567-0e02b2c3d479"


async def _compact_n_times(
    compactor: ContextCompactor, passes: int
) -> tuple[list[ChatMessage], list[CompactionOutcome]]:
    """Introduce the needle once, then run `passes` compactions over a growing dialogue.

    The needle rides in the *first* turn and in no later one, so it can only be found
    later if some ledger carried it forward. Each pass adds enough turns that the needle
    turn is well outside `keep_recent_turns`, which is what makes this a test of the
    ledger chain rather than of the recent window.
    """
    history: list[ChatMessage] = [
        ChatMessage(role=MessageRole.SYSTEM, content="Anchored instructions"),
        ChatMessage(role=MessageRole.USER, content=f"Deployment session UUID: {_NEEDLE}."),
    ]
    outcomes: list[CompactionOutcome] = []
    for p in range(1, passes + 1):
        history.extend(_dialogue(start=p * 10, pairs=3))
        outcome = await compactor.compact(history)
        outcomes.append(outcome)
        history = list(outcome.messages)
    return history, outcomes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summarizer",
    [None, _EmptySummarizer(), _UselessProseSummarizer(), _MockSummarizer()],
    ids=["no_summarizer", "empty_string", "useless_prose", "mock"],
)
async def test_a_needle_survives_three_compactions_whatever_the_summarizer_says(
    summarizer: LLMProviderProtocol | None,
) -> None:
    """Retention of a concrete value is a property of the compactor, not of the prose.

    The parametrization is the assertion: a summarizer that returns nothing, one that
    returns fluent content mentioning no fact, one that returns a plausible bullet list,
    and none at all must all retain the needle across three passes. Pre-#584 the two
    non-empty summarizers lost it at pass 1, because their reply replaced the record
    instead of being layered onto it — which is how the compaction battery measured 4.2%
    needle recall at cycle 3 offline while the summarizer-free path measured 87.5%.

    `max_ledgers=3` here so that three passes stay inside the ledger horizon; the
    horizon itself is pinned separately below.
    """
    compactor = ContextCompactor(keep_recent_turns=2, max_ledgers=3, summarizer=summarizer)

    history, outcomes = await _compact_n_times(compactor, passes=3)

    assert len(outcomes) == 3
    corpus = "\n".join(m.content or "" for m in history)
    assert _NEEDLE in corpus, (
        f"needle lost after 3 compactions with summarizer="
        f"{summarizer.provider_name if summarizer else None}"
    )
    # And it is a ledger carrying it, not a stray recent turn.
    ledger_corpus = "\n".join(m.content or "" for m in history if m.compaction_ledger)
    assert _NEEDLE in ledger_corpus


@pytest.mark.asyncio
async def test_the_ledger_cap_is_the_retention_horizon_and_it_is_accounted_for() -> None:
    """`max_ledgers` bounds how many passes a needle survives, and the loss is counted.

    Stated rather than papered over: at `max_ledgers=2` the pass-1 ledger is superseded
    when pass 3 emits its own, so a value that only that ledger carried is genuinely
    gone. `_retain_ledgers` already refuses to drop it silently, and this pins that the
    horizon equals the cap instead of leaving it to be rediscovered. It is the whole of
    the residual gap in the compaction battery (3 needles of 24 -> 87.5% at cycle 3);
    raising it is a `max_ledgers` decision bounded by #196's self-retriggering loop.
    """
    compactor = ContextCompactor(keep_recent_turns=2, max_ledgers=2, summarizer=None)

    history, _ = await _compact_n_times(compactor, passes=2)
    assert _NEEDLE in "\n".join(m.content or "" for m in history)

    # Pass 3 evicts the only ledger that carried it.
    history.extend(_dialogue(start=99, pairs=3))
    outcome = await compactor.compact(history)

    assert _NEEDLE not in "\n".join(m.content or "" for m in outcome.messages)
    assert outcome.superseded_ledger_count == 1
    assert compactor.supersession_reasons["ledger_cap"] >= 1
    # The note rides on the ledger this pass emitted, which is the last one.
    ledgers = [m.content or "" for m in outcome.messages if m.compaction_ledger]
    assert "not carried forward" in ledgers[-1]


@pytest.mark.asyncio
async def test_repeated_compaction_bounds_the_resident_block_with_a_summarizer() -> None:
    """The #196 headroom argument must hold on the path #584 actually enlarged.

    `test_repeated_compaction_bounds_the_resident_system_block` drives this session with
    no summarizer, so it never exercises the configuration where a ledger carries the
    structural record *and* the model's prose. That is precisely the configuration #584
    changed, and the claim that its worst case still clears the self-retriggering
    threshold lived only in a module comment. Measure it.
    """
    compactor = ContextCompactor(keep_recent_turns=4, max_ledgers=2, summarizer=_MockSummarizer())
    outputs = await _run_session(compactor, passes=6)

    ledger_counts = [sum(1 for m in out if m.compaction_ledger) for out in outputs]
    assert ledger_counts == [1, 2, 2, 2, 2, 2], (
        f"ledger count per pass grew past max_ledgers=2 with a summarizer: {ledger_counts}"
    )

    resident = [
        compactor.estimate_tokens([m for m in out if m.role == MessageRole.SYSTEM])
        for out in outputs
    ]
    print(f"\nresident tokens per pass (summarizer configured): {resident}")

    # Flat once the ledger budget is full: growth here is the unbounded context P5 exists
    # to prevent, and it would arrive as a compaction that retriggers itself.
    assert resident[-3] == resident[-2] == resident[-1], (
        f"resident block still growing at the last passes: {resident}"
    )

    # DEFAULT_MAX_LEDGERS' rationale puts the self-retriggering trigger at 5,734 tokens of
    # an 8K window. The prose adds up to `max_tokens=500` per ledger on top of the record.
    assert resident[-1] < 5_734, (
        f"resident block {resident[-1]} tokens reaches the self-retriggering trigger; "
        "the max_ledgers bound in compactor.py no longer holds with a summarizer"
    )

    # And the worst case the rationale actually reasons about: prose at the 500-token
    # ceiling, on every pass. A bound that holds only for a terse stub is not a bound.
    verbose = ContextCompactor(
        keep_recent_turns=4, max_ledgers=2, summarizer=_MaxTokensSummarizer()
    )
    verbose_outputs = await _run_session(verbose, passes=6)
    verbose_resident = [
        verbose.estimate_tokens([m for m in out if m.role == MessageRole.SYSTEM])
        for out in verbose_outputs
    ]
    print(f"resident tokens per pass (500-token prose): {verbose_resident}")

    assert verbose_resident[-3] == verbose_resident[-2] == verbose_resident[-1], (
        f"resident block still growing with max-length prose: {verbose_resident}"
    )
    assert verbose_resident[-1] < 5_734, (
        f"worst-case resident block {verbose_resident[-1]} tokens reaches the "
        "self-retriggering trigger (#196); max_ledgers must shrink or the prose must be "
        "bounded before it enters the ledger"
    )
