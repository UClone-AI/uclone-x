"""Unit tests for TokenBudgetManager and per-session compaction attribution (P5, P6, Issue #213, #226)."""

from __future__ import annotations

import pytest

from uclone_x.errors import BudgetExceededError
from uclone_x.llm import TokenBudgetManager, TokenUsage
from uclone_x.llm.models import TokenBudget, TokenCountSource


def test_budget_manager_initial_state_per_session() -> None:
    manager = TokenBudgetManager(default_max_tokens=100_000)

    decision = manager.check_budget("session_a")
    assert decision.allowed is True
    assert decision.remaining_tokens == 100_000

    budget = manager.get_budget("session_a")
    assert budget is not None
    assert budget.max_tokens == 100_000
    assert budget.used_input_tokens == 0


def test_budget_manager_multi_session_isolation() -> None:
    manager = TokenBudgetManager(default_max_tokens=50_000)

    # Session A usage
    manager.record_usage(
        "session_a",
        TokenUsage(provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=200),
    )
    # Session B usage
    manager.record_usage(
        "session_b",
        TokenUsage(
            provider="anthropic", model="claude-3-5-sonnet", input_tokens=5000, output_tokens=1000
        ),
    )

    budget_a = manager.get_budget("session_a")
    assert budget_a is not None
    assert budget_a.used_input_tokens == 1000
    assert dict(budget_a.per_provider) == {"openai": 1200}

    budget_b = manager.get_budget("session_b")
    assert budget_b is not None
    assert budget_b.used_input_tokens == 5000
    assert dict(budget_b.per_provider) == {"anthropic": 6000}

    # Check history isolation
    hist_a = manager.get_turn_history("session_a")
    assert len(hist_a) == 1
    assert hist_a[0].provider == "openai"

    hist_b = manager.get_turn_history("session_b")
    assert len(hist_b) == 1
    assert hist_b[0].provider == "anthropic"


def test_budget_manager_compaction_attribution_and_filtering() -> None:
    manager = TokenBudgetManager(default_max_tokens=200_000)

    # Record compactions for different sessions
    cmp_a1 = manager.record_compaction(
        reason="auto_threshold",
        original_tokens=60_000,
        compacted_tokens=15_000,
        kept_turns=4,
        session_id="session_alpha",
    )
    assert cmp_a1["session_id"] == "session_alpha"
    assert cmp_a1["saved_tokens"] == 45_000
    assert cmp_a1["compression_ratio_pct"] == 75.0

    cmp_a2 = manager.record_compaction(
        reason="manual_on_demand",
        original_tokens=50_000,
        compacted_tokens=20_000,
        kept_turns=4,
        session_id="session_alpha",
    )
    assert cmp_a2["session_id"] == "session_alpha"

    cmp_b1 = manager.record_compaction(
        reason="auto_threshold",
        original_tokens=80_000,
        compacted_tokens=20_000,
        kept_turns=2,
        session_id="session_beta",
    )
    assert cmp_b1["session_id"] == "session_beta"

    # Global summary has all 3
    global_summary = manager.get_summary()
    assert len(global_summary["compaction_history"]) == 3

    # Session alpha summary has 2
    alpha_summary = manager.get_summary(session_id="session_alpha")
    assert len(alpha_summary["compaction_history"]) == 2
    assert all(c["session_id"] == "session_alpha" for c in alpha_summary["compaction_history"])
    assert [c["reason"] for c in alpha_summary["compaction_history"]] == [
        "auto_threshold",
        "manual_on_demand",
    ]

    # Session beta summary has 1
    beta_summary = manager.get_summary(session_id="session_beta")
    assert len(beta_summary["compaction_history"]) == 1
    assert beta_summary["compaction_history"][0]["session_id"] == "session_beta"
    assert beta_summary["compaction_history"][0]["kept_turns"] == 2

    # Unknown session summary has 0
    empty_summary = manager.get_summary(session_id="session_gamma")
    assert len(empty_summary["compaction_history"]) == 0


def test_budget_manager_reset_session_cleans_compactions() -> None:
    manager = TokenBudgetManager()

    manager.record_usage(
        "sess_x",
        TokenUsage(
            provider="ollama",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
    )
    manager.record_compaction(
        reason="test_prune",
        original_tokens=1000,
        compacted_tokens=300,
        session_id="sess_x",
    )
    manager.record_compaction(
        reason="other_prune",
        original_tokens=2000,
        compacted_tokens=500,
        session_id="sess_y",
    )

    assert len(manager.get_summary(session_id="sess_x")["compaction_history"]) == 1
    assert len(manager.get_summary(session_id="sess_y")["compaction_history"]) == 1
    assert len(manager.get_summary()["compaction_history"]) == 2

    # Reset sess_x
    manager.reset_session("sess_x")

    assert manager.get_budget("sess_x") is None
    assert len(manager.get_turn_history("sess_x")) == 0
    assert len(manager.get_summary(session_id="sess_x")["compaction_history"]) == 0
    # sess_y compactions remain
    assert len(manager.get_summary(session_id="sess_y")["compaction_history"]) == 1
    assert len(manager.get_summary()["compaction_history"]) == 1


def test_budget_manager_set_and_configure_session() -> None:
    """`set_budget` stores the budget as given; `configure_session` builds one from limits."""
    manager = TokenBudgetManager()
    custom_budget = TokenBudget(max_tokens=20_000)
    manager.set_budget("custom_sess", custom_budget, provider_limits={"openai": 500})

    assert manager.get_budget("custom_sess") == custom_budget

    configured = manager.configure_session("conf_sess", max_tokens=30_000)
    assert configured.max_tokens == 30_000
    assert manager.get_budget("conf_sess") == configured


def test_the_session_token_ceiling_refuses_once_used_tokens_reach_it() -> None:
    """Input and output tokens together count against the ceiling; reaching it refuses (P6)."""
    manager = TokenBudgetManager()
    manager.configure_session("s", max_tokens=100)
    manager.record_usage("s", TokenUsage(provider="openai", input_tokens=60, output_tokens=39))

    below = manager.check_budget("s")
    assert (below.allowed, below.remaining_tokens) == (True, 1)

    manager.record_usage("s", TokenUsage(provider="openai", output_tokens=1))
    refused = manager.check_budget("s")
    assert refused.allowed is False
    assert refused.reason == "Session token limit exceeded: 100/100"
    assert refused.remaining_tokens == 0
    with pytest.raises(BudgetExceededError, match="Session token limit exceeded: 100/100"):
        manager.enforce_budget("s")


def test_a_provider_token_limit_refuses_that_provider_and_no_other() -> None:
    """A per-provider token quota is decoupled from the session ceiling and other providers (P5).

    The limit counts the provider's input and output tokens together. Under the limit the
    provider is allowed; at it, that provider is refused while another provider, and the
    session checked without a provider, still proceed.
    """
    manager = TokenBudgetManager()
    manager.set_budget(
        "s", TokenBudget(max_tokens=1_000_000), provider_limits={"ollama": 50, "openai": 50}
    )
    manager.record_usage("s", TokenUsage(provider="ollama", input_tokens=30, output_tokens=19))

    allowed = manager.check_budget("s", provider="ollama")
    assert (allowed.allowed, allowed.reason) == (True, None)

    manager.record_usage("s", TokenUsage(provider="ollama", output_tokens=1))
    refused = manager.check_budget("s", provider="ollama")
    assert refused.allowed is False
    assert refused.reason == "Provider 'ollama' token limit exceeded: 50/50"
    assert refused.remaining_tokens == 1_000_000 - 50
    with pytest.raises(BudgetExceededError, match="Provider 'ollama' token limit exceeded"):
        manager.enforce_budget("s", provider="ollama")

    assert manager.check_budget("s", provider="openai").allowed is True
    assert manager.check_budget("s", provider="anthropic").allowed is True
    assert manager.check_budget("s").allowed is True


def test_default_provider_limits_apply_to_a_session_created_by_default() -> None:
    """`default_provider_limits` are token limits every default-created session inherits."""
    manager = TokenBudgetManager(default_provider_limits={"ollama": 10})
    manager.record_usage("s", TokenUsage(provider="ollama", input_tokens=10))

    assert manager.check_budget("s", provider="ollama").reason == (
        "Provider 'ollama' token limit exceeded: 10/10"
    )


def test_a_refusal_that_estimates_contributed_to_says_so() -> None:
    """An estimated step is booked like a count, and a refusal it helped cause names it (#916).

    Both refusals carry the note after their figures; an allowed decision has no reason.
    """
    manager = TokenBudgetManager()
    manager.configure_session("s", max_tokens=100, provider_limits={"ollama": 20})
    manager.record_usage(
        "s",
        TokenUsage(provider="ollama", input_tokens=20, count_source=TokenCountSource.ESTIMATE),
    )
    note = " (includes 1 estimated step(s): the provider reported no usage)"

    assert manager.check_budget("s").allowed is True
    assert manager.check_budget("s", provider="ollama").reason == (
        f"Provider 'ollama' token limit exceeded: 20/20{note}"
    )

    manager.record_usage("s", TokenUsage(provider="openai", input_tokens=80))
    assert manager.check_budget("s").reason == f"Session token limit exceeded: 100/100{note}"


def test_reconfiguring_a_session_carries_its_token_counters_forward() -> None:
    """`configure_session` changes the limits, not what the session has already used.

    Dropping the counters would let a reconfigured session start its new ceiling and its
    provider quotas from zero.
    """
    manager = TokenBudgetManager()
    manager.configure_session("s", max_tokens=1_000_000)
    manager.record_usage("s", TokenUsage(provider="ollama", input_tokens=40, output_tokens=10))
    manager.configure_session("s", max_tokens=50, provider_limits={"ollama": 50})

    budget = manager.get_budget("s")
    assert budget is not None
    assert (budget.max_tokens, budget.used_input_tokens, budget.used_output_tokens) == (50, 40, 10)
    assert dict(budget.per_provider) == {"ollama": 50}
    assert manager.check_budget("s").allowed is False
    assert manager.check_budget("s", provider="ollama").allowed is False


def test_the_summary_breaks_usage_out_by_provider_in_tokens() -> None:
    """Each provider entry totals its tokens and lists its models once, in order of first use."""
    manager = TokenBudgetManager()
    manager.configure_session("s", max_tokens=1000)
    manager.record_usage("s", TokenUsage(provider="openai", model="gpt-4o", input_tokens=100))
    manager.record_usage(
        "s", TokenUsage(provider="openai", model="gpt-4o-mini", input_tokens=10, output_tokens=5)
    )
    manager.record_usage("s", TokenUsage(provider="openai", model="gpt-4o", output_tokens=20))
    manager.record_usage("s", TokenUsage(provider="ollama", input_tokens=7))
    manager.record_usage("other", TokenUsage(provider="anthropic", input_tokens=1))

    summary = manager.get_summary("s")

    assert summary["providers"] == {
        "openai": {
            "provider": "openai",
            "input_tokens": 110,
            "output_tokens": 25,
            "models": ["gpt-4o", "gpt-4o-mini"],
        },
        "ollama": {"provider": "ollama", "input_tokens": 7, "output_tokens": 0, "models": []},
    }
    assert (summary["total_tokens"], summary["prompt_tokens"], summary["completion_tokens"]) == (
        142,
        117,
        25,
    )
    assert summary["session_budget"] == {
        "max_tokens": 1000,
        "used_input_tokens": 117,
        "used_output_tokens": 25,
        "total_used_tokens": 142,
        "remaining_tokens": 858,
        "budget_used_pct": 14.2,
    }
    assert set(manager.get_summary()["providers"]) == {"openai", "ollama", "anthropic"}
