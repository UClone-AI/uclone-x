"""Token budget manager and quota controller (Principle 5 & Principle 6)."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from uclone_x.errors import BudgetExceededError
from uclone_x.llm.models import BudgetDecision, TokenBudget, TokenCountSource, TokenUsage
from uclone_x.llm.protocols import TokenBudgetManagerProtocol

__all__ = [
    "TokenBudgetManager",
    "TokenBudgetTracker",
]

#: The turns the running code belongs to: each `collect_turn_usage` it runs inside (#982).
#: A task copies its context when it is created, so the tasks a turn starts belong to it,
#: and a turn running in another request's context does not, whatever session it shares.
_CURRENT_TURNS: contextvars.ContextVar[tuple[list[TokenUsage], ...]] = contextvars.ContextVar(
    "uclone_x_current_turns", default=()
)


class TokenBudgetManager(TokenBudgetManagerProtocol):
    """Session-level and provider-level token budget manager.

    Enforces a session token ceiling and decoupled per-provider token quotas. Under
    Principle 6, budget violations fail fast and propagate as `BudgetExceededError`.
    Cost is not calculated: UClone-X counts tokens only (#1392).
    """

    def __init__(
        self,
        default_max_tokens: int = 1_000_000,
        default_provider_limits: Mapping[str, int] | None = None,
    ) -> None:
        self._default_max_tokens = default_max_tokens
        self._default_provider_limits: dict[str, int] = (
            dict(default_provider_limits) if default_provider_limits else {}
        )
        self._budgets: dict[str, TokenBudget] = {}
        self._session_provider_limits: dict[str, dict[str, int]] = {}
        self._turn_history: dict[str, list[TokenUsage]] = {}
        # The turns open on each session, each the list `collect_turn_usage` yields (#982).
        self._open_turns: dict[str, list[list[TokenUsage]]] = {}
        self._compaction_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def set_budget(
        self,
        session_id: str,
        budget: TokenBudget,
        provider_limits: Mapping[str, int] | None = None,
    ) -> None:
        """Explicitly set the active budget and optional per-provider token limits."""
        with self._lock:
            self._budgets[session_id] = budget
            if provider_limits is not None:
                self._session_provider_limits[session_id] = dict(provider_limits)

    def configure_session(
        self,
        session_id: str,
        max_tokens: int = 1_000_000,
        provider_limits: Mapping[str, int] | None = None,
    ) -> TokenBudget:
        """Create or configure a session budget with custom limits. Counters carry over."""
        with self._lock:
            existing = self._budgets.get(session_id)
            budget = TokenBudget(
                max_tokens=max_tokens,
                used_input_tokens=existing.used_input_tokens if existing else 0,
                used_output_tokens=existing.used_output_tokens if existing else 0,
                per_provider=dict(existing.per_provider) if existing else {},
            )
            self._budgets[session_id] = budget
            if provider_limits is not None:
                self._session_provider_limits[session_id] = dict(provider_limits)
            return budget

    def get_budget(self, session_id: str) -> TokenBudget | None:
        """Retrieve the active budget snapshot for a session."""
        with self._lock:
            return self._budgets.get(session_id)

    def _get_or_create_budget(self, session_id: str) -> TokenBudget:
        """Internal helper to get existing or create default budget for a session."""
        if session_id not in self._budgets:
            self._budgets[session_id] = TokenBudget(max_tokens=self._default_max_tokens)
            if self._default_provider_limits:
                self._session_provider_limits[session_id] = dict(self._default_provider_limits)
        return self._budgets[session_id]

    def record_usage(self, session_id: str, usage: TokenUsage) -> None:
        """Record token usage, attributed to `usage.provider`.

        An estimate (`usage.count_source is TokenCountSource.ESTIMATE`) is booked exactly
        like a count, and keeps its label on the ledger, where a refusal can state it (#916).
        """
        with self._lock:
            current = self._get_or_create_budget(session_id)

            new_per_prov: dict[str, int] = dict(current.per_provider)
            new_per_prov[usage.provider] = new_per_prov.get(usage.provider, 0) + (
                usage.input_tokens + usage.output_tokens
            )

            updated = TokenBudget(
                max_tokens=current.max_tokens,
                used_input_tokens=current.used_input_tokens + usage.input_tokens,
                used_output_tokens=current.used_output_tokens + usage.output_tokens,
                per_provider=new_per_prov,
            )
            self._budgets[session_id] = updated

            if session_id not in self._turn_history:
                self._turn_history[session_id] = []
            self._turn_history[session_id].append(usage)

            ours = _CURRENT_TURNS.get()
            for turn in self._open_turns.get(session_id, ()):
                if any(turn is mine for mine in ours):
                    turn.append(usage)

    @contextmanager
    def collect_turn_usage(self, session_id: str) -> Generator[list[TokenUsage], None, None]:
        """Collect the usages this turn books on `session_id`, as they are booked (#982).

        The ledger is keyed by session id, so a slice of `get_turn_history` taken across a
        turn holds every step booked on the session meanwhile, including another turn's when
        two overlap: a second client on the same agent, or two agents given one session id.
        A booking made inside this block, or in a task started from it, lands in the
        yielded list; one made by code running in another context does not. The list stays
        readable after the block closes, and nothing is added to it afterwards.
        """
        # A task started inside this block keeps the turn in its context for as long as it
        # runs, but it collects nothing once the block closes, because the turn has left
        # `_open_turns`. So a long-lived task first started mid-turn (the event bus
        # dispatcher, an agent started from a tool) books later steps to the session only.
        booked: list[TokenUsage] = []
        with self._lock:
            self._open_turns.setdefault(session_id, []).append(booked)
        token = _CURRENT_TURNS.set((*_CURRENT_TURNS.get(), booked))
        try:
            yield booked
        finally:
            _CURRENT_TURNS.reset(token)
            with self._lock:
                still_open = [turn for turn in self._open_turns[session_id] if turn is not booked]
                if still_open:
                    self._open_turns[session_id] = still_open
                else:
                    del self._open_turns[session_id]

    def check_budget(self, session_id: str, provider: str | None = None) -> BudgetDecision:
        """Report whether a session — optionally for one provider — may proceed.

        The session token ceiling is checked first, then `provider`'s token limit if the
        session has one.
        """
        with self._lock:
            budget = self._get_or_create_budget(session_id)
            history = self._turn_history.get(session_id, ())
            total_used_tokens = budget.used_input_tokens + budget.used_output_tokens
            rem_tokens = max(0, budget.max_tokens - total_used_tokens)
            # A refusal is where an estimated figure acts rather than merely misreports,
            # so a refusal that estimates contributed to says so (P6, #916).
            estimated_steps = sum(
                1 for usage in history if usage.count_source is TokenCountSource.ESTIMATE
            )
            estimate_note = (
                f" (includes {estimated_steps} estimated step(s): the provider reported no usage)"
                if estimated_steps
                else ""
            )

            if total_used_tokens >= budget.max_tokens:
                return BudgetDecision(
                    allowed=False,
                    reason=f"Session token limit exceeded: {total_used_tokens}/{budget.max_tokens}"
                    f"{estimate_note}",
                    remaining_tokens=0,
                )

            if provider is not None:
                prov_limits = self._session_provider_limits.get(
                    session_id, self._default_provider_limits
                )
                if provider in prov_limits:
                    prov_limit = prov_limits[provider]
                    prov_used = budget.per_provider.get(provider, 0)
                    if prov_used >= prov_limit:
                        return BudgetDecision(
                            allowed=False,
                            reason=f"Provider '{provider}' token limit exceeded: "
                            f"{prov_used}/{prov_limit}{estimate_note}",
                            remaining_tokens=rem_tokens,
                        )

            return BudgetDecision(allowed=True, reason=None, remaining_tokens=rem_tokens)

    def enforce_budget(self, session_id: str, provider: str | None = None) -> None:
        """Enforce budget limits immediately, raising `BudgetExceededError` on breach."""
        decision = self.check_budget(session_id=session_id, provider=provider)
        if not decision.allowed:
            raise BudgetExceededError(decision.reason or "Budget limit exceeded")

    def get_turn_history(self, session_id: str) -> tuple[TokenUsage, ...]:
        """Get chronological list of token usages for a session."""
        with self._lock:
            return tuple(self._turn_history.get(session_id, []))

    def reset_session(self, session_id: str) -> None:
        """Reset budget counters and history for a session."""
        with self._lock:
            self._budgets.pop(session_id, None)
            self._session_provider_limits.pop(session_id, None)
            self._turn_history.pop(session_id, None)
            self._compaction_history = [
                c for c in self._compaction_history if c.get("session_id") != session_id
            ]

    def record_compaction(
        self,
        reason: str,
        original_tokens: int,
        compacted_tokens: int,
        kept_turns: int = 4,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a context window compaction event in history."""
        with self._lock:
            saved = max(0, original_tokens - compacted_tokens)
            ratio = round((saved / original_tokens * 100.0), 2) if original_tokens > 0 else 0.0
            now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            entry: dict[str, Any] = {
                "id": f"cmp_{len(self._compaction_history) + 1}",
                "timestamp": now_iso,
                "reason": reason,
                "original_tokens": original_tokens,
                "compacted_tokens": compacted_tokens,
                "saved_tokens": saved,
                "compression_ratio_pct": ratio,
                "kept_turns": kept_turns,
                "session_id": session_id,
            }
            self._compaction_history.append(entry)
            return entry

    def get_summary(self, session_id: str | None = None) -> dict[str, Any]:
        """Return token budget metrics, per-provider attribution, and compaction history."""
        with self._lock:
            if session_id is not None:
                if session_id in self._budgets:
                    budget = self._budgets[session_id]
                    used_in = budget.used_input_tokens
                    used_out = budget.used_output_tokens
                    total_used = used_in + used_out
                    max_tok = budget.max_tokens
                else:
                    used_in = 0
                    used_out = 0
                    total_used = 0
                    max_tok = self._default_max_tokens
                usages = self._turn_history.get(session_id, [])
                compactions = [
                    c for c in self._compaction_history if c.get("session_id") == session_id
                ]
            else:
                used_in = sum(b.used_input_tokens for b in self._budgets.values())
                used_out = sum(b.used_output_tokens for b in self._budgets.values())
                total_used = used_in + used_out
                max_tok = (
                    sum(b.max_tokens for b in self._budgets.values())
                    if self._budgets
                    else self._default_max_tokens
                )
                usages = [u for turn_list in self._turn_history.values() for u in turn_list]
                compactions = list(self._compaction_history)

            rem_tokens = max(0, max_tok - total_used)
            pct_used = round((total_used / max_tok * 100.0), 2) if max_tok > 0 else 0.0

            # Provider breakdown
            providers: dict[str, dict[str, Any]] = {}
            for u in usages:
                prov = u.provider or "unknown"
                if prov not in providers:
                    providers[prov] = {
                        "provider": prov,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "models": [],
                    }
                providers[prov]["input_tokens"] += u.input_tokens
                providers[prov]["output_tokens"] += u.output_tokens
                if u.model and u.model not in providers[prov]["models"]:
                    providers[prov]["models"].append(u.model)

            roles: dict[str, dict[str, Any]] = {}

            return {
                "total_tokens": total_used,
                "prompt_tokens": used_in,
                "completion_tokens": used_out,
                "session_budget": {
                    "max_tokens": max_tok,
                    "used_input_tokens": used_in,
                    "used_output_tokens": used_out,
                    "total_used_tokens": total_used,
                    "remaining_tokens": rem_tokens,
                    "budget_used_pct": pct_used,
                },
                "providers": providers,
                "roles": roles,
                "compaction_history": compactions,
            }


# Backward-compatibility alias
TokenBudgetTracker = TokenBudgetManager
