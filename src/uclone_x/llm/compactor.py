"""Context compactor and semantic pruner (Principle 5)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.core.secrets import redact_credentials
from uclone_x.core.tool_results import (
    ResultBodies,
    handle_in,
    stored_result_stub,
)
from uclone_x.errors import UnmappableChatMessageError
from uclone_x.llm.models import (
    IMAGE_TOKEN_ESTIMATE,
    ChatMessage,
    CompactionOutcome,
    LedgerSource,
    LLMRequest,
    MessageRole,
    RenderedFrom,
    ToolCallRequest,
)
from uclone_x.llm.protocols import (
    ContextCompactorProtocol,
    LLMProviderProtocol,
    request_for_model,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_LEDGERS = 2
MAX_PERMITTED_LEDGERS = 5

# Rationale for DEFAULT_MAX_LEDGERS = 2 and MAX_PERMITTED_LEDGERS = 5 (#196, #204):
# In an 8K window, 2 resident heuristic ledgers consume ~1,950 tokens (24% of the 8K context limit),
# which leaves ample headroom below the 70% compaction trigger (5,734 tokens). This preserves
# both the most recent compaction summary and its immediate predecessor while guaranteeing
# that resident ledgers alone never cause a self-retriggering compaction loop.
#
# Since #584 a ledger written with a summarizer carries the structural record *and* the
# model's prose, which enlarges the resident block. Measured over 6 successive passes at
# max_ledgers=2 by `test_repeated_compaction_bounds_the_resident_block_with_a_summarizer`:
# the block plateaus at 1,296 tokens with a terse summarizer and at 2,274 tokens (27.8% of
# an 8K window) with prose at the 500-token `max_tokens` ceiling `_generate_llm_summary`
# sets. Larger than the 24% above, still well below the 5,734-token trigger, and flat
# rather than growing -- so the headroom argument holds with a summarizer as without one.
# Those figures are asserted by that test rather than argued here, because a bound stated
# only in a comment is a bound nothing checks.
#
# Bounded above at MAX_PERMITTED_LEDGERS (5):
# At max_ledgers=6 or higher, steady-state resident tokens reach >= 5,826 tokens (> 71% of an 8K window),
# which immediately re-crosses the 70% threshold upon turn entry and reproduces #196's severity
# regardless of dialogue pruning. Therefore max_ledgers is strictly bounded: 1 <= max_ledgers <= 5.

# This module's identity when it is itself the producer of a returned value, rather than
# relaying one a provider produced. Used for the heuristic ledger and the tool-output
# pruning pass, both of which are genuine local executions of the declared algorithm —
# not substitutions for a failed provider call — so they attribute as `path=primary`
# with `requested == served_by`, and `degraded` computes False.
_COMPACTOR_PROVIDER = "uclone_x.llm.compactor"

# How the heuristic ledger renders a turn whose `content` is `None`. It is a statement
# about the record — "nothing was recorded here" — and not a stand-in for the text that
# would have been there, which is what `msg.content or ""` produced: a turn holding
# `None` and a turn holding `""` rendered as the same empty bullet, so the model read one
# fact where the history held two (P6, #385). The marker is distinguishable, not escaped:
# a user whose message is literally this string yields the same bullet. That is stated
# rather than claimed away, and is strictly narrower than collapsing every absent content
# onto the empty string.
_NO_CONTENT_MARKER = "[no content recorded]"
_HEURISTIC_LEDGER_MODEL = "heuristic-ledger"
_TOOL_PRUNER_MODEL = "tool-output-pruner"


def estimate_text_tokens(text: str) -> int:
    """Estimate the tokens `text` costs: one per four UTF-8 bytes, and at least one.

    The shared token estimator (#939). The compaction trigger uses it, and so does every
    connector that has to stand in for a count the provider did not report; such a figure
    is labelled `TokenCountSource.ESTIMATE`, `MockLLMConnector`'s included (#983). So does
    `TurnExecutor._estimate_stream_usage`, the agent's estimate for a stream that sent no count,
    through the same `estimate_request_tokens` and `estimate_reply_tokens`, so a watched step
    and a headless step are estimated alike (#980).

    **Bytes, not characters.** Byte-level BPE tokenizers, such as OpenAI's and those of the
    Llama and Qwen families Ollama commonly serves, merge UTF-8 bytes, not code points.
    Four bytes to a token is the Latin-script ratio the character count already used, and
    ASCII is one byte a character, so Latin text keeps exactly the figure `(len + 3) // 4`
    gave it. A Hangul syllable or a CJK ideograph is three bytes, so it now costs three
    quarters of a token instead of one quarter. The quarter is what undercounted Korean by
    about half against `hermes3:8b`. Design doc `unified-conversations-and-room-ui.md` §6.7
    records that measurement and the error this estimate has on it.

    This is a point estimate, not a bound. Vocabularies differ, a tokenizer that is not
    byte-level makes the byte count only a heuristic, and chat-template tokens are not
    text, so the estimator claims nothing in either direction.
    """
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


def _tool_call_tokens(tool_call: ToolCallRequest) -> int:
    """A tool call's framing, its id and name, and its JSON arguments."""
    total = 4 + estimate_text_tokens(tool_call.id + tool_call.name)
    if tool_call.arguments:
        total += estimate_text_tokens(
            json.dumps(unwrap_immutable(tool_call.arguments), ensure_ascii=False)
        )
    return total


def estimate_message_tokens(messages: Sequence[ChatMessage]) -> int:
    """Estimate a message sequence: four tokens of framing per message, plus its text.

    An image counts `IMAGE_TOKEN_ESTIMATE` whatever its size: providers price a picture by
    its pixels, not its bytes, and the figure is the order of one ~1.2 MP screenshot.
    """
    total = 0
    for msg in messages:
        total += 4 + IMAGE_TOKEN_ESTIMATE * len(msg.images)
        if msg.content:
            total += estimate_text_tokens(msg.content)
        if msg.name:
            total += estimate_text_tokens(msg.name)
        for tool_call in msg.tool_calls:
            total += _tool_call_tokens(tool_call)
    return total


def _kept_ledger_origins(ledger_at: Sequence[int], retained: Sequence[ChatMessage]) -> list[int]:
    """The input indices of the prior ledgers a pass keeps (#1848).

    `_retain_ledgers` keeps the newest, so they are the last of `ledger_at`, the indices
    of every prior ledger in order.
    """
    return list(ledger_at[len(ledger_at) - len(retained) :])


def _turn_boundary_cut(dialog: Sequence[ChatMessage], keep: int) -> int | None:
    """Where to split `dialog` into summarised and kept parts: always a user turn's start.

    A `USER` message is the one place a cut can never split a tool-call group: every
    `TOOL` result follows the `ASSISTANT` message that asked for it and precedes the next
    user turn, so all of a group lies on one side. It also keeps the kept window starting
    on a user turn.

    The latest boundary at or before `len(dialog) - keep` is preferred, so at least
    `keep` messages are kept. When there is none, the earliest one after it is taken:
    fewer than `keep` messages are kept, but whole turns, and the last user turn is
    never summarised. A dialogue no longer than `keep` is not cut at all. Index 0 is
    excluded -- a cut there summarises nothing -- so `None` means there is nothing to
    summarise, and the caller only prunes. A single long turn therefore shrinks by
    pruning its tool results, not by summarising its own question away.
    """
    if len(dialog) <= keep:
        return None
    boundaries = [i for i in range(1, len(dialog)) if dialog[i].role == MessageRole.USER]
    at_or_before = [i for i in boundaries if i <= len(dialog) - keep]
    if at_or_before:
        return at_or_before[-1]
    return boundaries[0] if boundaries else None


def unseen_step_start(messages: Sequence[ChatMessage]) -> int | None:
    """Where the trailing tool-call group of `messages` starts, or `None` if there is none.

    The group is an `ASSISTANT` message carrying `tool_calls` followed only by its `TOOL`
    results, at the very end. Between two steps of a turn this is the step that just ran:
    the model has not seen its results yet, so the step budget may still cut them to
    shares, or withhold the step when it is refused (#1480, #1509). Nothing a request
    already carried is touched.
    """
    i = len(messages)
    while i > 0 and messages[i - 1].role == MessageRole.TOOL:
        i -= 1
    if i == len(messages) or i == 0:
        return None
    start = i - 1
    head = messages[start]
    if head.role != MessageRole.ASSISTANT or not head.tool_calls:
        return None
    return start


def estimate_request_tokens(request: LLMRequest) -> int:
    """Estimate a request's input: its messages and the tool definitions sent with them.

    The tool definitions are counted because providers count them. For an agent with a
    dozen tools, the schemas can outweigh the conversation.
    """
    total = estimate_message_tokens(request.messages)
    for tool in request.tools:
        schema = json.dumps(unwrap_immutable(tool.parameters), ensure_ascii=False)
        total += 4 + estimate_text_tokens(f"{tool.name} {tool.description} {schema}")
    return total


def estimate_reply_tokens(content: str | None, tool_calls: Sequence[ToolCallRequest] = ()) -> int:
    """Estimate a reply's output: its text and the tool calls it made, at least one."""
    total = estimate_text_tokens(content) if content else 0
    total += sum(_tool_call_tokens(tool_call) for tool_call in tool_calls)
    return max(1, total)


def _local_provenance(model: str) -> Provenance:
    """Attribution for a value this module computed itself."""
    ref = ServiceRef(provider=_COMPACTOR_PROVIDER, model=model)
    return Provenance(path=ExecutionPath.PRIMARY, requested=ref, served_by=ref)


# The one reason a ledger currently stops being carried forward. Kept as a reason-keyed
# mapping rather than a bare counter, following `TelemetryTracer.drop_reasons` (#192):
# a second cause can be added without changing what a caller reads.
_SUPERSEDED_BY_CAP = "ledger_cap"


class ContextCompactor(ContextCompactorProtocol):
    """Automatic context window compactor and semantic pruner.

    Triggered when conversation token consumption reaches a threshold (default 70%
    of context limit). Anchors system state and ontology, retains the most recent N
    turns, and compacts intermediate dialogue and tool outputs into a structured
    Session Progress Ledger.

    The emitted ledger carries `role=SYSTEM` and is therefore outside everything this
    class prunes: `_prune_tool_message` only rewrites `MessageRole.TOOL`, and the
    `keep_recent_turns` window is computed over `dialog_messages`, which excludes
    `SYSTEM`. So a compactor that passed its own prior output through untouched grew the
    permanently-resident block by one ledger per pass while shrinking the dialogue —
    measured on the pre-fix code at **939 tokens per pass with no summarizer**, which is
    the default, linear and unbounded, and which at an 8K context window puts the
    resident block alone over the 70% trigger in 7 compactions (issue #196).

    With a summarizer the rate is this module's own record *plus* the size of whatever
    summary the provider returns — up to roughly 500 tokens under the `max_tokens=500`
    that `_generate_llm_summary` sets. Unbounded above by the summarizer's verbosity, in
    other words — which is why the bound below cannot be made to depend on a summarizer
    being configured. `max_ledgers` bounds it in every configuration.

    A configured summarizer **adds to** the ledger rather than replacing it (#584): the
    structural record of the discarded turns is emitted on every pass, and the model's
    prose is layered on top. Retention of the concrete content of those turns is
    therefore a property of this code in every configuration, not of the provider's
    phrasing. `ledger_source` still says `LLM` when a model contributed text, and the
    outcome still relays that model's own provenance verbatim: those answer "who wrote
    the prose", which is the question they were introduced to answer (#183). One
    consequence is stated rather than hidden: because the record is now built on every
    pass, a `TOOL` message with no `name` is refused (see `_build_heuristic_ledger`)
    even when a summarizer is configured, where previously the summarizer path never
    reached that check.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        keep_recent_turns: int = 4,
        max_tool_output_chars: int = 500,
        summarizer: LLMProviderProtocol | None = None,
        max_ledgers: int = DEFAULT_MAX_LEDGERS,
        result_bodies: ResultBodies | None = None,
        tool_result_reader: bool = True,
    ) -> None:
        if not (1 <= max_ledgers <= MAX_PERMITTED_LEDGERS):
            raise ValueError(
                f"max_ledgers must be between 1 and {MAX_PERMITTED_LEDGERS}, got {max_ledgers}: "
                f"`compact()` always emits a ledger for turns just discarded (so budget < 1 is invalid), "
                f"and values > {MAX_PERMITTED_LEDGERS} allow resident ledgers to exceed 50% of the "
                "context window, causing self-retriggering compaction loops (#196, #204)."
            )
        self.threshold = threshold
        self.keep_recent_turns = keep_recent_turns
        self.max_tool_output_chars = max_tool_output_chars
        self.summarizer = summarizer
        self.max_ledgers = max_ledgers
        # The session's full tool-result bodies (#1848): where a pruned output is kept in
        # full and a stored result is read back to stub it. The agent sets it per pass
        # (`CompactionDriver`); without it, an output is truncated and a stored result
        # is left as it is.
        self.result_bodies = result_bodies
        # Whether the agent this compactor serves can call `tool_result_read`. The short
        # form of a stored result names that tool, so it is used only when the tool can
        # be called; otherwise the result is pruned like any output (#1422, P6).
        self.tool_result_reader = tool_result_reader
        # Ledgers this compactor stopped carrying forward, and why. A cap that discards
        # silently is the P6 shape #192 named -- a control that loses data without saying
        # so -- and it is worse here than for a span, because a ledger is the only
        # surviving record of dialogue turns already discarded. So the loss is counted on
        # an attribute a caller and a test can read, and noted in band on the replacement
        # ledger for a caller that only ever sees messages.
        self._superseded_ledger_count: int = 0
        self._supersession_reasons: dict[str, int] = {}

    @property
    def superseded_ledger_count(self) -> int:
        """Prior ledgers **this compactor instance** stopped carrying forward (#196).

        Per instance, not per session: a caller that constructs a `ContextCompactor` per
        turn gets a counter that resets with it, while the per-pass figure in each
        ledger's in-band note stays exact. A true session total would have to be derived
        from the input rather than accumulated here.
        """
        return self._superseded_ledger_count

    @property
    def supersession_reasons(self) -> Mapping[str, int]:
        """Supersession counts by reason — `ledger_cap` today."""
        return dict(self._supersession_reasons)

    def _record_supersession(self, count: int, reason: str) -> None:
        """Account for ledgers dropped from the resident block."""
        if count <= 0:
            return
        first_of_this_reason = reason not in self._supersession_reasons
        self._superseded_ledger_count += count
        self._supersession_reasons[reason] = self._supersession_reasons.get(reason, 0) + count
        # The counter is the observable; the log is a hint towards it. Warn once per
        # reason, as `TelemetryTracer._record_drop` does: a long session supersedes one
        # ledger per compaction forever, and a warning each time would bury the first.
        if first_of_this_reason:
            logger.warning(
                "Superseding compaction ledger(s) (%s); first occurrence, %d so far. "
                "Further supersessions for this reason are counted on "
                "`superseded_ledger_count` and logged at debug level.",
                reason,
                self._superseded_ledger_count,
            )
        else:
            logger.debug(
                "Superseded %d more compaction ledger(s) (%s); %d total",
                count,
                reason,
                self._superseded_ledger_count,
            )

    def _retain_ledgers(
        self, prior_ledgers: Sequence[ChatMessage], budget: int
    ) -> list[ChatMessage]:
        """Keep the newest `budget` prior ledgers, accounting for the rest.

        Newest-wins, and the choice is not free: a ledger is never fed back into
        summarization (it is `SYSTEM`, and only `dialog_messages` reach the summarizer),
        so a superseded ledger's content is genuinely lost rather than folded forward.
        The newest are kept because they cover the turns discarded most recently, which
        are the ones most likely to bear on the current turn; the loss of the rest is
        counted, and named on the replacement ledger.
        """
        split = max(len(prior_ledgers) - max(budget, 0), 0)
        self._record_supersession(split, _SUPERSEDED_BY_CAP)
        return list(prior_ledgers[split:])

    def estimate_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """Estimate token count across a sequence of messages (`estimate_message_tokens`)."""
        return estimate_message_tokens(messages)

    def should_compact_at(self, messages: Sequence[ChatMessage], threshold_tokens: int) -> bool:
        """Determine whether message context meets or exceeds an absolute token threshold (P5)."""
        if threshold_tokens <= 0 or not messages:
            return False
        return self.estimate_tokens(messages) >= threshold_tokens

    def should_compact(self, messages: Sequence[ChatMessage], context_limit: int) -> bool:
        """Determine whether message context exceeds the compaction threshold (70%)."""
        if context_limit <= 0 or not messages:
            return False
        threshold_tokens = int(context_limit * self.threshold)
        return self.should_compact_at(messages, threshold_tokens)

    def should_compact_request_at(self, request: LLMRequest, threshold_tokens: int) -> bool:
        """Whether a whole request -- messages and tool schemas -- meets `threshold_tokens`.

        Counted with `estimate_request_tokens`, the estimator the agent's usage figures
        already use, so the trigger and the reported size of a request agree (#1422).
        """
        if threshold_tokens <= 0 or not request.messages:
            return False
        return estimate_request_tokens(request) >= threshold_tokens

    def should_compact_request(self, request: LLMRequest, context_limit: int) -> bool:
        """Whether a whole request exceeds the compaction threshold of `context_limit`."""
        if context_limit <= 0 or not request.messages:
            return False
        return self.should_compact_request_at(request, int(context_limit * self.threshold))

    def prune_tool_message(self, msg: ChatMessage) -> ChatMessage:
        """Prune or offload long tool output message if it exceeds char limit.

        An excerpt or stub of a kept result (#1422) is not offloaded a second time: its
        full text is already kept, so it drops to a stub -- its handle and the start of
        the kept text. The handle is the one the message records (`rendered_from`), never
        one read out of its text (#1848): a page of `tool_result_read` names a handle in
        its header, but it is a full result, and is pruned like any output. A form whose
        kept result this session cannot read is left as it is: there is nothing to render
        a smaller form from, and its text is not a result to keep. The stub names
        `tool_result_read` only when the agent can call it.
        """
        if msg.role != MessageRole.TOOL or not msg.content:
            return msg
        kept = msg.rendered_from.handle if msg.rendered_from is not None else None
        if kept is not None:
            return self._restub_form(msg, kept)
        if msg.content.startswith("[Tool Output Truncated"):
            return msg
        if len(msg.content) <= self.max_tool_output_chars:
            return msg

        head_len = self.max_tool_output_chars // 2

        # Offloading needs a session: the reader resolves a handle only among the
        # session's own log entries, so a body kept for no session could never be read
        # back, and its stub would name a handle nothing resolves -- and, not recognised
        # as stored, be offloaded again under a small cap (#1653). Without one it is
        # truncated.
        if self.result_bodies is not None:
            return self._offload_tool_message(msg, self.result_bodies, head_len)

        redacted_full = redact_credentials(msg.content)
        tail_len = self.max_tool_output_chars // 4
        truncated_content = (
            f"[Tool Output Truncated (path=truncate, {len(msg.content)} chars total):\n"
            f"{redacted_full[:head_len]}\n"
            f"... [omitted {len(msg.content) - (head_len + tail_len)} chars] ...\n"
            f"{redacted_full[-tail_len:]}\n]"
        )
        return ChatMessage(
            role=msg.role,
            content=truncated_content,
            name=msg.name,
            tool_call_id=msg.tool_call_id,
            tool_calls=msg.tool_calls,
            form="excerpt",
        )

    def _restub_form(self, msg: ChatMessage, handle: str) -> ChatMessage:
        """`msg`, a form of the kept result `handle`, as a stub when that is smaller."""
        body = None if self.result_bodies is None else self.result_bodies.read(handle)
        if body is None or msg.content is None:
            return msg
        keep = self.max_tool_output_chars // 2
        stub = stored_result_stub(handle, body, keep_chars=keep, readable=self.tool_result_reader)
        if len(stub) >= len(msg.content):
            return msg
        return ChatMessage(
            role=msg.role,
            content=stub,
            name=msg.name,
            tool_call_id=msg.tool_call_id,
            tool_calls=msg.tool_calls,
            form="stub",
            rendered_from=RenderedFrom(handle=handle, limit=keep, readable=self.tool_result_reader),
        )

    def _prune_tool_message(self, msg: ChatMessage) -> ChatMessage:
        return self.prune_tool_message(msg)

    def _offload_tool_message(
        self, msg: ChatMessage, bodies: ResultBodies, head_len: int
    ) -> ChatMessage:
        """Keep oversized tool output in full in the session and stub it.

        The body is kept where ingest keeps one (#1422, #1848): an entry of the session's
        own log named by its content's handle, so a reused `tool_call_id` cannot replace
        another result. The stub is rendered from that same redacted body, in the form
        compaction gives a stored excerpt, naming `tool_result_read` only when the agent
        can call it (#1640).
        """
        assert msg.content is not None
        # Redacted before it is hashed or kept (#569); the stub's start is taken from the
        # same redacted text.
        body = redact_credentials(msg.content)
        handle = bodies.keep(body, tool_name=msg.name)
        reader_offered = self.tool_result_reader
        stub = stored_result_stub(
            handle,
            body,
            keep_chars=head_len,
            readable=reader_offered,
        )
        return ChatMessage(
            role=msg.role,
            content=stub,
            name=msg.name,
            tool_call_id=msg.tool_call_id,
            tool_calls=msg.tool_calls,
            form="stub",
            rendered_from=RenderedFrom(handle=handle, limit=head_len, readable=reader_offered),
        )

    def _build_heuristic_ledger(self, middle_messages: Sequence[ChatMessage]) -> str:
        """Construct structured Session Progress Ledger from middle dialogue turns.

        The ledger is not a wire payload, but it is not prose either: it replaces the
        turns it summarises in the history the model is given, so whatever it asserts
        becomes the model's account of what happened. Two sites here asserted things
        that had not happened (P6, #385):

        * `tool_name = msg.name or "tool"` named a tool for a `TOOL` message that
          carried none. That is the fabrication `gemini.py` refuses for
          `functionResponse.name` (#380), pointed at the one party the history exists to
          inform, and indistinguishable in the ledger text from a real tool called
          `tool`. It is now refused: `ui.app`'s rehydration already refuses a `TOOL`
          record with no name (#387/PR #388) and `agent/base.py` always passes
          `name=tc.name`, so a nameless `TOOL` message here means a caller built one
          directly and the ledger cannot honestly describe it.
        * `msg.content or ""` rendered a `USER` turn with no recorded content and one
          holding the empty string as the same empty bullet. `None` now renders as
          `_NO_CONTENT_MARKER`, which is a statement about the record rather than a
          substituted value. The marker is a distinguishable rendering, not an escaped
          one: a user who types that exact string produces the same bullet. That
          residual ambiguity is stated rather than claimed away, and it is strictly
          narrower than the previous one, which collapsed *every* absent content onto
          the empty string.
        * `elif msg.content:` on the `ASSISTANT` branch emitted **no bullet at all** for
          a turn holding `""`, so an assistant turn that said the empty string and one
          that recorded nothing both disappeared from the model's own history — the same
          filter-makes-two-inputs-one defect as the `SYSTEM` site PR #384's review found,
          reached by omission. Both now render, symmetrically with `USER`.

        Examined and left alone: `estimate_message_tokens`'s `if msg.content:` /`if msg.name:`.
        Those guard an arithmetic contribution to an estimate rather than a value that
        stands in for anything, and the whole difference `""` makes is one token of
        framing. Recorded because a list of nearby truthiness guards invites the
        assumption that all of them are suspect.
        """
        bullets: list[str] = []
        for i, msg in enumerate(middle_messages, 1):
            if msg.role == MessageRole.USER:
                if msg.content is None:
                    content_preview = _NO_CONTENT_MARKER
                elif len(msg.content) > 150:
                    content_preview = msg.content[:150] + "..."
                else:
                    content_preview = msg.content
                bullets.append(f"• User Request (turn {i}): {content_preview}")
            elif msg.role == MessageRole.ASSISTANT:
                if msg.tool_calls:
                    tools_used = ", ".join(tc.name for tc in msg.tool_calls)
                    bullets.append(f"• Agent Action (turn {i}): Invoked tool(s) [{tools_used}]")
                else:
                    if msg.content is None:
                        content_preview = _NO_CONTENT_MARKER
                    elif len(msg.content) > 150:
                        content_preview = msg.content[:150] + "..."
                    else:
                        content_preview = msg.content
                    bullets.append(f"• Agent Reasoning (turn {i}): {content_preview}")
            elif msg.role == MessageRole.TOOL:
                if msg.name is None or not msg.name.strip():
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role='tool', tool_call_id={msg.tool_call_id!r}) at "
                        f"turn {i} has name={msg.name!r}, so the ledger cannot say which "
                        "tool produced this result. It is not defaulted to 'tool', because "
                        "the ledger becomes the model's own record of the session: a "
                        "placeholder does not lose the name, it attributes the result to a "
                        "tool that never ran, under a name indistinguishable from a real "
                        "one (P6, #385)."
                    )
                # The kept result a form records; else a page's, whose header names it.
                handle = (
                    msg.rendered_from.handle
                    if msg.rendered_from is not None
                    else handle_in(msg.content)
                )
                if handle is not None:
                    bullets.append(
                        f"• Tool Result (turn {i}, {msg.name}): completed execution (stored result: {handle})"
                    )
                else:
                    bullets.append(f"• Tool Result (turn {i}, {msg.name}): completed execution")

        ledger_content = (
            f"[Context Auto-Compacted Summary: Heuristic Session Progress Ledger]\n"
            f"Preserved {len(middle_messages)} intermediate turns summarized below:\n"
            + "\n".join(bullets)
        )
        return redact_credentials(ledger_content)

    async def _generate_llm_summary(
        self, middle_messages: Sequence[ChatMessage]
    ) -> tuple[str, Provenance | None] | None:
        """Attempt to generate an LLM-assisted summary of middle messages.

        Returns the ledger text together with **the summarizer's own** `provenance`,
        forwarded verbatim, or `None` when no LLM ledger was produced. Returning the
        attribution alongside the text is what lets `compact` say which producer wrote
        the ledger without a caller having to substring-match the in-band label.

        The provenance is never synthesized here. If a connector attributed its response
        with `None`, that `None` travels — for the same reason `BaseAgent.execute_turn`
        propagates a connector's attribution including its absence: naming this module
        as the server of text a model produced would turn "not stated" into a positively
        asserted clean result.

        Per P6, provider exceptions propagate rather than silently falling back to
        an unattributed heuristic ledger.

        The remaining ambiguity is deliberately left alone and reported rather than
        changed here: a *configured* summarizer that returns empty content still falls
        back to the heuristic ledger. That fallback is now at least **attributable** —
        the outcome says `HEURISTIC` and carries this module's own provenance, so it is
        no longer silent. Whether an empty provider response should instead propagate as
        a failure is a question about the compaction algorithm's P6 posture, not about
        the agent wiring #183 adds, so it is filed separately rather than bundled.
        """
        if self.summarizer is None:
            return None

        summary_request = LLMRequest(
            model="",
            messages=(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You are an expert context summarizer. Summarize the key actions, decisions, and outcomes from the following conversation turns into a concise progress ledger.",
                ),
                *middle_messages,
                ChatMessage(
                    role=MessageRole.USER,
                    content="Provide a concise 3-5 bullet point Session Progress Ledger summarizing our progress and key state.",
                ),
            ),
            temperature=0.2,
            max_tokens=500,
            auto_compact=False,
        )
        # The turns summarised can hold images the summarizer's model cannot see (#2123).
        summary_request = await request_for_model(self.summarizer, summary_request)
        response = await self.summarizer.generate(summary_request)
        if response.content:
            return (
                f"[Context Auto-Compacted Summary: LLM Session Progress Ledger]\n"
                f"{redact_credentials(response.content.strip())}",
                response.provenance,
            )
        return None

    async def compact(self, messages: Sequence[ChatMessage]) -> CompactionOutcome:
        """Prune and summarize old turns, preserving recent history and system prompts.

        The returned `messages` never carry more than `max_ledgers` compaction ledgers,
        so the resident `SYSTEM` block is bounded across any number of successive passes.

        Returns a `CompactionOutcome` rather than a bare tuple so the pass says which
        producer wrote its ledger and carries P6 attribution for it. See
        `CompactionOutcome` for why the message-borne `compaction_ledger` flag cannot
        answer that question.
        """
        if not messages:
            return CompactionOutcome(
                messages=(),
                ledger_source=LedgerSource.NONE,
                provenance=_local_provenance(_TOOL_PRUNER_MODEL),
            )

        # 1. Anchored system state, split from the compaction artifacts sitting beside
        #    it. The split is on `compaction_ledger`, not on the in-band label: the label
        #    is provenance for a reader, and branching on its text would turn it into a
        #    control channel (#196).
        anchor_messages = [
            m for m in messages if m.role == MessageRole.SYSTEM and not m.compaction_ledger
        ]
        prior_ledgers = [
            m for m in messages if m.role == MessageRole.SYSTEM and m.compaction_ledger
        ]
        dialog_messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        # Where each output message comes from, by its index in `messages` (#1848): the
        # three groups above, in the same order, so a kept or pruned message names the
        # message it renders and the ledger this pass writes names none.
        anchor_at = [
            i
            for i, m in enumerate(messages)
            if m.role == MessageRole.SYSTEM and not m.compaction_ledger
        ]
        ledger_at = [
            i
            for i, m in enumerate(messages)
            if m.role == MessageRole.SYSTEM and m.compaction_ledger
        ]
        dialog_at = [i for i, m in enumerate(messages) if m.role != MessageRole.SYSTEM]

        # Where the summarised part ends. Only ever before a `USER` message (#1422): a
        # cut at `-keep_recent_turns` fell wherever the count landed, and could keep a
        # `TOOL` result whose call was summarised away, summarise a result whose call was
        # kept, or start the kept window on an `ASSISTANT` turn -- each a request some
        # providers refuse and all of them a history that no longer says what happened.
        cut = _turn_boundary_cut(dialog_messages, self.keep_recent_turns)

        # If dialogue length is within recent window, or no turn boundary lies before it,
        # only prune long tool outputs. No new ledger is emitted, so the whole budget is
        # available to the prior ones.
        if cut is None:
            retained_ledgers = self._retain_ledgers(prior_ledgers, self.max_ledgers)
            pruned_dialog = [self._prune_tool_message(m) for m in dialog_messages]
            kept_ledgers = _kept_ledger_origins(ledger_at, retained_ledgers)
            return CompactionOutcome(
                messages=(*anchor_messages, *retained_ledgers, *pruned_dialog),
                origins=(*anchor_at, *kept_ledgers, *dialog_at),
                ledger_source=LedgerSource.NONE,
                superseded_ledger_count=len(prior_ledgers) - len(retained_ledgers),
                provenance=_local_provenance(_TOOL_PRUNER_MODEL),
            )

        # 2. Split middle vs recent turns
        middle_messages = dialog_messages[:cut]
        recent_messages = dialog_messages[cut:]

        # Prune middle messages before summarization so oversized tool outputs
        # are safely offloaded to sandbox files and referenced in the ledger
        pruned_middle = [self._prune_tool_message(m) for m in middle_messages]

        # 3. Summarize middle turns. The producer is recorded as the summary is built,
        #    not inferred afterwards from the text: the in-band label is provenance for a
        #    human reader, and reading it back would make it a control channel (#196).
        #
        #    The structural record is built unconditionally and is always carried, with
        #    the summarizer's prose layered on top of it rather than substituted for it
        #    (#584). Before this, a configured summarizer *replaced* the record, so the
        #    only surviving trace of the discarded turns was whatever the model chose to
        #    write — and the concrete values in those turns (identifiers, paths, stated
        #    constraints) survived exactly to the extent that the model happened to
        #    restate them. Measured on the compaction battery with `--provider mock`,
        #    where the summarizer is a stub returning one canned sentence: needle recall
        #    fell to 75% / 35% / 4.2% across three passes, against 100% / 100% / 87.5%
        #    for the same battery on this module's own record. A summarizer scripted to
        #    restate every needle reached exactly the record's own 87.5% and no better,
        #    which is the point: prose is at best a tie and at worst a total loss, so
        #    retention must not be entrusted to it. That is P5's whole reason for
        #    existing, and it is not a property a provider can be relied on to supply.
        structural_record = self._build_heuristic_ledger(pruned_middle)
        llm_summary = await self._generate_llm_summary(pruned_middle)
        if llm_summary is not None:
            llm_text, ledger_provenance = llm_summary
            summary_text = f"{llm_text}\n\n{structural_record}"
            ledger_source = LedgerSource.LLM
        else:
            summary_text = structural_record
            ledger_provenance = _local_provenance(_HEURISTIC_LEDGER_MODEL)
            ledger_source = LedgerSource.HEURISTIC

        # 4. Make room for the ledger about to be emitted, and say in band what that
        #    cost. The count is read back off `_retain_ledgers` rather than recomputed,
        #    so the budget arithmetic lives in exactly one place; the note is built after
        #    the accounting so it can quote the running total, and appended after the
        #    summary so the label line stays byte-identical.
        #
        #    The note says "by this compactor", not "this session", and the distinction is
        #    load-bearing rather than pedantic: under per-turn construction -- the wiring
        #    the message-borne flag was chosen to survive, and the one #183 has not ruled
        #    out -- the cap still holds and the per-pass figure is still exact, but the
        #    running total resets with the instance. Measured, a fresh compactor per pass
        #    reports 1 at pass 8 where a shared one reports 6. In-band provenance must not
        #    state a magnitude it cannot know (issue #196 review).
        retained_ledgers = self._retain_ledgers(prior_ledgers, self.max_ledgers - 1)
        superseded_now = len(prior_ledgers) - len(retained_ledgers)
        if superseded_now:
            summary_text = (
                f"{summary_text}\n"
                f"[Superseded {superseded_now} earlier compaction ledger(s) at this pass "
                f"({self._superseded_ledger_count} by this compactor); their content is "
                f"not carried forward.]"
            )

        summary_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content=summary_text,
            compaction_ledger=True,
        )

        # 5. Prune oversized tool outputs in recent turns
        pruned_recent = [self._prune_tool_message(m) for m in recent_messages]

        kept_ledgers = _kept_ledger_origins(ledger_at, retained_ledgers)
        return CompactionOutcome(
            messages=(*anchor_messages, *retained_ledgers, summary_msg, *pruned_recent),
            origins=(*anchor_at, *kept_ledgers, None, *dialog_at[cut:]),
            ledger_source=ledger_source,
            superseded_ledger_count=superseded_now,
            provenance=ledger_provenance,
        )
