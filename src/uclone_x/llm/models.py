"""Data models for LLM-agnostic provider interfaces and token budgeting."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from uclone_x.core.immutable import ImmutableIntMapping, ImmutableJsonMapping
from uclone_x.core.provenance import Provenance


class FinishReason(StrEnum):
    """Why generation stopped, as an enum rather than a commented string."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"

    UNKNOWN = "unknown"
    """The provider reported a stop reason this connector does not enumerate, or
    reported none at all.

    It exists because every connector previously ended its `_map_finish_reason` with
    `return FinishReason.STOP`, so any value the mapper did not recognise — including
    the field being absent — was reported to the caller as a clean, complete
    generation. A response truncated for a reason the provider names and the connector
    does not enumerate arrived indistinguishable from one that finished normally, and
    the substitution ran in the direction of **success**, which is the direction
    nothing prompts anyone to check (P6, #385).

    `UNKNOWN` is not `ERROR`: the call succeeded and the content is real and billed.
    Only the *reason* it stopped is unrepresentable here, so the response is delivered
    with that fact stated rather than discarded or dressed up as `STOP`. A caller that
    must distinguish a complete answer from a truncated one has to treat `UNKNOWN` as
    "not established" and cannot read it as either."""


class ToolDefinition(BaseModel):
    """A tool as declared to a provider.

    Replaces `dict[str, Any]` in the provider contract. Issue 2026-09-02-011 reports
    exactly this: an untyped tool declaration is valid to a type checker, so P8's rule
    only bites if the contract models it. `parameters` stays a mapping because it is a
    JSON Schema document, but the envelope around it no longer is.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    description: str
    parameters: ImmutableJsonMapping = Field(
        default_factory=dict,
        description="JSON Schema for the tool's arguments.",
    )


class MessageRole(StrEnum):
    """Chat message sender role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallRequest(BaseModel):
    """Structured tool invocation request produced by an LLM."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: str
    name: str
    arguments: ImmutableJsonMapping = Field(default_factory=dict)


class RenderedFrom(BaseModel):
    """What a shortened tool result was rendered from, so it can be rendered again (#1848).

    `handle` names the full, redacted result the session log keeps (`tr_` and 16 hex
    digits). `limit` is the one size the form was cut to: for an `excerpt`, the cap in
    UTF-8 bytes (`excerpt_tool_result`'s `cap_bytes`); for a `stub`, the characters of the
    result's start it keeps (`stored_result_stub`'s `keep_chars`). `readable` says whether
    the text names `tool_result_read`. With these and the body, the form's text is a pure
    function (`core/context_state.render_form`), so a rebuild from the log needs no copy of
    the text as it was shown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    handle: str
    limit: int
    readable: bool


ImageMediaType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
"""The image formats every connector that sends images can send (#2107)."""

IMAGE_TOKEN_ESTIMATE: Final = 1_600
"""What one image is counted as when a request's tokens are estimated (#2107).

An image is billed by its pixels, not its bytes, so the byte estimator would count a
screenshot's base64 as a hundred thousand tokens. 1,600 is about what Anthropic bills an
image at its default size limit (pixels / 750, long edge 1,568), which is the largest of
the four providers' figures for a screenshot of that size, so the estimate does not run
low. A point estimate, like the text estimator.
"""

IMAGE_UNAVAILABLE_NOTE: Final = "[An image was here, but it is no longer available.]"
"""What a connector sends in place of an image whose bytes this process does not hold.

A message read back from a record keeps each image's digest, not its bytes; the bytes are
put back from the session's body store (`core/session_log.with_image_data`). When they
cannot be, the model is told an image was there rather than being sent nothing.
"""


CANNOT_SEE_PICTURES: Final = "This model cannot see pictures"
"""The words every note uses for a model that does not read images (#2107, #2123)."""

IMAGE_NOT_SEEN_NOTE: Final = f"[An image was here. {CANNOT_SEE_PICTURES}, so it was left out.]"
"""What a request to a model that does not read images says in place of each image (#2123).

A conversation can move to such a model after an earlier one was shown pictures. The
stored history keeps its images; only the request sent to this model carries this note
instead (`request_without_images`), so the provider is not sent input it would refuse.
"""


class ImagePart(BaseModel):
    """One image a message carries, beside its text (#2107, design `browser-agent.md` §3.2).

    `data` is the image as base64 and is never dumped: a session log body, a saved record
    and a request record keep the image's `digest` only, and the bytes are kept once, as a
    body of the session's context body store named by that digest, the way a full tool
    result is (#1848). In memory a part read back from a record has `data=None` until the
    bytes are put back; a connector sends `IMAGE_UNAVAILABLE_NOTE` for one that has none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    media_type: ImageMediaType
    digest: str = Field(
        description="SHA-256 of `data`, 64 lowercase hex characters: the name the bytes are "
        "kept under in the session's context body store."
    )
    data: str | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="The image, base64-encoded. Excluded from every dump.",
    )
    width: int | None = Field(default=None, description="Pixels, when the producer knows.")
    height: int | None = Field(default=None, description="Pixels, when the producer knows.")

    @model_validator(mode="after")
    def _digest_names_data(self) -> Self:
        if len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest):
            raise ValueError("An image's digest is 64 lowercase hex characters (SHA-256).")
        if self.data is not None and image_digest(self.data) != self.digest:
            raise ValueError("An image's digest is not the SHA-256 of its data.")
        return self

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        media_type: ImageMediaType,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> ImagePart:
        """An image part holding `raw`, encoded and named by its digest."""
        data = base64.b64encode(raw).decode("ascii")
        return cls(
            media_type=media_type, digest=image_digest(data), data=data, width=width, height=height
        )


def image_digest(data: str) -> str:
    """The name an image's base64 `data` is kept under: its SHA-256, as a body is named."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class ChatMessage(BaseModel):
    """Unified message envelope across Gemini, Claude, and OpenAI."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCallRequest, ...] = Field(default_factory=tuple)
    images: tuple[ImagePart, ...] = Field(
        default_factory=tuple,
        exclude_if=lambda value: not value,
        description="Images sent with the text, on a `USER` or `TOOL` message (#2107): a "
        "tool result's screenshot, say. Each is sent to the model as an image by the "
        "connectors that can (Anthropic, OpenAI, Gemini, Ollama); a dump keeps each one's "
        "digest and not its bytes (`ImagePart.data`). Left out of a dump when empty, so a "
        "message with no images serialises as it did before the field existed.",
    )
    compaction_ledger: bool = Field(
        default=False,
        description="True on a Session Progress Ledger emitted by `ContextCompactor` "
        "(issue #196). A ledger carries `role=SYSTEM`, so without this flag a compactor "
        "receiving its own prior output cannot tell an artifact it may supersede from "
        "the real system prompt it must anchor, and every pass added one more "
        "permanently-resident message. The flag travels on the message rather than in "
        "compactor state because `compact()` must stay a pure function of its input: a "
        "caller that constructs a fresh compactor per turn would otherwise fail to "
        "recognise prior ledgers and reintroduce the growth. It is not a control "
        "channel for ledger *kind* — the in-band `[Context Auto-Compacted Summary: "
        "LLM|Heuristic ...]` label carries that, and exposing it on the return type is "
        "issue #183's. Only meaningful on a `SYSTEM` message.",
    )

    form: Literal["excerpt", "stub"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="For a `TOOL` message that holds less than the result it answers with: "
        "`excerpt` for a head and tail within a cap, `stub` for the one-line form a "
        "compaction leaves (design `llm-request-layering.md` §5.8, *Forms*). Set where the "
        "shortening is decided -- ingest, the step budget, the compactor -- so the context "
        "state records it rather than reading it back from the text, which a tool's own "
        "output can imitate (#1854). `None` is the whole result, and is left out of a dump, "
        "so a message without a form serialises as it did before the field existed. Like "
        "`compaction_ledger`, it is not sent to a provider. Only meaningful on a `TOOL` "
        "message.",
    )

    rendered_from: RenderedFrom | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="For a `TOOL` message whose `form` was rendered from a result the "
        "session log keeps in full: that result's handle and the size it was cut to, set "
        "where the form is decided (ingest, the step budget, the compactor). The session "
        "log keeps a form as this record and no text, and every reader renders the text "
        "from the kept body (#1848). `None` for a whole result; a form with none -- a "
        "truncation with no session -- is never logged. Left out of a dump when `None`. "
        "Not sent to a provider.",
    )

    @model_validator(mode="after")
    def _validate_compaction_ledger_role(self) -> Self:
        if self.compaction_ledger and self.role != MessageRole.SYSTEM:
            raise ValueError(
                "compaction_ledger=True is only meaningful and permitted on MessageRole.SYSTEM "
                f"messages (got role={self.role.value!r}) (#204)"
            )
        if self.rendered_from is not None and self.form is None:
            raise ValueError(
                "rendered_from is only meaningful and permitted on a TOOL message that "
                "carries a form (#1848)"
            )
        if self.images and self.role not in (MessageRole.USER, MessageRole.TOOL):
            raise ValueError(
                f"images are only permitted on USER and TOOL messages "
                f"(got role={self.role.value!r}) (#2107)"
            )
        if self.form is not None and self.role != MessageRole.TOOL:
            raise ValueError(
                f"form={self.form!r} is only meaningful and permitted on MessageRole.TOOL "
                f"messages (got role={self.role.value!r}) (#1854)"
            )
        return self


class TokenCountSource(StrEnum):
    """Who produced a `TokenUsage`'s token figures (P6 provenance for a quantity, #916)."""

    PROVIDER = "provider"
    """The provider reported the count."""

    ESTIMATE = "estimate"
    """The provider reported no count, or left one out, and the figures stand in for it.

    Two producers: a connector completing a usage report that lacks a count
    (`connectors/base.py::resolve_token_counts`, with the UTF-8 byte estimator in
    `llm/compactor.py`, #939), and `TurnExecutor.invoke_model` for a stream that sent no usage
    (still a character-length heuristic, which undercounted Korean output by about half when
    measured once against a local model). Charged against the budget ceiling exactly like a
    count (#916)."""


class TokenUsage(BaseModel):
    """Token consumption. UClone-X counts tokens only; cost is out of scope (#1392)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: str = Field(
        description="Required. A defaulted provider meant unattributed usage aggregated "
        "into a bucket named 'unknown' instead of failing, which defeats P5's "
        "per-provider quota tracking.",
    )
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    count_source: TokenCountSource = Field(
        default=TokenCountSource.PROVIDER,
        description="Whether the token figures are the provider's count or an estimate "
        "standing in for one (#916). "
        "Defaulted to `PROVIDER` because a connector building usage from a provider "
        "response is the normal producer; a producer that substitutes figures must say so.",
    )
    cache_creation_input_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda created: created is None,
        description="Of `input_tokens`, how many the provider wrote to its prompt cache "
        "(Anthropic's `cache_creation_input_tokens`, #1371). A subset of `input_tokens`, "
        "never added to it again. `None` means the provider reported no figure, which is "
        "not 0 (P6); a record without one serializes exactly as it did before the field "
        "existed (#1844).",
    )
    cache_read_input_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda read: read is None,
        description="Of `input_tokens`, how many the provider served from its prompt cache "
        "(Anthropic's `cache_read_input_tokens`; OpenAI-compatible "
        "`prompt_tokens_details.cached_tokens`; Gemini's `cachedContentTokenCount`, #1371). "
        "A subset of `input_tokens`. `None` "
        "means the provider reported no figure, which is not 0 (P6); a record without one "
        "serializes exactly as it did before the field existed (#1844).",
    )

    @model_validator(mode="after")
    def _validate_or_derive_total(self) -> Self:
        computed = self.input_tokens + self.output_tokens
        if self.total_tokens == 0 and computed > 0:
            object.__setattr__(self, "total_tokens", computed)
        elif self.total_tokens != 0 and self.total_tokens != computed:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) does not match input_tokens + output_tokens ({computed})"
            )
        return self


def aggregate_token_usages(usages: Sequence[TokenUsage]) -> TokenUsage | None:
    """Sum token usage records across a sequence of model invocations (e.g. within a turn).

    Returns None if no usage records are provided (i.e. no model calls completed).
    `count_source` is `TokenCountSource.PROVIDER` only if every step's source was `PROVIDER`.
    Otherwise, the least certain source across steps is used (ESTIMATE < PROVIDER).
    A cache count is summed only when every step reported it; one step without a figure
    makes the sum unknown (`None`), never an undercount (#1371).
    """
    if not usages:
        return None
    input_tokens = sum(u.input_tokens for u in usages)
    output_tokens = sum(u.output_tokens for u in usages)
    total_tokens = sum(u.total_tokens for u in usages)

    provider = usages[-1].provider if usages else "unknown"
    model = usages[-1].model if usages else None

    if all(u.count_source == TokenCountSource.PROVIDER for u in usages):
        count_source = TokenCountSource.PROVIDER
    else:
        count_source = TokenCountSource.ESTIMATE

    return TokenUsage(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        count_source=count_source,
        cache_creation_input_tokens=_sum_if_all_reported(
            [u.cache_creation_input_tokens for u in usages]
        ),
        cache_read_input_tokens=_sum_if_all_reported([u.cache_read_input_tokens for u in usages]),
    )


def _sum_if_all_reported(counts: Sequence[int | None]) -> int | None:
    """The sum when every count is known, else `None`: a partial sum would understate."""
    known = [c for c in counts if c is not None]
    return sum(known) if len(known) == len(counts) else None


class ModelResponse(BaseModel):
    """Standardized response from an LLM provider."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    content: str | None = None
    thinking: str | None = Field(
        default=None,
        description="Reasoning or thinking content emitted by reasoning models (#695).",
    )
    tool_calls: tuple[ToolCallRequest, ...] = Field(default_factory=tuple)
    usage: TokenUsage = Field(
        description="Required. `TokenUsage` now names its provider, so it cannot be "
        "defaulted into existence — a response with unattributed usage is what broke "
        "P5's per-provider accounting.",
    )
    finish_reason: FinishReason | None = None
    model_name: str = "unknown"
    provenance: Provenance | None = Field(
        description="In-band attribution required by Principle 6. Explicit with no "
        "default: `None` is representable so a non-conformant value can be rejected by "
        "`require_provenance`, but it is never inherited silently.",
    )


class StreamChunk(BaseModel):
    """Normalized streaming delta chunk from an LLM provider."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    delta_content: str | None = None
    delta_thinking: str | None = Field(
        default=None,
        description="Incremental reasoning or thinking content delta (#695).",
    )
    tool_calls: tuple[ToolCallRequest, ...] = Field(default_factory=tuple)
    usage: TokenUsage | None = None
    finish_reason: FinishReason | None = None
    model: str | None = Field(
        default=None,
        description="The model the provider's stream says served this chunk, read from the "
        "wire and never filled in by the connector. `None` means the chunk named none. A "
        "streamed step is attributed from this, the way `generate` attributes from the "
        "response body, because the request often names no model at all (#1447).",
    )


class LLMRequest(BaseModel):
    """Unified request envelope for model inference."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: str | None = None
    messages: tuple[ChatMessage, ...] = Field(default_factory=tuple)
    tools: tuple[ToolDefinition, ...] = Field(default_factory=tuple)
    temperature: float = 0.7
    max_tokens: int | None = None
    auto_compact: bool = True
    compaction_threshold_tokens: int = 60_000
    context_window: int | None = Field(
        default=None,
        gt=0,
        description="The window the caller configured and counts this request against "
        "(`AgentLLMConfig.context_limit`). A connector whose server picks its own window "
        "sends it -- Ollama as `num_ctx` -- so the window served and the window counted "
        "agree (#1372). `None` leaves the choice to the connector: Ollama sends the "
        "`num_ctx` it last sent for the model, else `OLLAMA_CONTEXT_LENGTH` when that is a "
        "positive integer, else `DEFAULT_OLLAMA_NUM_CTX` (16384). The order is "
        "context_limit > OLLAMA_CONTEXT_LENGTH > 16384.",
    )
    thinking: bool | None = Field(
        default=None,
        description="Explicit control over provider reasoning/thinking tokens (e.g. Ollama think parameter). "
        "When None, default provider behavior is preserved.",
    )
    response_schema: ImmutableJsonMapping | None = Field(
        default=None,
        description="A JSON Schema the reply's content must satisfy (structured output). "
        "Ollama sends it as `format`. A connector with no faithful mapping raises "
        "`StructuredOutputUnsupportedError` rather than dropping it (P6).",
    )


def request_without_images(request: LLMRequest) -> LLMRequest:
    """`request` with each image replaced by `IMAGE_NOT_SEEN_NOTE` in its message's text (#2123).

    For a model that does not read images. The note follows the message's text, one per
    image, as a connector adds `IMAGE_UNAVAILABLE_NOTE`; a message that had only images
    carries only the notes. Messages without images, and `request` itself, are unchanged.
    """
    if not any(message.images for message in request.messages):
        return request
    messages: list[ChatMessage] = []
    for message in request.messages:
        if not message.images:
            messages.append(message)
            continue
        text = [message.content] if message.content else []
        text.extend(IMAGE_NOT_SEEN_NOTE for _ in message.images)
        messages.append(message.model_copy(update={"content": "\n\n".join(text), "images": ()}))
    return request.model_copy(update={"messages": tuple(messages)})


class LedgerSource(StrEnum):
    """Which producer wrote the Session Progress Ledger a compaction pass emitted.

    Exists because `ChatMessage.compaction_ledger` deliberately does not carry it: that
    flag answers "is this message a ledger", and its own field description says it "is
    not a control channel for ledger *kind* — the in-band `[Context Auto-Compacted
    Summary: LLM|Heuristic ...]` label carries that, and exposing it on the return type
    is issue #183's". This is that exposure.

    The alternative — having a caller substring-match the in-band label — would turn a
    human-readable provenance line into a load-bearing control channel, which is the
    coupling `ContextCompactor.compact` already refuses when splitting anchors from
    ledgers (issue #196).
    """

    NONE = "none"
    """No new ledger was emitted: the dialogue was within `keep_recent_turns`, so the
    pass only pruned oversized tool outputs."""

    LLM = "llm"
    """A configured summarizer produced the ledger text."""

    HEURISTIC = "heuristic"
    """The compactor's own structured ledger builder produced the text."""


class CompactionOutcome(BaseModel):
    """What one `ContextCompactor.compact` pass produced (P5, P6).

    `compact` previously returned a bare `tuple[ChatMessage, ...]`, which left a caller
    unable to say what produced the ledger it was handed — `_generate_llm_summary`
    returns `None` both when no summarizer is configured and when a configured one
    returns empty content, so the two cases were indistinguishable downstream.

    That matters for P6 rather than for tidiness. A compaction is a result crossing a
    component boundary, and an LLM-written ledger is a value a foundation model produced;
    publishing it with no attribution would be exactly the unattributed result P6
    forbids. Attribution is built **here**, by the only component that knows which path
    ran, and on the LLM path it is the summarizer's own `ModelResponse.provenance`
    forwarded verbatim — never synthesized, for the same reason `TurnExecutor.execute_turn`
    refuses to synthesize one for a model reply.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    messages: tuple[ChatMessage, ...] = Field(
        default_factory=tuple,
        description="The compacted context, in order: anchors, retained ledgers, the "
        "new ledger if any, then the recent window.",
    )
    origins: tuple[int | None, ...] = Field(
        default=(),
        description="Per message of `messages`, the index of the input message it renders "
        "-- kept as it was or pruned -- or `None` for a ledger this pass wrote (#1848). The "
        "caller derives the new epoch's forms from it. Empty when the compactor does not "
        "say; the caller then takes each message's form as recorded on it.",
    )
    ledger_source: LedgerSource = Field(
        description="Required. A defaulted source would let an unattributed ledger pass "
        "as a heuristic one, which is the distinction this type exists to carry.",
    )
    superseded_ledger_count: int = Field(
        default=0,
        description="Prior ledgers this pass stopped carrying forward, matching the "
        "figure named in band on the replacement ledger. Their content is genuinely "
        "lost: a ledger is never fed back into summarization (issue #196).",
    )
    provenance: Provenance | None = Field(
        description="In-band attribution required by Principle 6. Explicit with no "
        "default: `None` is representable so a non-conformant value can be rejected by "
        "`require_provenance`, but it is never inherited silently.",
    )


class TokenBudget(BaseModel):
    """Token limits for an agent session, and the tokens used against them.

    P5 requires "decoupled per-provider quota tracking", which the previous shape could
    not express: it had one undifferentiated pair of counters and no provider dimension
    at all, so a session spanning two providers could not be attributed to either.
    Cost is not tracked: UClone-X counts tokens only (#1392).

    `max_tokens` is `None` -- no session ceiling -- unless one is set explicitly. The
    budget is the user's system-wide paid-model limit (`llm-token-gateway.md` §4.6); this
    ledger attributes a session's tokens, and refuses only for a ceiling an embedder set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_tokens: int | None = None
    used_input_tokens: int = 0
    used_output_tokens: int = 0
    per_provider: ImmutableIntMapping = Field(
        default_factory=dict,
        description="Tokens used, input and output together, broken out by provider name, "
        "so a quota is enforceable per provider and not only per session.",
    )


class BudgetDecision(BaseModel):
    """The answer to "may this call proceed", with the reason attached.

    Replaces a bare `bool`. Under P6 a refusal a caller cannot explain is indistinguishable
    from an unexplained failure, and a quota ceiling is one of the failure classes the
    principle says must propagate rather than be worked around.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    allowed: bool
    reason: str | None = None
    remaining_tokens: int | None = None
