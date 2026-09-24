# Principle 5: LLM Provider Agnosticism, Token Management & Auto-Compaction

## Detailed Specification & Rationale

* **Core Law**: Token budgeting, quota tracking, and context compaction are strictly
  managed by the LLM layer, completely decoupled from agent business logic — and that
  layer is the *only* place a foundation-model provider may be named. Core agent, tool,
  and orchestration code depends solely on a provider-neutral interface.
* **Strict Rule**:
  - **Provider-neutral core (falsifiable)**:
    - No runtime module under `src/` outside `uclone_x.llm.connectors.*` may import a provider SDK
      (`anthropic`, `openai`, `google.genai`, `ollama`, or equivalent) or branch on
      provider identity (`if provider == "anthropic"` and the like). Test-only imports (`tests/`)
      are permitted solely to verify that mirror models (e.g. `ADKContentAdapter` types) agree
      with real SDK types. A lint rule checks this import boundary once code exists (see §Enforcement below).
    - No provider-native type (e.g. `anthropic.types.Message`, `openai.types.*`) may
      appear in an engine, agent, or tool function/method signature. Only the
      normalized `LLMRequest`, `ChatMessage`, and `TokenUsage` types defined in
      `docs/llm-agnostic-interface.md` cross that
      boundary.
    - All tool schemas, function-calling payloads, and streaming outputs are
      translated to and from provider-native shapes inside the adapter layer.
      Nothing upstream of it ever sees a provider's proprietary schema or prompt
      format.
    - Adding a new provider is satisfied by adding one connector module under
      `uclone_x.llm.connectors.*` that implements `BaseLLMClient`; it requires zero
      changes to engine, agent, or tool code outside that module.
  - **Token & quota governance**: The LLM abstraction layer owns token usage
    accounting, token ceilings, and request rate-limiting, decoupled from agent
    business logic. Because a `TokenUsage` record is produced by every connector on
    every call, accounting is available **per-provider** (and per-agent, per-session,
    per-user) even though callers only ever see the normalized type — agnosticism at
    the interface does not erase provider-level token attribution.
  - **Tokens only, no cost**: Cost or price calculation is out of scope for UClone-X.
    The LLM layer counts tokens only; it keeps no price table, estimates no monetary
    cost, and enforces no monetary ceiling.
  - **Automatic Context Compaction**: When conversation or working context nears
    window limits (e.g. 70%), the LLM layer triggers automated semantic pruning and
    structured summarization.
  - **Attributable Failover**: Provider failover is permitted only under P6's
    attributable-recovery conditions — a declared policy, in-band `provenance` on the
    response, a `PROVIDER_FAILOVER` event, and a correlated `failover.event` span. A
    span alone is not sufficient: telemetry informs a human, while `provenance`
    informs the calling agent. See
    [P6](p6-fail-fast-observability.md).
  - **Failover accounting is undecided**: which session's token budget a failover
    request charges is not specified here. It needs a P5 accounting rule — see
    `2026-09-02-007` §3. Do not close this gap by weakening either P5 or P6.
* **Why**: Prevents Out-Of-Context crashes, keeps agent code clean from token
  counting overhead, and keeps the router meaningful — "the LLM layer manages
  provider switching" is worth nothing if agent code is still free to import a
  provider SDK directly or special-case its wire format. The constraint that keeps
  the abstraction real is that nothing outside the adapter boundary is allowed to
  know a provider exists.
* **Enforcement**: Until `src/uclone_x/llm/` exists there is no import boundary to
  lint. Once it does, `./ucx test check` must gain a static check (e.g. a Ruff
  `TID251`/`flake8-tidy-imports` banned-import rule, or an import-linter contract)
  forbidding provider-SDK imports outside `uclone_x.llm.connectors.*`. Until that
  check lands, this rule is enforced by review.
* **Implementation Reference**: `docs/llm-agnostic-interface.md`

## Amendment Log

| Date | Principle | Change | Issue | Route | Author |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-09-04 | P5 | Option (A): Test-only import of provider SDKs (e.g. `google.genai` under optional dependency) is permitted exclusively within `tests/` for verifying adapter mirror model schema conformity. No runtime module under `src/` outside `uclone_x.llm.connectors.*` may import a provider SDK. | #381 | Tier A — approved by the project owner | a reviewer |
