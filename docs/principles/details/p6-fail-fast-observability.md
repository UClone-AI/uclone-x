# Principle 6: Fail-Fast & Zero Silent Fallbacks

## Detailed Specification & Rationale

* **Core Law**: A failure must either be repaired by a declared, attributable recovery that
  produces a genuine result, or surface immediately with its precise root cause. There is no
  third option. No component may hand its caller a value that the failed operation did not
  produce.

* **Strict Rule**: The law separates two behaviours that are often confused, and states one
  rule for each. The title's prohibition is absolute; the permission below is not a softening
  of it, because an attributable recovery is not a fallback in the forbidden sense.

  - **Forbidden unconditionally — silent substitution.** Substituting a value for the output
    of a failed operation is forbidden. This covers mock or stub data, an empty collection or
    empty string returned in place of a result, `None` or a zero value returned as if it were
    a success, a hardcoded default, and a stale cached value presented as fresh. No amount of
    logging, tracing, or commenting makes a substitution permissible. A telemetry span does
    not convert a substitution into a recovery.
  - **Permitted — declared, attributable recovery.** Re-attempting the *same* operation and
    returning a result that a real execution actually produced is permitted, whether the
    re-attempt hits the same provider (retry) or a different one (provider failover). It is
    permitted **only** when all four of the following hold; if any one is missing the
    behaviour falls back into the forbidden class above:
    1. **Declared in advance.** The retry or failover policy is explicit configuration, not
       an implicit `except` branch. An undeclared recovery is a silent one.
    2. **In-band on the result.** The returned envelope carries a `provenance` block
       (§ *In-Band Provenance*) naming the path taken and what actually served the request.
    3. **Announced on the decision plane.** A first-class `PROVIDER_FAILOVER` (or `RETRY`)
       `AgentEvent` is published to the event bus before the result event it describes, so
       the reasoning loop can observe the substitution of provider as it happens.
    4. **Correlated in telemetry.** An OpenTelemetry `failover.event` span is emitted, and
       its `span_id` appears in the in-band `provenance.attempts` entry, so the trace and
       the envelope describe the same event and neither can drift from the other.
  - **Unrepairable failures propagate.** When no declared recovery applies, or every attempt
    fails, the error propagates to the caller. It may be wrapped to add context; it may not
    be swallowed, downgraded to a warning, or converted into a successful return.
  - **Error classification never authorises substitution.** Fail-fast applies to *every*
    failure class without exception — infrastructure, transport, timeout, quota, validation,
    provider refusal, and business/domain errors alike. Classifying a failure decides only
    whether a *retry is eligible*; it never decides whether a *substitution is allowed*,
    because substitution is never allowed:

    | Failure class | Retry / failover eligible? | Substitution allowed? |
    | :--- | :--- | :--- |
    | Transport & infrastructure (HTTP 429/503, connection reset, timeout) | Yes, if declared | **No** |
    | Provider refusal / content policy | No | **No** |
    | Validation or schema violation | No | **No** |
    | Business / domain error | No | **No** |
    | Quota or budget ceiling exceeded | No — must propagate | **No** |

    Earlier text scoped this rule to "business errors". That word is removed: it placed
    infrastructure failures implicitly outside fail-fast and opened the escape hatch of
    classifying a failure as infrastructural, emitting a span, and returning something else.

* **In-Band Provenance**: A telemetry span records a failover for a human reading a trace
  later; it never enters the calling agent's reasoning context. An agent that cannot tell a
  primary result from a failover result cannot reason about the reliability of its own
  conclusions. Provenance therefore travels **with the value**, on the result envelope or
  response object crossing every component boundary, and not only in telemetry:

  | Field | Semantics |
  | :--- | :--- |
  | `provenance.path` | `"primary"` \| `"retry"` \| `"failover"`. Required, no default. Which path produced this value. |
  | `provenance.requested` | `{provider, model}` the caller asked for. |
  | `provenance.served_by` | `{provider, model}` that actually produced this value. |
  | `provenance.degraded` | `true` if and only if `served_by` differs from `requested`. |
  | `provenance.attempts` | Ordered list of failed attempts, one entry per attempt: `{provider, model, error_class, status_code, span_id}`. Empty if and only if `path == "primary"`. |

  - **Absence is a violation, not a default.** A consumer receiving a result envelope with no
    `provenance` must fail fast, and must never assume `"primary"`. A missing marker that is
    read as "nothing went wrong" is precisely the default-masquerading-as-a-real-answer that
    the first Strict Rule forbids.
  - **Why the bus event is not sufficient on its own.** The event bus applies a
    `BackpressurePolicy`, and under `DROP_OLDEST` or `DROP_INCOMING` a `PROVIDER_FAILOVER`
    event may legitimately be discarded under load. Provenance carried on the result itself
    cannot be dropped while the result survives, which is why requirement 2 above exists
    independently of requirement 3.
  - **Ordering is testable.** The bus orders delivery by `(priority, sequence)`. Publishing
    the `PROVIDER_FAILOVER` event before the result event, at a priority no lower than the
    result event's, guarantees a subscriber never observes a failover result before the
    notice that explains it.

* **Classification Procedure**: Any behaviour that runs after a failure is classified as
  permitted or forbidden by three questions, in order. The procedure is total — every
  behaviour reaches a verdict:

  1. **Does the caller receive a value that no real execution of the requested operation
     produced?** If yes → **forbidden**. Stop; telemetry is irrelevant here.
  2. **Was the operation genuinely re-attempted, and did a real execution produce the
     returned value?** If yes → **permitted if and only if** all four conditions of
     *declared, attributable recovery* hold; otherwise **forbidden**.
  3. **Otherwise the failure was not repaired** → the error must propagate. Returning any
     success value is **forbidden**.

* **Falsifiable Checks**: Compliance is verifiable, not a matter of judgement. Each of the
  following is directly assertable in a test:
  1. Force a total provider outage: the call **raises**; no result object is returned.
  2. Force a primary-only 503 with a declared failover: the returned envelope has
     `path == "failover"`, non-empty `attempts`, `served_by` naming the secondary, and
     `degraded == true`.
  3. A successful primary call has `path == "primary"` and `attempts == []`.
  4. For every result with `path != "primary"`, a matching `PROVIDER_FAILOVER` event exists
     on the bus with a lower `(priority, sequence)` than the result event.
  5. For every result with `path == "failover"`, each `attempts[*].span_id` resolves to an
     emitted `failover.event` span.
  6. A consumer handed a result envelope with `provenance` omitted raises rather than
     proceeding.

* **Why**: Silent fallbacks cause subtle hallucinations, obscure breaking bugs, and make agent
  behaviour non-deterministic and impossible to debug. Provider substitution is worse than a
  generic fallback: a model swap changes tool-calling behaviour, output formatting, and
  determinism, so an agent mid-plan may continue under assumptions that no longer hold. The
  agent must be able to learn that its result came from a degraded path while it can still act
  on that knowledge — which a collector-bound span can never tell it.

* **What this principle does not decide**: How a failover request is *billed* — which session
  budget it charges, and how a ceiling is re-checked at the fallback provider's different rate
  — belongs to **P5**, whose accounting rules govern it, and is still open
  (`2026-09-02-007` §3). P6 governs only
  whether a recovery is permitted and how it must be made visible.

* **Implementation Reference**: [`docs/telemetry-opentelemetry.md`](../../telemetry-opentelemetry.md),
  [`docs/llm-agnostic-interface.md`](../../llm-agnostic-interface.md) §4,
  [`docs/event-driven-agent-core.md`](../../event-driven-agent-core.md) §3
