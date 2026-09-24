"""Tests for the in-band provenance contract and the typed error taxonomy.

Each test in the first group is one of the falsifiable checks in
`docs/principles/details/p6-fail-fast-observability.md`, so a regression in the
contract fails the gate rather than being noticed in review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from provenance_ast import (
    constructed_path,
    construction_kind,
    is_provenance_expression,
    provenance_binding_names,
)
from pydantic import ValidationError

from uclone_x.core import AttemptRecord, ExecutionPath, Provenance, ServiceRef, require_provenance
from uclone_x.errors import (
    A2AError,
    MissingProvenanceError,
    TaskNotFoundError,
    UCloneXError,
    UnsupportedOperationError,
)
from uclone_x.llm.models import FinishReason, ModelResponse, TokenUsage


def test_primary_result_has_empty_attempts_and_is_not_degraded() -> None:
    """P6 check 3: a successful primary call has path 'primary' and attempts == []."""
    prov = Provenance.primary("anthropic", "claude-3-7-sonnet")

    assert prov.path is ExecutionPath.PRIMARY
    assert prov.attempts == ()
    assert prov.degraded is False


def test_failover_result_names_the_secondary_and_reports_degraded() -> None:
    """P6 check 2: a declared failover carries attempts, served_by and degraded."""
    prov = Provenance(
        path=ExecutionPath.FAILOVER,
        requested=ServiceRef(provider="google", model="gemini-2.5-pro"),
        served_by=ServiceRef(provider="anthropic", model="claude-3-7-sonnet"),
        attempts=(
            AttemptRecord(
                provider="google",
                model="gemini-2.5-pro",
                error_class="ServiceUnavailable",
                status_code=503,
                span_id="span-1",
            ),
        ),
    )

    assert prov.degraded is True
    assert prov.served_by.provider == "anthropic"
    assert prov.attempts[0].status_code == 503


def test_retry_on_the_same_service_is_not_degraded() -> None:
    """`degraded` tracks substitution, not failure: a retry that succeeded is not degraded."""
    ref = ServiceRef(provider="openai", model="gpt-4.1")
    prov = Provenance(
        path=ExecutionPath.RETRY,
        requested=ref,
        served_by=ref,
        attempts=(AttemptRecord(provider="openai", model="gpt-4.1", error_class="RateLimited"),),
    )

    assert prov.degraded is False
    assert prov.path is ExecutionPath.RETRY


def test_degraded_cannot_be_asserted_by_the_producer() -> None:
    """Issue #28: `degraded` is strictly computed from requested vs served_by, so peer-asserted degraded is stripped/recomputed."""
    req_ref = ServiceRef(provider="google", model="gemini-2.5-pro")
    served_ref = ServiceRef(provider="anthropic", model="claude-3-7-sonnet")
    attempt = AttemptRecord(
        provider="google",
        model="gemini-2.5-pro",
        error_class="ServiceUnavailable",
    )
    # If a peer asserts degraded=False on a substituted/failover result, it is stripped and recomputed as True
    prov = Provenance.model_validate(
        {
            "path": ExecutionPath.FAILOVER,
            "requested": req_ref,
            "served_by": served_ref,
            "attempts": (attempt,),
            "degraded": False,
        }
    )
    assert prov.degraded is True

    # Deserializing from model_dump() or model_dump_json() round-trips cleanly
    dumped = prov.model_dump()
    assert dumped["degraded"] is True
    restored = Provenance.model_validate(dumped)
    assert restored == prov
    assert restored.degraded is True


def test_all_provenance_bearing_envelopes_survive_wire_round_trip() -> None:
    """Issue #28: All 5 provenance-bearing envelopes round-trip through model_dump_json -> model_validate_json."""
    from uclone_x.a2a.models import TaskResult, TaskStatus
    from uclone_x.agent.models import TurnResult
    from uclone_x.sandbox.models import ExecutionResult, IsolationLevel
    from uclone_x.tools.models import ToolResult

    prov = Provenance.primary("anthropic", "claude-3-7-sonnet")

    # 1. Provenance itself
    prov_rt = Provenance.model_validate_json(prov.model_dump_json())
    assert prov_rt == prov
    assert prov_rt.degraded is False

    # 2. ModelResponse
    resp = ModelResponse(
        finish_reason=FinishReason.STOP,
        content="hello",
        usage=TokenUsage(provider="anthropic"),
        provenance=prov,
    )
    resp_rt = ModelResponse.model_validate_json(resp.model_dump_json())
    assert resp_rt == resp

    # 3. ToolResult
    tool_res = ToolResult(
        success=True,
        output="tool output",
        execution_time_ms=12.5,
        isolation_level=IsolationLevel.WORKSPACE,
        provenance=prov,
    )
    tool_rt = ToolResult.model_validate_json(tool_res.model_dump_json())
    assert tool_rt == tool_res

    # 4. ExecutionResult
    exec_res = ExecutionResult(
        exit_code=0,
        stdout="ok",
        stderr="",
        duration_ms=45.0,
        isolation_level=IsolationLevel.WORKSPACE,
        provenance=prov,
    )
    exec_rt = ExecutionResult.model_validate_json(exec_res.model_dump_json())
    assert exec_rt == exec_res

    # 5. TurnResult
    turn_res = TurnResult(
        turn_index=1,
        content="turn done",
        is_completed=True,
        provenance=prov,
    )
    turn_rt = TurnResult.model_validate_json(turn_res.model_dump_json())
    assert turn_rt == turn_res

    # 6. TaskResult
    task_res = TaskResult(
        task_id="t-123",
        status=TaskStatus.COMPLETED,
        output_data={"answer": "42"},
        provenance=prov,
    )
    task_rt = TaskResult.model_validate_json(task_res.model_dump_json())
    assert task_rt == task_res


def test_attempts_are_empty_if_and_only_if_the_path_is_primary() -> None:
    """P6 states the relation as an 'if and only if'; both directions are enforced."""
    ref = ServiceRef(provider="a")

    with pytest.raises(ValidationError, match="must be empty"):
        Provenance(
            path=ExecutionPath.PRIMARY,
            requested=ref,
            served_by=ref,
            attempts=(AttemptRecord(provider="a", error_class="Boom"),),
        )

    with pytest.raises(ValidationError, match="must be non-empty"):
        Provenance(path=ExecutionPath.FAILOVER, requested=ref, served_by=ServiceRef(provider="b"))


def test_a_primary_path_cannot_be_served_by_a_different_provider() -> None:
    """`path='primary'` with a different `served_by.provider` is rejected (#136, #149).

    P6 names the recovery paths by who served: a re-attempt hits "the same provider
    (retry) or a different one (provider failover)". A different provider serving with
    `path='primary'` therefore describes something P6 has no name for — and it is the
    encoding a substituting component reaches for, because checks 4 and 5 quantify over
    "every result with `path != 'primary'`" and so never look at it. This is the #136
    echo's shape.
    """
    with pytest.raises(ValidationError, match="cannot be served by a different provider"):
        Provenance(
            path=ExecutionPath.PRIMARY,
            requested=ServiceRef(provider="ollama", model="qwen2.5-coder:14b"),
            served_by=ServiceRef(provider="agent.core", model="BaseAgent"),
        )


def test_a_primary_path_may_resolve_a_model_alias_and_reports_it_degraded() -> None:
    """A provider-side alias is `primary` **and** `degraded` (issue #149).

    Gemini answers a request for `gemini-1.5-pro` with `modelVersion:
    gemini-1.5-pro-002`. One call, the requested provider answered, nothing failed — so
    `path` is `primary` and check 4 rightly does not apply, there being no failover to
    announce. But the caller is not talking to the model it named, and P6 wants that
    visible: "a model swap changes tool-calling behaviour, output formatting, and
    determinism, so an agent mid-plan may continue under assumptions that no longer
    hold."

    The first version of the sibling validator compared the whole `ServiceRef` and
    rejected this value, forbidding P6's headline case. Regression guard for that.
    """
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="gemini", model="gemini-1.5-pro"),
        served_by=ServiceRef(provider="gemini", model="gemini-1.5-pro-002"),
    )

    assert prov.path is ExecutionPath.PRIMARY
    assert prov.attempts == ()
    assert prov.degraded is True
    assert Provenance.primary("gemini", "gemini-1.5-pro", served_model="gemini-1.5-pro-002") == prov
    # `served_model` omitted still means "the requested model served itself".
    assert Provenance.primary("gemini", "gemini-1.5-pro").degraded is False


def test_a_primary_path_may_name_a_model_the_caller_did_not_specify() -> None:
    """A caller that named only a provider gets the served model back (issue #149).

    `ServiceRef.model` is optional, so `requested` can name a provider alone. The
    whole-`ServiceRef` comparison rejected this too — the caller asked for no particular
    model, yet reporting which one answered was unrepresentable on the primary path.
    """
    prov = Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="gemini"),
        served_by=ServiceRef(provider="gemini", model="gemini-1.5-pro"),
    )

    assert prov.degraded is True


def test_consumer_raises_on_absent_provenance() -> None:
    """P6 check 6: a consumer handed an envelope without provenance raises."""
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        content="hi",
        usage=TokenUsage(provider="test"),
        provenance=None,
    )

    with pytest.raises(MissingProvenanceError, match="refusing to treat it as a primary result"):
        require_provenance(response.provenance, "ModelResponse")


def test_provenance_must_be_stated_explicitly() -> None:
    """Absence is representable but never inherited: the field has no default."""
    with pytest.raises(ValidationError, match="provenance"):
        ModelResponse.model_validate({"content": "hi", "usage": {"provider": "test"}})


def test_require_provenance_returns_the_value_when_present() -> None:
    prov = Provenance.primary("anthropic")
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        content="hi",
        usage=TokenUsage(provider="test"),
        provenance=prov,
    )

    assert require_provenance(response.provenance, "ModelResponse") is prov


def test_provenance_is_frozen() -> None:
    prov = Provenance.primary("anthropic")

    with pytest.raises(ValidationError):
        prov.path = ExecutionPath.FAILOVER  # pyright: ignore[reportAttributeAccessIssue]


def test_error_taxonomy_is_rooted_and_carries_a2a_code_mappings() -> None:
    """A2A section 10.5: the fastpath raises the error a remote caller would receive."""
    assert issubclass(A2AError, UCloneXError)
    assert issubclass(MissingProvenanceError, UCloneXError)

    assert TaskNotFoundError.jsonrpc_code == -32001
    assert TaskNotFoundError.http_status == 404
    assert UnsupportedOperationError.jsonrpc_code == -32004
    assert UnsupportedOperationError.grpc_status == "FAILED_PRECONDITION"

    with pytest.raises(UCloneXError):
        raise TaskNotFoundError("t-1")


# ======================================================================================
# Static hygiene sweep: no defaulting of an absent provenance (issue #157)
# ======================================================================================


def _provenance_binding_names(tree: ast.Module) -> set[str]:
    """Delegates to the shared resolver (#188).

    This guard and the P6 checks-4/5 enumerator each resolved the name `Provenance`
    their own way and so shared a bypass: an alias defeated one, a module-qualified
    reference defeated both. One resolver now serves both, in `provenance_ast`.
    """
    return provenance_binding_names(tree)


def _is_provenance_construction(node: ast.AST, names: set[str]) -> bool:
    """True for any expression yielding a `Provenance`, produced or forwarded (#188).

    Now includes `Provenance(**payload)`, `Provenance.model_validate({...})` and
    `<prov>.model_copy(update={...})`. All three were constructible with
    `path=failover, degraded=True` and all three evaded the previous check, which only
    recognised `Provenance(...)` and `Provenance.<method>(...)` under the literal name.

    Forwarding forms count *here* even though the checks-4/5 enumerator excludes them:
    the question this guard asks is "did something manufacture attribution to fill an
    absence", and `x.provenance or Provenance.model_validate(stashed_dict)` does exactly
    that regardless of where the dict came from.
    """
    return is_provenance_expression(node, names)


def _provenance_valued_locals(scope: ast.AST) -> set[str]:
    """Names in `scope` bound from a `.provenance` read.

    Extracting a local is the first thing anyone does when the line gets long — and the
    sweep's own failure message, by naming the `or` form, pushes a reader straight
    towards it. Tracking it is therefore not an adversarial case but the likeliest one
    (#157 review). Single-assignment only; a name later rebound from something else is
    still tracked, which over-approximates in a direction that cannot hide a defect.
    """
    locals_: set[str] = set()
    for _ in range(2):  # one extra pass so `a = x.provenance; b = a` is reached
        for node in ast.walk(scope):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if value is None:
                continue
            sources = node.targets if isinstance(node, ast.Assign) else [node.target]
            is_prov = (isinstance(value, ast.Attribute) and value.attr == "provenance") or (
                isinstance(value, ast.Name) and value.id in locals_
            )
            if is_prov:
                for target in sources:
                    if isinstance(target, ast.Name):
                        locals_.add(target.id)
    return locals_


def _reads_provenance(node: ast.AST, locals_: set[str]) -> bool:
    """True for `<anything>.provenance` and for a local bound from one."""
    if isinstance(node, ast.Attribute) and node.attr == "provenance":
        return True
    return isinstance(node, ast.Name) and node.id in locals_


def _tests_provenance_absence(node: ast.AST, locals_: set[str]) -> bool:
    """True for `x.provenance is None`, `not x.provenance`, and the bare read."""
    if isinstance(node, ast.Compare) and _reads_provenance(node.left, locals_):
        return any(isinstance(op, ast.Is | ast.IsNot) for op in node.ops)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _reads_provenance(node.operand, locals_)
    return _reads_provenance(node, locals_)


def _scopes(tree: ast.AST) -> list[ast.AST]:
    """The module plus each function body, so local tracking does not leak across them."""
    return [tree] + [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _find_provenance_defaulting(tree: ast.AST) -> list[tuple[int, str]]:
    """Locate every place a construction fills in for an absent `provenance`."""
    names = _provenance_binding_names(tree) if isinstance(tree, ast.Module) else {"Provenance"}
    hits: dict[int, str] = {}
    for scope in _scopes(tree):
        locals_ = _provenance_valued_locals(scope)
        for node in ast.walk(scope):
            # `x.provenance or Provenance...(...)`
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
                values = node.values
                for left, right in zip(values, values[1:], strict=False):
                    if _reads_provenance(left, locals_) and _is_provenance_construction(
                        right, names
                    ):
                        hits[node.lineno] = "`x.provenance or Provenance...`"
            # `Provenance...(...) if <absence test> else ...`, and the mirrored form
            elif isinstance(node, ast.IfExp) and _tests_provenance_absence(node.test, locals_):
                if _is_provenance_construction(node.body, names) or _is_provenance_construction(
                    node.orelse, names
                ):
                    hits[node.lineno] = "conditional expression on an absent provenance"
            # `if x.provenance is None: ... Provenance...(...)`
            elif isinstance(node, ast.If) and _tests_provenance_absence(node.test, locals_):
                for branch in (node.body, node.orelse):
                    for stmt in branch:
                        if any(_is_provenance_construction(n, names) for n in ast.walk(stmt)):
                            hits[node.lineno] = "`if x.provenance is None:` then construct"
                            break
    return sorted(hits.items())


def test_no_component_manufactures_provenance_for_an_absent_one() -> None:
    """No `src/` module fills an absent `provenance` with one it constructs (#157).

    **Why this is a static sweep and not a validator.** The fabricated value is
    well-formed by construction — `Provenance.primary(agent_id)` yields
    `path=primary, requested == served_by, degraded=False`, which is exactly what an
    honest primary result looks like. Both runtime validators pass it (verified), and no
    schema rule can ever separate "the producer truthfully served this" from "a
    forwarder made it up", because the difference is not in the value. It is in *who
    called the constructor*, which only the source can show.

    **Why a sweep and not another call-site deletion.** #136 deleted this exact shape
    from `agent/base.py`. It reappeared in `a2a/server.py` within a day — and #136 is
    what made it reachable there, since `execute_turn` now propagates `None` verbatim.
    A deletion fixes one file; this fixes the shape.

    The rule is narrow on purpose: it does not forbid constructing `Provenance`, which
    every genuine producer must do. It forbids constructing one *in the branch where
    another object's provenance turned out to be absent* — the one context in which the
    constructor cannot be telling the truth, because absence is precisely what P6 says
    must not be defaulted.
    """
    src_root = Path(__file__).resolve().parents[2] / "src" / "uclone_x"
    offences: list[str] = []
    scanned = 0
    for path in sorted(src_root.rglob("*.py")):
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, shape in _find_provenance_defaulting(tree):
            offences.append(f"{path.relative_to(src_root.parent.parent)}:{lineno} — {shape}")

    assert scanned > 50, f"sweep only reached {scanned} modules; the glob is wrong"
    assert not offences, (
        "provenance is being defaulted rather than propagated or refused (P6, #157):\n  "
        + "\n  ".join(offences)
        + "\n\nPropagate the absent value, or refuse it with `require_provenance`. "
        "Do not construct one on behalf of a component that stated none."
    )


_DEFAULTING_SHAPES = [
    "record.provenance = turn.provenance or Provenance.primary(agent_id)",
    "record.provenance = result.provenance or Provenance(path=p, requested=r, served_by=r)",
    "x = Provenance.primary('h') if result.provenance is None else result.provenance",
    "x = result.provenance if result.provenance is not None else Provenance.primary('h')",
    "if result.provenance is None:\n    record.provenance = Provenance.primary('h')",
    "if not result.provenance:\n    record.provenance = Provenance.primary('h')",
    "if res.provenance is None:\n    if flag:\n        p = Provenance.primary('h')",
    # Extracted into a local first -- the likeliest rewrite of all, because the sweep's
    # own failure message names the `or` form and the line is long (#157 review).
    "p = turn.provenance\nrecord.provenance = p or Provenance.primary(aid)",
    "p = turn.provenance\nif p is None:\n    record.provenance = Provenance.primary(aid)",
    "a = turn.provenance\nb = a\nrecord.provenance = b or Provenance.primary(aid)",
    # Bound under an alias.
    "from uclone_x.core.provenance import Provenance as P\nx = t.provenance or P.primary('a')",
    "import uclone_x.core.provenance as m\nx = t.provenance or m.Provenance.primary('a')",
    # Forms verified constructible with `path=failover, degraded=True`, each a live
    # idiom in this tree rather than a contrivance (#188 item 1).
    "x = t.provenance or Provenance(**payload)",
    "x = t.provenance or Provenance.model_validate(stashed)",
    'x = t.provenance or Provenance.model_validate({"path": "failover"})',
    "x = t.provenance or base_prov.model_copy(update={'path': ExecutionPath.FAILOVER})",
    "if t.provenance is None:\n    x = Provenance.model_validate_json(raw)",
]

# Shapes this sweep does **not** catch, listed rather than left to be discovered.
# A syntactic rule cannot be complete; what it can be is honest about its edges. Each of
# these needs a deliberate act to write, unlike the local-variable and alias forms above,
# which are what a reader reaches for by accident.
_KNOWN_UNCOVERED_SHAPES = [
    # Construction moved behind a helper: the call site no longer names Provenance.
    "def _fallback():\n    return Provenance.primary('h')\nx = t.provenance or _fallback()",
    # Absence tested through a temporary boolean rather than on the value.
    "missing = t.provenance is None\nif missing:\n    x = Provenance.primary('h')",
    # Reflective access, in either position.
    "if getattr(t, 'provenance', None) is None:\n    x = Provenance.primary('h')",
    "x = t.provenance or getattr(Provenance, 'primary')('h')",
]

# Not an evasion, recorded so it is not re-investigated: pydantic rejects positional
# arguments to a model, so `Provenance(ExecutionPath.FAILOVER, req, srv)` raises at
# construction and cannot reach a consumer. The guard does not need to model it.
_NOT_AN_EVASION_POSITIONAL = "Provenance(ExecutionPath.FAILOVER, requested, served_by)"

_LEGITIMATE_SHAPES = [
    # A genuine producer constructing its own attribution.
    "provenance = Provenance.primary(provider='openai', model=m, served_model=s)",
    # Refusing, which is what an absence must lead to.
    "if result.provenance is None:\n    raise MissingProvenanceError('no provenance')",
    "prov = require_provenance(res.provenance, 'TurnResult')",
    # Propagating absence verbatim — the whole point.
    "record.provenance = turn_result.provenance",
    # Defaulting something that is not provenance.
    "record.model = result.model or 'gpt-4o'",
    # A `None` check on provenance that constructs something else entirely.
    "if result.provenance is None:\n    record.error = 'missing'",
]


@pytest.mark.parametrize("source", _DEFAULTING_SHAPES)
def test_the_provenance_sweep_detects_each_defaulting_shape(source: str) -> None:
    """The detector is the whole prevention mechanism, so its logic is pinned too.

    Mutation-checked: gutting `_find_provenance_defaulting` to return no hits passes the
    sweep itself — the sweep can only prove the tree is clean, never that it is still
    looking. These cases are what make silently disabling it fail.

    The listed shapes are the ways this defect has actually been written, plus the ways
    a reader would naturally rewrite it after the literal `or` form starts failing. The
    first version of this list asserted that claim without covering the two likeliest
    rewrites of all — extracting a local, and importing under an alias — so the list was
    the weakest exactly where the prose was most confident. Both are now covered; what
    remains uncovered is pinned next door in `_KNOWN_UNCOVERED_SHAPES` rather than
    described as complete.
    """
    assert _find_provenance_defaulting(ast.parse(source)), f"missed: {source!r}"


@pytest.mark.parametrize("source", _KNOWN_UNCOVERED_SHAPES)
def test_the_sweep_s_known_blind_spots_are_recorded_not_claimed(source: str) -> None:
    """Pins what the sweep does **not** catch, so the gap cannot be quietly assumed away.

    A syntactic sweep can never be complete, and `docs/a2a-protocol-spec.md` briefly
    claimed it caught the defect "in any of its syntactic forms" — false, and false in
    the one place a future reader checks instead of testing (#157 review). The remedy is
    not a wider claim but a recorded boundary.

    This test asserting these *evade* is deliberate and is not an invitation to leave
    them evading: if a later change makes one of them caught, this test fails, and the
    right response is to move that shape up into `_DEFAULTING_SHAPES`.
    """
    assert not _find_provenance_defaulting(ast.parse(source)), (
        f"now caught -- move this shape into _DEFAULTING_SHAPES: {source!r}"
    )


@pytest.mark.parametrize("source", _LEGITIMATE_SHAPES)
def test_the_provenance_sweep_does_not_flag_legitimate_construction(source: str) -> None:
    """The rule must not fire on a genuine producer, or it will be deleted (#157).

    A sweep with false positives is worse than none: the next builder to hit one removes
    it, and the protection goes with it. Constructing `Provenance` is required of every
    real producer; only constructing one *to fill an absence* is forbidden.
    """
    assert not _find_provenance_defaulting(ast.parse(source)), f"false positive: {source!r}"


# ======================================================================================
# The shared resolver's own behaviour (#188)
# ======================================================================================


_RESOLVER_CASES: list[tuple[str, str, str]] = [
    # source, expected ConstructionKind, expected PathVerdict
    ("Provenance(path=ExecutionPath.PRIMARY)", "PRODUCED", "PRIMARY"),
    ("Provenance(path=ExecutionPath.FAILOVER)", "PRODUCED", "NON_PRIMARY"),
    ("Provenance(path=ExecutionPath.RETRY)", "PRODUCED", "NON_PRIMARY"),
    ('Provenance(path="primary")', "PRODUCED", "PRIMARY"),
    ('Provenance(path="failover")', "PRODUCED", "NON_PRIMARY"),
    ("Provenance.primary('a')", "PRODUCED", "PRIMARY"),
    # Fails closed: an unresolvable path is UNKNOWN, never PRIMARY.
    ("Provenance(path=NON_PRIMARY_PATH)", "PRODUCED", "UNKNOWN"),
    ("Provenance(path=_PRIMARY_FALLBACK)", "PRODUCED", "UNKNOWN"),
    ("Provenance(**payload)", "PRODUCED", "UNKNOWN"),
    ("Provenance(path=compute())", "PRODUCED", "UNKNOWN"),
    # Alias and module-qualified bindings.
    ("P(path=ExecutionPath.FAILOVER)", "PRODUCED", "NON_PRIMARY"),
    ("m.Provenance(path=ExecutionPath.FAILOVER)", "PRODUCED", "NON_PRIMARY"),
    # Forwarding vs producing.
    ("Provenance.model_validate(d)", "FORWARDED", "UNKNOWN"),
    ("Provenance.model_validate_json(raw)", "FORWARDED", "UNKNOWN"),
    # An inline literal is authorship whatever the method is called, so the forwarding
    # exemption does not apply to it (#188 item 4).
    ('Provenance.model_validate({"path": "failover"})', "PRODUCED", "NON_PRIMARY"),
    ('Provenance.model_validate({"path": "primary"})', "PRODUCED", "PRIMARY"),
    ('base.model_copy(update={"path": ExecutionPath.FAILOVER})', "PRODUCED", "NON_PRIMARY"),
    # A copy that does not touch `path` is somebody else's model.
    ('entity.model_copy(update={"name": "x"})', "NONE", "UNKNOWN"),
    ('self._context.model_copy(update={"current_state": s})', "NONE", "UNKNOWN"),
    # Fails *open*, and pinned so it stays a known cost rather than a discovery: an
    # update dict built elsewhere is unreachable, because the receiver's type cannot be
    # resolved by name. See the asymmetry note in `provenance_ast`.
    ("base.model_copy(update=upd)", "NONE", "UNKNOWN"),
    ("base.model_copy(update=dict(path=FAILOVER))", "NONE", "UNKNOWN"),
]


@pytest.mark.parametrize(("source", "kind", "verdict"), _RESOLVER_CASES)
def test_the_shared_resolver_classifies_each_form(source: str, kind: str, verdict: str) -> None:
    """Pins `provenance_ast`, which both static guards now depend on (#188).

    Mutation testing is why this exists: flipping `_path_from_value`'s fallback from
    `UNKNOWN` to `PRIMARY` — turning the guard fail-open, the exact defect #188 item 2
    reports — passed every other test in the suite, because no module in `src/` currently
    contains an unresolvable path expression. The resolver's contract has to be asserted
    directly, not inferred from a tree that happens not to exercise it.

    The `NONE` rows are as load-bearing as the rest: a first draft treated every
    `model_copy` as provenance-related and swept in six unrelated pydantic models, and a
    guard that cries wolf is one the next builder deletes.
    """
    stmt = ast.parse(source).body[0]
    assert isinstance(stmt, ast.Expr)
    call = stmt.value
    assert isinstance(call, ast.Call)
    names = {"Provenance", "P", "m"}
    assert construction_kind(call, names).name == kind, source
    assert constructed_path(call, names).name == verdict, source


def test_the_shared_resolver_reads_real_import_bindings() -> None:
    """`provenance_binding_names` resolves from imports, not from a hard-coded string."""
    aliased = ast.parse("from uclone_x.core.provenance import Provenance as P\n")
    assert "P" in provenance_binding_names(aliased)

    qualified = ast.parse("import uclone_x.core.provenance as m\n")
    assert "m" in provenance_binding_names(qualified)

    # With no import at all the resolver still defends the default name rather than
    # resolving to nothing, which would disable both guards in one stroke.
    assert provenance_binding_names(ast.parse("x = 1\n")) == {"Provenance"}
