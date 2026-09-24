"""Unit tests for Layer 3 execution-mode gating and planning (#470).

Every assertion here is made on the *returned plan*. None inspects the source text of
`planner.py`: a source-text assertion is a grep moved inside a test function and is blind
to whether the code it matched is reachable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uclone_x.agent.planner import (
    HEURISTIC_PROVIDER,
    HEURISTIC_STRATEGY,
    ExecutionIntent,
    ExecutionPlan,
    PlanGenerator,
    PlanStep,
)
from uclone_x.core.provenance import ExecutionPath
from uclone_x.errors import PlanGenerationError

# The two queries from the #470 reproduction. Neither enumerates its own steps, so the
# heuristic must raise on both rather than return the old three-step filler.
UNDECOMPOSABLE = ("delete every file under /etc", "summarise this sentence")


def test_generate_plan_derives_distinct_steps_from_distinct_queries() -> None:
    """The #470 regression: two distinct queries must not yield identical step text.

    The negative (`!=`) is guarded by positive assertions first. A bare inequality can pass
    for the wrong reason -- for instance if both calls raised and were caught, or if steps
    were empty on both sides -- so each plan's step descriptions are pinned to the text
    actually present in its query before the two are compared.
    """
    generator = PlanGenerator()

    plan_a = generator.generate_plan("back up the database; then drop the stale index")
    plan_b = generator.generate_plan("compile the report; then email it to finance")

    descriptions_a = [step.description for step in plan_a.steps]
    descriptions_b = [step.description for step in plan_b.steps]

    # Positive first: the steps are the query's own clauses, not filler.
    assert descriptions_a == ["back up the database", "drop the stale index"]
    assert descriptions_b == ["compile the report", "email it to finance"]

    # Then the negative the acceptance criterion names.
    assert descriptions_a != descriptions_b

    # And the specific hardcoded plan #470 reported is gone from both.
    hardcoded = ["Analyze requirements", "Execute task", "Verify results"]
    assert descriptions_a != hardcoded
    assert descriptions_b != hardcoded


@pytest.mark.parametrize("query", UNDECOMPOSABLE)
def test_generate_plan_raises_rather_than_substituting_a_default(query: str) -> None:
    """P6: with no decomposition available, the call raises; no plan object is returned."""
    generator = PlanGenerator()

    with pytest.raises(PlanGenerationError) as excinfo:
        generator.generate_plan(query)

    assert query in str(excinfo.value)


@pytest.mark.parametrize("query", ["", "   ", "\n\n"])
def test_generate_plan_raises_on_empty_query(query: str) -> None:
    generator = PlanGenerator()

    with pytest.raises(PlanGenerationError):
        generator.generate_plan(query)


def test_generate_plan_splits_a_numbered_list() -> None:
    generator = PlanGenerator()

    plan = generator.generate_plan("1. clone the repo\n2. install deps\n3) run the gate")

    assert [step.description for step in plan.steps] == [
        "clone the repo",
        "install deps",
        "run the gate",
    ]
    assert all(isinstance(step, PlanStep) for step in plan.steps)
    assert all(step.status == "pending" for step in plan.steps)
    assert plan.status == "created"


def test_generate_plan_splits_bulleted_lines() -> None:
    generator = PlanGenerator()

    plan = generator.generate_plan("- fetch the manifest\n- verify the checksum")

    assert [step.description for step in plan.steps] == [
        "fetch the manifest",
        "verify the checksum",
    ]


def test_generate_plan_ignores_context_but_accepts_it() -> None:
    """`context` is in the signature for #475; the heuristic derives from `query` alone.

    Asserted rather than assumed, so that a future change which starts consulting `context`
    has to change a test that says so.
    """
    generator = PlanGenerator()
    query = "stage the files; then commit them"

    without_context = generator.generate_plan(query)
    with_context = generator.generate_plan(query, context={"session_id": "s-1"})

    assert [s.description for s in without_context.steps] == ["stage the files", "commit them"]
    assert [s.description for s in with_context.steps] == [
        s.description for s in without_context.steps
    ]


def test_plan_carries_heuristic_provenance() -> None:
    """AC 2: the envelope distinguishes a heuristic plan from an LLM-generated one."""
    generator = PlanGenerator()

    plan = generator.generate_plan("draft the memo; then circulate it")

    # Literals, not the module constants. Asserting `== HEURISTIC_PROVIDER` compares the
    # value against the same symbol that produced it, so a mutation of the constant moves
    # both sides and survives -- which is exactly what happened on the first mutation run.
    assert plan.provenance.path is ExecutionPath.PRIMARY
    assert plan.provenance.served_by.provider == "heuristic"
    assert plan.provenance.served_by.model == "clause-split-v1"
    assert plan.provenance.requested.provider == "heuristic"
    assert plan.provenance.degraded is False
    assert plan.provenance.attempts == ()

    # The exported constants are what an LLM-backed planner (#475) will be contrasted
    # against, so pin them too -- separately, so a drift in either is attributable.
    assert HEURISTIC_PROVIDER == "heuristic"
    assert HEURISTIC_STRATEGY == "clause-split-v1"


def test_execution_plan_cannot_be_built_without_provenance() -> None:
    """P6: "Absence is a violation, not a default." The field has no default to fall to."""
    with pytest.raises(ValidationError):
        ExecutionPlan(id="plan_x")  # type: ignore[call-arg]


def test_plan_ids_are_unique_per_call() -> None:
    generator = PlanGenerator()
    query = "open the ticket; then assign it"

    first = generator.generate_plan(query)
    second = generator.generate_plan(query)

    assert first.id != second.id
    assert {s.id for s in first.steps}.isdisjoint({s.id for s in second.steps})


def test_execution_intent_values() -> None:
    assert ExecutionIntent.DIRECT.value == "direct"
    assert ExecutionIntent.PLANNING.value == "planning"
