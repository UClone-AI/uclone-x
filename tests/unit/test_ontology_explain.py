"""Unit tests for `Closure.explain` as an *audit surface* (review finding on PR #159).

`explain` answers the question an operator asks before acting: "why does this hold, and
what would I have to revoke to make it stop?" An answer that names one of several live
supports is worse than one that admits it is partial, because the operator revokes the
named fact, sees the conclusion survive, and has no way to tell a bug from a second
reason.

The rule under test is therefore *well-foundedness as reachability*: a proof step is
reported when every premise is derivable from asserted leaves along some route that does
not pass through the step's own conclusion. Length is not a criterion. Circular support
(A justifies B justifies A) is the only thing excluded, because it proves nothing.

Closures here are hand-built rather than materialised. `Closure` accepts justifications
from anywhere, so the guarantee is pinned on the shapes that matter -- four parallel
supports, a bare cycle, a cycle beside a grounded route, mutual recursion -- rather than
on whichever of those a particular rule set happens to emit today.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from uclone_x.ontology.justification import Closure, Fact, ProofStep

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def leaf(name: str) -> Fact:
    """Build an asserted fact -- a leaf the closure rests on."""
    return Fact(subject=name, predicate="p", object="x")


def node(name: str) -> Fact:
    """Build a derived fact, which must therefore carry at least one justification."""
    return Fact(subject=name, predicate="p", object="x", derived=True)


def step(rule: str, premises: Sequence[Fact], conclusion: Fact) -> ProofStep:
    """Build a proof step from facts rather than raw ids, so the fixtures stay readable."""
    return ProofStep(
        rule=rule,
        premises=tuple(fact.id for fact in premises),
        conclusion=conclusion.id,
    )


def build(facts: Sequence[Fact], steps: Sequence[ProofStep]) -> Closure:
    """Assemble a closure from facts and steps, grouping steps by the fact they conclude."""
    grouped: dict[str, list[ProofStep]] = {}
    for one in steps:
        grouped.setdefault(one.conclusion, []).append(one)
    return build_with(facts, grouped)


def build_with(facts: Sequence[Fact], grouped: dict[str, list[ProofStep]]) -> Closure:
    """Assemble a closure from facts and pre-grouped justifications."""
    return Closure(
        facts={fact.id: fact for fact in facts},
        justifications={fact_id: tuple(steps) for fact_id, steps in grouped.items()},
    )


def rules_concluding(steps: Sequence[ProofStep], conclusion: Fact) -> set[str]:
    """Return the rule names of the reported steps that conclude `conclusion`."""
    return {one.rule for one in steps if one.conclusion == conclusion.id}


# --------------------------------------------------------------------------------------
# Fixtures: four parallel supports of differing lengths
# --------------------------------------------------------------------------------------


def four_support_closure() -> tuple[Closure, Fact, tuple[Fact, Fact, Fact, Fact]]:
    """Build the reviewer's case: one target held up by four independent, non-cyclic routes.

    The four supporting premises sit at derivation depths 0, 1, 2 and 3, so the four
    supports differ only in *length*. Nothing about them is circular: each route runs from
    its own asserted leaf and they share no facts.
    """
    leaf_a, leaf_b, leaf_c, leaf_d = leaf("la"), leaf("lb"), leaf("lc"), leaf("ld")
    # Route A: the target rests directly on an asserted leaf.
    # Route B: one derivation deep.
    route_b = node("b1")
    # Route C: two derivations deep.
    route_c1, route_c2 = node("c1"), node("c2")
    # Route D: three derivations deep.
    route_d1, route_d2, route_d3 = node("d1"), node("d2"), node("d3")
    target = node("target")

    closure = build(
        [
            leaf_a,
            leaf_b,
            leaf_c,
            leaf_d,
            route_b,
            route_c1,
            route_c2,
            route_d1,
            route_d2,
            route_d3,
            target,
        ],
        [
            step("lift_b", [leaf_b], route_b),
            step("lift_c1", [leaf_c], route_c1),
            step("lift_c2", [route_c1], route_c2),
            step("lift_d1", [leaf_d], route_d1),
            step("lift_d2", [route_d1], route_d2),
            step("lift_d3", [route_d2], route_d3),
            step("via_a", [leaf_a], target),
            step("via_b", [route_b], target),
            step("via_c", [route_c2], target),
            step("via_d", [route_d3], target),
        ],
    )
    return closure, target, (leaf_a, route_b, route_c2, route_d3)


# --------------------------------------------------------------------------------------
# The defect: parallel supports must all be reported
# --------------------------------------------------------------------------------------


def test_four_independent_supports_are_all_reported() -> None:
    """All four routes are in `justifications`; `explain` must report all four too.

    This is the failure that motivated the fix: an operator asks why the target holds,
    revokes the single fact `explain` named, and believes the conclusion is retracted. It
    survives through three other routes that were never shown.
    """
    closure, target, supports = four_support_closure()

    recorded = closure.justifications[target.id]
    assert len(recorded) == 4, "the fixture must record four supports for the filter to keep"

    steps = closure.explain(target.id)
    assert rules_concluding(steps, target) == {"via_a", "via_b", "via_c", "via_d"}

    # Each support is reported citing its own premise, so the four are distinguishable.
    cited = {premise for one in steps if one.conclusion == target.id for premise in one.premises}
    assert cited == {support.id for support in supports}

    # And every supporting route is itself explained down to its asserted leaf.
    reported = {one.rule for one in steps}
    assert {"lift_b", "lift_c1", "lift_c2", "lift_d1", "lift_d2", "lift_d3"} <= reported


def test_a_longer_support_is_not_discarded_for_being_longer() -> None:
    """Pin the boundary the old shortest-depth filter drew, so it cannot creep back.

    The old rule admitted a step only when its deepest premise was *strictly shallower*
    than the conclusion (`deepest < target`). Relaxing it by one (`deepest <= target`)
    changed the answer from one support to two and survived the reviewer's mutation run,
    because both branches were still taken.

    The fixture places one support one below that boundary, one exactly on it, and one
    one above it, so the test fails if such a comparison reappears at either offset:

    * with `deepest < depth(target)`  -> `via_b` and `via_c` are both dropped;
    * with `deepest <= depth(target)` -> `via_c` is dropped.

    Widening the other way is pinned by
    `test_a_circular_support_is_still_excluded_when_a_deeper_one_is_admitted`, which fails
    if the filter degenerates into admitting everything.
    """
    closure, target, (support_a, support_b, support_c, _) = four_support_closure()

    # The three depths that straddle the old boundary, asserted explicitly so that a
    # change to the fixture cannot quietly stop exercising it.
    assert closure.depth(target.id) == 1
    assert closure.depth(support_a.id) == 0  # one below the boundary
    assert closure.depth(support_b.id) == 1  # exactly on it
    assert closure.depth(support_c.id) == 2  # one above it

    reported = rules_concluding(closure.explain(target.id), target)
    assert "via_a" in reported, "a support shallower than the conclusion must be reported"
    assert "via_b" in reported, "a support exactly as deep as the conclusion must be reported"
    assert "via_c" in reported, "a support deeper than the conclusion must be reported"


def test_a_circular_support_is_still_excluded_when_a_deeper_one_is_admitted() -> None:
    """Widening the filter must not degrade into admitting everything.

    The same target carries a fourth, circular justification: it cites a fact whose only
    support is the target itself. Length is no longer a criterion, but groundedness is,
    so this one stays out while the deeper honest routes stay in.
    """
    closure, target, _ = four_support_closure()
    circular = node("circular")

    facts = [*closure.facts.values(), circular]
    grouped: dict[str, list[ProofStep]] = {
        fact_id: list(steps) for fact_id, steps in closure.justifications.items()
    }
    grouped[target.id].append(step("via_cycle", [circular], target))
    grouped[circular.id] = [step("back_to_target", [target], circular)]
    widened = build_with(facts, grouped)

    reported = rules_concluding(widened.explain(target.id), target)
    assert reported == {"via_a", "via_b", "via_c", "via_d"}
    assert "via_cycle" not in reported
    # The circular step is not deleted, only not offered as a reason.
    assert any(one.rule == "via_cycle" for one in widened.justifications[target.id])


# --------------------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------------------


def test_a_fact_supported_only_by_a_cycle_explains_nothing() -> None:
    """Two facts that justify only each other are grounded in nothing, so neither explains."""
    left, right = node("left"), node("right")
    closure = build(
        [left, right],
        [
            step("cycle_left", [right], left),
            step("cycle_right", [left], right),
        ],
    )
    assert closure.explain(left.id) == ()
    assert closure.explain(right.id) == ()
    # The cycle is recorded, just not reported as a reason.
    assert len(closure.justifications[left.id]) == 1


def test_a_cyclic_and_a_grounded_support_reports_only_the_grounded_one() -> None:
    """The cyclic route is dropped, the grounded route survives, and the walk terminates."""
    ground = leaf("ground")
    target = node("target")
    partner = node("partner")
    closure = build(
        [ground, target, partner],
        [
            step("from_ground", [ground], target),
            step("from_partner", [partner], target),
            step("from_target", [target], partner),
        ],
    )

    steps = closure.explain(target.id)
    assert [one.rule for one in steps] == ["from_ground"]

    # `partner` is genuinely grounded -- through the target -- so it explains itself, and
    # that walk terminates too rather than bouncing between the two facts.
    partner_steps = closure.explain(partner.id)
    assert [one.rule for one in partner_steps] == ["from_target", "from_ground"]


def test_mutual_recursion_terminates_and_reports_both_grounded_supports() -> None:
    """Two facts support each other *and* each rests on its own leaf.

    Unlike the bare cycle, each mutual step is well-founded: the premise is reachable from
    an asserted leaf without passing through the conclusion. Both are therefore genuine
    alternative supports and both are reported -- and the traversal still terminates,
    because a fact id is expanded at most once.
    """
    leaf_m, leaf_n = leaf("lm"), leaf("ln")
    first, second = node("first"), node("second")
    closure = build(
        [leaf_m, leaf_n, first, second],
        [
            step("first_from_leaf", [leaf_m], first),
            step("first_from_second", [second], first),
            step("second_from_leaf", [leaf_n], second),
            step("second_from_first", [first], second),
        ],
    )

    steps = closure.explain(first.id)
    assert {one.rule for one in steps} == {
        "first_from_leaf",
        "first_from_second",
        "second_from_leaf",
        "second_from_first",
    }
    assert len(steps) == len(set(steps)), "each step is reported once"


# --------------------------------------------------------------------------------------
# Contract edges: unknown ids, asserted facts, ordering
# --------------------------------------------------------------------------------------


def test_explain_refuses_an_unknown_fact_id_rather_than_returning_empty() -> None:
    """A miss must be loud. `()` would be indistinguishable from an asserted fact (P6)."""
    closure = build([leaf("only")], [])
    with pytest.raises(KeyError):
        closure.explain("0" * 64)


def test_an_asserted_fact_explains_itself_with_an_empty_chain() -> None:
    """An asserted leaf rests on nothing, so there is nothing to explain -- and no error."""
    ground = leaf("ground")
    target = node("target")
    closure = build([ground, target], [step("from_ground", [ground], target)])
    assert closure.explain(ground.id) == ()


def test_ordering_keeps_depths_descending_and_leaf_resting_steps_last() -> None:
    """The ordering contract, now that parallel supports crowd the result.

    Depths never increase and the final step bottoms out on asserted facts -- so a reader
    who stops at the end has reached the leaves. Note that a support may legitimately run
    through premises *deeper* than the conclusion, in which case those premises' own steps
    sort ahead of the queried fact's; that is the honest consequence of no longer hiding
    long routes.
    """
    closure, target, _ = four_support_closure()
    steps = closure.explain(target.id)

    depths = [closure.depth(one.conclusion) for one in steps]
    assert depths == sorted(depths, reverse=True)

    last = steps[-1]
    assert all(not closure.facts[premise].derived for premise in last.premises)


def test_ordering_leads_with_the_queried_fact_within_its_own_depth() -> None:
    """Among equally deep conclusions, the fact that was asked about is reported first."""
    closure, target, _ = four_support_closure()
    steps = closure.explain(target.id)

    same_depth = [one for one in steps if closure.depth(one.conclusion) == closure.depth(target.id)]
    own = [one for one in same_depth if one.conclusion == target.id]
    assert len(own) == 4
    assert same_depth[: len(own)] == own


def test_explain_is_deterministic_across_calls() -> None:
    """Two calls on the same closure return the identical tuple, memoisation included."""
    closure, target, _ = four_support_closure()
    assert closure.explain(target.id) == closure.explain(target.id)
