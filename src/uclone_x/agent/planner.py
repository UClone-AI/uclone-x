"""Execution Mode Gating and Planning (Layer 3).

`PlanGenerator` used to return the same three steps — "Analyze requirements", "Execute
task", "Verify results" — for every query, with only the UUIDs differing (#470). That is
the case Principle 6 names first among the things it forbids unconditionally: "a hardcoded
default" substituted for the output of work that was never done, handed to a caller that
"receives a value that no real execution of the requested operation produced".

Two rules follow from the principle and are implemented here rather than left to callers:

* **No plan is produced unless the query says what the steps are.** This generator carries
  no model. It can only *split* a query that already enumerates its own steps; it cannot
  infer a decomposition for one that does not. When the declared heuristic finds no
  structure to split on, `generate_plan` raises `PlanGenerationError`. P6's Classification
  Procedure reaches "forbidden" at question 1 for every alternative: returning three
  generic steps, returning a single step echoing the query, and returning an empty plan are
  all substitutions, and a more elaborate placeholder is a worse one, not a better one.

* **Every plan says how it was made.** `ExecutionPlan.provenance` is required and has no
  default, so a consumer can tell a clause-split heuristic plan from a model-authored one.
  P6: "Absence is a violation, not a default." The heuristic reports itself as the
  `heuristic` provider; an LLM-backed planner (#475) reports its own provider and model
  through the same field, which is what makes the two distinguishable at the call site.

**What the heuristic can and cannot do**, stated plainly so nobody mistakes its scope. It
can decompose a query that enumerates its steps: numbered or bulleted lines, newline-
separated instructions, and clauses joined by `;` / `then` / `, and then`. It cannot do
anything else. It has no task understanding, no ordering judgement, no notion of
dependency, and no ability to expand "summarise this sentence" or "delete every file under
/etc" into steps — those raise. It is a splitter with correct attribution, not a planner,
and it is deliberately narrow: a broad heuristic that always returns *something* would
reintroduce exactly the query-independent filler this module exists to remove.

`context` is accepted because `BaseAgent` passes its `AgentContext`, and because the
LLM-backed planner of #475 will need it. The heuristic does not consult it; it derives
steps from `query` alone. That is stated here rather than implied by an unused parameter.
"""

from __future__ import annotations

import re
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.provenance import Provenance
from uclone_x.errors import PlanGenerationError

__all__ = [
    "HEURISTIC_PROVIDER",
    "HEURISTIC_STRATEGY",
    "ExecutionIntent",
    "ExecutionPlan",
    "PlanGenerator",
    "PlanStep",
]

HEURISTIC_PROVIDER = "heuristic"
"""`provenance.served_by.provider` for a plan produced without a model.

An LLM-backed planner names its own provider here instead, which is the discriminant the
acceptance criteria of #470 ask for: heuristic versus LLM-generated, in band, on the value.
"""

HEURISTIC_STRATEGY = "clause-split-v1"
"""`provenance.served_by.model` for the heuristic: which splitter produced the steps."""

# Enumerated-list prefixes: "1.", "2)", "-", "*", "•" at the start of a line.
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")

# Clause separators the query itself supplies. Every one of these is an explicit
# enumeration marker written by the caller, never one inferred here. The first branch lets
# a semicolon absorb a following "then" / "and then" so the connective does not survive
# into the step text; the `\b` in the second stops "streng|then " from being a separator.
_CLAUSE_SEPARATOR = re.compile(
    r"\s*;\s*(?:\band\s+)?(?:\bthen\s+)?|\s*,?\s*\b(?:and\s+)?then\s+",
    re.IGNORECASE,
)


class ExecutionIntent(StrEnum):
    """Execution mode intent."""

    DIRECT = "direct"
    PLANNING = "planning"


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    model_config = ConfigDict(frozen=True, strict=True)

    id: str
    description: str
    status: str = "pending"


class ExecutionPlan(BaseModel):
    """An execution plan consisting of multiple steps.

    `provenance` is required and has no default. Principle 6 puts attribution on the value
    crossing the boundary, not only in telemetry, "so a caller cannot distinguish a real
    plan from this placeholder" stops being true of this envelope.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    id: str
    provenance: Provenance
    steps: tuple[PlanStep, ...] = Field(default_factory=tuple)
    status: str = "created"


class PlanGenerator:
    """Splits a self-enumerating query into checklist steps, or raises.

    See the module docstring for the exact scope of the heuristic and for why raising —
    rather than returning a richer placeholder — is the Principle 6 outcome when the query
    cannot be split.
    """

    def generate_plan(self, query: str, context: Any = None) -> ExecutionPlan:
        """Derive an execution plan from the enumeration `query` already contains.

        Args:
            query: The task text. Must enumerate its own steps — see the module docstring.
            context: Accepted for interface parity with an LLM-backed planner (#475) and
                not consulted by this heuristic.

        Returns:
            An `ExecutionPlan` whose steps came from `query`, carrying heuristic
            provenance.

        Raises:
            PlanGenerationError: `query` is empty, or contains no enumeration this
                heuristic can split. No plan is substituted in its place (P6).
        """
        del context  # Named in the signature for #475; deliberately unused here.

        segments = self._split(query)
        if len(segments) < 2:
            raise PlanGenerationError(
                "PlanGenerator cannot derive an execution plan from "
                f"{query!r}: it carries no enumeration to split on (numbered or bulleted "
                "lines, separate lines, or clauses joined by ';' / 'then' / 'and then'). "
                "This generator has no model and does not invent steps; returning a "
                "generic checklist would be the substituted default Principle 6 forbids. "
                "Supply an LLM-backed planner for queries that must be decomposed."
            )

        steps = tuple(
            PlanStep(id=f"step_{uuid.uuid4().hex}", description=segment) for segment in segments
        )
        return ExecutionPlan(
            id=f"plan_{uuid.uuid4().hex}",
            steps=steps,
            provenance=Provenance.primary(
                provider=HEURISTIC_PROVIDER,
                model=HEURISTIC_STRATEGY,
            ),
        )

    @staticmethod
    def _split(query: str) -> list[str]:
        """Return the step texts `query` enumerates, or fewer than two if it enumerates none."""
        lines = [_LIST_PREFIX.sub("", line).strip() for line in query.splitlines()]
        lines = [line for line in lines if line]
        if len(lines) > 1:
            return lines
        if not lines:
            return []
        return [part.strip() for part in _CLAUSE_SEPARATOR.split(lines[0]) if part.strip()]
