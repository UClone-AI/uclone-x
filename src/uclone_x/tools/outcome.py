"""Whether a tool call produced anything, as distinct from whether it succeeded.

`ToolResultStatus` is binary — SUCCESS or ERROR — and a search that matched nothing is a
success. So `{"total_matches": 0, "matches": []}` reaches the model as an unremarkable
result, and the model reads it as evidence the thing is not there.

Measured (#698): asked for a number in a named file, `qwen3:8b` reached for `file_search`
with a regex, the regex missed, and it answered *"The line ... does not appear in the
file"*. The number was there. That was one call of the median one call per problem, against
a median declared horizon of six (#697): the turn did not run out of budget, it ran out of
willingness, and the empty result is what ended it.

A miss is not an absence. This module says which of the three a call was, so the runtime
can say so too:

* `ERRORED` — the call failed. The *call* was wrong: a bad path, a rejected boundary.
  Retrying a corrected call is the reasonable next move.
* `EMPTY` — the call succeeded and produced nothing. The *query* was wrong, not the call.
  A different query, or a different tool, is the reasonable next move.
* `PRODUCTIVE` — the call succeeded and produced something. Whether it answered the
  question is not knowable here; only the model can judge that.

**What this deliberately does not do** is decide what happens next. Retrying, re-querying
and giving up are turn-level decisions with turn-level costs, and putting them here would
bury a policy inside a classifier. The classifier's job is to stop three different things
looking identical.
"""

from __future__ import annotations

from collections.abc import Sized
from enum import StrEnum
from typing import Any, cast

from uclone_x.tools.models import ToolResult

#: Keys whose value is a count of what a call found. Checked before the generic emptiness
#: rules because a payload can be structurally non-empty and still report nothing --
#: `{"query": ..., "total_matches": 0, "matches": []}` has three keys and no findings.
_COUNT_KEYS: tuple[str, ...] = ("total_matches", "match_count", "result_count", "count")

#: Keys holding the findings themselves. An empty sequence under one of these is the same
#: statement as a zero count, made structurally.
_COLLECTION_KEYS: tuple[str, ...] = ("matches", "results", "items", "entries", "files")


class ToolOutcome(StrEnum):
    """What a tool call produced, beyond whether it raised."""

    ERRORED = "errored"
    EMPTY = "empty"
    PRODUCTIVE = "productive"


def classify_tool_outcome(result: ToolResult) -> ToolOutcome:
    """Classify one tool result as errored, empty, or productive.

    Conservative in one direction on purpose: anything not recognisably empty is
    `PRODUCTIVE`. A false `EMPTY` would tell the model its query found nothing when it
    found something, which is worse than staying silent -- the runtime would be inventing
    an absence, which is the very failure this module exists to name.
    """
    if not result.success:
        return ToolOutcome.ERRORED
    payload: Any = result.output
    return classify_payload_shape(payload)


def classify_payload_shape(output: Any) -> ToolOutcome:
    """The shape rules, over a plain `object`.

    Taken as `object` rather than `JsonValue`: the latter is a recursive union that
    pyright strict cannot narrow through `isinstance` without a cast at every branch, and
    six casts to satisfy a type checker would bury the three rules this function exists to
    state.
    """
    if output is None:
        return ToolOutcome.EMPTY
    if isinstance(output, str):
        return ToolOutcome.EMPTY if not output.strip() else ToolOutcome.PRODUCTIVE
    if isinstance(output, (list, tuple, set)):
        sized = cast("Sized", output)
        return ToolOutcome.EMPTY if len(sized) == 0 else ToolOutcome.PRODUCTIVE
    if isinstance(output, dict):
        return _classify_mapping(cast("dict[str, object]", output))
    return ToolOutcome.PRODUCTIVE


def _classify_mapping(mapping: dict[str, object]) -> ToolOutcome:
    """A mapping can be structurally non-empty and still report nothing found.

    `{"query": ..., "total_matches": 0, "matches": []}` is the measured payload: three
    keys, no findings. A plain "is the dict empty" rule calls that productive, which is
    how the misread survived.
    """
    if not mapping:
        return ToolOutcome.EMPTY
    for key in _COUNT_KEYS:
        value = mapping.get(key)
        # `bool` is an `int` in Python, and `{"count": False}` is a flag, not a count.
        if isinstance(value, int) and not isinstance(value, bool) and value == 0:
            return ToolOutcome.EMPTY
    for key in _COLLECTION_KEYS:
        if key in mapping:
            collection = mapping[key]
            if isinstance(collection, (list, tuple, set)):
                if len(cast("Sized", collection)) == 0:
                    return ToolOutcome.EMPTY
    return ToolOutcome.PRODUCTIVE


#: Appended to the tool message when a call succeeded and found nothing. Addressed to the
#: reading the measurement caught -- a miss read as an absence -- and phrased without
#: naming a replacement tool, because naming one steers the choice and that steer was
#: measured at one problem in ten, inside the noise of an unrepeated run (#698).
EMPTY_RESULT_NOTE = (
    "[This call succeeded and matched nothing. A miss is not evidence that the thing is "
    "absent: the query may be wrong, or the wrong tool for it. Do not conclude absence "
    "from this result alone.]"
)
