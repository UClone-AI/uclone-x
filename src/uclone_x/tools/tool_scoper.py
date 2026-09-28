"""Per-turn tool scoping: kept for the `tool_strategy` eval only, not on any request path.

**No production path uses this module.** The agent's tools layer is a pinned base set that
host binding may only append to (`uclone_x.tools.tool_binder`, design §5.1): re-choosing
the advertised tools on every turn changed the tools layer on 11 of 19 request pairs in
#1670, and every empty reply it caused was a call to a tool it had withheld, which Ollama
drops without an error. The embedding-backed `SemanticToolScoper` was removed with it.

`LexicalToolScoper` stays because the eval's arm B measures it as the baseline host
binding replaced. Its notice promised that a withheld tool still runs when called by
name; the agent now refuses a call to a name its request did not declare (F12), so the
notice describes the eval's harness, not the agent.

What the scoper still fixes relative to its predecessor: it is called lexical because it
scores literal token overlap, and a query no tool scores above zero on withholds nothing
rather than returning the first `top_k` tools in registry order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from uclone_x.llm.models import ToolDefinition

if TYPE_CHECKING:
    from collections.abc import Sequence

# How many withheld tool names the notice lists before it stops naming them. The count is
# always exact; the names are a courtesy that must not itself become the context problem
# scoping exists to solve.
MAX_NAMED_WITHHELD = 20


def _by_name(tools: Sequence[ToolDefinition]) -> tuple[ToolDefinition, ...]:
    """`tools` in name order: the tools layer of the request (design §5.1).

    A scorer decides *which* tools are advertised, never their order. Score order would
    move a schema within the request prefix whenever a query ranked it differently, so the
    rank is kept on `ToolScopingResult.scores` and out of `selected`.
    """
    return tuple(sorted(tools, key=lambda tool: tool.name))


@dataclass(frozen=True)
class ToolScopingResult:
    """The tools advertised this turn, what was withheld, and by what method."""

    selected: tuple[ToolDefinition, ...]
    withheld: tuple[str, ...] = ()
    method: str = "none"
    # Why nothing was withheld, when nothing was. Carried so a turn that scoped nothing
    # is distinguishable from a turn that was never scoped.
    reason: str | None = None
    scores: tuple[tuple[str, float], ...] = field(default=())
    matched_skills: tuple[str, ...] = field(default=())

    def notice(self) -> str:
        """The model-facing section naming the withheld capabilities, or `""`.

        Empty when nothing was withheld: a notice that says "0 tools were withheld" on
        every unscoped turn is noise, and the absence of the section already says it.
        """
        if not self.withheld and not self.matched_skills:
            return ""
        lines: list[str] = ["[Tool Scoping]"]
        if self.withheld:
            named = self.withheld[:MAX_NAMED_WITHHELD]
            lines.extend(
                [
                    f"{len(self.selected)} of {len(self.selected) + len(self.withheld)} registered "
                    f"tools are advertised this turn, selected by {self.method}.",
                    f"{len(self.withheld)} were withheld: " + ", ".join(named),
                ]
            )
            if len(self.withheld) > len(named):
                lines.append(f"...and {len(self.withheld) - len(named)} more, not named here.")
            lines.append(
                "A withheld tool is registered and still executes: emit a call for it by name "
                "in this turn and it will run, exactly as an advertised one would. Its schema "
                "is not shown here, so give the arguments the tool documents. Do not report a "
                "task impossible because its tool is missing from this list."
            )
        if self.matched_skills:
            lines.append(
                f"Relevant approved skills for this context: {', '.join(self.matched_skills)}. "
                "Call load_skill(skill_name=...) to activate them if needed."
            )
        return "\n".join(lines)


class ToolScoperProtocol(Protocol):
    """Selects the tools advertised for one turn.

    Declared `async` because an embedding-backed scoper calls an embedder, which is I/O.
    A synchronous implementation satisfies it with an `async def` that never awaits.
    """

    async def scope_tools(self, query: str, tools: tuple[ToolDefinition, ...]) -> ToolScopingResult:
        """Return the advertised subset, the withheld names, and the method used."""
        ...


class LexicalToolScoper:
    """Scores tools by literal token overlap with the query. Not semantic.

    The eval's per-turn baseline (arm B); no agent is built with it.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def score_tool(self, query: str, tool: ToolDefinition) -> int:
        """Overlap score: an exact name mention dominates a description-word match."""
        lower_query = query.lower()
        score = 0
        if tool.name.lower() in lower_query:
            score += 10
        for word in tool.description.lower().split():
            if len(word) > 4 and word in lower_query:
                score += 1
        return score

    async def scope_tools(self, query: str, tools: tuple[ToolDefinition, ...]) -> ToolScopingResult:
        """Select the top-k tools by lexical overlap, withholding nothing without signal."""
        if not query:
            return ToolScopingResult(selected=tools, reason="no query to score against")
        if len(tools) <= self.top_k:
            return ToolScopingResult(
                selected=tools, reason=f"{len(tools)} tools is within the top-{self.top_k} budget"
            )

        scored = [(self.score_tool(query, tool), index, tool) for index, tool in enumerate(tools)]
        if all(score == 0 for score, _, _ in scored):
            # No tool overlaps the query. Ranking by an all-zero score is registry order
            # wearing a rank's clothes, so this withholds nothing rather than guessing.
            return ToolScopingResult(
                selected=tools, method="lexical overlap", reason="no tool scored above zero"
            )

        # The name breaks ties, so which tools an equal score keeps does not depend on the
        # order they were passed in.
        scored.sort(key=lambda entry: (-entry[0], entry[2].name))
        selected = tuple(tool for _, _, tool in scored[: self.top_k])
        withheld = tuple(tool.name for _, _, tool in scored[self.top_k :])
        return ToolScopingResult(
            selected=_by_name(selected),
            withheld=withheld,
            method="lexical overlap",
            scores=tuple((tool.name, float(score)) for score, _, tool in scored),
        )


__all__ = [
    "MAX_NAMED_WITHHELD",
    "LexicalToolScoper",
    "ToolScoperProtocol",
    "ToolScopingResult",
]
