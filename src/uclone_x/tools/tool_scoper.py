"""Per-turn tool scoping: advertise a subset of the registry, and say so.

Two things are fixed here relative to the `SemanticToolScoper` this module replaces.

**The name.** That class scored a tool by lowercase substring overlap between the query
and the tool's name and description, and called the result semantic. It is lexical, and
the class is now called what it does. No embedding-backed scoper exists yet; when one is
written it is the one entitled to the other word, and `ToolScoperProtocol` is the seam it
plugs into. Until then the word is simply not claimed by anything here.

**The silence.** Withholding a tool withholds a *capability*, and the previous
implementation withheld without telling the model — indistinguishable, from inside the
turn, from a registry that never held the tool. A model that does not know a capability
was withheld reports the task impossible instead of asking for it. So scoping returns a
`ToolScopingResult` carrying what was withheld and by which method, and the turn carries
that notice into the context.

A third correction is in `LexicalToolScoper` itself: a query no tool scores above zero on
used to return the first `top_k` tools in registry order, which is a selection with no
signal behind it presented as a selection. It now withholds nothing and says the method
found no signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from uclone_x.llm.models import ToolDefinition

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from uclone_x.llm.protocols import EmbedderProtocol

# How many withheld tool names the notice lists before it stops naming them. The count is
# always exact; the names are a courtesy that must not itself become the context problem
# scoping exists to solve.
MAX_NAMED_WITHHELD = 20


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


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

    Kept because it needs no embedder and therefore always runs — P0's default
    composition has to work before anything optional is configured. It is the only scoper
    in the tree: an embedding-backed one would implement `ToolScoperProtocol` beside it.
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

        # `index` breaks ties, so equal scores keep registry order rather than whatever
        # order the sort happened to produce.
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        selected = tuple(tool for _, _, tool in scored[: self.top_k])
        withheld = tuple(tool.name for _, _, tool in scored[self.top_k :])
        return ToolScopingResult(
            selected=selected,
            withheld=withheld,
            method="lexical overlap",
            scores=tuple((tool.name, float(score)) for score, _, tool in scored),
        )


class SemanticToolScoper:
    """Scores and scopes tools and skills using text embeddings (Principle 5 & 6).

    Embeds tools once, caching vectors in memory. On each turn, embeds the query
    and ranks tools by cosine similarity. Tools above `threshold` are selected up
    to `top_k`. If no tool reaches `threshold`, advertises only `always_include`
    (or nothing), saving prompt context on pure chat.

    If the embedder fails or is unavailable, falls back gracefully to LexicalToolScoper.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        *,
        top_k: int = 5,
        threshold: float = 0.30,
        always_include: Sequence[str] = (),
        skills_provider: Callable[[], Sequence[tuple[str, str]]] | None = None,
    ) -> None:
        self.embedder = embedder
        self.top_k = top_k
        self.threshold = threshold
        self.always_include = frozenset(always_include)
        self.skills_provider = skills_provider
        self._tool_cache: dict[tuple[str, str], tuple[float, ...]] = {}
        self._skill_cache: dict[tuple[str, str], tuple[float, ...]] = {}
        self._fallback_scoper = LexicalToolScoper(top_k=top_k)

    async def scope_tools(self, query: str, tools: tuple[ToolDefinition, ...]) -> ToolScopingResult:
        """Select relevant tools using semantic embedding similarity."""
        if not query or not query.strip():
            return ToolScopingResult(selected=tools, reason="no query to score against")

        try:
            # 1. Ensure tools are embedded and cached
            needed_tools: list[str] = []
            needed_keys: list[tuple[str, str]] = []
            for t in tools:
                k = (t.name, t.description or "")
                if k not in self._tool_cache:
                    needed_keys.append(k)
                    needed_tools.append(f"{t.name}: {t.description or ''}")

            if needed_tools:
                tool_vectors = await self.embedder.embed(needed_tools)
                for k, vec in zip(needed_keys, tool_vectors, strict=True):
                    self._tool_cache[k] = vec

            # 2. Embed query
            query_vectors = await self.embedder.embed([query])
            if not query_vectors:
                return await self._fallback_scoper.scope_tools(query, tools)
            query_vec = query_vectors[0]

            # 3. Score tools
            scored_tools: list[tuple[float, int, ToolDefinition]] = []
            for index, t in enumerate(tools):
                k = (t.name, t.description or "")
                t_vec = self._tool_cache.get(k)
                sim = _cosine_similarity(query_vec, t_vec) if t_vec else 0.0
                scored_tools.append((sim, index, t))

            # 4. Sort and filter
            scored_tools.sort(key=lambda entry: (-entry[0], entry[1]))

            selected_tools: list[ToolDefinition] = []
            withheld_names: list[str] = []

            for sim, _, tool in scored_tools:
                is_always = tool.name in self.always_include
                if (sim >= self.threshold or is_always) and len(selected_tools) < self.top_k:
                    selected_tools.append(tool)
                else:
                    withheld_names.append(tool.name)

            # 5. Check skills if provider is given
            matched_skills: list[str] = []
            if self.skills_provider is not None:
                try:
                    skills = self.skills_provider()
                    needed_skills: list[str] = []
                    needed_skill_keys: list[tuple[str, str]] = []
                    for s_name, s_desc in skills:
                        sk = (s_name, s_desc)
                        if sk not in self._skill_cache:
                            needed_skill_keys.append(sk)
                            needed_skills.append(f"{s_name}: {s_desc}")
                    if needed_skills:
                        s_vecs = await self.embedder.embed(needed_skills)
                        for sk, s_vec in zip(needed_skill_keys, s_vecs, strict=True):
                            self._skill_cache[sk] = s_vec

                    skill_scored: list[tuple[float, str]] = []
                    for s_name, s_desc in skills:
                        sk = (s_name, s_desc)
                        s_vec = self._skill_cache.get(sk)
                        if s_vec:
                            s_sim = _cosine_similarity(query_vec, s_vec)
                            if s_sim >= self.threshold:
                                skill_scored.append((s_sim, s_name))
                    skill_scored.sort(key=lambda entry: -entry[0])
                    matched_skills = [name for _, name in skill_scored[:3]]
                except Exception:
                    pass

            return ToolScopingResult(
                selected=tuple(selected_tools),
                withheld=tuple(withheld_names),
                method="semantic similarity",
                scores=tuple((tool.name, round(sim, 4)) for sim, _, tool in scored_tools),
                matched_skills=tuple(matched_skills),
                reason=(
                    "no tool met similarity threshold"
                    if not selected_tools
                    else f"selected {len(selected_tools)} tools above {self.threshold} threshold"
                ),
            )
        except Exception:
            return await self._fallback_scoper.scope_tools(query, tools)


__all__ = [
    "MAX_NAMED_WITHHELD",
    "LexicalToolScoper",
    "SemanticToolScoper",
    "ToolScoperProtocol",
    "ToolScopingResult",
]
