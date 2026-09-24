"""Built-in tools for agent interaction with CrossSessionMemory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uclone_x.core.provenance import Provenance
from uclone_x.memory.retrieval import FactRanking
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

#: How every memory tool's arguments are validated. **Not strict**, because these arguments
#: are a model's JSON: strict validation accepts only a Python `tuple` for `tuple[str, ...]`,
#: which no JSON decoder produces, so every `record_memory_fact` that passed `tags` -- as
#: the advertised schema's `array` invites -- was refused, and nothing was saved (#1375).
#: Unknown keys are still refused, by name.
_MODEL_ARGUMENTS = ConfigDict(frozen=True, extra="forbid")


class RecordMemoryFactParams(BaseModel):
    """Parameters for recording a verified cross-session fact."""

    model_config = _MODEL_ARGUMENTS

    subject: str = Field(description="The entity or domain subject of the fact.")
    predicate: str = Field(description="The relation or attribute name.")
    object_value: str = Field(description="The value or statement asserted.")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]."
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple, description="Optional categorization tags."
    )

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, v: object) -> tuple[str, ...] | object:
        if isinstance(v, (list, set)):
            items = cast(list[object] | set[object], v)
            return tuple(str(x) for x in items)
        return v


class RecordMemoryFactTool(BaseTool[RecordMemoryFactParams]):
    """Tool for recording a durable cross-session memory fact carrying P6 provenance."""

    name: str = "record_memory_fact"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    not_run_note: ClassVar[str] = "Nothing was saved to memory."
    description: str = (
        "Record a verified cross-session fact (subject, predicate, object_value) into durable memory. "
        "Carries in-band P6 provenance and automatically supersedes conflicting prior assertions."
    )

    def __init__(self, memory: CrossSessionMemory) -> None:
        super().__init__()
        self._memory = memory

    async def run(self, params: RecordMemoryFactParams, context: ToolContext) -> str:
        agent_id = context.agent_id or "default"
        session_id = context.session_id or "default"
        provenance = Provenance.primary(
            provider=f"agent.{agent_id}",
            model="memory",
        )
        try:
            fact = self._memory.record_fact(
                subject=params.subject,
                predicate=params.predicate,
                object_value=params.object_value,
                provenance=provenance,
                source_session_id=session_id,
                confidence=params.confidence,
                tags=params.tags,
            )
        except ValueError as exc:
            # The store refuses an empty field before it touches anything, so this one
            # sentence is true of every `ValueError` it raises. Said in the result because
            # the model reads the result, and "subject cannot be empty" alone does not tell
            # it that the fact is not in memory (#1375).
            raise ValueError(f"Nothing was saved to memory: {exc}") from exc
        output = (
            f"Successfully recorded memory fact '{fact.fact_id}': "
            f"{fact.summary()} (confidence={fact.confidence:.2f})"
        )
        if fact.contradicts_fact_id:
            output += f" [superseded conflicting fact '{fact.contradicts_fact_id}']"
        return output


class RetractMemoryFactParams(BaseModel):
    """Parameters for retracting a memory fact."""

    model_config = _MODEL_ARGUMENTS

    fact_id: str = Field(description="ID of the memory fact to retract.")
    reason: str = Field(description="Reason for retraction.")


class RetractMemoryFactTool(BaseTool[RetractMemoryFactParams]):
    """Tool for retracting a memory fact with explicit audit trail."""

    name: str = "retract_memory_fact"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Explicitly retract an outdated or incorrect cross-session memory fact. "
        "Preserves audit provenance while removing the fact from active injection."
    )

    def __init__(self, memory: CrossSessionMemory) -> None:
        super().__init__()
        self._memory = memory

    async def run(self, params: RetractMemoryFactParams, context: ToolContext) -> str:
        agent_id = context.agent_id or "default"
        session_id = context.session_id or "default"
        provenance = Provenance.primary(
            provider=f"agent.{agent_id}",
            model="memory",
        )
        fact = self._memory.retract_fact(
            fact_id=params.fact_id,
            reason=params.reason,
            provenance=provenance,
            session_id=session_id,
        )
        return f"Successfully retracted memory fact '{fact.fact_id}': {fact.retraction_reason}"


class ReadOnlyMemory:
    """A memory store reached through its search alone, so nothing holding it can write.

    What a sub-agent gets when its parent shares its memory (#1431). The store rewrites its
    whole document on every save, with no lock, so a second writer would overwrite the
    parent's own saves. This view offers `search_facts` and nothing else: no record, no
    retract, no save.
    """

    __slots__ = ("_memory",)

    def __init__(self, memory: CrossSessionMemory) -> None:
        self._memory = memory

    async def search_facts(
        self,
        query: str,
        top_k: int = 5,
        include_retracted: bool = False,
        subject: str | None = None,
        predicate: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> FactRanking:
        """The store's own ranking, read without changing the document."""
        return await self._memory.search_facts(
            query=query,
            top_k=top_k,
            include_retracted=include_retracted,
            subject=subject,
            predicate=predicate,
            tags=tags,
        )


class QueryMemoryFactsParams(BaseModel):
    """Parameters for querying cross-session memory facts."""

    model_config = _MODEL_ARGUMENTS

    query: str = Field(
        default="",
        description=(
            "Natural-language description of what you are trying to recall. Results are "
            "ranked against it, and the reply states which method produced the ranking. "
            "Leave empty to list the most recent matching facts instead."
        ),
    )
    top_k: int = Field(
        default=5, ge=1, le=50, description="Maximum number of ranked facts to return."
    )
    subject: str | None = Field(default=None, description="Optional subject to filter by.")
    predicate: str | None = Field(default=None, description="Optional predicate to filter by.")
    tag: str | None = Field(default=None, description="Optional tag to filter by.")
    include_retracted: bool = Field(
        default=False, description="Whether to include retracted facts."
    )


class QueryMemoryFactsTool(BaseTool[QueryMemoryFactsParams]):
    """Tool for querying active or historical cross-session memory facts."""

    name: str = "query_memory_facts"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Recall cross-session memory facts. Give a natural-language `query` to have facts "
        "ranked by relevance, and/or `subject` / `predicate` / `tag` to filter exactly. "
        "The reply names the ranking method, because a lexical ranking and a semantic one "
        "fail differently and an empty result means different things under each."
    )

    def __init__(self, memory: CrossSessionMemory | ReadOnlyMemory) -> None:
        super().__init__()
        self._memory = memory

    async def run(self, params: QueryMemoryFactsParams, context: ToolContext) -> str:
        tags_seq: Sequence[str] | None = (params.tag,) if params.tag else None
        ranking = await self._memory.search_facts(
            query=params.query,
            top_k=params.top_k,
            include_retracted=params.include_retracted,
            subject=params.subject,
            predicate=params.predicate,
            tags=tags_seq,
        )
        if not ranking.ranked:
            # The method is reported on the empty path too — this is the case where its
            # absence does the damage, since "no matching memory facts" reads as a fact
            # about memory rather than about the matcher.
            return (
                f"No matching memory facts found. {ranking.describe()}"
                if ranking.considered
                else "No matching memory facts found. Memory holds no facts for these filters."
            )

        lines = [ranking.describe()]
        for entry in ranking.ranked:
            fact = entry.fact
            status = " [RETRACTED]" if fact.retracted else ""
            lines.append(
                f"- ID: {fact.fact_id} | {fact.summary()} | "
                f"session: {fact.source_session_id} | score: {entry.score:.3f}{status}"
            )
        return "\n".join(lines)
