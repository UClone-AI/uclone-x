"""Durable cross-session memory store for typed facts with P6 provenance.

A clone's facts are the edges of its knowledge file (`knowledge.sqlite3`, uGraph design §5
step 3): `CrossSessionMemory` is the clone's `MemoryFact` surface over a
`SqliteKnowledgeStore`. Each write is one sqlite transaction, so two processes writing to
one clone lose nothing, and nothing is held between calls but the embedding index.

Provides structured persistence, conflict resolution (a later fact on the same subject and
predicate ends the earlier one's belief, and both stay), retraction management, bounded
progressive disclosure for prompt injection, and a synthesis adapter according to
Principles P6, P7, P8, and P9.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from uclone_x.core.agent_home import AgentHome
from uclone_x.core.provenance import Provenance, require_provenance
from uclone_x.core.set_aside import set_aside, set_aside_unreadable
from uclone_x.errors import MemoryStoreUnreadableError
from uclone_x.knowledge.sqlite_store import KnowledgeUnreadableError, SqliteKnowledgeStore
from uclone_x.llm.protocols import EmbedderProtocol
from uclone_x.memory.graph import fact_of, retract_edge, supersede_edge, write_fact
from uclone_x.memory.legacy_import import import_legacy_memory
from uclone_x.memory.models import FactOrigin, MemoryFact, fold_name, utc_now_iso
from uclone_x.memory.retrieval import FactRanking, rank_facts
from uclone_x.memory.vector_store import BruteForceVectorStore

logger = logging.getLogger(__name__)

#: The scope of a store made without an agent: tests and tools that hold no clone.
LOCAL_AGENT_ID = "local"


def set_aside_database(path: Path) -> str | None:
    """Move a database we could not read, and its `-wal` / `-shm` files, out of the way.

    The sidecars go too: a write-ahead log left beside a new file would be replayed into it.
    Returns where the main file went, or `None` when it could not be moved.
    """
    try:
        moved = str(set_aside_unreadable(path))
    except OSError as exc:
        logger.error("Could not set aside unreadable knowledge file %s: %s", path, exc)
        return None
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists():
            try:
                set_aside(sidecar, ".unreadable-")
            except OSError as exc:
                logger.error("Could not set aside %s: %s", sidecar, exc)
    return moved


def _statements_now(store: SqliteKnowledgeStore) -> dict[str, tuple[str, str, str]]:
    """`fact_id -> (subject, predicate, object)` of what holds now: the reasoner's premises."""
    return store.statements_at("wall", utc_now_iso())


@dataclass(frozen=True)
class SavedMemory:
    """What a clone's knowledge file holds, read for display.

    `facts`: the active facts, oldest first. `statements`: the same ones that hold now (a
    fact with a stated end is listed but not here), keyed by `fact_id`, for the ontology
    reasoner.
    """

    facts: list[MemoryFact]
    statements: dict[str, tuple[str, str, str]]


def default_cross_session_memory(username: str) -> CrossSessionMemory:
    """Build the memory store a host wires when it has no opinion about the path.

    The knowledge file lives inside that clone's own directory (`AgentHome`), beside the file
    recording its opaque id, so one clone's durable state is one directory. A `memory.json`
    left there by an earlier build is imported into it once (`legacy_import`).

    It replaces a derivation that folded every character it disliked into `_`:
    `a.b`, `a/b`, `a b` and `a_b` all resolved to `a_b.json`, which is four agents
    sharing one memory with nothing reporting it. `AgentHome.for_handle` refuses such
    a name instead, naming the agent that carries it (P6).

    `username` is a clone's id, or its handle resolved to the id (clone-data-scopes §3.3). A name
    no clone carries is refused, never given a home: a clone is created in one step with
    its id (§3.4), so there is no lazy home and nothing here mints.

    Raises:
        CloneNotFoundError: No clone is called `username`.
        DuplicateHandleError: More than one clone is.
        AgentHomeError: `username` cannot be a clone's name.
    """
    home = AgentHome.for_ref(username)
    return CrossSessionMemory(
        storage_path=home.knowledge_path,
        agent_id=home.agent_id(),
        legacy_path=home.memory_path,
    )


def read_saved_memory(username: str) -> SavedMemory | None:
    """What `username`'s knowledge file holds, read without making a home or a damaged file worse.

    For a reader that shows what a clone remembers (the dock's Remembers, #1401). It reads the
    same file `default_cross_session_memory` writes, so it sees a fact saved by any process.

    Retracted and superseded facts are left out: they are no longer injected or recalled, so a
    list of what the clone remembers that showed them would be wrong about the clone.

    Returns `None` when the clone has neither a knowledge file nor a `memory.json` to import,
    which is not the same answer as a file holding no facts.

    It opens the file read-only and never moves a damaged one aside (a read must not): it
    raises instead. The one write it can make is the one-time import of a `memory.json` that
    has not been imported yet, because without it the answer would be missing facts.

    Raises:
        MemoryStoreUnreadableError: The file is there and cannot be read. Its message is
            plain; the path and the reason are on `path` and `cause`.
        AgentHomeError: `username` names no clone, more than one, or cannot be a name.
    """
    home = AgentHome.for_ref(username)
    knowledge, legacy = home.knowledge_path, home.memory_path
    if not knowledge.is_file() and not legacy.is_file():
        return None
    agent_id = home.recorded_agent_id() or home.agent_id()
    try:
        if legacy.is_file():
            store = SqliteKnowledgeStore(knowledge, agent_id)
            try:
                import_legacy_memory(store, legacy)
            except (KnowledgeUnreadableError, sqlite3.OperationalError):
                raise
            except Exception as exc:
                raise MemoryStoreUnreadableError(
                    "The saved memory could not be read.",
                    path=legacy,
                    cause=f"{type(exc).__name__}: {exc}",
                ) from exc
        else:
            store = SqliteKnowledgeStore(knowledge, agent_id, create=False)
        facts = [fact_of(row) for row in store.rows()]
        statements = _statements_now(store)
    except KnowledgeUnreadableError as exc:
        raise MemoryStoreUnreadableError(
            "The saved memory could not be read.", path=exc.path, cause=exc.cause
        ) from exc
    active = [fact for fact in facts if not fact.retracted]
    active.sort(key=lambda fact: fact.created_at)
    return SavedMemory(active, statements)


def read_saved_facts(username: str) -> list[MemoryFact] | None:
    """The active facts of `username`'s memory, oldest first: `read_saved_memory`'s `facts`."""
    saved = read_saved_memory(username)
    return None if saved is None else saved.facts


class CrossSessionMemory:
    """Manages cross-session memory facts across distinct agent sessions.

    Key capabilities:
    1. Structured, typed facts (subject, predicate, object_value) carrying in-band P6 provenance.
    2. Automatic conflict detection and supersession (conflicts_with).
    3. Explicit retraction tracking (retract_fact) preserving audit history.
    4. Bounded progressive disclosure prompt formatting to prevent unbounded context growth.
    5. Integration as synthesis input for SkillSynthesizer (P9).

    Facts live in a `SqliteKnowledgeStore`: at `storage_path`, or in memory when it is
    `None`. `agent_id` names the scope (`clone:<agent id>`); a file made for another id is
    refused. `legacy_path` is a `memory.json` of an earlier build, imported once.
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        max_facts_in_prompt: int = 10,
        embedder: EmbedderProtocol | None = None,
        *,
        agent_id: str = LOCAL_AGENT_ID,
        legacy_path: Path | str | None = None,
    ) -> None:
        self._storage_path = Path(storage_path) if storage_path is not None else None
        self._max_facts_in_prompt = max(1, max_facts_in_prompt)
        # Optional by design. Without it, `search_facts` still works and reports that it
        # ranked lexically; it never presents a lexical ranking as a semantic one.
        self._embedder = embedder
        self._vector_store: BruteForceVectorStore | None = (
            None
            if embedder is None
            else BruteForceVectorStore(
                dimensions=embedder.dimensions, model_name=embedder.model_name
            )
        )
        # Set when a store existed and could not be read. Carried rather than swallowed:
        # an empty memory and an unreadable memory are different claims, and the second
        # one reaches the model as "you have no prior facts" if nobody says otherwise.
        self._load_failure: str | None = None
        self._store = self._open(agent_id)
        if legacy_path is not None and Path(legacy_path).is_file():
            self._import_legacy(Path(legacy_path))

    def _open(self, agent_id: str) -> SqliteKnowledgeStore:
        """Open the knowledge file; one that cannot be read is set aside and a new one made.

        Set aside rather than recreated over: the next write would otherwise bury the one
        copy of those facts. `load_failure` makes the gap visible in the prompt instead of
        letting it read as "no prior facts" (P6).
        """
        try:
            return SqliteKnowledgeStore(self._storage_path, agent_id)
        except KnowledgeUnreadableError as exc:
            if self._storage_path is None:
                raise
            moved = set_aside_database(self._storage_path)
            self._load_failure = (
                f"the store at {self._storage_path} could not be read ({exc.cause})"
                + (f"; it was moved to {moved}" if moved else "")
            )
            logger.error("Failed to load cross-session memory: %s", self._load_failure)
            return SqliteKnowledgeStore(self._storage_path, agent_id)

    def _import_legacy(self, legacy_path: Path) -> None:
        """Import an earlier build's `memory.json` once; one that cannot be read is set aside."""
        try:
            import_legacy_memory(self._store, legacy_path)
        except sqlite3.OperationalError:
            raise
        except Exception as exc:
            try:
                moved: str | None = str(set_aside_unreadable(legacy_path))
            except OSError as move_exc:
                moved = None
                logger.error("Could not set aside %s: %s", legacy_path, move_exc)
            self._load_failure = f"the store at {legacy_path} could not be read ({exc})" + (
                f"; it was moved to {moved}" if moved else ""
            )
            logger.error("Failed to import cross-session memory: %s", self._load_failure)

    @property
    def storage_path(self) -> Path | None:
        """Configured on-disk path of the knowledge file, if any."""
        return self._storage_path

    @property
    def knowledge(self) -> SqliteKnowledgeStore:
        """The clone's knowledge store: the facts are its edges."""
        return self._store

    @property
    def load_failure(self) -> str | None:
        """Why the on-disk store could not be read, if it could not be."""
        return self._load_failure

    @property
    def max_facts_in_prompt(self) -> int:
        """Maximum number of memory facts injected into system prompt."""
        return self._max_facts_in_prompt

    def statements_now(self) -> dict[str, tuple[str, str, str]]:
        """`fact_id -> (subject, predicate, object)` of the facts that hold now."""
        return _statements_now(self._store)

    def record_fact(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        provenance: Provenance,
        source_session_id: str,
        confidence: float = 1.0,
        tags: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
        auto_retract_conflicts: bool = True,
        origin: FactOrigin = "saved",
        source_room_id: str | None = None,
        source_turn_id: str | None = None,
        fact_id: str | None = None,
        valid_until: str | None = None,
    ) -> MemoryFact:
        """Record a new verified cross-session fact carrying P6 provenance.

        `origin`, `source_room_id` and `source_turn_id` say how, and in which conversation
        and turn, the fact was learned (clone-knowledge-graph design §3.2).

        `fact_id` is for a writer that must be idempotent across processes: a second write
        that names an id already held returns the fact held and changes nothing. Every
        other caller leaves it to the default.

        `valid_until` is the end a person stated for the fact; it fills the edge's
        `valid.end`. A fact with an end is history: it neither supersedes nor is
        superseded (`MemoryFact.conflicts_with`).

        The subject is resolved through the knowledge store's entity resolver: an exact name
        is the entity already there, and any other is a new one.
        """
        verified_prov = require_provenance(provenance, "MemoryFact")

        subj = subject.strip()
        pred = predicate.strip()
        val = object_value.strip()
        sess = source_session_id.strip()

        if not subj:
            raise ValueError("subject cannot be empty")
        if not pred:
            raise ValueError("predicate cannot be empty")
        if not val:
            raise ValueError("object_value cannot be empty")
        if not sess:
            raise ValueError("source_session_id cannot be empty")

        new_fact = MemoryFact(
            subject=subj,
            predicate=pred,
            object_value=val,
            provenance=verified_prov,
            source_session_id=sess,
            confidence=confidence,
            tags=tuple(tags),
            metadata=dict(metadata or {}),
            origin=origin,
            source_room_id=source_room_id,
            source_turn_id=source_turn_id,
            valid_until=valid_until,
        )
        if fact_id is not None:
            new_fact = new_fact.model_copy(update={"fact_id": fact_id})

        superseded: list[str] = []
        with self._store.transaction() as tx:
            if fact_id is not None:
                held = tx.row(fact_id)
                if held is not None:
                    return fact_of(held)
            if auto_retract_conflicts:
                for row in tx.rows():
                    existing = fact_of(row)
                    if existing.conflicts_with(new_fact):
                        superseded.append(existing.fact_id)
                        supersede_edge(
                            tx, existing.fact_id, by=new_fact.fact_id, at=new_fact.created_at
                        )
                        logger.info(
                            "Fact %s superseded and retracted by conflicting fact %s",
                            existing.fact_id,
                            new_fact.fact_id,
                        )
            if superseded:
                new_fact = new_fact.model_copy(update={"contradicts_fact_id": superseded[-1]})
            write_fact(tx, new_fact)
        # Same reason as the explicit retraction path: a retracted fact's vector is an
        # index entry with no corpus row behind it, and a store that supersedes often would
        # grow one per supersession forever.
        for gone in superseded:
            self.forget_vector(gone)
        return new_fact

    def retract_fact(
        self,
        fact_id: str,
        reason: str,
        provenance: Provenance,
        session_id: str | None = None,
    ) -> MemoryFact:
        """Explicitly retract a fact by ID, preserving the audit trail.

        The fact is read inside the write, so one another process saved a moment ago can be
        retracted, and a retraction made elsewhere is not repeated.
        """
        require_provenance(provenance, "retract_fact")
        clean_reason = reason.strip() or "No retraction reason provided"
        with self._store.transaction() as tx:
            held = tx.row(fact_id)
            if held is None:
                raise KeyError(f"Memory fact '{fact_id}' not found")
            current = fact_of(held)
            if current.retracted:
                return current
            now = utc_now_iso()
            retract_edge(tx, fact_id, reason=clean_reason, at=now, updated_at=now)
            retracted = current.model_copy(
                update={
                    "retracted": True,
                    "retraction_reason": clean_reason,
                    "retracted_at": now,
                    "updated_at": now,
                }
            )
        # A retracted fact is excluded from every candidate set, so its vector is dead
        # weight in the index. Dropping it here keeps the index the same size as the
        # corpus it claims to describe. After the write, so a failed one leaves it indexed.
        self.forget_vector(fact_id)
        logger.info("Retracted fact %s: %s", fact_id, clean_reason)
        return retracted

    def correct_fact(
        self,
        fact_id: str,
        value: str,
        provenance: Provenance,
        reason: str = "corrected by the user",
    ) -> MemoryFact:
        """A person's correction: a `corrected` fact that supersedes `fact_id` (design §3.6).

        The new fact keeps the old one's subject and predicate, and where it was learned
        (session, conversation, turn), so it stays in the same place in Remembers. It is
        written at confidence `1.0` with origin `corrected`, and names the old fact in
        `contradicts_fact_id`. The old fact is kept, its belief ended (`expired_at`) with
        `reason`. Any other active fact on the same subject and predicate is superseded as
        `record_fact` does.

        The fact is read inside the write, and all of it is one transaction: a failure
        leaves the old fact as it was, so a retried correction does the whole edit again.

        Raises:
            KeyError: No fact has that id.
            ValueError: The fact is already retracted, or `value` is blank.
        """
        require_provenance(provenance, "correct_fact")
        clean_value = value.strip()
        if not clean_value:
            raise ValueError("value cannot be empty")
        superseded: list[str] = []
        with self._store.transaction() as tx:
            held = tx.row(fact_id)
            if held is None:
                raise KeyError(f"Memory fact '{fact_id}' not found")
            current = fact_of(held)
            if current.retracted:
                raise ValueError(f"Memory fact '{fact_id}' is retracted and cannot be corrected")
            if current.object_value == clean_value:
                return current

            corrected = MemoryFact(
                subject=current.subject,
                predicate=current.predicate,
                object_value=clean_value,
                provenance=require_provenance(provenance, "MemoryFact"),
                source_session_id=current.source_session_id,
                confidence=1.0,
                tags=current.tags,
                origin="corrected",
                source_room_id=current.source_room_id,
                source_turn_id=current.source_turn_id,
                contradicts_fact_id=current.fact_id,
            )
            now = corrected.created_at
            superseded.append(current.fact_id)
            supersede_edge(tx, current.fact_id, by=corrected.fact_id, at=now, reason=reason)
            for row in tx.rows():
                existing = fact_of(row)
                if existing.conflicts_with(corrected):
                    superseded.append(existing.fact_id)
                    supersede_edge(tx, existing.fact_id, by=corrected.fact_id, at=now)
            write_fact(tx, corrected)
        for fact_id_gone in superseded:
            self.forget_vector(fact_id_gone)
        logger.info("Fact %s corrected as %s", current.fact_id, corrected.fact_id)
        return corrected

    def refresh(self) -> None:
        """Drop the embeddings of facts another writer withdrew since this object last looked.

        The facts themselves are read from the file on every call, so a reader that decides
        from them (the knowledge extractor asking whether a person corrected a fact, or
        already knows it) sees a Forget or Correct made through another process (#1404). What
        is held between calls is the embedding index, and this keeps it the size of the
        corpus it describes.
        """
        if self._vector_store is None:
            return
        for fact in self.list_facts(include_retracted=True):
            if fact.retracted:
                self.forget_vector(fact.fact_id)

    def reinforce_fact(self, fact_id: str, turn_id: str) -> MemoryFact:
        """Record that the fact was observed again in `turn_id` (design §3.3 step 5).

        The fact itself is unchanged: its value, confidence and origin stay as saved. The
        turn is appended to `metadata["observations"]` (once), and `updated_at` moves.

        Raises:
            KeyError: No fact has that id.
            ValueError: The fact is retracted.
        """
        with self._store.transaction() as tx:
            held = tx.row(fact_id)
            if held is None:
                raise KeyError(f"Memory fact '{fact_id}' not found")
            current = fact_of(held)
            if current.retracted:
                raise ValueError(f"Memory fact '{fact_id}' is retracted and cannot be reinforced")
            seen = current.metadata.get("observations")
            observations = (
                [str(turn) for turn in cast("list[object]", seen)] if isinstance(seen, list) else []
            )
            if turn_id in observations:
                return current
            now = utc_now_iso()
            metadata = {**current.metadata, "observations": [*observations, turn_id]}
            tx.update_edge(fact_id, record={"metadata": metadata, "updated_at": now})
            return current.model_copy(update={"metadata": metadata, "updated_at": now})

    def get_fact(self, fact_id: str) -> MemoryFact | None:
        """Look up a fact by ID."""
        held = self._store.row(fact_id)
        return None if held is None else fact_of(held)

    def list_facts(
        self,
        include_retracted: bool = False,
        subject: str | None = None,
        predicate: str | None = None,
        tags: Sequence[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryFact]:
        """List memory facts matching filter criteria.

        `subject` and `predicate` match folded (`fold_name`, NFC included), so a subject
        stored decomposed before #1895 matches the composed form asked for now (#1893).
        """
        results: list[MemoryFact] = []
        target_tags = set(tags) if tags else None
        target_subj = fold_name(subject) if subject else None
        target_pred = fold_name(predicate) if predicate else None

        for fact in (fact_of(row) for row in self._store.rows()):
            if not include_retracted and fact.retracted:
                continue
            if fact.confidence < min_confidence:
                continue
            if target_subj and fold_name(fact.subject) != target_subj:
                continue
            if target_pred and fold_name(fact.predicate) != target_pred:
                continue
            if target_tags and not target_tags.issubset(set(fact.tags)):
                continue
            results.append(fact)

        results.sort(key=lambda f: f.created_at)
        return results

    async def search_facts(
        self,
        query: str,
        top_k: int = 5,
        include_retracted: bool = False,
        subject: str | None = None,
        predicate: str | None = None,
        tags: Sequence[str] | None = None,
        min_confidence: float = 0.0,
    ) -> FactRanking:
        """Rank facts against a natural-language query, naming the ranking method.

        The structured filters still apply and still narrow the candidate set; the query
        orders what survives them. An agent that knows how a fact was filed keeps the exact
        lookup it always had, and one that only remembers what the fact was *about* now has
        a way to ask.
        """
        candidates = self.list_facts(
            include_retracted=include_retracted,
            subject=subject,
            predicate=predicate,
            tags=tags,
            min_confidence=min_confidence,
        )
        return await rank_facts(
            query=query,
            facts=candidates,
            top_k=top_k,
            embedder=self._embedder,
            vector_store=self._vector_store,
        )

    @property
    def vector_store(self) -> BruteForceVectorStore | None:
        """The embedding index, or `None` when no embedder is wired.

        Exposed because what the index holds is not observable from a search result: a
        retracted fact is filtered out of the ranking either way, so only reading the
        index shows whether its entry was actually dropped.
        """
        return self._vector_store

    def forget_vector(self, fact_id: str) -> None:
        """Drop a fact's cached embedding so a rewritten fact is not matched by its old text."""
        if self._vector_store is not None:
            self._vector_store.remove(fact_id)

    def find_conflicts(self, fact: MemoryFact) -> list[MemoryFact]:
        """Find active facts that conflict with the given fact."""
        return [f for f in self.list_facts() if f.conflicts_with(fact)]

    def format_prompt_section(
        self,
        max_facts: int | None = None,
        min_confidence: float = 0.5,
        relevance_tags: Sequence[str] | None = None,
    ) -> str:
        """Format progressive disclosure prompt section bounded to avoid unbounded growth."""
        active_facts = self.list_facts(
            include_retracted=False,
            tags=relevance_tags,
            min_confidence=min_confidence,
        )
        if not active_facts:
            if self._load_failure is not None:
                return (
                    "[Cross-Session Memory Facts]\n"
                    f"UNAVAILABLE: {self._load_failure}. Prior facts exist but cannot be read; "
                    "do not treat this turn as evidence that none were recorded."
                )
            return ""

        # `fact_id` breaks ties, so facts of equal confidence and age do not come out in
        # insertion order.
        active_facts.sort(key=lambda f: (f.confidence, f.created_at, f.fact_id), reverse=True)

        limit = max_facts if max_facts is not None else self._max_facts_in_prompt
        selected = active_facts[:limit]

        lines = [
            "[Cross-Session Memory Facts]",
            "The following verified durable facts from prior sessions are active:",
        ]
        # The source session and the confidence rank the facts but are not rendered: they
        # change the text between sessions without telling the model anything it acts on.
        for f in selected:
            lines.append(f"- {f.summary()}")

        if len(active_facts) > limit:
            omitted = len(active_facts) - limit
            lines.append(
                f"({omitted} additional facts omitted for prompt bounds; query memory to view)"
            )

        return "\n".join(lines)

    # ----------------------------------------------------------------------
    # P9 SkillSynthesizer Input Adapter
    # ----------------------------------------------------------------------
    def to_skill_workflow_steps(
        self,
        subject: str | None = None,
        tags: Sequence[str] | None = None,
        min_confidence: float = 0.7,
    ) -> list[str]:
        """Extract durable memory facts as sequential actionable steps for SkillSynthesizer (P9)."""
        facts = self.list_facts(
            include_retracted=False,
            subject=subject,
            tags=tags,
            min_confidence=min_confidence,
        )
        if not facts:
            return []

        steps: list[str] = []
        for fact in facts:
            steps.append(
                f"Apply verified rule '{fact.subject} {fact.predicate}': {fact.object_value}"
            )
        return steps
