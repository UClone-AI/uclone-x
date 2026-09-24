"""Durable cross-session memory store for typed facts with P6 provenance.

Provides structured persistence, conflict resolution, retraction management,
bounded progressive disclosure for prompt injection, and synthesis / ontology
promotion adapters according to Principles P6, P7, P8, and P9.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from uclone_x.core.agent_home import AgentHome
from uclone_x.core.provenance import Provenance, require_provenance
from uclone_x.errors import MemoryStoreUnreadableError
from uclone_x.llm.protocols import EmbedderProtocol
from uclone_x.memory.models import MemoryFact, utc_now_iso
from uclone_x.memory.retrieval import FactRanking, rank_facts
from uclone_x.memory.vector_store import BruteForceVectorStore
from uclone_x.ontology.models import EvidenceRecord

if TYPE_CHECKING:
    from uclone_x.ontology.engine import OntologyEngine
    from uclone_x.ontology.models import OntologyConcept

logger = logging.getLogger(__name__)

_DOCUMENT_VERSION = "1.0.0"


def _document_digest(serialized: str) -> str:
    """Identify a document by its bytes, so "unchanged" is not a guess about timestamps."""
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parse_facts(content: str) -> dict[str, MemoryFact]:
    """Read a document into facts by id, raising rather than returning a partial set."""
    data = json.loads(content)
    parsed: dict[str, MemoryFact] = {}
    for item in data.get("facts", []):
        fact = MemoryFact.model_validate_json(json.dumps(item))
        parsed[fact.fact_id] = fact
    return parsed


def _supersedes(candidate: MemoryFact, held: MemoryFact) -> bool:
    """Whether `candidate` is the later word on a fact both writers hold.

    Retraction is monotonic -- a fact is never un-retracted -- so a retracted copy always
    wins, regardless of clocks. Otherwise the later `updated_at` wins, and a tie keeps
    what we already hold, because a tie carries no evidence for changing it.
    """
    if candidate.retracted != held.retracted:
        return candidate.retracted
    return candidate.updated_at > held.updated_at


def default_cross_session_memory(username: str) -> CrossSessionMemory:
    """Build the memory store a host wires when it has no opinion about the path.

    The document lives inside that agent's own home (`AgentHome`), beside the file
    recording its opaque id, so one agent's durable state is one directory.

    It replaces a derivation that folded every character it disliked into `_`:
    `a.b`, `a/b`, `a b` and `a_b` all resolved to `a_b.json`, which is four agents
    sharing one memory with nothing reporting it. `AgentHome.for_username` refuses such
    a name instead, naming the agent that carries it (P6).

    The id is minted here, on the way past, because this is the one call that knows both
    the username and that the agent is being brought up. A home with memory in it and no
    id would be a directory this layout cannot explain.
    """
    home = AgentHome.for_username(username)
    home.agent_id()
    return CrossSessionMemory(storage_path=home.memory_path)


def read_saved_facts(username: str) -> list[MemoryFact] | None:
    """The facts `username`'s memory holds on disk, oldest first, read without changing anything.

    For a reader that shows what a clone remembers (the dock's Remembers, #1401). It reads the
    same document `default_cross_session_memory` writes -- the file every process holding
    that agent's store saves to after each fact -- so it sees a fact saved by any of them.

    Retracted facts are left out: a retracted fact is no longer injected or recalled, so a
    list of what the clone remembers that showed it would be wrong about the clone.

    Returns `None` when no memory document has been saved for the agent, which is not the
    same answer as a document holding no facts.

    Never writes. Constructing a `CrossSessionMemory` over an unreadable document moves it
    aside (so the next save cannot overwrite it); a read must not, so this parses the
    document itself and raises instead.

    Raises:
        MemoryStoreUnreadableError: The document is there and cannot be read. Its message is
            plain; the path and the parser's words are on `path` and `cause`.
        AgentHomeError: `username` cannot name an agent home.
    """
    path = AgentHome.for_username(username).memory_path
    if not path.is_file():
        return None
    try:
        facts = _parse_facts(path.read_text(encoding="utf-8"))
    except Exception as exc:  # the same breadth `load` treats as unreadable
        raise MemoryStoreUnreadableError(
            "The saved memory could not be read.", path=path, cause=f"{type(exc).__name__}: {exc}"
        ) from exc
    active = [fact for fact in facts.values() if not fact.retracted]
    active.sort(key=lambda fact: fact.created_at)
    return active


class CrossSessionMemory:
    """Manages cross-session memory facts across distinct agent sessions.

    Key capabilities:
    1. Structured, typed facts (subject, predicate, object_value) carrying in-band P6 provenance.
    2. Automatic conflict detection and supersession (conflicts_with).
    3. Explicit retraction tracking (retract_fact) preserving audit history.
    4. Bounded progressive disclosure prompt formatting to prevent unbounded context growth.
    5. Integration as synthesis input for SkillSynthesizer (P9) and promotion input for OntologyEngine (P7).
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        max_facts_in_prompt: int = 10,
        embedder: EmbedderProtocol | None = None,
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
        self._facts: dict[str, MemoryFact] = {}
        # Set when a store existed and could not be read. Carried rather than swallowed:
        # an empty memory and an unreadable memory are different claims, and the second
        # one reaches the model as "you have no prior facts" if nobody says otherwise.
        self._load_failure: str | None = None
        # The document we last agreed with, by digest. `save` compares the file against it
        # to tell "nobody else wrote" from "somebody did", and only the second case costs
        # a merge.
        self._witness: str | None = None
        if self._storage_path is not None and self._storage_path.is_file():
            self.load()

    @property
    def storage_path(self) -> Path | None:
        """Configured on-disk storage path, if any."""
        return self._storage_path

    @property
    def load_failure(self) -> str | None:
        """Why the on-disk store could not be read, if it could not be."""
        return self._load_failure

    @property
    def max_facts_in_prompt(self) -> int:
        """Maximum number of memory facts injected into system prompt."""
        return self._max_facts_in_prompt

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
    ) -> MemoryFact:
        """Record a new verified cross-session fact carrying P6 provenance."""
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
        )

        contradicted_id: str | None = None
        if auto_retract_conflicts:
            for existing in list(self._facts.values()):
                if existing.conflicts_with(new_fact):
                    contradicted_id = existing.fact_id
                    retracted = existing.model_copy(
                        update={
                            "retracted": True,
                            "retraction_reason": f"Superseded by fact {new_fact.fact_id}",
                            "retracted_at": utc_now_iso(),
                            "updated_at": utc_now_iso(),
                        }
                    )
                    self._facts[existing.fact_id] = retracted
                    # Same reason as the explicit retraction path: a retracted fact's
                    # vector is an index entry with no corpus row behind it, and a store
                    # that supersedes often would grow one per supersession forever.
                    self.forget_vector(existing.fact_id)
                    logger.info(
                        "Fact %s superseded and retracted by conflicting fact %s",
                        existing.fact_id,
                        new_fact.fact_id,
                    )

        if contradicted_id is not None:
            new_fact = new_fact.model_copy(update={"contradicts_fact_id": contradicted_id})

        self._facts[new_fact.fact_id] = new_fact
        self.save()
        return new_fact

    def retract_fact(
        self,
        fact_id: str,
        reason: str,
        provenance: Provenance,
        session_id: str | None = None,
    ) -> MemoryFact:
        """Explicitly retract a fact by ID, preserving the audit trail."""
        require_provenance(provenance, "retract_fact")
        if fact_id not in self._facts:
            raise KeyError(f"Memory fact '{fact_id}' not found")

        current = self._facts[fact_id]
        if current.retracted:
            return current

        clean_reason = reason.strip() or "No retraction reason provided"
        retracted = current.model_copy(
            update={
                "retracted": True,
                "retraction_reason": clean_reason,
                "retracted_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
        )
        self._facts[fact_id] = retracted
        # A retracted fact is excluded from every candidate set, so its vector is dead
        # weight in the index. Dropping it here keeps the index the same size as the
        # corpus it claims to describe.
        self.forget_vector(fact_id)
        self.save()
        logger.info("Retracted fact %s: %s", fact_id, clean_reason)
        return retracted

    def get_fact(self, fact_id: str) -> MemoryFact | None:
        """Look up a fact by ID."""
        return self._facts.get(fact_id)

    def list_facts(
        self,
        include_retracted: bool = False,
        subject: str | None = None,
        predicate: str | None = None,
        tags: Sequence[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryFact]:
        """List memory facts matching filter criteria."""
        results: list[MemoryFact] = []
        target_tags = set(tags) if tags else None
        target_subj = subject.strip().lower() if subject else None
        target_pred = predicate.strip().lower() if predicate else None

        for fact in self._facts.values():
            if not include_retracted and fact.retracted:
                continue
            if fact.confidence < min_confidence:
                continue
            if target_subj and fact.subject.strip().lower() != target_subj:
                continue
            if target_pred and fact.predicate.strip().lower() != target_pred:
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
        return [f for f in self._facts.values() if f.conflicts_with(fact)]

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

        active_facts.sort(key=lambda f: (f.confidence, f.created_at), reverse=True)

        limit = max_facts if max_facts is not None else self._max_facts_in_prompt
        selected = active_facts[:limit]

        lines = [
            "[Cross-Session Memory Facts]",
            "The following verified durable facts from prior sessions are active:",
        ]
        for f in selected:
            lines.append(
                f"- {f.summary()} (session: {f.source_session_id}, confidence: {f.confidence:.2f})"
            )

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

    # ----------------------------------------------------------------------
    # P7 Evolving Ontology Grounding & Promotion Adapter
    # ----------------------------------------------------------------------
    def export_ontology_evidence(
        self,
        subject: str,
        predicate: str,
    ) -> EvidenceRecord | None:
        """Aggregate cross-session memory facts into an EvidenceRecord for ontology induction (P7)."""
        norm_subj = subject.strip().lower()
        norm_pred = predicate.strip().lower()

        matching_facts = [
            f
            for f in self._facts.values()
            if f.subject.strip().lower() == norm_subj and f.predicate.strip().lower() == norm_pred
        ]
        if not matching_facts:
            return None

        active_facts = [f for f in matching_facts if not f.retracted]
        retracted_facts = [f for f in matching_facts if f.retracted]

        if not active_facts:
            return None

        sessions = tuple(sorted({f.source_session_id for f in active_facts}))
        contradictions = tuple(
            sorted(
                {
                    f"{f.fact_id}: {f.object_value} (retracted: {f.retraction_reason or 'retracted'})"
                    for f in retracted_facts
                }
            )
        )
        avg_conf = sum(f.confidence for f in active_facts) / len(active_facts)

        return EvidenceRecord(
            observation_count=len(active_facts),
            first_seen=min(f.created_at for f in active_facts),
            last_seen=max(f.created_at for f in active_facts),
            source_session=sessions[0] if sessions else None,
            originating_sessions=sessions,
            contradicting_observations=contradictions,
            confidence=avg_conf,
        )

    def promote_fact_to_ontology_concept(
        self,
        ontology: OntologyEngine,
        fact_id: str,
        approver: str = "human:developer",
        force: bool = False,
        min_observations: int = 2,
        min_distinct_sessions: int = 2,
    ) -> OntologyConcept:
        """Promote a memory fact into an induced or enforcing concept in OntologyEngine (P7)."""
        fact = self.get_fact(fact_id)
        if fact is None:
            raise KeyError(f"Memory fact '{fact_id}' not found")
        if fact.retracted:
            raise ValueError(f"Cannot promote retracted fact '{fact_id}' to ontology")

        evidence = self.export_ontology_evidence(fact.subject, fact.predicate)
        if evidence is None:
            raise ValueError(f"No valid observational evidence for fact '{fact_id}'")

        from uclone_x.ontology.models import OntologyConcept

        existing = ontology.get_concept(fact.subject)
        if existing is None:
            concept = ontology.induce_concept(
                name=fact.subject,
                attributes={fact.predicate: fact.object_value},
                source_session=fact.source_session_id,
                confidence=fact.confidence,
            )
            updated_concept = concept.model_copy(update={"evidence": evidence})
            ontology._concepts[fact.subject] = updated_concept  # pyright: ignore[reportPrivateUsage]

        promoted = ontology.promote(
            name=fact.subject,
            approver=approver,
            force=force,
            min_observations=min_observations,
            min_distinct_sessions=min_distinct_sessions,
        )
        if not isinstance(promoted, OntologyConcept):
            raise TypeError(f"Expected promoted concept, got {type(promoted)}")
        return promoted

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------
    def save(self) -> None:
        """Persist this agent's facts, keeping whatever another writer recorded meanwhile.

        The document is written whole, and until #1125 it was written from this object's
        facts alone. One agent is reachable through several processes at once -- the
        dashboard, `ucx run`, an A2A or ACP shell -- and each loads the document when it
        starts. Every fact the others recorded after that load was erased by the next
        `os.replace`, with nothing raised and nothing logged: the store simply had fewer
        facts than it was told, which is the substitution P6 forbids.

        So the file is re-read here and compared against the document we last agreed with.
        Unchanged, this costs one read. Changed, the writes are folded together by
        `fact_id` before serializing -- facts are immutable and their ids are unique, so a
        union is well defined, and `_supersedes` settles an id both writers hold. The
        merged set becomes this object's own view too, otherwise the next save would
        re-erase what this one just preserved.

        **This is a merge, not a lock**, for the reason `SessionStore.save` gives for its
        revision precondition (#240): `fcntl.flock` has no Windows equivalent and would
        ship a branch that cannot be exercised where verification runs, and an `O_EXCL`
        lockfile turns a crash into an agent that can never write again. The window left
        open is the one between this read and the `os.replace` below -- microseconds, and
        bounded by it -- against a window that previously spanned a whole session.
        `test_the_merge_is_a_check_not_a_lock` pins that limit so this docstring cannot
        drift into claiming serialisability.
        """
        if self._storage_path is None:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._absorb_concurrent_writes()
        dumped = {
            "version": _DOCUMENT_VERSION,
            "facts": [fact.model_dump(mode="json") for fact in self._facts.values()],
        }
        serialized = json.dumps(dumped, indent=2)

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._storage_path.parent,
            delete=False,
            encoding="utf-8",
        )
        try:
            temp_file.write(serialized)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            os.replace(temp_file.name, self._storage_path)
            self._witness = _document_digest(serialized)
        except BaseException:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise

    def _absorb_concurrent_writes(self) -> None:
        """Fold in whatever another process wrote since we last agreed with the document.

        A document we cannot read is quarantined and why is recorded in `_load_failure`.
        That deliberately does not stop the write: refusing here would discard this
        session's own facts to protect a file that has already been set aside.

        It quarantines directly rather than calling `load`, and the read is inside the
        `try` with the parse. `load`'s *success* path assigns `self._facts` outright, so
        a document that failed the first read and parsed on `load`'s second -- another
        process having replaced it in between -- silently dropped every fact this object
        held and was about to write, which is the P6 substitution this method exists to
        remove. And a file that cannot be decoded or opened at all (a non-UTF-8 document,
        EACCES) is exactly as unreadable as one that will not parse: leaving `read_text`
        outside the guard sent `UnicodeDecodeError` out through `record_fact`, quarantining
        nothing and failing every write from then on.
        """
        if self._storage_path is None:
            return

        try:
            if not self._storage_path.is_file():
                return
            content = self._storage_path.read_text(encoding="utf-8")
            if _document_digest(content) == self._witness:
                return
            theirs = _parse_facts(content)
        except Exception as unreadable:
            self._quarantine_unreadable(unreadable)
            return

        for fact_id, their_fact in theirs.items():
            held = self._facts.get(fact_id)
            if held is None or _supersedes(their_fact, held):
                self._facts[fact_id] = their_fact
                if their_fact.retracted:
                    # Same invariant `retract_fact` keeps: the index describes the
                    # corpus, so a fact withdrawn elsewhere leaves the index here too.
                    self.forget_vector(fact_id)

    def _quarantine_unreadable(self, exc: Exception) -> None:
        """Move a store we could not read aside, and record why.

        It is moved rather than left in place: the next `save()` would otherwise
        overwrite the one copy of those facts with whatever this session happens to
        hold, and the loss would be unrecoverable. `_load_failure` makes the gap
        visible in the prompt instead of letting it read as "no prior facts" (P6).

        It touches nothing this object holds. Both callers reach it with facts they
        still intend to write.
        """
        if self._storage_path is None:
            return

        quarantine = self._storage_path.with_suffix(
            self._storage_path.suffix + f".unreadable-{int(time.time())}"
        )
        try:
            os.replace(self._storage_path, quarantine)
            moved_to: str | None = str(quarantine)
        except OSError as move_exc:
            moved_to = None
            logger.error(
                "Could not quarantine unreadable memory store %s: %s",
                self._storage_path,
                move_exc,
            )
        # No document is agreed with any more, so the next save re-reads whatever takes
        # this path's place rather than trusting a digest of the file that was moved aside.
        self._witness = None
        self._load_failure = f"the store at {self._storage_path} could not be read ({exc})" + (
            f"; it was moved to {moved_to}" if moved_to else ""
        )
        logger.error("Failed to load cross-session memory: %s", self._load_failure)

    def load(self) -> None:
        """Load memory facts from storage_path."""
        if self._storage_path is None or not self._storage_path.is_file():
            return

        try:
            content = self._storage_path.read_text(encoding="utf-8")
            self._facts = _parse_facts(content)
            self._witness = _document_digest(content)
            self._load_failure = None
        except Exception as exc:
            self._quarantine_unreadable(exc)
