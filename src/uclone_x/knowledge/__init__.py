"""uGraph: one fact model and one entity resolver for clone memory and story codex.

See the uGraph design document. The model, the shared fold, the resolver and the two views
of a point (`edges_at`: approved, or approved and proposed) are pure. `SqliteKnowledgeStore`
and `ClonePolicy` are the clone vessel (step 3). The story vessel reads its YAML codex into
this model in the story package (step 2), since the codex files are the story's own.
"""

from __future__ import annotations

from uclone_x.knowledge.fold import bigram_overlap, bigrams, fold
from uclone_x.knowledge.models import (
    Edge,
    Entity,
    EntityRef,
    Episode,
    Interval,
)
from uclone_x.knowledge.policy import CLONE_POLICY, ClonePolicy
from uclone_x.knowledge.resolver import EntityResolver
from uclone_x.knowledge.sqlite_store import (
    KnowledgeTransaction,
    KnowledgeUnreadableError,
    ScopeMismatchError,
    SqliteKnowledgeStore,
)
from uclone_x.knowledge.store import (
    APPROVED,
    APPROVED_OR_PROPOSED,
    EdgeRow,
    KnowledgeStore,
    edges_at,
    facts_at,
    holds_at,
)

__all__ = [
    "APPROVED",
    "APPROVED_OR_PROPOSED",
    "CLONE_POLICY",
    "ClonePolicy",
    "Edge",
    "EdgeRow",
    "Entity",
    "EntityRef",
    "EntityResolver",
    "Episode",
    "Interval",
    "KnowledgeStore",
    "KnowledgeTransaction",
    "KnowledgeUnreadableError",
    "ScopeMismatchError",
    "SqliteKnowledgeStore",
    "bigram_overlap",
    "bigrams",
    "edges_at",
    "facts_at",
    "fold",
    "holds_at",
]
