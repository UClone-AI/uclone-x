"""Cross-session memory subsystem for UClone-X (Issue #476).

Provides structured, typed facts carrying P6 in-band provenance,
explicit retraction and contradiction handling, bounded progressive
disclosure prompt injection, and synthesis / ontology promotion inputs.
"""

from __future__ import annotations

from uclone_x.memory.models import MemoryFact, RetractionRecord
from uclone_x.memory.retrieval import FactRanking, RankedFact, rank_facts
from uclone_x.memory.store import (
    CrossSessionMemory,
    default_cross_session_memory,
)
from uclone_x.memory.tools import (
    QueryMemoryFactsParams,
    QueryMemoryFactsTool,
    ReadOnlyMemory,
    RecordMemoryFactParams,
    RecordMemoryFactTool,
    RetractMemoryFactParams,
    RetractMemoryFactTool,
)
from uclone_x.memory.vector_store import (
    BruteForceVectorStore,
    VectorMatch,
    VectorStoreProtocol,
    deserialize_vector,
    serialize_vector,
)

__all__ = [
    "BruteForceVectorStore",
    "CrossSessionMemory",
    "FactRanking",
    "MemoryFact",
    "QueryMemoryFactsParams",
    "QueryMemoryFactsTool",
    "RankedFact",
    "ReadOnlyMemory",
    "RecordMemoryFactParams",
    "RecordMemoryFactTool",
    "RetractMemoryFactParams",
    "RetractMemoryFactTool",
    "RetractionRecord",
    "VectorMatch",
    "VectorStoreProtocol",
    "default_cross_session_memory",
    "deserialize_vector",
    "rank_facts",
    "serialize_vector",
]
