"""Put a `MemoryFact` with a chosen id and time into a `CrossSessionMemory`.

`record_fact` stamps a fact with the time it is called, which a test about ordering by age
cannot use. These write the fact the way the import does: as an edge of the clone's
knowledge store, through the same resolver, in one transaction.
"""

from __future__ import annotations

from uclone_x.memory.graph import write_fact
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.store import CrossSessionMemory


def seed_fact(memory: CrossSessionMemory, fact: MemoryFact) -> MemoryFact:
    """Write `fact` as it is, without superseding anything; return it."""
    with memory.knowledge.transaction() as tx:
        write_fact(tx, fact)
    return fact
