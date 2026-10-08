"""The one-time import of a clone's `memory.json` into its knowledge file (uGraph §5 step 3).

`memory.json` was the clone's whole-document memory until the knowledge store replaced it.
A clone that has one gets its facts copied into `knowledge.sqlite3` once, inside one
transaction that also records, in the file's `meta`, that it happened; then `memory.json`
is set aside (`.imported-<time>`), not deleted. Nothing reads that format afterwards (owner
2026-09-28: no old-format readers): the set-aside copy is for a person, and a file named
`memory.json` that turns up later is set aside again without being read.

This module is the only reader of the old format, and it reads it to import it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from uclone_x.core.set_aside import set_aside
from uclone_x.knowledge.sqlite_store import SqliteKnowledgeStore
from uclone_x.memory.graph import write_fact
from uclone_x.memory.models import MemoryFact

__all__ = ["IMPORT_MARKER", "IMPORTED_META_KEY", "import_legacy_memory", "read_legacy_facts"]

logger = logging.getLogger(__name__)

#: The `meta` key set, in the import's own transaction, to the number of facts imported.
IMPORTED_META_KEY = "legacy_memory_imported"
#: What the old file is renamed to: `memory.json.imported-<time>`.
IMPORT_MARKER = ".imported-"


def read_legacy_facts(path: Path) -> list[MemoryFact]:
    """The facts `path` holds, all or an exception: never a partial set.

    Raises:
        Exception: The file cannot be read or parsed, or a fact in it is not a `MemoryFact`.
            The caller decides whether that is damage to report or to set aside.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return [MemoryFact.model_validate_json(json.dumps(item)) for item in data.get("facts", [])]


def import_legacy_memory(store: SqliteKnowledgeStore, path: Path) -> bool:
    """Copy the facts of `path` into `store` once, then set `path` aside. Whether it copied.

    Raises:
        Exception: `path` could not be read (`read_legacy_facts`). Nothing was written and
            the file is where it was.

    The meta flag is checked and set inside the write transaction, so two processes that
    both see the old file import it once: the second waits for the first's commit, finds the
    flag, and only sets the file aside if it is still there.
    """
    with store.transaction() as tx:
        copied = tx.meta(IMPORTED_META_KEY) is None
        facts = read_legacy_facts(path) if copied else []
        if copied:
            for fact in sorted(facts, key=lambda f: f.created_at):
                write_fact(tx, fact)
            tx.set_meta(IMPORTED_META_KEY, str(len(facts)))
    try:
        aside = set_aside(path, IMPORT_MARKER)
    except FileNotFoundError:
        return copied  # another process set it aside after its own import
    if copied:
        logger.info("Imported %d facts from %s; set aside as %s", len(facts), path, aside)
    else:
        logger.warning("%s appeared after the import; set aside unread as %s", path, aside)
    return copied
