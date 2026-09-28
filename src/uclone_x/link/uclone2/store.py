"""`LinkStore`: every uClone2 link on this machine, in one owner-only JSON file.

The file holds credentials, so it is written the way `tools/mcp_manager.py` writes
`mcp_servers.json`: created `0600` before a byte of it exists, replaced atomically so a
crash mid-write cannot lose every link, and under a sidecar lock so the dashboard and the
CLI cannot each read the old file and have the later write drop the earlier one's link.

This is the one place the raw token is serialised. Views (`LinkRecord.view`) never carry it.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from uclone_x.link.uclone2.models import LinkRecord, LinkView

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None

__all__ = [
    "LINKS_DIR_ENV_VAR",
    "LinkStore",
    "LinkStoreError",
    "default_store_path",
]

#: Redirects the links directory, as `UCLONE_AGENTS_DIR` redirects agent homes.
LINKS_DIR_ENV_VAR = "UCLONE_LINKS_DIR"

_DEFAULT_LINKS_DIR = Path.home() / ".uclone" / "links"
_FILE_NAME = "uclone2.json"
_FORMAT_VERSION = 1

_UNREADABLE = "저장된 uClone2 연결 정보를 읽을 수 없습니다. 파일이 손상되었을 수 있습니다"


class LinkStoreError(Exception):
    """The links file exists and cannot be used. The message is a plain sentence.

    `kind` names the case for a head that words it in its own language: `unreadable`
    here, and `not_saved_undone` / `not_saved_not_undone` from `service.link_uclone2`.
    """

    def __init__(self, message: str, kind: str = "unreadable") -> None:
        super().__init__(message)
        self.kind = kind


def default_store_path() -> Path:
    """`~/.uclone/links/uclone2.json`, or the same file under `UCLONE_LINKS_DIR`."""
    override = os.environ.get(LINKS_DIR_ENV_VAR)
    root = Path(override).expanduser() if override else _DEFAULT_LINKS_DIR
    return root / _FILE_NAME


class LinkStore:
    """Load, change and write back the links file, one whole-file change at a time."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else default_store_path()

    @property
    def path(self) -> Path:
        return self._path

    def records(self) -> list[LinkRecord]:
        """Every stored link, oldest first. A missing file is no links."""
        return self._read()

    def views(self) -> list[LinkView]:
        """Every stored link with its token masked."""
        return [record.view() for record in self._read()]

    def get(self, link_id: str) -> LinkRecord | None:
        return next((r for r in self._read() if r.link_id == link_id), None)

    def put(self, record: LinkRecord) -> None:
        """Add `record`, replacing any record with its `link_id` or its `bot_id`.

        One uClone2 clone has one live token: linking it again revokes the earlier one on
        the server, so an earlier record for the same `bot_id` holds a dead credential.
        """

        def change(records: list[LinkRecord]) -> list[LinkRecord]:
            kept = [r for r in records if r.link_id != record.link_id and r.bot_id != record.bot_id]
            return [*kept, record]

        self._update(change)

    def replace(self, record: LinkRecord) -> bool:
        """Overwrite the record with `record.link_id`; `False` when there is none."""
        found = False

        def change(records: list[LinkRecord]) -> list[LinkRecord]:
            nonlocal found
            out: list[LinkRecord] = []
            for r in records:
                if r.link_id == record.link_id:
                    found = True
                    out.append(record)
                else:
                    out.append(r)
            return out

        self._update(change)
        return found

    def update(self, link_id: str, **changes: object) -> LinkRecord | None:
        """Change fields of the record with `link_id` in one locked read-change-write.

        Reads the record inside the lock, so a change made meanwhile by another writer (the
        CLI, or another session) is kept rather than overwritten from a stale copy. `None`
        when there is no such record.
        """
        updated: LinkRecord | None = None

        def change(records: list[LinkRecord]) -> list[LinkRecord]:
            nonlocal updated
            out: list[LinkRecord] = []
            for r in records:
                if r.link_id == link_id:
                    updated = r.model_copy(update=changes)
                    out.append(updated)
                else:
                    out.append(r)
            return out

        self._update(change)
        return updated

    def remove(self, link_id: str) -> bool:
        """Delete the record with `link_id`; `False` when there is none."""
        found = False

        def change(records: list[LinkRecord]) -> list[LinkRecord]:
            nonlocal found
            kept = [r for r in records if r.link_id != link_id]
            found = len(kept) != len(records)
            return kept

        self._update(change)
        return found

    # --- file ---------------------------------------------------------------

    def _read(self) -> list[LinkRecord]:
        if not self._path.exists():
            return []
        try:
            data = cast(object, json.loads(self._path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            raise LinkStoreError(_UNREADABLE) from None
        raw_links = cast(dict[str, object], data).get("links") if isinstance(data, dict) else None
        if not isinstance(raw_links, list):
            raise LinkStoreError(_UNREADABLE)
        try:
            return [LinkRecord.model_validate(item) for item in cast(list[object], raw_links)]
        except ValidationError:
            # `from None`: a validation error quotes the input, and the input holds tokens.
            raise LinkStoreError(_UNREADABLE) from None

    def _update(self, change: Callable[[list[LinkRecord]], list[LinkRecord]]) -> None:
        with self._locked():
            self._write(change(self._read()))

    def _write(self, records: list[LinkRecord]) -> None:
        payload = {
            "version": _FORMAT_VERSION,
            "links": [
                # The one place the token is written out in the clear: the store file.
                {**r.model_dump(mode="json"), "token": r.token.get_secret_value()}
                for r in records
            ],
        }
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".uclone2-links.")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                # Without this the rename can reach disk before the bytes do, and a crash
                # leaves an empty file where the token was (as `room/store.py` does).
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @contextmanager
    def _locked(self) -> Generator[None]:
        """Hold a sidecar lock for one read-change-replace (the file itself is replaced)."""
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if fcntl is None:  # pragma: no cover - not POSIX
            yield
            return
        lock = self._path.parent / f".{self._path.name}.lock"
        fd = os.open(lock, os.O_RDONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)
