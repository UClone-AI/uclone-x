"""The dashboard's sessions: one `LinkSession` per link that should be online.

The dashboard's lifespan calls `start()` on the way up and `shutdown()` on the way down;
`ucx link run` does the same without a dashboard. With no link record there is no session
and nothing is dialled.

One runtime per links file (`runtime_lock.py`): the first supervisor with a link to dial
holds the lock until `shutdown()`. Another finds it held, dials nothing, and tries again
every `claim_retry_s`, so a dashboard opened while `ucx link run` serves the links takes
them over once that stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable

from uclone_x.link.uclone2.client import LinkError, LinkFailure, Uclone2LinkClient
from uclone_x.link.uclone2.models import LinkRecord, RemoteClone
from uclone_x.link.uclone2.runtime_lock import RuntimeLock, RuntimeRole, runtime_lock_path
from uclone_x.link.uclone2.service import (
    UnlinkOutcome,
    forget_local,
    link_uclone2,
    retry_pending_unlinks,
    unlink,
)
from uclone_x.link.uclone2.session import TERMINAL_STATES, LinkSession, LinkSessionState
from uclone_x.link.uclone2.store import LinkStore, LinkStoreError

__all__ = ["CLAIM_RETRY_S", "LinkSupervisor", "SessionFactory", "StateListener"]

logger = logging.getLogger(__name__)

SessionFactory = Callable[[LinkRecord, LinkStore, Uclone2LinkClient], LinkSession]
#: Told each state change of each session, with the record the session runs.
StateListener = Callable[[LinkRecord, LinkSessionState], None]

#: How often a supervisor that found the links held elsewhere tries to take them.
CLAIM_RETRY_S = 15.0


def _default_factory(
    record: LinkRecord, store: LinkStore, client: Uclone2LinkClient
) -> LinkSession:
    return LinkSession(record, store=store, client=client)


def _should_run(record: LinkRecord) -> bool:
    return record.enabled and not record.paused and not record.unlink_pending


class LinkSupervisor:
    """Starts, pauses and stops the sessions of every stored link."""

    def __init__(
        self,
        *,
        store: LinkStore | None = None,
        client: Uclone2LinkClient | None = None,
        session_factory: SessionFactory = _default_factory,
        role: RuntimeRole = RuntimeRole.DASHBOARD,
        on_state: StateListener | None = None,
        claim_retry_s: float = CLAIM_RETRY_S,
    ) -> None:
        # Resolved at `start()`, not here: the links folder is read from the environment
        # when the dashboard comes up, not when its app object is built.
        self._given_store = store
        self._store = store if store is not None else LinkStore()
        self._client = client if client is not None else Uclone2LinkClient()
        self._factory = session_factory
        self._sessions: dict[str, LinkSession] = {}
        self._retry_task: asyncio.Task[None] | None = None
        self._retried: list[LinkRecord] = []
        self._retry_started = False
        self._role = role
        self._on_state = on_state
        self._claim_retry_s = claim_retry_s
        self._lock: RuntimeLock | None = None
        self._claim_task: asyncio.Task[None] | None = None

    def _records(self) -> list[LinkRecord]:
        try:
            return self._store.records()
        except LinkStoreError as err:
            # The dashboard still comes up; the section reads the store and says the same.
            logger.warning("uClone2 links not started: %s", err)
            return []

    async def start(self) -> None:
        """Retry *해제 대기* unlinks in the background and start every link that should run."""
        await self.start_unlink_retry()
        for record in self._records():
            if _should_run(record):
                self._start(record)

    async def start_unlink_retry(self) -> None:
        """Retry *해제 대기* unlinks in the background; once per supervisor.

        `start()` begins with it. `ucx link run` calls it first and waits on
        `unlinks_retried()`, so it can say what finished before it dials anything.
        """
        if self._retry_started:
            return
        self._retry_started = True
        if self._given_store is None:
            self._store = LinkStore()
        if any(r.unlink_pending for r in self._records()):
            self._retry_task = asyncio.create_task(self._retry_unlinks())

    async def unlinks_retried(self) -> list[LinkRecord]:
        """Wait for `start()`'s retry of *해제 대기* unlinks; the records it finished."""
        if self._retry_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(self._retry_task)
        return list(self._retried)

    def runnable(self) -> list[LinkRecord]:
        """The stored links `start()` dials: enabled, not paused, not pending unlink."""
        return [r for r in self._records() if _should_run(r)]

    # --- one runtime per links file ----------------------------------------------------

    def claim(self) -> bool:
        """Take the links for this process; `False` when another runtime holds them."""
        try:
            return self._runtime_lock().acquire()
        except OSError as err:
            # A lock that cannot even be opened (a read-only folder) is no evidence of a
            # second runtime: dial, as before the lock existed.
            logger.warning("uClone2 runtime lock not taken: %s", type(err).__name__)
            return True

    @property
    def _waiting_to_claim(self) -> bool:
        return self._claim_task is not None and not self._claim_task.done()

    @property
    def held_elsewhere(self) -> bool:
        """Another runtime on this machine holds the links, as the lock says now.

        Asked of the lock on every read, not only of this head's retry: a dashboard opened
        with no links has nothing to retry, and must still see the `ucx link run` started
        after it.
        """
        if self._waiting_to_claim:
            return True
        if self._lock is not None and self._lock.held:
            return False
        try:
            return self._runtime_lock().held_by_another()
        except OSError:
            return False

    def holder(self) -> RuntimeRole | None:
        """Who holds the links, as the lock file says; `None` when unknown."""
        return self._runtime_lock().holder()

    def _runtime_lock(self) -> RuntimeLock:
        if self._lock is None:
            self._lock = RuntimeLock(runtime_lock_path(self._store.path), self._role)
        return self._lock

    async def _claim_later(self) -> None:
        while True:
            await asyncio.sleep(self._claim_retry_s)
            if self.claim():
                logger.info("uClone2 links taken over from another runtime on this machine")
                break
        for record in self.runnable():
            self._start(record)

    def _start(self, record: LinkRecord) -> None:
        if record.link_id in self._sessions:
            return
        if not self.claim():  # another runtime has the links: dial nothing
            if not self._waiting_to_claim:
                logger.info("uClone2 links are held by another runtime on this machine")
                self._claim_task = asyncio.create_task(self._claim_later())
            return
        session = self._factory(record, self._store, self._client)
        self._sessions[record.link_id] = session
        if self._on_state is not None:
            listener = self._on_state
            session.listen(lambda state: listener(record, state))
        session.start()

    async def _retry_unlinks(self) -> None:
        try:
            done = await retry_pending_unlinks(store=self._store, client=self._client)
        except (LinkError, LinkStoreError) as err:
            logger.info("uClone2 pending unlinks not retried: %s", type(err).__name__)
            return
        self._retried = list(done)
        for record in done:
            logger.info("uClone2 link %s unlinked on retry", record.link_id)

    @property
    def store(self) -> LinkStore:
        """The store the sessions run from; the Settings routes read the records here."""
        return self._store

    async def link(
        self,
        pasted: str,
        *,
        choose_local: Callable[[RemoteClone], str],
        server_url: str | None = None,
    ) -> LinkRecord:
        """Link from a pasted URL or code, and start the new link's session at once.

        A link made here does not wait for the next dashboard start (§3.2 step 5). The
        record it replaces -- the same uClone2 clone linked before -- holds a token the
        exchange just revoked, so that record's session is stopped without a logout.

        Refused before the code is spent while another runtime on this machine holds the
        links: it would not dial the new link until it restarts, and this one could not.
        """
        if not self.claim():
            raise LinkError(LinkFailure.ELSEWHERE)
        try:
            record = await link_uclone2(
                pasted,
                choose_local=choose_local,
                store=self._store,
                client=self._client,
                server_url=server_url,
            )
        except BaseException:
            if not self._sessions and self._lock is not None:
                # Nothing is dialled here, so `ucx link run` may have the links after all.
                self._lock.release()
            raise
        kept = {r.link_id for r in self._records()}
        for link_id in [i for i in self._sessions if i not in kept]:
            await self._drop(link_id, logout=False)
        if _should_run(record):
            self._start(record)
        return record

    async def unlink(self, link_id: str) -> UnlinkOutcome:
        """Unlink in uClone2, then stop the session.

        Once uClone2 has answered it revokes the token and closes the socket itself, so no
        `bye{logout}` is sent. When it could not be reached the record is kept as
        *해제 대기* and the session is logged out, so the clone is not left answering.
        """
        outcome = await unlink(link_id, store=self._store, client=self._client)
        await self._drop(link_id, logout=outcome is UnlinkOutcome.PENDING)
        return outcome

    async def forget_local(self, link_id: str) -> LinkRecord:
        """Remove a *해제 대기* record from this machine only (see `service.forget_local`)."""
        record = forget_local(link_id, store=self._store)
        await self._drop(link_id, logout=False)
        return record

    async def _drop(self, link_id: str, *, logout: bool) -> None:
        session = self._sessions.pop(link_id, None)
        if session is not None:
            await session.stop(logout=logout, final=LinkSessionState.OFFLINE)

    def session(self, link_id: str) -> LinkSession | None:
        return self._sessions.get(link_id)

    def states(self) -> dict[str, LinkSessionState]:
        """Each started link's state; a link with no session is not in the map."""
        return {link_id: s.state for link_id, s in self._sessions.items()}

    async def set_online(self, link_id: str, online: bool) -> LinkRecord:
        """The user's *온라인으로 전환* / *오프라인으로 전환*, kept across restarts.

        Going offline sends `bye{logout}`, so uClone2 shows the clone offline at once.
        """
        if self.held_elsewhere:
            # The runtime that holds the links would not see the change until it restarts.
            raise LinkError(LinkFailure.ELSEWHERE)
        record = self._store.update(link_id, paused=not online)
        if record is None:
            raise LinkError(LinkFailure.NOT_FOUND)
        session = self._sessions.get(link_id)
        if not online:
            if session is not None:
                del self._sessions[link_id]
                await session.stop(logout=True, final=LinkSessionState.PAUSED)
        else:
            if not _should_run(record):
                # A link the server ended stays disabled: its session is kept, so the state
                # the user needs to read (for one, 연결이 끝났습니다) stays in `states()`.
                return record
            if session is not None and session.state in TERMINAL_STATES:
                # Stopped earlier (by the user, or by the server for a reason a retry may
                # now get past, such as an update): switching on dials afresh.
                del self._sessions[link_id]
                await session.stop(logout=False, final=session.state)
            self._start(record)
        return record

    async def shutdown(self) -> None:
        """Log every session out, concurrently, each bounded by its own logout wait."""
        if self._retry_task is not None and not self._retry_task.done():
            self._retry_task.cancel()
            with contextlib.suppress(BaseException):
                await self._retry_task
        if self._claim_task is not None and not self._claim_task.done():
            self._claim_task.cancel()
            with contextlib.suppress(BaseException):
                await self._claim_task
        # A session the server stopped keeps its state, but its background work (a profile
        # refresh still in flight) is cancelled all the same: `stop` without a logout does it.
        await asyncio.gather(
            *(
                s.stop(logout=False, final=s.state)
                if s.state in _FINAL_ON_SERVER
                else s.stop(logout=True, final=LinkSessionState.OFFLINE)
                for s in list(self._sessions.values())
            ),
            return_exceptions=True,
        )
        # Last: another runtime may take the links only once every bye has been written.
        if self._lock is not None:
            self._lock.release()


#: Stopped by the server: shutdown leaves the state the user needs to read as it is.
_FINAL_ON_SERVER = frozenset(
    {
        LinkSessionState.ENDED,
        LinkSessionState.REPLACED,
        LinkSessionState.UPDATE_REQUIRED,
        LinkSessionState.PROTOCOL_ERROR,
    }
)
