"""What a head calls to link, refresh and unlink: the store and the client, in order.

The CLI calls these today and the dashboard's `/api/links*` routes will call the same
ones, so the rules below -- a stored link replaces the clone's earlier one, a revoked
token disables the record instead of deleting it, an unreachable unlink is kept as
*해제 대기* -- hold for every head alike.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from uclone_x.link.uclone2.client import (
    LinkError,
    LinkFailure,
    Uclone2LinkClient,
    parse_connect_input,
)
from uclone_x.link.uclone2.models import LinkRecord, RemoteClone
from uclone_x.link.uclone2.store import LinkStore, LinkStoreError

__all__ = [
    "FALLBACK_CLONE",
    "LINK_ID_PREFIX",
    "ProfileRefresh",
    "UnlinkOutcome",
    "check_local_choice",
    "forget_local",
    "local_chooser",
    "link_uclone2",
    "refresh_profile",
    "retry_pending_unlinks",
    "unlink",
]

logger = logging.getLogger(__name__)

LINK_ID_PREFIX = "lnk_"

_NOT_SAVED = (
    "uClone2와 연결했지만 이 컴퓨터에 저장하지 못해서 연결을 되돌렸습니다. "
    "저장 공간과 권한을 확인한 뒤 uClone2에서 새 코드를 받아 다시 연결하십시오"
)
_NOT_SAVED_NOT_UNDONE = (
    "uClone2와 연결했지만 이 컴퓨터에 저장하지 못했습니다. "
    "uClone2의 클론 페이지에서 연결을 해제한 뒤 새 코드로 다시 연결하십시오"
)


#: The clone that answers when none is named and none shares the uClone2 clone's name.
FALLBACK_CLONE = "clone"


def check_local_choice(named: str | None, known: list[str]) -> None:
    """Refuse, before the exchange, a choice that cannot be met: the code is single use.

    A named clone must exist here, and with none named there must be some clone at all.
    Whether one shares the uClone2 clone's name is known only after the exchange.
    """
    if (named is not None and named not in known) or not known:
        raise LinkError(LinkFailure.NO_LOCAL_CLONE)


def local_chooser(named: str | None, known: list[str]) -> Callable[[RemoteClone], str]:
    """The local clone for a uClone2 clone: the named one, else one with the same name
    (case-insensitively), else the default `clone` -- but only a clone that exists here.

    With none of those, `NO_MATCHING_CLONE`: `link_uclone2` then undoes the exchange, so
    the uClone2 clone is not left linked to a name nothing on this machine answers to.
    """

    def choose(remote: RemoteClone) -> str:
        if named is not None:
            if named in known:
                return named
            raise LinkError(LinkFailure.NO_LOCAL_CLONE)
        by_folded = {name.casefold(): name for name in known}
        for candidate in (remote.username, remote.display_name):
            if candidate.casefold() in by_folded:
                return by_folded[candidate.casefold()]
        if FALLBACK_CLONE in known:
            return FALLBACK_CLONE
        raise LinkError(LinkFailure.NO_MATCHING_CLONE)

    return choose


def _now() -> datetime:
    return datetime.now(UTC)


def _new_link_id(taken: set[str]) -> str:
    while True:
        candidate = f"{LINK_ID_PREFIX}{secrets.token_hex(4)}"
        if candidate not in taken:
            return candidate


async def link_uclone2(
    pasted: str,
    *,
    choose_local: Callable[[RemoteClone], str],
    store: LinkStore,
    client: Uclone2LinkClient,
    server_url: str | None = None,
) -> LinkRecord:
    """Exchange a pasted connect URL or code and store the link it creates.

    The pasted text is checked before anything is sent, so a typo costs no code.
    `choose_local` picks the local clone once the uClone2 clone is known; the head asks
    the user, or defaults by name.

    The code is single use, so once the exchange succeeds a failure to save is not left
    as a dangling link: the token is used once more to unlink, and the user is told to
    start again. If even that fails, they are told to unlink in uClone2.
    """
    parsed = parse_connect_input(pasted, server_url)
    result = await client.connect(parsed.server_url, parsed.code)
    now = _now()
    try:
        taken = {r.link_id for r in store.records()}
        record = LinkRecord(
            link_id=_new_link_id(taken),
            server_url=parsed.server_url,
            bot_id=result.bot_id,
            remote_username=result.clone.username,
            remote_display_name=result.clone.display_name,
            remote_avatar_url=result.clone.avatar_url,
            local_agent_id=choose_local(result.clone),
            token=result.token,
            ws_url=result.ws_url,
            created_at=now,
        )
        store.put(record)
    except BaseException as err:
        # Whatever stopped the save -- the disk, the store, or `choose_local` itself -- the
        # server side of the link exists and nothing here remembers its token. Undo it.
        logger.warning("uClone2 link for bot %s could not be saved; undoing it", result.bot_id)
        try:
            await client.unlink(parsed.server_url, result.token)
        except LinkError:
            undone = False
        else:
            undone = True
        if isinstance(err, (OSError, LinkStoreError)) or (
            isinstance(err, LinkError) and not undone
        ):
            # A refusal from `choose_local` says the link was undone; when it was not, the
            # user has to end it in uClone2, which is what this sentence tells them.
            raise (
                LinkStoreError(_NOT_SAVED, "not_saved_undone")
                if undone
                else LinkStoreError(_NOT_SAVED_NOT_UNDONE, "not_saved_not_undone")
            ) from None
        raise
    logger.info("Linked local clone %s to uClone2 bot %s", record.local_agent_id, record.bot_id)
    return record


@dataclass(frozen=True)
class ProfileRefresh:
    """The record after a `GET /linked/self`, and what the head should say about it."""

    record: LinkRecord
    #: The owner changed the persona or avatar in uClone2 since the last read. The local
    #: persona is left alone -- after the link it is the user's own -- and the head offers
    #: to re-seed it instead.
    profile_changed: bool
    pending: int
    online: bool


def _require(store: LinkStore, link_id: str) -> LinkRecord:
    record = store.get(link_id)
    if record is None:
        raise LinkError(LinkFailure.NOT_FOUND)
    return record


async def refresh_profile(
    link_id: str, *, store: LinkStore, client: Uclone2LinkClient
) -> ProfileRefresh:
    """Read the link's live profile and keep the card's copy of it current.

    A revoked token (401) marks the record disabled and re-raises, so the head says the
    link is over; the record stays until the user removes it.
    """
    record = _require(store, link_id)
    try:
        current = await client.get_self(record.server_url, record.token)
    except LinkError as err:
        if err.failure is LinkFailure.TOKEN_INVALID:
            store.replace(record.model_copy(update={"enabled": False}))
            logger.info("uClone2 link %s was ended on the server; marked disabled", link_id)
        raise
    changed = (
        record.clone_updated_at is not None and current.clone_updated_at > record.clone_updated_at
    )
    updated = record.model_copy(
        update={
            "remote_username": current.clone.username,
            "remote_display_name": current.clone.display_name,
            "remote_avatar_url": current.clone.avatar_url,
            "clone_updated_at": current.clone_updated_at,
        }
    )
    store.replace(updated)
    return ProfileRefresh(
        record=updated, profile_changed=changed, pending=current.pending, online=current.online
    )


class UnlinkOutcome(StrEnum):
    #: uClone2 confirmed (or the link was already gone there) and the record is deleted.
    REMOVED = "removed"
    #: uClone2 could not be reached: the record is kept as *해제 대기* and retried later.
    PENDING = "pending"


async def unlink(link_id: str, *, store: LinkStore, client: Uclone2LinkClient) -> UnlinkOutcome:
    """Unlink with the link's own token, then delete the record.

    The record is deleted only once uClone2 has answered. Deleting it on an unreachable
    server would leave the clone linked and offline there, with paid obligations piling
    up until they expire -- and with nothing left on this machine that could end it.
    """
    record = _require(store, link_id)
    try:
        await client.unlink(record.server_url, record.token)
    except LinkError as err:
        logger.info("uClone2 unlink of %s deferred (%s)", link_id, err.diagnostic or err.failure)
        store.replace(record.model_copy(update={"unlink_pending": True}))
        return UnlinkOutcome.PENDING
    store.remove(link_id)
    logger.info("Unlinked uClone2 link %s", link_id)
    return UnlinkOutcome.REMOVED


def forget_local(link_id: str, *, store: LinkStore) -> LinkRecord:
    """Delete a *해제 대기* record from this machine only, without asking uClone2.

    For a record whose unlink uClone2 keeps refusing: the retry would never succeed and
    nothing else could remove it. The link may still exist in uClone2 -- the head says so
    -- and the owner ends it there. Any other record is refused (`NOT_PENDING`): an
    unlink that has not been tried yet must go through uClone2 first.
    """
    record = _require(store, link_id)
    if not record.unlink_pending:
        raise LinkError(LinkFailure.NOT_PENDING)
    store.remove(record.link_id)
    logger.info("uClone2 link %s removed from this machine only", link_id)
    return record


async def retry_pending_unlinks(*, store: LinkStore, client: Uclone2LinkClient) -> list[LinkRecord]:
    """Retry every *해제 대기* unlink; the records that are now gone. Called on start."""
    done: list[LinkRecord] = []
    for record in store.records():
        if record.unlink_pending and (
            await unlink(record.link_id, store=store, client=client) is UnlinkOutcome.REMOVED
        ):
            done.append(record)
    return done
