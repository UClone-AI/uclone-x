"""The link service: link, refresh the profile, unlink with the *해제 대기* rule."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.uclone2_fake import (
    BOT_ID,
    CONNECT,
    CONNECT_URL,
    DELETE_SELF,
    EDITED_CLONE,
    GET_SELF,
    ORIGIN,
    TOKEN,
    FakeUclone2,
    internals_in,
    self_body,
)
from uclone_x.link.uclone2.client import LinkError, LinkFailure, Uclone2LinkClient
from uclone_x.link.uclone2.models import LinkRecord, RemoteClone
from uclone_x.link.uclone2.service import (
    UnlinkOutcome,
    link_uclone2,
    refresh_profile,
    retry_pending_unlinks,
    unlink,
)
from uclone_x.link.uclone2.store import LinkStore, LinkStoreError


@pytest.fixture
def store(tmp_path: Path) -> LinkStore:
    return LinkStore(tmp_path / "links" / "uclone2.json")


def _local(_: RemoteClone) -> str:
    return "haru"


async def _linked(store: LinkStore, fake: FakeUclone2) -> LinkRecord:
    client = Uclone2LinkClient(transport=fake.transport)
    return await link_uclone2(CONNECT_URL, choose_local=_local, store=store, client=client)


async def test_linking_stores_the_record_for_the_chosen_local_clone(store: LinkStore) -> None:
    fake = FakeUclone2()
    record = await _linked(store, fake)

    assert record.link_id.startswith("lnk_")
    assert record.server_url == ORIGIN
    assert record.bot_id == BOT_ID
    assert record.local_agent_id == "haru"
    assert record.remote_display_name == "Haru"
    assert record.token.get_secret_value() == TOKEN
    assert store.records() == [record]


async def test_bad_input_costs_no_request(store: LinkStore) -> None:
    fake = FakeUclone2()
    with pytest.raises(LinkError):
        await link_uclone2(
            "not a code",
            choose_local=_local,
            store=store,
            client=Uclone2LinkClient(transport=fake.transport),
        )
    assert fake.requests == []
    assert store.records() == []


async def test_a_link_that_cannot_be_saved_is_undone_on_the_server(
    store: LinkStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/link/uclone2/service.py :: await client.unlink(parsed.server_url, result.token)
    Becomes: pass
    """
    fake = FakeUclone2()

    def refuse(record: LinkRecord) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "put", refuse)
    with pytest.raises(LinkStoreError) as caught:
        await _linked(store, fake)

    assert len(fake.calls(DELETE_SELF)) == 1
    assert "되돌렸습니다" in str(caught.value)
    assert internals_in(str(caught.value)) == []
    assert caught.value.__cause__ is None


async def test_a_link_whose_local_clone_cannot_be_chosen_is_undone_on_the_server(
    store: LinkStore,
) -> None:
    """Any failure after the code is spent undoes the server link, not only a disk error.

    Killed by: src/uclone_x/link/uclone2/service.py :: except BaseException as err:
    Becomes: except (OSError, LinkStoreError) as err:
    """
    fake = FakeUclone2()

    def broken(_: RemoteClone) -> str:
        raise RuntimeError("no clone")

    with pytest.raises(RuntimeError):
        await link_uclone2(
            CONNECT_URL,
            choose_local=broken,
            store=store,
            client=Uclone2LinkClient(transport=fake.transport),
        )
    assert len(fake.calls(DELETE_SELF)) == 1
    assert store.records() == []


async def test_a_link_neither_saved_nor_undone_says_to_unlink_in_uclone2(
    store: LinkStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeUclone2(unreachable={DELETE_SELF})

    def refuse(record: LinkRecord) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "put", refuse)
    with pytest.raises(LinkStoreError) as caught:
        await _linked(store, fake)
    assert "uClone2의 클론 페이지에서 연결을 해제" in str(caught.value)
    assert internals_in(str(caught.value)) == []


async def test_refresh_keeps_the_card_current_and_notices_an_edited_profile(
    store: LinkStore,
) -> None:
    fake = FakeUclone2()
    record = await _linked(store, fake)
    client = Uclone2LinkClient(transport=fake.transport)

    first = await refresh_profile(record.link_id, store=store, client=client)
    assert not first.profile_changed, "the first read has nothing to compare with"
    assert first.pending == 3

    fake.self_answer = self_body(clone=EDITED_CLONE, updated_at="2026-09-28T01:00:00Z")
    second = await refresh_profile(record.link_id, store=store, client=client)
    assert second.profile_changed
    stored = store.get(record.link_id)
    assert stored is not None
    assert stored.remote_display_name == "Haru (봄)"

    third = await refresh_profile(record.link_id, store=store, client=client)
    assert not third.profile_changed


async def test_a_revoked_token_disables_the_record_without_deleting_it(store: LinkStore) -> None:
    fake = FakeUclone2()
    record = await _linked(store, fake)
    fake.answer(GET_SELF, 401)

    with pytest.raises(LinkError) as caught:
        await refresh_profile(
            record.link_id, store=store, client=Uclone2LinkClient(transport=fake.transport)
        )
    assert caught.value.failure is LinkFailure.TOKEN_INVALID
    stored = store.get(record.link_id)
    assert stored is not None and not stored.enabled


@pytest.mark.parametrize("status", [204, 401])
async def test_unlink_removes_the_record_once_uclone2_answers(
    store: LinkStore, status: int
) -> None:
    fake = FakeUclone2()
    record = await _linked(store, fake)
    fake.answer(DELETE_SELF, status)

    outcome = await unlink(
        record.link_id, store=store, client=Uclone2LinkClient(transport=fake.transport)
    )
    assert outcome is UnlinkOutcome.REMOVED
    assert store.records() == []


async def test_an_unreachable_unlink_is_kept_pending_and_retried(store: LinkStore) -> None:
    """Killed by: src/uclone_x/link/uclone2/service.py :: store.replace(record.model_copy(update={"unlink_pending": True}))
    Becomes: store.remove(link_id)
    """
    fake = FakeUclone2()
    record = await _linked(store, fake)
    fake.unreachable.add(DELETE_SELF)
    client = Uclone2LinkClient(transport=fake.transport)

    assert await unlink(record.link_id, store=store, client=client) is UnlinkOutcome.PENDING
    stored = store.get(record.link_id)
    assert stored is not None and stored.unlink_pending

    assert await retry_pending_unlinks(store=store, client=client) == []
    assert store.get(record.link_id) is not None

    fake.unreachable.clear()
    done = await retry_pending_unlinks(store=store, client=client)
    assert [r.link_id for r in done] == [record.link_id]
    assert store.records() == []
    assert len(fake.calls(DELETE_SELF)) == 3


async def test_a_link_that_is_not_there_is_a_plain_not_found(store: LinkStore) -> None:
    fake = FakeUclone2()
    with pytest.raises(LinkError) as caught:
        await unlink("lnk_nothere", store=store, client=Uclone2LinkClient(transport=fake.transport))
    assert caught.value.failure is LinkFailure.NOT_FOUND
    assert fake.requests == []


async def test_connect_refusal_leaves_no_record(store: LinkStore) -> None:
    fake = FakeUclone2()
    fake.answer(CONNECT, 409)
    with pytest.raises(LinkError):
        await _linked(store, fake)
    assert store.records() == []
