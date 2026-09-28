"""`LinkStore`: the owner-only links file, and the views that carry no token."""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from tests.support.uclone2_fake import TOKEN, TOKEN_TAIL, internals_in
from uclone_x.link.uclone2.models import LinkRecord
from uclone_x.link.uclone2.store import (
    LINKS_DIR_ENV_VAR,
    LinkStore,
    LinkStoreError,
    default_store_path,
)


def _record(link_id: str = "lnk_00000001", bot_id: str = "bot_a", token: str = TOKEN) -> LinkRecord:
    return LinkRecord(
        link_id=link_id,
        server_url="https://uclone.ai",
        bot_id=bot_id,
        remote_username="haru",
        local_agent_id="clone",
        token=SecretStr(token),
        ws_url="wss://uclone.ai/api/v4/linked/ws",
        created_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


@pytest.fixture
def store(tmp_path: Path) -> LinkStore:
    return LinkStore(tmp_path / "links" / "uclone2.json")


def test_a_missing_file_is_no_links(store: LinkStore) -> None:
    assert store.records() == []
    assert store.views() == []
    assert not store.path.exists()


def test_the_file_is_owner_only_from_its_first_write(store: LinkStore) -> None:
    """Killed by: src/uclone_x/link/uclone2/store.py :: os.fchmod(fd, 0o600)
    Becomes: os.fchmod(fd, 0o644)
    """
    store.put(_record())

    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    leftovers = [
        p.name for p in store.path.parent.iterdir() if p.name.startswith(".uclone2-links.")
    ]
    assert leftovers == []


def test_a_record_round_trips_with_its_token(store: LinkStore) -> None:
    store.put(_record())
    [loaded] = store.records()
    assert loaded.token.get_secret_value() == TOKEN
    assert loaded == _record()
    assert json.loads(store.path.read_text())["version"] == 1


def test_views_carry_a_masked_hint_and_no_token(store: LinkStore) -> None:
    store.put(_record())
    [view] = store.views()
    assert view.token_hint == f"ucl_…{TOKEN_TAIL}"
    dumped = view.model_dump_json()
    assert TOKEN not in dumped
    assert "token" not in type(view).model_fields
    assert TOKEN not in repr(store.records())


def test_linking_the_same_clone_again_replaces_its_dead_record(store: LinkStore) -> None:
    store.put(_record("lnk_00000001", "bot_a", "ucl_old_token_aaaaaaaaaaaa"))
    store.put(_record("lnk_00000002", "bot_b"))
    store.put(_record("lnk_00000003", "bot_a"))
    assert [(r.link_id, r.bot_id) for r in store.records()] == [
        ("lnk_00000002", "bot_b"),
        ("lnk_00000003", "bot_a"),
    ]


def test_replace_and_remove_report_whether_the_link_existed(store: LinkStore) -> None:
    store.put(_record())
    assert store.replace(_record().model_copy(update={"enabled": False}))
    replaced = store.get("lnk_00000001")
    assert replaced is not None and not replaced.enabled
    assert not store.replace(_record("lnk_missing"))
    assert store.remove("lnk_00000001")
    assert not store.remove("lnk_00000001")
    assert store.records() == []


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '{"version": 1}',
        '["a list"]',
        json.dumps({"version": 1, "links": [{"link_id": "x", "token": TOKEN}]}),
    ],
)
def test_an_unusable_file_is_a_plain_sentence_that_quotes_nothing(
    store: LinkStore, content: str
) -> None:
    """A validation error quotes its input, and this input holds a token."""
    store.path.parent.mkdir(parents=True)
    store.path.write_text(content)
    with pytest.raises(LinkStoreError) as caught:
        store.records()
    err = caught.value
    assert internals_in(str(err)) == []
    assert err.__cause__ is None
    assert err.__suppress_context__ or err.__context__ is None
    assert TOKEN not in str(err) and TOKEN not in repr(err)


def test_the_links_directory_follows_its_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LINKS_DIR_ENV_VAR, str(tmp_path / "elsewhere"))
    assert default_store_path() == tmp_path / "elsewhere" / "uclone2.json"
    monkeypatch.delenv(LINKS_DIR_ENV_VAR)
    assert default_store_path() == Path.home() / ".uclone" / "links" / "uclone2.json"
