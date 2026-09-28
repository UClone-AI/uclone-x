"""``ucx link``: link, list and remove, as a user reads them.

What these pin is what reaches the terminal: plain sentences, the token only as
``ucl_…`` and four characters, and no status code, URL, error code or exception name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

import uclone_x.cli.commands.link as link_cli
from tests.support.uclone2_fake import (
    CODE,
    CONNECT,
    CONNECT_URL,
    DELETE_SELF,
    TOKEN,
    TOKEN_TAIL,
    FakeUclone2,
    internals_in,
)
from uclone_x.cli.main import app
from uclone_x.link.uclone2.client import Uclone2LinkClient
from uclone_x.link.uclone2.store import LINKS_DIR_ENV_VAR, LinkStore

runner = CliRunner()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeUclone2:
    server = FakeUclone2()
    monkeypatch.setenv(LINKS_DIR_ENV_VAR, str(tmp_path / "links"))
    monkeypatch.setattr(
        link_cli, "make_client", lambda: Uclone2LinkClient(transport=server.transport)
    )
    monkeypatch.setattr(link_cli, "local_clone_names", lambda: ["clone", "haru", "mina"])
    return server


def _flat(text: str) -> str:
    return " ".join(text.split())


def _run(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["link", *args])
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    return result.exit_code, result.output


def _linked_id() -> str:
    [record] = LinkStore().records()
    return record.link_id


def test_linking_names_both_clones_and_matches_by_name(fake: FakeUclone2) -> None:
    code, out = _run("uclone2", CONNECT_URL)
    assert code == 0, out
    assert "로컬 클론 'haru'" in out
    assert "@haru" in out
    assert _linked_id() in out
    assert internals_in(out) == []


def test_without_a_same_named_clone_the_default_answers_and_says_so(
    fake: FakeUclone2, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(link_cli, "local_clone_names", lambda: ["clone"])
    code, out = _run("uclone2", CONNECT_URL)
    assert code == 0, out
    assert "로컬 클론 'clone'" in out
    assert "기본 클론이 답합니다" in _flat(out)


def test_an_unknown_clone_is_refused_before_the_code_is_spent(fake: FakeUclone2) -> None:
    code, out = _run("uclone2", CONNECT_URL, "--clone", "nobody")
    assert code == 2
    assert "'nobody' 클론이 없습니다" in out
    assert fake.requests == []


def test_an_expired_code_prints_the_design_copy_and_nothing_else(fake: FakeUclone2) -> None:
    fake.answer(CONNECT, 400)
    code, out = _run("uclone2", CONNECT_URL)
    assert code == 1
    assert _flat(out) == "코드가 만료되었거나 이미 사용되었습니다. uClone2에서 새로 받으십시오"


@pytest.mark.parametrize("status", [400, 409, 429, 503])
def test_every_refusal_is_plain_copy(fake: FakeUclone2, status: int) -> None:
    fake.answer(CONNECT, status)
    code, out = _run("uclone2", CONNECT_URL)
    assert code == 1
    assert internals_in(out) == []


def test_an_unreachable_server_is_plain_copy(fake: FakeUclone2) -> None:
    fake.unreachable.add(CONNECT)
    code, out = _run("uclone2", CONNECT_URL)
    assert code == 1
    assert "uClone2에 닿을 수 없습니다" in out
    assert internals_in(out) == []


def test_pasted_garbage_is_refused_without_echoing_it(fake: FakeUclone2) -> None:
    code, out = _run("uclone2", f"https://evil.example/steal/{CODE}")
    assert code == 1
    assert internals_in(out) == []
    assert fake.requests == []


@pytest.mark.parametrize(
    "pasted", [f"https://[::1/link/{CODE}", f"https://uclone.example:abc/link/{CODE}"]
)
def test_an_unparseable_address_is_plain_copy_not_a_traceback(
    fake: FakeUclone2, pasted: str
) -> None:
    result = runner.invoke(app, ["link", "uclone2", pasted])
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert result.exit_code == 1
    out = result.output
    assert "Traceback" not in out
    assert "붙여 넣은 내용이 uClone2 연결 주소나 코드가 아닙니다" in _flat(out)
    assert CODE not in out
    assert TOKEN not in out
    assert internals_in(out) == []
    assert fake.requests == []


def test_list_masks_the_token(fake: FakeUclone2) -> None:
    """Killed by: src/uclone_x/cli/commands/link.py :: Text(view.token_hint),
    Becomes: Text(LinkStore().get(view.link_id).token.get_secret_value()),
    """
    _run("uclone2", CONNECT_URL)
    code, out = _run("list")
    assert code == 0
    assert TOKEN not in out
    assert TOKEN_TAIL in out
    assert "연결됨" in out
    assert "@haru" in out


def test_an_empty_list_says_how_to_link(fake: FakeUclone2) -> None:
    code, out = _run("list")
    assert code == 0
    assert "./ucx link uclone2" in out


def test_remove_unlinks_and_forgets(fake: FakeUclone2) -> None:
    _run("uclone2", CONNECT_URL)
    code, out = _run("remove", _linked_id())
    assert code == 0
    assert "연결을 해제했습니다" in out
    assert LinkStore().records() == []


def test_an_unreachable_remove_is_kept_pending_and_finished_on_the_next_run(
    fake: FakeUclone2,
) -> None:
    _run("uclone2", CONNECT_URL)
    link_id = _linked_id()
    fake.unreachable.add(DELETE_SELF)

    code, out = _run("remove", link_id)
    assert code == 1
    assert "해제 대기" in out
    assert internals_in(out) == []
    code, out = _run("list")
    assert "해제 대기" in out

    fake.unreachable.clear()
    code, out = _run("list")
    assert "미뤄 둔 연결 해제를 마쳤습니다" in out
    assert LinkStore().records() == []


def test_a_pending_unlink_can_be_removed_from_this_computer_only(fake: FakeUclone2) -> None:
    """`remove --local` drops a *해제 대기* record without asking uClone2, and says the link
    may still be there.

    Killed by: src/uclone_x/link/uclone2/service.py :: store.remove(record.link_id)
    Becomes: pass
    """
    _run("uclone2", CONNECT_URL)
    link_id = _linked_id()
    fake.unreachable.add(DELETE_SELF)
    code, out = _run("remove", link_id)
    assert f"./ucx link remove {link_id} --local" in _flat(out)

    calls = len(fake.requests)
    code, out = _run("remove", link_id, "--local")
    assert code == 0, out
    assert "uClone2에는 연결이 남아 있을 수 있으니" in _flat(out)
    assert LinkStore().records() == []
    assert len(fake.requests) == calls, "a local-only removal asked uClone2"


def test_local_removal_is_refused_for_a_link_that_is_not_pending(fake: FakeUclone2) -> None:
    _run("uclone2", CONNECT_URL)
    code, out = _run("remove", _linked_id(), "--local")
    assert code == 1
    assert "해제 대기 중이 아닙니다" in _flat(out)
    assert len(LinkStore().records()) == 1


def test_with_no_local_clone_at_all_the_code_is_not_spent(
    fake: FakeUclone2, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(link_cli, "local_clone_names", list[str])
    code, out = _run("uclone2", CONNECT_URL)
    assert code == 2
    assert "답할 클론이 없습니다" in _flat(out)
    assert fake.requests == []


def test_without_a_same_named_clone_or_the_default_the_link_is_undone(
    fake: FakeUclone2, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(link_cli, "local_clone_names", lambda: ["mina"])
    code, out = _run("uclone2", CONNECT_URL)
    assert code == 1
    assert "기본 클론도 없어서 연결을 되돌렸습니다" in _flat(out)
    assert LinkStore().records() == []
    assert internals_in(out) == []


def test_removing_an_unknown_link_is_plain_copy(fake: FakeUclone2) -> None:
    code, out = _run("remove", "lnk_nothere")
    assert code == 1
    assert "그런 연결이 없습니다" in out


def test_no_command_prints_or_logs_the_token(
    fake: FakeUclone2, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    outputs = [_run("uclone2", CONNECT_URL)[1], _run("list")[1]]
    link_id = _linked_id()
    fake.unreachable.add(DELETE_SELF)
    outputs.append(_run("remove", link_id)[1])
    fake.unreachable.clear()
    outputs.append(_run("list")[1])

    assert caplog.records
    for text in [*outputs, *(r.getMessage() for r in caplog.records)]:
        assert TOKEN not in text
        assert CODE not in text
