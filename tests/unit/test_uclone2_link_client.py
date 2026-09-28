"""`Uclone2LinkClient` and `parse_connect_input` against an in-process uClone2.

The fake answers only as the vendored contract allows (see `tests/support/uclone2_fake.py`).
The tests that feed the client a server outside the contract build their own transport.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic import SecretStr

from tests.support.uclone2_fake import (
    BOT_ID,
    CODE,
    CONNECT,
    CONNECT_URL,
    DELETE_SELF,
    GET_SELF,
    ORIGIN,
    TOKEN,
    TOKEN_TAIL,
    FakeUclone2,
    internals_in,
)
from uclone_x.link.uclone2.client import (
    MESSAGES,
    PRODUCTION_ORIGIN,
    LinkError,
    LinkFailure,
    Uclone2LinkClient,
    UnlinkResult,
    parse_connect_input,
)
from uclone_x.link.uclone2.models import mask_token


def _client(fake: FakeUclone2) -> Uclone2LinkClient:
    return Uclone2LinkClient(transport=fake.transport, runtime_version="9.9.9")


def _off_contract(status: int, body: bytes = b"") -> Uclone2LinkClient:
    """A server that answers outside the contract: every route, `status` and `body`."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return Uclone2LinkClient(transport=httpx.MockTransport(handle))


def _secret_free(err: BaseException) -> None:
    """Nothing reachable from `err` names the token or the code."""
    seen = [str(err), repr(err), *map(str, err.args)]
    assert err.__cause__ is None
    assert err.__context__ is None or TOKEN not in repr(err.__context__)
    seen.append(getattr(err, "diagnostic", ""))
    for text in seen:
        assert TOKEN not in text
        assert CODE not in text


# --- parsing -------------------------------------------------------------------------


def test_a_connect_url_carries_its_own_origin() -> None:
    parsed = parse_connect_input(f"  {CONNECT_URL}/ \n")
    assert parsed.server_url == ORIGIN
    assert parsed.code.get_secret_value() == CODE


def test_a_bare_code_goes_to_production_unless_a_server_is_named() -> None:
    assert parse_connect_input(CODE).server_url == PRODUCTION_ORIGIN
    assert parse_connect_input(CODE, "http://127.0.0.1:8080").server_url == "http://127.0.0.1:8080"


@pytest.mark.parametrize(
    ("text", "server"),
    [
        ("http://uclone.example/link/Ab3dE6gH9j", None),  # plain http off this machine
        ("https://uclone.example/join/Ab3dE6gH9j", None),  # not a connect path
        ("https://uclone.example/link/Ab3dE6gH9j?x=1", None),
        ("https://user:pw@uclone.example/link/Ab3dE6gH9j", None),
        ("https://uclone.example/link/short", None),
        ("Ab3dE6gH9", None),  # nine characters
        ("Ab3dE6gH9j!", None),
        ("Ab3dE6gH9j", "http://uclone.example"),
        ("Ab3dE6gH9j", "https://uclone.example/api"),
    ],
)
def test_input_that_is_not_a_connect_url_or_code_is_refused_before_sending(
    text: str, server: str | None
) -> None:
    with pytest.raises(LinkError) as caught:
        parse_connect_input(text, server)
    assert caught.value.failure is LinkFailure.BAD_INPUT


def test_a_server_given_with_a_url_is_refused_rather_than_guessed() -> None:
    with pytest.raises(LinkError) as caught:
        parse_connect_input(CONNECT_URL, "https://uclone.ai")
    assert caught.value.failure is LinkFailure.SERVER_WITH_URL


@pytest.mark.parametrize(
    "text",
    [
        f"https://[::1/link/{CODE}",  # `urlsplit` raises on the unbalanced bracket
        f"https://uclone.example:abc/link/{CODE}",  # `.port` raises on a non-number
    ],
)
def test_an_address_urlsplit_cannot_read_is_bad_input_not_a_crash(text: str) -> None:
    with pytest.raises(LinkError) as caught:
        parse_connect_input(text)
    assert caught.value.failure is LinkFailure.BAD_INPUT
    _secret_free(caught.value)


@pytest.mark.parametrize("server", ["https://[::1", "https://uclone.example:abc"])
def test_a_server_urlsplit_cannot_read_is_bad_input_not_a_crash(server: str) -> None:
    with pytest.raises(LinkError) as caught:
        parse_connect_input(CODE, server)
    assert caught.value.failure is LinkFailure.BAD_INPUT


@pytest.mark.parametrize("server", ["https://uclone.example:abc", "https://[::1"])
async def test_an_address_httpx_cannot_build_is_bad_input_not_a_crash(server: str) -> None:
    """`httpx.InvalidURL` is not an `httpx.HTTPError`; it must not escape `_send`.

    Killed by: src/uclone_x/link/uclone2/client.py :: except httpx.InvalidURL:
    Becomes: except ZeroDivisionError:
    """
    fake = FakeUclone2()
    with pytest.raises(LinkError) as caught:
        await _client(fake).connect(server, SecretStr(CODE))
    assert caught.value.failure is LinkFailure.BAD_INPUT
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    _secret_free(caught.value)
    assert fake.requests == []


# --- connect -------------------------------------------------------------------------


async def test_connect_posts_the_code_in_the_body_and_returns_the_token() -> None:
    fake = FakeUclone2()
    result = await _client(fake).connect(ORIGIN, SecretStr(CODE))

    assert result.token.get_secret_value() == TOKEN
    assert result.bot_id == BOT_ID
    assert result.clone.username == "haru"
    [request] = fake.calls(CONNECT)
    assert str(request.url) == f"{ORIGIN}/api/v4/linked/connect"
    assert CODE not in str(request.url)
    assert json.loads(request.content) == {
        "code": CODE,
        "runtime": {"name": "uclone-x", "version": "9.9.9"},
    }
    assert "authorization" not in request.headers


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (400, LinkFailure.CODE_INVALID),
        (409, LinkFailure.CAP_REACHED),
        (429, LinkFailure.RATE_LIMITED),
        (503, LinkFailure.DISABLED),
    ],
)
async def test_each_connect_refusal_is_a_plain_sentence(status: int, failure: LinkFailure) -> None:
    fake = FakeUclone2()
    fake.answer(CONNECT, status)
    with pytest.raises(LinkError) as caught:
        await _client(fake).connect(ORIGIN, SecretStr(CODE))

    err = caught.value
    assert err.failure is failure
    assert str(err) == MESSAGES[failure]
    assert internals_in(str(err)) == []
    assert err.diagnostic.startswith(str(status))
    _secret_free(err)


async def test_an_expired_code_reads_as_the_design_says() -> None:
    fake = FakeUclone2()
    fake.answer(CONNECT, 400)
    with pytest.raises(LinkError) as caught:
        await _client(fake).connect(ORIGIN, SecretStr(CODE))
    assert (
        str(caught.value) == "코드가 만료되었거나 이미 사용되었습니다. uClone2에서 새로 받으십시오"
    )


def test_every_failure_message_is_plain_copy() -> None:
    assert set(MESSAGES) == set(LinkFailure)
    for failure, message in MESSAGES.items():
        assert internals_in(message) == [], failure


async def test_an_unreachable_server_is_a_plain_sentence_without_the_cause() -> None:
    fake = FakeUclone2(unreachable={CONNECT})
    with pytest.raises(LinkError) as caught:
        await _client(fake).connect(ORIGIN, SecretStr(CODE))
    assert caught.value.failure is LinkFailure.UNREACHABLE
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__
    _secret_free(caught.value)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, b"internal"),
        (302, b""),
        (200, b"<html>not json</html>"),
        (200, b'{"status":"success","data":{"bot_id":"x"}}'),
    ],
)
async def test_an_answer_outside_the_contract_is_a_server_error(status: int, body: bytes) -> None:
    with pytest.raises(LinkError) as caught:
        await _off_contract(status, body).connect(ORIGIN, SecretStr(CODE))
    assert caught.value.failure is LinkFailure.SERVER_ERROR
    assert internals_in(str(caught.value)) == []


async def test_a_malformed_connect_answer_does_not_quote_the_token() -> None:
    """A validation error quotes its input; for `connect` the input is the token.

    Killed by: src/uclone_x/link/uclone2/client.py :: raise LinkError(LinkFailure.SERVER_ERROR, f"{response.status_code} unexpected body")
    Becomes: raise LinkError(LinkFailure.SERVER_ERROR, f"{response.status_code} {response.text}")
    """
    body = json.dumps({"status": "success", "data": {"token": TOKEN, "protocol": 2}}).encode()
    with pytest.raises(LinkError) as caught:
        await _off_contract(200, body).connect(ORIGIN, SecretStr(CODE))
    _secret_free(caught.value)


# --- self and unlink -----------------------------------------------------------------


async def test_get_self_sends_the_token_only_as_a_bearer_header() -> None:
    fake = FakeUclone2()
    current = await _client(fake).get_self(ORIGIN, SecretStr(TOKEN))

    assert current.bot_id == BOT_ID
    assert current.pending == 3
    [request] = fake.calls(GET_SELF)
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in str(request.url)
    assert request.content == b""


async def test_a_revoked_token_reads_as_the_link_being_over() -> None:
    fake = FakeUclone2()
    fake.answer(GET_SELF, 401)
    with pytest.raises(LinkError) as caught:
        await _client(fake).get_self(ORIGIN, SecretStr(TOKEN))
    assert caught.value.failure is LinkFailure.TOKEN_INVALID
    assert str(caught.value) == (
        "이 연결은 끝났습니다 — uClone2에서 해제했거나 다른 곳에서 다시 연결했습니다"
    )
    _secret_free(caught.value)


async def test_unlink_deletes_with_the_token() -> None:
    fake = FakeUclone2()
    assert await _client(fake).unlink(ORIGIN, SecretStr(TOKEN)) is UnlinkResult.UNLINKED
    [request] = fake.calls(DELETE_SELF)
    assert request.headers["authorization"] == f"Bearer {TOKEN}"


async def test_unlinking_a_link_already_gone_counts_as_done() -> None:
    fake = FakeUclone2()
    fake.answer(DELETE_SELF, 401)
    assert await _client(fake).unlink(ORIGIN, SecretStr(TOKEN)) is UnlinkResult.ALREADY_GONE


@pytest.mark.parametrize("unreachable", [True, False])
async def test_an_unlink_that_does_not_land_raises(unreachable: bool) -> None:
    fake = FakeUclone2(unreachable={DELETE_SELF} if unreachable else set())
    if not unreachable:
        fake.answer(DELETE_SELF, 503)
    with pytest.raises(LinkError) as caught:
        await _client(fake).unlink(ORIGIN, SecretStr(TOKEN))
    expected = LinkFailure.UNREACHABLE if unreachable else LinkFailure.DISABLED
    assert caught.value.failure is expected
    _secret_free(caught.value)


async def test_the_client_does_not_follow_a_redirect_with_the_token() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(307, headers={"Location": "https://elsewhere.example/steal"})

    client = Uclone2LinkClient(transport=httpx.MockTransport(handle))
    with pytest.raises(LinkError):
        await client.get_self(ORIGIN, SecretStr(TOKEN))
    assert len(seen) == 1


# --- the token never shows -----------------------------------------------------------


async def test_no_log_line_from_any_path_names_the_token_or_the_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every route, every declared answer, a network failure and a malformed body.

    Killed by: src/uclone_x/link/uclone2/client.py :: logger.info("uClone2 %s %s could not be sent: %s", method, path, type(err).__name__)
    Becomes: logger.info("uClone2 %s %s could not be sent: %s", method, headers, type(err).__name__)
    """
    caplog.set_level(logging.DEBUG)
    for route, statuses in {CONNECT: (200, 400, 409, 429, 503), GET_SELF: (200, 401, 503)}.items():
        for status in statuses:
            fake = FakeUclone2()
            fake.answer(route, status)
            client = _client(fake)
            try:
                if route == CONNECT:
                    await client.connect(ORIGIN, SecretStr(CODE))
                else:
                    await client.get_self(ORIGIN, SecretStr(TOKEN))
            except LinkError:
                pass
    for route in (GET_SELF, DELETE_SELF):
        unreachable = _client(FakeUclone2(unreachable={route}))
        with pytest.raises(LinkError):
            if route == GET_SELF:
                await unreachable.get_self(ORIGIN, SecretStr(TOKEN))
            else:
                await unreachable.unlink(ORIGIN, SecretStr(TOKEN))
    malformed = json.dumps({"data": {"token": TOKEN}}).encode()
    with pytest.raises(LinkError):
        await _off_contract(200, malformed).connect(ORIGIN, SecretStr(CODE))

    assert caplog.records, "the paths above log; an empty capture would prove nothing"
    for record in caplog.records:
        line = record.getMessage()
        assert TOKEN not in line
        assert CODE not in line


def test_the_connect_result_and_its_mask_do_not_print_the_token() -> None:
    fake_token = SecretStr(TOKEN)
    assert mask_token(fake_token) == f"ucl_…{TOKEN_TAIL}"
    assert TOKEN not in repr(fake_token)
