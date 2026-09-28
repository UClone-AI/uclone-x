"""``ucx link``: link a local clone to its clone in uClone2, list, remove, and run the links.

A head over ``uclone_x.link.uclone2.service``: every rule about what is stored, what is
kept and what is retried lives there, so the dashboard's Settings section will behave the
same. This module only chooses the local clone, runs the call and says what happened.

What it prints is for the user: plain sentences, the token only ever as ``ucl_…`` and
four characters, and no status code, URL or exception name. Server-supplied names are
printed with markup off, so a uClone2 username cannot style or hide the output.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from uclone_x.link.uclone2.client import LinkError, Uclone2LinkClient
from uclone_x.link.uclone2.models import LinkRecord, LinkView
from uclone_x.link.uclone2.runtime_lock import RuntimeRole
from uclone_x.link.uclone2.service import (
    FALLBACK_CLONE,
    UnlinkOutcome,
    check_local_choice,
    forget_local,
    link_uclone2,
    local_chooser,
    retry_pending_unlinks,
    unlink,
)
from uclone_x.link.uclone2.store import LinkStore, LinkStoreError

if TYPE_CHECKING:
    # `websockets` (the `http` extra) comes with these; `link run` imports them when it runs.
    from uclone_x.link.uclone2.session import LinkSessionState
    from uclone_x.link.uclone2.supervisor import LinkSupervisor, StateListener

console = Console()

link_app = typer.Typer(
    name="link",
    help="Link a local clone to uClone2 so it answers its guestbook and Square there",
    no_args_is_help=True,
)


def local_clone_names() -> list[str]:
    """The local clones a link can speak through: the persona registry's names."""
    from uclone_x.agent.persona_registry import get_default_persona_registry

    registry = get_default_persona_registry(workspace_root=Path.cwd().resolve())
    return [p.name for p in registry.list_personas()]


def make_client() -> Uclone2LinkClient:
    """The client every command uses; a seam for tests."""
    return Uclone2LinkClient()


def make_store() -> LinkStore:
    return LinkStore()


def make_supervisor(
    store: LinkStore, client: Uclone2LinkClient, on_state: StateListener
) -> LinkSupervisor:
    """The supervisor `link run` serves the links with; a seam for tests."""
    # Deferred and guarded, as `cli/commands/a2a.py` does: the sessions need `websockets`,
    # which a `uclone-x[cli]` install does not have, and `cli/main.py` imports this module.
    try:
        from uclone_x.link.uclone2.supervisor import LinkSupervisor
    except ImportError as exc:
        from uclone_x.errors import MissingDependencyError

        raise MissingDependencyError(
            extra="http", package="websockets", feature="uClone2 links (ucx link run)"
        ) from exc
    return LinkSupervisor(store=store, client=client, role=RuntimeRole.LINK_RUN, on_state=on_state)


def _say(text: str, style: str | None = None) -> None:
    console.print(Text(text, style=style or ""))


def _refuse(message: str, code: int = 1) -> typer.Exit:
    _say(message, "red")
    return typer.Exit(code=code)


def _state(view: LinkView) -> str:
    if view.unlink_pending:
        return "해제 대기 — 다음에 다시 시도합니다"
    if not view.enabled:
        return "연결이 끝났습니다"
    return "연결됨"


def _finish_pending(store: LinkStore, client: Uclone2LinkClient) -> None:
    """Retry any unlink left *해제 대기*; each command run is this head's "next start"."""
    if not any(r.unlink_pending for r in store.records()):
        return
    for record in asyncio.run(retry_pending_unlinks(store=store, client=client)):
        _say(f"미뤄 둔 연결 해제를 마쳤습니다: uClone2 @{record.remote_username}")


@link_app.command("uclone2")
def link_uclone2_command(
    connect: Annotated[
        str,
        typer.Argument(help="The connect URL from uClone2 (https://…/link/<code>), or the code"),
    ],
    clone: Annotated[
        str | None,
        typer.Option("--clone", help="The local clone that answers (default: the same name)"),
    ] = None,
    server: Annotated[
        str | None,
        typer.Option("--server", help="The uClone2 server, for a bare code only"),
    ] = None,
) -> None:
    """Link a local clone to a uClone2 clone with the connect URL uClone2 gave you."""
    store = make_store()
    client = make_client()
    known = local_clone_names()
    if clone is not None and known and clone not in known:
        # Checked before the exchange: the code is single use, and a typo here must not
        # spend it.
        raise _refuse(
            f"이 컴퓨터에 '{clone}' 클론이 없습니다. 있는 클론: {', '.join(known)}", code=2
        )
    try:
        check_local_choice(clone, known)
    except LinkError as err:
        raise _refuse(str(err), code=2) from None
    try:
        _finish_pending(store, client)
        record = asyncio.run(
            link_uclone2(
                connect,
                choose_local=local_chooser(clone, known),
                store=store,
                client=client,
                server_url=server,
            )
        )
    except (LinkError, LinkStoreError) as err:
        raise _refuse(str(err)) from None
    _say(
        f"연결했습니다: 로컬 클론 '{record.local_agent_id}' ↔ uClone2 @{record.remote_username}",
        "green",
    )
    if (
        clone is None
        and record.local_agent_id == FALLBACK_CLONE
        and (
            FALLBACK_CLONE
            not in (record.remote_username.lower(), record.remote_display_name.lower())
        )
    ):
        _say(
            "같은 이름의 로컬 클론이 없어서 기본 클론이 답합니다. "
            "다른 클론이 답하게 하려면 연결을 해제하고 --clone으로 골라 다시 연결하십시오"
        )
    _say(f"연결 번호: {record.link_id}")


@link_app.command("list")
def list_command() -> None:
    """Show every uClone2 link on this machine (tokens masked)."""
    store = make_store()
    try:
        _finish_pending(store, make_client())
        views = store.views()
    except (LinkError, LinkStoreError) as err:
        raise _refuse(str(err)) from None
    if not views:
        _say("연결된 uClone2 클론이 없습니다. ./ucx link uclone2 <연결 주소>로 연결하십시오")
        return
    table = Table(title="uClone2 연결")
    for column in ("번호", "로컬 클론", "uClone2", "상태", "토큰"):
        table.add_column(column)
    for view in views:
        table.add_row(
            Text(view.link_id),
            Text(view.local_agent_id),
            Text(f"@{view.remote_username}"),
            Text(_state(view)),
            Text(view.token_hint),
        )
    console.print(table)


@link_app.command("remove")
def remove_command(
    link_id: Annotated[str, typer.Argument(help="The link's number, from `ucx link list`")],
    local: Annotated[
        bool,
        typer.Option(
            "--local",
            help="Only for a link left pending unlink: delete it from this machine without "
            "asking uClone2. The link may still exist in uClone2; end it there.",
        ),
    ] = False,
) -> None:
    """Unlink: uClone2 takes the clone offline and back from this machine."""
    store = make_store()
    if local:
        try:
            record = forget_local(link_id, store=store)
        except (LinkError, LinkStoreError) as err:
            raise _refuse(str(err)) from None
        _say(
            f"이 컴퓨터에서 uClone2 @{record.remote_username} 연결 기록을 지웠습니다. "
            "uClone2에는 연결이 남아 있을 수 있으니 uClone2의 클론 페이지에서 해제하십시오",
            "green",
        )
        return
    client = make_client()
    try:
        record = store.get(link_id)
        outcome = asyncio.run(unlink(link_id, store=store, client=client))
    except (LinkError, LinkStoreError) as err:
        raise _refuse(str(err)) from None
    name = f"@{record.remote_username}" if record is not None else link_id
    if outcome is UnlinkOutcome.REMOVED:
        _say(f"연결을 해제했습니다: uClone2 {name}", "green")
        return
    raise _refuse(
        f"uClone2에 닿지 않아 {name} 연결 해제를 끝내지 못했습니다. "
        "해제 대기로 두고, 다음에 ./ucx link를 실행할 때 다시 시도합니다. "
        f"계속 해제되지 않으면 ./ucx link remove {link_id} --local로 이 컴퓨터에서만 지울 수 있습니다"
    )


# --- link run ---------------------------------------------------------------------------

_HELD_BY = {
    RuntimeRole.DASHBOARD: (
        "이 컴퓨터의 대시보드가 이미 uClone2 연결을 맡고 있습니다. "
        "대시보드를 닫은 뒤 다시 실행하십시오"
    ),
    RuntimeRole.LINK_RUN: "이 컴퓨터에서 ./ucx link run이 이미 실행 중입니다",
}
_HELD_BY_SOMEONE = (
    "이 컴퓨터에서 실행 중인 다른 UClone-X가 이미 uClone2 연결을 맡고 있습니다. "
    "그쪽을 멈춘 뒤 다시 실행하십시오"
)


def _state_line(record: LinkRecord, state: LinkSessionState) -> str:
    from uclone_x.link.uclone2.session import STATE_TEXT  # loaded with the supervisor

    stamp = datetime.now().strftime("%H:%M:%S")
    return (
        f"{stamp} uClone2 @{record.remote_username} ({record.local_agent_id}): {STATE_TEXT[state]}"
    )


async def _serve_links(store: LinkStore, client: Uclone2LinkClient) -> int:
    """`link run`: keep every enabled link online until a signal, then log them all out."""
    supervisor = make_supervisor(store, client, lambda r, s: _say(_state_line(r, s)))
    # The *해제 대기* retry runs first and on its own, so it is reported before anything
    # else and happens even when nothing is left to dial.
    await supervisor.start_unlink_retry()
    for record in await supervisor.unlinks_retried():
        _say(f"미뤄 둔 연결 해제를 마쳤습니다: uClone2 @{record.remote_username}")
    runnable = supervisor.runnable()
    if not runnable:
        _say(
            "켜 둘 uClone2 연결이 없습니다. ./ucx link uclone2 <연결 주소>로 연결하거나, "
            "오프라인으로 전환한 연결은 대시보드에서 다시 켜십시오"
        )
        return 0
    if not supervisor.claim():
        holder = supervisor.holder()
        _say(_HELD_BY[holder] if holder is not None else _HELD_BY_SOMEONE, "red")
        return 1

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    handled: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stop.set)
            handled.append(sig)
    try:
        _say(f"uClone2 연결 {len(runnable)}개를 켭니다. 멈추려면 Ctrl-C를 누르십시오")
        await supervisor.start()
        sessions = [s for r in runnable if (s := supervisor.session(r.link_id)) is not None]
        stopped = asyncio.create_task(stop.wait())
        ended = asyncio.gather(*(s.wait() for s in sessions), return_exceptions=True)
        await asyncio.wait({stopped, ended}, return_when=asyncio.FIRST_COMPLETED)
        stopped.cancel()
        ended.cancel()
        with contextlib.suppress(BaseException):
            await ended
        signalled = stop.is_set()
        # A deliberate stop: each session sends `bye{logout}`, so uClone2 shows the clone
        # offline at once (each bounded by its logout wait, all in parallel).
        await supervisor.shutdown()
    finally:
        for sig in handled:
            loop.remove_signal_handler(sig)
    if signalled:
        _say("uClone2에서 오프라인으로 전환하고 멈췄습니다")
        return 0
    _say("켜 둔 uClone2 연결이 모두 멈췄습니다. 위의 안내를 확인하십시오", "red")
    return 1


@link_app.command("run")
def run_command() -> None:
    """Keep every enabled link online without the dashboard, until Ctrl-C (for an always-on machine)."""
    store = make_store()
    client = make_client()
    try:
        code = asyncio.run(_serve_links(store, client))
    except (LinkError, LinkStoreError) as err:
        raise _refuse(str(err)) from None
    raise typer.Exit(code=code)
