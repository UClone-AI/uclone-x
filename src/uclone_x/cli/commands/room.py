"""`ucx room` — create a multi-agent room, change who is in it, and talk to it.

Before this the feature had no user-reachable path at all: the only way to use a room was
to assemble a `RoomState` in Python and call `RoomStore.save` by hand. The CLI comes before
the head deliberately — it is verifiable without a browser, so the Core's seams are
exercised by something a test can drive end to end.

**Every command here is a shell (P8).** Creation, roster edits and the join/leave record
live in `uclone_x.room.service`; turn-taking lives in `RoomOrchestrator`; the selector
chain is assembled by `build_selector_chain`. What this module contributes is argument
parsing, a table, and turning a `RoomError` into an exit code instead of a traceback. If a
rule about rooms appears below, it is in the wrong file.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from uclone_x.core.provenance import Provenance, ServiceRef
from uclone_x.errors import RoomError, UCloneXError
from uclone_x.room.models import (
    ParticipantKind,
    RoomMessage,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
)
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.protocols import RoomOrchestratorProtocol
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import build_selector_chain
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore

console = Console()

room_app = typer.Typer(
    name="room",
    help="Create and drive multi-agent conversation rooms",
    no_args_is_help=True,
)


def build_service() -> RoomService:
    """The room service over the configured store (`UCLONE_ROOM_DIR`).

    A function and not a module-level singleton: the store resolves its directory at
    construction, so a captured instance would pin whatever the environment said at import
    time and ignore a later change.
    """
    return RoomService(RoomStore())


def build_orchestrator(
    *,
    store: RoomStore,
    policy: RoomPolicy,
    provider: str | None = None,
    model: str | None = None,
) -> RoomOrchestratorProtocol:
    """Assemble the orchestrator that `say` drives.

    Constructed per invocation and given no event bus: `post` returns only once every turn
    has landed, and a CLI that prints the result afterwards has nothing to stream to. The
    bus being optional is what lets this stay a four-line assembly.
    """
    from uclone_x.agent.composition import HostDependencies
    from uclone_x.agent.models import AgentLLMConfig
    from uclone_x.agent.persona_registry import get_default_persona_registry
    from uclone_x.agent.session import SessionStore
    from uclone_x.cli.commands.run import get_default_llm
    from uclone_x.engine.event_bus import EventBus
    from uclone_x.memory.store import default_cross_session_memory
    from uclone_x.telemetry import TelemetryTracer
    from uclone_x.tools.registry import create_default_registry

    llm = get_default_llm(provider)
    host = HostDependencies(
        bus=EventBus(),
        llm=llm,
        tools=create_default_registry(),
        tracer=TelemetryTracer(),
        store=SessionStore(),
    )
    resolver = RoomAgentResolver(
        host,
        llm_config=None if model is None else AgentLLMConfig(model_name=model),
        # One store per participant, so `record_memory_fact` reaches the seated agent's
        # own file. The host deliberately carries no `memory=`: a single store there is
        # read back by every agent in the room as its own recollection.
        memory_factory=default_cross_session_memory,
        persona_registry=get_default_persona_registry(),
    )
    return RoomOrchestrator(
        store=store,
        selectors=build_selector_chain(policy, provider=llm),
        resolver=resolver,
    )


def _fail(message: str) -> None:
    """Report a Core refusal and exit non-zero, rather than showing a traceback.

    The message is escaped because it quotes the user back to themselves — a room id, a
    title, a participant id — and Rich reads `[...]` in it as markup. An unbalanced tag in
    the text being refused would raise `MarkupError` *inside the error path*, replacing
    the refusal with a traceback: the failure this function exists to prevent, reached by
    the input most likely to have caused the refusal in the first place.
    """
    console.print(f"[bold red]✖[/bold red] {escape(message)}")
    raise typer.Exit(code=1)


def _policy(max_turns: int, responder: str) -> RoomPolicy:
    """The room's policy, with a validation refusal phrased as the flag that set it.

    `RoomPolicy` refuses a negative ceiling and one the selector window cannot hold, and
    both refusals name a *field*. A user typed `--max-turns`, so that is what the message
    must name; the alternative, which this replaces, was a pydantic traceback.
    """
    try:
        return RoomPolicy(
            max_agent_turns_per_human_message=max_turns, default_responder_id=responder
        )
    except ValidationError as exc:
        _fail(
            f"--max-turns {max_turns} is not a ceiling this room can keep: "
            f"{'; '.join(e['msg'] for e in exc.errors())}"
        )
        raise  # unreachable: `_fail` exits. Kept so the return type holds.


# --------------------------------------------------------------------------------------
# Rooms
# --------------------------------------------------------------------------------------


@room_app.command("create")
def room_create(
    title: str = typer.Argument(..., help="What to call this room"),
    room_id: str | None = typer.Option(None, "--id", help="Room id (generated when omitted)"),
    human: Annotated[
        list[str] | None, typer.Option("--human", help="Seat a human participant (repeatable)")
    ] = None,
    agent: Annotated[
        list[str] | None, typer.Option("--agent", help="Seat an agent (repeatable)")
    ] = None,
    persona: Annotated[
        list[str] | None,
        typer.Option(
            "--persona",
            help="AGENT=one line describing what that agent is for (repeatable)",
        ),
    ] = None,
    alias: Annotated[
        list[str] | None,
        typer.Option("--alias", help="AGENT=extra name it answers to in a mention (repeatable)"),
    ] = None,
    responder: str | None = typer.Option(
        None,
        "--responder",
        help="Agent that answers an unaddressed message; must be one of --agent",
    ),
    max_turns: int = typer.Option(
        3, "--max-turns", help="Agent utterances allowed between two human ones"
    ),
) -> None:
    """Create a room and seat its participants."""
    # `None` rather than `[]` as the default: a mutable default shared across invocations
    # is the bug B008 names, and typer treats both the same way.
    humans = list(human or ())
    agents = list(agent or ())
    # Keyed rather than positional. `--agent a --persona x --agent b --persona y` pairs by
    # counting, which silently mis-assigns the moment an option is reordered or omitted;
    # `--persona b=y` names its subject, so a mismatch is reported against a name instead
    # of landing on the wrong agent.
    try:
        personas = _keyed(persona, "--persona", agents)
        aliases = _keyed(alias, "--alias", agents)
    except ValueError as exc:
        _fail(str(exc))
        return
    service = build_service()

    if responder is not None and responder not in agents:
        # Checked before anything is written. `DefaultResponderSelector` raises on a
        # responder that is not an agent of the room, so accepting this would produce a
        # room that fails on its first unaddressed message — a failure several steps away
        # from the flag that caused it.
        _fail(
            f"--responder {responder!r} is not among the agents being seated "
            f"({', '.join(agents) or 'none'}). A room whose designated responder is not in "
            f"it fails on the first unaddressed message."
        )

    policy = _policy(max_turns, responder or "")
    try:
        state = service.create(title, room_id=room_id, policy=policy)
    except UCloneXError as exc:
        _fail(str(exc))
        return

    try:
        for human_id in humans:
            state = service.add_participant(state.room_id, human_id, kind=ParticipantKind.HUMAN)
        for agent_id in agents:
            raw_persona = personas.get(agent_id, "")
            state = service.add_participant(
                state.room_id,
                agent_id,
                kind=ParticipantKind.AGENT,
                persona_summary=raw_persona,
                aliases=tuple(aliases.get(agent_id, "").split(","))
                if aliases.get(agent_id)
                else (),
                persona=raw_persona,
            )
    except UCloneXError as exc:
        # One command, one outcome. A seat the Core refuses — a second human, a roster
        # this build will not serve — must not leave a room the user did not ask for
        # standing under the id they chose, which is also the id their corrected command
        # will try to use.
        service.delete(state.room_id)
        _fail(str(exc))
        return

    console.print(
        f"[bold green]✔[/bold green] Created room [cyan]{escape(state.room_id)}[/cyan] — "
        f"[bold]{escape(state.title)}[/bold] "
        f"({len(state.participants)} participant(s))"
    )


def _keyed(values: list[str] | None, flag: str, agents: list[str]) -> dict[str, str]:
    """Parse repeated `AGENT=VALUE` options, refusing an agent nobody is seating.

    Refused here rather than passed through, because the failure is otherwise silent: an
    unrecognised name simply never reaches a participant, and the operator gets a room whose
    agent has no persona without being told which flag was ignored.
    """
    parsed: dict[str, str] = {}
    for raw in values or ():
        name, sep, value = raw.partition("=")
        if not sep or not name.strip():
            raise ValueError(
                f"{flag} takes AGENT=VALUE, so that it names the agent it describes; got {raw!r}"
            )
        if name not in agents:
            raise ValueError(
                f"{flag} names {name!r}, which is not among the agents being seated "
                f"({', '.join(agents) or 'none'})"
            )
        existing = parsed.get(name)
        parsed[name] = f"{existing},{value}" if existing else value
    return parsed


@room_app.command("responder")
def room_responder(
    room_id: str = typer.Argument(..., help="Room to change"),
    agent_id: str | None = typer.Argument(None, help="Agent that answers unaddressed messages"),
    clear: bool = typer.Option(False, "--clear", help="Leave the room with no responder"),
) -> None:
    """Name the agent that answers an unaddressed message, or clear it.

    Without this the setting was write-once and lose-once: it could only be given to an
    agent seated by `create`, and `remove` clears it when that agent leaves — correctly,
    since a responder who is not in the room fails the next unaddressed message — after
    which nothing could name another.
    """
    service = build_service()
    try:
        state = service.get(room_id)
    except UCloneXError as exc:
        _fail(str(exc))
        return

    if clear:
        chosen = ""
    elif agent_id is None:
        _fail("Name an agent, or pass --clear to leave the room without a responder.")
        return
    else:
        seated = {p.id for p in state.participants if p.kind is ParticipantKind.AGENT}
        if agent_id not in seated:
            _fail(
                f"{agent_id!r} is not a seated agent of this room ({', '.join(sorted(seated)) or 'none'}). "
                f"A room whose designated responder is not in it fails on the first "
                f"unaddressed message."
            )
            return
        chosen = agent_id

    try:
        updated = service.set_default_responder(room_id, chosen)
    except UCloneXError as exc:
        _fail(str(exc))
        return

    if chosen:
        console.print(
            f"[bold green]✔[/bold green] [cyan]{chosen}[/cyan] now answers unaddressed "
            f"messages in [cyan]{updated.room_id}[/cyan]"
        )
    else:
        console.print(
            f"[bold green]✔[/bold green] [cyan]{updated.room_id}[/cyan] now has no "
            f"default responder; an unaddressed message falls to whatever selector follows"
        )


@room_app.command("list")
def room_list() -> None:
    """List stored rooms by title."""
    summaries = build_service().list_rooms()
    if not summaries:
        console.print(
            f"[yellow]No rooms in the store ({RoomStore().storage_dir}).[/yellow]\n"
            "[dim]Create one with './ucx room create \"<title>\"'.[/dim]"
        )
        return

    table = Table(
        title=f"🗣 Rooms ({len(summaries)})", caption=f"Storage: {RoomStore().storage_dir}"
    )
    table.add_column("Title", style="bold")
    table.add_column("Room ID", style="cyan", no_wrap=True)
    table.add_column("Agents", style="magenta")
    table.add_column("Humans")
    table.add_column("Rows", justify="right")
    table.add_column("Updated", style="dim")

    for s in summaries:
        table.add_row(
            escape(s.title) or "[dim](untitled)[/dim]",
            escape(s.room_id),
            escape(", ".join(s.agent_ids)) or "[dim]none[/dim]",
            escape(", ".join(s.human_ids)) or "[dim]none[/dim]",
            str(s.message_count),
            s.updated_at[:19].replace("T", " "),
        )
    console.print(table)


@room_app.command("show")
def room_show(
    room_id: str = typer.Argument(..., help="Room to render"),
    tail: int = typer.Option(0, "--tail", "-n", help="Show only the last N rows (0 for all)"),
) -> None:
    """Render a room's roster and transcript, membership rows included."""
    if tail < 0:
        # Silently rendering the whole room for `--tail -5` answers a different question
        # than the one asked, and looks like the room is shorter or longer than it is.
        _fail(f"--tail {tail} asks for a negative number of rows; use 0 for the whole room.")
    try:
        state = build_service().get(room_id)
    except UCloneXError as exc:
        _fail(str(exc))
        return

    roster = (
        ", ".join(f"{escape(p.id)} ({p.kind.value})" for p in state.participants) or "nobody yet"
    )
    console.print(
        Panel(
            f"[bold]{escape(state.title) or '(untitled)'}[/bold]\n"
            f"[dim]{escape(state.room_id)} · updated {state.updated_at}[/dim]\n"
            f"[bold]Roster:[/bold] {roster}\n"
            f"[bold]Responder:[/bold] "
            f"{escape(state.policy.default_responder_id) or '[dim]none[/dim]'}",
            title="🗣 Room",
            border_style="cyan",
        )
    )

    rows = state.transcript[-tail:] if tail > 0 else state.transcript
    if not rows:
        console.print("[dim]Nothing has been said yet.[/dim]")
        return
    for message in rows:
        console.print(_render_row(message))
        attribution = _render_attribution(message)
        if attribution:
            console.print(attribution)

    decision = state.last_decision
    if decision is not None and decision.verdict is SelectionVerdict.SILENCE:
        # A decided silence has no utterance to hang itself on, so it is the one outcome
        # that vanishes entirely unless it is rendered here. Without it a room that
        # concluded, a room whose ceiling was reached and a room whose selectors are
        # broken all present as a room that stopped.
        said_why = f": {escape(decision.reasoning)} " if decision.reasoning else " "
        console.print(
            f"[dim]— the room fell quiet{said_why}(decided by {escape(decision.selector)})[/dim]"
        )
        if decision.provenance is not None:
            console.print(f"[dim]  decided via {_served(decision.provenance)}[/dim]")

    # The remedy belongs beside the failure. Rendering the error alone leaves the user
    # where the record found them — retyping the question, which starts a fresh turn
    # budget for a turn the failure has already paid for.
    _offer_retry(state)


def _render_row(message: RoomMessage) -> str:
    """One transcript row, with a membership row visibly not somebody talking."""
    if not message.is_utterance:
        return f"[dim]{message.seq:>3} ·· {escape(message.content)}[/dim]"
    if message.error is not None:
        return (
            f"[bold red]{message.seq:>3} ✖ {escape(message.sender_id)}[/bold red] "
            f"[red]turn failed: {escape(message.error)}[/red]"
        )
    return (
        f"[bold cyan]{message.seq:>3} {escape(message.sender_id)}[/bold cyan]  "
        f"{escape(message.content)}"
    )


def _service_label(ref: ServiceRef) -> str:
    """`provider/model`, or the provider alone where the reference names no model."""
    return escape(f"{ref.provider}/{ref.model}" if ref.model else ref.provider)


def _served(provenance: Provenance) -> str:
    """The model that actually answered — never the one that was asked for.

    P6 makes the two legally different on a `primary` path (the provider-side alias case),
    and a head that rendered the request would report the model the operator *chose*
    rather than the one that served, which is the substitution P6 wants visible.
    """
    return _service_label(provenance.served_by)


def _render_attribution_of(provenance: Provenance | None) -> str:
    """`served by <model>`, naming the requested one too where the two diverged."""
    if provenance is None:
        return ""
    if provenance.degraded:
        return f"served by {_served(provenance)} (requested {_service_label(provenance.requested)})"
    return f"served by {_served(provenance)}"


def _render_attribution(message: RoomMessage) -> str:
    """The dim line under an utterance: which model answered, and who gave it the floor.

    Both halves, because they answer different questions and the transcript carries them
    separately (`Provenance` on the message, and on the `SpeakerDecision` beside it). The
    selector's own model is rendered as a bare `via <model>`: whether *it* was served
    something other than what was asked for is a second-order question, and nesting one
    parenthesised divergence inside another buries the one that matters.

    A failed turn keeps its line. Which model was serving *when it failed* is exactly what
    a reader of a failure wants, and dropping it would make the outcome the room records
    most carefully the one it explains least.
    """
    if not message.is_utterance:
        return ""
    parts = [p for p in (_render_attribution_of(message.provenance),) if p]
    decision = message.decision
    if decision is not None:
        chose = f"chosen by {escape(decision.selector)}"
        if decision.provenance is not None:
            chose += f" via {_served(decision.provenance)}"
        parts.append(chose)
    if not parts:
        return ""
    return f"[dim]      ↳ {' · '.join(parts)}[/dim]"


# --------------------------------------------------------------------------------------
# Roster
# --------------------------------------------------------------------------------------


@room_app.command("add")
def room_add(
    room_id: str = typer.Argument(..., help="Room to seat the participant in"),
    participant_id: str = typer.Argument(..., help="Agent id, or the human's user id"),
    human: bool = typer.Option(False, "--human", help="Seat a human rather than an agent"),
    display_name: str = typer.Option("", "--name", help="Shown in the transcript"),
    persona: str = typer.Option(
        "",
        "--persona",
        help="One line describing what this agent is for; it is a routing input, not decoration",
    ),
    alias: Annotated[
        list[str] | None,
        typer.Option("--alias", help="Extra name this participant answers to in a mention"),
    ] = None,
) -> None:
    """Seat a participant, and record the join in the transcript."""
    try:
        state = build_service().add_participant(
            room_id,
            participant_id,
            kind=ParticipantKind.HUMAN if human else ParticipantKind.AGENT,
            display_name=display_name,
            persona_summary=persona,
            aliases=tuple(alias or ()),
            persona=persona,
        )
    except UCloneXError as exc:
        _fail(str(exc))
        return

    seated = next(p for p in state.participants if p.id == participant_id)
    console.print(
        f"[bold green]✔[/bold green] {escape(participant_id)} joined [cyan]{escape(room_id)}[/cyan]"
        + (f" — session [dim]{escape(seated.session_id)}[/dim]" if seated.session_id else "")
    )


@room_app.command("remove")
def room_remove(
    room_id: str = typer.Argument(..., help="Room to remove the participant from"),
    participant_id: str = typer.Argument(..., help="Participant to unseat"),
) -> None:
    """Unseat a participant, and record the departure in the transcript."""
    try:
        state = build_service().remove_participant(room_id, participant_id)
    except UCloneXError as exc:
        _fail(str(exc))
        return

    console.print(
        f"[bold green]✔[/bold green] {escape(participant_id)} left [cyan]{escape(room_id)}[/cyan]\n"
        f"[dim]{escape(state.transcript[-1].content)}[/dim]"
    )


# --------------------------------------------------------------------------------------
# Talking
# --------------------------------------------------------------------------------------


@room_app.command("say")
def room_say(
    room_id: str = typer.Argument(..., help="Room to speak in"),
    sender_id: str = typer.Argument(..., help="Human participant speaking"),
    message: str = typer.Argument(..., help="What to say; @name addresses a participant"),
    provider: str | None = typer.Option(None, "--provider", help="LLM provider override"),
    model: str | None = typer.Option(None, "--model", help="Model for the room's agents"),
) -> None:
    """Post a human utterance and drive the agent turns it causes to a stop."""
    service = build_service()
    try:
        before = service.get(room_id)
    except UCloneXError as exc:
        _fail(str(exc))
        return

    orchestrator = _orchestrator_or_exit(policy=before.policy, provider=provider, model=model)
    try:
        after: RoomState = asyncio.run(orchestrator.post(room_id, sender_id, message))
    except RoomError as exc:
        _fail(str(exc))
        return
    except UCloneXError as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return

    _render_new_rows(before, after)
    fresh = [m for m in after.transcript if m.seq > len(before.transcript)]
    if not any(m.is_utterance and m.sender_id != sender_id for m in fresh):
        # Recorded silence is a decision, and saying nothing about it would present the
        # room as broken rather than as quiet — the conflation the verdicts exist to end.
        reason = (
            after.last_decision.reasoning
            if after.last_decision is not None
            else "no agent took the floor"
        )
        console.print(f"[dim]— the room stayed quiet: {escape(reason)}[/dim]")
    _offer_retry(after)


@room_app.command("retry")
def room_retry(
    room_id: str = typer.Argument(..., help="Room whose failed turn should be re-run"),
    provider: str | None = typer.Option(None, "--provider", help="LLM provider override"),
    model: str | None = typer.Option(None, "--model", help="Model for the room's agents"),
) -> None:
    """Re-run a turn that failed, without spending a fresh turn from the room's budget.

    The alternative the user had was to retype the question, which appends a second human
    message, resets the turn budget, and leaves the failure answered by a fresh exchange
    rather than by the turn that owed the answer.
    """
    service = build_service()
    try:
        before = service.get(room_id)
    except UCloneXError as exc:
        _fail(str(exc))
        return

    orchestrator = _orchestrator_or_exit(policy=before.policy, provider=provider, model=model)
    try:
        after: RoomState = asyncio.run(orchestrator.retry(room_id))
    except RoomError as exc:
        _fail(str(exc))
        return
    except UCloneXError as exc:
        _fail(f"{type(exc).__name__}: {exc}")
        return

    _render_new_rows(before, after)
    _offer_retry(after)


def _orchestrator_or_exit(
    *, policy: RoomPolicy, provider: str | None, model: str | None
) -> RoomOrchestratorProtocol:
    """Assemble the orchestrator, reporting a setup refusal instead of raising it.

    Assembly resolves a model client, so the ordinary ways this command is used wrongly —
    a mistyped `--provider`, no endpoint configured, a model this host cannot serve — all
    fail *here*, before the `try` that turns a `RoomError` into a message. Left outside
    it, they reached the user as a traceback, which is the one thing this module claims
    not to do.
    """
    try:
        return build_orchestrator(store=RoomStore(), policy=policy, provider=provider, model=model)
    except UCloneXError as setup_failure:
        _fail(str(setup_failure))
        raise  # unreachable: `_fail` exits.


def _render_new_rows(before: RoomState, after: RoomState) -> None:
    """Print whatever appeared in the transcript between two reads of the room."""
    for row in (m for m in after.transcript if m.seq > len(before.transcript)):
        console.print(_render_row(row))
        attribution = _render_attribution(row)
        if attribution:
            console.print(attribution)


def _offer_retry(state: RoomState) -> None:
    """Name the remedy when the room's last utterance is a failed turn."""
    last = state.last_utterance
    if last is not None and last.error is not None:
        console.print(f"[dim]— retry that turn with:[/dim] ucx room retry {escape(state.room_id)}")
