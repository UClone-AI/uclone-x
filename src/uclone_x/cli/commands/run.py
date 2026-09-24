"""Interactive Terminal Chat REPL and single-turn runner for UClone-X BaseAgent (Issue #54)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from uclone_x.agent.loop import (
    LoopJob,
    LoopScheduler,
    LoopStatus,
    LoopTickResult,
    parse_loop_command_input,
)
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    AgentState,
    TurnResult,
)
from uclone_x.agent.prompts import compose_system_prompt
from uclone_x.agent.session import SessionStore
from uclone_x.core.agent_home import AgentHomeError
from uclone_x.core.failure_journal import record_failure
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import (
    LLMError,
    MissingDependencyError,
    PathTraversalError,
    SessionIdCollisionError,
)
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.llm.models import MessageRole
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.sandbox.models import NoIsolation, WorkspaceIsolation
from uclone_x.telemetry import TelemetryTracer, create_telemetry_exporter
from uclone_x.tools.protocols import ToolRegistryProtocol
from uclone_x.tools.registry import create_default_registry

logger = logging.getLogger(__name__)
# Every value this module interpolates into a Rich string that the code did not write --
# model replies, the user's messages, agent names, session ids, model names, tool names and
# descriptions, exception text -- goes through `escape()` (#960). Values the code does write
# (a connector's `provider_name`, enum values, counts) are interpolated as they are. Rich
# reads `[...]` as markup, so `arr[i]` printed as `arr`, and an unbalanced `[/bold]` raised
# `MarkupError` after a turn that had succeeded. `emoji=False` for the same reason:
# `:thumbs_up:` in a reply is text, not a request for a glyph.
console = Console(emoji=False)
# Everything that is not the agent's reply. A script reads the reply from stdout, so a
# failure printed there is indistinguishable from an answer that happens to quote one.
err_console = Console(stderr=True, emoji=False)

FAILED_TURN_EXIT_CODE = 1
"""Exit status of `ucx run --prompt` when the turn failed (#953).

The same status the command already used for an LLM setup error, and the one the room
commands use for a refusal. Documented in `docs/cli-specification.md`.
"""

UNSAVED_SESSION_EXIT_CODE = 3
"""Exit status when the turn answered but the session could not be saved (#957).

The reply is already on stdout, so a caller must be able to tell this apart both from a
failed turn (1: there is no reply) and from success (0): the reply is real, but a later
`--session-id` call will not resume it. Exit 0 here was a silent loss (P6). A turn that
failed *and* was not saved exits `FAILED_TURN_EXIT_CODE`; the missing reply is the larger
fact, and both reports are on stderr.
"""

UNWRITTEN_REPLY_EXIT_CODE = 1
"""Exit status when the turn answered and was saved, but the reply could not be written (#970).

A closed pipe (`BrokenPipeError`), any other error writing to the descriptor (`OSError`,
#996), or a stdout encoding that cannot represent the reply (`UnicodeEncodeError`). If the
save failed too, the status is still this one. It shares `1` with a failed turn: either way
a script received no complete reply, and `1` is the status this case already had, as an unhandled
traceback, before the failure was caught. Whether it deserves a status of its own is left
open on #970. What changed is that the session is saved before the write, so `--session-id`
resumes the turn.
"""


#: The model `ucx run` falls back to when `--model` names none. One name for what was
#: three copies of `model or "qwen3:8b"`, which had already drifted out of the `--model`
#: help text ("default depends on provider").
DEFAULT_CLI_MODEL = "qwen3:8b"


def _report_failed_turn(failure: str) -> None:
    """Report a failed turn on stderr, labelled so it cannot be read as agent output.

    Escaped because the text is an exception message, and Rich reads `[...]` in it as
    markup: an unbalanced tag would raise `MarkupError` inside the error path.
    """
    err_console.print(f"[bold red]✖ Execution failed:[/bold red] {escape(failure)}")


def _report_blocked_turn(reason: str) -> None:
    """Report a turn a `PRE_TURN` hook refused, naming the hook as the cause.

    Under the generic `Execution failed:` label, a hook's own reason (`PII detected in
    prompt`) did not say that a hook, rather than the model or the provider, refused.
    """
    err_console.print(f"[bold red]✖ Turn blocked by hook:[/bold red] {escape(reason)}")


def _blocked_by_hook(result: TurnResult) -> bool:
    """Whether a `PRE_TURN` hook refused the turn, read from its structured stop reason.

    Not from `content`: until #970 this compared `content` with the engine's sentence
    `Turn execution blocked by hook: <reason>`, so rewording that sentence in `agent/base.py`
    relabelled every hook block, and a failure whose content read alike became one.
    """
    return result.stop_reason == "blocked_by_hook"


def _report_unsaved_session(save_error: str) -> None:
    """Report on stderr that the session was not written, after a turn that answered."""
    err_console.print(
        f"[bold red]✖ Session not saved:[/bold red] {escape(save_error)}\n"
        "[dim]The reply was not recorded; --session-id will not resume it.[/dim]"
    )


def _report_interrupted_save(label: str) -> None:
    """Report that an interrupt landed inside a session save, which `130` alone hid (#970).

    The status said "interrupted", not that the save never finished. `SessionStore` writes
    atomically, so what stands is the session as of its last completed save.
    """
    err_console.print(
        f"[bold red]✖ {label}:[/bold red] interrupted before the save completed\n"
        "[dim]Saves are atomic: the session is as it was at its last completed save.[/dim]"
    )


#: Control characters rendered as visible carets before a reply reaches a terminal (#1282),
#: built once. C0 except `\n` and `\t` -- the two a reply legitimately uses for structure --
#: plus DEL and C1.
#:
#: The caret forms are the conventional ones (`cat -v`, `stty`): a C0 byte `c` shows as
#: `^` + `chr(c + 0x40)`, so ESC is `^[`, BEL is `^G` and CR is `^M`; DEL is `^?`. A C1 byte
#: is shown as the 7-bit escape sequence it is equivalent to -- `0x9B` (CSI) as `^[[` -- so a
#: sequence written in 8-bit form is displayed the same way as the same sequence written in
#: 7-bit form, rather than as a different-looking thing the reader has to recognise separately.
_CARET_TRANSLATION: dict[int, str] = {
    **{c: f"^{chr(c + 0x40)}" for c in range(0x00, 0x20) if chr(c) not in "\n\t"},
    0x7F: "^?",
    **{c: f"^[{chr(c - 0x40)}" for c in range(0x80, 0xA0)},
}


def _neutralise_control_sequences(text: str) -> str:
    """Render control characters as visible carets so a terminal displays them, not obeys them.

    A reply is model output, and on a tool-using turn it can contain text a third party put
    there. Written raw to a terminal, it decides what that terminal does: the measured case
    is OSC 52, which writes the user's clipboard, but the family also covers CSI cursor moves
    that overwrite what is already on screen and CR that rewrites the current line (#1282).

    Neutralising rather than deleting is deliberate. The bytes are still visible as `^[`,
    `^G`, `^M`, so a reader can see that something was there and what it was -- a silent
    deletion would hide a manipulation attempt as effectively as executing it hides itself.

    `\\n` and `\\t` are kept: they are how a reply is laid out, not how it is weaponised, and
    a terminal renders both as whitespace rather than as a command.
    """
    return text.translate(_CARET_TRANSLATION)


def _stdout_is_a_terminal() -> bool:
    """Whether stdout is a terminal, answering `False` when the descriptor cannot say.

    `isatty()` raises `ValueError` on a closed stream and `OSError` on a descriptor that
    cannot be queried, and `AttributeError` when `sys.stdout` has been replaced by an object
    that does not implement it at all -- which is not hypothetical: the double in
    `test_an_interrupt_while_the_reply_is_written_leaves_the_turn_saved` is one. All three
    are the same answer: this stdout cannot say.

    This is asked on the paths that exist precisely because stdout can be broken (#996), so
    the question must be total. `False` is the safe default in both directions: it keeps the
    byte-fidelity guarantee a redirecting caller depends on, and it cannot turn a reply that
    was going to be written into an unhandled exception on the way out.
    """
    try:
        return sys.stdout.isatty()
    except (OSError, ValueError, AttributeError):  # `io.UnsupportedOperation` subclasses two
        return False


def _write_reply(reply_text: str) -> str | None:
    """Write the reply to stdout as plain text; return why it could not be, or `None`.

    Plain bytes, not Rich (#960). Escaping alone is not enough for a script: Rich also
    wraps at the console width (80 columns when stdout is a pipe) and expands tabs, so the
    reply on stdout would still not be the text the model wrote.

    **A terminal is a renderer; a pipe is a byte channel.** On a terminal the reply's control
    characters are neutralised first (#1282), because there they are instructions the terminal
    obeys. Through a pipe nothing is touched: a caller redirecting `ucx run` into a file or
    another program is entitled to the exact bytes, and rewriting them would be data
    corruption. The two requirements only conflict if the behaviour is unconditional.
    """
    if _stdout_is_a_terminal():
        # Spelled with the keyword to keep this call textually distinct from the REPL's, so
        # that each site's `Killed by:` declaration names a substring unique in this file --
        # `./swx mutate` refuses an anchor that matches twice, and the two sites are pinned
        # by different tests.
        reply_text = _neutralise_control_sequences(text=reply_text)
    try:
        sys.stdout.write(f"{reply_text}\n")
        sys.stdout.flush()
    except (OSError, UnicodeError) as write_err:
        # A reader that went away, a descriptor that cannot be written, or an encoding that
        # cannot hold the reply. The session was saved before this write (#970); what is
        # lost is this copy of the reply. Any `OSError`, not only `BrokenPipeError` (#996):
        # `EBADF`, `EIO` and `ENOSPC` leave the reply buffered just the same.
        if isinstance(write_err, OSError):
            _discard_unflushable_stdout()
        return f"{type(write_err).__name__}: {write_err}"
    return None


def _discard_unflushable_stdout() -> None:
    """Point file descriptor 1 at `os.devnull` after a write to it failed.

    The reply is still in stdout's buffer, and the interpreter flushes that buffer again at
    shutdown -- into the same failure, which prints `Exception ignored ... BrokenPipeError`
    (or `OSError`) and changes the exit status to 120. For `EPIPE` this is the remedy the
    Python documentation gives for `SIGPIPE`; a descriptor opened for reading fails the same
    way with `EBADF` (#996). An encoding error leaves nothing buffered, so it needs no
    redirect. A stdout with no descriptor (a test runner's buffer) has nothing to redirect.
    """
    try:
        descriptor = sys.stdout.fileno()
    except (OSError, ValueError):  # `io.UnsupportedOperation` subclasses both
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, descriptor)
    finally:
        os.close(devnull)


def _report_unwritten_reply(write_error: str, *, saved: bool) -> None:
    """Report on stderr that the reply did not reach stdout, and whether the turn was kept."""
    note = "\n[dim]The turn was saved; --session-id resumes it.[/dim]" if saved else ""
    err_console.print(
        f"[bold red]✖ Reply not written to stdout:[/bold red] {escape(write_error)}{note}"
    )


def get_default_llm(provider: str | None = None) -> LLMProviderProtocol:
    """Resolve an LLM connector, honouring the environment when no provider is named.

    `None` means "resolve from configuration", which is what the factory's precedence is
    for: `LLM_PROVIDER`, then a credential variable, then an Ollama endpoint, then a named
    refusal. The default was the literal `"ollama"` (#539 follow-up), and that had two
    costs. The refusal added for #533 was unreachable from `./ucx run`, so the fix was true
    of the library and false of the product. And the CLI ignored `LLM_PROVIDER` and every
    credential variable outright: a user who exported `OPENAI_API_KEY` still got Ollama.
    """
    return create_llm_connector(provider=provider, fallback_to_mock=False)


async def run_agent_repl_async(
    agent_name: str = "default",
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    prompt: str | None = None,
    tools: ToolRegistryProtocol | None = None,
    session_id: str | None = None,
    reset: bool = False,
    compact: bool = False,
    store: SessionStore | None = None,
    isolation: str = "workspace",
    workspace_dir: str | Path | None = None,
) -> None:
    """Asynchronous interactive chat loop or single-turn runner with BaseAgent.

    `session_id`, `reset` and `compact` implement #183 requirement 4's CLI surface. The
    session is persisted through the Core `SessionStore` (P8), so `--session-id` resumes
    the conversation that ID had rather than starting a fresh one that merely shares a
    name.
    """
    bus = EventBus()
    llm = create_llm_connector(provider=provider, fallback_to_mock=False)
    active_tools = tools if tools is not None else create_default_registry()
    exporter = create_telemetry_exporter()
    tracer = TelemetryTracer()

    effective_model = model or DEFAULT_CLI_MODEL

    # `compose_system_prompt`, not a hand-rolled sentence (#1206): the composed default
    # is the only place `HONEST_REPORTING` / `ARTIFACT_REPORTING` / `IMAGE_GENERATION` /
    # `ENVIRONMENT_REPAIR` are defined, and every non-CLI entry point (the web UI,
    # persona registry) already uses it.
    #
    # `model_name` here does not decide the steerability framing and no test can pin it
    # to `effective_model` rather than `model`: `BaseAgent._system_prompt` re-frames the
    # prompt through `adapt_system_prompt(base, llm_config.model_name)` on every turn, so
    # the family framing always follows the model that serves. It is passed for the same
    # reason the constant exists -- one name for the model this run uses.
    default_system = system_prompt or compose_system_prompt(model_name=effective_model)

    isolation_policy = NoIsolation() if isolation == "none" else WorkspaceIsolation()
    config = AgentConfig(
        agent_id=agent_name,
        name=agent_name,
        system_prompt=default_system,
        llm_config=AgentLLMConfig(
            model_name=effective_model,
            temperature=0.7,
            max_tokens=2048,
        ),
        isolation=isolation_policy,
    )

    session_store = store if store is not None else SessionStore()
    # `is None`, not `or`: an omitted flag means "the agent's default session", while
    # `--session-id ""` is a bad argument and must reach the name rule rather than being
    # silently redirected at the default. Same idiom the Core routes were corrected for.
    effective_workspace = (
        Path(workspace_dir).resolve() if workspace_dir is not None else Path.cwd().resolve()
    )
    context = AgentContext(
        session_id=f"sess_{agent_name}" if session_id is None else session_id,
        agent_id=agent_name,
        workspace_root=effective_workspace,
    )
    from uclone_x.agent.composition import HostDependencies, compose_agent
    from uclone_x.memory.store import default_cross_session_memory

    host = HostDependencies(
        bus=bus,
        llm=llm,
        tools=active_tools,
        tracer=tracer,
        store=session_store,
        # P7's durable facts are worth nothing to a head that never wires the store:
        # `BaseAgent` registers the memory tools only when it is given one.
        memory=default_cross_session_memory(agent_name),
    )
    agent = compose_agent(config=config, host=host, context=context)

    # With `--prompt`, stdout carries the reply and nothing else (#960), so the notices a
    # person reads -- resumed, reset, compacted, the spinner -- go to stderr. The REPL is
    # read by a person only, and keeps everything on stdout.
    notices = err_console if prompt else console

    # Resume before starting: `switch_session` and the bus subscription are bound at
    # `start()`, and a hydrate that replaced history afterwards would race the loop.
    hydrated = agent.hydrate_session()
    if hydrated is not None:
        notices.print(
            f"[dim]Resumed session [bold]{escape(hydrated.session_id)}[/bold] — "
            f"{len(hydrated.messages)} message(s), {hydrated.turn_counter} turn(s).[/dim]"
        )
    if reset:
        reset_state = agent.reset_session()
        notices.print(
            f"[yellow]✔ Session '{escape(reset_state.session_id)}' reset before start "
            f"({len(reset_state.messages)} anchored message(s)).[/yellow]"
        )

    await agent.start()

    if compact:
        compaction = await agent.compact_session(reason="cli_startup_flag")
        notices.print(
            f"[yellow]✔ Compacted '{escape(compaction.session_id)}' "
            f"via {compaction.ledger_source.value} ledger: "
            f"{compaction.messages_before} -> {compaction.messages_after} messages "
            f"({compaction.compression_ratio_pct}% est. tokens saved).[/yellow]"
        )

    # Non-interactive single-shot execution mode
    if prompt:
        failure: str | None = None
        blocked = False
        reply_text = ""
        with err_console.status(
            f"[bold green]{escape(agent_name)}[/bold green] is reasoning...", spinner="dots"
        ):
            try:
                async with tracer.agent_turn_span(
                    agent_id=agent_name,
                    session_id=f"sess_{agent_name}",
                    turn_index=1,
                    extra_attributes={
                        "mode": "single-shot",
                        "provider": llm.provider_name,
                        "model": effective_model,
                    },
                ):
                    result = await agent.execute_turn(prompt)
                # `execute_turn` returns most failures rather than raising them: an
                # unreachable provider arrives as `error` with empty `content`, which
                # printed `(No response content)` as the reply and exited 0.
                failure = result.error
                blocked = _blocked_by_hook(result)
                if failure is None:
                    reply_text = result.content
            except MissingDependencyError:  # the launcher reports it, not this turn
                # `uclone_x.shells.entry` names the extra to install; relabelling it as a
                # turn failure would hide which one.
                raise
            except Exception as e:
                # A failed turn is the failure a user would want to report, and
                # the one they cannot reconstruct afterwards. Recorded only with
                # consent; a no-op otherwise, and it never raises.
                record_failure(e, context={"surface": "cli.run", "single_shot": True})
                logger.exception("Single-shot turn failed for agent %s", agent_name)
                failure = f"{type(e).__name__}: {e}"
            finally:
                completed = tracer.get_completed_spans()
                if completed:
                    # Single-shot: the process exits next, so draining is not what
                    # matters here -- symmetry with the REPL branch is, so a failed
                    # export is accounted for the same way in both.
                    try:
                        await exporter.export_spans(completed)
                    except Exception as export_err:
                        logger.warning("Failed to export telemetry spans: %s", export_err)
                        tracer.drop_unexported(completed, "cli_export_failed")
                    else:
                        tracer.discard_exported(completed)
        # Persist here too, or `--session-id` would be meaningless across successive
        # single-shot invocations — which is the main way a script drives the CLI. And
        # persist *before* the reply is written (#970): the write can fail on its own -- a
        # reader that closed the pipe, an encoding that cannot hold the reply -- and while
        # it came first, that failure left before the save and took an answered turn with
        # it, the same class of loss as #960's `MarkupError`.
        # A failure is reported *before* the save, though (#996): a second Ctrl+C can land
        # inside the save, and the interrupt leaves without reaching anything below it.
        if failure is not None and blocked:
            _report_blocked_turn(failure)
        elif failure is not None:
            # Never where the reply goes, and never into the session as the agent's
            # message: `--session-id` would feed it back to the model as something it said.
            _report_failed_turn(failure)
        save_failure: str | None = None
        try:
            agent.persist_session()
        except KeyboardInterrupt:
            _report_interrupted_save("Session not saved")
            raise
        except Exception as persist_err:
            logger.warning("Failed to persist session state: %s", persist_err)
            save_failure = f"{type(persist_err).__name__}: {persist_err}"
            _report_unsaved_session(save_failure)
        write_failure: str | None = None
        if failure is None:
            err_console.print(f"\n[bold green]{escape(agent_name)}[/bold green]:")
            if reply_text:
                write_failure = _write_reply(reply_text)
                if write_failure is not None:
                    _report_unwritten_reply(write_failure, saved=save_failure is None)
            else:
                # Not `(No response content)` on stdout: a script would read the
                # placeholder as the reply.
                err_console.print("[dim](No response content)[/dim]")
        await agent.stop()
        if failure is not None:
            # Exit 0 here was the whole of #953: a script cannot tell a failed turn from
            # an answer by anything but the status (P6).
            raise typer.Exit(code=FAILED_TURN_EXIT_CODE)
        if write_failure is not None:
            # No reply reached the script. That outranks an unsaved session, as it does
            # for a failed turn; both reports are already on stderr.
            raise typer.Exit(code=UNWRITTEN_REPLY_EXIT_CODE)
        if save_failure is not None:
            # The reply stands, but the next `--session-id` call will not see it (#957).
            raise typer.Exit(code=UNSAVED_SESSION_EXIT_CODE)
        return

    # Interactive REPL mode
    console.print(
        Panel(
            f"[bold cyan]⚡ UClone-X Autonomous Agent REPL[/bold cyan]\n"
            f"[dim]Agent ID:[/dim] [green]{escape(agent_name)}[/green] | "
            f"[dim]Provider:[/dim] [yellow]{llm.provider_name}[/yellow] | "
            f"[dim]Model:[/dim] [magenta]{escape(model or 'auto')}[/magenta]\n\n"
            f"[dim]Commands: [bold]/exit[/bold] (quit), [bold]/reset[/bold] (clear memory), "
            f"[bold]/compact[/bold] (compact context), "
            f"[bold]/history[/bold] (view log), [bold]/status[/bold] (state)[/dim]",
            border_style="cyan",
        )
    )

    def _on_loop_tick(job: LoopJob, result: LoopTickResult) -> None:
        if result.skipped:
            console.print(
                f"\n[dim]⏱ [Loop {escape(job.job_id)}] Skipped tick: previous run still active.[/dim]"
            )
            return
        status_color = "green" if result.success else "red"
        console.print(
            f"\n[bold yellow]⏱ [Loop {escape(job.job_id)} Tick #{result.tick_index}][/bold yellow] "
            f"[dim]({result.duration_seconds:.1f}s)[/dim]"
        )
        if result.success:
            console.print(
                f"[{status_color}]{escape(result.content or '(No response content)')}[/{status_color}]"
            )
        else:
            console.print(
                f"[{status_color}]✖ Error: {escape(result.error or 'unknown error')}[/{status_color}]"
            )
        if job.status == LoopStatus.COMPLETED:
            console.print(f"[bold green]✔ Loop '{escape(job.job_id)}' completed.[/bold green]")
        elif job.status == LoopStatus.FAILED:
            console.print(
                f"[bold red]✖ Loop '{escape(job.job_id)}' terminated due to consecutive failures.[/bold red]"
            )

    loop_scheduler = LoopScheduler(agent, on_tick_completed=_on_loop_tick)

    turn_count = 0
    session_unsaved = False
    try:
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                break

            clean_input = user_input.strip()
            if not clean_input:
                continue

            # Handle slash commands
            if clean_input in ("/exit", "/quit", "exit", "quit"):
                break
            elif clean_input == "/help":
                console.print(
                    "[bold]Available REPL Commands:[/bold]\n"
                    "  • [cyan]/status[/cyan]            - View active agent state machine status\n"
                    "  • [cyan]/history[/cyan]           - Print full conversation history turns\n"
                    "  • [cyan]/tools[/cyan]             - List registered tools and MCP servers\n"
                    "  • [cyan]/ontology[/cyan]          - View active ontology knowledge graph\n"
                    "  • [cyan]/reset[/cyan]             - Clear session dialogue memory\n"
                    "  • [cyan]/compact[/cyan]           - Compact session context now (P5)\n"
                    "  • [cyan]/loop <intvl> <cmd>[/cyan] - Schedule recurring prompt (e.g. /loop 5m check tests)\n"
                    "  • [cyan]/loop list[/cyan]           - List active recurring loops\n"
                    "  • [cyan]/loop stop <id>[/cyan]       - Stop recurring loop (or '/loop stop all')\n"
                    "  • [cyan]/help[/cyan]              - Display this command manual\n"
                    "  • [cyan]/exit[/cyan]              - Terminate REPL session\n"
                )
                continue
            elif clean_input == "/loop" or clean_input == "/loop help":
                console.print(
                    "[bold]Usage for /loop (Recurring execution):[/bold]\n"
                    "  • [cyan]/loop <interval> <prompt>[/cyan] - Start loop (e.g. '/loop 5m ./ucx test check' or '/loop 5분마다 테스트 돌려줘')\n"
                    "  • [cyan]/loop list[/cyan]                 - List active recurring loops\n"
                    "  • [cyan]/loop stop <job_id>[/cyan]        - Stop specific loop (or '/loop stop all')\n"
                    "  • [cyan]/loop status <job_id>[/cyan]      - Inspect loop execution details\n"
                )
                continue
            elif clean_input == "/loop list":
                jobs = loop_scheduler.list_jobs()
                if not jobs:
                    console.print("[dim]No active recurring loops.[/dim]")
                else:
                    l_table = Table(title="⏱ Recurring Loops")
                    l_table.add_column("Job ID", style="bold cyan")
                    l_table.add_column("Interval", justify="right")
                    l_table.add_column("Runs", justify="right")
                    l_table.add_column("Status")
                    l_table.add_column("Prompt")
                    for j in jobs:
                        status_style = "green" if j.status == LoopStatus.ACTIVE else "dim"
                        l_table.add_row(
                            escape(j.job_id),
                            f"{j.interval_seconds:.0f}s",
                            str(j.runs_count),
                            f"[{status_style}]{j.status.value}[/{status_style}]",
                            escape(j.prompt[:40] + ("..." if len(j.prompt) > 40 else "")),
                        )
                    console.print(l_table)
                continue
            elif clean_input.startswith("/loop stop"):
                parts = clean_input.split(maxsplit=2)
                target = parts[2].strip() if len(parts) > 2 else ""
                if not target:
                    console.print("[yellow]Usage: /loop stop <job_id> or /loop stop all[/yellow]")
                elif target == "all":
                    stopped = loop_scheduler.cancel_all()
                    console.print(f"[yellow]✔ Stopped {stopped} active loop(s).[/yellow]")
                else:
                    if loop_scheduler.cancel_job(target):
                        console.print(f"[yellow]✔ Loop '{escape(target)}' cancelled.[/yellow]")
                    else:
                        console.print(
                            f"[red]✖ Loop '{escape(target)}' not found or already inactive.[/red]"
                        )
                continue
            elif clean_input.startswith("/loop status"):
                parts = clean_input.split(maxsplit=2)
                target = parts[2].strip() if len(parts) > 2 else ""
                if not target:
                    console.print("[yellow]Usage: /loop status <job_id>[/yellow]")
                else:
                    job = loop_scheduler.get_job(target)
                    if not job:
                        console.print(f"[red]✖ Loop '{escape(target)}' not found.[/red]")
                    else:
                        s_table = Table(show_header=False, box=None)
                        s_table.add_row("Job ID:", escape(job.job_id))
                        s_table.add_row("Status:", escape(job.status.value))
                        s_table.add_row("Interval:", f"{job.interval_seconds:.0f}s")
                        s_table.add_row("Runs:", str(job.runs_count))
                        s_table.add_row("Prompt:", escape(job.prompt))
                        if job.last_result:
                            s_table.add_row(
                                "Last Run Duration:", f"{job.last_result.duration_seconds:.1f}s"
                            )
                            s_table.add_row("Last Run Success:", str(job.last_result.success))
                        console.print(
                            Panel(
                                s_table, title=f"Loop Status: {escape(target)}", border_style="dim"
                            )
                        )
                continue
            elif clean_input.startswith("/loop "):
                raw_arg = clean_input[len("/loop ") :].strip()
                try:
                    intvl, loop_prompt = parse_loop_command_input(raw_arg)
                    new_job = loop_scheduler.add_job(interval_seconds=intvl, prompt=loop_prompt)
                    console.print(
                        f"[bold green]✔ Started recurring loop [cyan]{escape(new_job.job_id)}[/cyan] "
                        f"(every {intvl:.0f}s):[/bold green] {escape(loop_prompt)}"
                    )
                except ValueError as val_err:
                    console.print(
                        f"[bold red]✖ Failed to create loop:[/bold red] {escape(str(val_err))}"
                    )
                continue
            elif clean_input == "/reset":
                # Was: `agent._history = [ChatMessage(SYSTEM, config.system_prompt or "")]`
                # behind a `reportPrivateUsage` pragma. Three defects went with it — an
                # empty SYSTEM message when no prompt was configured (a state
                # `__init__` never produces), neither turn counter reset, and a fourth
                # reset semantics that no other caller shared. `reset_session` is the
                # Core API and the single semantics (#183).
                state = agent.reset_session()
                turn_count = 0
                console.print(
                    f"[yellow]✔ Session '{escape(state.session_id)}' reset "
                    f"({len(state.messages)} anchored message(s), turn counter 0).[/yellow]"
                )
                continue
            elif clean_input == "/compact":
                with console.status("[bold green]Compacting context...", spinner="dots"):
                    compaction = await agent.compact_session(reason="cli_on_demand")
                console.print(
                    f"[yellow]✔ Compacted '{escape(compaction.session_id)}' via "
                    f"{compaction.ledger_source.value} ledger: "
                    f"{compaction.messages_before} -> {compaction.messages_after} messages, "
                    f"{compaction.tokens_before} -> {compaction.tokens_after} est. tokens "
                    f"({compaction.compression_ratio_pct}% saved).[/yellow]"
                )
                continue
            elif clean_input == "/status":
                status_table = Table(show_header=False, box=None)
                status_table.add_row("State:", f"[green]{agent.state.value}[/green]")
                status_table.add_row("Agent ID:", escape(agent.agent_id))
                status_table.add_row("Session ID:", escape(agent.session_id))
                status_table.add_row("History Messages:", str(len(agent.history)))
                status_table.add_row("Turns Executed:", str(agent.get_session().turn_counter))
                console.print(Panel(status_table, title="Agent Status", border_style="dim"))
                continue
            elif clean_input == "/history":
                for idx, msg in enumerate(agent.history):
                    role_color = "cyan" if msg.role == MessageRole.USER else "green"
                    # The label is escaped too: `[user #0]` is itself valid tag syntax, and
                    # Rich swallowed it, so no line of `/history` ever showed its role.
                    label = escape(f"[{msg.role.value} #{idx}]")
                    console.print(
                        f"[{role_color}]{label}[/{role_color}] {escape(str(msg.content))}"
                    )
                continue
            elif clean_input == "/tools":
                registered = active_tools.list_tools()
                if not registered:
                    console.print("[dim]No tools currently registered in tool registry.[/dim]")
                else:
                    t_table = Table(title="🔧 Registered Tools")
                    t_table.add_column("Tool Name", style="bold cyan")
                    t_table.add_column("Description")
                    for t in registered:
                        t_table.add_row(escape(t.name), escape(t.description))
                    console.print(t_table)
                continue
            elif clean_input == "/ontology":
                console.print("[bold purple]🧠 Ontology Knowledge Graph:[/bold purple]")
                console.print(f"  • Agent Domain: [cyan]{escape(agent_name)}[/cyan]")
                console.print(
                    "  • Tiers: [green]Asserted[/green], [yellow]Induced Enforcing[/yellow], [blue]Candidate[/blue]"
                )
                continue

            turn_count += 1
            failure_in_turn: str | None = None
            blocked = False
            reply_text = ""
            with console.status(
                f"[bold green]{escape(agent_name)}[/bold green] is reasoning...", spinner="dots"
            ):
                try:
                    async with tracer.agent_turn_span(
                        agent_id=agent_name,
                        session_id=f"sess_{agent_name}",
                        turn_index=turn_count,
                        extra_attributes={
                            "mode": "interactive",
                            "provider": llm.provider_name,
                            "model": effective_model,
                        },
                    ):
                        result = await agent.execute_turn(clean_input)
                    failure_in_turn = result.error
                    blocked = _blocked_by_hook(result)
                    if failure_in_turn is None:
                        reply_text = result.content or "(No response content)"
                except MissingDependencyError:  # as in single-shot mode: the launcher's
                    raise
                except Exception as e:
                    record_failure(e, context={"surface": "cli.run", "single_shot": False})
                    logger.exception("REPL turn failed for agent %s", agent_name)
                    failure_in_turn = f"{type(e).__name__}: {e}"
                finally:
                    completed = tracer.get_completed_spans()
                    if completed:
                        # The buffer must be drained every turn or it grows for the life
                        # of the REPL, but `tracer.clear()` used to run here whether or
                        # not the export succeeded, so a collector outage silently threw
                        # the turn's spans away behind an export warning (#187).
                        # `drop_unexported` still drops -- at-most-once is the deliberate
                        # choice over re-sending spans the collector may have taken --
                        # but the loss is now counted on `tracer.buffer_evicted_span_count`
                        # instead of vanishing.
                        try:
                            await exporter.export_spans(completed)
                        except Exception as export_err:
                            logger.warning("Failed to export telemetry spans: %s", export_err)
                            tracer.drop_unexported(completed, "cli_export_failed")
                        else:
                            tracer.discard_exported(completed)

            if failure_in_turn is None:
                # Neutralised unconditionally here, and *before* `escape()` (#1282). The REPL
                # prompts and reads, so its output is always for a person; unlike `--prompt`
                # it is already not byte-faithful (Rich wraps and styles it), so no caller can
                # depend on these bytes and there is no pipe case to preserve.
                #
                # Until #1282 this line's protection was an accident: Rich's markup escaping
                # and highlighter broke the sequence up, so the same OSC 52 payload arrived as
                # `\x1b\x1b[1m]\x1b[0m\x1b[1;36m52\x1b[0m;c;…` -- mangled, but by nothing that
                # promised to. `highlight=False`, a highlighter change, or a move off
                # `console.print` would have removed it with nothing failing.
                #
                # The order is load-bearing and is the reason this is a separate statement:
                # neutralising turns `\x1b` into `^[`, which *introduces* a `[` that the
                # preceding `escape()` never saw. Escaping first and neutralising after feeds
                # that fresh `[` to Rich as the start of a markup tag. It only bites when what
                # follows the ESC parses as a tag name, which is why a single payload does not
                # show it: `\x1b]52;…` comes out the same either way, but `\x1bbold]X` renders
                # as `^[bold]X` in this order and as `^X` in the other -- Rich swallowing the
                # marker and the text after it. Pinned by
                # `test_the_repl_neutralises_before_escaping_so_rich_cannot_eat_the_caret`.
                reply_text = _neutralise_control_sequences(reply_text)
                console.print(
                    f"\n[bold green]{escape(agent_name)}[/bold green]:\n{escape(reply_text)}"
                )
            elif blocked:
                _report_blocked_turn(failure_in_turn)
            else:
                # The REPL stays alive -- the person at it can retry -- but the failure is
                # labelled as one and not printed as the agent's reply. It was printed
                # twice before #953, the second time as `[Fallback Error]` under the
                # agent's name.
                _report_failed_turn(failure_in_turn)

            # Persist per turn, not only in the `finally`. A REPL killed mid-session, or
            # one whose process dies, otherwise loses every turn since it started —
            # and `execute_turn` deliberately does not persist, so nothing else would.
            try:
                agent.persist_session()
            except Exception as persist_err:
                # Best-effort per turn, and exit-status-neutral: the persist on the way
                # out retries it, and that one decides the exit status.
                logger.warning("Failed to persist session after a turn: %s", persist_err)
                err_console.print(
                    f"[yellow]⚠ Turn completed but the session was not saved: "
                    f"{escape(str(persist_err))}[/yellow]"
                )

            if agent.state == AgentState.ERROR:
                agent.transition_to(AgentState.IDLE)

    finally:
        loop_scheduler.cancel_all()
        # Persist before stopping. A REPL session that vanished on exit was the whole
        # reason `--session-id` would otherwise be decorative.
        try:
            agent.persist_session()
        except KeyboardInterrupt:
            # Keeps its 130 (an exception already leaving the REPL keeps its status), but
            # 130 alone did not say this save never finished (#970). Under `asyncio.run` a
            # first Ctrl+C lands at the next `await`, after the save; a second lands here.
            _report_interrupted_save("Session not saved on exit")
            raise
        except Exception as persist_err:
            logger.warning("Failed to persist session state: %s", persist_err)
            err_console.print(
                f"[bold red]✖ Session not saved on exit:[/bold red] "
                f"{escape(f'{type(persist_err).__name__}: {persist_err}')}"
            )
            session_unsaved = True
        await agent.stop()
        console.print(
            f"\n[dim]Session finished. Completed [bold]{turn_count}[/bold] turns with "
            f"[bold]{escape(agent_name)}[/bold].[/dim]"
        )
    # After the `finally`, so an exception already leaving the loop keeps its own status.
    if session_unsaved:
        # The same fact as a single-shot run's unsaved turn (#957): whatever changed since
        # the last successful save is gone, and a later `--session-id` will not see it.
        raise typer.Exit(code=UNSAVED_SESSION_EXIT_CODE)


def run_agent_repl(
    agent_name: str = "default",
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    prompt: str | None = None,
    tools: ToolRegistryProtocol | None = None,
    session_id: str | None = None,
    reset: bool = False,
    compact: bool = False,
    store: SessionStore | None = None,
    isolation: str = "workspace",
    workspace_dir: str | Path | None = None,
) -> None:
    """Synchronous entry point launching the agent REPL event loop."""
    try:
        asyncio.run(
            run_agent_repl_async(
                agent_name=agent_name,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                prompt=prompt,
                tools=tools,
                session_id=session_id,
                reset=reset,
                compact=compact,
                store=store,
                isolation=isolation,
                workspace_dir=workspace_dir,
            )
        )
    except AgentHomeError as exc:
        # The agent name becomes the directory holding that agent's id and memory, so a
        # name no directory can carry is a usage error on the positional argument -- the
        # same class as `--session-id ../../etc/passwd` below, and reported the same way
        # rather than as the framed traceback it was.
        console.print(f"[bold red]Invalid agent name:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc
    except PathTraversalError as exc:
        # Was an unhandled traceback on `--session-id ../../etc/passwd`: the name rule
        # fired correctly in the Core and the CLI printed a stack trace at the user.
        # A refused argument is a usage error, not a crash.
        console.print(f"[bold red]Invalid --session-id:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc
    except SessionIdCollisionError as exc:
        # Same usage-error class as the above and a different cause: the id is legal, and
        # this filesystem cannot tell it apart from a session that already exists (#256).
        # Reported here rather than left to hydrate as a traceback, and reported *instead*
        # of resuming: the alternative — which is what `e8b3e2f` did — was to hydrate the
        # other session's conversation, print "Resumed session 'SessA'" after the user
        # asked for `SESSA`, and overwrite it on the first turn.
        console.print(
            f"[bold red]Unusable --session-id[/bold red] {escape(repr(exc.asked_session_id))}: "
            f"a session named {escape(repr(exc.record_session_id))} already occupies "
            f"{escape(str(exc.path))}, and this filesystem does not distinguish the two names.\n"
            f"[dim]{escape(str(exc))}[/dim]"
        )
        raise typer.Exit(code=2) from exc
    except LLMError as exc:
        # `LLMError`, not `LLMProviderError`: those two are **siblings**, and a missing
        # credential raises `LLMCredentialsNotConfiguredError`. Catching only the
        # narrower `LLMProviderError` clause would let a configuration defect exit as an
        # unhandled traceback.
        #
        # Recorded here and not only around `execute_turn`: this is the failure a new
        # user is most likely to hit, and it happens while the agent is being built,
        # so the turn-level hooks never see it. Measured — an end-to-end run with
        # consent granted produced an empty journal until this line existed.
        record_failure(exc, context={"surface": "cli.run", "phase": "startup"})
        console.print(f"[bold red]LLM Error:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        # 130, the status Typer gives an interrupt everywhere else in the CLI. Returning
        # normally exited 0, so an interrupted `--prompt` run read as a completed one.
        err_console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=130) from None
