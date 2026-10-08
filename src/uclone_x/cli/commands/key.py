"""``ucx key``: save, list and remove the API keys of model connections.

A key is saved in the settings file on the connection it is for (model-gateway §3.2: one key
per connection), through the settings module's one writer -- the same file and the same
writer the dashboard's Settings panel uses, so a key saved here is the key Settings shows,
and the reverse. ``ucx key set gemini`` saves the key on the connection called ``gemini``,
adding it when there is none; ``ucx key set gpu-box`` on a saved connection of that name. Nothing here writes
``.env`` or exports a variable: ``ucx key setup`` used to write ``GEMINI_API_KEY`` into a
``.env`` file while Settings wrote the settings file, and which of the two a request used
depended on which was loaded first.

A key variable in the environment (``GEMINI_API_KEY`` and the rest) still outranks the
saved key -- a CI job has to be able to override the file -- and ``ucx key list`` says
when one does.

What each provider is called, which variables carry its key, and where a key is created
all come from the provider table, ``uclone_x.llm.providers``.
"""

from __future__ import annotations

import contextlib
import logging
import webbrowser
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from uclone_x.core.logging_setup import reason_is_in_the_log
from uclone_x.llm.connections import Connection, saved_connections
from uclone_x.llm.connectors.saved_choice import (
    delete_api_key,
    save_api_key,
    settings_data,
    settings_file,
    update_settings_file,
)
from uclone_x.llm.providers import PROVIDERS, ProviderSpec, env_key, spec_for

logger = logging.getLogger(__name__)

console = Console()

#: What `ucx key set` says when the settings file could not be read and was kept aside
#: (#1921), in the words Settings uses for the same case. No path and no cause: the log has
#: both, and neither is something the person can act on.
SETTINGS_SET_ASIDE_NOTICE = (
    "This version could not open your saved settings. It kept that file unchanged beside "
    "a new one and saved the key in the new one, so settings saved before, such as other "
    "API keys, are not used now."
)

#: What `ucx key set` says when the key could not be saved. Plain: the cause, which names
#: the settings file's path, goes to the log (#1921).
KEY_NOT_SAVED_NOTICE = "The key was not saved: the settings file could not be read or written."

#: What `ucx key remove` says when the settings file could not be read or written.
KEY_NOT_REMOVED_NOTICE = (
    "The key was not removed: the settings file could not be read or written, so it was "
    "left as it is."
)

key_app = typer.Typer(
    name="key",
    help="Save and manage API keys for hosted LLM providers (Google, Anthropic, OpenAI)",
    no_args_is_help=True,
)

#: The providers ``ucx key`` manages: every provider in the table that reads a key.
KEYED_PROVIDERS: tuple[ProviderSpec, ...] = tuple(
    spec for spec in PROVIDERS.values() if spec.key_env_vars
)

#: The hosted providers the interactive ``setup`` offers, in the order it lists them.
_SETUP_CHOICES: tuple[str, ...] = ("gemini", "anthropic", "openai")


def sanitize_key(key: str) -> str:
    """``key`` without surrounding whitespace or one pair of surrounding quotes."""
    cleaned = key.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def mask_key(key: str) -> str:
    """Enough of ``key`` to recognise it, never enough to use it."""
    if len(key) <= 8:
        return "****"
    return f"{key[:6]}...{key[-4:]}"


def _keyed_spec(provider: str) -> ProviderSpec:
    """The provider ``provider`` names, or a plain refusal listing the ones that take a key."""
    spec = spec_for(provider)
    if spec is None or not spec.key_env_vars:
        known = ", ".join(s.id for s in KEYED_PROVIDERS)
        console.print(
            f"[red]There is no connection or provider called {escape(provider.strip())!r} "
            f"that takes a key.[/red] Choose a saved connection, or one of: {known}."
        )
        raise typer.Exit(code=2)
    return spec


def _target(name: str) -> tuple[str, ProviderSpec]:
    """``(connection id, its kind's spec)`` a key named ``name`` is saved on.

    A saved connection's id names it; otherwise ``name`` must be a kind that takes a key,
    which names the connection whose id is that kind.
    """
    clean = name.strip()
    saved = next((c for c in saved_connections(settings_data()) if c.id == clean), None)
    if saved is not None:
        spec = spec_for(saved.kind)
        if spec is not None:
            return saved.id, spec
    spec = _keyed_spec(clean)
    return spec.id, spec


def _save(spec: ProviderSpec, raw_key: str, conn_id: str | None = None) -> None:
    """Save ``raw_key`` on connection ``conn_id`` (of kind ``spec``), and say what outranks it."""
    target = conn_id or spec.id
    clean_key = sanitize_key(raw_key)
    if not clean_key:
        console.print("[red]The key is empty, so nothing was saved.[/red]")
        raise typer.Exit(code=1)
    if spec.key_hint is not None and not clean_key.startswith(spec.key_hint):
        # A warning, not a refusal: providers change their key formats (a current Google
        # key starts `AQ.`), and a refused real key is worse than a saved typo.
        console.print(
            f"[yellow]Note: {spec.display_name} keys usually start with "
            f"'{spec.key_hint}'. It was saved anyway; check it if requests are refused.[/yellow]"
        )
    try:
        kept_aside = _save_over_unreadable(target, clean_key)
    except (OSError, ValueError) as exc:
        logger.warning("The %s key was not saved: %s", target, exc)
        refusal = f"{KEY_NOT_SAVED_NOTICE} {reason_is_in_the_log()}"
        console.print(f"[red]{escape(refusal)}[/red]")
        raise typer.Exit(code=1) from exc
    if kept_aside:
        # Nothing here names a file: the log has where the earlier one went.
        console.print(
            f"[bold green]Saved[/bold green] the {spec.display_name} key "
            f"([green]{mask_key(clean_key)}[/green])."
        )
        console.print(f"[yellow]{escape(SETTINGS_SET_ASIDE_NOTICE)}[/yellow]")
    else:
        console.print(
            f"[bold green]Saved[/bold green] the {spec.display_name} key "
            f"([green]{mask_key(clean_key)}[/green]) in {escape(str(settings_file()))}."
        )
    overriding = env_key(spec.id)
    if overriding is not None and target == spec.id:
        console.print(
            f"[yellow]{overriding[1]} is set in this environment and is used instead of the "
            f"saved key until it is unset.[/yellow]"
        )


def _save_over_unreadable(provider_id: str, key: str) -> bool:
    """Save ``key``, first keeping an unreadable settings file aside; whether one was (#1921).

    ``save_api_key`` refuses to merge into a file it cannot read, and its refusal names the
    file's path, which used to be printed as the reason. The file may be a newer build's
    settings, keys included, so it is not written over: it is kept beside its name, as a
    Settings save in the app keeps it, and the key is saved into a new file. This command
    holds no other settings, so the new file starts with the key alone.

    Raises:
        OSError: The file could not be kept aside, or the key could not be written.
        ValueError: The file was unreadable again when the key was saved (another writer),
            or the refusal was not about an unreadable file.
    """
    try:
        save_api_key(provider_id, key)
        return False
    except ValueError as refusal:
        if update_settings_file({}, replace_unreadable_with={}) is None:
            raise  # the file was readable: the refusal is about something else
        logger.warning("Settings could not be read before a key was saved: %s", refusal)
    save_api_key(provider_id, key)
    return True


@key_app.command("set")
def set_key(
    provider: Annotated[
        str,
        typer.Argument(
            help="Connection the key is for: a saved connection, or gemini | anthropic | "
            "openai | vllm"
        ),
    ],
    api_key: Annotated[
        str | None,
        typer.Option("--key", "-k", help="The key (prompted for, hidden, when omitted)"),
    ] = None,
) -> None:
    """Save a connection's API key in the settings file, keeping every other connection's key."""
    conn_id, spec = _target(provider)
    raw = (
        api_key
        if api_key is not None
        else typer.prompt(f"{spec.display_name} API key", hide_input=True)
    )
    _save(spec, raw, conn_id)


@key_app.command("remove")
def remove_key(
    provider: Annotated[str, typer.Argument(help="Connection whose saved key to remove")],
) -> None:
    """Remove a connection's saved API key. Other keys and the environment are kept."""
    conn_id, spec = _target(provider)
    try:
        delete_api_key(conn_id)
    except (OSError, ValueError) as exc:
        logger.warning("The %s key was not removed: %s", conn_id, exc)
        refusal = f"{KEY_NOT_REMOVED_NOTICE} {reason_is_in_the_log()}"
        console.print(f"[red]{escape(refusal)}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Removed the saved {spec.display_name} key, if there was one.")
    overriding = env_key(spec.id)
    if overriding is not None:
        console.print(
            f"[yellow]{overriding[1]} is still set in this environment, so requests keep "
            f"using it.[/yellow]"
        )


@key_app.command("list")
def list_keys() -> None:
    """Show each connection's saved key (masked) and any environment variable overriding it."""
    rows: list[Connection] = saved_connections(settings_data())
    table = Table(title=f"API keys ({settings_file()})", border_style="cyan")
    table.add_column("Connection", style="bold white")
    table.add_column("Saved key", style="green")
    table.add_column("Environment", style="yellow")
    table.add_column("Get a key", style="blue")

    listed: set[str] = set()
    for conn in rows:
        spec = spec_for(conn.kind)
        if spec is None or not spec.key_env_vars:
            continue
        listed.add(conn.id)
        table.add_row(conn.id, *_key_cells(spec, conn.key, overridable=conn.id == spec.id))
    for spec in KEYED_PROVIDERS:
        if spec.id not in listed:
            table.add_row(spec.id, *_key_cells(spec, None, overridable=True))

    console.print(table)


def _key_cells(
    spec: ProviderSpec, stored: str | None, *, overridable: bool
) -> tuple[str, str, str]:
    """The saved, environment and key-page cells of one connection's row."""
    overriding = env_key(spec.id) if overridable else None
    saved_cell = mask_key(stored) if stored else "[dim]not saved[/dim]"
    env_cell = (
        f"{overriding[1]} ({mask_key(overriding[0])}) overrides the saved key"
        if overriding is not None
        else (f"[dim]{spec.key_env_vars[0]} not set[/dim]" if overridable else "")
    )
    return saved_cell, env_cell, spec.console_url or ""


@key_app.command("setup")
def setup_key(
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider to configure: gemini | anthropic | openai"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--key", "-k", help="API key value (skips the prompt)"),
    ] = None,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-open-browser",
            help="Open the provider's page for creating a key",
        ),
    ] = True,
) -> None:
    """Walk through choosing a provider and saving its API key in the settings file."""
    chosen = spec_for(provider) if provider else None
    if chosen is None or chosen.id not in _SETUP_CHOICES:
        console.print("Which provider is the key for?")
        for number, provider_id in enumerate(_SETUP_CHOICES, start=1):
            console.print(f"  [{number}] {PROVIDERS[provider_id].display_name}")
        answer = typer.prompt(f"Choose (1-{len(_SETUP_CHOICES)})", default="1").strip()
        index = int(answer) - 1 if answer.isdigit() else 0
        chosen = PROVIDERS[_SETUP_CHOICES[index if 0 <= index < len(_SETUP_CHOICES) else 0]]

    if open_browser and not api_key and chosen.console_url:
        console.print(f"Opening {chosen.console_url} to create a key.")
        with contextlib.suppress(Exception):
            webbrowser.open(chosen.console_url)

    raw = (
        api_key
        if api_key is not None
        else typer.prompt(f"{chosen.display_name} API key", hide_input=True)
    )
    _save(chosen, raw)
