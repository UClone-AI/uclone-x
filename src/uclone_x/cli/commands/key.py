"""``ucx key``: save, list and remove the API keys for hosted LLM providers.

A key is saved in the settings file, under the provider it is for, through the settings
module's one writer -- the same file and the same writer the dashboard's Settings panel
uses, so a key saved here is the key Settings shows, and the reverse. Nothing here writes
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
import webbrowser
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from uclone_x.llm.connectors.saved_choice import (
    api_keys,
    delete_api_key,
    save_api_key,
    settings_data,
    settings_file,
)
from uclone_x.llm.providers import PROVIDERS, ProviderSpec, env_key, spec_for

console = Console()

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
            f"[red]There is no provider called {escape(provider.strip())!r} that takes a key.[/red] "
            f"Choose one of: {known}."
        )
        raise typer.Exit(code=2)
    return spec


def _save(spec: ProviderSpec, raw_key: str) -> None:
    """Save ``raw_key`` for ``spec``, and say where it went and what outranks it."""
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
        save_api_key(spec.id, clean_key)
    except (OSError, ValueError) as exc:
        console.print(f"[red]The key was not saved:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[bold green]Saved[/bold green] the {spec.display_name} key "
        f"([green]{mask_key(clean_key)}[/green]) in {escape(str(settings_file()))}."
    )
    overriding = env_key(spec.id)
    if overriding is not None:
        console.print(
            f"[yellow]{overriding[1]} is set in this environment and is used instead of the "
            f"saved key until it is unset.[/yellow]"
        )


@key_app.command("set")
def set_key(
    provider: Annotated[
        str, typer.Argument(help="Provider the key is for: gemini | anthropic | openai | vllm")
    ],
    api_key: Annotated[
        str | None,
        typer.Option("--key", "-k", help="The key (prompted for, hidden, when omitted)"),
    ] = None,
) -> None:
    """Save a provider's API key in the settings file, keeping every other provider's key."""
    spec = _keyed_spec(provider)
    raw = (
        api_key
        if api_key is not None
        else typer.prompt(f"{spec.display_name} API key", hide_input=True)
    )
    _save(spec, raw)


@key_app.command("remove")
def remove_key(
    provider: Annotated[str, typer.Argument(help="Provider whose saved key to remove")],
) -> None:
    """Remove a provider's saved API key. Other providers' keys and the environment are kept."""
    spec = _keyed_spec(provider)
    try:
        delete_api_key(spec.id)
    except (OSError, ValueError) as exc:
        console.print(f"[red]The key was not removed:[/red] {escape(str(exc))}")
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
    """Show each provider's saved key (masked) and any environment variable overriding it."""
    saved = api_keys(settings_data())
    table = Table(title=f"API keys ({settings_file()})", border_style="cyan")
    table.add_column("Provider", style="bold white")
    table.add_column("Saved key", style="green")
    table.add_column("Environment", style="yellow")
    table.add_column("Get a key", style="blue")

    for spec in KEYED_PROVIDERS:
        stored = saved.get(spec.id)
        overriding = env_key(spec.id)
        saved_cell = mask_key(stored) if stored else "[dim]not saved[/dim]"
        env_cell = (
            f"{overriding[1]} ({mask_key(overriding[0])}) overrides the saved key"
            if overriding is not None
            else f"[dim]{spec.key_env_vars[0]} not set[/dim]"
        )
        table.add_row(spec.display_name, saved_cell, env_cell, spec.console_url or "")

    console.print(table)


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
