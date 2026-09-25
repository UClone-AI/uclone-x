"""CLI commands for public LLM API key onboarding and setup."""

from __future__ import annotations

import contextlib
import os
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

key_app = typer.Typer(
    name="key",
    help="Setup and manage API keys for public LLM providers (Gemini, Claude, OpenAI)",
    no_args_is_help=True,
)


@dataclass(frozen=True)
class ProviderKeyInfo:
    name: str
    env_var: str
    url: str
    prefix: str
    pattern: str
    tip: str
    alt_env_var: str | None = None


PROVIDER_INFO: dict[str, ProviderKeyInfo] = {
    "gemini": ProviderKeyInfo(
        name="Google Gemini",
        env_var="GEMINI_API_KEY",
        alt_env_var="GOOGLE_API_KEY",
        url="https://aistudio.google.com/app/apikey",
        prefix="AIzaSy",
        pattern=r"^AIzaSy[A-Za-z0-9_-]{33}$",
        tip="Google AI Studio: 신용카드 없이 분당 15회 무료(Free Tier) 사용 가능",
    ),
    "anthropic": ProviderKeyInfo(
        name="Anthropic Claude",
        env_var="ANTHROPIC_API_KEY",
        url="https://console.anthropic.com/settings/keys",
        prefix="sk-ant-",
        pattern=r"^sk-ant-[A-Za-z0-9_-]{20,}$",
        tip="Anthropic Console: 종량제 Credit 충전 필요",
    ),
    "openai": ProviderKeyInfo(
        name="OpenAI",
        env_var="OPENAI_API_KEY",
        url="https://platform.openai.com/api-keys",
        prefix="sk-",
        pattern=r"^sk-(?:proj-)?[A-Za-z0-9_-]{20,}$",
        tip="OpenAI Platform: 프로젝트 또는 사용자 비밀 키 생성",
    ),
}


def sanitize_key(key: str) -> str:
    cleaned = key.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:6]}...{key[-4:]}"


def find_env_path(start_dir: Path | None = None) -> Path:
    current = start_dir or Path.cwd()
    for directory in [current, *current.parents]:
        env_file = directory / ".env"
        if env_file.exists():
            return env_file
        if (directory / ".git").exists():
            return env_file
    return current / ".env"


def update_env_file(env_path: Path, key_name: str, key_val: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key_name}=") or stripped.startswith(f"export {key_name}="):
            prefix = "export " if stripped.startswith("export ") else ""
            new_lines.append(f'{prefix}{key_name}="{key_val}"')
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f'{key_name}="{key_val}"')

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@key_app.command("list")
def list_keys() -> None:
    """List status of all configured public LLM API keys."""
    table = Table(title="Public LLM API Key Status", border_style="cyan")
    table.add_column("Provider", style="bold white")
    table.add_column("Environment Variable", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Console Link", style="blue")

    for info in PROVIDER_INFO.values():
        key_val = os.getenv(info.env_var)
        if not key_val and info.alt_env_var:
            key_val = os.getenv(info.alt_env_var)

        status = (
            f"[green]Configured ({mask_key(key_val)})[/green]" if key_val else "[red]Not Set[/red]"
        )
        table.add_row(info.name, info.env_var, status, info.url)

    console.print(table)


@key_app.command("setup")
def setup_key(
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider to configure: gemini | anthropic | openai"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--key", "-k", help="API key value (skips interactive prompt)"),
    ] = None,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-open-browser",
            help="Open browser console to generate key automatically",
        ),
    ] = True,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", "-e", help="Path to .env file to update"),
    ] = None,
) -> None:
    """Interactively setup and validate a public LLM API key."""
    console.print("\n[bold cyan]🔑 Public LLM API Key Setup Wizard[/bold cyan]\n")

    chosen_provider = provider.lower().strip() if provider else None
    if not chosen_provider or chosen_provider not in PROVIDER_INFO:
        console.print("설정할 LLM 공급자를 선택하세요:")
        console.print("  [1] [bold]Google Gemini[/bold] (Free Tier 제공, 분당 15회 무료)")
        console.print("  [2] [bold]Anthropic Claude[/bold] (Claude 3.5 Sonnet)")
        console.print("  [3] [bold]OpenAI[/bold] (GPT-4o)")

        choice = typer.prompt("선택 (1-3)", default="1").strip()
        mapping = {"1": "gemini", "2": "anthropic", "3": "openai"}
        chosen_provider = mapping.get(choice, "gemini")

    info = PROVIDER_INFO[chosen_provider]
    console.print(f"\n[cyan]▶ {info.name}[/cyan] 설정을 진행합니다.")
    console.print(f"  [dim]💡 {info.tip}[/dim]\n")

    if open_browser and not api_key:
        console.print(
            f"🌐 기본 브라우저에서 키 발급 페이지를 엽니다: [underline]{info.url}[/underline]"
        )
        with contextlib.suppress(Exception):
            webbrowser.open(info.url)

    key_input = api_key
    if not key_input:
        key_input = typer.prompt(
            f"발급받은 {info.name} API Key를 입력하세요 ({info.prefix}...)",
            hide_input=True,
        )

    clean_key = sanitize_key(key_input)
    if not clean_key:
        console.print("[red]❌ 빈 키가 입력되어 취소되었습니다.[/red]")
        raise typer.Exit(code=1)

    # Validate pattern
    if not clean_key.startswith(info.prefix):
        console.print(
            f"[yellow]⚠️ 경고: {info.name} 키는 일반적으로 '{info.prefix}'로 시작합니다.[/yellow]"
        )

    target_env = env_file or find_env_path()
    update_env_file(target_env, info.env_var, clean_key)
    os.environ[info.env_var] = clean_key

    console.print(
        f"\n[bold green]✔ 성공:[/bold green] {info.name} API Key가 안전하게 저장되었습니다!"
    )
    console.print(f"  - 파일: [white]{target_env}[/white]")
    console.print(f"  - 키: [green]{mask_key(clean_key)}[/green]\n")
