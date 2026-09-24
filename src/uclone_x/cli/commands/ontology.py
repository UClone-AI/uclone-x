"""CLI commands for tiered LinkML domain ontologies, deterministic validation, and review."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from rich.console import Console
from rich.table import Table

from uclone_x.errors import (
    OntologyPromotionError,
    OntologyRetractionBlockedError,
    OntologyViolationError,
    UnparseableDirectiveError,
)
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import (
    OntologyAxiom,
    OntologyConcept,
    OntologyTier,
)

ontology_app = typer.Typer(
    name="ontology",
    help="Manage tiered LinkML domain ontologies, deterministic validation, and induction",
    no_args_is_help=True,
)

console = Console()


def _repo_root() -> Path:
    """Anchor paths to the repository root, or current directory if not in git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def _ontology_file(agent_name: str = "default", custom_dir: Path | None = None) -> Path:
    """Resolve the ontology YAML file path for an agent."""
    if custom_dir is not None:
        target_dir = custom_dir
    else:
        target_dir = _repo_root() / "ontology"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{agent_name}.yaml"


def _load_engine(
    agent_name: str = "default", custom_dir: Path | None = None
) -> tuple[OntologyEngine, Path]:
    """Load or initialize an OntologyEngine for the specified agent."""
    file_path = _ontology_file(agent_name, custom_dir)
    engine = OntologyEngine(agent_id=agent_name)
    if file_path.is_file():
        engine.load_from_yaml(file_path)
    return engine, file_path


@ontology_app.command("list")
def ontology_list(
    tier: Annotated[
        str | None,
        typer.Option(
            "--tier",
            "-t",
            help="Filter by tier: asserted | induced_enforcing | induced_candidate | all",
        ),
    ] = None,
    agent: Annotated[
        str, typer.Option("--agent", "-a", help="Agent ontology identifier")
    ] = "default",
    dir_path: Annotated[
        Path | None, typer.Option("--dir", "-d", help="Custom ontology directory")
    ] = None,
) -> None:
    """List concepts, relations, and axioms in the agent's domain ontology."""
    engine, _ = _load_engine(agent, dir_path)

    filter_tier: OntologyTier | None = None
    if tier and tier.lower() != "all":
        try:
            filter_tier = OntologyTier(tier.lower())
        except ValueError:
            console.print(
                f"[bold red]✖ Invalid tier:[/bold red] '{tier}'. "
                f"Must be one of: {', '.join(t.value for t in OntologyTier)} or 'all'"
            )
            raise typer.Exit(code=1) from None

    concepts = engine.list_concepts(filter_tier)
    relations = engine.list_relations(filter_tier)
    axioms = engine.list_axioms(filter_tier)

    if not concepts and not relations and not axioms:
        console.print(f"[yellow]No ontology elements found for agent '{agent}'.[/yellow]")
        return

    table = Table(title=f"🧠 Domain Ontology Knowledge Graph: {agent}")
    table.add_column("Element / Triplet", style="bold cyan", no_wrap=True)
    table.add_column("Kind", style="magenta")
    table.add_column("Tier", style="bold")
    table.add_column("Prec", justify="right", style="dim")
    table.add_column("Conf", justify="right")
    table.add_column("Obs", justify="right", style="dim")
    table.add_column("Hash (SHA-256)", style="dim")

    for c in concepts:
        obs_str = str(c.evidence.observation_count) if c.evidence else "-"
        tier_style = (
            "[cyan]asserted[/cyan]"
            if c.tier == OntologyTier.ASSERTED
            else (
                "[green]induced_enforcing[/green]"
                if c.tier == OntologyTier.INDUCED_ENFORCING
                else "[yellow]induced_candidate[/yellow]"
            )
        )
        table.add_row(
            c.name,
            "Concept",
            tier_style,
            str(c.precedence),
            f"{c.confidence:.2f}",
            obs_str,
            c.content_hash[:12] + "…" if c.content_hash else "-",
        )

    for r in relations:
        obs_str = str(r.evidence.observation_count) if r.evidence else "-"
        tier_style = (
            "[cyan]asserted[/cyan]"
            if r.tier == OntologyTier.ASSERTED
            else (
                "[green]induced_enforcing[/green]"
                if r.tier == OntologyTier.INDUCED_ENFORCING
                else "[yellow]induced_candidate[/yellow]"
            )
        )
        arrow = "->" if r.is_directed else "<->"
        table.add_row(
            f"{r.source_entity} {arrow} {r.predicate} {arrow} {r.target_entity}",
            "Relation",
            tier_style,
            str(r.precedence),
            f"{r.confidence:.2f}",
            obs_str,
            r.content_hash[:12] + "…" if r.content_hash else "-",
        )

    for a in axioms:
        obs_str = str(a.evidence.observation_count) if a.evidence else "-"
        tier_style = (
            "[cyan]asserted[/cyan]"
            if a.tier == OntologyTier.ASSERTED
            else (
                "[green]induced_enforcing[/green]"
                if a.tier == OntologyTier.INDUCED_ENFORCING
                else "[yellow]induced_candidate[/yellow]"
            )
        )
        table.add_row(
            f"{a.name} ({a.subject_entity})",
            "Axiom",
            tier_style,
            str(a.precedence),
            f"{a.confidence:.2f}",
            obs_str,
            a.content_hash[:12] + "…" if a.content_hash else "-",
        )

    console.print(table)
    console.print(
        f"[dim]Version: {engine.version} | Asserted Content Hash: {engine.content_hash[:20]}…[/dim]"
    )


@ontology_app.command("teach")
def ontology_teach(
    directive_or_name: Annotated[
        str | None, typer.Argument(help="Natural language directive or concept name")
    ] = None,
    concept: Annotated[
        str | None, typer.Option("--concept", "-c", help="Explicit concept name to teach")
    ] = None,
    parent: Annotated[
        str | None, typer.Option("--parent", "-p", help="Parent concept type")
    ] = None,
    attributes: Annotated[
        str | None,
        typer.Option("--attributes", "--attrs", help="Comma-separated name:type attribute pairs"),
    ] = None,
    required: Annotated[
        str | None,
        typer.Option("--required", "-r", help="Comma-separated list of required field names"),
    ] = None,
    relation: Annotated[
        str | None,
        typer.Option(
            "--relation",
            "--rel",
            help="Relationship in format 'SourceEntity:predicate:TargetEntity'",
        ),
    ] = None,
    axiom: Annotated[
        str | None,
        typer.Option(
            "--axiom",
            help="Axiom rule in format 'RuleName:SubjectEntity:Predicate:ObjectValue'",
        ),
    ] = None,
    agent: Annotated[
        str, typer.Option("--agent", "-a", help="Agent ontology identifier")
    ] = "default",
    dir_path: Annotated[
        Path | None, typer.Option("--dir", "-d", help="Custom ontology directory")
    ] = None,
) -> None:
    """Explicitly teach an asserted concept, relation, or invariant rule to the agent."""
    engine, file_path = _load_engine(agent, dir_path)

    # 1. Relation option
    if relation:
        parts = relation.split(":")
        if len(parts) != 3:
            console.print(
                "[bold red]✖ Invalid relation format.[/bold red] Expected 'Source:predicate:Target'"
            )
            raise typer.Exit(code=1)
        rel_obj = engine.teach_relation(
            source_entity=parts[0].strip(),
            predicate=parts[1].strip(),
            target_entity=parts[2].strip(),
        )
        engine.save_to_yaml(file_path)
        console.print(
            f"[bold green]✔ Taught asserted relation:[/bold green] "
            f"[cyan]{rel_obj.source_entity} -> {rel_obj.predicate} -> {rel_obj.target_entity}[/cyan] "
            f"(version: {engine.version}, content_hash: [dim]{engine.content_hash[:16]}…[/dim])"
        )
        return

    # 2. Axiom option
    if axiom:
        parts = axiom.split(":")
        if len(parts) < 4:
            console.print(
                "[bold red]✖ Invalid axiom format.[/bold red] Expected 'Name:Subject:Predicate:Value'"
            )
            raise typer.Exit(code=1)
        ax_obj = engine.teach_axiom(
            name=parts[0].strip(),
            subject_entity=parts[1].strip(),
            predicate=parts[2].strip(),
            object_value=":".join(parts[3:]).strip(),
        )
        engine.save_to_yaml(file_path)
        console.print(
            f"[bold green]✔ Taught asserted axiom:[/bold green] [cyan]{ax_obj.name}[/cyan] "
            f"(version: {engine.version}, content_hash: [dim]{engine.content_hash[:16]}…[/dim])"
        )
        return

    # 3. Concept / Directive option
    if not concept and not directive_or_name:
        console.print("[bold red]✖ Please specify a concept name or directive.[/bold red]")
        raise typer.Exit(code=1)

    attrs_dict: dict[str, str] = {}
    if attributes:
        for pair in attributes.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                attrs_dict[k.strip()] = v.strip()
            elif pair.strip():
                attrs_dict[pair.strip()] = "str"

    req_list: list[str] = []
    if required:
        req_list = [r.strip() for r in required.split(",") if r.strip()]

    if concept:
        c_obj = engine.teach_concept(
            name=concept,
            parent_type=parent,
            attributes=attrs_dict,
            required_fields=req_list,
        )
        res_elem_name = c_obj.name
    elif directive_or_name:
        if attributes or required or parent:
            c_obj = engine.teach_concept(
                name=directive_or_name,
                parent_type=parent,
                attributes=attrs_dict,
                required_fields=req_list,
            )
            res_elem_name = c_obj.name
        else:
            try:
                elem_obj = engine.teach_directive(directive_or_name)
            except UnparseableDirectiveError as err:
                console.print(f"[bold red]✖ Unparseable directive:[/bold red] {err}")
                raise typer.Exit(code=1) from None

            if isinstance(elem_obj, (OntologyConcept, OntologyAxiom)):
                res_elem_name = elem_obj.name
            else:
                res_elem_name = (
                    f"{elem_obj.source_entity}->{elem_obj.predicate}->{elem_obj.target_entity}"
                )
    else:
        console.print("[bold red]✖ Please specify a concept name or directive.[/bold red]")
        raise typer.Exit(code=1)

    engine.save_to_yaml(file_path)
    console.print(
        f"[bold green]✔ Taught asserted element:[/bold green] [cyan]{res_elem_name}[/cyan] "
        f"(tier: [green]asserted[/green], version: {engine.version}, content_hash: [dim]{engine.content_hash[:16]}…[/dim])"
    )


@ontology_app.command("review")
def ontology_review(
    agent: Annotated[
        str, typer.Option("--agent", "-a", help="Agent ontology identifier")
    ] = "default",
    dir_path: Annotated[
        Path | None, typer.Option("--dir", "-d", help="Custom ontology directory")
    ] = None,
    promote: Annotated[
        str | None,
        typer.Option("--promote", "-p", help="Promote candidate term to induced_enforcing"),
    ] = None,
    demote: Annotated[
        str | None,
        typer.Option("--demote", help="Demote an enforcing term back to candidate staging"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force promotion bypassing observation thresholds"),
    ] = False,
) -> None:
    """Review induced candidates, promote verified concepts, or demote contradicted terms."""
    engine, file_path = _load_engine(agent, dir_path)

    if promote:
        try:
            res = engine.promote(promote, force=force)
            engine.save_to_yaml(file_path)
            if isinstance(res, (OntologyConcept, OntologyAxiom)):
                res_name = res.name
            else:
                res_name = f"{res.source_entity}->{res.predicate}->{res.target_entity}"
            console.print(
                f"[bold green]✔ Promoted candidate:[/bold green] [cyan]{res_name}[/cyan] "
                f"to [green]induced_enforcing[/green] tier."
            )
            return
        except OntologyPromotionError as exc:
            console.print(f"[bold red]✖ Promotion rejected:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    if demote:
        try:
            res = engine.demote(demote)
            engine.save_to_yaml(file_path)
            if isinstance(res, (OntologyConcept, OntologyAxiom)):
                res_name = res.name
            else:
                res_name = f"{res.source_entity}->{res.predicate}->{res.target_entity}"
            console.print(
                f"[bold yellow]✔ Demoted element:[/bold yellow] [cyan]{res_name}[/cyan] "
                f"to [yellow]induced_candidate[/yellow] tier."
            )
            return
        except OntologyViolationError as exc:
            console.print(f"[bold red]✖ Demotion failed:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    # Default: List candidates awaiting review
    candidates = engine.list_concepts(OntologyTier.INDUCED_CANDIDATE)
    cand_relations = engine.list_relations(OntologyTier.INDUCED_CANDIDATE)
    cand_axioms = engine.list_axioms(OntologyTier.INDUCED_CANDIDATE)

    if not candidates and not cand_relations and not cand_axioms:
        console.print("[green]No candidate ontology elements awaiting review.[/green]")
        return

    table = Table(title=f"🔍 Candidate Review Staging: {agent}")
    table.add_column("Element", style="bold cyan", no_wrap=True)
    table.add_column("Kind", style="magenta")
    table.add_column("Obs", justify="right")
    table.add_column("First Seen", style="dim")
    table.add_column("Last Seen", style="dim")
    table.add_column("Contradictions", style="yellow")
    table.add_column("Eligible", justify="center")

    for c in candidates:
        ev = c.evidence
        obs = ev.observation_count if ev else 1
        first = ev.first_seen[:10] if ev else "-"
        last = ev.last_seen[:10] if ev else "-"
        contra_count = len(ev.contradicting_observations) if ev else 0
        eligible = obs >= 5 and contra_count == 0
        table.add_row(
            c.name,
            "Concept",
            str(obs),
            first,
            last,
            str(contra_count),
            "[green]Yes[/green]" if eligible else "[yellow]No (<5 obs)[/yellow]",
        )

    for r in cand_relations:
        ev = r.evidence
        obs = ev.observation_count if ev else 1
        first = ev.first_seen[:10] if ev else "-"
        last = ev.last_seen[:10] if ev else "-"
        contra_count = len(ev.contradicting_observations) if ev else 0
        eligible = obs >= 5 and contra_count == 0
        table.add_row(
            f"{r.source_entity} -> {r.predicate} -> {r.target_entity}",
            "Relation",
            str(obs),
            first,
            last,
            str(contra_count),
            "[green]Yes[/green]" if eligible else "[yellow]No[/yellow]",
        )

    for a in cand_axioms:
        ev = a.evidence
        obs = ev.observation_count if ev else 1
        first = ev.first_seen[:10] if ev else "-"
        last = ev.last_seen[:10] if ev else "-"
        contra_count = len(ev.contradicting_observations) if ev else 0
        eligible = obs >= 5 and contra_count == 0
        table.add_row(
            a.name,
            "Axiom",
            str(obs),
            first,
            last,
            str(contra_count),
            "[green]Yes[/green]" if eligible else "[yellow]No[/yellow]",
        )

    console.print(table)
    console.print(
        "[dim]Run with [bold]--promote <name>[/bold] to promote an eligible candidate to enforcing tier.[/dim]"
    )


@ontology_app.command("forget")
def ontology_forget(
    name: Annotated[
        str, typer.Argument(help="Name of concept, relation, or axiom to retract/forget")
    ],
    agent: Annotated[
        str, typer.Option("--agent", "-a", help="Agent ontology identifier")
    ] = "default",
    dir_path: Annotated[
        Path | None, typer.Option("--dir", "-d", help="Custom ontology directory")
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Force retraction and cascade delete asserted dependents"
        ),
    ] = False,
) -> None:
    """Retract an ontology element, checking dependencies to prevent orphaned invariants."""
    engine, file_path = _load_engine(agent, dir_path)

    try:
        retracted = engine.forget(name, force=force)
        engine.save_to_yaml(file_path)
        console.print(
            f"[bold green]✔ Retracted ontology element(s):[/bold green] [cyan]{', '.join(retracted)}[/cyan]"
        )
    except OntologyRetractionBlockedError as exc:
        console.print(f"[bold red]✖ Retraction blocked:[/bold red] {exc}")
        console.print("[yellow]Use --force to cascade delete dependent asserted elements.[/yellow]")
        raise typer.Exit(code=1) from exc
    except OntologyViolationError as exc:
        console.print(f"[bold red]✖ Retraction failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@ontology_app.command("validate")
def ontology_validate(
    concept: Annotated[
        str, typer.Argument(help="Concept name in domain ontology to validate against")
    ],
    data: Annotated[
        str | None, typer.Option("--data", help="JSON formatted payload string to validate")
    ] = None,
    file_path: Annotated[
        Path | None, typer.Option("--file", "-f", help="Path to JSON file containing payload")
    ] = None,
    content_hash: Annotated[
        str | None, typer.Option("--content-hash", "--hash", help="Pinned content hash snapshot")
    ] = None,
    agent: Annotated[
        str, typer.Option("--agent", "-a", help="Agent ontology identifier")
    ] = "default",
    dir_path: Annotated[
        Path | None, typer.Option("--dir", "-d", help="Custom ontology directory")
    ] = None,
) -> None:
    """Validate input/output payload against active compiled ontology schema."""
    engine, _ = _load_engine(agent, dir_path)

    payload: dict[str, Any] = {}
    if data:
        try:
            parsed: Any = json.loads(data)
            if isinstance(parsed, dict):
                payload = cast(dict[str, Any], parsed)
            else:
                console.print("[bold red]✖ Payload must be a JSON object (dict).[/bold red]")
                raise typer.Exit(code=1)
        except json.JSONDecodeError as exc:
            console.print(f"[bold red]✖ Invalid JSON data:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
    elif file_path:
        if not file_path.is_file():
            console.print(f"[bold red]✖ Data file not found:[/bold red] {file_path}")
            raise typer.Exit(code=1)
        try:
            parsed_f: Any = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(parsed_f, dict):
                payload = cast(dict[str, Any], parsed_f)
            else:
                console.print("[bold red]✖ File content must be a JSON object (dict).[/bold red]")
                raise typer.Exit(code=1)
        except json.JSONDecodeError as exc:
            console.print(f"[bold red]✖ Invalid JSON file:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        console.print("[bold red]✖ Please provide --data <json_str> or --file <path>.[/bold red]")
        raise typer.Exit(code=1)

    result = engine.validate_entity(concept, payload, pinned_content_hash=content_hash)

    table = Table(title=f"🛡️ Ontology Schema Validation: {concept}")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    valid_style = "[green]True (PASSED)[/green]" if result.is_valid else "[red]False (FAILED)[/red]"
    tier_str = result.matched_tier.value if result.matched_tier else "unknown"

    table.add_row("Entity Concept", concept)
    table.add_row("Is Valid", valid_style)
    table.add_row("Matched Tier", tier_str)
    table.add_row("Latency", f"{result.latency_ms:.3f} ms")
    table.add_row("Content Hash", result.content_hash or "-")

    if result.errors:
        errs_formatted = "\n".join(f"[red]•[/red] {e}" for e in result.errors)
        table.add_row("Errors", errs_formatted)
    else:
        table.add_row("Errors", "[green]None[/green]")

    if result.warnings:
        warns_formatted = "\n".join(f"[yellow]•[/yellow] {w}" for w in result.warnings)
        table.add_row("Warnings", warns_formatted)

    console.print(table)

    if not result.is_valid:
        raise typer.Exit(code=1)
