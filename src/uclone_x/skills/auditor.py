"""Skill Auditor security verification and registry implementation.

Implements SkillAuditorProtocol and SkillRegistryProtocol with fail-closed
security evaluation, AST static analysis, and quarantine lifecycle gating.
"""

from __future__ import annotations

import ast
import asyncio
import errno
import hashlib
import logging
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import yaml

from uclone_x.errors import SkillAuditError, SkillNotApprovedError
from uclone_x.sandbox.models import (
    AVAILABLE_ISOLATION_LEVELS,
    IsolationLevel,
    is_weaker_isolation,
)
from uclone_x.skills.approvals import SkillApprovalLedger, SkillPin
from uclone_x.skills.models import (
    AuditVerdict,
    AutoApprovalPolicy,
    SkillAuditReport,
    SkillManifest,
    SkillOrigin,
    SkillStatus,
)
from uclone_x.skills.protocols import SkillAuditorProtocol, SkillProtocol, SkillStoreProtocol
from uclone_x.skills.refusals import SkillRefusalCode, refusal_reason

logger = logging.getLogger(__name__)

__all__ = [
    "RUNTIME_SKILL_STORE_DIRNAME",
    "FileSystemSkillStore",
    "InMemorySkillStore",
    "Skill",
    "SkillAuditor",
    "SkillRefusal",
    "SkillRegistry",
    "compute_skill_sha256",
    "copy_skill_package",
    "load_approved_skills",
    "load_runtime_skill_registry",
    "load_skill_from_dir",
    "manifest_from_dict",
    "parse_skill_markdown",
    "runtime_skill_store_dir",
    "save_skill",
    "serialize_skill_markdown",
]

# Critical dangerous function calls / builtins
DANGEROUS_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "__import__",
        "compile",
        "globals",
        "locals",
    }
)

# Dangerous OS/subprocess functions
DANGEROUS_OS_CALLS: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "popen2",
        "popen3",
        "popen4",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "kill",
        "killpg",
        "remove",
        "unlink",
        "rmdir",
    }
)

# Dangerous subprocess methods
DANGEROUS_SUBPROCESS_CALLS: frozenset[str] = frozenset(
    {
        "call",
        "check_call",
        "check_output",
        "run",
        "Popen",
    }
)

# Dangerous modules
DANGEROUS_MODULES: frozenset[str] = frozenset(
    {
        "pty",
        "ctypes",
    }
)

# Network modules (medium risk if not sandboxed)
NETWORK_MODULES: frozenset[str] = frozenset(
    {
        "socket",
        "http.client",
        "urllib.request",
        "requests",
        "httpx",
        "aiohttp",
    }
)

# Prompt injection patterns
PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard all previous instructions",
    "disregard previous instructions",
    "disregard system prompt",
    "override system prompt",
    "bypass security policy",
    "bypass safety filters",
    "<system>",
    "</system>",
    "[system_prompt]",
    "[/system_prompt]",
)

# Shell command escalation patterns
DANGEROUS_SHELL_PATTERNS: tuple[str, ...] = (
    "rm -rf",
    "sudo ",
    "curl ",
    "wget ",
    "chmod +x",
    "nc -e",
    "dd if=",
    ":(){ :|:& };:",
)


class Skill:
    """Concrete implementation of SkillProtocol representing a loaded modular skill package."""

    def __init__(
        self,
        manifest: SkillManifest,
        instructions_markdown: str,
        directory: Path | None = None,
    ) -> None:
        self._manifest = manifest
        self._instructions_markdown = instructions_markdown
        self._directory = directory

    @property
    def manifest(self) -> SkillManifest:
        """Skill metadata and frontmatter."""
        return self._manifest

    @property
    def instructions_markdown(self) -> str:
        """Markdown procedural knowledge."""
        return self._instructions_markdown

    @property
    def directory(self) -> Path | None:
        """Skill package root directory on disk, if loaded from disk."""
        return self._directory


def parse_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Parse a SKILL.md file into frontmatter dictionary and markdown body."""
    if not text.startswith("---"):
        raise ValueError("Document has no leading YAML frontmatter block starting with '---'")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Document frontmatter block is not closed with '---'")
    yaml_content = parts[1]
    instructions = parts[2].lstrip()
    raw_data: object = yaml.safe_load(yaml_content)
    if not isinstance(raw_data, dict):
        raise ValueError("YAML frontmatter must be a mapping/dictionary")
    raw_dict = cast(dict[object, object], raw_data)
    data_dict: dict[str, Any] = {str(k): v for k, v in raw_dict.items()}
    return data_dict, instructions


def manifest_from_dict(data: dict[str, Any]) -> SkillManifest:
    """Construct a SkillManifest from a dictionary extracted from frontmatter."""
    unknown_keys = set(data.keys()) - set(SkillManifest.model_fields.keys())
    if unknown_keys:
        raise ValueError(
            f"Unknown field(s) in SKILL.md frontmatter: {sorted(unknown_keys)}. "
            f"Allowed fields are: {sorted(SkillManifest.model_fields.keys())}"
        )

    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("Skill manifest is missing required 'name' field")

    origin_val = data.get("origin", SkillOrigin.SYNTHESIZED)
    if isinstance(origin_val, str):
        origin = SkillOrigin(origin_val)
    elif isinstance(origin_val, SkillOrigin):
        origin = origin_val
    else:
        origin = SkillOrigin.SYNTHESIZED

    status_val = data.get("status", SkillStatus.PENDING)
    if isinstance(status_val, str):
        status = SkillStatus(status_val)
    elif isinstance(status_val, SkillStatus):
        status = status_val
    else:
        status = SkillStatus.PENDING

    isolation_val = data.get("requested_isolation")
    if isolation_val is not None and isinstance(isolation_val, str):
        requested_isolation = IsolationLevel(isolation_val)
    elif isinstance(isolation_val, IsolationLevel):
        requested_isolation = isolation_val
    else:
        requested_isolation = None

    scripts_val: object = data.get("scripts")
    scripts_list: list[str] = []
    if isinstance(scripts_val, list):
        for item in cast(list[object], scripts_val):
            scripts_list.append(str(item))
    elif isinstance(scripts_val, tuple):
        for item in cast(tuple[object, ...], scripts_val):
            scripts_list.append(str(item))

    tags_val: object = data.get("tags")
    tags_list: list[str] = []
    if isinstance(tags_val, list):
        for item in cast(list[object], tags_val):
            tags_list.append(str(item))
    elif isinstance(tags_val, tuple):
        for item in cast(tuple[object, ...], tags_val):
            tags_list.append(str(item))

    family_sections_val: object = data.get("family_sections", False)
    if not isinstance(family_sections_val, bool):
        raise ValueError(
            "SKILL.md frontmatter 'family_sections' must be true or false, "
            f"not {family_sections_val!r}"
        )

    return SkillManifest(
        name=name,
        description=str(data.get("description", "")),
        version=str(data.get("version", "0.1.0")),
        author=str(data["author"]) if data.get("author") else None,
        origin=origin,
        status=status,
        requested_isolation=requested_isolation,
        scripts=tuple(scripts_list),
        tags=tuple(tags_list),
        entrypoint=str(data["entrypoint"]) if data.get("entrypoint") else None,
        family_sections=family_sections_val,
        content_sha256=str(data["content_sha256"]) if data.get("content_sha256") else None,
        approved_by=str(data["approved_by"]) if data.get("approved_by") else None,
        approved_at=str(data["approved_at"]) if data.get("approved_at") else None,
        rejected_by=str(data["rejected_by"]) if data.get("rejected_by") else None,
        rejected_at=str(data["rejected_at"]) if data.get("rejected_at") else None,
        rejection_reason=str(data["rejection_reason"]) if data.get("rejection_reason") else None,
    )


#: A line of a `SKILL.md` frontmatter that holds the package's own digest. The digest rule
#: leaves exactly this out -- the key at the start of the line, one space, and 64 lowercase
#: hex digits, bare or in single quotes -- so a file can carry its own digest (#1751). Any
#: other spelling of the key is hashed like every other line: were it left out too, a
#: value such as a YAML anchor could change what the rest of the frontmatter says without
#: changing the digest.
_OWN_DIGEST_LINE: Final[re.Pattern[bytes]] = re.compile(
    rb"content_sha256: (?:[0-9a-f]{64}|'[0-9a-f]{64}')\r?\n?"
)


def _skill_md_digest_bytes(data: bytes) -> bytes:
    """The bytes of a `SKILL.md` that its digest covers: all of them but its own digest line.

    The frontmatter is what `parse_skill_markdown` reads as one: from the leading `---` to
    the next `---`. Only a line inside it can be left out, so the same line in the body is
    hashed.
    """
    if not data.startswith(b"---"):
        return data
    end = data.find(b"---", 3)
    if end == -1:
        return data
    kept = (
        line
        for line in data[:end].splitlines(keepends=True)
        if _OWN_DIGEST_LINE.fullmatch(line) is None
    )
    return b"".join(kept) + data[end:]


def compute_skill_sha256(skill_dir: Path) -> str:
    """Compute a deterministic SHA-256 digest of a skill package.

    **The digest rule** (#1720, #1751), the one statement of it: the walk hashes each regular
    file it lists whose name does not start with a dot, by its path inside the package and
    its bytes, in sorted path order, except that the package's own `SKILL.md` is hashed
    without the frontmatter line `content_sha256: <64 hex digits>`. So writing the digest
    into the file does not change the digest, and a stored digest can match its own file.

    A link to a file is hashed through. A link to a folder is not descended into, so files
    under it are not in the digest, and neither are FIFOs, sockets, devices or broken links.
    A folder the walk cannot list, a file it cannot read, or a link that loops under a name
    that does not start with a dot raises `SkillAuditError` naming it instead of being
    skipped.
    """
    hasher = hashlib.sha256()
    if skill_dir.is_file():
        data = skill_dir.read_bytes()
        hasher.update(_skill_md_digest_bytes(data) if skill_dir.name == "SKILL.md" else data)
        return hasher.hexdigest()

    if not skill_dir.exists() or not skill_dir.is_dir():
        raise SkillAuditError(f"Cannot compute hash for invalid directory: {skill_dir}")

    try:
        for path in sorted(_package_entries(skill_dir)):
            if _is_file(path) and not path.name.startswith("."):
                rel_path = path.relative_to(skill_dir).as_posix()
                hasher.update(rel_path.encode("utf-8"))
                data = path.read_bytes()
                hasher.update(_skill_md_digest_bytes(data) if rel_path == "SKILL.md" else data)
            elif not path.name.startswith(".") and path.is_symlink():
                _refuse_a_loop(path)
    except OSError as exc:
        raise SkillAuditError(
            f"The skill package could not be read in full, so it cannot be audited: "
            f"'{_inside(skill_dir, exc.filename)}' could not be read ({exc.strerror})."
        ) from exc
    return hasher.hexdigest()


def copy_skill_package(skill_dir: Path, into: Path) -> None:
    """Copy into `into` exactly the files the digest rule hashes, each read once (#1777).

    The copy is what `ucx skill approve` audits, asks about and pins, so the digest it pins
    is of the bytes it checked even if the package changes while the person is being asked.
    It walks the package as `compute_skill_sha256` does, so the copy's digest is the
    package's digest at the moment each file was read: a link to a file is copied as the
    file, and a link to a folder, a dot-named file and anything that is not a regular file
    are left out. Raises `SkillAuditError` as `compute_skill_sha256` does.
    """
    if not skill_dir.is_dir():
        raise SkillAuditError(f"Cannot copy an invalid skill directory: {skill_dir}")
    # Read everything first, then write: an error while reading is the package's, and one
    # while writing is the copy's, which is not reported as the package being unreadable.
    files: list[tuple[Path, bytes]] = []
    try:
        for path in sorted(_package_entries(skill_dir)):
            if _is_file(path) and not path.name.startswith("."):
                files.append((path.relative_to(skill_dir), path.read_bytes()))
            elif not path.name.startswith(".") and path.is_symlink():
                _refuse_a_loop(path)
    except OSError as exc:
        raise SkillAuditError(
            f"The skill package could not be read in full, so it cannot be audited: "
            f"'{_inside(skill_dir, exc.filename)}' could not be read ({exc.strerror})."
        ) from exc
    for relative, data in files:
        target = into / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


#: The errors after which `Path.is_file()` on Python 3.11 to 3.13 answers False rather than
#: raising: the entry is missing, or is a link that loops or leads nowhere.
_NOT_A_FILE = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})


def _is_file(path: Path) -> bool:
    """`Path.is_file()` as Python 3.11 to 3.13 answer it, on every version.

    Python 3.14's `is_file()` answers False for any error, so a file whose stat is refused
    (a folder the walk can list but not search) would be left out of the digest instead of
    failing the audit. Here only the errors that mean "not a file" answer False; any other
    is raised, and `compute_skill_sha256` turns it into `SkillAuditError`.
    """
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError as exc:
        if exc.errno in _NOT_A_FILE:
            return False
        raise


def _refuse_a_loop(link: Path) -> None:
    """Raise `ELOOP` for a link that loops; any other link that is not a file is left out.

    `is_file()` answers False for a link that loops, so the walk would leave it out of the
    digest while the story loader refuses it. Raising here makes the audit refuse it too. A
    link to a folder or a broken link is still left out, as before, so the digest of every
    package without a looping link is unchanged.
    """
    try:
        link.stat()
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise


def _package_entries(skill_dir: Path) -> list[Path]:
    """Every entry under `skill_dir`, as `rglob("*")` lists them, but raising on a folder
    it cannot list where `rglob` would skip it silently. Links to folders are listed and
    not descended into, as `rglob` does."""

    def _refuse(error: OSError) -> None:
        raise error

    entries: list[Path] = []
    for folder, dirnames, filenames in os.walk(skill_dir, onerror=_refuse):
        entries.extend(Path(folder) / name for name in (*dirnames, *filenames))
    return entries


def _inside(skill_dir: Path, filename: object) -> str:
    """`filename` as a path inside the package, for a message; never a path outside it."""
    if isinstance(filename, Path):
        path = filename
    elif isinstance(filename, str):
        path = Path(filename)
    else:
        return "."
    try:
        return path.relative_to(skill_dir).as_posix()
    except ValueError:
        return path.name


def serialize_skill_markdown(manifest: SkillManifest, instructions: str) -> str:
    """Serialize a SkillManifest and instructions markdown into standard SKILL.md format."""
    data: dict[str, Any] = {
        "name": manifest.name,
        "description": manifest.description,
        "version": manifest.version,
    }
    if manifest.author:
        data["author"] = manifest.author
    data["origin"] = manifest.origin.value
    data["status"] = manifest.status.value
    if manifest.requested_isolation is not None:
        data["requested_isolation"] = manifest.requested_isolation.value
    if manifest.entrypoint:
        data["entrypoint"] = manifest.entrypoint
    if manifest.scripts:
        data["scripts"] = list(manifest.scripts)
    if manifest.tags:
        data["tags"] = list(manifest.tags)
    if manifest.family_sections:
        data["family_sections"] = True
    if manifest.content_sha256:
        data["content_sha256"] = manifest.content_sha256
    if manifest.approved_by:
        data["approved_by"] = manifest.approved_by
    if manifest.approved_at:
        data["approved_at"] = manifest.approved_at
    if manifest.rejected_by:
        data["rejected_by"] = manifest.rejected_by
    if manifest.rejected_at:
        data["rejected_at"] = manifest.rejected_at
    if manifest.rejection_reason:
        data["rejection_reason"] = manifest.rejection_reason

    yaml_str = yaml.dump(data, sort_keys=False)
    instructions_clean = instructions.strip()
    if instructions_clean:
        return f"---\n{yaml_str}---\n\n{instructions_clean}\n"
    return f"---\n{yaml_str}---\n"


def load_skill_from_dir(skill_dir: Path) -> Skill:
    """Load a skill package from a directory containing SKILL.md."""
    if not skill_dir.exists() or not skill_dir.is_dir():
        raise SkillAuditError(f"Skill directory '{skill_dir}' does not exist or is not a directory")

    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise SkillAuditError(f"Missing required 'SKILL.md' in '{skill_dir}'")

    text = skill_file.read_text(encoding="utf-8")
    try:
        data, instructions = parse_skill_markdown(text)
        manifest = manifest_from_dict(data)
    except Exception as exc:
        raise SkillAuditError(f"Failed to parse skill package in '{skill_dir}': {exc}") from exc

    return Skill(manifest=manifest, instructions_markdown=instructions, directory=skill_dir)


def save_skill(skill_dir: Path, manifest: SkillManifest, instructions: str) -> None:
    """Save/update a skill package SKILL.md in the given directory."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    content = serialize_skill_markdown(manifest, instructions)
    skill_file.write_text(content, encoding="utf-8")


class _PythonASTSecurityVisitor(ast.NodeVisitor):
    """AST visitor inspecting Python source code for security violations and high-risk operations."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.critical_risks: list[str] = []
        self.medium_risks: list[str] = []
        self.low_risks: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name
            if name in DANGEROUS_MODULES:
                self.critical_risks.append(
                    f"Dangerous module import '{name}' in {self.filename}:{node.lineno}"
                )
            elif name in NETWORK_MODULES:
                self.medium_risks.append(
                    f"Network module import '{name}' in {self.filename}:{node.lineno}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            if node.module in DANGEROUS_MODULES:
                self.critical_risks.append(
                    f"Dangerous module import '{node.module}' in {self.filename}:{node.lineno}"
                )
            elif node.module in NETWORK_MODULES:
                self.medium_risks.append(
                    f"Network module import '{node.module}' in {self.filename}:{node.lineno}"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct calls (e.g. eval(), exec())
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DANGEROUS_CALLS:
                self.critical_risks.append(
                    f"Dangerous builtin function call '{func_name}()' in {self.filename}:{node.lineno}"
                )

        # Check attribute calls (e.g. os.system(), subprocess.run(), shutil.rmtree())
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            # Check os.system, os.popen, etc.
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                if module_name == "os" and attr_name in DANGEROUS_OS_CALLS:
                    self.critical_risks.append(
                        f"Dangerous OS call 'os.{attr_name}()' in {self.filename}:{node.lineno}"
                    )
                elif module_name == "subprocess" and attr_name in DANGEROUS_SUBPROCESS_CALLS:
                    self.medium_risks.append(
                        f"Process execution 'subprocess.{attr_name}()' in {self.filename}:{node.lineno}"
                    )
                elif module_name == "shutil" and attr_name == "rmtree":
                    self.critical_risks.append(
                        f"Recursive deletion 'shutil.rmtree()' in {self.filename}:{node.lineno}"
                    )

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            val_lower = node.value.lower()
            for pattern in DANGEROUS_SHELL_PATTERNS:
                if pattern in val_lower:
                    self.critical_risks.append(
                        f"Dangerous shell command pattern '{pattern}' in {self.filename}:{node.lineno}"
                    )
        self.generic_visit(node)


class SkillAuditor:
    """Security auditor for dynamic skills, implementing SkillAuditorProtocol.

    Enforces fail-closed evaluation, AST static analysis, isolation ceiling checks,
    and configurable auto-approval policies (P9 / Issue 2026-09-02-002 / 2026-09-02-041).
    """

    def __init__(
        self,
        policy: AutoApprovalPolicy = AutoApprovalPolicy.SAFE_ONLY,
        isolation_floor: IsolationLevel = IsolationLevel.WORKSPACE,
        auditor_version: str = "0.1.0",
        available_levels: frozenset[IsolationLevel] = AVAILABLE_ISOLATION_LEVELS,
    ) -> None:
        if isolation_floor not in available_levels:
            raise SkillAuditError(
                f"Configured isolation_floor '{isolation_floor.value}' has no available backend runner "
                f"(available: {sorted(lvl.value for lvl in available_levels)})"
            )
        self._policy = policy
        self._isolation_floor = isolation_floor
        self._auditor_version = auditor_version
        self._available_levels = available_levels

    @property
    def policy(self) -> AutoApprovalPolicy:
        """Current auto-approval mode."""
        return self._policy

    @property
    def isolation_floor(self) -> IsolationLevel:
        """The weakest isolation a skill may run under, decided by the runtime."""
        return self._isolation_floor

    async def audit_skill(self, skill_dir: Path) -> SkillAuditReport:
        """Audit a skill package before activation.

        Performs fail-closed static AST analysis, prompt injection detection,
        isolation clamping checks, and policy enforcement.
        """
        return await asyncio.to_thread(self._audit_sync, skill_dir)

    def _audit_sync(self, skill_dir: Path) -> SkillAuditReport:
        if not skill_dir.exists() or not skill_dir.is_dir():
            raise SkillAuditError(f"Skill directory does not exist: {skill_dir}")

        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            raise SkillAuditError(f"Missing required SKILL.md in {skill_dir}")

        content_sha256 = compute_skill_sha256(skill_dir)
        skill = load_skill_from_dir(skill_dir)
        manifest = skill.manifest

        critical_risks: list[str] = []
        medium_risks: list[str] = []
        low_risks: list[str] = []

        # 1. Isolation Policy Check (P3 / Issue 2026-09-02-002, #62)
        # Check if requested isolation has an available runner backend
        if (
            manifest.requested_isolation is not None
            and manifest.requested_isolation not in self._available_levels
        ):
            critical_risks.append(
                f"Requested isolation level '{manifest.requested_isolation.value}' has no available backend runner "
                f"(available: {sorted(lvl.value for lvl in self._available_levels)})"
            )

        # Synthesized skills cannot grant themselves host execution (IsolationLevel.NONE)
        if manifest.origin == SkillOrigin.SYNTHESIZED:
            if manifest.requested_isolation is IsolationLevel.NONE:
                critical_risks.append(
                    "Synthesized skill requested unisolated host execution (IsolationLevel.NONE)"
                )
            if manifest.requested_isolation is not None and is_weaker_isolation(
                manifest.requested_isolation, self._isolation_floor
            ):
                medium_risks.append(
                    f"Requested isolation '{manifest.requested_isolation.value}' is weaker "
                    f"than runtime floor '{self._isolation_floor.value}'"
                )

        # 2. Prompt Injection Check on markdown instructions
        instructions_lower = skill.instructions_markdown.lower()
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern in instructions_lower:
                critical_risks.append(
                    f"Prompt injection / override pattern detected in instructions: '{pattern}'"
                )

        for pattern in DANGEROUS_SHELL_PATTERNS:
            if pattern in instructions_lower:
                critical_risks.append(
                    f"Dangerous shell command pattern detected in instructions: '{pattern}'"
                )

        # 3. Static AST Analysis on all Python scripts in the package
        for py_file in skill_dir.rglob("*.py"):
            if py_file.is_file():
                try:
                    code = py_file.read_text(encoding="utf-8")
                    tree = ast.parse(code, filename=py_file.name)
                    visitor = _PythonASTSecurityVisitor(filename=py_file.name)
                    visitor.visit(tree)
                    critical_risks.extend(visitor.critical_risks)
                    medium_risks.extend(visitor.medium_risks)
                    low_risks.extend(visitor.low_risks)
                except SyntaxError as exc:
                    critical_risks.append(f"Syntax error in script '{py_file.name}': {exc}")
                except Exception as exc:
                    critical_risks.append(f"Failed to analyze script '{py_file.name}': {exc}")

        all_risks = tuple(critical_risks + medium_risks + low_risks)

        # 4. Calculate Risk Score (0.0 to 1.0)
        risk_score: float = 0.0
        if critical_risks:
            risk_score = min(1.0, 0.8 + (len(critical_risks) - 1) * 0.1)
        elif medium_risks:
            risk_score = min(0.7, 0.4 + (len(medium_risks) - 1) * 0.1)
        elif low_risks:
            risk_score = min(0.3, 0.1 * len(low_risks))
        else:
            risk_score = 0.0

        # 5. Determine Verdict and Safety according to Policy
        is_safe: bool
        verdict: AuditVerdict

        if critical_risks or risk_score >= 0.7:
            is_safe = False
            verdict = AuditVerdict.REJECT
        elif medium_risks or risk_score >= 0.2:
            is_safe = False
            verdict = AuditVerdict.REQUIRE_HUMAN_REVIEW
        else:
            # Clean skill with low/zero risk
            if self._policy is AutoApprovalPolicy.NEVER:
                # NEVER auto-approve policy: safe skills still require explicit human review
                is_safe = True
                verdict = AuditVerdict.REQUIRE_HUMAN_REVIEW
            else:
                # SAFE_ONLY or ALWAYS
                is_safe = True
                verdict = AuditVerdict.APPROVE

        return SkillAuditReport(
            skill_name=manifest.name,
            is_safe=is_safe,
            recommendation=verdict,
            risk_score=risk_score,
            detected_risks=all_risks,
            auditor_version=self._auditor_version,
            content_sha256=content_sha256,
        )


def _summary_entry(
    manifest: SkillManifest, report: SkillAuditReport | None, refusal: SkillRefusal | None
) -> dict[str, Any]:
    """One skill as `SkillRegistry.get_summary` lists it; a refused one as `quarantined`."""
    if report is not None:
        audit_report_dict: dict[str, Any] = {
            "skill_name": report.skill_name,
            "is_safe": report.is_safe,
            "recommendation": (
                report.recommendation.value
                if hasattr(report.recommendation, "value")
                else str(report.recommendation)
            ),
            "risk_score": report.risk_score if report.risk_score is not None else 0.0,
            "detected_risks": list(report.detected_risks),
            "auditor_version": report.auditor_version or "0.1.0",
            "content_sha256": report.content_sha256 or "",
        }
    else:
        audit_report_dict = {
            "skill_name": manifest.name,
            "is_safe": manifest.status == SkillStatus.ACTIVE,
            "recommendation": (
                AuditVerdict.APPROVE.value
                if manifest.status == SkillStatus.ACTIVE
                else AuditVerdict.REQUIRE_HUMAN_REVIEW.value
            ),
            "risk_score": 0.0 if manifest.status == SkillStatus.ACTIVE else 0.5,
            "detected_risks": [],
            "auditor_version": "0.1.0",
            "content_sha256": manifest.content_sha256 or "",
        }
    status = SkillStatus.QUARANTINED if refusal is not None else manifest.status
    return {
        "name": manifest.name,
        "description": manifest.description,
        "version": manifest.version,
        "author": manifest.author or "unknown",
        "origin": (
            manifest.origin.value if hasattr(manifest.origin, "value") else str(manifest.origin)
        ),
        "status": status.value,
        "isolation_level": (
            manifest.requested_isolation.value
            if manifest.requested_isolation is not None
            else "workspace"
        ),
        "content_sha256": manifest.content_sha256 or "",
        "scripts": list(manifest.scripts),
        "tags": list(manifest.tags),
        "approved_by": manifest.approved_by,
        "approved_at": manifest.approved_at,
        "rejected_by": manifest.rejected_by,
        "rejected_at": manifest.rejected_at,
        "rejection_reason": manifest.rejection_reason,
        "not_loaded_reason": refusal.reason if refusal is not None else None,
        "not_loaded_code": refusal.code if refusal is not None else None,
        "not_loaded_params": {"name": manifest.name} if refusal is not None else None,
        "audit_report": audit_report_dict,
    }


class SkillRegistry:
    """Registry for hot-reloading skill discovery and runtime management.

    Implements SkillRegistryProtocol with quarantine enforcement.
    """

    def __init__(
        self,
        skills_dir: Path | None = None,
        *,
        store: SkillStoreProtocol | None = None,
    ) -> None:
        if skills_dir is not None and store is not None:
            raise ValueError("Give a SkillRegistry a skills_dir or a store, not both")
        self._store: SkillStoreProtocol | None = (
            FileSystemSkillStore(skills_dir) if skills_dir is not None else store
        )
        self._skills: dict[str, SkillProtocol] = {}
        self._audit_reports: dict[str, SkillAuditReport] = {}
        self._refused: dict[str, SkillRefusal] = {}

    def register(self, skill: SkillProtocol, report: SkillAuditReport) -> None:
        """Register a skill, admitting it only on a passing audit of *this* code."""
        if report.skill_name != skill.manifest.name:
            raise SkillNotApprovedError(
                f"Audit report for '{report.skill_name}' does not match skill '{skill.manifest.name}'"
            )

        if not report.content_sha256 or not skill.manifest.content_sha256:
            raise SkillNotApprovedError(
                f"Missing content hash binding for skill '{skill.manifest.name}': "
                f"report={report.content_sha256}, manifest={skill.manifest.content_sha256}"
            )

        if report.content_sha256 != skill.manifest.content_sha256:
            raise SkillNotApprovedError(
                f"Audit report content hash '{report.content_sha256}' does not match "
                f"skill content hash '{skill.manifest.content_sha256}'"
            )

        if report.is_safe and report.recommendation is AuditVerdict.APPROVE:
            self._skills[skill.manifest.name] = skill
            self._audit_reports[skill.manifest.name] = report
            return

        raise SkillNotApprovedError(
            f"Skill '{skill.manifest.name}' is not approved for registration: "
            f"verdict={report.recommendation.value}, is_safe={report.is_safe}"
        )

    def get(self, name: str) -> SkillProtocol | None:
        """Retrieve an *active* skill by name. Quarantined packages are not returned."""
        return self._skills.get(name)

    def get_audit_report(self, name: str) -> SkillAuditReport | None:
        """Retrieve the security audit report for a registered skill."""
        return self._audit_reports.get(name)

    def list_skills(self) -> list[SkillProtocol]:
        """List all active skills."""
        return list(self._skills.values())

    @property
    def store_missing(self) -> bool:
        """Whether this registry reads a skill folder that does not exist (#1721).

        A process started outside a project with a store (an installed release, or another
        folder) resolves a store that is not there, loads nothing, and would otherwise say
        nothing. A registry with no store at all (a test, or an injected one) is not missing
        one. Read at the time of asking, so a folder created later is seen.
        """
        return isinstance(self._store, FileSystemSkillStore) and not self._store.root.is_dir()

    def get_summary(self) -> dict[str, Any]:
        """Return JSON-serializable list of registered skills and security audit summary.

        `store_missing` says the skill folder this registry reads does not exist, so the
        Settings Skills panel can say why nothing is listed (#1721).

        A skill the last reload refused (#1720) is listed too, as `quarantined`, with the
        reason in plain words in `not_loaded_reason`, so the Settings Skills panel shows it
        rather than letting it disappear.
        """
        skills_list: list[dict[str, Any]] = []
        for skill in self._skills.values():
            manifest = skill.manifest
            skills_list.append(
                _summary_entry(manifest, self._audit_reports.get(manifest.name), None)
            )
        for name, refusal in self._refused.items():
            if name not in self._skills:
                skills_list.append(_summary_entry(refusal.manifest, refusal.report, refusal))

        active_count = sum(1 for s in skills_list if s["status"] == SkillStatus.ACTIVE.value)
        pending_count = sum(1 for s in skills_list if s["status"] == SkillStatus.PENDING.value)
        quarantined_count = sum(
            1 for s in skills_list if s["status"] == SkillStatus.QUARANTINED.value
        )

        return {
            "skills": skills_list,
            "total": len(skills_list),
            "store_missing": self.store_missing,
            "summary": {
                "total_skills": len(skills_list),
                "active_count": active_count,
                "pending_count": pending_count,
                "quarantined_count": quarantined_count,
            },
        }

    async def scan(self, skills_dir: Path) -> tuple[SkillManifest, ...]:
        """Discover packages on disk and return their manifests without activating any."""
        return await asyncio.to_thread(self._scan_sync, skills_dir)

    def _scan_sync(self, skills_dir: Path) -> tuple[SkillManifest, ...]:
        if not skills_dir.exists() or not skills_dir.is_dir():
            return ()

        manifests: list[SkillManifest] = []
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.is_file():
                    try:
                        skill = load_skill_from_dir(child)
                        manifests.append(skill.manifest)
                    except Exception as exc:
                        manifests.append(
                            SkillManifest(
                                name=child.name,
                                description=f"Unparseable skill package: {exc}",
                                origin=SkillOrigin.SYNTHESIZED,
                                status=SkillStatus.REJECTED,
                                rejection_reason=f"Package failed to parse: {exc}",
                            )
                        )
        return tuple(manifests)

    async def reload_approved(
        self,
        skills_dir: Path | None = None,
        auditor: SkillAuditorProtocol | None = None,
    ) -> tuple[SkillProtocol, ...]:
        """Load the approved skills of `skills_dir`, or else of this registry's store, into it."""
        store = FileSystemSkillStore(skills_dir) if skills_dir is not None else self._store
        if store is None:
            return ()
        reloaded: list[SkillProtocol] = []
        for skill, report in await store.load_approved(auditor):
            try:
                self.register(skill, report)
            except SkillNotApprovedError as exc:
                logger.warning("The skill '%s' was not loaded: %s", skill.manifest.name, exc)
                continue
            reloaded.append(skill)
        if isinstance(store, FileSystemSkillStore):
            # A skill refused now is taken out even if an earlier reload admitted it: an
            # edit after approval must stop it at the next reload, not at the next restart.
            self._refused = {refusal.manifest.name: refusal for refusal in store.refused}
            for name in self._refused:
                self._skills.pop(name, None)
                self._audit_reports.pop(name, None)
        return tuple(reloaded)


@dataclass(frozen=True)
class SkillRefusal:
    """A skill the store did not load, why (`code`), and its audit when it got that far.

    `report` is None when the package could not be read or its audit could not finish; the
    panel then lists it from `manifest`, which for an unreadable package holds only its folder
    name. `reason` is the English sentence (`uclone_x.skills.refusals`); the panel words `code`
    in the person's language.
    """

    manifest: SkillManifest
    report: SkillAuditReport | None
    code: SkillRefusalCode

    @property
    def reason(self) -> str:
        """The English sentence for `code`, for the log, the CLI and an older head."""
        return refusal_reason(self.code, self.manifest.name)


def _unreadable_manifest(folder: Path) -> SkillManifest:
    """A stand-in for a package whose `SKILL.md` could not be read: its folder name only."""
    return SkillManifest(
        name=folder.name,
        description="",
        origin=SkillOrigin.SYNTHESIZED,
        status=SkillStatus.QUARANTINED,
    )


class FileSystemSkillStore:
    """The skills in a directory of `<name>/SKILL.md` packages, each audited as it loads.

    This is the store every head uses (`SkillRegistry(skills_dir=...)`). A directory that
    does not exist holds no skills; one that cannot be listed raises `OSError`, which
    `load_approved_skills` logs.

    An active package loads only when its audit approves it **and** its current digest is
    the one pinned for its name (#1720): in the person's approvals ledger, or, for a skill
    that ships with the code, in `SHIPPED_SKILL_PINS`. The digest written in the package is
    not consulted; the package cannot vouch for itself. What was refused, and why, is kept in
    `refused` for the Settings Skills panel.
    """

    def __init__(self, root: Path, approvals: SkillApprovalLedger | None = None) -> None:
        self._root = root
        self._approvals = approvals if approvals is not None else SkillApprovalLedger()
        self._refused: tuple[SkillRefusal, ...] = ()

    @property
    def root(self) -> Path:
        """The directory the packages live in."""
        return self._root

    @property
    def refused(self) -> tuple[SkillRefusal, ...]:
        """The packages the last `load_approved` did not load: an active one it refused, or
        one it could not read at all."""
        return self._refused

    def _read_pins(self) -> dict[str, SkillPin]:
        """The person's pins; an unreadable ledger is logged and approves nothing (P6)."""
        try:
            return self._approvals.read()
        except SkillAuditError as exc:
            logger.warning(
                "The skill approvals ledger at %s could not be read, so only shipped skills "
                "can load: %s",
                self._approvals.path,
                exc.__cause__ or exc,
            )
            return {}

    async def load_approved(
        self, auditor: SkillAuditorProtocol | None = None
    ) -> tuple[tuple[SkillProtocol, SkillAuditReport], ...]:
        """Audit every package marked active and return the approved ones with their reports."""
        target_dir = self._root
        if not target_dir.exists() or not target_dir.is_dir():
            self._refused = ()
            return ()

        active_auditor = auditor or SkillAuditor(policy=AutoApprovalPolicy.SAFE_ONLY)
        pins = await asyncio.to_thread(self._read_pins)
        approved: list[tuple[SkillProtocol, SkillAuditReport]] = []
        refused: list[SkillRefusal] = []
        for child in sorted(target_dir.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                # Why a package failed to load or to be audited, or why an active one was
                # not approved; logged as a warning, not at debug, because an active skill
                # that stops loading is otherwise invisible and the tools that read its data
                # quietly lose it (P6). A package that is not active is skipped unlogged.
                dropped: str | None = None
                skill: Skill | None = None
                try:
                    skill = load_skill_from_dir(child)
                    if skill.manifest.status == SkillStatus.ACTIVE:
                        report = await active_auditor.audit_skill(child)
                        name = skill.manifest.name
                        digests = self._approvals.approved_digests(name, pins)
                        code: SkillRefusalCode
                        if not (report.is_safe and report.recommendation is AuditVerdict.APPROVE):
                            dropped = (
                                "it is marked active, but its audit did not approve it "
                                f"(safe: {report.is_safe}, recommendation: "
                                f"{report.recommendation.value}, "
                                f"{len(report.detected_risks)} risk(s) found)"
                            )
                            code = "failed_safety_check"
                        elif report.content_sha256 not in digests:
                            pinned = ", ".join(sorted(digests)) or "none"
                            dropped = f"its content ({report.content_sha256}) is not the approved version ({pinned})"
                            if digests:
                                code = "changed_after_approval"
                            elif skill.manifest.approved_by:
                                # Approved with `ucx skill approve` before approvals were pinned
                                # (#1776), so the approval lives only in the file (#1777).
                                code = "approved_before_pins"
                            else:
                                code = "never_approved"
                        else:
                            bound_manifest = skill.manifest.model_copy(
                                update={"content_sha256": report.content_sha256}
                            )
                            bound_skill = Skill(
                                manifest=bound_manifest,
                                instructions_markdown=skill.instructions_markdown,
                                directory=child,
                            )
                            approved.append((bound_skill, report))
                            continue
                        refused.append(SkillRefusal(skill.manifest, report, code))
                except Exception as exc:
                    # Logged in full; the panel is told only which of the two it was (#1777).
                    dropped = str(exc)
                    if skill is None:
                        unreadable = _unreadable_manifest(child)
                        refused.append(SkillRefusal(unreadable, None, "unreadable"))
                    elif skill.manifest.status == SkillStatus.ACTIVE:
                        refused.append(SkillRefusal(skill.manifest, None, "check_not_finished"))
                if dropped is not None:
                    logger.warning("The skill '%s' was not loaded: %s", child.name, dropped)
        self._refused = tuple(refused)
        return tuple(approved)


class InMemorySkillStore:
    """Skills held in memory with the reports that approved them; no directory is read.

    For a host that keeps skills somewhere other than local folders (a database, a
    service), and for tests. There are no files to audit, so `auditor` is ignored and
    the given reports stand; the registry's `register` still refuses a report that does
    not approve, or does not bind, the skill it is paired with.
    """

    def __init__(self, entries: Iterable[tuple[SkillProtocol, SkillAuditReport]] = ()) -> None:
        self._entries = tuple(entries)

    async def load_approved(
        self, auditor: SkillAuditorProtocol | None = None
    ) -> tuple[tuple[SkillProtocol, SkillAuditReport], ...]:
        """Every held skill marked active, with its report."""
        del auditor
        return tuple(
            (skill, report)
            for skill, report in self._entries
            if skill.manifest.status is SkillStatus.ACTIVE
        )


#: The runtime skill store's directory, under the repository root (PRD FR-5.1).
#: `ucx-agent-skills`, not `skills`: the store belongs to the Runtime Layer (`ucx agent`),
#: and under the shorter name it twice collected Builder workflow prose instead -- which
#: surfaced as an unaudited `pending` package. Builder skills live in `swarm/skills/`.
#: See `ucx-agent-skills/README.md`.
RUNTIME_SKILL_STORE_DIRNAME: Final[str] = "ucx-agent-skills"

#: What a CLI head says at startup when the skill folder it resolved does not exist (#1721).
#: Plain words: the folder's name is what `ucx skill` writes, so it is the user's word too.
NO_SKILL_STORE_NOTICE: Final[str] = (
    f"No skills are loaded: there is no {RUNTIME_SKILL_STORE_DIRNAME} folder in the project "
    "UClone-X was started from. To see how to add a skill, run: ucx skill --help"
)


def runtime_skill_store_dir(start: Path | None = None) -> Path:
    """Where the runtime skill store is, for a process working in `start` (default: cwd).

    The store is `<git top level>/ucx-agent-skills`, or `<start>/ucx-agent-skills` outside
    a git checkout. This is the one resolver: `ucx skill` reads and writes the store it
    names, and the heads load approved skills from it at startup, so the two cannot drift
    onto different directories. It only resolves; it never creates the directory.
    """
    base = start if start is not None else Path.cwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=base,
            capture_output=True,
            text=True,
            check=True,
        )
        root = Path(out.stdout.strip())
    except (subprocess.CalledProcessError, OSError):
        root = base
    return root / RUNTIME_SKILL_STORE_DIRNAME


async def load_approved_skills(registry: SkillRegistry) -> tuple[SkillProtocol, ...]:
    """Load the approved skills in `registry`'s store into it, at a head's startup (P9).

    Every package marked `active` is audited again and registered only on a passing audit
    of its current bytes (`reload_approved`). A registry with no store, or a store that
    does not exist, loads nothing; a store that cannot be read is logged and loads nothing.
    So a missing or broken store never stops a head from starting.
    """
    try:
        return await registry.reload_approved()
    except OSError as exc:
        logger.warning("The runtime skill store could not be read, so no skill is loaded: %s", exc)
        return ()


async def load_runtime_skill_registry(skills_dir: Path | None = None) -> SkillRegistry:
    """A registry over the runtime skill store, with its approved skills already loaded.

    The store defaults to `runtime_skill_store_dir()`. This is what a CLI head hands
    `HostDependencies.skills`, so that `load_skill` is registered on its agent. When that
    folder does not exist, one line on stderr says so (#1721); stderr, so a head that speaks
    a protocol on stdout (`ucx acp serve`) is not disturbed.
    """
    registry = SkillRegistry(
        skills_dir=skills_dir if skills_dir is not None else runtime_skill_store_dir()
    )
    await load_approved_skills(registry)
    if registry.store_missing:
        print(NO_SKILL_STORE_NOTICE, file=sys.stderr)
    return registry
