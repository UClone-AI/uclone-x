"""Persona storage subsystem and protocols (Issue #538 / #892)."""

from __future__ import annotations

import difflib
import logging
import os
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.agent.models import (
    AgentLLMConfig,
    ModelTier,
    PersonaDefinition,
)
from uclone_x.agent.prompts import compose_system_prompt, composed_default_prompts
from uclone_x.core.agent_home import AgentHomeError, refuse_an_unusable_username

logger = logging.getLogger(__name__)

BUILTIN_PERSONAS_DIR = Path(__file__).resolve().parent.parent / "personas"
DEFAULT_WORKSPACE_PERSONAS_SUBDIR = ".uclone/personas"
DEFAULT_PERSONA_NAME: Final[str] = "clone"


class PersonaLoadError(Exception):
    """A persona file could not be loaded, and the load is not continued without it.

    Skipping the file was the previous behaviour -- a `logger.warning` and a `continue`.
    That is the failure mode P6 forbids: a user who mistypes a key in their own persona
    gets a runtime with one fewer agent in it and a line in a log they are not reading,
    which is indistinguishable from having written no file at all. An unloadable persona
    is an error against the file that caused it.
    """


class PersonaWriteError(Exception):
    """A persona could not be written. Subclasses carry the HTTP meaning of the refusal."""


class PersonaWriteRefused(PersonaWriteError):
    """The request cannot become a valid persona file where the loader reads (422)."""


class PersonaWriteConflict(PersonaWriteError):
    """The write would replace something it was not asked to replace (409)."""


class PersonaNotFound(PersonaWriteError):
    """An edit named a persona no loaded file defines (404)."""


class PersonaDraft(BaseModel):
    """What a head sends to create or edit a persona: the file's contents, field by field.

    Strict, and closed to unknown keys, so a value the file cannot hold is refused instead of
    coerced (`"yes"` is not `true`) or dropped (a misspelt key does not quietly vanish). The
    name is a plain string here on purpose: it is checked by `save_persona` against the same
    rule the loader applies, so the head and a restart agree on which names exist.

    `system_prompt` is the persona's own text. `append_default_prompt` is the loader's flag
    for appending the composed default prompt, kept as a flag so an edit does not freeze a
    copy of that prompt into the file.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    role: str = Field(min_length=1)
    description: str = ""
    system_prompt: str = Field(min_length=1)
    append_default_prompt: bool = False
    allowed_tools: list[str] = Field(default_factory=list[str])
    model_name: str | None = None
    model_tier: Literal["inherit", "fast", "pro", "flash_lite", "custom"] = "inherit"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    enable_write_tools: bool = False
    enable_subagent_tools: bool = False
    a2a_peers: list[str] = Field(default_factory=list[str])


def split_appended_default_prompt(prompt: str) -> tuple[str, bool]:
    """Undo the loader's `append_default_prompt`: the persona's own text, and whether it was on.

    The loader stores the composed prompt, so a head that shows the stored text and saves it
    back would write the whole default prompt into the file as the persona's own words, and
    no later change to the default would reach it again.

    The appended default depends on the persona's tools (#1424), so the suffix is matched
    against every composition the loader can append, not against one.
    """
    for composed in composed_default_prompts():
        suffix = f"\n\n{composed}"
        if prompt.endswith(suffix):
            return prompt[: -len(suffix)], True
    return prompt, False


def default_prompt_for(persona: PersonaDefinition) -> str:
    """The default prompt `append_default_prompt` appends for `persona` (#1424).

    Composed from what the persona can actually call: its `granted_tools` (an empty list
    is no restriction, so every fragment), with the write tools dropped when
    `enable_write_tools` is off, since those are refused to it. Depends only on the
    persona's file, so it is the same text on every load.
    """
    return compose_system_prompt(
        tools=persona.granted_tools or None,
        writes_permitted=persona.enable_write_tools,
    )


def _with_default_prompt(persona: PersonaDefinition) -> PersonaDefinition:
    """`persona` with its default prompt appended to its own text."""
    return persona.model_copy(
        update={"system_prompt": f"{persona.system_prompt}\n\n{default_prompt_for(persona)}"}
    )


def _refuse_an_unwritable_name(name: str) -> None:
    """Refuse a name that cannot be a persona file name, before any path is built from it."""
    try:
        refuse_an_unusable_username(name)
    except AgentHomeError as exc:
        raise PersonaWriteRefused(
            f"{name!r} cannot be a persona name: {exc} The name is both the file name and "
            f"the agent id, so choose one that fits that rule."
        ) from exc


class _LiteralBlockDumper(yaml.SafeDumper):
    """Writes multi-line strings as `|` blocks, so a saved prompt stays readable in the file."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    style = "|" if "\n" in value else None
    node: yaml.Node = dumper.represent_scalar(  # pyright: ignore[reportUnknownMemberType]
        "tag:yaml.org,2002:str", value, style=style
    )
    return node


_LiteralBlockDumper.add_representer(str, _represent_str)


def _file_contents(draft: PersonaDraft) -> dict[str, Any]:
    """The mapping a persona file holds, in the keys the loader reads."""
    llm_config: dict[str, Any] = {
        "model_tier": draft.model_tier,
        "temperature": float(draft.temperature),
    }
    if draft.model_name is not None:
        llm_config["model_name"] = draft.model_name
    if draft.max_tokens is not None:
        llm_config["max_tokens"] = draft.max_tokens
    contents: dict[str, Any] = {
        "name": draft.name,
        "role": draft.role,
        "description": draft.description,
        "append_default_prompt": draft.append_default_prompt,
        "system_prompt": draft.system_prompt,
        "allowed_tools": list(draft.allowed_tools),
        "llm_config": llm_config,
        "enable_write_tools": draft.enable_write_tools,
        "enable_subagent_tools": draft.enable_subagent_tools,
    }
    if draft.a2a_peers:
        # Written only when set, so a persona that calls no one keeps the file it had.
        contents["a2a_peers"] = list(draft.a2a_peers)
    return contents


@runtime_checkable
class PersonaStoreProtocol(Protocol):
    """Protocol for persona storage backends."""

    def get_persona(self, name: str) -> PersonaDefinition | None:
        """Get persona definition by unique name."""
        ...

    def list_personas(self) -> Sequence[PersonaDefinition]:
        """Return sequence of all persona definitions."""
        ...

    def save_persona(self, draft: PersonaDraft, *, create: bool) -> PersonaDefinition:
        """Save a persona draft (create or update) and return the definition."""
        ...

    def has_persona(self, name: str) -> bool:
        """Check if a persona with the given name exists."""
        ...

    def source_of(self, name: str) -> Path | str | None:
        """Return the source location/identifier of the persona, or None."""
        ...


class YamlFilePersonaStore:
    """File-backed YAML persona store for a specific directory."""

    def __init__(
        self,
        directory: Path | str,
        *,
        tool_names: Collection[str] | None = None,
        read_only: bool = False,
    ) -> None:
        self._directory = Path(directory).resolve()
        self._tool_names: frozenset[str] | None = (
            frozenset(tool_names) if tool_names is not None else None
        )
        self._read_only = read_only
        self._personas: dict[str, PersonaDefinition] = {}
        self._sources: dict[str, Path] = {}
        self.reload()

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def validates_tools(self) -> bool:
        return self._tool_names is not None

    def reload(self) -> None:
        """Discover and load personas from the directory."""
        self._personas.clear()
        self._sources.clear()
        if not self._directory.is_dir():
            return
        for file_path in sorted(self._directory.glob("*.yaml")) + sorted(
            self._directory.glob("*.yml")
        ):
            try:
                self._load_file(file_path)
            except PersonaLoadError:
                raise
            except Exception as exc:
                raise PersonaLoadError(f"{file_path}: {exc}") from exc

    def _load_file(self, file_path: Path) -> None:
        persona = self._parse_file(file_path)
        self._personas[persona.name] = persona
        self._sources[persona.name] = file_path

    def _parse_file(self, file_path: Path, *, label: Path | None = None) -> PersonaDefinition:
        source = label or file_path
        persona = self._parse_text(file_path.read_text(encoding="utf-8"), source)
        self._validate_tools(persona, source=source)
        return persona

    def _parse_text(self, raw_text: str, file_path: Path) -> PersonaDefinition:
        raw_obj: object = yaml.safe_load(raw_text)
        if not isinstance(raw_obj, dict):
            raise PersonaLoadError(
                f"{file_path}: a persona file must be a YAML mapping, got {type(raw_obj).__name__}"
            )
        raw_dict = cast(dict[object, object], raw_obj)
        data: dict[str, Any] = {str(k): v for k, v in raw_dict.items()}

        allowed: object = data.get("allowed_tools")
        if isinstance(allowed, (list, tuple, set)):
            allowed_seq = cast(Sequence[object], allowed)
            data["allowed_tools"] = tuple(str(x) for x in allowed_seq)
        elif allowed is None:
            data["allowed_tools"] = ()

        peers: object = data.get("a2a_peers")
        if isinstance(peers, (list, tuple)):
            peer_seq = cast(Sequence[object], peers)
            data["a2a_peers"] = tuple(str(x) for x in peer_seq)
        elif peers is None:
            data["a2a_peers"] = ()

        raw_name: object = data.get("name") or data.get("id") or file_path.stem
        persona_name = str(raw_name)
        data["name"] = persona_name

        try:
            refuse_an_unusable_username(persona_name)
        except AgentHomeError as exc:
            raise PersonaLoadError(
                f"{file_path}: persona name {persona_name!r} cannot be an agent name, and a "
                f"persona's name is the agent id the dashboard sends: {exc}"
            ) from exc

        append_default: object = data.pop("append_default_prompt", False)
        if not isinstance(append_default, bool):
            raise PersonaLoadError(
                f"{file_path}: 'append_default_prompt' must be true or false, got "
                f"{type(append_default).__name__}"
            )
        if append_default and not isinstance(data.get("system_prompt"), str):
            raise PersonaLoadError(
                f"{file_path}: 'append_default_prompt' needs a 'system_prompt' to append to"
            )

        raw_llm: object = data.get("llm_config") or data.get("llm")
        if isinstance(raw_llm, dict):
            raw_llm_dict = cast(dict[object, object], raw_llm)
            llm_dict: dict[str, Any] = {str(k): v for k, v in raw_llm_dict.items()}
            tier: object = llm_dict.get("model_tier")
            if isinstance(tier, str):
                try:
                    llm_dict["model_tier"] = ModelTier(tier)
                except ValueError as exc:
                    raise PersonaLoadError(
                        f"{file_path}: 'model_tier' {tier!r} is not a tier; use one of "
                        f"{', '.join(t.value for t in ModelTier)}"
                    ) from exc
            data["llm_config"] = AgentLLMConfig(**llm_dict)
            data.pop("llm", None)

        persona = PersonaDefinition.model_validate(data)
        return _with_default_prompt(persona) if append_default else persona

    def _validate_tools(self, persona: PersonaDefinition, *, source: Path | str) -> None:
        if self._tool_names is None or not persona.allowed_tools:
            return
        unknown = [name for name in persona.allowed_tools if name not in self._tool_names]
        if not unknown:
            return
        known = sorted(self._tool_names)
        details = "; ".join(
            f"{name!r}"
            + (
                f" (did you mean {', '.join(repr(m) for m in matches)}?)"
                if (matches := difflib.get_close_matches(name, known, n=2, cutoff=0.5))
                else ""
            )
            for name in unknown
        )
        raise PersonaLoadError(
            f"{source}: persona {persona.name!r} declares tool(s) that are not registered: "
            f"{details}. Registered: {', '.join(known)}"
        )

    def get_persona(self, name: str) -> PersonaDefinition | None:
        return self._personas.get(name)

    def has_persona(self, name: str) -> bool:
        return name in self._personas

    def list_personas(self) -> Sequence[PersonaDefinition]:
        personas = list(self._personas.values())
        return sorted(
            personas,
            key=lambda p: (0 if p.name == DEFAULT_PERSONA_NAME else 1, p.name),
        )

    def source_of(self, name: str) -> Path | None:
        return self._sources.get(name)

    def save_persona(
        self, draft: PersonaDraft, *, create: bool, allow_override: bool = False
    ) -> PersonaDefinition:
        if self._read_only:
            raise PersonaWriteConflict(f"Store for directory '{self._directory}' is read-only.")
        _refuse_an_unwritable_name(draft.name)
        if create and draft.name in self._personas:
            raise PersonaWriteConflict(
                f"a persona named {draft.name!r} already exists. Edit it, or choose another name."
            )
        if not allow_override:
            if not create and draft.name not in self._personas:
                raise PersonaNotFound(f"no persona named {draft.name!r} is loaded.")

        directory = self._directory
        source = self._sources.get(draft.name)
        in_place = source is not None and source.parent == directory
        target = source if in_place and source is not None else directory / f"{draft.name}.yaml"
        if target != source and (target.exists() or target.is_symlink()):
            raise PersonaWriteConflict(
                f"{target} already exists and defines a different persona. Rename or remove it."
            )
        resolved_target = target.resolve()
        if resolved_target.parent != directory.resolve():
            raise PersonaWriteRefused(
                f"{target} resolves to {resolved_target}, outside the personas directory "
                f"{directory}; it is not written through. Replace the link with a file."
            )

        directory.mkdir(parents=True, exist_ok=True)
        text = yaml.dump(
            _file_contents(draft), Dumper=_LiteralBlockDumper, allow_unicode=True, sort_keys=False
        )
        fd, temp_name = tempfile.mkstemp(dir=directory, prefix=f".{draft.name}.", suffix=".tmp")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            try:
                persona = self._parse_file(temp_path, label=target)
            except (PersonaLoadError, Exception) as exc:
                raise PersonaWriteRefused(str(exc)) from exc
            os.replace(temp_path, target)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        self._personas[persona.name] = persona
        self._sources[persona.name] = target
        return persona


class InMemoryPersonaStore:
    """In-memory dictionary store for testing and transient backends."""

    def __init__(
        self,
        personas: Sequence[PersonaDefinition] | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        self._read_only = read_only
        self._personas: dict[str, PersonaDefinition] = {p.name: p for p in (personas or [])}

    @property
    def read_only(self) -> bool:
        return self._read_only

    def get_persona(self, name: str) -> PersonaDefinition | None:
        return self._personas.get(name)

    def has_persona(self, name: str) -> bool:
        return name in self._personas

    def list_personas(self) -> Sequence[PersonaDefinition]:
        personas = list(self._personas.values())
        return sorted(
            personas,
            key=lambda p: (0 if p.name == DEFAULT_PERSONA_NAME else 1, p.name),
        )

    def source_of(self, name: str) -> Path | str | None:
        return f"<memory:{name}>" if name in self._personas else None

    def register_persona(self, persona: PersonaDefinition) -> None:
        """Register or overwrite a persona in memory."""
        self._personas[persona.name] = persona

    def save_persona(
        self, draft: PersonaDraft, *, create: bool, allow_override: bool = False
    ) -> PersonaDefinition:
        if self._read_only:
            raise PersonaWriteConflict("InMemoryPersonaStore is read-only.")
        try:
            refuse_an_unusable_username(draft.name)
        except AgentHomeError as exc:
            raise PersonaWriteRefused(
                f"{draft.name!r} cannot be a persona name: {exc} The name is both the file name and "
                f"the agent id, so choose one that fits that rule."
            ) from exc
        if create and self.has_persona(draft.name):
            raise PersonaWriteConflict(
                f"a persona named {draft.name!r} already exists. Edit it, or choose another name."
            )
        if not create and not allow_override and not self.has_persona(draft.name):
            raise PersonaNotFound(f"no persona named {draft.name!r} is loaded.")

        llm_dict: dict[str, Any] = {
            "model_tier": ModelTier(draft.model_tier),
            "temperature": float(draft.temperature),
        }
        if draft.model_name is not None:
            llm_dict["model_name"] = draft.model_name
        if draft.max_tokens is not None:
            llm_dict["max_tokens"] = draft.max_tokens

        persona = PersonaDefinition(
            name=draft.name,
            role=draft.role,
            description=draft.description,
            system_prompt=draft.system_prompt,
            allowed_tools=tuple(draft.allowed_tools),
            llm_config=AgentLLMConfig(**llm_dict),
            enable_write_tools=draft.enable_write_tools,
            enable_subagent_tools=draft.enable_subagent_tools,
            a2a_peers=tuple(draft.a2a_peers),
        )
        if draft.append_default_prompt:
            persona = _with_default_prompt(persona)
        self._personas[persona.name] = persona
        return persona


class CompositePersonaStore:
    """Chains multiple PersonaStoreProtocol instances with precedence (first found wins).

    Reads search stores in given sequence order.
    Writes delegate to the primary writable store.
    """

    def __init__(
        self,
        stores: Sequence[PersonaStoreProtocol],
        writable_store: PersonaStoreProtocol | None = None,
        overridable_dirs: Sequence[Path] | None = None,
    ) -> None:
        self._stores: list[PersonaStoreProtocol] = list(stores)
        self._overridable_dirs: tuple[Path, ...] = (
            tuple(overridable_dirs) if overridable_dirs is not None else (BUILTIN_PERSONAS_DIR,)
        )
        if writable_store is not None:
            self._writable_store: PersonaStoreProtocol | None = writable_store
        else:
            writable = None
            for s in self._stores:
                if not getattr(s, "read_only", False):
                    writable = s
                    break
            self._writable_store = writable

    @property
    def stores(self) -> Sequence[PersonaStoreProtocol]:
        return tuple(self._stores)

    @property
    def writable_store(self) -> PersonaStoreProtocol | None:
        return self._writable_store

    def get_persona(self, name: str) -> PersonaDefinition | None:
        for store in self._stores:
            persona = store.get_persona(name)
            if persona is not None:
                return persona
        return None

    def has_persona(self, name: str) -> bool:
        return any(store.has_persona(name) for store in self._stores)

    def source_of(self, name: str) -> Path | str | None:
        for store in self._stores:
            if store.has_persona(name):
                return store.source_of(name)
        return None

    def list_personas(self) -> Sequence[PersonaDefinition]:
        seen: set[str] = set()
        result: list[PersonaDefinition] = []
        for store in self._stores:
            for p in store.list_personas():
                if p.name not in seen:
                    seen.add(p.name)
                    result.append(p)
        return sorted(
            result,
            key=lambda p: (0 if p.name == DEFAULT_PERSONA_NAME else 1, p.name),
        )

    def reload(self) -> None:
        for store in self._stores:
            reload_fn = getattr(store, "reload", None)
            if callable(reload_fn):
                reload_fn()

    def save_persona(self, draft: PersonaDraft, *, create: bool) -> PersonaDefinition:
        if self._writable_store is None:
            raise PersonaWriteConflict(
                "this runtime has no workspace directory, so it has nowhere to save a persona."
            )
        other_has = any(
            s.has_persona(draft.name) for s in self._stores if s is not self._writable_store
        )
        if create and other_has:
            raise PersonaWriteConflict(
                f"a persona named {draft.name!r} already exists. Edit it, or choose another name."
            )
        if not create and not self.has_persona(draft.name):
            raise PersonaNotFound(f"no persona named {draft.name!r} is loaded.")

        source = self.source_of(draft.name)
        writable_dir = getattr(self._writable_store, "directory", None)
        if (
            source is not None
            and isinstance(source, Path)
            and writable_dir is not None
            and source.parent not in (writable_dir, *self._overridable_dirs)
        ):
            raise PersonaWriteConflict(
                f"persona {draft.name!r} is defined in {source}, which is not the workspace "
                f"directory {writable_dir}. Edit that file directly."
            )

        save_fn: Any = getattr(self._writable_store, "save_persona", None)
        if callable(save_fn):
            try:
                return cast(PersonaDefinition, save_fn(draft, create=create, allow_override=True))
            except TypeError:
                return self._writable_store.save_persona(draft, create=create)
        return self._writable_store.save_persona(draft, create=create)
