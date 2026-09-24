"""Generic BaseTool abstraction with declarative Pydantic schemas and strict containment."""

from __future__ import annotations

import abc
import inspect
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

from uclone_x.core.provenance import Provenance
from uclone_x.errors import PathTraversalError
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.protocols import ToolProtocol

TParams = TypeVar("TParams", bound=BaseModel)
TTool = TypeVar("TTool", bound=ToolProtocol)


def tool_writes_files(tool: object) -> bool:
    """Whether executing `tool` can create, modify or delete a file on the host (#1167).

    This is the whole definition of a "write tool", the thing a persona's
    `enable_write_tools: false` refuses. The rule (owner-delegated ruling on #1167,
    ):

    * It is **declared on the tool** as a `writes_files` attribute and never inferred from
      the name. Name-based notions contradicted each other -- the approval hook's
      prefixes caught `run_command` but not `bash_run`, which is the same class under
      another name, and neither `file_write` nor `file_edit`, until #1463 made
      `HumanApprovalHook._is_destructive` read this flag first; the evaluation answerer's
      `WITHHELD_TOOLS` names the two file tools and keeps the shell. A name is not a
      capability.
    * **Anything that can put bytes on the host counts**, the shells included: the flag is
      a promise to the person who set it. Writing the agent's *own* state (its memory
      facts, its plan, which skills it loaded) does not count; those are not the user's
      files.
    * **Undeclared means writing.** A tool that does not declare the attribute cannot be
      classified -- an MCP server's tool, a handler registered as a `LocalTool` -- and a
      restriction must not be widened by what cannot be classified.
    """
    declared: object = getattr(tool, "writes_files", True)
    return declared is not False


def tool_spawns_subagents(tool: object) -> bool:
    """Whether executing `tool` starts another agent -- what `enable_subagent_tools` refuses.

    Declared, like `writes_files`, and **undeclared means no**. The asymmetry is
    deliberate: a tool can only start an agent through `BaseAgent.spawn_subagent`, which
    refuses on the same flag, so an undeclared tool that somehow reached it is still stopped
    there. No such backstop exists for a tool that writes a file.
    """
    declared: object = getattr(tool, "spawns_subagents", False)
    return declared is True


def drop_shadowed_aliases(tools: Sequence[TTool]) -> list[TTool]:
    """`tools` without any alias whose canonical tool is also in it (#1424).

    One behaviour is advertised under one name. A tool registered only for compatibility
    declares `alias_of = "<canonical name>"`; it stays registered, so a call to it still
    resolves and a persona that lists it still loads, but a request that already carries
    the canonical tool does not also carry the alias. A list holding the alias alone -- a
    persona that permits only the old name -- keeps it, so that persona keeps a shell.
    """
    names = {tool.name for tool in tools}
    return [tool for tool in tools if getattr(tool, "alias_of", None) not in names]


class BaseTool(abc.ABC, Generic[TParams]):
    """Generic base tool providing declarative schema generation, validation, and containment."""

    name: str = ""
    description: str = ""
    params_type: type[TParams] | None = None
    #: Whether executing this tool can create, modify or delete a file on the host. See
    #: `tool_writes_files`. `True` here, so a subclass that says nothing is refused under
    #: `enable_write_tools: false` rather than trusted; every read-only tool says `False`.
    writes_files: ClassVar[bool] = True
    #: Whether executing this tool starts another agent. See `tool_spawns_subagents`.
    spawns_subagents: ClassVar[bool] = False
    #: What a call refused on its arguments did *not* do, in words the model reads next to
    #: the refusal. A refusal that names only a field and a type leaves the model to guess
    #: whether anything happened, and one model guessed that it had (#1375). A tool whose
    #: effect has a plainer name says it: the memory tool says nothing was saved.
    not_run_note: ClassVar[str] = "The tool did not run, so nothing was done."

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        params_type: type[TParams] | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if params_type is not None:
            self.params_type = params_type

        self._path_validator = PathValidator()

    @classmethod
    def _resolve_params_type(cls) -> type[TParams]:
        """Resolve the TParams type from class definition or generic arguments."""
        if cls.params_type is not None:
            return cls.params_type

        # Check __orig_bases__ on class or subclasses
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is not None and issubclass(origin, BaseTool):
                args = get_args(base)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    return args[0]  # type: ignore[return-value]

        # Check MRO for any generic BaseTool
        for base_cls in cls.__mro__:
            for base in getattr(base_cls, "__orig_bases__", ()):
                args = get_args(base)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                    return args[0]  # type: ignore[return-value]

        raise TypeError(
            f"Tool '{cls.__name__}' must define `params_type` or inherit from `BaseTool[YourParams]`"
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """Return JSON Schema for the tool parameters model."""
        model_cls = self._resolve_params_type()
        return model_cls.model_json_schema()

    def resolve_safe_path(self, target_path: str | Path, workspace_root: Path) -> Path:
        """Enforce path containment within workspace_root using PathValidator."""
        return self._path_validator.resolve_safe_path(Path(target_path), workspace_root)

    def resolve_read_path(self, target_path: str | Path, context: ToolContext) -> tuple[Path, Path]:
        """Resolve a path a read-only tool may read, and the root that admitted it.

        With no read roots this is `resolve_safe_path` against the workspace, unchanged.
        With read roots, a path that names an existing workspace entry still means that
        entry; otherwise an absolute or `~` path inside one of `context.read_roots` is
        accepted. Only read-only tools call this; a writing tool keeps `resolve_safe_path`,
        so a read root is never writable.
        """
        workspace = context.require_workspace()
        if not context.read_roots:
            return self.resolve_safe_path(target_path, workspace), workspace.resolve()
        try:
            inside = self.resolve_safe_path(target_path, workspace)
        except PathTraversalError:
            inside = None
        # A writing tool never expands `~`: `~/notes` names `<workspace>/~/notes` for it. A
        # file there is therefore what the clone wrote, and it wins over the read roots.
        if inside is not None and inside.exists():
            return inside, workspace.resolve()
        candidate = Path(target_path).expanduser()
        if candidate.is_absolute():
            for root in context.read_roots:
                try:
                    return self.resolve_safe_path(candidate, root), root.resolve()
                except PathTraversalError:
                    continue
        if inside is not None:
            return inside, workspace.resolve()
        allowed = ", ".join(str(root) for root in context.read_roots)
        raise PathTraversalError(
            f"Path '{target_path}' is outside the workspace '{workspace}' and every "
            f"read-only folder ({allowed}). Pass an absolute path inside one of them."
        )

    @staticmethod
    def display_path(path: Path, root: Path, context: ToolContext) -> str:
        """How a result names `path`: workspace-relative, or absolute under a read root.

        Absolute for a read root because a relative name would resolve against the
        workspace when the model passes it back.
        """
        workspace = context.require_workspace().resolve()
        if root == workspace:
            return str(path.relative_to(workspace))
        return str(path)

    async def execute(
        self,
        params: dict[str, Any] | ToolContext | None = None,
        context: ToolContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the tool with parameter validation, execution timing, and error handling."""
        start_time = time.perf_counter()
        tool_identifier = self.name or self.__class__.__name__

        # 1. Resolve context and parameters dictionary
        actual_context: ToolContext
        actual_params_dict: dict[str, Any]

        if isinstance(params, ToolContext):
            actual_context = params
            actual_params_dict = kwargs
        elif isinstance(context, ToolContext):
            actual_context = context
            if isinstance(params, dict):
                actual_params_dict = {**params, **kwargs}
            else:
                actual_params_dict = kwargs
        elif "context" in kwargs and isinstance(kwargs["context"], ToolContext):
            actual_context = kwargs.pop("context")
            if isinstance(params, dict):
                actual_params_dict = {**params, **kwargs}
            else:
                actual_params_dict = kwargs
        else:
            return ToolResult(
                success=False,
                error="Tool execution requires a valid ToolContext",
                execution_time_ms=0.0,
                isolation_level=None,
                provenance=Provenance.primary(
                    provider="local.builtin",
                    model=tool_identifier,
                ),
            )

        # 2. Parameter parsing and validation against TParams
        try:
            model_cls = self._resolve_params_type()
            validated_params = model_cls.model_validate(actual_params_dict)
        except ValidationError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=(
                    f"Parameter validation failed for tool '{tool_identifier}'. "
                    f"{self.not_run_note} {e}"
                ),
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(
                    provider="local.builtin",
                    model=tool_identifier,
                ),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Failed to parse parameters for tool '{tool_identifier}': {e}",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(
                    provider="local.builtin",
                    model=tool_identifier,
                ),
            )

        # 3. Execution invocation
        try:
            # Check parameter order of run method
            sig = inspect.signature(self.run)
            param_names = [p.name for p in sig.parameters.values() if p.name != "self"]
            if len(param_names) >= 2 and param_names[0] == "context":
                res = self.run(actual_context, validated_params)  # type: ignore[arg-type]
            else:
                res = self.run(validated_params, actual_context)

            if inspect.isawaitable(res):
                output = await res
            else:
                output = res

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=True,
                output=output,
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(
                    provider="local.builtin",
                    model=tool_identifier,
                ),
            )
        except PathTraversalError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Path traversal violation in tool '{tool_identifier}': {e}",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(
                    provider="local.builtin",
                    model=tool_identifier,
                ),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=f"Tool execution failed for '{tool_identifier}': {type(e).__name__}: {e}",
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(
                    provider="local.builtin",
                    model=tool_identifier,
                ),
            )

    @abc.abstractmethod
    def run(self, params: TParams, context: ToolContext) -> Any:
        """Abstract execution logic to be implemented by concrete tools."""
        ...
