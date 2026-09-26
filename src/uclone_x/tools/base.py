"""Generic BaseTool abstraction with declarative Pydantic schemas and strict containment."""

from __future__ import annotations

import abc
import inspect
import logging
import os
import secrets
import stat
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
)
from urllib.parse import quote

from pydantic import BaseModel, ValidationError

from uclone_x.core.provenance import Provenance
from uclone_x.errors import PathTraversalError, PlainRefusalError
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.protocols import ToolProtocol

logger = logging.getLogger(__name__)

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


def tool_call_writes_files(tool: object, arguments: object) -> bool:
    """Whether this call of `tool` can write a file: what a call's record says (#1584).

    `tool_writes_files` is a property of the tool, and it is what a persona's
    `enable_write_tools: false` refuses: a tool that can write under any action is a write
    tool. A single call is narrower. A tool whose `action` argument picks between reading
    and writing declares `read_actions`, the actions that only read, and a call to one of
    them did not write -- so a room does not count it as a file the conversation wrote.
    Undeclared means none: every call of a writing tool may write.
    """
    if not tool_writes_files(tool):
        return False
    declared: object = getattr(tool, "read_actions", None)
    if not isinstance(declared, frozenset) or not isinstance(arguments, dict):
        return True
    action: object = cast(dict[str, object], arguments).get("action")
    return not (isinstance(action, str) and action in cast(frozenset[object], declared))


def in_story_library(resolved: Path, workspace_root: Path) -> bool:
    """Whether `resolved`, a path `resolve_safe_path` returned, is in `<workspace>/stories`.

    Decided by what the path *is*, not how it was spelled (#1583). `resolved` has had its
    symlinks followed already, so a link pointing into the library arrives as the library
    path. Two further ways to name the library remain, and both are refused:

    * **Another spelling of the folder name.** A case-insensitive disk opens `Stories`,
      `ſtories` or `ﬆories` as `stories`, so the first component under the workspace is
      compared after Unicode case folding. This is checked whether or not the library exists
      yet, so no file can be planted where a story will be created.
    * **The same folder under another name**, such as `stories` being a link to another
      folder in the workspace: every existing folder on the way to `resolved` is compared by
      file identity with the library.

    A file hardlinked into the library under another name is not detected. `file_write`,
    `file_edit`, both `generate_image` tools and `character_sheet` write a new file and then
    give it the name (`replace_file`), so writing such a link detaches it and
    leaves the story's copy unchanged. A new tool that writes a model-chosen path has to do
    the same.
    """
    # Imported here: `uclone_x.story` imports `uclone_x.tools.models`, which loads this module.
    from uclone_x.story.schemas import STORIES_DIRNAME

    root = workspace_root.resolve()
    parts = resolved.relative_to(root).parts
    if parts and parts[0].casefold() == STORIES_DIRNAME:
        return True
    library = root / STORIES_DIRNAME
    if not library.exists():
        return False
    for folder in (resolved, *resolved.parents):
        if folder == root:
            return False
        if folder.exists() and folder.samefile(library):
            return True
    return False


def artifact_content_url(rel_path: str) -> str:
    """The rooted URL the UI serves a workspace file at -- the one link a reply should carry.

    Rooted, with no scheme or host: the page resolves it against wherever it was opened
    from. The image tools' results name no absolute filesystem path beside it. They used
    to carry `absolute_path`, and a model handed `/.../ucx-fresh-test2-pypi022/work/...`
    next to `/api/artifacts/content?...` joined the two into
    `https://ucx-fresh-test2-pypi022/work/api/artifacts/...`, a link to nowhere (#1618).
    The path is percent-encoded (`/` kept readable) so a space or a parenthesis in a file
    name cannot end a markdown link early.
    """
    return f"/api/artifacts/content?path={quote(rel_path, safe='/')}"


def replace_file(path: Path, data: bytes) -> None:
    """Write `data` to `path` as a new file that then takes the name.

    Writing into the existing file would also change every other name for it, so a
    hardlink planted outside the story library could rewrite a story (#1583). A new
    file detaches that name instead. `file_write`, `file_edit`, both `generate_image`
    tools and `character_sheet` write through this.

    * **The mode is kept.** A file that already exists keeps its permission bits, so a
      `0600` file stays private; a new file gets the default the umask gives (#1589).
    * **The temporary name is short**, in the same folder so the rename stays on one
      disk, and hidden. Adding a suffix to `path.name` instead made a name that is
      itself near the 255-byte limit fail as too long (#1589).
    * **The mode is set before any data is written**, so the new contents of a `0600`
      file never sit in a file others can read, even for a moment (#1589 follow-up a).
    * **The descriptor is closed on every path**, including a failure to wrap it in a
      file object (#1589 follow-up d).
    """
    try:
        mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = None
    tmp_file = path.parent / f".ucx-{secrets.token_hex(6)}.tmp"
    # O_EXCL: never write through something already at the temporary name, a link included.
    descriptor = os.open(tmp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        try:
            if mode is not None:
                _set_mode(descriptor, tmp_file, mode)
            handle = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            handle.write(data)
        os.replace(tmp_file, path)
    except BaseException:
        tmp_file.unlink(missing_ok=True)
        raise


def _set_mode(descriptor: int, path: Path, mode: int) -> None:
    """Give the open file `mode`. `os.fchmod` is missing on Windows before Python 3.13."""
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, mode)
    else:
        os.chmod(path, mode)


def tool_spawns_subagents(tool: object) -> bool:
    """Whether executing `tool` starts another agent -- what `enable_subagent_tools` refuses.

    Declared, like `writes_files`, and **undeclared means no**. The asymmetry is
    deliberate: a tool can only start an agent through `BaseAgent.spawn_subagent`, which
    refuses on the same flag, so an undeclared tool that somehow reached it is still stopped
    there. No such backstop exists for a tool that writes a file.
    """
    declared: object = getattr(tool, "spawns_subagents", False)
    return declared is True


def tool_opens_story(tool: object) -> bool:
    """Whether a successful call of `tool` can change which story its conversation has open.

    Declared, and **undeclared means no**: a tool that says nothing cannot move a
    conversation's story. A declaring tool names the story in its output under
    `uclone_x.story.OPEN_STORY_KEY`, and the runtime reads that key only from a tool that
    declares this (#1555) -- the shape of an output is not a capability.
    """
    declared: object = getattr(tool, "opens_story", False)
    return declared is True


def tool_needs_room(tool: object) -> bool:
    """Whether `tool` is kept out of a turn made outside a conversation (a room).

    Declared, and undeclared means no. A tool declares it when its schema would cost every
    such request room in the window for nothing a turn there can use: every call is
    refused, or -- `story_library` -- the one call that works there, 'list', lists stories
    that can be opened only inside a conversation (#1556, #1576). It says what is offered,
    not what a call does: a direct call of such a tool still runs, and is refused or
    answered by the tool itself.
    """
    declared: object = getattr(tool, "needs_room", False)
    return declared is True


def tool_call_needs_approval(tool: object, arguments: object) -> bool:
    """Whether this call of `tool` may run only once a person has approved it (#1557).

    Declared on the tool as `approval_actions`, the values of its `action` argument that
    need a person; undeclared means none. The runtime asks the person before such a call
    (`HookRunner.run_hooks` answers `ASK` whatever the hooks said) and tells the tool it
    was approved through `ToolContext.approved_by_person`, which only the runtime sets --
    nothing the model sends can say a call was approved.
    """
    declared: object = getattr(tool, "approval_actions", None)
    if not isinstance(declared, frozenset):
        return False
    if not isinstance(arguments, dict):
        return False
    action: object = cast(dict[str, object], arguments).get("action")
    return isinstance(action, str) and action in cast(frozenset[object], declared)


def tool_approval_timeout_note(tool: object) -> str | None:
    """What `tool` says when its approval request went unanswered, or `None` (#1557).

    The runtime refuses such a call (fail-closed). A tool whose refusal the person will
    hear about declares `approval_timeout_note` in plain words, so they are told what did
    not happen and why, rather than that a request "timed out".
    """
    declared: object = getattr(tool, "approval_timeout_note", None)
    return declared if isinstance(declared, str) and declared else None


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


#: Type names pydantic puts in an error's location when a value fails every member of a
#: union (`('size', 'int')`, `('size', 'list[str]')`). They are not argument names.
_UNION_TAGS = frozenset(
    {"str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple", "none", "any"}
)


class _Names:
    """The names an error's location can hold for `params_model`, gathered from the model.

    `arguments` are the field names and aliases at every depth: a part that is one of them
    is an argument, whatever its spelling -- an alias may start with a capital or hold a
    hyphen (#1602). `forms` are what pydantic puts in the location to say which member of a
    union of models it tried: the member's class name, or the tag value of a union with a
    discriminator. They are not arguments either.
    """

    def __init__(self, params_model: type[BaseModel] | None) -> None:
        self.arguments: set[str] = set()
        self.forms: set[str] = set()
        self._seen: set[type[BaseModel]] = set()
        if params_model is not None:
            self._visit(params_model)

    def _visit(self, annotation: object) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in self._seen:
                return
            self._seen.add(annotation)
            self.forms.add(annotation.__name__)
            for name, field in annotation.model_fields.items():
                self.arguments.add(name)
                for alias in (field.alias, field.validation_alias):
                    if isinstance(alias, str):
                        self.arguments.add(alias)
                if isinstance(field.discriminator, str):
                    self.forms.update(_tag_values(field.annotation, field.discriminator))
                self._visit(field.annotation)
            return
        if get_origin(annotation) is Annotated:
            # A discriminator on a union inside another type (`Annotated[A | B, ...] | None`)
            # stays in the metadata rather than moving to the field.
            inner, *metadata = get_args(annotation)
            for meta in metadata:
                tag = getattr(meta, "discriminator", None)
                if isinstance(tag, str):
                    self.forms.update(_tag_values(inner, tag))
            self._visit(inner)
            return
        for arg in get_args(annotation):
            self._visit(arg)


def _tag_values(annotation: object, discriminator: str) -> set[str]:
    """The values of `discriminator` that pick a member of the union `annotation`."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        field = annotation.model_fields.get(discriminator)
        if field is None:
            return set()
        return {str(value) for value in get_args(field.annotation)}
    values: set[str] = set()
    for arg in get_args(annotation):
        values |= _tag_values(arg, discriminator)
    return values


def _is_argument_part(part: int | str, names: _Names) -> bool:
    if isinstance(part, int):
        return True
    if part in names.arguments:
        return True
    if not part or part in names.forms or part.lower() in _UNION_TAGS or part[0].isupper():
        return False
    return not any(mark in part for mark in "[]-(),: ")


def _argument_path(loc: Sequence[int | str], names: _Names) -> str:
    text = ""
    for part in loc:
        if not _is_argument_part(part, names):
            continue
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text


def _form_at(loc: Sequence[int | str], names: _Names) -> int | None:
    """Where in `loc` pydantic names the member of a union of models it tried, if it does."""
    for index, part in enumerate(loc):
        if isinstance(part, str) and part not in names.arguments and part in names.forms:
            return index
    return None


def _plural(count: object, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _expected(error: Mapping[str, Any]) -> str:
    """What was expected of the value `error` is about, in words, with the limit it broke."""
    kind = str(error.get("type", ""))
    ctx: Mapping[str, Any] = error.get("ctx") or {}
    if kind in {"value_error", "assertion_error"} and ctx.get("error") is not None:
        # The tool's own validator wrote this sentence; it is passed on without the
        # validator's "Value error, " prefix.
        return f": {str(ctx['error']).rstrip('.')}"
    if kind.startswith(PLAIN_ERROR_PREFIX) and error.get("msg"):
        # A `PydanticCustomError` the tool opted in: its message is the tool's own sentence
        # (#1602). Every other custom type -- pydantic raises about thirty of its own, some
        # quoting the value they were given -- keeps the plain fallback below.
        return f": {str(error['msg']).rstrip('.')}"
    if kind == "union_tag_invalid" and "discriminator" in ctx and "expected_tags" in ctx:
        # The tag the call gave is not repeated; the ones that would do are.
        return f"should have {ctx['discriminator']} set to one of {ctx['expected_tags']}"
    if kind in {"literal_error", "enum"} and "expected" in ctx:
        allowed = str(ctx["expected"])
        if " or " not in allowed:  # one allowed value: a union member's tag, say
            return f"should be {allowed}"
        return f"should be one of {allowed}"
    if kind == "string_too_short":
        least = ctx.get("min_length", 1)
        if least == 1:
            return "should not be empty"
        return f"should be at least {_plural(least, 'character', 'characters')} long"
    if kind == "string_too_long" and "max_length" in ctx:
        return f"should be at most {_plural(ctx['max_length'], 'character', 'characters')} long"
    if kind == "too_short" and "min_length" in ctx:
        return f"should have at least {_plural(ctx['min_length'], 'item', 'items')}"
    if kind == "too_long" and "max_length" in ctx:
        return f"should have at most {_plural(ctx['max_length'], 'item', 'items')}"
    for key, words in (
        ("ge", "at least"),
        ("gt", "more than"),
        ("le", "at most"),
        ("lt", "less than"),
    ):
        if kind.startswith(("greater_than", "less_than")) and key in ctx:
            return f"should be {words} {ctx[key]}"
    if kind == "string_pattern_mismatch" and "pattern" in ctx:
        return f"is not in the form it needs (it has to match {ctx['pattern']})"
    return _EXPECTED.get(kind, "has a value of the wrong kind")


#: The error type a tool's own `PydanticCustomError` starts with, for its message to reach
#: the refusal: `PydanticCustomError("plain_name_taken", "that name is already taken")`.
#: Opt-in, because pydantic raises `PydanticCustomError` itself (`zoneinfo_str`,
#: `import_error`, `sequence_str`, ...), with messages that name its own types or repeat the
#: value given; none of them starts with this. Write the message for the person reading
#: the conversation, and put a value in it only if they should see it (#1602).
PLAIN_ERROR_PREFIX = "plain_"

_EXPECTED: dict[str, str] = {
    "missing": "is missing",
    "extra_forbidden": "is not an argument this tool takes",
    "string_type": "should be text",
    "int_type": "should be a whole number",
    "int_parsing": "should be a whole number",
    "int_from_float": "should be a whole number",
    "float_type": "should be a number",
    "float_parsing": "should be a number",
    "decimal_type": "should be a number",
    "bool_type": "should be true or false",
    "bool_parsing": "should be true or false",
    "list_type": "should be a list",
    "tuple_type": "should be a list",
    "set_type": "should be a list",
    "dict_type": "should be an object of named values",
    "model_type": "should be an object of named values",
    "model_attributes_type": "should be an object of named values",
    "none_required": "should be left out",
}


def _problem_lines(problems: Mapping[str, list[str]]) -> list[str]:
    lines: list[str] = []
    for where, reasons in problems.items():
        for reason in reasons:
            if reason.startswith(": "):  # the validator's own sentence
                lines.append(f"'{where}'{reason}" if where else reason[2:])
        kept = [reason for reason in reasons if not reason.startswith(": ")]
        if len(kept) > 1 and all(reason.startswith("should be ") for reason in kept):
            # A value that fit no member of a union: "should be a whole number or text".
            kept = [kept[0], *(reason.removeprefix("should be ") for reason in kept[1:])]
        said = " or ".join(kept)
        if said:
            lines.append(f"'{where}' {said}" if where else f"the arguments {said}")
    return lines


def describe_invalid_arguments(
    tool: str, error: ValidationError, params_model: type[BaseModel] | None, not_done: str
) -> str:
    """The refusal of a call to `tool` whose arguments `params_model` did not accept (#1570).

    The model reads it and calls again, so it names each argument that did not fit, says
    what was expected of it -- the allowed values, the limit it broke, the missing name --
    and, when an argument was missing or unknown, lists the ones the tool takes. It says
    that `not_done` (see `BaseTool.not_run_note`), so nobody reads the refusal as a call
    that ran (#1375). The validator's own report is not shown: its wording, type and class
    names and its documentation links mean nothing to a person reading the conversation.
    """
    names = _Names(params_model)
    # Keyed by the argument a line is about. A plain entry holds what was expected of it;
    # a union-of-models entry holds, per member it could have been, what that member
    # expected of each of its parts.
    problems: dict[str, list[str]] = {}
    unions: dict[str, dict[str, dict[str, list[str]]]] = {}
    order: list[tuple[bool, str]] = []
    names_matter = False
    for item in error.errors():
        kind = str(item.get("type", ""))
        names_matter = names_matter or kind in {"missing", "extra_forbidden"}
        loc: Sequence[int | str] = item.get("loc", ())
        expected = _expected(cast(Mapping[str, Any], item))
        form = _form_at(loc, names)
        if form is None:
            where = _argument_path(loc, names)
            reasons = problems.setdefault(where, [])
            if (False, where) not in order:
                order.append((False, where))
        else:
            where = _argument_path(loc[:form], names)
            members = unions.setdefault(where, {})
            if (True, where) not in order:
                order.append((True, where))
            reasons = members.setdefault(str(loc[form]), {}).setdefault(
                _argument_path(loc, names), []
            )
        if expected not in reasons:
            reasons.append(expected)
    lines: list[str] = []
    for is_union, where in order:
        if not is_union:
            if where not in unions:
                lines.extend(_problem_lines({where: problems[where]}))
            continue
        # A value that fit no member of a union of models: what each member wanted, as one
        # line, so the model can pick one and supply all of it (#1602). A member that is not
        # a model (`int | Sketch`) left its error on the argument itself; it is one more
        # form, not a second thing to fix.
        members = list(unions[where].values())
        if where in problems:
            members.insert(0, {where: problems[where]})
        if all(set(member) == {where} for member in members):
            # Every form is about the value itself: "should be a whole number or an object".
            reasons = [reason for member in members for reason in member[where]]
            lines.extend(_problem_lines({where: list(dict.fromkeys(reasons))}))
            continue
        wanted = [" and ".join(_problem_lines(member)) for member in members]
        wanted = list(dict.fromkeys(wanted))
        if len(wanted) == 1:
            lines.append(wanted[0])
        else:
            subject = f"'{where}'" if where else "the arguments"
            lines.append(
                f"{subject} did not fit any of the forms it can take: either "
                + ", or ".join(wanted)
            )
    shown = lines[:5]
    more = len(lines) - len(shown)
    tail = f" (and {more} more)" if more else ""
    text = (
        f"The call to '{tool}' was refused because its arguments did not fit: "
        f"{'; '.join(shown)}{tail}. {not_done}"
    )
    if names_matter and params_model is not None:
        takes = ", ".join(
            f"{field.alias or name} (required)" if field.is_required() else field.alias or name
            for name, field in params_model.model_fields.items()
        )
        text += f" It takes: {takes}." if takes else " It takes no arguments."
    return f"{text} Call it again with the arguments corrected."


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
    #: Whether a successful call can change the conversation's open story. See
    #: `tool_opens_story`.
    opens_story: ClassVar[bool] = False
    #: Whether the tool is kept out of a turn outside a conversation (a room). See
    #: `tool_needs_room`.
    needs_room: ClassVar[bool] = False
    #: The values of the `action` argument that only read, on a tool that writes under its
    #: other actions. See `tool_call_writes_files`.
    read_actions: ClassVar[frozenset[str]] = frozenset()
    #: The values of the `action` argument that run only once a person approves the call.
    #: See `tool_call_needs_approval`.
    approval_actions: ClassVar[frozenset[str]] = frozenset()
    #: What the refusal says when nobody answered the approval request. See
    #: `tool_approval_timeout_note`.
    approval_timeout_note: ClassVar[str | None] = None
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

    def resolve_write_path(self, target_path: str | Path, workspace_root: Path) -> Path:
        """`resolve_safe_path`, refusing a path inside the story library (#1583).

        What a general tool writes to a path the model chose. A story is written only by
        the story tools, which check the lease, the digest and a person's approval; a
        general tool writing under `stories/` skipped all three. Reading a story file stays
        allowed: read-only tools keep `resolve_read_path`.

        Raises:
            PathTraversalError: the path escapes the workspace.
            PlainRefusalError: the path is in the story library (`in_story_library`).
        """
        resolved = self.resolve_safe_path(target_path, workspace_root)
        if in_story_library(resolved, workspace_root):
            raise PlainRefusalError(
                f"'{target_path}' is in the story library, so it was not written. Stories "
                "are changed with the story tools (story_manuscript, story_outline, "
                "story_codex), which check which conversation is writing the story and ask "
                "a person before a codex change. Reading the file is still allowed."
            )
        return resolved

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
        model_cls: type[TParams] | None = None
        try:
            model_cls = self._resolve_params_type()
            validated_params = model_cls.model_validate(actual_params_dict)
        except ValidationError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=describe_invalid_arguments(tool_identifier, e, model_cls, self.not_run_note),
                execution_time_ms=elapsed_ms,
                isolation_level=actual_context.isolation.level,
                provenance=Provenance.primary(
                    provider="local.builtin",
                    model=tool_identifier,
                ),
            )
        except Exception:
            # A validator that raised something other than a refusal: the log gets the
            # cause, the model a sentence that does not show it.
            logger.warning("%s: its arguments could not be checked", tool_identifier, exc_info=True)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=(
                    f"The arguments of the call to '{tool_identifier}' could not be checked, "
                    f"so the call was refused. {self.not_run_note}"
                ),
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
        except PlainRefusalError as e:
            # Already written for a person: passed on as it is, without a class name.
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                success=False,
                error=str(e),
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
