"""The tools layer of a request, as a selectable module (#2188, design §5.1).

A request's tools layer (layer 1) is built one of several ways. Each is a module, and a
clone runs one of them for the life of its seat:

- ``native`` -- today's behaviour, and the default on every provider. Where the host has a
  binder (Ollama), the base set and `search_tools` are pinned and host binding appends
  catalog tools once per user message; everywhere else every held tool is pinned.
- ``pinned`` -- every held tool is declared on every request, binder or not (the
  `tool_strategy` eval's arm A).
- ``bound`` -- the base set is pinned and host binding appends catalog tools once per user
  message, with no `search_tools` (the eval's arm F). With no binder, or once binding
  fails, every held tool is pinned, as in ``native``.

- ``k_act`` -- K + act: no tool is declared. The base set is described in the system text,
  a tool binding adds is described once on the user message that bound it, and the host
  reads the reply's ``<tool_call>`` text and runs it through the same dispatch
  (`agent/k_act.py`). It binds as ``bound`` does, with no `search_tools`. Opt-in only:
  no provider defaults to it.

**One place selects.** `select_tools_module` reads the clone's own setting, then the
provider's default. `BaseAgent` calls it once, when it builds its tool invoker, so a 1:1
chat and a room seat of one clone, both built by `build_clone`, run the same module.

**Recorded, and changed only at an epoch boundary.** Each request's `ContextSnapshot`
records the module it was built under (absent for ``native``, so a default record is
written as before). A request whose module differs from the one the session last sent
under declares `EPOCH_TOOLS_MODULE_CHANGED`, which opens a new epoch even when the
conversation only grew. A module is fixed when the seat is built, so that request is
always the first of a turn.

**No migration.** A recorded module this build does not know is refused in plain words
(`UnknownToolsModuleError`), never read as some other module.

The layer functions here (`pinned_layer`, `grow_bound`, `bound_layer`) are the ones the
`tool_strategy` eval's arms A and F run, and arm K runs `agent/k_act.py`, so the eval
measures this code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

from uclone_x.core.context_state import EPOCH_TOOLS_MODULE_CHANGED
from uclone_x.errors import PlainRefusalError
from uclone_x.llm.models import ToolDefinition

__all__ = [
    "DEFAULT_TOOLS_MODULE",
    "EPOCH_TOOLS_MODULE_CHANGED",
    "K_ACT_PROVIDERS",
    "PROVIDER_TOOLS_MODULES",
    "TOOLS_MODULES",
    "TOOLS_MODULE_PROVIDER_MESSAGE",
    "UNKNOWN_RECORDED_TOOLS_MODULE_MESSAGE",
    "UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE",
    "ToolsModuleName",
    "ToolsModuleUnsupportedError",
    "UnknownToolsModuleError",
    "bound_layer",
    "grow_bound",
    "pinned_layer",
    "recorded_tools_module",
    "select_tools_module",
    "snapshot_tools_module",
]

ToolsModuleName = Literal["native", "pinned", "bound", "k_act"]

#: Every module this build knows, in the order a person is told them.
TOOLS_MODULES: Final[tuple[ToolsModuleName, ...]] = ("native", "pinned", "bound", "k_act")

#: The module a clone runs when nothing chooses another.
DEFAULT_TOOLS_MODULE: Final[ToolsModuleName] = "native"

#: Per provider, the module a clone with no setting of its own runs. A provider not listed
#: runs `DEFAULT_TOOLS_MODULE`. Empty: every provider stays on ``native`` until the #2170
#: checks pass, and the PR that flips Ollama adds its entry here.
PROVIDER_TOOLS_MODULES: Final[Mapping[str, ToolsModuleName]] = MappingProxyType({})

#: Refusal of a clone setting that names no module this build has. Plain words: it names
#: the choices a person can write, and no class, file or field.
UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE: Final = (
    "This clone is set to offer its tools in a way this version does not have. "
    "Choose native, pinned, bound or k_act, or leave the setting out."
)

#: Refusal of a recorded module this build does not know: a conversation saved by a newer
#: version, or by one whose module has since been removed. It is not read as another one.
UNKNOWN_RECORDED_TOOLS_MODULE_MESSAGE: Final = (
    "This conversation was saved by a version that offered tools in a way this version "
    "does not have, so it cannot continue here."
)


class UnknownToolsModuleError(PlainRefusalError):
    """A tools module name this build does not know, refused in plain words.

    `name` carries the unknown name for the log; the message never quotes it, since a
    record's text is not something to show a person verbatim.
    """

    def __init__(self, message: str, *, name: str) -> None:
        super().__init__(message, reason_code="unknown_tools_module")
        self.name = name


#: The providers ``k_act`` may run on: Ollama, whose chat API accepts a tool result after an
#: assistant message that declared no structured call, and the scripted ``mock`` stand-in
#: the tests and offline runs use. A hosted API (OpenAI, Anthropic, Gemini) or vLLM's
#: OpenAI-compatible server refuses such a history at the first tool step, so ``k_act`` is
#: refused there when the seat is built, never in the middle of a conversation.
K_ACT_PROVIDERS: Final[frozenset[str]] = frozenset({"ollama", "mock"})

#: Refusal of ``k_act`` on a provider it cannot run on. Plain words: what the setting does,
#: where it works, and the two ways out; no setting name, module name or provider id.
TOOLS_MODULE_PROVIDER_MESSAGE: Final = (
    "This clone is set to write its tool calls as plain text, which works only with a "
    "model served by Ollama. Connect it to an Ollama model, or choose another way to offer "
    "its tools."
)


class ToolsModuleUnsupportedError(UnknownToolsModuleError):
    """A known tools module the clone's provider cannot run, refused when the seat is built.

    A kind of `UnknownToolsModuleError`, so every head that refuses an unknown setting in
    its plain sentence refuses this one the same way. `provider` is kept for the log.
    """

    def __init__(self, *, name: str, provider: str) -> None:
        super().__init__(TOOLS_MODULE_PROVIDER_MESSAGE, name=name)
        self.reason_code = "tools_module_unsupported"
        self.provider = provider


def _known(name: str) -> ToolsModuleName | None:
    return name if name in TOOLS_MODULES else None


def select_tools_module(setting: str | None, provider: str) -> ToolsModuleName:
    """The module a clone runs: its own setting, else its provider's default.

    The one place a module is chosen (#2188). An empty or absent setting chooses nothing.

    Raises:
        UnknownToolsModuleError: The setting names a module this build does not have.
        ToolsModuleUnsupportedError: The setting names ``k_act`` and the provider is not
            one it runs on (`K_ACT_PROVIDERS`).
    """
    chosen = (setting or "").strip()
    if chosen:
        known = _known(chosen)
        if known is None:
            raise UnknownToolsModuleError(UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE, name=chosen)
        if known == "k_act" and provider.strip().lower() not in K_ACT_PROVIDERS:
            raise ToolsModuleUnsupportedError(name=known, provider=provider)
        return known
    return PROVIDER_TOOLS_MODULES.get(provider.strip().lower(), DEFAULT_TOOLS_MODULE)


def recorded_tools_module(recorded: str | None) -> ToolsModuleName:
    """The module a `ContextSnapshot` records: absent is ``native``.

    Raises:
        UnknownToolsModuleError: The record names a module this build does not have.
    """
    if recorded is None:
        return DEFAULT_TOOLS_MODULE
    known = _known(recorded)
    if known is None:
        raise UnknownToolsModuleError(UNKNOWN_RECORDED_TOOLS_MODULE_MESSAGE, name=recorded)
    return known


def snapshot_tools_module(module: ToolsModuleName) -> str | None:
    """What a `ContextSnapshot` records for `module`: nothing for ``native``.

    Leaving the default out keeps a default record, and its snapshot ids, byte for byte
    what they were before modules existed.
    """
    return None if module == DEFAULT_TOOLS_MODULE else module


def pinned_layer(held: Iterable[ToolDefinition]) -> list[ToolDefinition]:
    """``pinned``: every held tool, in the order given (the caller's canonical order)."""
    return list(held)


def grow_bound(bound: list[str], hits: Iterable[str], held: Iterable[str] = ()) -> list[str]:
    """Append to `bound` the `hits` it lacks, sorted by name; return what was appended.

    The bound set only grows (design §5.1): a binding appends its new tools sorted among
    themselves, after every tool bound before, and never removes one. A name in `held`
    (the pinned base set) is never bound.
    """
    added = sorted(set(hits) - set(bound) - set(held))
    bound.extend(added)
    return added


def bound_layer(
    base: Sequence[ToolDefinition], catalog: Iterable[ToolDefinition], bound: Sequence[str]
) -> list[ToolDefinition]:
    """``bound`` (and ``native`` with a binder): `base`, then each bound catalog tool.

    The bound tools follow in the order they were bound; a bound name the catalog no
    longer holds is skipped.
    """
    by_name = {tool.name: tool for tool in catalog}
    return [*base, *(by_name[name] for name in bound if name in by_name)]
