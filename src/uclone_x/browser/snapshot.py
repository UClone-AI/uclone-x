"""The page as a clone reads it: an accessibility-tree snapshot with refs (design §3.2).

Pure functions over the node list `Accessibility.getFullAXTree` returns. A snapshot lists
the page's landmarks, headings, text and controls, one line each, and gives every control a
ref (`e12`) that later actions take. Refs come from a `RefTable` the caller keeps for the
life of one page load, so the same element keeps its ref across snapshots and a diff can
say what changed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

INTERACTIVE_ROLES = frozenset(
    {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "listbox",
        "option",
        "checkbox",
        "radio",
        "switch",
        "slider",
        "spinbutton",
        "tab",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "treeitem",
    }
)
_LANDMARK_ROLES = frozenset(
    {
        "banner",
        "navigation",
        "main",
        "form",
        "search",
        "complementary",
        "contentinfo",
        "region",
        "dialog",
        "alertdialog",
        "alert",
    }
)
_CONTAINER_ROLES = frozenset({"list", "listitem", "table", "row", "tabpanel", "menu", "tree"})
# A list item's or cell's accessible name is the text inside it, which its children show.
_UNNAMED_CONTAINERS = frozenset({"listitem", "row"})
_TEXT_ROLES = frozenset({"StaticText", "heading", "image"})
_DIALOG_ROLES = frozenset({"dialog", "alertdialog"})
_STATE_NAMES = ("checked", "selected", "expanded", "pressed", "disabled", "required", "readonly")
_TRUE_VALUES = frozenset({True, "true", "mixed"})

LIST_ITEMS_SHOWN = 10
DEFAULT_MAX_CHARS = 12_000  # about 3k tokens (§3.2)
_NAME_CHARS = 120


class RefTable:
    """Refs for one page load: each element keeps its ref until the page navigates."""

    def __init__(self) -> None:
        self._by_node: dict[int, str] = {}
        self._by_ref: dict[str, int] = {}
        self._names: dict[str, str] = {}

    def ref_for(self, backend_id: int, name: str = "") -> str:
        """The element's ref, issuing the next one on first sight; remembers its name."""
        ref = self._by_node.get(backend_id)
        if ref is None:
            ref = f"e{len(self._by_node) + 1}"
            self._by_node[backend_id] = ref
            self._by_ref[ref] = backend_id
        if name:
            self._names[ref] = name
        return ref

    def name_for(self, ref: str) -> str:
        """The accessible name the element had when last listed, or "" (the Browser tab's
        overlay label and the conversation's step line, design §3.3-§3.4)."""
        return self._names.get(ref, "")

    def node_for(self, ref: str) -> int | None:
        """The DOM node a ref names, or `None` if this page load never issued it."""
        return self._by_ref.get(ref)


@dataclass(frozen=True)
class Entry:
    """One line of a snapshot."""

    role: str
    name: str
    depth: int
    ref: str | None = None
    value: str = ""
    states: tuple[str, ...] = ()
    level: int | None = None
    more: int = 0  # set on the line that stands for collapsed siblings

    def line(self) -> str:
        """The entry as the clone reads it."""
        pad = "  " * self.depth
        if self.more:
            return f"{pad}- …and {self.more} more"
        role = "text" if self.role == "StaticText" else self.role
        parts = [f"{pad}- {role}"]
        if self.name:
            parts.append(f'"{self.name}"')
        attrs: list[str] = []
        if self.ref:
            attrs.append(f"ref={self.ref}")
        if self.level is not None:
            attrs.append(f"level={self.level}")
        attrs.extend(self.states)
        if attrs:
            parts.append(f"[{', '.join(attrs)}]")
        if self.value:
            parts.append(f'value="{self.value}"')
        return " ".join(parts)


@dataclass
class _Tree:
    nodes: dict[str, dict[str, Any]] = field(default_factory=lambda: {})
    root: str | None = None


def _value(node: Mapping[str, Any], key: str) -> str:
    raw = node.get(key)
    if isinstance(raw, dict):
        inner = cast(dict[str, Any], raw).get("value")
        if inner is not None:
            return " ".join(str(inner).split())
    return ""


def _clip(text: str) -> str:
    return text if len(text) <= _NAME_CHARS else text[: _NAME_CHARS - 1] + "…"


def _properties(node: Mapping[str, Any]) -> dict[str, object]:
    props: dict[str, object] = {}
    raw = node.get("properties")
    if not isinstance(raw, list):
        return props
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            continue
        prop = cast(dict[str, Any], item)
        value = prop.get("value")
        props[str(prop.get("name"))] = (
            cast(dict[str, Any], value).get("value") if isinstance(value, dict) else None
        )
    return props


def _index(nodes: Sequence[Mapping[str, Any]]) -> _Tree:
    tree = _Tree()
    for node in nodes:
        node_id = str(node.get("nodeId"))
        tree.nodes[node_id] = dict(node)
        if tree.root is None and "parentId" not in node:
            tree.root = node_id
    return tree


def _children(tree: _Tree, node: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids = node.get("childIds")
    if not isinstance(ids, list):
        return []
    return [tree.nodes[str(c)] for c in cast(list[object], ids) if str(c) in tree.nodes]


def build_entries(
    nodes: Sequence[Mapping[str, Any]], refs: RefTable, *, collapse: bool = True
) -> list[Entry]:
    """The snapshot's lines, in reading order, with an open dialog listed first."""
    tree = _index(nodes)
    if tree.root is None:
        return []
    root = tree.nodes[tree.root]
    dialogs: list[dict[str, Any]] = []
    _collect_dialogs(tree, root, dialogs)
    entries: list[Entry] = []
    for dialog in dialogs:
        _walk(tree, dialog, 0, "", refs, entries, collapse, skip=())
    skip = tuple(str(d.get("nodeId")) for d in dialogs)
    for child in _children(tree, root):
        _walk(tree, child, 0, _value(root, "name"), refs, entries, collapse, skip=skip)
    return entries


def _collect_dialogs(tree: _Tree, node: Mapping[str, Any], out: list[dict[str, Any]]) -> None:
    for child in _children(tree, node):
        if not child.get("ignored") and _value(child, "role") in _DIALOG_ROLES:
            out.append(child)
        else:
            _collect_dialogs(tree, child, out)


def _walk(
    tree: _Tree,
    node: dict[str, Any],
    depth: int,
    parent_name: str,
    refs: RefTable,
    out: list[Entry],
    collapse: bool,
    *,
    skip: tuple[str, ...],
) -> None:
    if str(node.get("nodeId")) in skip:
        return
    role = _value(node, "role")
    if role in {"InlineTextBox", "LineBreak"}:
        return
    if node.get("ignored"):
        for child in _children(tree, node):
            _walk(tree, child, depth, parent_name, refs, out, collapse, skip=skip)
        return
    name = _clip(_value(node, "name"))
    shown = _entry_for(node, role, name, parent_name, depth, refs)
    if shown is not None:
        out.append(shown)
    mark = len(out)
    child_depth = depth + 1 if shown is not None else depth
    children = _children(tree, node)
    if role in INTERACTIVE_ROLES:
        # A control's text is its name; only controls inside it (a select's options) show.
        children = [c for c in children if _value(c, "role") in INTERACTIVE_ROLES]
    shown_items = 0
    hidden_items = 0
    for child in children:
        is_item = _value(child, "role") == "listitem"
        if collapse and role == "list" and is_item:
            if shown_items >= LIST_ITEMS_SHOWN:
                hidden_items += 1
                continue
            shown_items += 1
        # Text repeating the line above it (a heading's own text node) is not shown twice.
        above = shown.name if shown is not None else parent_name
        _walk(tree, child, child_depth, above, refs, out, collapse, skip=skip)
    if hidden_items:
        out.append(Entry(role="more", name="", depth=child_depth, more=hidden_items))
    if shown is not None and len(out) == mark and role in _CONTAINER_ROLES:
        # A list, row or table with nothing readable in it is layout, not content.
        out.pop()


def _entry_for(
    node: Mapping[str, Any],
    role: str,
    name: str,
    parent_name: str,
    depth: int,
    refs: RefTable,
) -> Entry | None:
    props = _properties(node)
    states = tuple(s for s in _STATE_NAMES if props.get(s) in _TRUE_VALUES)
    if role in INTERACTIVE_ROLES:
        backend = node.get("backendDOMNodeId")
        ref = refs.ref_for(backend, name) if isinstance(backend, int) else None
        return Entry(
            role=role,
            name=name,
            depth=depth,
            ref=ref,
            value=_clip(_value(node, "value")),
            states=states,
        )
    if role in _LANDMARK_ROLES and (name or role != "region"):
        return Entry(role=role, name=name, depth=depth, states=states)
    if role in _UNNAMED_CONTAINERS:
        return Entry(role=role, name="", depth=depth)
    if role in _CONTAINER_ROLES:
        return Entry(role=role, name=name, depth=depth)
    if role in _TEXT_ROLES and name and name != parent_name:
        level = props.get("level") if role == "heading" else None
        return Entry(
            role=role,
            name=name,
            depth=depth,
            level=level if isinstance(level, int) else None,
        )
    return None


def render(entries: Iterable[Entry], *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """The snapshot as text, cut at `max_chars` with a note that says how to go on."""
    lines: list[str] = []
    used = 0
    for entry in entries:
        line = entry.line()
        if used + len(line) + 1 > max_chars:
            lines.append(
                "- …the snapshot stops here to stay short; use find to reach elements "
                "further down the page"
            )
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else "(The page shows nothing a clone can read.)"


def diff(before: Sequence[Entry], after: Sequence[Entry]) -> str:
    """What changed on a page that did not navigate: added, removed and changed lines."""
    old_by_ref = {e.ref: e for e in before if e.ref}
    new_by_ref = {e.ref: e for e in after if e.ref}
    out: list[str] = []
    for ref, entry in new_by_ref.items():
        previous = old_by_ref.get(ref)
        if previous is None:
            out.append("+ " + _bare(entry))
        elif _bare(previous) != _bare(entry):
            out.append("~ " + _bare(entry))
    out.extend("- " + _bare(e) for ref, e in old_by_ref.items() if ref not in new_by_ref)
    old_text = Counter(_bare(e) for e in before if not e.ref and not e.more)
    new_text = Counter(_bare(e) for e in after if not e.ref and not e.more)
    out.extend("+ " + line for line in (new_text - old_text).elements())
    out.extend("- " + line for line in (old_text - new_text).elements())
    return "\n".join(out) if out else "Nothing on the page changed."


def _bare(entry: Entry) -> str:
    """The entry's line without its indent and bullet, for a diff to mark."""
    return entry.line().lstrip().removeprefix("- ")


def find(entries: Sequence[Entry], query: str, *, limit: int = 20) -> list[Entry]:
    """Entries whose role, name or value contains `query`, controls first."""
    needle = query.casefold().strip()
    if not needle:
        return []
    hits = [
        e
        for e in entries
        if not e.more
        and (needle in e.name.casefold() or needle in e.value.casefold() or needle == e.role)
    ]
    hits.sort(key=lambda e: e.ref is None)
    return [Entry(e.role, e.name, 0, e.ref, e.value, e.states, e.level) for e in hits[:limit]]
