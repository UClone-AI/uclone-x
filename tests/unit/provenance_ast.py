"""Shared AST resolution for the two static Principle 6 guards (issue #188).

Two guards read `src/` looking for `Provenance` values:

* `test_provenance.py` — no component may *default* an absent provenance (#157).
* `test_p6_failover_compliance.py` — every producer of a non-primary provenance must be
  classified against P6 checks 4 and 5 (#148).

They were written a week apart and each resolved the name `Provenance` its own way, so
they shared a bypass: an aliased import defeated one, a module-qualified reference
defeated both. Two guards, one hole. The resolution lives here now so that fixing it
fixes both — which is the whole reason #188 is one issue rather than two.

With hosted CI removed (#177), these two guards are a large share of the enforcement that
still runs, so their blind spots are load-bearing rather than academic.
"""

from __future__ import annotations

import ast
from enum import Enum, auto

__all__ = [
    "ConstructionKind",
    "PathVerdict",
    "constructed_path",
    "construction_kind",
    "is_provenance_expression",
    "provenance_binding_names",
]


class ConstructionKind(Enum):
    """How a `Provenance` value came into being at a call site."""

    #: `Provenance(...)`, `Provenance.primary(...)`, `Provenance.model_construct(...)`.
    #: The caller is asserting attribution, so P6's producer obligations apply.
    PRODUCED = auto()
    #: `Provenance.model_validate(...)` and `<prov>.model_copy(update=...)`. The value is
    #: being re-materialised from one that already existed. See `_FORWARDING_RATIONALE`.
    FORWARDED = auto()
    #: Not a `Provenance` expression at all.
    NONE = auto()


#: **Asymmetry worth knowing before extending this module.** `constructed_path` fails
#: *closed*: anything it cannot prove is `UNKNOWN`, which callers must treat as
#: non-primary. `construction_kind` for `model_copy` fails *open*: it only recognises an
#: inline dict literal naming `path`, so indirect forms are classified `NONE` and drop out
#: silently. That is not an oversight — the receiver's type is unresolvable by name, and
#: the only closed alternative (treat every `model_copy` as provenance-related) produced
#: six false positives. The two directions are traded differently and on purpose.


class PathVerdict(Enum):
    """What can be *proved* about the `path` of a constructed `Provenance`."""

    PRIMARY = auto()
    NON_PRIMARY = auto()
    #: Not provable from the source — `**kwargs`, a name, a call, a dict built elsewhere.
    #: Callers must treat this as non-primary: the guard fails closed (#188 item 2).
    UNKNOWN = auto()


def provenance_binding_names(tree: ast.AST) -> set[str]:
    """Every local name in this module that refers to the `Provenance` class.

    Resolved from the module's real import statements, so `import Provenance as P` and
    `import uclone_x.core.provenance as m` are not escape hatches. Both guards previously
    assumed the literal string "Provenance".
    """
    direct: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "Provenance":
                    direct.add(alias.asname or alias.name)
                elif alias.name == "provenance" and (node.module or "").endswith("uclone_x.core"):
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("core.provenance"):
                    modules.add(alias.asname or alias.name.split(".")[0])
    # A module binding is used as `<mod>.Provenance(...)`, which `_root_name` reaches.
    return (direct | modules) or {"Provenance"}


def _root_name(expr: ast.expr) -> str | None:
    """The leftmost `Name` of an attribute chain: `a.b.c` -> "a"."""
    while isinstance(expr, ast.Attribute):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _names_provenance(expr: ast.expr, names: set[str]) -> bool:
    """True when `expr` denotes the `Provenance` class under any binding."""
    if isinstance(expr, ast.Name):
        return expr.id in names
    if isinstance(expr, ast.Attribute):
        # `mod.Provenance`, `uclone_x.core.provenance.Provenance`
        if expr.attr == "Provenance":
            return True
        return isinstance(expr.value, ast.Name) and expr.value.id in names
    return False


#: Constructors that assert new attribution.
_PRODUCING_METHODS = frozenset({"primary", "model_construct"})
#: Re-materialisers of a value produced elsewhere — a peer's provenance entering over the
#: A2A wire (`a2a/wire.py`).
_FORWARDING_METHODS = frozenset({"model_validate", "model_validate_json"})


def _inline_dicts(node: ast.Call, keyword: str | None = None) -> list[ast.Dict]:
    """Dict *literals* written at this call site, positionally or under `keyword`."""
    dicts = [a for a in node.args if isinstance(a, ast.Dict)]
    dicts += [
        kw.value
        for kw in node.keywords
        if isinstance(kw.value, ast.Dict) and (keyword is None or kw.arg == keyword)
    ]
    return dicts


def _rewrites_path(node: ast.Call) -> bool:
    """True for `model_copy(update={... "path": ... })` with an **inline** dict literal.

    Narrower than "rewrites `path`", deliberately: a dict built elsewhere and passed by
    name (`upd = {...}; model_copy(update=upd)`), or built by `dict(path=...)`, is not
    reached. Widening would require treating every `model_copy` as provenance-related,
    since the receiver's type cannot be resolved by name — and that swept in six unrelated
    models when it was tried. Recorded rather than papered over.
    """
    return any(
        isinstance(key, ast.Constant) and key.value == "path"
        for d in _inline_dicts(node, "update")
        for key in d.keys
    )


def _authors_an_inline_payload(node: ast.Call) -> bool:
    """True when the call's payload is a dict literal written right here.

    `Provenance.model_validate(peer_payload)` re-materialises a value produced elsewhere.
    `Provenance.model_validate({"path": "failover", ...})` does not: the fields are being
    *authored* at this call site, and the method name is the only thing that made it look
    like forwarding. Same reasoning that promotes a path-rewriting `model_copy` — an
    inline literal is authorship, whatever the method is called (#188 item 4).
    """
    return bool(_inline_dicts(node))


def construction_kind(node: ast.AST, names: set[str]) -> ConstructionKind:
    """Classify a call node as producing, forwarding, or unrelated to `Provenance`."""
    if not isinstance(node, ast.Call):
        return ConstructionKind.NONE
    func = node.func

    # Method calls are classified by the method first: `Provenance.model_validate(...)`
    # names the class but re-materialises an existing value rather than asserting a new
    # one, so the class-name test must not claim it.
    if isinstance(func, ast.Attribute):
        on_the_class = _names_provenance(func.value, names) or _root_name(func) in names
        if func.attr in _FORWARDING_METHODS and on_the_class:
            # The exemption keys on where the value came from, not on the method name.
            return (
                ConstructionKind.PRODUCED
                if _authors_an_inline_payload(node)
                else ConstructionKind.FORWARDED
            )
        if func.attr == "model_copy" and _rewrites_path(node):
            # `prov.model_copy(update={"path": FAILOVER, ...})`. The receiver is a value,
            # so its type cannot be resolved by name — but an update that rewrites `path`
            # is asserting a path this value did not have, which is *producing*
            # attribution wearing a copy's clothing. A `model_copy` that leaves `path`
            # alone is somebody else's model and none of this guard's business, which is
            # what keeps every `ontology/` and `skills/` model out of the results.
            return ConstructionKind.PRODUCED
        if on_the_class:
            return ConstructionKind.PRODUCED

    if _names_provenance(func, names):
        return ConstructionKind.PRODUCED
    return ConstructionKind.NONE


def is_provenance_expression(node: ast.AST, names: set[str]) -> bool:
    """True for any expression that yields a `Provenance`, produced or forwarded."""
    return construction_kind(node, names) is not ConstructionKind.NONE


def _path_from_value(value: ast.expr) -> PathVerdict:
    """Read a `path=` argument. Fails closed: only a proven PRIMARY returns PRIMARY."""
    # `ExecutionPath.PRIMARY` — match the member, never a substring of the source text.
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        if value.value.id == "ExecutionPath":
            return PathVerdict.PRIMARY if value.attr == "PRIMARY" else PathVerdict.NON_PRIMARY
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return PathVerdict.PRIMARY if value.value == "primary" else PathVerdict.NON_PRIMARY
    return PathVerdict.UNKNOWN


def constructed_path(node: ast.Call, names: set[str]) -> PathVerdict:
    """What the `path` of this construction can be *proved* to be.

    The predecessor asked `"PRIMARY" not in ast.unparse(value)`, which failed **open**:
    any identifier containing that substring — `NON_PRIMARY_PATH`, `_PRIMARY_FALLBACK` —
    excluded the site from the guard exactly where a guard matters most (#188 item 2).
    This resolves the enum member instead, and returns `UNKNOWN` rather than guessing.
    """
    if construction_kind(node, names) is ConstructionKind.NONE:
        return PathVerdict.UNKNOWN

    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "primary":
        return PathVerdict.PRIMARY  # the classmethod hard-codes PRIMARY

    # `Provenance(**payload)` / `Provenance(**{...})`: nothing is provable.
    if any(kw.arg is None for kw in node.keywords):
        return PathVerdict.UNKNOWN

    for kw in node.keywords:
        if kw.arg == "path":
            return _path_from_value(kw.value)

    # `model_copy(update={"path": ...})` and `model_validate({...})` carry the path inside
    # a dict literal, when it is a literal at all.
    dicts = [a for a in node.args if isinstance(a, ast.Dict)]
    dicts += [kw.value for kw in node.keywords if isinstance(kw.value, ast.Dict)]
    for d in dicts:
        for key, value in zip(d.keys, d.values, strict=False):
            if isinstance(key, ast.Constant) and key.value == "path":
                return _path_from_value(value)

    if isinstance(func, ast.Attribute) and func.attr in _FORWARDING_METHODS:
        return PathVerdict.UNKNOWN
    # A bare `Provenance(...)` with no `path=` does not validate at runtime (the field is
    # required and has no default), so this is unreachable in working code.
    return PathVerdict.UNKNOWN
