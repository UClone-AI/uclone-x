"""The thirteen entailment capabilities, and the tier gate that decides which may fire.

Nine fixed rules derive facts, three detect contradictions, and a thirteenth admits *safe
Horn rules*. That is the set the ten experiences in issue #138 actually need, not OWL 2
RL's sixty, and every one of them records the premises it consumed so that
`Closure.explain` has something to walk.

**Why a thirteenth.** The twelve handle taxonomy, identity and consistency: they propagate
along subclass, subproperty, domain and range, and they know inverse, transitive,
symmetric and functional. None of them can express a **conjunctive body with a join**, and
a *derived* class needs exactly that::

    type(?s,'Skill') ^ hasVerdict(?s,?v) ^ type(?v,'AuditVerdict.APPROVE')
      ^ verdictSubject(?v,?h) ^ contentSha256(?s,?h) -> type(?s,'Approved')

The join on `?h` is the load-bearing part: it binds the audit to *this* content, so a
verdict cannot be inherited by different code. Without it, an incident like #22 -- where
an artifact asserted its own admission status -- stays representable.

**The Horn grammar.** `OntologyAxiom.rule_expression` is a bare `str` with no grammar, so
this module defines one, deliberately tiny, and **rejects anything that does not parse**
rather than guessing::

    rule      := body "->" head  |  head ":-" body      ("=>" and "<-" also accepted)
    body      := atom ( ("^" | "&" | "," ) atom )*
    head      := atom
    atom      := name "(" term ")"              -- a class atom, meaning type(term, name)
               | name "(" term "," term ")"
    term      := "?" ident                      -- a variable, always "?"-prefixed
               | "'" text "'" | '"' text '"'    -- a quoted constant
               | bareword                       -- an unquoted constant

Four restrictions keep the form safe, and each is checked at interpretation time with the
failure reported on `unsupported_axioms`:

* **Safety** -- every variable in the head must appear in the body. This is what bounds
  the derivable facts to constants that already exist, and so what makes termination hold.
* **No negation, disjunction or comparison.** Equality is expressed by sharing a variable,
  as `?h` does above. Tokens that would express them (`not`, `or`, `!`, `;`, `|`, `<`,
  `>`, `=`, `¬`, `\\+`) are refused outright rather than parsed as ordinary names -- a
  negation silently read as a class atom would invert the rule's meaning.
* **Exactly one head atom**, and at least one body atom.
* **A rule may not derive schema.** A head over *any* member of `SCHEMA_PREDICATES` is
  refused -- not merely `hornRule`. A rule deriving `subClassOf`, `domain`, `inverseOf`
  or a cardinality is not a smaller version of a rule deriving `hornRule`; it is the same
  escalation by another door. `declaresCategory(?c,?p) -> subClassOf(?c,?p)` reads as a
  benign taxonomy convenience, and it lets any writer of a `declaresCategory` triple mint
  a subsumption edge, which the fixed rules then propagate types along. Since the whole
  tier and authority model rests on schema being asserted by a human, a rule that
  manufactures schema from data dissolves it. The refusal names the offending head.

Since the grammar has no function symbols, no rule can invent a constant: this is Datalog,
and forward chaining over it reaches a fixed point.

**The tier gate is structural, not a filter.** Every rule in this module takes an
`AxiomIndex` and nothing else as its source of schema. `AxiomIndex.__init__` is the only
way to build one, and it admits a schema statement only when that statement is **asserted
in both senses**: its tier is `OntologyTier.ASSERTED` *and* it was not derived. There is
consequently no code path -- not a forgotten `if`, not a caller who skipped a step -- by
which an `INDUCED_ENFORCING` or `INDUCED_CANDIDATE` axiom can reach a rule: the index type
cannot represent one. This is the safety rule the tiers exist for. One wrong induction from
three observations would otherwise poison the graph with a thousand derived facts, each of
them looking exactly as justified as a true one.

**Why the tier check alone was not enough.** Tier and derivation are different questions,
and only checking the first left schema manufacturable from asserted data. A conclusion
inherits its premises' tier, so a rule fed *asserted* data yields an *asserted* derived
fact -- which passed a tier-only gate untouched. Two routes exploited that, and the second
needs no Horn rule at all:

* `declaresCategory(?c,?p) -> subClassOf(?c,?p)`, now refused by the head guard above;
* an asserted `subPropertyOf(promotedTo, subClassOf)`, which makes the fixed
  subproperty-value rule turn every `promotedTo` triple into a subsumption edge.

The head guard cannot see the second, because no Horn rule is involved. So the index
refuses `fact.derived` as well, and the two defences are independent by design: schema
that a rule computed may never license another rule, whichever rule computed it.

The cost is a real one and worth naming: `subClassOf(A,C)` derived by subclass transitivity
is now refused as a licensing edge too, and reported. Nothing is lost by it -- that edge is
redundant, since type propagation already walks `A -> B -> C` one step at a time -- but the
refusal does appear on `unsupported_axioms`, so a reader sees an entry for an edge that cost
them nothing. Reporting it anyway is the module's standing bargain: an unused refusal is
noise, an unreported one is the defect Principle 6 forbids.

**Schema statements are facts.** An interpretable asserted `OntologyAxiom` is seeded into
the closure as an ordinary `Fact` -- `subClassOf(Reviewer, Agent)` and so on -- before
chaining begins. That is what lets a proof step cite it: `ProofStep.premises` holds
`Fact.id` values, so an axiom that is not a fact cannot appear in a justification, and an
explanation that omits the axiom it used is not an explanation.

**Refusal is reported, never silent, and an unparsable axiom is inert in full.** An axiom
this module cannot interpret derives nothing *and* is recorded as an `UnsupportedAxiom` on
the closure. In particular an axiom carrying a `rule_expression` that does not parse is
voided **entirely**, including its structured fields. The risk is not symmetric: a rule
that does nothing is recoverable, whereas a rule silently stripped of a narrowing
condition fires more often than its author intended -- which, for an authority-granting
rule, is the shape of incident #22 itself.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Final, NamedTuple, TypeVar

from uclone_x.ontology.justification import (
    DISJOINT_WITH,
    DOMAIN,
    FUNCTIONAL_PROPERTY,
    HORN_RULE,
    INVERSE_FUNCTIONAL_PROPERTY,
    INVERSE_OF,
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    RANGE,
    SUBCLASS_OF,
    SUBPROPERTY_OF,
    SYMMETRIC_PROPERTY,
    TRANSITIVE_PROPERTY,
    TYPE_PREDICATE,
    Fact,
    Inconsistency,
    UnsupportedAxiom,
    canonical_predicate,
    compute_fact_id,
)
from uclone_x.ontology.models import OntologyAxiom, OntologyTier

__all__ = [
    "DERIVATION_RULES",
    "DETECTION_RULES",
    "DISJOINT_WITH",
    "DOMAIN",
    "FUNCTIONAL_PROPERTY",
    "HORN_RULE",
    "INVERSE_FUNCTIONAL_PROPERTY",
    "INVERSE_OF",
    "KIND_CARDINALITY",
    "KIND_DISJOINT",
    "KIND_FUNCTIONAL",
    "KIND_INVERSE_FUNCTIONAL",
    "MAX_CARDINALITY",
    "MIN_CARDINALITY",
    "RANGE",
    "RULE_DOMAIN_TYPE",
    "RULE_HORN",
    "RULE_INVERSE_OF",
    "RULE_PROPERTY_SYMMETRY",
    "RULE_PROPERTY_TRANSITIVITY",
    "RULE_RANGE_TYPE",
    "RULE_SUBCLASS_TRANSITIVITY",
    "RULE_SUBPROPERTY_TRANSITIVITY",
    "RULE_SUBPROPERTY_VALUE_PROPAGATION",
    "RULE_TYPE_PROPAGATION",
    "SCHEMA_PREDICATES",
    "SUBCLASS_OF",
    "SUBPROPERTY_OF",
    "SYMMETRIC_PROPERTY",
    "TRANSITIVE_PROPERTY",
    "TRUE_VALUE",
    "AxiomIndex",
    "Derivation",
    "FactView",
    "HornAtom",
    "HornParse",
    "HornRule",
    "HornTerm",
    "IndexedHornRule",
    "canonical_horn_text",
    "interpret_axioms",
    "normalize_axiom_kind",
    "parse_horn_rule",
]

_K = TypeVar("_K")

# --------------------------------------------------------------------------------------
# Schema vocabulary
#
# The canonical predicate names live in `justification.py`, because a predicate's spelling
# is part of a fact's identity; they are re-exported here so that rule code and its callers
# read one vocabulary.
# --------------------------------------------------------------------------------------

TRUE_VALUE: Final = "true"
"""Object of a unary characteristic fact, e.g. `transitiveProperty(partOf, true)`."""

_BINARY_KINDS: Final = frozenset(
    {SUBCLASS_OF, SUBPROPERTY_OF, DOMAIN, RANGE, INVERSE_OF, DISJOINT_WITH}
)
_UNARY_KINDS: Final = frozenset(
    {
        TRANSITIVE_PROPERTY,
        SYMMETRIC_PROPERTY,
        FUNCTIONAL_PROPERTY,
        INVERSE_FUNCTIONAL_PROPERTY,
    }
)
_CARDINALITY_KINDS: Final = frozenset({MAX_CARDINALITY, MIN_CARDINALITY})

SCHEMA_PREDICATES: Final = frozenset(
    _BINARY_KINDS | _UNARY_KINDS | _CARDINALITY_KINDS | {HORN_RULE}
)
"""Predicates the reasoner reads as schema rather than as data.

`hornRule` is one of them, which is what puts Horn rules behind the same structural tier
gate as every other schema statement: `AxiomIndex` is the only thing that can turn one
into something a rule reads, and it admits only asserted facts.

This frozenset is also exactly the set of predicates refused as a Horn *head*, so adding a
member here extends that refusal automatically. That coupling is deliberate: a schema
predicate a rule could derive but the head guard did not know about would be a hole opened
by omission rather than by decision.
"""

_KIND_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "subclassof": SUBCLASS_OF,
        "subpropertyof": SUBPROPERTY_OF,
        "domain": DOMAIN,
        "range": RANGE,
        "inverseof": INVERSE_OF,
        "transitive": TRANSITIVE_PROPERTY,
        "transitiveproperty": TRANSITIVE_PROPERTY,
        "symmetric": SYMMETRIC_PROPERTY,
        "symmetricproperty": SYMMETRIC_PROPERTY,
        "functional": FUNCTIONAL_PROPERTY,
        "functionalproperty": FUNCTIONAL_PROPERTY,
        "inversefunctional": INVERSE_FUNCTIONAL_PROPERTY,
        "inversefunctionalproperty": INVERSE_FUNCTIONAL_PROPERTY,
        "disjointwith": DISJOINT_WITH,
        "maxcardinality": MAX_CARDINALITY,
        "mincardinality": MIN_CARDINALITY,
    }
)

_PREFIX_PATTERN: Final = re.compile(r"^(?:rdf|rdfs|owl)\s*:\s*", re.IGNORECASE)
_CARDINALITY_PATTERN: Final = re.compile(r"^([^\s:]+)\s*:\s*(\d+)$")

# Rule names, exported so that callers assert against a constant rather than a literal.
RULE_SUBCLASS_TRANSITIVITY: Final = "subclass_transitivity"
RULE_TYPE_PROPAGATION: Final = "type_propagation"
RULE_SUBPROPERTY_TRANSITIVITY: Final = "subproperty_transitivity"
RULE_SUBPROPERTY_VALUE_PROPAGATION: Final = "subproperty_value_propagation"
RULE_DOMAIN_TYPE: Final = "domain_type"
RULE_RANGE_TYPE: Final = "range_type"
RULE_INVERSE_OF: Final = "inverse_of"
RULE_PROPERTY_TRANSITIVITY: Final = "property_transitivity"
RULE_PROPERTY_SYMMETRY: Final = "property_symmetry"
RULE_HORN: Final = "horn_rule"
"""Prefix of a Horn firing's rule name; the recorded step reads `horn_rule:<rule name>`."""

# Inconsistency kinds.
KIND_DISJOINT: Final = "disjoint"
KIND_CARDINALITY: Final = "cardinality"
KIND_FUNCTIONAL: Final = "functional"
KIND_INVERSE_FUNCTIONAL: Final = "inverse_functional"


def normalize_axiom_kind(raw: str) -> str | None:
    """Map an axiom `predicate` onto a canonical schema predicate, or `None` if unknown.

    Accepts the namespace prefixes and the snake/kebab spellings that arrive from hand
    authoring (`rdfs:subClassOf`, `sub_class_of`, `SUBCLASSOF`). Normalisation is applied
    to axiom *fields* only; a `Fact` must already use the canonical spelling, because a
    fact's identity is its exact triple and two spellings would be two facts.
    """
    stripped = _PREFIX_PATTERN.sub("", raw.strip())
    collapsed = stripped.replace("_", "").replace("-", "").replace(" ", "").lower()
    return _KIND_ALIASES.get(collapsed)


def _parse_cardinality(object_value: str) -> tuple[str, int] | None:
    """Parse a `"property:N"` cardinality argument, or return `None` if malformed."""
    match = _CARDINALITY_PATTERN.match(object_value.strip())
    if match is None:
        return None
    return (match.group(1), int(match.group(2)))


# --------------------------------------------------------------------------------------
# The thirteenth capability: safe Horn rules
# --------------------------------------------------------------------------------------


class HornTerm(NamedTuple):
    """A rule argument: either a variable to bind, or a constant to match literally."""

    value: str
    is_variable: bool


class HornAtom(NamedTuple):
    """One `predicate(subject, object)` pattern in a rule body or head."""

    predicate: str
    subject: HornTerm
    object: HornTerm


class HornRule(NamedTuple):
    """A safe Horn rule: a conjunctive body entailing exactly one head atom."""

    body: tuple[HornAtom, ...]
    head: HornAtom

    def variables(self) -> frozenset[str]:
        """Return every variable name appearing anywhere in the rule."""
        names: set[str] = set()
        for atom in (*self.body, self.head):
            for term in (atom.subject, atom.object):
                if term.is_variable:
                    names.add(term.value)
        return frozenset(names)


class IndexedHornRule(NamedTuple):
    """A rule admitted by `AxiomIndex`, with the asserted fact that licensed it."""

    rule: HornRule
    fact_id: str
    name: str


class HornParse(NamedTuple):
    """The outcome of parsing a rule: a rule, or the reason there is none.

    Parsing reports rather than raises because every caller needs the reason as text, to
    put on `unsupported_axioms` where an operator will read it.
    """

    rule: HornRule | None
    reason: str


_ARROWS_BODY_FIRST: Final = ("->", "=>")
_ARROWS_HEAD_FIRST: Final = (":-", "<-")
_ARROWS: Final = (*_ARROWS_HEAD_FIRST, *_ARROWS_BODY_FIRST)
_CONJUNCTION_SEPARATORS: Final = frozenset({"^", "&", ",", "∧"})
_ATOM_PATTERN: Final = re.compile(r"^([A-Za-z_][A-Za-z0-9_.:\-]*)\s*\((.*)\)$", re.DOTALL)
_VARIABLE_PATTERN: Final = re.compile(r"^\?([A-Za-z0-9_]+)$")
_FORBIDDEN_PATTERN: Final = re.compile(
    r"\\\+|¬|[;|!<>=]|(?<![A-Za-z0-9_])(?:not|or|neq|isnot)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _split_top_level(text: str, separators: frozenset[str]) -> list[str] | None:
    """Split on separators outside quotes and parentheses, or return `None` if unbalanced."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            current.append(char)
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and char in separators:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if depth != 0 or quote is not None:
        return None
    parts.append("".join(current))
    return parts


def _find_arrow(text: str) -> tuple[int, str] | None:
    """Locate the first implication arrow outside a quoted constant."""
    quote: str | None = None
    for position, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            continue
        for arrow in _ARROWS:
            if text.startswith(arrow, position):
                return (position, arrow)
    return None


def _parse_term(text: str) -> HornTerm | None:
    """Parse one argument into a variable or a constant, or return `None` if malformed."""
    stripped = text.strip()
    if not stripped:
        return None
    variable = _VARIABLE_PATTERN.match(stripped)
    if variable is not None:
        return HornTerm(value=variable.group(1), is_variable=True)
    if stripped.startswith("?"):
        return None
    if len(stripped) >= 2 and stripped[0] in ("'", '"') and stripped[-1] == stripped[0]:
        inner = stripped[1:-1]
        return None if stripped[0] in inner else HornTerm(value=inner, is_variable=False)
    if "'" in stripped or '"' in stripped or any(c.isspace() for c in stripped):
        return None
    return HornTerm(value=stripped, is_variable=False)


def _parse_atom(text: str) -> HornAtom | None:
    """Parse `name(term)` or `name(term, term)`, or return `None` if malformed.

    A one-argument atom is a class atom: `Skill(?s)` means `type(?s, 'Skill')`, which is
    what lets a rule body require membership rather than merely mention a subject.
    """
    match = _ATOM_PATTERN.match(text.strip())
    if match is None:
        return None
    name = canonical_predicate(match.group(1))
    arguments = _split_top_level(match.group(2), frozenset({","}))
    if arguments is None:  # pragma: no cover - the enclosing split rejects this first
        # An atom can only be unbalanced if the body or head containing it was, and
        # `parse_horn_rule` splits those with the same tracker before reaching here.
        return None
    terms = [_parse_term(argument) for argument in arguments]
    if any(term is None for term in terms):
        return None
    parsed = [term for term in terms if term is not None]
    if len(parsed) == 1:
        return HornAtom(
            predicate=TYPE_PREDICATE,
            subject=parsed[0],
            object=HornTerm(value=name, is_variable=False),
        )
    if len(parsed) == 2:
        return HornAtom(predicate=name, subject=parsed[0], object=parsed[1])
    return None


def parse_horn_rule(text: str) -> HornParse:
    """Parse a rule expression under the restricted Horn grammar documented in this module.

    Returns the rule, or the reason it was refused. Refusal is the default for anything the
    grammar does not cover: a best-effort reading of a rule the author wrote differently is
    how a constraint quietly becomes broader than intended.
    """
    stripped = text.strip()
    if not stripped:
        return HornParse(None, "the expression is empty")
    found = _find_arrow(stripped)
    if found is None:
        return HornParse(None, "no implication arrow ('->' or ':-') was found")
    position, arrow = found
    left = stripped[:position]
    right = stripped[position + len(arrow) :]
    body_text, head_text = (left, right) if arrow in _ARROWS_BODY_FIRST else (right, left)

    # A second arrow is checked before the forbidden-token scan: arrows themselves contain
    # characters that scan forbids ('>', '='), so the clearer diagnosis has to come first.
    if _find_arrow(body_text) is not None or _find_arrow(head_text) is not None:
        return HornParse(None, "more than one implication arrow was found")

    for half, label in ((body_text, "body"), (head_text, "head")):
        forbidden = _FORBIDDEN_PATTERN.search(half)
        if forbidden is not None:
            return HornParse(
                None,
                f"the {label} uses '{forbidden.group(0)}'; negation, disjunction and "
                "comparison are not part of this grammar",
            )

    atom_texts = _split_top_level(body_text, _CONJUNCTION_SEPARATORS)
    if atom_texts is None:
        return HornParse(None, "the body has unbalanced parentheses or quotes")
    body: list[HornAtom] = []
    for atom_text in atom_texts:
        if not atom_text.strip():
            return HornParse(None, "the body has an empty conjunct")
        atom = _parse_atom(atom_text)
        if atom is None:
            return HornParse(None, f"body atom '{atom_text.strip()}' does not parse")
        body.append(atom)

    head_atoms = _split_top_level(head_text, _CONJUNCTION_SEPARATORS)
    if head_atoms is None or len(head_atoms) != 1:
        return HornParse(None, "the head must be exactly one atom")
    head = _parse_atom(head_atoms[0])
    if head is None:
        return HornParse(None, f"head atom '{head_atoms[0].strip()}' does not parse")
    if head.predicate in SCHEMA_PREDICATES:
        return HornParse(
            None,
            f"a rule may not derive schema: the head predicate '{head.predicate}' is a schema "
            "predicate, so the rule would rewrite the schema that licenses it",
        )

    rule = HornRule(body=tuple(body), head=head)
    body_variables = {
        term.value for atom in rule.body for term in (atom.subject, atom.object) if term.is_variable
    }
    unsafe = sorted(
        term.value
        for term in (head.subject, head.object)
        if term.is_variable and term.value not in body_variables
    )
    if unsafe:
        return HornParse(
            None,
            f"the rule is unsafe: head variable(s) {', '.join('?' + n for n in unsafe)} "
            "never appear in the body, so the conclusion is not bound to anything",
        )
    return HornParse(rule, "")


def _term_text(term: HornTerm) -> str:
    """Render a term in canonical form: variables `?x`, constants always quoted."""
    return f"?{term.value}" if term.is_variable else f"'{term.value}'"


def _atom_text(atom: HornAtom) -> str:
    """Render an atom in canonical two-argument form."""
    return f"{atom.predicate}({_term_text(atom.subject)},{_term_text(atom.object)})"


def canonical_horn_text(rule: HornRule) -> str:
    """Render a rule in one canonical spelling, so equivalent rules are one fact.

    A rule is stored as a `Fact`, and a fact's identity is its triple; two spellings of the
    same rule must therefore collapse to the same text or the closure would hold both.
    """
    body = " ^ ".join(_atom_text(atom) for atom in rule.body)
    return f"{body} -> {_atom_text(rule.head)}"


def interpret_axioms(
    axioms: Iterable[OntologyAxiom],
) -> tuple[tuple[Fact, ...], tuple[UnsupportedAxiom, ...]]:
    """Translate axioms into asserted schema facts, reporting every one that is refused.

    An axiom takes exactly one of two forms, decided by whether `rule_expression` is set:

    * **empty `rule_expression`** -- a structured axiom. `predicate` names the axiom kind,
      `subject_entity` is its first argument and `object_value` its second.
    * **non-empty `rule_expression`** -- a Horn rule, and the expression is then the
      axiom's *entire* semantics. The structured fields are descriptive only, and **any
      non-empty `predicate` beside a rule is reported as not applied** -- whether or not it
      names a schema kind this module recognises.

      Reporting only the *recognised* ones, as this function once did, had the reporting
      exactly backwards. An author who writes `predicate="equivalentClass"` beside a rule
      believes `Approved = Skill n ...` is in force; `equivalentClass` is not a supported
      kind, so under the old test it produced no refusal at all and the belief was never
      contradicted. The unrecognised predicate is the one more likely to be load-bearing in
      its author's mind, precisely because nothing else in the system mentions it.

    Keeping the two forms exclusive means every axiom has one interpretation, never a
    blend of two that could double-count or disagree.

    An axiom is refused, and left wholly inert, when:

    * its tier is not `ASSERTED` -- the safety rule of issue #138;
    * its `rule_expression` does not parse under the restricted Horn grammar. The whole
      axiom is voided, structured fields included. A `rule_expression` that fails to parse
      may have been the *narrowing* half of the author's intent, and an axiom silently
      stripped of its narrowing half fires more often than intended -- for an
      authority-granting rule, that is incident #22 exactly. A rule that does nothing is
      recoverable; a rule broader than its author believes is not;
    * its `predicate` is not a known schema kind -- notably the `predicate == object_value`
      constraint axioms that `OntologyEngine.validate_entity` reads, which are record
      validation and not entailment;
    * its arguments are missing or malformed for its kind.

    The remaining `OntologyAxiom` fields -- `description`, `confidence`, `evidence`,
    `content_hash`, `precedence` -- are metadata carrying no rule content, and are not
    interpreted as semantics by any rule here.
    """
    facts: list[Fact] = []
    refused: list[UnsupportedAxiom] = []
    for axiom in axioms:
        if axiom.tier is not OntologyTier.ASSERTED:
            refused.append(
                UnsupportedAxiom(
                    source=axiom.name,
                    tier=axiom.tier,
                    reason=(
                        f"tier '{axiom.tier.value}' does not enter the closure; only asserted "
                        "axioms may derive facts"
                    ),
                )
            )
            continue
        if axiom.rule_expression.strip():
            parsed = parse_horn_rule(axiom.rule_expression)
            if parsed.rule is None:
                refused.append(
                    UnsupportedAxiom(
                        source=axiom.name,
                        tier=axiom.tier,
                        reason=(
                            f"rule_expression does not parse as a Horn rule ({parsed.reason}); "
                            "the axiom is wholly inert, structured fields included"
                        ),
                    )
                )
                continue
            declared = axiom.predicate.strip()
            if declared:
                unrecognised = (
                    ""
                    if normalize_axiom_kind(declared) is not None
                    else f"; '{declared}' is not a supported axiom kind in any case"
                )
                refused.append(
                    UnsupportedAxiom(
                        source=axiom.name,
                        tier=axiom.tier,
                        reason=(
                            f"structured predicate '{axiom.predicate}' was not applied: a "
                            "parsable rule_expression is the axiom's entire semantics"
                            f"{unrecognised}"
                        ),
                    )
                )
            facts.append(
                Fact(
                    subject=axiom.name,
                    predicate=HORN_RULE,
                    object=canonical_horn_text(parsed.rule),
                    tier=OntologyTier.ASSERTED,
                    derived=False,
                )
            )
            continue
        kind = normalize_axiom_kind(axiom.predicate)
        if kind is None:
            refused.append(
                UnsupportedAxiom(
                    source=axiom.name,
                    tier=axiom.tier,
                    # The kinds a predicate may name: `hornRule` is a schema predicate but
                    # not one of them, since a Horn rule is only a `rule_expression` (#1601).
                    reason=(
                        f"predicate '{axiom.predicate}' is not a supported axiom kind "
                        f"(supported: {', '.join(sorted(set(_KIND_ALIASES.values())))})"
                    ),
                )
            )
            continue
        subject = axiom.subject_entity.strip()
        if not subject:
            refused.append(
                UnsupportedAxiom(
                    source=axiom.name,
                    tier=axiom.tier,
                    reason=f"axiom kind '{kind}' needs a subject_entity, which is empty",
                )
            )
            continue
        object_value = axiom.object_value.strip()
        if kind in _BINARY_KINDS:
            if not object_value:
                refused.append(
                    UnsupportedAxiom(
                        source=axiom.name,
                        tier=axiom.tier,
                        reason=f"axiom kind '{kind}' needs an object_value, which is empty",
                    )
                )
                continue
        elif kind in _CARDINALITY_KINDS:
            if _parse_cardinality(object_value) is None:
                refused.append(
                    UnsupportedAxiom(
                        source=axiom.name,
                        tier=axiom.tier,
                        reason=(
                            f"axiom kind '{kind}' needs object_value of the form "
                            f"'property:N', got '{axiom.object_value}'"
                        ),
                    )
                )
                continue
        else:
            if object_value and object_value.lower() != TRUE_VALUE:
                refused.append(
                    UnsupportedAxiom(
                        source=axiom.name,
                        tier=axiom.tier,
                        reason=(
                            f"axiom kind '{kind}' takes no argument, but object_value is "
                            f"'{axiom.object_value}'"
                        ),
                    )
                )
                continue
            object_value = TRUE_VALUE
        facts.append(
            Fact(
                subject=subject,
                predicate=kind,
                object=object_value,
                tier=OntologyTier.ASSERTED,
                derived=False,
            )
        )
    return (tuple(facts), tuple(refused))


# --------------------------------------------------------------------------------------
# Indexes
# --------------------------------------------------------------------------------------


class FactView:
    """Read-only lookup indexes over a fact store, rebuilt once per chaining round."""

    __slots__ = (
        "by_pred_object",
        "by_pred_subject",
        "by_predicate",
        "facts",
        "types_by_class",
        "types_by_subject",
    )

    def __init__(self, facts: Mapping[str, Fact]) -> None:
        """Index `facts` by predicate, by predicate/subject, by predicate/object and by type."""
        by_predicate: dict[str, list[Fact]] = {}
        by_pred_subject: dict[tuple[str, str], list[Fact]] = {}
        by_pred_object: dict[tuple[str, str], list[Fact]] = {}
        types_by_subject: dict[str, list[Fact]] = {}
        types_by_class: dict[str, list[Fact]] = {}
        for fact in facts.values():
            by_predicate.setdefault(fact.predicate, []).append(fact)
            by_pred_subject.setdefault((fact.predicate, fact.subject), []).append(fact)
            by_pred_object.setdefault((fact.predicate, fact.object), []).append(fact)
            if fact.predicate == TYPE_PREDICATE:
                types_by_subject.setdefault(fact.subject, []).append(fact)
                types_by_class.setdefault(fact.object, []).append(fact)
        self.facts: Mapping[str, Fact] = facts
        self.by_predicate: Mapping[str, tuple[Fact, ...]] = _freeze_lists(by_predicate)
        self.by_pred_subject: Mapping[tuple[str, str], tuple[Fact, ...]] = _freeze_lists(
            by_pred_subject
        )
        self.by_pred_object: Mapping[tuple[str, str], tuple[Fact, ...]] = _freeze_lists(
            by_pred_object
        )
        self.types_by_subject: Mapping[str, tuple[Fact, ...]] = _freeze_lists(types_by_subject)
        self.types_by_class: Mapping[str, tuple[Fact, ...]] = _freeze_lists(types_by_class)


def _freeze_lists(source: Mapping[_K, list[Fact]]) -> Mapping[_K, tuple[Fact, ...]]:
    """Return a read-only mapping of tuples from a mapping of accumulator lists."""
    return MappingProxyType({key: tuple(value) for key, value in source.items()})


class AxiomIndex:
    """Schema edges licensed exclusively by facts that are asserted and underived.

    This class is the tier gate. Its constructor is the only way to obtain an index, and it
    admits a schema fact only when `fact.tier is OntologyTier.ASSERTED` **and**
    `fact.derived` is `False`; every rule in this module reads its schema from an index and
    from nowhere else. An induced axiom therefore derives nothing, not because a caller
    remembered to filter but because the only type a rule accepts is one that cannot hold
    it.

    The two conditions catch different things and neither implies the other. The tier check
    stops a *guess* from acting as schema. The `derived` check stops schema a rule
    *computed* from acting as schema on the next round -- which the tier check cannot do,
    because a conclusion drawn from asserted premises is itself asserted. Without it, an
    asserted `subPropertyOf(promotedTo, subClassOf)` is enough to let any `promotedTo`
    triple mint a subsumption edge that the fixed rules then propagate types along.

    Facts that were refused admission are kept on `refusals` so the closure can report
    them, rather than dropping them where nobody would see the loss.
    """

    __slots__ = (
        "disjoint_with",
        "domain_of",
        "functional",
        "horn_rules",
        "inverse_functional",
        "inverse_of",
        "max_cardinality",
        "min_cardinality",
        "range_of",
        "refusals",
        "subclass_of",
        "subproperty_of",
        "symmetric",
        "transitive",
    )

    def __init__(self, facts: Iterable[Fact]) -> None:
        """Index the asserted schema facts in `facts`; refuse and record everything else."""
        subclass: dict[str, list[tuple[str, str]]] = {}
        subproperty: dict[str, list[tuple[str, str]]] = {}
        domain: dict[str, list[tuple[str, str]]] = {}
        range_of: dict[str, list[tuple[str, str]]] = {}
        inverse: dict[str, list[tuple[str, str]]] = {}
        disjoint: dict[str, list[tuple[str, str]]] = {}
        maximums: dict[str, list[tuple[str, int, str]]] = {}
        minimums: dict[str, list[tuple[str, int, str]]] = {}
        transitive: dict[str, str] = {}
        symmetric: dict[str, str] = {}
        functional: dict[str, str] = {}
        inverse_functional: dict[str, str] = {}
        horn_rules: list[IndexedHornRule] = []
        refusals: list[UnsupportedAxiom] = []

        binary_targets: Mapping[str, dict[str, list[tuple[str, str]]]] = {
            SUBCLASS_OF: subclass,
            SUBPROPERTY_OF: subproperty,
            DOMAIN: domain,
            RANGE: range_of,
            INVERSE_OF: inverse,
        }
        unary_targets: Mapping[str, dict[str, str]] = {
            TRANSITIVE_PROPERTY: transitive,
            SYMMETRIC_PROPERTY: symmetric,
            FUNCTIONAL_PROPERTY: functional,
            INVERSE_FUNCTIONAL_PROPERTY: inverse_functional,
        }

        for fact in facts:
            if fact.predicate not in SCHEMA_PREDICATES:
                continue
            if fact.tier is not OntologyTier.ASSERTED:
                refusals.append(
                    UnsupportedAxiom(
                        source=fact.id,
                        tier=fact.tier,
                        reason=(
                            f"schema statement {fact.triple} is tier '{fact.tier.value}'; only "
                            "asserted schema may license a rule"
                        ),
                    )
                )
                continue
            if fact.derived:
                refusals.append(
                    UnsupportedAxiom(
                        source=fact.id,
                        tier=fact.tier,
                        reason=(
                            f"schema statement {fact.triple} was derived, not asserted; a derived "
                            "schema edge may not license further derivation"
                        ),
                    )
                )
                continue
            if fact.predicate in binary_targets:
                binary_targets[fact.predicate].setdefault(fact.subject, []).append(
                    (fact.object, fact.id)
                )
            elif fact.predicate in unary_targets:
                unary_targets[fact.predicate][fact.subject] = fact.id
            elif fact.predicate == DISJOINT_WITH:
                # Disjointness is symmetric; both directions are indexed from one statement.
                disjoint.setdefault(fact.subject, []).append((fact.object, fact.id))
                disjoint.setdefault(fact.object, []).append((fact.subject, fact.id))
            elif fact.predicate == HORN_RULE:
                horn = parse_horn_rule(fact.object)
                if horn.rule is None:
                    refusals.append(
                        UnsupportedAxiom(
                            source=fact.id,
                            tier=fact.tier,
                            reason=(
                                f"rule '{fact.subject}' does not parse as a Horn rule "
                                f"({horn.reason})"
                            ),
                        )
                    )
                    continue
                horn_rules.append(
                    IndexedHornRule(rule=horn.rule, fact_id=fact.id, name=fact.subject)
                )
            else:
                parsed = _parse_cardinality(fact.object)
                if parsed is None:
                    refusals.append(
                        UnsupportedAxiom(
                            source=fact.id,
                            tier=fact.tier,
                            reason=(
                                f"cardinality statement {fact.triple} needs an object of the "
                                "form 'property:N'"
                            ),
                        )
                    )
                    continue
                target = maximums if fact.predicate == MAX_CARDINALITY else minimums
                target.setdefault(fact.subject, []).append((parsed[0], parsed[1], fact.id))

        self.subclass_of: Mapping[str, tuple[tuple[str, str], ...]] = _freeze_pairs(subclass)
        self.subproperty_of: Mapping[str, tuple[tuple[str, str], ...]] = _freeze_pairs(subproperty)
        self.domain_of: Mapping[str, tuple[tuple[str, str], ...]] = _freeze_pairs(domain)
        self.range_of: Mapping[str, tuple[tuple[str, str], ...]] = _freeze_pairs(range_of)
        self.inverse_of: Mapping[str, tuple[tuple[str, str], ...]] = _freeze_pairs(inverse)
        self.disjoint_with: Mapping[str, tuple[tuple[str, str], ...]] = _freeze_pairs(disjoint)
        self.max_cardinality: Mapping[str, tuple[tuple[str, int, str], ...]] = MappingProxyType(
            {key: tuple(value) for key, value in maximums.items()}
        )
        self.min_cardinality: Mapping[str, tuple[tuple[str, int, str], ...]] = MappingProxyType(
            {key: tuple(value) for key, value in minimums.items()}
        )
        self.transitive: Mapping[str, str] = MappingProxyType(dict(transitive))
        self.symmetric: Mapping[str, str] = MappingProxyType(dict(symmetric))
        self.functional: Mapping[str, str] = MappingProxyType(dict(functional))
        self.inverse_functional: Mapping[str, str] = MappingProxyType(dict(inverse_functional))
        self.horn_rules: tuple[IndexedHornRule, ...] = tuple(horn_rules)
        self.refusals: tuple[UnsupportedAxiom, ...] = tuple(refusals)


def _freeze_pairs(
    source: Mapping[str, list[tuple[str, str]]],
) -> Mapping[str, tuple[tuple[str, str], ...]]:
    """Return a read-only mapping of tuples from a mapping of accumulator lists."""
    return MappingProxyType({key: tuple(value) for key, value in source.items()})


class Derivation(NamedTuple):
    """One rule firing: the conclusion triple and the premise fact ids that produced it."""

    rule: str
    premises: tuple[str, ...]
    subject: str
    predicate: str
    object: str


# --------------------------------------------------------------------------------------
# The nine derivation rules
# --------------------------------------------------------------------------------------


def rule_subclass_transitivity(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """subClassOf(A, B) and subClassOf(B, C) entail subClassOf(A, C)."""
    del view
    for child, parents in index.subclass_of.items():
        for parent, first_id in parents:
            for grandparent, second_id in index.subclass_of.get(parent, ()):
                yield Derivation(
                    RULE_SUBCLASS_TRANSITIVITY,
                    (first_id, second_id),
                    child,
                    SUBCLASS_OF,
                    grandparent,
                )


def rule_type_propagation(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """type(x, A) and subClassOf(A, B) entail type(x, B)."""
    for subclass, parents in index.subclass_of.items():
        for type_fact in view.types_by_class.get(subclass, ()):
            for parent, license_id in parents:
                yield Derivation(
                    RULE_TYPE_PROPAGATION,
                    (type_fact.id, license_id),
                    type_fact.subject,
                    TYPE_PREDICATE,
                    parent,
                )


def rule_subproperty_transitivity(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """subPropertyOf(p, q) and subPropertyOf(q, r) entail subPropertyOf(p, r)."""
    del view
    for sub, supers in index.subproperty_of.items():
        for middle, first_id in supers:
            for top, second_id in index.subproperty_of.get(middle, ()):
                yield Derivation(
                    RULE_SUBPROPERTY_TRANSITIVITY,
                    (first_id, second_id),
                    sub,
                    SUBPROPERTY_OF,
                    top,
                )


def rule_subproperty_value_propagation(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """p(x, y) and subPropertyOf(p, q) entail q(x, y)."""
    for prop, supers in index.subproperty_of.items():
        for fact in view.by_predicate.get(prop, ()):
            for super_property, license_id in supers:
                yield Derivation(
                    RULE_SUBPROPERTY_VALUE_PROPAGATION,
                    (fact.id, license_id),
                    fact.subject,
                    super_property,
                    fact.object,
                )


def rule_domain_type(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """p(x, y) and domain(p, C) entail type(x, C)."""
    for prop, classes in index.domain_of.items():
        for fact in view.by_predicate.get(prop, ()):
            for class_name, license_id in classes:
                yield Derivation(
                    RULE_DOMAIN_TYPE,
                    (fact.id, license_id),
                    fact.subject,
                    TYPE_PREDICATE,
                    class_name,
                )


def rule_range_type(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """p(x, y) and range(p, C) entail type(y, C)."""
    for prop, classes in index.range_of.items():
        for fact in view.by_predicate.get(prop, ()):
            for class_name, license_id in classes:
                yield Derivation(
                    RULE_RANGE_TYPE,
                    (fact.id, license_id),
                    fact.object,
                    TYPE_PREDICATE,
                    class_name,
                )


def rule_inverse_of(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """p(x, y) and inverseOf(p, q) entail q(y, x)."""
    for prop, inverses in index.inverse_of.items():
        for fact in view.by_predicate.get(prop, ()):
            for inverse_property, license_id in inverses:
                yield Derivation(
                    RULE_INVERSE_OF,
                    (fact.id, license_id),
                    fact.object,
                    inverse_property,
                    fact.subject,
                )


def rule_property_transitivity(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """p(x, y), p(y, z) and transitiveProperty(p) entail p(x, z)."""
    for prop, license_id in index.transitive.items():
        for first in view.by_predicate.get(prop, ()):
            for second in view.by_pred_subject.get((prop, first.object), ()):
                yield Derivation(
                    RULE_PROPERTY_TRANSITIVITY,
                    (first.id, second.id, license_id),
                    first.subject,
                    prop,
                    second.object,
                )


def rule_property_symmetry(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """p(x, y) and symmetricProperty(p) entail p(y, x)."""
    for prop, license_id in index.symmetric.items():
        for fact in view.by_predicate.get(prop, ()):
            yield Derivation(
                RULE_PROPERTY_SYMMETRY,
                (fact.id, license_id),
                fact.object,
                prop,
                fact.subject,
            )


def _unify(term: HornTerm, value: str, bindings: dict[str, str]) -> bool:
    """Match a term against a value, binding a free variable, or reject on conflict.

    A variable repeated within a rule -- the join that binds an audit to its own content
    hash -- is enforced here: the second occurrence must agree with the first.
    """
    if not term.is_variable:
        return term.value == value
    bound = bindings.get(term.value)
    if bound is None:
        bindings[term.value] = value
        return True
    return bound == value


def _candidates(view: FactView, atom: HornAtom, bindings: Mapping[str, str]) -> tuple[Fact, ...]:
    """Return the facts that could match an atom, narrowed by whatever is already bound."""
    subject = (
        atom.subject.value if not atom.subject.is_variable else bindings.get(atom.subject.value)
    )
    object_value = (
        atom.object.value if not atom.object.is_variable else bindings.get(atom.object.value)
    )
    if subject is not None and object_value is not None:
        found = view.facts.get(compute_fact_id(subject, atom.predicate, object_value))
        return () if found is None else (found,)
    if subject is not None:
        return view.by_pred_subject.get((atom.predicate, subject), ())
    if object_value is not None:
        return view.by_pred_object.get((atom.predicate, object_value), ())
    return view.by_predicate.get(atom.predicate, ())


def _match_body(
    atoms: tuple[HornAtom, ...],
    position: int,
    view: FactView,
    bindings: dict[str, str],
) -> Iterator[tuple[dict[str, str], tuple[str, ...]]]:
    """Enumerate every way the body matches the closure, with the facts each match used."""
    if position == len(atoms):
        yield (dict(bindings), ())
        return
    atom = atoms[position]
    for fact in _candidates(view, atom, bindings):
        extended = dict(bindings)
        pairs = ((atom.subject, fact.subject), (atom.object, fact.object))
        if not all(_unify(term, value, extended) for term, value in pairs):
            continue
        for final_bindings, premises in _match_body(atoms, position + 1, view, extended):
            yield (final_bindings, (fact.id, *premises))


def _ground(term: HornTerm, bindings: Mapping[str, str]) -> str:
    """Resolve a head term against the bindings; safety guarantees a variable is bound."""
    return bindings[term.value] if term.is_variable else term.value


def rule_horn(index: AxiomIndex, view: FactView) -> Iterator[Derivation]:
    """Fire every asserted Horn rule: a conjunctive body entails its head.

    The premises recorded are the body facts the match consumed *plus the rule fact
    itself, so `explain` shows both the evidence and the rule that read it -- for a
    derived class such as `Approved`, that is the verdict, the hash join, and the rule,
    which is what E1 asks for.
    """
    for indexed in index.horn_rules:
        for bindings, premises in _match_body(indexed.rule.body, 0, view, {}):
            deduplicated = tuple(dict.fromkeys((*premises, indexed.fact_id)))
            yield Derivation(
                f"{RULE_HORN}:{indexed.name}",
                deduplicated,
                _ground(indexed.rule.head.subject, bindings),
                indexed.rule.head.predicate,
                _ground(indexed.rule.head.object, bindings),
            )


DERIVATION_RULES: Final[tuple[Callable[[AxiomIndex, FactView], Iterator[Derivation]], ...]] = (
    rule_subclass_transitivity,
    rule_type_propagation,
    rule_subproperty_transitivity,
    rule_subproperty_value_propagation,
    rule_domain_type,
    rule_range_type,
    rule_inverse_of,
    rule_property_transitivity,
    rule_property_symmetry,
    rule_horn,
)


# --------------------------------------------------------------------------------------
# The three contradiction detectors
# --------------------------------------------------------------------------------------


def detect_disjoint(index: AxiomIndex, view: FactView) -> Iterator[Inconsistency]:
    """Report an individual that is typed into two classes declared disjoint."""
    for subject, type_facts in view.types_by_subject.items():
        by_class: dict[str, Fact] = {fact.object: fact for fact in type_facts}
        for class_name, others in index.disjoint_with.items():
            if class_name not in by_class:
                continue
            for other, license_id in others:
                if other == class_name or other not in by_class or class_name > other:
                    continue
                conflicting = sorted((by_class[class_name].id, by_class[other].id))
                yield Inconsistency(
                    kind=KIND_DISJOINT,
                    facts=(*conflicting, license_id),
                    explanation=(
                        f"'{subject}' is typed as both '{class_name}' and '{other}', which are "
                        "declared disjoint; one of the two type assertions is wrong."
                    ),
                )


def detect_cardinality(index: AxiomIndex, view: FactView) -> Iterator[Inconsistency]:
    """Report an individual holding more property values than a max, or fewer than a min.

    Cardinality is scoped to a class, so only individuals already typed into that class are
    checked. A property-global minimum would otherwise flag every individual in the graph
    that simply does not use the property, which is noise rather than a finding.
    """
    for class_name, constraints in index.max_cardinality.items():
        for prop, limit, license_id in constraints:
            for type_fact in view.types_by_class.get(class_name, ()):
                values = view.by_pred_subject.get((prop, type_fact.subject), ())
                distinct = {fact.object: fact for fact in values}
                if len(distinct) > limit:
                    value_ids = sorted(fact.id for fact in distinct.values())
                    yield Inconsistency(
                        kind=KIND_CARDINALITY,
                        facts=(*value_ids, type_fact.id, license_id),
                        explanation=(
                            f"'{type_fact.subject}' is a '{class_name}' and has "
                            f"{len(distinct)} distinct '{prop}' values, but at most "
                            f"{limit} is allowed."
                        ),
                    )
    for class_name, constraints in index.min_cardinality.items():
        for prop, floor, license_id in constraints:
            for type_fact in view.types_by_class.get(class_name, ()):
                values = view.by_pred_subject.get((prop, type_fact.subject), ())
                distinct = {fact.object: fact for fact in values}
                if len(distinct) < floor:
                    value_ids = sorted(fact.id for fact in distinct.values())
                    yield Inconsistency(
                        kind=KIND_CARDINALITY,
                        facts=(*value_ids, type_fact.id, license_id),
                        explanation=(
                            f"'{type_fact.subject}' is a '{class_name}' and has "
                            f"{len(distinct)} distinct '{prop}' values, but at least "
                            f"{floor} is required."
                        ),
                    )


def detect_functional(index: AxiomIndex, view: FactView) -> Iterator[Inconsistency]:
    """Report functional and inverse-functional violations, the two identity contradictions."""
    for prop, license_id in index.functional.items():
        subjects = sorted({fact.subject for fact in view.by_predicate.get(prop, ())})
        for subject in subjects:
            values = view.by_pred_subject.get((prop, subject), ())
            distinct = {value.object: value for value in values}
            if len(distinct) > 1:
                conflicting = sorted(value.id for value in distinct.values())
                yield Inconsistency(
                    kind=KIND_FUNCTIONAL,
                    facts=(*conflicting, license_id),
                    explanation=(
                        f"'{prop}' is functional, so '{subject}' may have one value, but it "
                        f"has {len(distinct)}: {', '.join(sorted(distinct))}."
                    ),
                )
    for prop, license_id in index.inverse_functional.items():
        objects = sorted({fact.object for fact in view.by_predicate.get(prop, ())})
        for object_value in objects:
            holders = view.by_pred_object.get((prop, object_value), ())
            distinct = {holder.subject: holder for holder in holders}
            if len(distinct) > 1:
                conflicting = sorted(holder.id for holder in distinct.values())
                yield Inconsistency(
                    kind=KIND_INVERSE_FUNCTIONAL,
                    facts=(*conflicting, license_id),
                    explanation=(
                        f"'{prop}' is inverse-functional, so '{object_value}' identifies one "
                        f"subject, but {len(distinct)} claim it: {', '.join(sorted(distinct))}."
                    ),
                )


DETECTION_RULES: Final[tuple[Callable[[AxiomIndex, FactView], Iterator[Inconsistency]], ...]] = (
    detect_disjoint,
    detect_cardinality,
    detect_functional,
)
