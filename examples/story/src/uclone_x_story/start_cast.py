# pyright: reportPrivateUsage=false
"""Stage 2 (Cast, places, family relations, and mid-story character introductions)."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError

from uclone_x_story.continuity import INVERSE_ROLE, family_role, gender_of
from uclone_x_story.kg_plan import split_entity_name
from uclone_x_story.library import StoryError
from uclone_x_story.names import id_for_name
from uclone_x_story.proposals import new_entry_clash
from uclone_x_story.scene_illustration import strip_non_ascii
from uclone_x_story.schemas import Proposal, status_of
from uclone_x_story.start_state import (
    _SYSTEM,
    MAX_CAST_TRIES,
    MAX_CHARACTERS,
    MAX_PLACES,
    _aliases,
    _brief_lines,
    _RunBase,
    _text,
    _texts,
    json_object,
)
from uclone_x_story.tools import NewEntryParams, new_entry_draft

#: The reason a cast reply could not be used, as the model is told it when asked again.
_CAST_RETRY: Mapping[str, str] = {
    "unreadable": "it was not one valid JSON object -- a key had an extra quotation mark "
    'before it (write {"name": ...}, never { " "name": ...}), a value held a double quote, '
    "or it was cut off. Keep every profile and appearance to one short sentence, and write "
    "quotation marks inside a value as ‘ ’",
    "empty": "it listed no character with a name. The story needs its main characters",
    "taken": "every character it listed is already in the story. Propose the characters "
    "this premise still needs, under other names",
}


def normalize_cast_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Split a combined role/name string (e.g. ``"검은 고양이 '먹물'"``, ``"'먹물'"``)
    into ``name`` with any role or epithet kept in ``aliases``."""
    out = dict(entry)
    raw_name = _text(out.get("name"))
    proper, extra = split_entity_name(raw_name)
    if proper and proper != raw_name:
        out["name"] = proper
        out["aliases"] = [*extra, *_texts(out.get("aliases"))]
    return out


def _cast_relations(
    found: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """The family the cast reply names: each character's codex `relations:` (the other's id
    to what this character is to them) by name, and the gender a gendered kinship word
    gives ("daughter" is female).

    A character's ``family`` lists each relative and what the relative is to it, as in "my
    father". Each relation is written on both sides (the father's relation to the child is
    ``parent``, the child's to the father ``child``), with ids made as
    `new_entry_draft` makes them. qwen3:8b sometimes wrote the word from the other side (a
    daughter listing her father as "father" in one run, as "daughter" in another), so a
    gendered word that contradicts the relative's stated gender and matches the
    character's own is read the other way round, and one that matches neither is dropped.
    A relation to a name not in the cast, to oneself, or in a word `family_role` does not
    know is dropped; the first relation of a pair wins.
    """
    raw = found.get("characters")
    people = [
        normalize_cast_entry(cast(dict[str, Any], c))
        for c in (cast(list[object], raw) if isinstance(raw, list) else [])
        if isinstance(c, dict)
    ][:MAX_CHARACTERS]
    ids = {
        _text(c.get("name")).casefold(): (_text(c.get("name")), id_for_name(_text(c.get("name"))))
        for c in people
        if _text(c.get("name"))
    }
    for c in people:
        cname = _text(c.get("name"))
        if cname:
            target = ids[cname.casefold()]
            for alias in _aliases(c.get("aliases"), cname):
                ids.setdefault(alias.casefold(), target)
    stated = {
        _text(c.get("name")).casefold(): gender_of(c.get("gender"))
        for c in people
        if _text(c.get("name"))
    }
    relations: dict[str, dict[str, str]] = {name: {} for name, _ in ids.values()}
    gender: dict[str, str] = {}
    for person in people:
        name, own = ids.get(_text(person.get("name")).casefold(), ("", None))
        family = person.get("family")
        for relation in cast(list[object], family) if isinstance(family, list) else []:
            if not isinstance(relation, dict):
                continue
            rel = cast(dict[str, Any], relation)
            relative, relative_id = ids.get(_text(rel.get("relative")).casefold(), ("", None))
            found_role = family_role(rel.get("is"))
            if not own or not relative_id or relative_id == own or found_role is None:
                continue
            role, word_gender = found_role
            holder, holder_id, other, other_id = relative, relative_id, name, own
            if word_gender is not None and stated.get(relative.casefold()) not in (
                None,
                word_gender,
            ):
                if stated.get(name.casefold()) != word_gender:
                    continue
                holder, holder_id, other, other_id = name, own, relative, relative_id
            relations[holder].setdefault(other_id, role)
            relations[other].setdefault(holder_id, INVERSE_ROLE[role])
            if word_gender is not None:
                gender.setdefault(holder, word_gender)
    return relations, gender


def _cast_items(found: Mapping[str, Any]) -> set[str]:
    """The names a cast reply lists as objects (`items`), case folded. Not proposed here:
    the settled plot proposes the story's items (`_kg_params`); here they only keep an
    object out of the cast."""
    raw = found.get("items")
    listed = cast(list[object], raw) if isinstance(raw, list) else []
    names: set[str] = set()
    for item in listed:
        if isinstance(item, dict) and (name := _text(cast(dict[str, Any], item).get("name"))):
            names.add(name.casefold())
        elif isinstance(item, str) and item.strip():
            names.add(item.strip().casefold())
    return names


def _cast_params(found: Mapping[str, Any], notes: list[str]) -> list[NewEntryParams]:
    """The entries a cast reply asks for, each shaped as a new codex entry.

    A character's state holds its ``status`` and its ``gender`` (female or male; "other"
    stays in the visual block only), and its `relations:` its family (`_cast_relations`),
    which the continuity check holds each scene to.
    """
    family, family_gender = _cast_relations(found)
    objects = _cast_items(found)
    asked: list[NewEntryParams] = []
    for kind, cap in (("characters", MAX_CHARACTERS), ("places", MAX_PLACES)):
        raw = found.get(kind)
        items = cast(list[object], raw)[:cap] if isinstance(raw, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = (
                normalize_cast_entry(cast(dict[str, Any], item))
                if kind == "characters"
                else cast(dict[str, Any], item)
            )
            name = _text(entry.get("name"))
            if not name:
                continue
            if name.casefold() in objects:
                # The reply lists it as an object too: it is not cast (#2301, "은시계").
                notes.append(f"'{name}' is an object the story turns on, so it was not cast.")
                continue
            gender = _text(entry.get("gender")).lower()
            if gender not in ("female", "male", "other"):
                gender = family_gender.get(name, "")
            status = status_of(entry.get("status"))
            facts: dict[str, Any] = {"status": status}
            if gender in ("female", "male"):
                facts["gender"] = gender
            visual_tags: list[str] = []
            if kind == "characters":
                for raw_tag in _texts(entry.get("visual_tags")):
                    cleaned = strip_non_ascii(raw_tag)
                    if cleaned:
                        for part in (p.strip() for p in cleaned.split(",") if p.strip()):
                            if part not in visual_tags:
                                visual_tags.append(part)
            try:
                asked.append(
                    NewEntryParams(
                        kind=kind,  # pyright: ignore[reportArgumentType]
                        name=name,
                        aliases=_aliases(entry.get("aliases"), name),
                        profile=_text(entry.get("profile")),
                        # Where the premise leaves each character when the story opens --
                        # the missing father is missing, not alive -- so the continuity
                        # checks hold a scene to the premise.
                        state=facts if kind == "characters" else {},
                        relations=family.get(name, {}) if kind == "characters" else {},
                        appearance=(_text(entry.get("appearance")) or None)
                        if kind == "characters"
                        else None,
                        gender=cast(Any, gender)
                        if kind == "characters" and gender in ("female", "male", "other")
                        else None,
                        visual_tags=visual_tags,
                    )
                )
            except ValidationError:
                notes.append(f"'{name}' did not fit a codex entry and was left out.")
    return asked


def _cast_ids(cast_lines: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """The cast's ids by `(kind, name)`, the name or alias casefolded."""
    out: dict[tuple[str, str], str] = {
        (str(c.get("kind")), str(c.get("name", "")).casefold()): str(c.get("id"))
        for c in cast_lines
    }
    for c in cast_lines:
        kind = str(c.get("kind"))
        cid = str(c.get("id"))
        for alias in _texts(c.get("aliases")):
            out.setdefault((kind, alias.casefold()), cid)
    return out


#: What a chapter plan is asked about the characters it brings in (`_new_characters`).
_NEW_CHARACTERS_ASK = (
    "If this chapter brings in a character who is not in the cast above and is family of "
    "someone in it, list them under new_characters, with what that cast member is to them "
    '(as in "my father": father, mother, son, daughter, brother, sister, husband, wife, '
    "grandfather, grandmother, grandson or granddaughter). [] for none.\n"
)
_NEW_CHARACTERS_SHAPE = (
    '"new_characters": [{"name": ..., "profile": one sentence, "gender": "female" or '
    '"male", "family": [{"relative": a name from the cast above, "is": what that '
    "relative is to this new character}]}]"
)


def _new_characters(
    found: Mapping[str, Any] | None, known: Mapping[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    """The new characters a chapter plan brings in with family among the known characters.

    A name the story already has is not new; a relation to a name it does not have, or in
    a word `family_role` does not know, is dropped, and a new character left with no relation
    is not kept (a scene that brings them in proposes them when it is written, through
    `grow_scene`). A gender the new character's reply leaves out is not guessed.
    """
    if found is None:
        return []
    raw = found.get("new_characters")
    people = cast(list[object], raw) if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for person in people[:MAX_CHARACTERS]:
        if not isinstance(person, dict):
            continue
        p = cast(dict[str, Any], person)
        name = _text(p.get("name"))
        if not name or name.casefold() in known or name.casefold() in seen:
            continue
        given = p.get("family")
        family: list[dict[str, str]] = []
        for relation in cast(list[object], given) if isinstance(given, list) else []:
            if not isinstance(relation, dict):
                continue
            rel = cast(dict[str, Any], relation)
            relative = known.get(_text(rel.get("relative")).casefold())
            word = _text(rel.get("is"))
            role = family_role(word)
            if relative is None or role is None:
                continue
            if any(f["relative_id"] == relative[1] for f in family):
                continue
            family.append(
                {"relative": relative[0], "is": word, "relative_id": relative[1], "role": role[0]}
            )
        if not family:
            continue
        seen.add(name.casefold())
        gender = gender_of(p.get("gender"))
        out.append(
            {
                "name": name,
                "profile": _text(p.get("profile")),
                "gender": gender or "",
                "family": family,
            }
        )
    return out


class _RunCastMixin(_RunBase):
    """Stage 2 cast generation and mid-story character introductions."""

    async def _cast(self) -> None:
        """Stage 2: characters and places, each proposed as a new codex entry.

        A reply that yields no character is asked again, told why (`_CAST_RETRY`): no
        readable JSON (`json_object` already reads past a stray quotation mark before a
        key, which lost qwen3:8b's whole cast for a premise that quotes a name), no named
        character, or only characters the story has already. After
        `MAX_CAST_TRIES` the flow stops and says why, rather than outlining and writing a
        story with no one in it.
        """
        state = self.state
        premise = state.premise()
        prompt = (
            f"Story premise:\n{premise.title}: {premise.premise}\n\n"
            f"Direction:\n{_brief_lines(state.brief)}\n\n"
            f"List the main characters (at most {MAX_CHARACTERS}) and places (at most "
            f"{MAX_PLACES}) this story needs.\n"
            'Reply as JSON: {"characters": [{"name": ..., "aliases": [the other names the '
            "story's prose will call them by, such as a short form or role], "
            '"profile": who they are and what '
            'they want, in one sentence, "appearance": how they look, in a few words, '
            '"visual_tags": [3 to 5 short English Danbooru tags for hair, eyes, and outfit, '
            'e.g. "short black hair", "brown eyes", "wool coat"], '
            '"gender": "female", "male" or "other", "status": where they stand when the '
            'story begins, from the premise: "alive", "missing", "dead" or "unknown", '
            '"family": [{"relative": another character\'s name, "is": what that relative is '
            'to this character, as in "my father": "father", "mother", "son", "daughter", '
            '"brother", "sister", "husband", "wife", "grandfather", "grandmother", '
            '"grandson" or "granddaughter"}]}], '
            '"places": [{"name": ..., "profile": what it is, in a sentence}], '
            '"items": [{"name": ..., "profile": what it is, in a sentence}]}'
            "\nA character is someone who acts: a person, or another being. Give every "
            "character a proper name (if the premise calls someone only by a role such as "
            "‘소녀’ or ‘고양이’, give them a proper name and put the role word in aliases). "
            "An object the story turns on goes under items, never under characters or "
            "places. Give family only where the premise implies it, and [] for none. A "
            "daughter lists her father "
            'as {"relative": his name, "is": "father"}.'
            "\nA character the premise says has vanished is missing, one it says has died is "
            "dead, and one whose fate the story sets out to learn is unknown. Inside a value, "
            "write quotation marks as ‘ ’, never as a double quote."
        )
        notes = list(state.notes)
        saved: list[Proposal] = []
        cast_lines: list[dict[str, Any]] = []
        asked = prompt
        reason: str | None = "empty"
        tries = 0
        for _ in range(MAX_CAST_TRIES):
            tries += 1
            reply = await self.model.complete(
                asked, system=_SYSTEM, temperature=0.7, max_tokens=2000
            )
            found = json_object(reply)
            if found is None:
                reason = "unreadable"
            else:
                wanted = _cast_params(found, notes)
                reason = await self._propose(wanted, saved, cast_lines, notes)
            if reason is None:
                break
            asked = (
                f"{prompt}\n\nYour last reply could not be used: {_CAST_RETRY[reason]}. "
                "Reply again with the JSON object only."
            )
        if tries > 1:
            notes.append(f"The cast was asked for {tries} times.")
        if reason is not None:
            notes.append(f"No character was proposed after {tries} tries ({reason}).")
            self._save(
                cast=cast_lines,
                cast_proposals=[p.id for p in saved],
                cast_failed=reason,
                notes=notes,
            )
            return
        self._save(
            stage="review_cast",
            cast=cast_lines,
            cast_proposals=[p.id for p in saved],
            cast_failed=None,
            notes=notes,
        )

    async def _propose(
        self,
        wanted: Sequence[NewEntryParams],
        saved: list[Proposal],
        cast_lines: list[dict[str, Any]],
        notes: list[str],
    ) -> str | None:
        """Propose `wanted`; `None` once the story has a proposed character, else why not."""
        pending = [p for p, _ in self.work.proposals()[0] if p.status == "pending"]
        taken = False
        for entry_params in wanted:
            try:
                draft = new_entry_draft(entry_params, room_id=self.room_id, context=self.context)
            except StoryError as exc:
                notes.append(str(exc))
                continue
            clash = new_entry_clash(
                self.work, draft.kind, draft.new_entry(), pending=[*pending, *saved]
            )
            if clash is not None:
                if entry_params.kind == "characters":
                    taken = True
                notes.append(f"'{entry_params.name}' was not proposed: {clash}.")
                continue
            proposal = self.work.add_proposal(draft, room_id=self.room_id)
            saved.append(proposal)
            cast_lines.append(
                {
                    "id": draft.entry_id,
                    "kind": draft.kind,
                    "name": entry_params.name,
                    **({"aliases": entry_params.aliases} if entry_params.aliases else {}),
                    "profile": entry_params.profile,
                    **({"status": entry_params.state["status"]} if entry_params.state else {}),
                }
            )
        if any(c["kind"] == "characters" for c in cast_lines):
            return None
        return "taken" if taken else "empty"

    def _known_characters(self) -> dict[str, tuple[str, str]]:
        """Every character the story has, by folded name or alias: (name, id). The cast
        (proposed or approved), the codex's characters, and those an earlier plan
        introduced (proposed), so a later plan neither brings them in again nor loses
        them as a relative."""
        known: dict[str, tuple[str, str]] = {}
        for item in self.work.codex().items:
            if item.kind == "characters":
                for name in (item.entry.name, *item.entry.aliases):
                    if name:
                        known[name.casefold()] = (item.entry.name, item.entry.id)
        for c in self.state.cast:
            if c.get("kind") == "characters" and c.get("name"):
                pair = (str(c["name"]), str(c.get("id")))
                known[str(c["name"]).casefold()] = pair
                for alias in _texts(c.get("aliases")):
                    known.setdefault(alias.casefold(), pair)
        for k in self.state.introduced:
            if k.get("change") == "new_entry":
                known.setdefault(str(k["name"]).casefold(), (str(k["name"]), str(k["entry_id"])))
        return known

    def _introduce(self, number: int) -> list[dict[str, Any]]:
        """The new characters of chapter `number`'s accepted plan, proposed with their family.

        Each is a new-entry proposal whose `relations:` hold the role the new character has
        to the relative (the inverse of what the relative is to it), as `_cast_relations`
        writes the cast's. A relative the codex has (approved) also gets a progression
        proposal at the chapter's first scene, a relation to the new character: the kinship
        on both sides. A relative that is itself still a pending proposal has no entry to
        progress yet, so the relation is kept on the new character's side only. Nothing
        is approved here; a name the story already has, or has pending, is not proposed.
        """
        if not self.state.details_new:
            return []
        outline, _ = self.work.require_outline()
        if not 1 <= number <= len(outline.chapters) or not outline.chapters[number - 1].scenes:
            return []
        at = outline.chapters[number - 1].scenes[0].id
        pending = [p for p, _ in self.work.proposals()[0] if p.status == "pending"]
        notes = list(self.state.notes)
        added: list[dict[str, Any]] = []
        for person in self.state.details_new:
            family = cast(list[dict[str, str]], person.get("family") or [])
            facts: dict[str, Any] = {"status": "alive"}
            relations = {f["relative_id"]: INVERSE_ROLE[f["role"]] for f in family}
            gender = str(person.get("gender") or "")
            draft_fn = getattr(
                sys.modules.get("uclone_x_story.start"), "new_entry_draft", new_entry_draft
            )
            try:
                draft = draft_fn(
                    NewEntryParams(
                        kind="characters",
                        name=str(person["name"]),
                        profile=str(person.get("profile") or ""),
                        state=facts,
                        relations=relations,
                        gender=cast(Any, gender) if gender in ("female", "male") else None,
                    ),
                    room_id=self.room_id,
                    context=self.context,
                )
            except StoryError as exc:
                notes.append(str(exc))
                continue
            except ValidationError:
                # Plain words, not the validator's own text (#2382).
                notes.append(
                    f"'{person['name']}' was not proposed: what was read for them does not fit a codex entry."
                )
                continue
            clash = new_entry_clash(self.work, "characters", draft.new_entry(), pending=pending)
            if clash is not None:
                notes.append(f"'{person['name']}' was not proposed: {clash}.")
                continue
            proposal = self.work.add_proposal(draft, room_id=self.room_id)
            pending.append(proposal)
            added.append(
                {
                    "proposal": proposal.id,
                    "kind": "characters",
                    "entry_id": proposal.entry_id,
                    "name": str(person["name"]),
                    "chapter": number,
                    "change": "new_entry",
                }
            )
            for f in family:
                loaded = self.work.entry("characters", f["relative_id"])
                if loaded is None:  # a relative still pending: this side only
                    continue
                try:
                    back = Proposal.model_validate(
                        {
                            "id": "p000",
                            "kind": "characters",
                            "entry_id": f["relative_id"],
                            "change": {
                                "progression": {
                                    "at": at,
                                    "relations": {proposal.entry_id: f["role"]},
                                }
                            },
                            "proposed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                            "room_id": self.room_id,
                            "agent_id": self.context.agent_id,
                            "entry_digest": loaded[1],
                        }
                    )
                except (ValidationError, ValueError):
                    continue
                saved = self.work.add_proposal(back, room_id=self.room_id)
                added.append(
                    {
                        "proposal": saved.id,
                        "kind": "characters",
                        "entry_id": f["relative_id"],
                        "name": f["relative"],
                        "chapter": number,
                        "change": "relation",
                    }
                )
        self._save(introduced=[*self.state.introduced, *added], details_new=[], notes=notes)
        return added


__all__ = [
    "_CAST_RETRY",
    "_NEW_CHARACTERS_ASK",
    "_NEW_CHARACTERS_SHAPE",
    "_RunCastMixin",
    "_cast_ids",
    "_cast_items",
    "_cast_params",
    "_cast_relations",
    "_new_characters",
    "normalize_cast_entry",
]
