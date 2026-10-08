"""What a vessel decides for itself (uGraph design §3.1): the part that is not the model.

A policy is data, not code: a store reads it and a later step reads it again. The clone's
is here; the story's arrives with its store (step 2).

This module is pure.
"""

from __future__ import annotations

from dataclasses import dataclass

from uclone_x.knowledge.models import Axis, EdgeStatus

__all__ = [
    "CLONE_POLICY",
    "CLONE_RESERVED_ENTITIES",
    "PERSON_ENTITY",
    "PROJECT_ENTITY",
    "SELF_ENTITY",
    "ClonePolicy",
]

#: The three entities every clone has a fact about from its first day, by id. Their ids are
#: their names, so a name resolves to them by the ordinary exact stage. `user` is the person
#: the clone works for, `project` this workspace's durable preferences, and `self` the clone
#: itself (#2016, #2074). `memory.models` files a fact under these names.
PERSON_ENTITY = "user"
PROJECT_ENTITY = "project"
SELF_ENTITY = "self"
#: `(id, kind)`; created the first time a fact names them, through the resolver like any
#: other entity, and never given another id.
CLONE_RESERVED_ENTITIES = (
    (PERSON_ENTITY, "person"),
    (PROJECT_ENTITY, "project"),
    (SELF_ENTITY, "clone"),
)


@dataclass(frozen=True)
class ClonePolicy:
    """The clone's vessel: real time, and a fact is believed as soon as it is written.

    `initial_status`: a new edge starts `approved`, with no person in between (design G3).
    `axis`: `valid` is a wall-clock interval.
    `reserved`: the entities whose id is their name (`CLONE_RESERVED_ENTITIES`).

    There is no switch for merging a near candidate (resolver stage 3): in this step a near
    name is a new entity, and stage 4 (a model confirms) with the undoable merge is step 5
    (owner question Q2).
    """

    initial_status: EdgeStatus = "approved"
    axis: Axis = "wall"
    reserved: tuple[tuple[str, str], ...] = CLONE_RESERVED_ENTITIES


CLONE_POLICY = ClonePolicy()
