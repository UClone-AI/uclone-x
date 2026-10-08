"""Where an extension's HTTP routes are mounted on the web head (#2205).

The head hands each extension's `routes` (`extensions.Extension.routes`) one
`ExtensionRouteContext`: the app, the room stack, the person gate, and the head's
cross-origin refusal. The routes are the extension's; this module only builds what they
are given and calls them, after the core's own routes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request

from uclone_x.extensions import mount_extension_routes

if TYPE_CHECKING:
    from uclone_x.ui.person import PersonGate
    from uclone_x.ui.rooms import RoomStack

__all__ = ["ExtensionRouteContext", "register_extension_routes"]


@dataclass(frozen=True)
class ExtensionRouteContext:
    """What an extension's routes are given."""

    app: FastAPI
    stack: RoomStack
    #: Asks whether a request came from a window this server confirmed (#1589 item 6).
    person: PersonGate
    #: Refuses a request another site's page made (#2146).
    refuse_cross_origin: Callable[[Request], None]

    def turn_busy(self, room_id: str) -> bool:
        """Whether conversation `room_id` is answering now, as the Files screen asks."""
        # A retry runs its turn inside the request, not as a cascade, so `turn_in_flight`
        # alone misses it (#1578).
        return self.stack.turn_in_flight(room_id) or self.stack.turn_unlanded(room_id)


def register_extension_routes(
    app: FastAPI,
    stack: RoomStack,
    person: PersonGate,
    *,
    refuse_cross_origin: Callable[[Request], None],
) -> list[str]:
    """Mount every extension's routes; return the names of the extensions that added some."""
    return mount_extension_routes(
        ExtensionRouteContext(
            app=app, stack=stack, person=person, refuse_cross_origin=refuse_cross_origin
        )
    )
