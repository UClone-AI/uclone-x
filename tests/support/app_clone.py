"""A clone built as the desktop app builds it, for tests that need one agent (#1893 item 4).

Every head builds a clone through `build_clone` over the app's scope; a room seat adds only
its framing, display name and transport, and a 1:1 conversation is a one-seat room whose
seat has no framing. `AgentSessionManager.get_or_create_agent`, which tests used to reach
for, was a second entry to that builder that nothing in the app called any more, with a
chat-only agent cache and a transcript rebuild of its own. It is gone; these helpers take
the path the app takes.
"""

from __future__ import annotations

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.clone_builder import build_clone
from uclone_x.ui.app import AgentSessionManager


def app_clone(
    mgr: AgentSessionManager,
    clone_id: str,
    session_id: str | None = None,
    *,
    model_name: str | None = None,
) -> BaseAgent:
    """Clone `clone_id` in `session_id`, built by `build_clone` from `mgr`'s app scope.

    Resumed from its session record before it is returned, as a room seat is before its
    first turn. `session_id` defaults to `sess_<clone_id>`.
    """
    built = build_clone(
        mgr.app_scope(),
        clone_id=clone_id,
        session_id=session_id or f"sess_{clone_id}",
        model_name=model_name,
    )
    built.agent.hydrate_session()
    return built.agent
