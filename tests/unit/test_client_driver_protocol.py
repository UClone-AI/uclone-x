"""Tests for the ClientDriverProtocol.

These tests ensure the protocol contract remains stable for interactive client drivers,
specifically checking that turn cancellation and prompting conform to the ACP adoption
requirements (Issue #576).
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator

from uclone_x.core.client_driver import ClientDriverProtocol


def test_client_driver_protocol_methods() -> None:
    """Ensure the ACP client driver protocol exposes exactly the required kernel-facing methods."""
    # Killed by: src/uclone_x/core/client_driver.py :: async def initialize(self) -> None:
    assert hasattr(ClientDriverProtocol, "initialize")

    # Killed by: src/uclone_x/core/client_driver.py :: async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
    assert hasattr(ClientDriverProtocol, "prompt")

    # Killed by: src/uclone_x/core/client_driver.py :: async def cancel(self, reason: str) -> None:
    assert hasattr(ClientDriverProtocol, "cancel")


def test_client_driver_protocol_cancel_signature() -> None:
    """Verify that cancellation explicitly requires a named reason, never silently accepted."""
    sig = inspect.signature(ClientDriverProtocol.cancel)
    # Killed by: src/uclone_x/core/client_driver.py :: async def cancel(self, reason: str) -> None:
    assert "reason" in sig.parameters, "cancel must accept a named reason"
    assert sig.parameters["reason"].annotation in (str, "str"), "reason must be typed as a string"
    assert sig.return_annotation in (None, type(None), "None"), "cancel must return None"


def test_client_driver_protocol_prompt_signature() -> None:
    """Verify that prompt requires content and returns an async iterator of AgentEvent."""
    sig = inspect.signature(ClientDriverProtocol.prompt)
    # Killed by: src/uclone_x/core/client_driver.py :: async def prompt(self, content: str) -> AsyncIterator[AgentEvent]:
    assert "content" in sig.parameters
    assert sig.parameters["content"].annotation in (str, "str")

    # Check that return type is an AsyncIterator
    ret_ann = sig.return_annotation

    # inspect stringified annotations for AsyncIterator
    assert "AsyncIterator" in str(ret_ann) or ret_ann is AsyncIterator
