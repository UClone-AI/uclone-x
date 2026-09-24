"""Adapters bridging UClone-X's core harness into the `uclone2` / Google ADK ecosystem.

Issue #367 names three deliverables for this package. Only the first —
`ADKContentAdapter` — is implemented here. `PostgresSessionStore` and
`EventBusMattermostBridge` are not decomposed into cards yet, and the former implements
`SessionStoreProtocol` while PR #292 is still open against `SessionStore.save`'s revision
precondition (#248), so writing a second implementation of that protocol now would build
against a contract under change.
"""

from __future__ import annotations

from uclone_x.adapters.uclone2.adk_content import (
    ADK_ROLE_MODEL,
    ADK_ROLE_USER,
    TOOL_RESULT_KEY,
    ADKContent,
    ADKContentAdapter,
    ADKFunctionCall,
    ADKFunctionResponse,
    ADKPart,
)

__all__ = [
    "ADK_ROLE_MODEL",
    "ADK_ROLE_USER",
    "TOOL_RESULT_KEY",
    "ADKContent",
    "ADKContentAdapter",
    "ADKFunctionCall",
    "ADKFunctionResponse",
    "ADKPart",
]
