"""Data models and JSON-RPC definitions for ACP (Agent Client Protocol) v0.12.1."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

ACP_PROTOCOL_VERSION: Final[str] = "0.12.1"
SERVER_NAME: Final[str] = "uclone-x"

# Standard JSON-RPC 2.0 and ACP Error Codes
PARSE_ERROR: Final[int] = -32700
INVALID_REQUEST: Final[int] = -32600
METHOD_NOT_FOUND: Final[int] = -32601
INVALID_PARAMS: Final[int] = -32602
INTERNAL_ERROR: Final[int] = -32603

# Application error codes
SESSION_NOT_FOUND: Final[int] = -32000
TURN_CANCELLED: Final[int] = -32001
UNSUPPORTED_MCP_TRANSPORT: Final[int] = -32002
PERMISSION_DENIED: Final[int] = -32003


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request model."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | list[Any] | None = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response model."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = None
    result: Any = None
    error: dict[str, Any] | None = None


class JSONRPCNotification(BaseModel):
    """JSON-RPC 2.0 Notification model (no id)."""

    model_config = ConfigDict(extra="ignore")

    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict[str, Any] | list[Any] | None = None


def make_jsonrpc_request(
    method: str,
    params: dict[str, Any] | list[Any] | None = None,
    req_id: int | str | None = 1,
) -> dict[str, Any]:
    """Format a JSON-RPC 2.0 request dictionary."""
    req: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
    }
    if params is not None:
        req["params"] = params
    return req


def make_jsonrpc_response(req_id: int | str | None, result: Any) -> dict[str, Any]:
    """Format a JSON-RPC 2.0 success response dictionary."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


def make_jsonrpc_error(
    req_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """Format a JSON-RPC 2.0 error response dictionary."""
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": err,
    }


def make_jsonrpc_notification(
    method: str, params: dict[str, Any] | list[Any] | None = None
) -> dict[str, Any]:
    """Format a JSON-RPC 2.0 notification dictionary."""
    notif: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        notif["params"] = params
    return notif


class ACPCapabilities(BaseModel):
    """Capability declaration of the ACP server (derived from §2 & §3.3)."""

    model_config = ConfigDict(frozen=True)

    session: dict[str, Any] = Field(
        default_factory=lambda: {
            "load": True,
            "cancel": True,
            "modes": ["auto", "ask", "readonly"],
        }
    )
    prompt: dict[str, Any] = Field(default_factory=lambda: {"streaming": True})
    permissions: dict[str, Any] = Field(default_factory=lambda: {"request": True})


class ACPSessionState(BaseModel):
    """Runtime tracking of an active ACP session."""

    session_id: str
    mode: str = "auto"
    config: dict[str, Any] = Field(default_factory=dict)
    cwd: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
