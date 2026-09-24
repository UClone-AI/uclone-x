"""Google A2A v1.0.1 subsystem: Discovery, local fastpath, and remote REST/SSE wire protocol."""

from uclone_x.a2a.discovery import A2ADiscoveryService
from uclone_x.a2a.http_transport import A2AHttpTransport
from uclone_x.a2a.in_memory import A2AInMemoryTransport
from uclone_x.a2a.models import (
    AgentCard,
    TaskMessage,
    TaskResult,
    TaskStatus,
    WireProtocolType,
)
from uclone_x.a2a.protocols import (
    A2ADiscoveryProtocol,
    A2ATransportProtocol,
)

__all__ = [
    "A2ADiscoveryProtocol",
    "A2ADiscoveryService",
    "A2AHttpTransport",
    "A2AInMemoryTransport",
    "A2ATransportProtocol",
    "AgentCard",
    "TaskMessage",
    "TaskResult",
    "TaskStatus",
    "WireProtocolType",
]
