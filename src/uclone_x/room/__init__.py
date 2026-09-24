"""Multi-agent chat rooms: a shared transcript over agents that keep separate sessions."""

from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomMessageKind,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
    TurnState,
)
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.protocols import (
    RoomAgentResolverProtocol,
    RoomOrchestratorProtocol,
    RoomStoreProtocol,
    SpeakerSelectorProtocol,
)
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import (
    MENTION_PATTERN,
    DefaultResponderSelector,
    LLMSpeakerSelector,
    MentionSelector,
    SoleAgentSelector,
    build_selector_chain,
)
from uclone_x.room.service import (
    SESSION_ID_PREFIX,
    SESSION_ID_SEPARATOR,
    RoomService,
    RoomSummary,
    participant_session_id,
)
from uclone_x.room.store import RoomStore
from uclone_x.room.turn_summary import (
    TurnDocument,
    TurnNotFoundError,
    TurnSummary,
    summarize_turn,
)

__all__ = [
    "DefaultResponderSelector",
    "LLMSpeakerSelector",
    "MENTION_PATTERN",
    "MentionSelector",
    "Participant",
    "ParticipantKind",
    "RoomAgentResolver",
    "RoomAgentResolverProtocol",
    "RoomMessage",
    "RoomMessageKind",
    "RoomOrchestrator",
    "RoomOrchestratorProtocol",
    "RoomPolicy",
    "RoomService",
    "RoomState",
    "RoomStore",
    "RoomStoreProtocol",
    "RoomSummary",
    "SESSION_ID_PREFIX",
    "SESSION_ID_SEPARATOR",
    "participant_session_id",
    "SelectionVerdict",
    "SpeakerDecision",
    "SpeakerRequest",
    "SpeakerSelectorProtocol",
    "SoleAgentSelector",
    "TurnDocument",
    "TurnNotFoundError",
    "TurnState",
    "TurnSummary",
    "build_selector_chain",
    "summarize_turn",
]
