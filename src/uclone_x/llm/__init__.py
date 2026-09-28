"""LLM-Agnostic subsystem: Multi-provider abstraction, token budgeting, and context compaction."""

# The connector adapters are re-exported lazily (PEP 562). Importing any module in this
# package runs this file first, so an eager `from uclone_x.llm.connectors import ...` here
# made `import uclone_x.llm.models` -- a kernel contract `core/` reads -- load every
# connector adapter and `httpx` with it (#1775). `from uclone_x.llm import
# MockLLMConnector` still works; it now loads the connectors on first use.

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.models import (
    BudgetDecision,
    ChatMessage,
    CompactionOutcome,
    FinishReason,
    LedgerSource,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenBudget,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.llm.protocols import (
    ContextCompactorProtocol,
    EmbedderProtocol,
    LLMProviderProtocol,
    TokenBudgetManagerProtocol,
)

if TYPE_CHECKING:
    from uclone_x.llm.connectors import (
        DEFAULT_EMBEDDING_DIMENSIONS,
        DEFAULT_EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS_ENV_VAR,
        EMBEDDING_MODEL_ENV_VAR,
        AnthropicConnector,
        BaseLLMConnector,
        GeminiConnector,
        MockLLMConnector,
        OllamaConnector,
        OllamaEmbedder,
        OpenAIConnector,
        VLLMConnector,
        create_llm_connector,
        has_configured_vllm_endpoint,
        normalize_ollama_base_url,
        normalize_vllm_base_url,
        resolve_embedding_dimensions,
        resolve_embedding_model,
        resolve_ollama_base_url,
        resolve_ollama_model,
        resolve_vllm_base_url,
        resolve_vllm_model,
    )

#: The names re-exported from `uclone_x.llm.connectors`, loaded on first access.
_CONNECTOR_NAMES = frozenset(
    {
        "DEFAULT_EMBEDDING_DIMENSIONS",
        "DEFAULT_EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS_ENV_VAR",
        "EMBEDDING_MODEL_ENV_VAR",
        "AnthropicConnector",
        "BaseLLMConnector",
        "GeminiConnector",
        "MockLLMConnector",
        "OllamaConnector",
        "OllamaEmbedder",
        "OpenAIConnector",
        "VLLMConnector",
        "create_llm_connector",
        "has_configured_vllm_endpoint",
        "normalize_ollama_base_url",
        "normalize_vllm_base_url",
        "resolve_embedding_dimensions",
        "resolve_embedding_model",
        "resolve_ollama_base_url",
        "resolve_ollama_model",
        "resolve_vllm_base_url",
        "resolve_vllm_model",
    }
)


def __getattr__(name: str) -> Any:
    """Load a connector name on first access, so importing this package loads none."""
    if name in _CONNECTOR_NAMES:
        value = getattr(importlib.import_module("uclone_x.llm.connectors"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicConnector",
    "BaseLLMConnector",
    "BudgetDecision",
    "ChatMessage",
    "CompactionOutcome",
    "ContextCompactor",
    "ContextCompactorProtocol",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS_ENV_VAR",
    "EMBEDDING_MODEL_ENV_VAR",
    "EmbedderProtocol",
    "FinishReason",
    "GeminiConnector",
    "LLMProviderProtocol",
    "LLMRequest",
    "LedgerSource",
    "MessageRole",
    "MockLLMConnector",
    "ModelResponse",
    "OllamaConnector",
    "OllamaEmbedder",
    "OpenAIConnector",
    "VLLMConnector",
    "StreamChunk",
    "TokenBudget",
    "TokenBudgetManager",
    "TokenBudgetManagerProtocol",
    "TokenCountSource",
    "TokenUsage",
    "ToolCallRequest",
    "ToolDefinition",
    "create_llm_connector",
    "has_configured_vllm_endpoint",
    "normalize_ollama_base_url",
    "normalize_vllm_base_url",
    "resolve_embedding_dimensions",
    "resolve_embedding_model",
    "resolve_ollama_base_url",
    "resolve_ollama_model",
    "resolve_vllm_base_url",
    "resolve_vllm_model",
]
