"""LLM provider connector implementations (Principle 5 & Principle 6)."""

from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.connectors.ollama import (
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    OllamaConnector,
    normalize_ollama_base_url,
    resolve_ollama_base_url,
    resolve_ollama_model,
    resolve_ollama_timeout,
)
from uclone_x.llm.connectors.ollama_embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS_ENV_VAR,
    EMBEDDING_MODEL_ENV_VAR,
    OllamaEmbedder,
    resolve_embedding_dimensions,
    resolve_embedding_model,
)
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.connectors.vllm import (
    VLLM_ENDPOINT_ENV_VARS,
    VLLM_MODEL_ENV_VAR,
    VLLMConnector,
    has_configured_vllm_endpoint,
    normalize_vllm_base_url,
    resolve_vllm_base_url,
    resolve_vllm_model,
)

__all__ = [
    "AnthropicConnector",
    "BaseLLMConnector",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_OLLAMA_TIMEOUT_SECONDS",
    "EMBEDDING_DIMENSIONS_ENV_VAR",
    "EMBEDDING_MODEL_ENV_VAR",
    "GeminiConnector",
    "MockLLMConnector",
    "OllamaConnector",
    "OllamaEmbedder",
    "OpenAIConnector",
    "VLLMConnector",
    "VLLM_ENDPOINT_ENV_VARS",
    "VLLM_MODEL_ENV_VAR",
    "create_llm_connector",
    "has_configured_vllm_endpoint",
    "normalize_ollama_base_url",
    "normalize_vllm_base_url",
    "resolve_embedding_dimensions",
    "resolve_embedding_model",
    "resolve_ollama_base_url",
    "resolve_ollama_model",
    "resolve_ollama_timeout",
    "resolve_vllm_base_url",
    "resolve_vllm_model",
]
