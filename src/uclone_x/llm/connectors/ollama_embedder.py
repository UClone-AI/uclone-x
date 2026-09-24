"""Ollama-backed text embedder (Principle 5 & Principle 6).

The default adapter behind `EmbedderProtocol`. It speaks Ollama's `/api/embed`, which is
the same endpoint the local reasoning connector already targets, so a machine that can run
this framework offline can also embed offline with no second service.

What this deliberately does not do: return a zero vector when the endpoint is down. The
prior art this port draws from did, and the consequence is that an unreachable embedding
service is indistinguishable, at the call site, from a corpus with nothing relevant in it —
every cosine similarity against a zero vector is exactly 0.0. Every failure here raises.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import Sequence
from typing import Any, cast

import httpx

from uclone_x.errors import EmbeddingDimensionError, EmbeddingError
from uclone_x.llm.connectors.ollama import describe_transport_error, resolve_ollama_base_url

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ENV_VAR = "UCLONE_EMBEDDING_MODEL"
EMBEDDING_DIMENSIONS_ENV_VAR = "UCLONE_EMBEDDING_DIMENSIONS"

DEFAULT_EMBEDDING_MODEL = "bge-m3"
"""Default embedding model.

Chosen multilingual-first. The model carried by the prior art (`all-MiniLM-L6-v2`, 384d)
ships in this repository's sibling with a comment conceding that its Korean similarity
scores fall below its own relevance threshold — i.e. its authors measured that it does not
embed Korean usefully and kept it anyway. `bge-m3` is trained multilingually and is the
candidate this default names until `evals/suites` measures a better one on the bilingual
set; the eval, not this constant, is the evidence.
"""

DEFAULT_EMBEDDING_DIMENSIONS = 1024
"""Width of `DEFAULT_EMBEDDING_MODEL`'s vectors."""


def resolve_embedding_model(model: str | None = None) -> str:
    """Resolve the embedding model from argument, then environment, then the default."""
    if model and model.strip():
        return model.strip()
    configured = os.getenv(EMBEDDING_MODEL_ENV_VAR)
    if configured is not None and configured.strip():
        return configured.strip()
    return DEFAULT_EMBEDDING_MODEL


def resolve_embedding_dimensions(dimensions: int | None = None) -> int:
    """Resolve the declared vector width from argument, then environment, then the default.

    A non-numeric or non-positive environment value raises rather than falling back to the
    default: the width decides what a vector store will accept for the life of the index,
    and silently substituting a different one is how a store ends up holding two widths.
    """
    if dimensions is not None:
        # `bool` is an `int`, and `True <= 0` is False, so an accidental boolean would
        # otherwise become a one-component width. Refused here for the same reason a
        # boolean is refused as a vector component.
        if isinstance(dimensions, bool):
            raise EmbeddingError(f"Embedding dimensions must be an integer, got {dimensions!r}.")
        if dimensions <= 0:
            raise EmbeddingError(f"Embedding dimensions must be positive, got {dimensions}.")
        return dimensions
    configured = os.getenv(EMBEDDING_DIMENSIONS_ENV_VAR)
    if configured is not None and configured.strip():
        try:
            parsed = int(configured.strip())
        except ValueError as exc:
            raise EmbeddingError(
                f"{EMBEDDING_DIMENSIONS_ENV_VAR} must be an integer, got {configured!r}."
            ) from exc
        if parsed <= 0:
            raise EmbeddingError(f"{EMBEDDING_DIMENSIONS_ENV_VAR} must be positive, got {parsed}.")
        return parsed
    return DEFAULT_EMBEDDING_DIMENSIONS


class OllamaEmbedder:
    """Embed text through an Ollama endpoint's `/api/embed`."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = resolve_embedding_model(model)
        self._dimensions = resolve_embedding_dimensions(dimensions)
        self.base_url = resolve_ollama_base_url(base_url)
        self.timeout = timeout
        self._http_client = http_client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed each text, in input order. Every failure raises; nothing is substituted."""
        if not texts:
            return ()

        url = f"{self.base_url}/api/embed"
        payload: dict[str, Any] = {"model": self._model, "input": list(texts)}
        client = self._http_client or httpx.AsyncClient()
        should_close = self._http_client is None
        try:
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise EmbeddingError(
                    f"Ollama embedding endpoint returned status {resp.status_code}: {resp.text}"
                )
            payload_body: object = resp.json()
        except httpx.RequestError as exc:
            raise EmbeddingError(
                f"Failed to reach the Ollama embedding endpoint at {url}: "
                f"{describe_transport_error(exc)}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingError(
                f"Invalid JSON from the Ollama embedding endpoint: {describe_transport_error(exc)}"
            ) from exc
        except asyncio.CancelledError:
            logger.debug("Ollama embed call cancelled by caller")
            raise
        finally:
            if should_close:
                await client.aclose()

        # An annotation is not a check: `resp.json()` is any JSON value, and a reverse
        # proxy or an OpenAI-shaped endpoint reached by mistake answers 200 with a list.
        # Calling `.get` on that raises `AttributeError`, which no caller of a declared
        # `EmbeddingError` contract catches.
        if not isinstance(payload_body, dict):
            raise EmbeddingError(
                f"Ollama embedding endpoint returned a JSON "
                f"{type(payload_body).__name__}, not an object. "
                f"Check that {url} is an Ollama embedding endpoint."
            )
        return self._parse_vectors(cast("dict[str, Any]", payload_body), expected_count=len(texts))

    def _parse_vectors(
        self, data: dict[str, Any], expected_count: int
    ) -> tuple[tuple[float, ...], ...]:
        """Read the response body as one vector per input, refusing anything else."""
        raw_value: object = data.get("embeddings")
        if not isinstance(raw_value, list):
            raise EmbeddingError(
                f"Ollama embedding response carried no 'embeddings' list "
                f"(keys: {sorted(data)}). Model '{self._model}' may not be an embedding model."
            )
        raw = cast("list[object]", raw_value)
        if len(raw) != expected_count:
            raise EmbeddingError(
                f"Asked Ollama for {expected_count} embeddings and got {len(raw)}. "
                f"Pairing them by position would attribute a vector to the wrong text."
            )

        vectors: list[tuple[float, ...]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, list):
                raise EmbeddingError(
                    f"Embedding {index} from Ollama is {type(item).__name__}, not a list."
                )
            components = cast("list[object]", item)
            vector = tuple(_as_float(component, index) for component in components)
            if len(vector) != self._dimensions:
                raise EmbeddingDimensionError(self._model, self._dimensions, len(vector))
            vectors.append(vector)
        return tuple(vectors)


def _as_float(value: object, index: int) -> float:
    """Read one vector component, refusing anything that is not a real number.

    `bool` is excluded even though it is an `int`: a boolean in an embedding is a decoding
    mistake, and silently reading it as 1.0 would bury the mistake inside a plausible vector.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EmbeddingError(
            f"Embedding {index} from Ollama holds a non-numeric component "
            f"of type {type(value).__name__}."
        )
    number = float(value)
    # `json.loads` accepts bare `NaN` and `Infinity`, so a corrupt body arrives here as a
    # float rather than as a decode error. Neither has a direction: a NaN component makes
    # every later score comparison False, which is how a corrupt vector wins a top_k it
    # cannot be compared into.
    if not math.isfinite(number):
        raise EmbeddingError(
            f"Embedding {index} from Ollama holds a non-finite component ({number}). "
            f"A vector that is not finite has no direction and cannot be ranked."
        )
    return number
