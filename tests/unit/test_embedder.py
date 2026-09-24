"""The embedder's contract: every failure raises, and no vector is ever substituted."""

from __future__ import annotations

import httpx
import pytest

from uclone_x.errors import EmbeddingDimensionError, EmbeddingError
from uclone_x.llm.connectors.ollama_embedder import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS_ENV_VAR,
    OllamaEmbedder,
    resolve_embedding_dimensions,
    resolve_embedding_model,
)


def _client(handler: object) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)  # pyright: ignore[reportArgumentType]
    return httpx.AsyncClient(transport=transport)


@pytest.mark.asyncio
async def test_vectors_come_back_in_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=_client(handler))

    assert await embedder.embed(["first", "second"]) == ((1.0, 0.0), (0.0, 1.0))


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_raises_instead_of_returning_a_zero_vector() -> None:
    """The defect this whole seam exists to prevent: a transport error read as "no match".

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: except httpx.RequestError as exc:
    Becomes: except NotImplementedError as exc:
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=_client(handler))

    with pytest.raises(EmbeddingError, match="Failed to reach"):
        await embedder.embed(["anything"])


@pytest.mark.asyncio
async def test_a_non_200_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    embedder = OllamaEmbedder(model="missing", dimensions=2, http_client=_client(handler))

    with pytest.raises(EmbeddingError, match="status 404"):
        await embedder.embed(["anything"])


@pytest.mark.asyncio
async def test_a_wrong_width_vector_is_refused_rather_than_padded() -> None:
    """A store indexes on the declared width, so a short vector is a defect, not a result.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: raise EmbeddingDimensionError(self._model, self._dimensions, len(vector))
    Becomes: vector = vector + (0.0,) * (self._dimensions - len(vector))
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    embedder = OllamaEmbedder(model="test", dimensions=4, http_client=_client(handler))

    with pytest.raises(EmbeddingDimensionError) as excinfo:
        await embedder.embed(["anything"])
    assert excinfo.value.expected == 4
    assert excinfo.value.actual == 1


@pytest.mark.asyncio
async def test_a_short_batch_is_refused_rather_than_paired_by_position() -> None:
    """Fewer vectors than inputs would attach each vector to the wrong text.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: if len(raw) != expected_count:
    Becomes: if False:
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=_client(handler))

    with pytest.raises(EmbeddingError, match="Asked Ollama for 2 embeddings and got 1"):
        await embedder.embed(["first", "second"])


@pytest.mark.asyncio
async def test_a_response_with_no_embeddings_key_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "not an embedding model"})

    embedder = OllamaEmbedder(model="qwen3:8b", dimensions=2, http_client=_client(handler))

    with pytest.raises(EmbeddingError, match="no 'embeddings' list"):
        await embedder.embed(["anything"])


@pytest.mark.asyncio
async def test_a_boolean_component_is_not_read_as_a_number() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[True, 0.5]]})

    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=_client(handler))

    with pytest.raises(EmbeddingError, match="non-numeric component"):
        await embedder.embed(["anything"])


@pytest.mark.asyncio
async def test_no_texts_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("embedding an empty batch must not reach the endpoint")

    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=_client(handler))

    assert await embedder.embed([]) == ()


def test_the_default_model_is_the_multilingual_one() -> None:
    assert resolve_embedding_model(None) == DEFAULT_EMBEDDING_MODEL
    assert resolve_embedding_dimensions(None) == DEFAULT_EMBEDDING_DIMENSIONS
    assert resolve_embedding_model("  custom-model  ") == "custom-model"


def test_a_non_numeric_configured_width_raises_rather_than_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A substituted width would let one store hold vectors of two widths.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: parsed = int(configured.strip())
    Becomes: parsed = DEFAULT_EMBEDDING_DIMENSIONS
    """
    monkeypatch.setenv(EMBEDDING_DIMENSIONS_ENV_VAR, "wide")

    with pytest.raises(EmbeddingError, match="must be an integer"):
        resolve_embedding_dimensions(None)


@pytest.mark.asyncio
async def test_a_caller_owned_client_survives_the_call() -> None:
    """Closing a client the embedder did not create breaks whatever else shares it.

    An embedder wired into an agent is handed the host's pooled client. Closing it after one
    embed leaves every later request on that client raising `RuntimeError` from httpx, far
    from here — and the second embed through this embedder fails the same way.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: should_close = self._http_client is None
    Becomes: should_close = True
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    client = _client(handler)
    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=client)

    await embedder.embed(["one"])

    assert client.is_closed is False
    assert await embedder.embed(["two"]) == ((1.0, 0.0),)


@pytest.mark.asyncio
async def test_a_client_the_embedder_created_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other branch: an embedder with no client given leaks a connection pool per call.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: await client.aclose()
    Becomes: pass
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    created: list[httpx.AsyncClient] = []

    real_client_cls = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        client = real_client_cls(transport=httpx.MockTransport(handler))
        created.append(client)
        return client

    monkeypatch.setattr("uclone_x.llm.connectors.ollama_embedder.httpx.AsyncClient", _factory)
    embedder = OllamaEmbedder(model="test", dimensions=2)

    assert await embedder.embed(["one"]) == ((1.0, 0.0),)

    assert len(created) == 1
    assert created[0].is_closed is True


@pytest.mark.asyncio
async def test_a_nan_component_in_the_response_is_refused() -> None:
    """`json.loads` accepts bare `NaN`, so a corrupt body arrives as a float, not an error.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: if not math.isfinite(number):
    Becomes: if False:
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"embeddings": [[NaN, 0.5]]}')

    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=_client(handler))

    with pytest.raises(EmbeddingError, match="finite"):
        await embedder.embed(["one"])


@pytest.mark.asyncio
async def test_a_json_body_that_is_not_an_object_is_refused() -> None:
    """An OpenAI-shaped endpoint reached by mistake answers 200 with a list.

    `.get` on that raises `AttributeError`, which no caller of a declared `EmbeddingError`
    contract catches — the failure arrives as the wrong type, far from here.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: if not isinstance(payload_body, dict):
    Becomes: if False:
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[1.0, 0.0]])

    embedder = OllamaEmbedder(model="test", dimensions=2, http_client=_client(handler))

    with pytest.raises(EmbeddingError, match="not an object"):
        await embedder.embed(["one"])


def test_a_boolean_width_is_not_read_as_one_dimension() -> None:
    """`True` is an `int` and `True <= 0` is False, so it would pass as a width of 1.

    Killed by: src/uclone_x/llm/connectors/ollama_embedder.py :: if isinstance(dimensions, bool):
    Becomes: if False:
    """
    with pytest.raises(EmbeddingError, match="must be an integer"):
        resolve_embedding_dimensions(True)  # pyright: ignore[reportArgumentType]
