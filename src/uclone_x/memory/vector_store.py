"""Vector storage extension point and its complete default (P0).

P0 requires a working default composition *and* a declared, consumed extension point. The
two halves are split here deliberately:

* `VectorStoreProtocol` is the seam. An installation that outgrows brute force swaps in an
  hnswlib- or pgvector-backed implementation without touching the memory subsystem.
* `BruteForceVectorStore` is the default, and it is pure Python on purpose. Core's declared
  dependencies are pydantic, pydantic-settings, pyyaml and httpx; making retrieval depend on
  numpy would make an optional accelerator a hard requirement of the default path. Exact
  cosine over a few thousand facts costs microseconds, which is the regime cross-session
  memory actually lives in.

The *storage form* is fixed even though the search layer is not: vectors are held as
little-endian float32, the width every mainstream vector index uses. That is what makes a
later swap a search-layer change rather than a re-embedding of the whole corpus — the bytes
on disk are already in the form the replacement wants.
"""

from __future__ import annotations

import math
import sys
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from uclone_x.errors import EmbeddingDimensionError, EmbeddingError

FLOAT32_TYPECODE = "f"
#: Bytes per component in the storage form, used to reject a truncated payload.
FLOAT32_ITEMSIZE = 4
"""Type code of the fixed storage form: IEEE-754 binary32, stored little-endian."""


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Render a vector in the fixed storage form: little-endian float32.

    Byte order is pinned rather than left native so that a store written on one machine
    reads correctly on another. On a little-endian host this is a no-op.
    """
    buffer = array(FLOAT32_TYPECODE, vector)
    if sys.byteorder != "little":
        buffer.byteswap()
    return buffer.tobytes()


def deserialize_vector(payload: bytes) -> tuple[float, ...]:
    """Read the fixed storage form back into a vector.

    Raises:
        EmbeddingError: If `payload` is not a whole number of float32 components. A
            truncated store file is a failure of this module, and it says so in this
            module's own failure type rather than in `array`'s `ValueError`.
    """
    if len(payload) % FLOAT32_ITEMSIZE:
        raise EmbeddingError(
            f"A stored vector is {len(payload)} bytes, which is not a whole number of "
            f"{FLOAT32_ITEMSIZE}-byte float32 components. The store file is truncated."
        )
    buffer = array(FLOAT32_TYPECODE)
    buffer.frombytes(payload)
    if sys.byteorder != "little":
        buffer.byteswap()
    return tuple(buffer)


@dataclass(frozen=True)
class VectorMatch:
    """One search hit: the key that was stored, and its cosine similarity to the query.

    `score` is reported rather than left implicit because a ranking with no scores cannot
    be thresholded, and a caller that cannot threshold has no way to tell "the best of a
    bad set" from "a good match" (P6).
    """

    key: str
    score: float


class VectorStoreProtocol(Protocol):
    """Protocol for vector storage and nearest-neighbour search."""

    @property
    def dimensions(self) -> int:
        """Width every stored vector must have."""
        ...

    @property
    def model_name(self) -> str:
        """Embedding model whose vectors this store holds.

        Vectors from different models are not comparable, so a store records which model
        wrote it (P6). Mixing two models silently produces a ranking that looks ordinary
        and means nothing.
        """
        ...

    def upsert(self, key: str, vector: Sequence[float]) -> None:
        """Store `vector` under `key`, replacing any vector already stored there.

        Raises:
            EmbeddingDimensionError: If the vector's width is not `dimensions`.
            EmbeddingError: If the vector has zero magnitude, which has no direction and
                so scores 0.0 against every query.
        """
        ...

    def keys(self) -> tuple[str, ...]:
        """Stored keys. Declared on the protocol so a caller can tell which items are
        already indexed without re-embedding them to find out."""
        ...

    def remove(self, key: str) -> bool:
        """Remove `key`, reporting whether it was present."""
        ...

    def search(self, vector: Sequence[float], top_k: int) -> tuple[VectorMatch, ...]:
        """Return up to `top_k` stored keys, by descending cosine similarity.

        Raises:
            EmbeddingDimensionError: If the query's width is not `dimensions`.
            EmbeddingError: If the query has zero magnitude.
        """
        ...

    def __len__(self) -> int:
        """Number of vectors held."""
        ...


class BruteForceVectorStore:
    """Exact cosine search over every stored vector, in pure Python.

    Vectors are normalized on write, so search is a dot product and nothing is recomputed
    per query. The normalized form is the one stored and returned: this store is an index,
    not the system of record for the vectors themselves.
    """

    def __init__(self, dimensions: int, model_name: str) -> None:
        if dimensions <= 0:
            raise EmbeddingError(f"A vector store needs a positive width, got {dimensions}.")
        self._dimensions = dimensions
        self._model_name = model_name
        self._vectors: dict[str, array[float]] = {}

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def __len__(self) -> int:
        return len(self._vectors)

    def keys(self) -> tuple[str, ...]:
        """Stored keys, in insertion order."""
        return tuple(self._vectors)

    def get(self, key: str) -> tuple[float, ...] | None:
        """Return the normalized vector stored under `key`, or None if absent."""
        stored = self._vectors.get(key)
        return None if stored is None else tuple(stored)

    def upsert(self, key: str, vector: Sequence[float]) -> None:
        self._vectors[key] = self._normalize(vector)

    def remove(self, key: str) -> bool:
        return self._vectors.pop(key, None) is not None

    def clear(self) -> None:
        """Drop every stored vector."""
        self._vectors.clear()

    def search(self, vector: Sequence[float], top_k: int) -> tuple[VectorMatch, ...]:
        query = self._normalize(vector)
        if top_k <= 0 or not self._vectors:
            return ()
        matches = [
            VectorMatch(key=key, score=_dot(query, stored)) for key, stored in self._vectors.items()
        ]
        # Ties break on insertion order, not on key text: `sorted` is stable and the dict
        # preserves insertion order, so equally similar facts stay in the order they were
        # recorded. Sorting by `(-score, key)` would silently rank by name.
        matches.sort(key=lambda match: -match.score)
        return tuple(matches[:top_k])

    def serialize(self) -> dict[str, bytes]:
        """Render every stored vector in the fixed storage form, for persistence."""
        return {key: serialize_vector(stored) for key, stored in self._vectors.items()}

    def load(self, payloads: Iterable[tuple[str, bytes]]) -> None:
        """Replace the store's contents with vectors in the fixed storage form."""
        loaded: dict[str, array[float]] = {}
        for key, payload in payloads:
            loaded[key] = self._normalize(deserialize_vector(payload))
        self._vectors = loaded

    def _normalize(self, vector: Sequence[float]) -> array[float]:
        if len(vector) != self._dimensions:
            raise EmbeddingDimensionError(self._model_name, self._dimensions, len(vector))
        magnitude = math.sqrt(sum(component * component for component in vector))
        # `isfinite`, not `!= 0.0`. A NaN component gives a NaN magnitude, which is not
        # zero, and dividing by it stores a NaN vector whose every comparison is False --
        # `sort` then puts it wherever it was inserted and it wins top_k with score NaN.
        # A component large enough to overflow the sum of squares gives an infinite
        # magnitude, and dividing by that stores exactly the all-zero vector this guard
        # exists to refuse. Both are failed embeddings wearing the shape of a value.
        if not math.isfinite(magnitude) or magnitude == 0.0:
            raise EmbeddingError(
                f"A vector with magnitude {magnitude} was offered to the vector store. "
                "Only a finite, non-zero magnitude has a direction; a zero one scores 0.0 "
                "against every query and a NaN one compares False against every score, so "
                "either would be indexed as a fact that matches nothing -- or everything. "
                "This is the shape a failed embedding call takes when its failure is "
                "swallowed; it is refused here rather than stored."
            )
        # Round-tripped through float32 on the way in, so a vector read back from disk and
        # one held in memory compare bit-for-bit rather than drifting in the last places.
        return array(FLOAT32_TYPECODE, (component / magnitude for component in vector))


def _dot(left: array[float], right: array[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))
