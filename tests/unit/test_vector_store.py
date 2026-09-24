"""The default vector store: exact cosine, a fixed storage form, and no zero vectors."""

from __future__ import annotations

import pytest

from uclone_x.errors import EmbeddingDimensionError, EmbeddingError
from uclone_x.memory.vector_store import (
    BruteForceVectorStore,
    deserialize_vector,
    serialize_vector,
)


def test_search_ranks_by_cosine_similarity() -> None:
    store = BruteForceVectorStore(dimensions=3, model_name="test")
    store.upsert("aligned", [1.0, 0.0, 0.0])
    store.upsert("diagonal", [1.0, 1.0, 0.0])
    store.upsert("orthogonal", [0.0, 0.0, 1.0])

    matches = store.search([1.0, 0.0, 0.0], top_k=2)

    assert [match.key for match in matches] == ["aligned", "diagonal"]
    assert matches[0].score == pytest.approx(1.0)
    assert matches[1].score == pytest.approx(0.7071, abs=1e-4)


def test_magnitude_does_not_change_the_ranking() -> None:
    """Cosine is direction, so a longer vector of the same direction ranks identically.

    Killed by: src/uclone_x/memory/vector_store.py :: return array(FLOAT32_TYPECODE, (component / magnitude for component in vector))
    Becomes: return array(FLOAT32_TYPECODE, vector)
    """
    store = BruteForceVectorStore(dimensions=2, model_name="test")
    store.upsert("short", [1.0, 0.0])
    store.upsert("long", [0.0, 100.0])

    matches = store.search([0.0, 1.0], top_k=2)

    assert matches[0].key == "long"
    assert matches[0].score == pytest.approx(1.0)


def test_a_zero_vector_is_refused() -> None:
    """A zero vector is the shape a swallowed embedding failure takes; it is not stored.

    Killed by: src/uclone_x/memory/vector_store.py :: if not math.isfinite(magnitude) or magnitude == 0.0:
    Becomes: if not math.isfinite(magnitude):
    """
    store = BruteForceVectorStore(dimensions=3, model_name="test")

    with pytest.raises(EmbeddingError, match="magnitude 0"):
        store.upsert("nothing", [0.0, 0.0, 0.0])
    assert len(store) == 0


def test_a_wrong_width_vector_is_refused() -> None:
    store = BruteForceVectorStore(dimensions=3, model_name="test")

    with pytest.raises(EmbeddingDimensionError):
        store.upsert("short", [1.0, 0.0])


def test_upsert_replaces_and_remove_reports_presence() -> None:
    store = BruteForceVectorStore(dimensions=2, model_name="test")
    store.upsert("key", [1.0, 0.0])
    store.upsert("key", [0.0, 1.0])

    assert len(store) == 1
    assert store.search([0.0, 1.0], top_k=1)[0].score == pytest.approx(1.0)
    assert store.remove("key") is True
    assert store.remove("key") is False


def test_equal_scores_keep_insertion_order() -> None:
    """Ties must not be resolved by key text — that is ranking by name.

    Killed by: src/uclone_x/memory/vector_store.py :: matches.sort(key=lambda match: -match.score)
    Becomes: matches.sort(key=lambda match: (-match.score, match.key))
    """
    store = BruteForceVectorStore(dimensions=2, model_name="test")
    store.upsert("zeta", [1.0, 0.0])
    store.upsert("alpha", [1.0, 0.0])

    assert [match.key for match in store.search([1.0, 0.0], top_k=2)] == ["zeta", "alpha"]


def test_top_k_of_zero_or_less_returns_nothing() -> None:
    store = BruteForceVectorStore(dimensions=2, model_name="test")
    store.upsert("key", [1.0, 0.0])

    assert store.search([1.0, 0.0], top_k=0) == ()


def test_the_storage_form_is_little_endian_float32() -> None:
    """The form is pinned so a later index swap is a search change, not a re-embedding.

    Killed by: src/uclone_x/memory/vector_store.py :: FLOAT32_TYPECODE = "f"
    Becomes: FLOAT32_TYPECODE = "d"
    """
    payload = serialize_vector([1.0, 0.0])

    assert len(payload) == 8
    assert payload[:4] == b"\x00\x00\x80?"
    assert deserialize_vector(payload) == (1.0, 0.0)


def test_a_serialized_store_reloads_to_the_same_ranking() -> None:
    store = BruteForceVectorStore(dimensions=2, model_name="test")
    store.upsert("a", [1.0, 0.0])
    store.upsert("b", [0.0, 1.0])
    payloads = store.serialize()

    reloaded = BruteForceVectorStore(dimensions=2, model_name="test")
    reloaded.load(payloads.items())

    assert reloaded.keys() == ("a", "b")
    assert store.search([1.0, 0.2], top_k=2) == reloaded.search([1.0, 0.2], top_k=2)


def test_a_store_needs_a_positive_width() -> None:
    with pytest.raises(EmbeddingError, match="positive width"):
        BruteForceVectorStore(dimensions=0, model_name="test")


def test_a_nan_component_is_refused_rather_than_indexed() -> None:
    """A NaN vector compares False against every score, so `sort` leaves it where it was.

    It then holds whatever top_k slot it was inserted into, with score NaN, for every query
    — a fact that matches everything. The zero-magnitude guard alone does not catch it: a
    NaN magnitude is not equal to 0.0.

    Killed by: src/uclone_x/memory/vector_store.py :: if not math.isfinite(magnitude) or magnitude == 0.0:
    Becomes: if magnitude == 0.0:
    """
    store = BruteForceVectorStore(dimensions=3, model_name="test")

    with pytest.raises(EmbeddingError, match="magnitude"):
        store.upsert("nan", [float("nan"), 1.0, 0.0])
    assert len(store) == 0


def test_a_magnitude_that_overflows_is_refused() -> None:
    """The other non-finite direction, and it lands on the value the guard exists to stop.

    Squaring 1e200 overflows to `inf`, and dividing a finite component by `inf` is exactly
    0.0 — so without the finiteness check this stores the all-zero vector.

    Killed by: src/uclone_x/memory/vector_store.py :: if not math.isfinite(magnitude) or magnitude == 0.0:
    Becomes: if magnitude == 0.0:
    """
    store = BruteForceVectorStore(dimensions=3, model_name="test")

    with pytest.raises(EmbeddingError, match="magnitude"):
        store.upsert("huge", [1e200, 1e200, 1e200])
    assert len(store) == 0


def test_a_query_vector_is_held_to_the_same_bar() -> None:
    store = BruteForceVectorStore(dimensions=3, model_name="test")
    store.upsert("one", [1.0, 0.0, 0.0])

    with pytest.raises(EmbeddingError, match="magnitude"):
        store.search([float("nan"), 0.0, 0.0], top_k=1)


def test_a_truncated_store_payload_says_so_in_this_modules_failure_type() -> None:
    """`array.frombytes` raises `ValueError`, which no caller of an embedding path catches.

    Killed by: src/uclone_x/memory/vector_store.py :: if len(payload) % FLOAT32_ITEMSIZE:
    Becomes: if False:
    """
    with pytest.raises(EmbeddingError, match="truncated"):
        deserialize_vector(b"\x00\x00\x00")
