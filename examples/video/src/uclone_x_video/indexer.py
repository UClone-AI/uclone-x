"""Frame-level and shot-level multimodal semantic indexing and retrieval (`uclone_x_video.indexer`).

Follows P5 and P6:
* Uses `EmbedderProtocol` and `BruteForceVectorStore` (`serialize_vector` / `deserialize_vector`
  in little-endian float32) when an embedder is wired.
* Reports `method` explicitly on every search response (`embedding similarity (<model>)` or
  `lexical overlap (no embedder configured)`), including on empty results.
* Never substitutes a zero vector or silently degrades to lexical ranking when an embedder
  call fails (`EmbeddingError` / `EmbeddingDimensionError` propagates).
* Clusters matching frames by shot boundary (`shot_in_frame..shot_out_frame`) so every hit
  returns an editable frame span ready for `video_timeline` or `bake_derived_asset`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import EmbeddingDimensionError, EmbeddingError
from uclone_x.memory.vector_store import (
    BruteForceVectorStore,
    deserialize_vector,
    serialize_vector,
)
from uclone_x_video.library import VideoError
from uclone_x_video.schemas import FrameIndexEntry, MediaAsset, frames_to_timecode
from uclone_x_video.work import INDEX_DIR, VideoWork

if TYPE_CHECKING:
    from uclone_x.llm.protocols import EmbedderProtocol

__all__ = [
    "VECTOR_MANIFEST_FILE",
    "FrameRangeHit",
    "FrameSearchRanking",
    "FrameSeedSpec",
    "default_seed_frames_for_asset",
    "index_asset_frames",
    "index_asset_frames_sync",
    "search_project_frames",
]

VECTOR_MANIFEST_FILE = f"{INDEX_DIR}/vectors_meta.json"

_TOKEN_RE = re.compile(r"[a-z0-9\uac00-\ud7a3_]+")


class FrameSeedSpec(BaseModel):
    """Input seed describing a shot or sampled frame when indexing a media asset."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    frame_no: int = Field(ge=0)
    shot_in_frame: int = Field(ge=0)
    shot_out_frame: int = Field(ge=1)
    shot_type: Literal["wide", "medium", "close_up", "extreme_close_up", "aerial", "insert"] = (
        "medium"
    )
    camera_motion: Literal["static", "pan", "tilt", "tracking", "handheld", "zoom"] = "static"
    caption: str = Field(min_length=1)
    visual_tags: list[str] = Field(default_factory=list)
    transcript_slice: str = ""


class FrameRangeHit(BaseModel):
    """One ranked frame/shot segment returned by `search_project_frames`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    frame_id: str
    asset_id: str
    shot_id: str
    peak_frame: int
    in_frame: int
    out_frame: int
    in_timecode: str
    out_timecode: str
    duration_frames: int
    shot_type: str
    camera_motion: str
    score: float
    caption: str
    visual_tags: list[str]
    transcript_slice: str = ""


class FrameSearchRanking(BaseModel):
    """Ranked frame search results with explicit retrieval method attribution (P6)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    query: str
    method: str
    total_indexed_frames: int
    hits: list[FrameRangeHit]


def _frame_search_text(entry: FrameIndexEntry) -> str:
    parts = [entry.caption, " ".join(entry.visual_tags), entry.shot_type, entry.camera_motion]
    if entry.transcript_slice:
        parts.append(entry.transcript_slice)
    return " ".join(p for p in parts if p).strip()


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _lexical_score(query: str, entry: FrameIndexEntry) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    doc_text = _frame_search_text(entry).casefold()
    doc_tokens = _tokenize(doc_text)
    if not doc_tokens:
        return 0.0
    exact_matches = len(q_tokens & doc_tokens)
    substring_matches = sum(
        0.6 for qt in q_tokens if qt not in doc_tokens and len(qt) >= 2 and qt in doc_text
    )
    raw = (exact_matches + substring_matches) / len(q_tokens)
    return round(min(1.0, raw), 4)


def default_seed_frames_for_asset(asset: MediaAsset) -> list[FrameSeedSpec]:
    """Generate deterministic multi-shot frame seeds covering `asset.total_frames`."""
    total = asset.total_frames
    fps = asset.fps
    shot_len = max(1, min(150, max(fps, total // 3)))
    seeds: list[FrameSeedSpec] = []
    shot_templates: tuple[
        tuple[
            Literal["wide", "medium", "close_up", "extreme_close_up", "aerial", "insert"],
            Literal["static", "pan", "tilt", "tracking", "handheld", "zoom"],
            str,
            list[str],
        ],
        ...,
    ] = (
        (
            "wide",
            "pan",
            f"Establishing wide shot in {asset.filename} showing the ambient setting and lighting",
            ["establishing", "wide", "ambient", "opening"],
        ),
        (
            "close_up",
            "static",
            f"Close-up highlight moment in {asset.filename} with warm expression and key action",
            ["close_up", "highlight", "expression", "focus", "action"],
        ),
        (
            "medium",
            "tracking",
            f"Dynamic medium tracking shot in {asset.filename} transitioning toward the closing beat",
            ["medium", "tracking", "transition", "closing", "broll"],
        ),
    )

    cursor = 0
    shot_idx = 0
    while cursor < total:
        out_f = min(total, cursor + shot_len)
        mid_f = cursor + (out_f - cursor) // 2
        shot_type, motion, desc, tags = shot_templates[shot_idx % len(shot_templates)]
        seeds.append(
            FrameSeedSpec(
                frame_no=mid_f,
                shot_in_frame=cursor,
                shot_out_frame=out_f,
                shot_type=shot_type,
                camera_motion=motion,
                caption=desc,
                visual_tags=list(tags),
                transcript_slice="",
            )
        )
        cursor = out_f
        shot_idx += 1
    return seeds


def _save_vector_index(
    work: VideoWork,
    *,
    model_name: str,
    dimensions: int,
    vectors_by_frame_id: dict[str, tuple[float, ...]],
) -> None:
    """Persist frame embeddings in little-endian float32 (`index/vectors.bin` + `vectors_meta.json`)."""
    store = BruteForceVectorStore(dimensions=dimensions, model_name=model_name)
    ordered_keys: list[str] = sorted(vectors_by_frame_id)
    chunks: list[bytes] = []
    for key in ordered_keys:
        vec = vectors_by_frame_id[key]
        store.upsert(key, vec)
        chunks.append(serialize_vector(vec))

    bin_path = work.project_dir / INDEX_DIR / "vectors.bin"
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.write_bytes(b"".join(chunks))

    meta_payload = {
        "model_name": model_name,
        "dimensions": dimensions,
        "frame_ids": ordered_keys,
    }
    work.library.write_file(
        work.project_id,
        VECTOR_MANIFEST_FILE,
        json.dumps(meta_payload, sort_keys=True, indent=2) + "\n",
        room_id=work.room_id,
        expected_digest=(
            existing.digest
            if (existing := work.library.read_file(work.project_id, VECTOR_MANIFEST_FILE))
            is not None
            else None
        ),
    )


def _load_vector_store(
    work: VideoWork, *, expected_model: str, expected_dimensions: int
) -> BruteForceVectorStore | None:
    """Load `index/vectors.bin` into a `BruteForceVectorStore` if present and matching the embedder."""
    meta_file = work.library.read_file(work.project_id, VECTOR_MANIFEST_FILE)
    bin_path = work.project_dir / INDEX_DIR / "vectors.bin"
    if meta_file is None or not bin_path.is_file():
        return None
    try:
        loaded_meta: Any = json.loads(meta_file.text)
    except json.JSONDecodeError as exc:
        raise EmbeddingError(f"'{VECTOR_MANIFEST_FILE}' is not valid JSON.") from exc
    if not isinstance(loaded_meta, dict):
        raise EmbeddingError(f"'{VECTOR_MANIFEST_FILE}' must be a JSON object.")
    meta = cast(dict[str, Any], loaded_meta)
    stored_model = meta.get("model_name")
    stored_dims = meta.get("dimensions")
    raw_frame_ids = meta.get("frame_ids")
    if (
        not isinstance(stored_model, str)
        or not isinstance(stored_dims, int)
        or not isinstance(raw_frame_ids, list)
    ):
        raise EmbeddingError(
            f"'{VECTOR_MANIFEST_FILE}' is missing model_name, dimensions, or frame_ids."
        )
    frame_ids = [str(fid) for fid in cast(list[Any], raw_frame_ids) if isinstance(fid, str)]
    if stored_model != expected_model or stored_dims != expected_dimensions:
        return None

    raw_bytes = bin_path.read_bytes()
    all_floats = deserialize_vector(raw_bytes)
    expected_floats = len(frame_ids) * stored_dims
    if len(all_floats) != expected_floats:
        raise EmbeddingError(
            f"'index/vectors.bin' has {len(all_floats)} floats, expected {expected_floats} "
            f"for {len(frame_ids)} frames at width {stored_dims}."
        )
    store = BruteForceVectorStore(dimensions=stored_dims, model_name=stored_model)
    for idx, fid in enumerate(frame_ids):
        offset = idx * stored_dims
        store.upsert(fid, all_floats[offset : offset + stored_dims])
    return store


def index_asset_frames_sync(
    work: VideoWork,
    *,
    asset_id: str,
    seeds: Sequence[FrameSeedSpec] | None = None,
) -> tuple[list[FrameIndexEntry], list[FrameIndexEntry]]:
    """Synchronously index frame metadata for `asset_id` into `index/frames.jsonl`."""
    work.library.require_writable(work.project_id, work.room_id)
    loaded = work.read_asset(asset_id)
    if loaded is None:
        raise VideoError(f"Media asset {asset_id!r} was not found.")
    asset, asset_digest = loaded

    chosen_seeds = list(seeds) if seeds else default_seed_frames_for_asset(asset)
    new_entries: list[FrameIndexEntry] = []
    for idx, seed in enumerate(chosen_seeds, start=1):
        if seed.shot_out_frame > asset.total_frames:
            raise VideoError(
                f"Frame seed shot_out_frame ({seed.shot_out_frame}) exceeds asset "
                f"{asset_id!r} total_frames ({asset.total_frames})."
            )
        pts_ms = round((seed.frame_no / asset.fps) * 1000)
        tc = frames_to_timecode(seed.frame_no, asset.fps)
        entry = FrameIndexEntry(
            frame_id=f"{asset_id}_f{seed.frame_no:06d}",
            asset_id=asset_id,
            frame_no=seed.frame_no,
            pts_ms=pts_ms,
            timecode=tc,
            shot_id=f"shot_{idx:02d}",
            shot_in_frame=seed.shot_in_frame,
            shot_out_frame=seed.shot_out_frame,
            shot_type=seed.shot_type,
            camera_motion=seed.camera_motion,
            caption=seed.caption,
            visual_tags=list(seed.visual_tags),
            transcript_slice=seed.transcript_slice,
        )
        new_entries.append(entry)

    existing_all = [e for e in work.read_frame_index() if e.asset_id != asset_id]
    combined = sorted(
        [*existing_all, *new_entries],
        key=lambda e: (e.asset_id, e.frame_no),
    )
    work.write_frame_index(combined)
    updated_asset = asset.model_copy(
        update={
            "index_status": "indexed",
            "indexed_frame_count": len(new_entries),
        }
    )
    work.write_asset(updated_asset, expected_digest=asset_digest)
    return new_entries, combined


async def index_asset_frames(
    work: VideoWork,
    *,
    asset_id: str,
    seeds: Sequence[FrameSeedSpec] | None = None,
    embedder: EmbedderProtocol | None = None,
) -> list[FrameIndexEntry]:
    """Index frames/shots for `asset_id` into `index/frames.jsonl` (and `index/vectors.bin` if `embedder` is given)."""
    new_entries, combined = index_asset_frames_sync(work, asset_id=asset_id, seeds=seeds)

    if embedder is not None and combined:
        texts = [_frame_search_text(e) for e in combined]
        vectors = await embedder.embed(texts)
        if len(vectors) != len(combined):
            raise EmbeddingError(
                f"Embedder {embedder.model_name!r} returned {len(vectors)} vectors for "
                f"{len(combined)} frames."
            )
        for vec in vectors:
            if len(vec) != embedder.dimensions:
                raise EmbeddingDimensionError(embedder.model_name, embedder.dimensions, len(vec))
        by_id = {entry.frame_id: vec for entry, vec in zip(combined, vectors, strict=True)}
        _save_vector_index(
            work,
            model_name=embedder.model_name,
            dimensions=embedder.dimensions,
            vectors_by_frame_id=by_id,
        )

    return new_entries


def _entry_to_hit(entry: FrameIndexEntry, score: float, fps: int) -> FrameRangeHit:
    return FrameRangeHit(
        frame_id=entry.frame_id,
        asset_id=entry.asset_id,
        shot_id=entry.shot_id,
        peak_frame=entry.frame_no,
        in_frame=entry.shot_in_frame,
        out_frame=entry.shot_out_frame,
        in_timecode=frames_to_timecode(entry.shot_in_frame, fps),
        out_timecode=frames_to_timecode(entry.shot_out_frame, fps),
        duration_frames=entry.shot_out_frame - entry.shot_in_frame,
        shot_type=entry.shot_type,
        camera_motion=entry.camera_motion,
        score=round(score, 4),
        caption=entry.caption,
        visual_tags=list(entry.visual_tags),
        transcript_slice=entry.transcript_slice,
    )


def _deduplicate_shot_hits(hits: Sequence[FrameRangeHit], top_k: int) -> list[FrameRangeHit]:
    """Merge hits landing in the same `(asset_id, shot_id)` keeping the highest-scoring peak frame."""
    best_by_shot: dict[tuple[str, str], FrameRangeHit] = {}
    for hit in hits:
        key = (hit.asset_id, hit.shot_id)
        existing = best_by_shot.get(key)
        if existing is None or hit.score > existing.score:
            best_by_shot[key] = hit
    ordered = sorted(
        best_by_shot.values(),
        key=lambda h: (-h.score, h.asset_id, h.in_frame),
    )
    return ordered[: max(1, top_k)]


async def search_project_frames(
    work: VideoWork,
    *,
    query: str,
    asset_id: str | None = None,
    shot_type: str | None = None,
    top_k: int = 5,
    embedder: EmbedderProtocol | None = None,
) -> FrameSearchRanking:
    """Search indexed video frames by semantic similarity (when `embedder` is wired) or lexical overlap.

    Always labels `method` explicitly and never falls back to lexical or zero vectors if an
    embedder is wired and fails (P6).
    """
    clean_query = query.strip()
    if not clean_query:
        raise VideoError("Frame search query must not be empty.")

    all_entries = work.read_frame_index()
    filtered = [
        e
        for e in all_entries
        if (asset_id is None or e.asset_id == asset_id)
        and (shot_type is None or e.shot_type == shot_type)
    ]
    assets_fps = {a.asset_id: a.fps for a, _ in work.list_assets()}
    default_fps = work.record().fps

    if embedder is not None:
        method = f"embedding similarity ({embedder.model_name})"
        if not filtered:
            return FrameSearchRanking(
                query=clean_query,
                method=method,
                total_indexed_frames=0,
                hits=[],
            )
        query_vecs = await embedder.embed([clean_query])
        if len(query_vecs) != 1:
            raise EmbeddingError(
                f"Embedder {embedder.model_name!r} returned {len(query_vecs)} vectors for 1 query."
            )
        query_vec = query_vecs[0]
        if len(query_vec) != embedder.dimensions:
            raise EmbeddingDimensionError(embedder.model_name, embedder.dimensions, len(query_vec))

        store = _load_vector_store(
            work,
            expected_model=embedder.model_name,
            expected_dimensions=embedder.dimensions,
        )
        entries_by_id = {e.frame_id: e for e in filtered}
        if store is None or not set(entries_by_id).issubset(set(store.keys())):
            texts = [_frame_search_text(e) for e in filtered]
            doc_vecs = await embedder.embed(texts)
            if len(doc_vecs) != len(filtered):
                raise EmbeddingError(
                    f"Embedder {embedder.model_name!r} returned {len(doc_vecs)} vectors for "
                    f"{len(filtered)} frames."
                )
            store = BruteForceVectorStore(
                dimensions=embedder.dimensions, model_name=embedder.model_name
            )
            for entry, vec in zip(filtered, doc_vecs, strict=True):
                store.upsert(entry.frame_id, vec)

        matches = store.search(query_vec, top_k=len(filtered))
        raw_hits: list[FrameRangeHit] = []
        for match in matches:
            matched_entry = entries_by_id.get(match.key)
            if matched_entry is not None:
                fps = assets_fps.get(matched_entry.asset_id, default_fps)
                raw_hits.append(_entry_to_hit(matched_entry, match.score, fps))
        return FrameSearchRanking(
            query=clean_query,
            method=method,
            total_indexed_frames=len(filtered),
            hits=_deduplicate_shot_hits(raw_hits, top_k),
        )

    method = "lexical overlap (no embedder configured)"
    if not filtered:
        return FrameSearchRanking(
            query=clean_query,
            method=method,
            total_indexed_frames=0,
            hits=[],
        )

    scored: list[FrameRangeHit] = []
    for entry in filtered:
        score = _lexical_score(clean_query, entry)
        if score > 0.0:
            fps = assets_fps.get(entry.asset_id, default_fps)
            scored.append(_entry_to_hit(entry, score, fps))

    return FrameSearchRanking(
        query=clean_query,
        method=method,
        total_indexed_frames=len(filtered),
        hits=_deduplicate_shot_hits(scored, top_k),
    )
