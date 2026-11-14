"""Cluster-shared provider model catalog cache backed by `CacheClientProtocol`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from uclone_x.core.cluster_protocols import CacheClientProtocol

__all__ = [
    "ValkeyCatalogCache",
]


class ValkeyCatalogCache:
    """Provider model catalog shared by worker pods, with a refresh stampede lock.

    Both keys are cache keys with a TTL: losing either to eviction costs one extra upstream
    `/models` request, never correctness.
    """

    def __init__(self, cache: CacheClientProtocol, default_ttl_seconds: int = 600) -> None:
        if default_ttl_seconds < 1:
            raise ValueError(f"default_ttl_seconds must be >= 1, got {default_ttl_seconds}")
        self._cache = cache
        self._default_ttl_seconds = default_ttl_seconds

    @staticmethod
    def _catalog_key(provider: str) -> bytes:
        clean = provider.strip().lower()
        if not clean:
            raise ValueError("provider must not be blank")
        return f"catalog:models:{{{clean}}}".encode()

    @staticmethod
    def _lock_key(provider: str) -> bytes:
        clean = provider.strip().lower()
        if not clean:
            raise ValueError("provider must not be blank")
        return f"lock:catalog:{{{clean}}}".encode()

    async def get_models(self, provider: str) -> tuple[Mapping[str, Any], ...] | None:
        """Read cached model metadata entries for `provider`, or `None` on a cache miss."""
        raw = await self._cache.get(self._catalog_key(provider))
        if raw is None:
            return None
        try:
            decoded: Any = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if not isinstance(decoded, list):
            return None
        items: list[Mapping[str, Any]] = []
        for entry in cast(list[Any], decoded):
            if isinstance(entry, dict):
                items.append(dict(cast(dict[str, Any], entry)))
        return tuple(items)

    async def set_models(
        self,
        provider: str,
        models: Sequence[Mapping[str, Any]],
        ttl_seconds: int | None = None,
    ) -> None:
        """Store serialized model metadata entries for `provider` with a TTL."""
        exptime = self._default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if exptime < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {exptime}")
        payload = json.dumps(
            [dict(entry) for entry in models],
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        await self._cache.set_cache(self._catalog_key(provider), payload, ttl_seconds=exptime)

    async def acquire_refresh_lock(self, provider: str, ttl_seconds: int = 15) -> bool:
        """Acquire a short-lived stampede lock before refreshing `provider` upstream."""
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {ttl_seconds}")
        return await self._cache.add_cache(self._lock_key(provider), b"1", ttl_seconds=ttl_seconds)

    async def release_refresh_lock(self, provider: str) -> bool:
        """Release the refresh lock for `provider`."""
        return await self._cache.delete(self._lock_key(provider)) > 0

    async def invalidate(self, provider: str) -> bool:
        """Delete cached model catalog entries for `provider`."""
        return await self._cache.delete(self._catalog_key(provider)) > 0
