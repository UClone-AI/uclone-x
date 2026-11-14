"""`CacheClientProtocol` over a Valkey cluster, through the `valkey` client (valkey-py, MIT).

This module is a thin adapter: cluster routing, MOVED/ASK redirects, failover discovery,
connection pooling and EVALSHA/NOSCRIPT handling are the library's. It adds only the rule the
protocol states — cache writes always carry a TTL, coordination writes never do — and one
client per event loop (`LoopLocal`).

`valkey` is imported lazily, so `uclone_x` imports without the `cluster` extra installed.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

from uclone_x.core.cluster_protocols import CacheClientProtocol
from uclone_x.core.loop_local import LoopLocal

__all__ = ["ValkeyClusterCache"]


def _split_host_port(endpoint: str) -> tuple[str, int]:
    host, sep, port = endpoint.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"Valkey endpoint must be host:port, got {endpoint!r}")
    return host, int(port)


class ValkeyClusterCache(CacheClientProtocol):
    """Async Valkey cluster client implementing `CacheClientProtocol`."""

    def __init__(
        self,
        *,
        nodes: Sequence[str],
        password: str,
        address_map: Mapping[str, str] | None = None,
        socket_timeout_seconds: float = 5.0,
    ) -> None:
        if not nodes:
            raise ValueError("ValkeyClusterCache needs at least one startup node")
        if not password:
            raise ValueError("ValkeyClusterCache needs the cluster password")
        self._nodes = tuple(_split_host_port(n) for n in nodes)
        self._password = password
        # Nodes announce their compose hostnames (`valkey-1:6379`); a client outside the
        # compose network reaches the same node on a published port.
        self._address_map = {
            _split_host_port(k): _split_host_port(v) for k, v in (address_map or {}).items()
        }
        self._socket_timeout = socket_timeout_seconds
        self._scripts: dict[str, Any] = {}
        self._clients: LoopLocal[Any] = LoopLocal(self._connect, self._disconnect)

    def _remap(self, address: tuple[str, int]) -> tuple[str, int]:
        return self._address_map.get(address, address)

    async def _connect(self) -> Any:
        cluster_mod: Any = importlib.import_module("valkey.asyncio.cluster")
        client: Any = cluster_mod.ValkeyCluster(
            startup_nodes=[cluster_mod.ClusterNode(host, port) for host, port in self._nodes],
            password=self._password,
            address_remap=self._remap,
            socket_timeout=self._socket_timeout,
            socket_connect_timeout=self._socket_timeout,
        )
        await client.initialize()
        return client

    @staticmethod
    async def _disconnect(client: Any) -> None:
        await client.aclose()

    async def raw_client(self) -> Any:
        """The library client for this event loop (for operational probes and tests)."""
        return await self._clients.get()

    async def get(self, key: bytes) -> bytes | None:
        client = await self._clients.get()
        value: bytes | None = await client.get(key)
        return value

    async def multi_get(self, *keys: bytes) -> tuple[bytes | None, ...]:
        """Fetch `keys` in order; keys of one session share a hash tag, so this is one MGET."""
        if not keys:
            return ()
        client = await self._clients.get()
        values: list[bytes | None] = await client.mget_nonatomic(list(keys))
        return tuple(values)

    @staticmethod
    def _require_ttl(ttl_seconds: int) -> int:
        if ttl_seconds < 1:
            raise ValueError(
                f"cache keys must carry a TTL >= 1 second so volatile-lru may evict them, "
                f"got {ttl_seconds}"
            )
        return ttl_seconds

    async def set_cache(self, key: bytes, value: bytes, *, ttl_seconds: int) -> None:
        ttl = self._require_ttl(ttl_seconds)
        client = await self._clients.get()
        await client.set(key, value, ex=ttl)

    async def add_cache(self, key: bytes, value: bytes, *, ttl_seconds: int) -> bool:
        ttl = self._require_ttl(ttl_seconds)
        client = await self._clients.get()
        stored: bool | None = await client.set(key, value, ex=ttl, nx=True)
        return bool(stored)

    async def delete(self, *keys: bytes) -> int:
        if not keys:
            return 0
        client = await self._clients.get()
        removed = 0
        for key in keys:  # one DEL per key: keys of different slots cannot share a DEL
            removed += int(await client.delete(key))
        return removed

    async def hash_get_all(self, key: bytes) -> Mapping[bytes, bytes]:
        client = await self._clients.get()
        fields: Mapping[bytes, bytes] = await client.hgetall(key)
        return fields

    async def incr(self, key: bytes, amount: int = 1) -> int:
        client = await self._clients.get()
        value: int = await client.incrby(key, amount)
        return value

    async def run_script(
        self, source: str, keys: Sequence[bytes], args: Sequence[bytes | str | int]
    ) -> Any:
        client = await self._clients.get()
        script = self._scripts.get(source)
        if script is None:
            script = client.register_script(source)
            self._scripts[source] = script
        return await script(keys=list(keys), args=list(args), client=client)

    async def replication_status(self) -> tuple[int, int]:
        """`(primaries, primaries with at least one online replica)`, as the primaries report it.

        A freshly formed cluster serves every slot before its replicas have attached and
        synced; until they have, losing a primary loses its slots. Readiness waits for this.
        """
        client = await self._clients.get()
        await client.nodes_manager.initialize()  # refresh topology: replicas attach after slots
        primaries: list[Any] = list(client.get_primaries())
        ready = 0
        for node in primaries:
            info: dict[str, Any] = await client.info("replication", target_nodes=node)
            states: list[str] = [
                str(cast(dict[str, Any], v).get("state"))
                for k, v in info.items()
                if k.startswith("slave") and isinstance(v, dict)
            ]
            if "online" in states:
                ready += 1
        return len(primaries), ready

    async def ping(self) -> bool:
        client = await self._clients.get()
        return bool(await client.ping())

    async def close(self) -> None:
        await self._clients.aclose()
