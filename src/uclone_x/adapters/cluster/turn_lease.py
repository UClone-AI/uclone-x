"""Fenced turn lease and cancel tombstones as atomic Valkey scripts.

Coordination keys never carry a TTL, so `maxmemory-policy volatile-lru` can never evict them
(the cache keys beside them always do carry one). Expiry the protocol needs — the lease
deadline and a tombstone's lifetime — is stored in the value and compared with the server
clock (`TIME`) inside the script, so all workers share one clock and one atomic step.

Keys of one conversation share the `{conversation_id}` hash tag and therefore one cluster slot,
which is what lets a single script read the tombstone, check the lease and bump the fence:

* `lease:turn:{id}` — JSON `{worker_id, turn_id, fencing_token, deadline_ms}`
* `fence:{id}`      — monotonic fencing counter (`INCR`)
* `cancel:{id}`     — hash `turn_id -> expiry_ms`; expired fields are pruned on write
"""

from __future__ import annotations

import json
from typing import Any, Final

from uclone_x.core.cluster_protocols import (
    CacheClientProtocol,
    TurnLeaseConflictError,
    TurnLeaseProtocol,
    TurnLeaseToken,
    TurnPreemptedError,
)
from uclone_x.errors import UCloneXError

__all__ = ["ValkeyTurnLeaseManager"]

_NOW_MS: Final[str] = """
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
"""

#: KEYS: lease, fence, cancel. ARGV: turn_id, worker_id, ttl_ms.
#: Returns {'cancelled'} | {'held', holder_worker_id} | {'acquired', fencing_token, deadline_ms}.
ACQUIRE_SCRIPT: Final[str] = (
    _NOW_MS
    + """
local tomb = redis.call('HGET', KEYS[3], ARGV[1])
if tomb and tonumber(tomb) > now_ms then
  return {'cancelled'}
end
local cur = redis.call('GET', KEYS[1])
if cur then
  local held = cjson.decode(cur)
  if tonumber(held['deadline_ms']) > now_ms then
    return {'held', held['worker_id']}
  end
end
local fence = redis.call('INCR', KEYS[2])
local deadline = now_ms + tonumber(ARGV[3])
redis.call('SET', KEYS[1], cjson.encode({
  worker_id = ARGV[2], turn_id = ARGV[1], fencing_token = fence, deadline_ms = deadline
}))
return {'acquired', fence, deadline}
"""
)

#: KEYS: lease. ARGV: worker_id, fencing_token, turn_id, ttl_ms. Returns 1 if extended.
#: The turn is compared too (#2468): a failover can lose a fence `INCR`, so one worker can hold
#: the same fence for a later turn, and an old turn's token must not extend that lease.
RENEW_SCRIPT: Final[str] = (
    _NOW_MS
    + """
local cur = redis.call('GET', KEYS[1])
if not cur then return 0 end
local held = cjson.decode(cur)
if held['worker_id'] ~= ARGV[1] or tonumber(held['fencing_token']) ~= tonumber(ARGV[2])
   or held['turn_id'] ~= ARGV[3] then
  return 0
end
if tonumber(held['deadline_ms']) <= now_ms then return 0 end
held['deadline_ms'] = now_ms + tonumber(ARGV[4])
redis.call('SET', KEYS[1], cjson.encode(held))
return 1
"""
)

#: KEYS: lease. ARGV: worker_id, fencing_token, turn_id. Deletes only the caller's own lease,
#: of that turn (see `RENEW_SCRIPT`).
RELEASE_SCRIPT: Final[str] = """
local cur = redis.call('GET', KEYS[1])
if not cur then return 0 end
local held = cjson.decode(cur)
if held['worker_id'] ~= ARGV[1] or tonumber(held['fencing_token']) ~= tonumber(ARGV[2])
   or held['turn_id'] ~= ARGV[3] then
  return 0
end
return redis.call('DEL', KEYS[1])
"""

#: KEYS: cancel. ARGV: turn_id, ttl_ms. Records the tombstone and prunes expired ones.
TOMBSTONE_SCRIPT: Final[str] = (
    _NOW_MS
    + """
local fields = redis.call('HGETALL', KEYS[1])
for i = 1, #fields, 2 do
  if tonumber(fields[i + 1]) <= now_ms then
    redis.call('HDEL', KEYS[1], fields[i])
  end
end
redis.call('HSET', KEYS[1], ARGV[1], now_ms + tonumber(ARGV[2]))
return 1
"""
)

#: KEYS: cancel. ARGV: turn_id. Returns 1 while an unexpired tombstone exists.
IS_CANCELLED_SCRIPT: Final[str] = (
    _NOW_MS
    + """
local tomb = redis.call('HGET', KEYS[1], ARGV[1])
if tomb and tonumber(tomb) > now_ms then return 1 end
return 0
"""
)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class ValkeyTurnLeaseManager(TurnLeaseProtocol):
    """Valkey-backed turn lease with monotonic fencing tokens and cancel tombstones."""

    def __init__(self, cache: CacheClientProtocol, *, default_ttl_seconds: int = 60) -> None:
        if default_ttl_seconds < 1:
            raise ValueError(f"default_ttl_seconds must be >= 1, got {default_ttl_seconds}")
        self._cache = cache
        self._default_ttl_seconds = default_ttl_seconds

    @staticmethod
    def lease_key(conversation_id: str) -> bytes:
        return f"lease:turn:{{{conversation_id}}}".encode()

    @staticmethod
    def fence_key(conversation_id: str) -> bytes:
        return f"fence:{{{conversation_id}}}".encode()

    @staticmethod
    def cancel_key(conversation_id: str) -> bytes:
        return f"cancel:{{{conversation_id}}}".encode()

    async def acquire(
        self, conversation_id: str, turn_id: str, worker_id: str, ttl_seconds: int | None = None
    ) -> TurnLeaseToken:
        """Check the tombstone and take the lease in one atomic server-side step."""
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        outcome: list[Any] = await self._cache.run_script(
            ACQUIRE_SCRIPT,
            [
                self.lease_key(conversation_id),
                self.fence_key(conversation_id),
                self.cancel_key(conversation_id),
            ],
            [turn_id, worker_id, effective_ttl * 1000],
        )
        verdict = _text(outcome[0])
        if verdict == "cancelled":
            raise TurnPreemptedError(conversation_id, turn_id)
        if verdict == "held":
            raise TurnLeaseConflictError(conversation_id, turn_id, holder=_text(outcome[1]))
        if verdict != "acquired":
            raise UCloneXError(
                f"Unexpected lease script reply for {conversation_id!r}: {outcome!r}"
            )
        return TurnLeaseToken(
            conversation_id=conversation_id,
            turn_id=turn_id,
            worker_id=worker_id,
            fencing_token=int(outcome[1]),
            ttl_seconds=effective_ttl,
        )

    async def renew(self, token: TurnLeaseToken) -> bool:
        """Extend the lease iff this token's holder, fence and turn hold it, and it has not lapsed."""
        renewed = await self._cache.run_script(
            RENEW_SCRIPT,
            [self.lease_key(token.conversation_id)],
            [token.worker_id, token.fencing_token, token.turn_id, token.ttl_seconds * 1000],
        )
        return bool(renewed)

    async def release(self, token: TurnLeaseToken) -> None:
        """Release the lease iff this token's holder, fence and turn still own it."""
        await self._cache.run_script(
            RELEASE_SCRIPT,
            [self.lease_key(token.conversation_id)],
            [token.worker_id, token.fencing_token, token.turn_id],
        )

    async def current_holder(self, conversation_id: str) -> TurnLeaseToken | None:
        """Return the stored lease, expired or not (for diagnostics and tests)."""
        raw = await self._cache.get(self.lease_key(conversation_id))
        if raw is None:
            return None
        held: dict[str, Any] = json.loads(raw)
        return TurnLeaseToken(
            conversation_id=conversation_id,
            turn_id=str(held["turn_id"]),
            worker_id=str(held["worker_id"]),
            fencing_token=int(held["fencing_token"]),
            ttl_seconds=self._default_ttl_seconds,
        )

    async def write_cancel_tombstone(
        self, conversation_id: str, turn_id: str, ttl_seconds: int = 300
    ) -> None:
        """Record a pre-start or mid-turn interrupt tombstone for `turn_id`."""
        await self._cache.run_script(
            TOMBSTONE_SCRIPT,
            [self.cancel_key(conversation_id)],
            [turn_id, max(1, ttl_seconds) * 1000],
        )

    async def is_cancelled(self, conversation_id: str, turn_id: str) -> bool:
        """Return `True` while an unexpired tombstone exists for `(conversation_id, turn_id)`."""
        cancelled = await self._cache.run_script(
            IS_CANCELLED_SCRIPT, [self.cancel_key(conversation_id)], [turn_id]
        )
        return bool(cancelled)
