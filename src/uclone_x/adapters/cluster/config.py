"""Configuration for the cluster adapters, read from the stack's own env file.

There are no default secrets. `ucx cluster up` generates the Postgres password, the Valkey
password, the Centrifugo API key and the Centrifugo HMAC secret once, into the stack env file
(`.ucx-cluster/stack.env` beside `docker-compose.yml`, gitignored), and compose and this module
read the same file. A missing value raises `ClusterConfigError` naming it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import UCloneXError

__all__ = [
    "REQUIRED_STACK_SECRETS",
    "VALKEY_NODE_COUNT",
    "ClusterCachePolicy",
    "ClusterConfigError",
    "ClusterRuntimeConfig",
    "STACK_ENV_FILE_VAR",
    "read_stack_env",
    "stack_env_path",
]

#: Environment variable naming the stack env file; `ucx cluster up` sets it for the processes
#: it starts.
STACK_ENV_FILE_VAR: Final[str] = "UCX_CLUSTER_ENV_FILE"

#: Secrets `ucx cluster up` generates; compose refuses to start without them.
REQUIRED_STACK_SECRETS: Final[tuple[str, ...]] = (
    "UCX_POSTGRES_PASSWORD",
    "UCX_VALKEY_PASSWORD",
    "UCX_CENTRIFUGO_API_KEY",
    "UCX_CENTRIFUGO_HMAC_SECRET",
)

#: Valkey cluster nodes in `docker-compose.yml` (3 primaries + 3 replicas).
VALKEY_NODE_COUNT: Final[int] = 6

_DEFAULT_PORTS: Final[Mapping[str, str]] = {
    "UCX_KAFKA_PORT": "19092",
    "UCX_POSTGRES_PORT": "15432",
    "UCX_CENTRIFUGO_PORT": "18000",
    **{f"UCX_VALKEY_PORT_{i}": str(17000 + i) for i in range(1, VALKEY_NODE_COUNT + 1)},
}


class ClusterConfigError(UCloneXError):
    """Raised when the stack env file is missing or lacks a required value."""


def stack_env_path(compose_dir: Path | None = None) -> Path:
    """Where the stack env file is: `$UCX_CLUSTER_ENV_FILE`, else `.ucx-cluster/stack.env`."""
    override = os.environ.get(STACK_ENV_FILE_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return (compose_dir or Path.cwd()) / ".ucx-cluster" / "stack.env"


def read_stack_env(path: Path) -> dict[str, str]:
    """Parse a `KEY=VALUE` env file (blank lines and `#` comments ignored)."""
    if not path.is_file():
        raise ClusterConfigError(
            f"The cluster stack env file {path} does not exist. Run `ucx cluster up` once to "
            "generate it."
        )
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


class ClusterCachePolicy(BaseModel):
    """TTLs for cache keys (evictable under volatile-lru) and the lease deadline."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    head_ttl_seconds: int = Field(default=3600, ge=1)
    blob_ttl_seconds: int = Field(default=86400, ge=1)
    catalog_ttl_seconds: int = Field(default=600, ge=1)
    lease_ttl_seconds: int = Field(default=60, ge=1)


class ClusterRuntimeConfig(BaseModel):
    """Endpoints and secrets of one running cluster stack. Every secret is required."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    postgres_dsn: str = Field(min_length=1)
    valkey_nodes: tuple[str, ...] = Field(min_length=1)
    valkey_password: str = Field(min_length=1)
    valkey_address_map: Mapping[str, str] = Field(default_factory=dict[str, str])
    kafka_bootstrap_servers: tuple[str, ...] = Field(min_length=1)
    centrifugo_url: str = Field(min_length=1)
    centrifugo_api_key: str = Field(min_length=1)
    centrifugo_hmac_secret: str = Field(min_length=1)
    cache_policy: ClusterCachePolicy = Field(default_factory=ClusterCachePolicy)
    max_concurrent_turns: int = Field(default=16, ge=1)

    @classmethod
    def from_stack_env(cls, path: Path, *, host: str = "127.0.0.1") -> ClusterRuntimeConfig:
        """Build the config a host-side client needs from the stack env file at `path`."""
        env = read_stack_env(path)
        missing = [name for name in REQUIRED_STACK_SECRETS if not env.get(name)]
        if missing:
            raise ClusterConfigError(
                f"The cluster stack env file {path} lacks {', '.join(missing)}. Run "
                "`ucx cluster up` to generate the missing secrets."
            )

        def port(name: str) -> str:
            return env.get(name) or _DEFAULT_PORTS[name]

        user = env.get("UCX_POSTGRES_USER") or "uclone"
        database = env.get("UCX_POSTGRES_DB") or "uclone_cluster"
        valkey_ports = [port(f"UCX_VALKEY_PORT_{i}") for i in range(1, VALKEY_NODE_COUNT + 1)]
        return cls(
            postgres_dsn=(
                f"postgresql://{user}:{env['UCX_POSTGRES_PASSWORD']}@{host}:"
                f"{port('UCX_POSTGRES_PORT')}/{database}"
            ),
            valkey_nodes=tuple(f"{host}:{p}" for p in valkey_ports),
            valkey_password=env["UCX_VALKEY_PASSWORD"],
            # Nodes announce `valkey-N:6379` (their compose hostnames); from the host the same
            # node is the published port.
            valkey_address_map={
                f"valkey-{i}:6379": f"{host}:{p}" for i, p in enumerate(valkey_ports, start=1)
            },
            kafka_bootstrap_servers=(f"{host}:{port('UCX_KAFKA_PORT')}",),
            centrifugo_url=f"http://{host}:{port('UCX_CENTRIFUGO_PORT')}",
            centrifugo_api_key=env["UCX_CENTRIFUGO_API_KEY"],
            centrifugo_hmac_secret=env["UCX_CENTRIFUGO_HMAC_SECRET"],
        )
