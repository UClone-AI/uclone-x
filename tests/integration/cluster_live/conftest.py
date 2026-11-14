"""Fixtures for the live cluster suite: a stack wired to the running docker compose cluster.

Opt-in only (`cluster_live` marker, `--cluster-live`): `ucx test changed`/`check` collect
these tests and skip them, and never start docker. Run it with

    ./ucx cluster up            # once; generates .ucx-cluster/stack.env
    ./ucx test cluster          # or: pytest tests/integration/cluster_live --cluster-live

The suite kills and restarts containers of the stack its env file names (a Valkey primary,
the Kafka broker), so point `UCX_CLUSTER_ENV_FILE` at a stack of your own. A missing or
unreachable stack fails the run; it is never skipped once `--cluster-live` was asked for.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

from uclone_x.adapters.cluster.config import (
    ClusterRuntimeConfig,
    read_stack_env,
    stack_env_path,
)
from uclone_x.adapters.cluster.valkey_cache import ValkeyClusterCache
from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack

REPO_ROOT = Path(__file__).resolve().parents[3]


def stack_env_file() -> Path:
    return stack_env_path(REPO_ROOT)


def compose_project() -> str:
    return read_stack_env(stack_env_file()).get("UCX_CLUSTER_PROJECT") or "uclone-x-cluster"


def container(service: str) -> str:
    """Container name compose gives the first replica of `service` in this stack."""
    return f"{compose_project()}-{service}-1"


def docker(*args: str) -> str:
    done = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


async def eventually(probe: Any, *, timeout: float = 60.0, interval: float = 0.5) -> Any:
    """Await `probe()` until it returns a truthy value without raising; return that value."""
    deadline = asyncio.get_running_loop().time() + timeout
    last: BaseException | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            value = await probe()
            if value:
                return value
        except Exception as exc:  # the probe is expected to fail while the cluster recovers
            last = exc
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s (last error: {last!r})")


@pytest.fixture(scope="session")
def cluster_config() -> ClusterRuntimeConfig:
    os.environ.setdefault("UCX_CLUSTER_ENV_FILE", str(stack_env_file()))
    return ClusterRuntimeConfig.from_stack_env(stack_env_file())


@pytest.fixture
async def stack(cluster_config: ClusterRuntimeConfig) -> AsyncIterator[ClusterRuntimeStack]:
    built = ClusterRuntimeStack.create_docker_stack(config=cluster_config)
    await built.bootstrap_schema()
    try:
        yield built
    finally:
        await built.close()


@pytest.fixture
async def second_stack(
    cluster_config: ClusterRuntimeConfig,
) -> AsyncIterator[ClusterRuntimeStack]:
    """A second, independent set of clients: another worker process as far as the cluster sees."""
    built = ClusterRuntimeStack.create_docker_stack(config=cluster_config)
    try:
        yield built
    finally:
        await built.close()


async def raw_valkey(stack: ClusterRuntimeStack) -> Any:
    """The `valkey` library client of `stack`, for probes the protocol does not expose."""
    return await cast(ValkeyClusterCache, stack.cache).raw_client()
