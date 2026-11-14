"""One paid call's tokens: the row every usage store keeps, whoever keeps it.

Kept in the kernel so that a store in `adapters/` (the cluster's) and the file store in
`llm/usage/store.py` share it without importing each other.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.llm.models import TokenCountSource

__all__ = ["USAGE_RETENTION", "UsageEntry"]

#: How long rows are kept: the longest window (7 days) and a day to spare.
USAGE_RETENTION = timedelta(days=8)


class UsageEntry(BaseModel):
    """One paid call's tokens, input and output together."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    at: datetime = Field(description="When the call finished, timezone-aware.")
    tokens: int = Field(ge=0)
    provider: str
    model: str | None = None
    count_source: TokenCountSource = TokenCountSource.PROVIDER
    cached_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Of `tokens`, the input tokens the provider served from its prompt cache "
        "(`TokenUsage.cache_read_input_tokens`). `None` means the provider reported no "
        "figure, and a row written before the column existed reads as `None`; either way "
        "the whole row counts as new, because nothing shows any of it was cached.",
    )

    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Of `tokens`, what the call sent. `None` where a store keeps no split.",
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Of `tokens`, what the call received. `None` where a store keeps no split.",
    )

    @property
    def new_tokens(self) -> int:
        """`tokens` less the cached input: what the provider did not already hold."""
        return self.tokens - (self.cached_tokens or 0)
