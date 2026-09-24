"""Multi-layer Semantic Model Router.

Implements Layer 1 of the semantic routing architecture (Issue #189).
Classifies requests into FAST_TIER or DEPTH_TIER based on heuristic and semantic rules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from uclone_x.llm.models import LLMRequest


class LLMTier(StrEnum):
    """The tiers of models available for routing."""

    FAST_TIER = "fast_tier"
    DEPTH_TIER = "depth_tier"


class SemanticModelRouter:
    """Classifies incoming LLM requests into an execution tier."""

    def route(self, request: LLMRequest, context: Any = None) -> LLMTier:
        """Classify incoming request complexity to determine the tier.

        Uses heuristic lexical matching or query complexity.
        """
        if len(request.tools) > 5:
            return LLMTier.DEPTH_TIER

        for msg in request.messages:
            content = msg.content
            if content and len(content) > 1000:
                return LLMTier.DEPTH_TIER

            if content:
                lower_content = content.lower()
                if any(
                    kw in lower_content
                    for kw in ["analyze", "plan", "complex", "system", "architect"]
                ):
                    return LLMTier.DEPTH_TIER

        return LLMTier.FAST_TIER
