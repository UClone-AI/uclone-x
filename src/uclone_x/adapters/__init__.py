"""Non-invasive integration adapters between UClone-X and foreign ecosystems (#367).

Nothing in this package may import a foreign provider SDK. P5 states the rule without
qualification: "No module outside `uclone_x.llm.connectors.*` may import a provider SDK
(`anthropic`, `openai`, `google.genai`, `ollama`, or equivalent)". An adapter package is
the *legitimate place* for translation under the same principle — "All tool schemas,
function-calling payloads, and streaming outputs are translated to and from
provider-native shapes inside the adapter layer" — but the translation is written against
a locally-declared mirror of the foreign shape, never against the foreign package.
"""

from __future__ import annotations

__all__: list[str] = []
