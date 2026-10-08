"""The browser tool's share of an 8K-window request, as one number two tests agree on (#2164).

`test_browser_tool.py` holds the browser's advertised definition under
`BROWSER_DEFINITION_TOKEN_CEILING`. `test_tool_result_ingest.py`'s 8K-window test sends every
default tool, and pads the browser up to that ceiling first, so it passes or fails the same
whatever the browser costs today. A schema that grows therefore fails the schema-size test,
which names the cause, and not the 8K test, which does not.
"""

from __future__ import annotations

import json
from typing import Final

from uclone_x.llm.compactor import estimate_text_tokens
from uclone_x.tools.protocols import ToolProtocol
from uclone_x.tools.schema import advertised_tool_parameters

#: The most the browser's definition may cost, counted as `browser_definition_tokens` counts.
#:
#: Twenty-five tokens (about a hundred bytes) above what it measured when this was set, on
#: `b54323b5` plus #2164's trim: room for the next argument or two, and the one after that
#: has to be paid for elsewhere. The current figure: `browser_definition_tokens(BrowserTool())`.
#:
#: The ceiling is not free to rise. On an 8K window the turn-start compaction threshold is
#: 70% of the window, and the system prompt with every default tool sits a few dozen tokens
#: under it; the 8K-window ingest test is the one that says by how much.
BROWSER_DEFINITION_TOKEN_CEILING: Final = 325


def browser_definition_tokens(tool: ToolProtocol) -> int:
    """What a tool's definition costs in a request, the way `estimate_request_tokens` counts it.

    Name, description and the advertised parameter schema, plus four tokens of framing.
    """
    schema = json.dumps(advertised_tool_parameters(tool), ensure_ascii=False)
    return 4 + estimate_text_tokens(f"{tool.name} {tool.description} {schema}")
