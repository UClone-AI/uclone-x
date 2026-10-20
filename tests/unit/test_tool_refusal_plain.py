"""A call to a tool outside an agent's range is refused in plain words (#2196 item 1).

The refusal used to name the agent's id and list its `allowed_tools`: a permission list and
an internal id, which reach the person's screen through the turn's tool record and tell the
model nothing it can act on. It is now one plain sentence, kept in core i18n in English and
Korean (`uclone_x.i18n.tool_refusals`), and still a different answer from "no such tool".
"""

from __future__ import annotations

import json
from importlib import resources
from typing import get_args

from tests.unit.test_agent_base import CannedToolResponseConnector, DummyEchoParams
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.i18n.language import Language
from uclone_x.i18n.tool_refusals import (
    TOOL_REFUSAL_TEXT,
    ToolRefusalCode,
    tool_refusal_text,
)
from uclone_x.llm.models import ToolCallRequest
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry

OUT_OF_RANGE_EN = (
    "This assistant is not set up to use that tool, so it was not run. "
    "Use only the tools listed in this request."
)
OUT_OF_RANGE_KO = (
    "이 어시스턴트는 그 도구를 쓰도록 설정되어 있지 않아 실행하지 않았습니다. "
    "이 요청에 나온 도구만 사용하십시오."
)


class _Forbidden(BaseTool[DummyEchoParams]):
    name = "forbidden_tool"
    description = "Must not run here."

    def run(self, params: DummyEchoParams, context: ToolContext) -> str:
        raise AssertionError("an out-of-range tool ran")


async def _refused() -> tuple[str, str]:
    """The tool message and the record's error for a call outside the agent's range."""
    reg = ToolRegistry()
    reg.register(_Forbidden())
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="scoped_agent_7f3a",
            name="ScopedAgent",
            system_prompt="",
            allowed_tools=("some_other_tool", "another_tool"),
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=CannedToolResponseConnector(
            content="Calling a tool it was never shown",
            tool_calls=(
                ToolCallRequest(id="call_1", name="forbidden_tool", arguments={"text": "x"}),
            ),
        ),
        tools=reg,
    )
    result = await agent.execute_turn("Go")
    record = result.tool_executions[0]
    assert record.status is ToolResultStatus.ERROR
    message = next(m for m in agent.history if m.tool_call_id == "call_1")
    return str(message.content), record.error or ""


async def test_the_refusal_is_the_plain_sentence() -> None:
    """
    Killed by: src/uclone_x/agent/tool_execution.py :: err_msg = tool_refusal_text("tool.out_of_range", "en")
    Becomes: err_msg = f"Tool '{effective_tc.name}' is not in agent '{self.agent_id}' allowed_tools {tuple(self._config.allowed_tools)!r}"
    """
    content, error = await _refused()

    assert content == error == OUT_OF_RANGE_EN


async def test_the_refusal_shows_no_id_and_no_permission_list() -> None:
    """
    Killed by: src/uclone_x/agent/tool_execution.py :: err_msg = tool_refusal_text("tool.out_of_range", "en")
    Becomes: err_msg = f"Tool '{effective_tc.name}' is not in agent '{self.agent_id}' allowed_tools {tuple(self._config.allowed_tools)!r}"
    """
    content, error = await _refused()

    for text in (content, error):
        assert "scoped_agent_7f3a" not in text
        assert "allowed_tools" not in text
        assert "some_other_tool" not in text and "another_tool" not in text
        assert "not found" not in text  # "not allowed" is not "not there"


def test_the_catalog_says_it_in_english_and_korean() -> None:
    """
    Killed by: src/uclone_x/i18n/tool_refusals.py :: if language == "en":
    Becomes: if True:
    """
    assert tool_refusal_text("tool.out_of_range", "en") == OUT_OF_RANGE_EN
    assert tool_refusal_text("tool.out_of_range", "ko") == OUT_OF_RANGE_KO


def test_every_code_has_a_sentence_in_every_language() -> None:
    codes: tuple[ToolRefusalCode, ...] = get_args(ToolRefusalCode)
    languages: tuple[Language, ...] = get_args(Language)
    assert set(TOOL_REFUSAL_TEXT) == set(codes)
    for language in languages:
        if language == "en":
            continue
        source = resources.files("uclone_x.i18n").joinpath("locales", language)
        catalog = json.loads(source.joinpath("tool_refusals.json").read_text("utf-8"))
        assert set(catalog) == set(codes), language
        assert all(isinstance(v, str) and v.strip() for v in catalog.values())
