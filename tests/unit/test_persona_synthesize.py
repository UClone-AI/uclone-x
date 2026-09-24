from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, StreamChunk
from uclone_x.ui.app import create_ui_app


class _UnreachableLLM:
    """A provider that is configured but does not answer, like a stopped Ollama."""

    @property
    def provider_name(self) -> str:
        return "unreachable"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        raise ConnectionError("connection refused by localhost:11434")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        raise ConnectionError("connection refused by localhost:11434")
        yield  # pragma: no cover


def _client(tmp_path: Path, llm: object) -> TestClient:
    return TestClient(create_ui_app(workspace_dir=tmp_path, llm=llm))  # type: ignore[arg-type]


@pytest.fixture
def offline_client(tmp_path: Path) -> TestClient:
    return _client(tmp_path, _UnreachableLLM())


def test_synthesize_persona_prompt_is_written_by_the_model(tmp_path: Path) -> None:
    llm = MockLLMConnector(default_response="너는 집밥 한식을 알려주는 요리사다.")
    res = _client(tmp_path, llm).post(
        "/api/personas/synthesize",
        json={
            "name": "요리사",
            "role": "한식 셰프",
            "description": "집에서 만들 수 있는 한식 레시피를 알려준다",
            "allowed_tools": ["web_search"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data == {
        "system_prompt": "너는 집밥 한식을 알려주는 요리사다.",
        "source": "llm",
        "model": "mock-model",
    }
    assert llm.call_count == 1


def test_synthesize_persona_prompt_sends_every_filled_field(tmp_path: Path) -> None:
    # The unscripted mock echoes the last message it was sent, so the reply is the request.
    res = _client(tmp_path, MockLLMConnector()).post(
        "/api/personas/synthesize",
        json={
            "name": "scout",
            "role": "Researcher",
            "description": "Finds sources",
            "allowed_tools": ["web_search", "web_fetch"],
        },
    )
    prompt = res.json()["system_prompt"]
    assert "Name: scout" in prompt
    assert "Role: Researcher" in prompt
    assert "Description: Finds sources" in prompt
    assert "Tools it can use: web_search, web_fetch" in prompt


def test_synthesize_persona_prompt_strips_an_enclosing_code_fence(tmp_path: Path) -> None:
    llm = MockLLMConnector(default_response="```markdown\nYou are Scout.\n- Cite sources.\n```")
    res = _client(tmp_path, llm).post("/api/personas/synthesize", json={"name": "scout"})
    assert res.json()["system_prompt"] == "You are Scout.\n- Cite sources."


def test_synthesize_persona_prompt_falls_back_when_the_model_is_silent(tmp_path: Path) -> None:
    res = _client(tmp_path, MockLLMConnector(default_response="   ")).post(
        "/api/personas/synthesize", json={"name": "scout"}
    )
    data = res.json()
    assert data["source"] == "template"
    assert data["fallback_reason"] == "the model returned an empty reply"
    assert data["system_prompt"].startswith("You are Scout, the Specialized AI Assistant")


def test_synthesize_persona_prompt_falls_back_and_names_the_cause(
    offline_client: TestClient,
) -> None:
    res = offline_client.post(
        "/api/personas/synthesize",
        json={
            "name": "code_reviewer",
            "role": "Senior Code Reviewer",
            "description": "Reviews pull requests and provides architectural feedback.",
            "allowed_tools": ["file_read", "file_edit", "generate_image"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["source"] == "template"
    assert data["fallback_reason"] == "connection refused by localhost:11434"
    prompt = data["system_prompt"]
    assert "You are Code Reviewer, the Senior Code Reviewer in UClone-X." in prompt
    assert "Reviews pull requests and provides architectural feedback." in prompt
    assert "Safe File Operations" in prompt
    assert "Visual Generation" in prompt


def test_synthesize_persona_prompt_validation_error(offline_client: TestClient) -> None:
    res = offline_client.post(
        "/api/personas/synthesize",
        json={"name": "", "role": "", "description": "", "allowed_tools": []},
    )
    assert res.status_code == 400
    assert "Provide at least a name, role, or description" in res.json()["detail"]


def test_synthesize_persona_prompt_tool_directives(offline_client: TestClient) -> None:
    res = offline_client.post(
        "/api/personas/synthesize",
        json={
            "name": "researcher",
            "role": "Deep Web Researcher",
            "description": "Scrapes and synthesizes web knowledge.",
            "allowed_tools": ["web_search", "bash_run", "delegate_subagent"],
        },
    )
    assert res.status_code == 200
    prompt = res.json()["system_prompt"]
    assert "Grounded Information" in prompt
    assert "Command Execution" in prompt
    assert "Subagent Delegation" in prompt
