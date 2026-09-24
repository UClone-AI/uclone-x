from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ToolDefinition
from uclone_x.llm.router import LLMTier, SemanticModelRouter


def test_semantic_router_fast_tier():
    router = SemanticModelRouter()
    req = LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="Hello!"),))
    tier = router.route(req)
    assert tier == LLMTier.FAST_TIER


def test_semantic_router_depth_tier_tools():
    router = SemanticModelRouter()
    tools = tuple(
        ToolDefinition(name=f"tool_{i}", description="mock", parameters={}) for i in range(6)
    )
    req = LLMRequest(
        messages=(ChatMessage(role=MessageRole.USER, content="Hello!"),),
        tools=tools,
    )
    tier = router.route(req)
    assert tier == LLMTier.DEPTH_TIER


def test_semantic_router_depth_tier_length():
    router = SemanticModelRouter()
    req = LLMRequest(messages=(ChatMessage(role=MessageRole.USER, content="A" * 1001),))
    tier = router.route(req)
    assert tier == LLMTier.DEPTH_TIER


def test_semantic_router_depth_tier_keyword():
    router = SemanticModelRouter()
    req = LLMRequest(
        messages=(ChatMessage(role=MessageRole.USER, content="Can you analyze this?"),)
    )
    tier = router.route(req)
    assert tier == LLMTier.DEPTH_TIER
