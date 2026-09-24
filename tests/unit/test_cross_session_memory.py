"""Unit tests for cross-session memory facts and synthesis adapters (Issue #476)."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.errors import MissingProvenanceError
from uclone_x.llm.models import MessageRole
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.skills.synthesizer import SkillSynthesizer
from uclone_x.tools.models import ToolContext


def _test_provenance(provider: str = "agent.test") -> Provenance:
    return Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider=provider, model="test-model"),
        served_by=ServiceRef(provider=provider, model="test-model"),
    )


def test_memory_fact_creation_and_provenance_enforcement() -> None:
    """MemoryFact requires P6 in-band Provenance and validates typed fields."""
    prov = _test_provenance()
    fact = MemoryFact(
        subject="user_preferences",
        predicate="preferred_theme",
        object_value="dark",
        provenance=prov,
        source_session_id="sess_alpha",
        confidence=0.95,
        tags=("ui", "theme"),
    )
    assert fact.subject == "user_preferences"
    assert fact.predicate == "preferred_theme"
    assert fact.object_value == "dark"
    assert fact.provenance == prov
    assert fact.confidence == 0.95
    assert fact.tags == ("ui", "theme")
    assert fact.retracted is False
    assert fact.summary() == "user_preferences: preferred_theme -> dark"

    with pytest.raises(MissingProvenanceError):
        MemoryFact(  # pyright: ignore[reportCallIssue]
            subject="user",
            predicate="name",
            object_value="Kenny",
            provenance=None,  # pyright: ignore[reportArgumentType]
            source_session_id="sess_alpha",
        )


def test_memory_fact_conflicts_detection() -> None:
    """conflicts_with detects contradictory statements on same subject/predicate.

    Killed by: src/uclone_x/memory/models.py :: return same_subject and same_predicate and different_value
    Becomes: return False
    """
    prov = _test_provenance()
    fact1 = MemoryFact(
        fact_id="mem_1",
        subject="build_system",
        predicate="target_version",
        object_value="py311",
        provenance=prov,
        source_session_id="sess_1",
    )
    fact2 = MemoryFact(
        fact_id="mem_2",
        subject="build_system",
        predicate="target_version",
        object_value="py312",
        provenance=prov,
        source_session_id="sess_2",
    )
    fact_same = MemoryFact(
        fact_id="mem_3",
        subject="build_system",
        predicate="target_version",
        object_value="py311",
        provenance=prov,
        source_session_id="sess_3",
    )
    fact_other = MemoryFact(
        fact_id="mem_4",
        subject="compiler",
        predicate="target_version",
        object_value="py312",
        provenance=prov,
        source_session_id="sess_4",
    )

    assert fact1.conflicts_with(fact2) is True
    assert fact2.conflicts_with(fact1) is True
    assert fact1.conflicts_with(fact_same) is False
    assert fact1.conflicts_with(fact_other) is False
    assert fact1.conflicts_with(fact1) is False

    fact2_retracted = fact2.model_copy(update={"retracted": True})
    assert fact1.conflicts_with(fact2_retracted) is False


def test_cross_session_memory_auto_retract_conflicts() -> None:
    """Recording a conflicting fact automatically retracts the superseded fact.

    Killed by: src/uclone_x/memory/store.py :: if auto_retract_conflicts:
    Becomes: if False:
    """
    memory = CrossSessionMemory()
    prov = _test_provenance()

    fact1 = memory.record_fact(
        subject="api_client",
        predicate="base_url",
        object_value="https://api.staging.example.com",
        provenance=prov,
        source_session_id="session_stage",
        confidence=0.8,
    )
    assert fact1.retracted is False

    fact2 = memory.record_fact(
        subject="api_client",
        predicate="base_url",
        object_value="https://api.prod.example.com",
        provenance=prov,
        source_session_id="session_prod",
        confidence=0.99,
        auto_retract_conflicts=True,
    )

    fact1_updated = memory.get_fact(fact1.fact_id)
    assert fact1_updated is not None
    assert fact1_updated.retracted is True
    assert "Superseded" in (fact1_updated.retraction_reason or "")
    assert fact2.contradicts_fact_id == fact1.fact_id

    active_facts = memory.list_facts(include_retracted=False)
    assert len(active_facts) == 1
    assert active_facts[0].fact_id == fact2.fact_id

    all_facts = memory.list_facts(include_retracted=True)
    assert len(all_facts) == 2


def test_cross_session_memory_explicit_retraction() -> None:
    """Explicit retraction updates status and preserves audit history."""
    memory = CrossSessionMemory()
    prov = _test_provenance()

    fact = memory.record_fact(
        subject="user_settings",
        predicate="notifications",
        object_value="slack",
        provenance=prov,
        source_session_id="sess_10",
    )

    retracted = memory.retract_fact(
        fact_id=fact.fact_id,
        reason="User disabled Slack alerts",
        provenance=prov,
        session_id="sess_11",
    )
    assert retracted.retracted is True
    assert retracted.retraction_reason == "User disabled Slack alerts"
    assert retracted.retracted_at is not None

    re_retracted = memory.retract_fact(
        fact_id=fact.fact_id,
        reason="Second retraction attempt",
        provenance=prov,
    )
    assert re_retracted.retracted_at == retracted.retracted_at

    with pytest.raises(KeyError, match="not found"):
        memory.retract_fact("nonexistent_id", "any reason", provenance=prov)


def test_cross_session_memory_bounded_prompt_injection() -> None:
    """format_prompt_section applies progressive disclosure with bounded size.

    Killed by: src/uclone_x/memory/store.py :: limit = max_facts if max_facts is not None else self._max_facts_in_prompt
    Becomes: limit = 999999
    """
    memory = CrossSessionMemory(max_facts_in_prompt=2)
    prov = _test_provenance()

    for i in range(5):
        memory.record_fact(
            subject=f"component_{i}",
            predicate="status",
            object_value="healthy",
            provenance=prov,
            source_session_id=f"sess_{i}",
            confidence=0.7 + (i * 0.05),
        )

    prompt_section = memory.format_prompt_section()
    assert "[Cross-Session Memory Facts]" in prompt_section
    assert "component_4" in prompt_section
    assert "component_3" in prompt_section
    assert "3 additional facts omitted for prompt bounds" in prompt_section

    # Empty memory produces empty prompt section
    empty_memory = CrossSessionMemory()
    assert empty_memory.format_prompt_section() == ""


def test_cross_session_memory_persistence(tmp_path: Path) -> None:
    """Memory facts atomically persist to disk and reload across sessions."""
    db_file = tmp_path / "memory_store.json"
    mem1 = CrossSessionMemory(storage_path=db_file)
    prov = _test_provenance()

    f1 = mem1.record_fact(
        subject="service",
        predicate="port",
        object_value="8080",
        provenance=prov,
        source_session_id="sess_a",
    )
    f2 = mem1.record_fact(
        subject="service",
        predicate="host",
        object_value="localhost",
        provenance=prov,
        source_session_id="sess_a",
    )
    mem1.retract_fact(f1.fact_id, "Port changed", prov)

    # Load in new instance
    mem2 = CrossSessionMemory(storage_path=db_file)
    assert mem2.get_fact(f1.fact_id) is not None
    assert mem2.get_fact(f1.fact_id).retracted is True  # pyright: ignore[reportOptionalMemberAccess]
    assert mem2.get_fact(f2.fact_id) is not None
    assert mem2.get_fact(f2.fact_id).retracted is False  # pyright: ignore[reportOptionalMemberAccess]
    assert len(mem2.list_facts(include_retracted=False)) == 1


def test_cross_session_memory_p9_skill_synthesizer_input() -> None:
    """CrossSessionMemory provides workflow steps directly into SkillSynthesizer."""
    memory = CrossSessionMemory()
    prov = _test_provenance()

    memory.record_fact(
        subject="pr_workflow",
        predicate="pre_check",
        object_value="run quality gate before commit",
        provenance=prov,
        source_session_id="sess_1",
        tags=("workflow",),
    )
    memory.record_fact(
        subject="pr_workflow",
        predicate="commit_format",
        object_value="use atomic conventional commits",
        provenance=prov,
        source_session_id="sess_1",
        tags=("workflow",),
    )

    steps = memory.to_skill_workflow_steps(subject="pr_workflow")
    assert len(steps) == 2
    assert any("pre_check" in s for s in steps)

    synthesizer = SkillSynthesizer()
    code = synthesizer.generate_skill_code("pr_routine", steps)
    assert "pr_routine" in code
    assert "run quality gate before commit" in code


def test_cross_session_memory_p7_ontology_evidence_and_promotion() -> None:
    """Aggregates multi-session facts into EvidenceRecord and promotes into OntologyEngine.

    Killed by: src/uclone_x/memory/store.py :: observation_count=len(active_facts),
    Becomes: observation_count=0,
    """
    memory = CrossSessionMemory()
    prov = _test_provenance()

    # Two distinct sessions observe the same fact
    f1 = memory.record_fact(
        subject="UserEntity",
        predicate="role",
        object_value="admin",
        provenance=prov,
        source_session_id="session_101",
    )
    memory.record_fact(
        subject="UserEntity",
        predicate="role",
        object_value="admin",
        provenance=prov,
        source_session_id="session_102",
    )

    evidence = memory.export_ontology_evidence("UserEntity", "role")
    assert evidence is not None
    assert evidence.observation_count == 2
    assert evidence.session_count == 2
    assert "session_101" in evidence.distinct_sessions
    assert "session_102" in evidence.distinct_sessions

    ontology = OntologyEngine()
    promoted = memory.promote_fact_to_ontology_concept(
        ontology=ontology,
        fact_id=f1.fact_id,
        force=True,
    )
    assert promoted.name == "UserEntity"
    assert promoted.attributes.get("role") == "admin"


@pytest.mark.asyncio
async def test_cross_session_memory_agent_wiring_and_tools() -> None:
    """BaseAgent wires memory, registers memory tools, and injects facts into prompts.

    Killed by: src/uclone_x/agent/base.py :: memory_section = self._memory.format_prompt_section()
    Becomes: memory_section = ""
    """
    memory = CrossSessionMemory()
    prov = _test_provenance()
    memory.record_fact(
        subject="codebase",
        predicate="package_manager",
        object_value="uv",
        provenance=prov,
        source_session_id="init_session",
    )

    config = AgentConfig(agent_id="agent_mem_test", name="MemAgent")
    agent = BaseAgent(config=config, memory=memory)

    # Verify property and tools
    assert agent.memory is memory
    assert agent.tools is not None
    assert agent.tools.get("record_memory_fact") is not None
    assert agent.tools.get("retract_memory_fact") is not None
    assert agent.tools.get("query_memory_facts") is not None

    # Test prompt injection: at the tail of the request, never in the system turn, which
    # must hold still for the provider's cached prefix to survive a recorded fact.
    turn_messages = agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
    tail = turn_messages[-1]
    assert tail.role is MessageRole.USER
    assert tail.content is not None
    assert "[Cross-Session Memory Facts]" in tail.content
    assert "codebase: package_manager -> uv" in tail.content
    assert "[Cross-Session Memory Facts]" not in (turn_messages[0].content or "")

    # Test record_memory_fact tool execution
    record_tool = agent.tools.get("record_memory_fact")
    assert record_tool is not None
    ctx = ToolContext(agent_id="agent_mem_test", session_id="test_sess")
    res = await record_tool.execute(
        {
            "subject": "python_version",
            "predicate": "minimum",
            "object_value": "3.11",
        },
        ctx,
    )
    assert res.success is True
    assert isinstance(res.output, str)
    assert "python_version: minimum -> 3.11" in res.output

    # Test query_memory_facts tool
    query_tool = agent.tools.get("query_memory_facts")
    assert query_tool is not None
    q_res = await query_tool.execute(
        {"subject": "python_version"},
        ctx,
    )
    assert q_res.success is True
    assert isinstance(q_res.output, str)
    assert "python_version" in q_res.output

    # Test retract_memory_fact tool
    facts = memory.list_facts(subject="python_version")
    assert len(facts) == 1
    fact_id = facts[0].fact_id

    retract_tool = agent.tools.get("retract_memory_fact")
    assert retract_tool is not None
    r_res = await retract_tool.execute(
        {"fact_id": fact_id, "reason": "Bumped to 3.12"},
        ctx,
    )
    assert r_res.success is True
    assert isinstance(r_res.output, str)
    assert "Successfully retracted" in r_res.output
    fact_retracted = memory.get_fact(fact_id)
    assert fact_retracted is not None
    assert fact_retracted.retracted is True


def test_record_memory_fact_params_accepts_json_list_tags() -> None:
    """RecordMemoryFactParams coerces JSON list inputs for tags into a tuple."""
    from uclone_x.memory.tools import RecordMemoryFactParams

    params = RecordMemoryFactParams.model_validate(
        {
            "subject": "user",
            "predicate": "favorite_color",
            "object_value": "teal",
            "tags": ["personal", "preference"],
        }
    )
    assert params.tags == ("personal", "preference")
    assert isinstance(params.tags, tuple)
