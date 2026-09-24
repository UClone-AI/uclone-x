# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Pre-release qualification scenario test for creative workspace workflow.

This scenario validates the full end-to-end creative workflow:
Stage 1: Document & Artifact Generation (RFC #837 ArtifactsDock payload readiness & safety)
Stage 2: Deterministic Local Image Generation (RFC #849 zero-setup silicon dispatch & provenance)
Stage 3: Dynamic Knowledge Graph Triples & Traceability (RFC #837 Graph projection & filtering)

Governed by Issue #853 and AGENTS.md Pre-Release Qualification Tier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import OntologyConcept, OntologyTier
from uclone_x.tools.builtin.comfy_client import ComfyClient
from uclone_x.tools.builtin.comfy_image_tool import ComfyImageGenTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import AgentSessionManager, create_ui_app


@pytest.mark.pre_release
@pytest.mark.asyncio
async def test_user_creative_workflow_scenario(tmp_path: Path) -> None:
    """Pre-release qualification: Docs authoring -> Image generation -> Knowledge Graph accumulation."""
    workspace_dir = tmp_path / "creative_workspace"
    workspace_dir.mkdir(parents=True)
    storage_dir = tmp_path / "sessions"
    storage_dir.mkdir(parents=True)
    session_id = "sess_creative_flow_001"
    agent_id = "creative_planner_agent"

    bus = EventBus(maxsize=100)
    await bus.start()
    _ = SessionStore(storage_dir=storage_dir)

    # ----------------------------------------------------------------------------------
    # Stage 1: Document & Artifact Generation (ArtifactsDock payload readiness & safety)
    # ----------------------------------------------------------------------------------
    # 1.1: Agent writes a project roadmap document into the workspace docs/ directory
    docs_dir = workspace_dir / "docs" / "design"
    docs_dir.mkdir(parents=True, exist_ok=True)
    roadmap_doc = docs_dir / "fantasy_card_game_design.md"
    roadmap_doc.write_text(
        "# Fantasy Card Game Design Specification\n\n"
        "## Overview\n"
        "A tactical card game featuring procedural characters.\n\n"
        "## Key Systems\n"
        "- Battle Engine\n"
        "- Procedural Illustration Generation\n",
        encoding="utf-8",
    )

    # 1.2: Agent creates structured workspace artifacts
    art_dir = workspace_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    concept_doc = art_dir / "character_concepts.md"
    concept_doc.write_text(
        "# Character Concepts & Archetypes\n\nFlame Knight: High melee burst, fiery armor.\n",
        encoding="utf-8",
    )

    # 1.3: Session-specific tool output artifact
    session_tool_dir = workspace_dir / ".sandbox" / "tool_artifacts" / session_id
    session_tool_dir.mkdir(parents=True, exist_ok=True)
    tool_doc = session_tool_dir / "card_balance_analysis.md"
    tool_doc.write_text(
        "# Balance Analysis Matrix\n\nSimulated 10,000 matches with balanced win-rate across factions.\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------------------
    # Stage 2: Deterministic Local Image Generation (RFC #849 silicon dispatch & provenance)
    # ----------------------------------------------------------------------------------
    # Mock ComfyClient to emulate local silicon / remote CUDA generation without external server
    mock_client = MagicMock(spec=ComfyClient)
    mock_client.queue_prompt = AsyncMock(return_value="prompt_creative_999")
    mock_client.wait_for_output = AsyncMock(return_value=["card_flame_knight.png"])
    # 1x1 transparent PNG payload
    dummy_png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63f8ffffff3f0005fe02fedccc59e70000000049454e44ae426082"
    )
    mock_client.download_image = AsyncMock(return_value=dummy_png)

    image_tool = ComfyImageGenTool(client=mock_client)
    tools = ToolRegistry()
    tools.register(image_tool)

    turn_idx = 1
    image_prompt = "Epic warrior flame knight with blazing golden armor and molten broadsword"
    # Derive deterministic seed as defined in RFC #849
    seed_hash = hashlib.md5(f"{session_id}__{turn_idx}__{image_prompt}".encode()).hexdigest()[:8]
    deterministic_seed = int(seed_hash, 16)

    tool_context = ToolContext(
        session_id=session_id,
        agent_id=agent_id,
        workspace_root=workspace_dir,
    )

    # Execute image generation
    image_result = await image_tool.execute(
        params={
            "prompt": image_prompt,
            "seed": deterministic_seed,
            "output_path": f"artifacts/images/flame_knight_{deterministic_seed}.png",
            "width": 512,
            "height": 512,
        },
        context=tool_context,
    )

    assert image_result.success is True
    assert isinstance(image_result.output, dict)
    out_dict = cast(dict[str, Any], image_result.output)
    assert out_dict["seed"] == deterministic_seed
    assert out_dict["prompt"] == image_prompt
    assert "artifacts/images/flame_knight" in out_dict["path"]

    # Verify physical file written to workspace safely
    generated_file = workspace_dir / out_dict["path"]
    assert generated_file.is_file()
    assert generated_file.read_bytes() == dummy_png

    # ----------------------------------------------------------------------------------
    # Stage 3: Dynamic Knowledge Graph Reasoning & Integration
    # ----------------------------------------------------------------------------------
    ontology_engine = OntologyEngine(agent_id=agent_id)
    ontology_engine.register_entity(
        OntologyConcept(
            name="CardGameDesign",
            parent_type="Specification",
            tier=OntologyTier.ASSERTED,
            attributes={"title": "string", "version": "string"},
            required_fields=("title",),
        )
    )
    ontology_engine.register_entity(
        OntologyConcept(
            name="FlameKnightCard",
            tier=OntologyTier.INDUCED_ENFORCING,
            attributes={"attack": "int", "defense": "int"},
            required_fields=("attack", "defense"),
        )
    )
    ontology_engine.induce_relation(
        source_entity="CardGameDesign",
        predicate="specifies",
        target_entity="FlameKnightCard",
        source_session=session_id,
        confidence=0.98,
    )

    # ----------------------------------------------------------------------------------
    # Stage 4: Verify End-to-End API Projections (ArtifactsDock & Knowledge Graph endpoints)
    # ----------------------------------------------------------------------------------
    session_manager = AgentSessionManager(
        bus=bus,
        storage_dir=storage_dir,
        workspace_dir=workspace_dir,
        ontology_engine=ontology_engine,
        tools=tools,
    )

    app = create_ui_app(
        static_dir=tmp_path / "ui_static",
        session_manager=session_manager,
    )
    client = TestClient(app)

    # 4.1 Verify /api/artifacts enumeration
    art_res = client.get(f"/api/artifacts?session_id={session_id}")
    assert art_res.status_code == 200
    art_data = cast(dict[str, Any], art_res.json())
    artifacts = art_data["artifacts"]
    assert art_data["total"] >= 2

    artifact_paths = {a["path"] for a in artifacts}
    assert "artifacts/character_concepts.md" in artifact_paths
    assert f".sandbox/tool_artifacts/{session_id}/card_balance_analysis.md" in artifact_paths
    # A document in the workspace's docs/ is readable, but it is not listed as an artifact.
    assert "docs/design/fantasy_card_game_design.md" not in artifact_paths

    # Verify structured metadata fields
    concept_item = next(a for a in artifacts if a["path"] == "artifacts/character_concepts.md")
    assert concept_item["title"] == "Character Concepts & Archetypes"
    assert concept_item["id"].startswith("art_")
    assert concept_item["size_bytes"] > 0
    assert "created_at" in concept_item
    assert "modified_at" in concept_item

    # 4.2 Verify /api/artifacts/content reading and mime-types
    content_res = client.get("/api/artifacts/content?path=docs/design/fantasy_card_game_design.md")
    assert content_res.status_code == 200
    assert "text/markdown" in content_res.headers.get("content-type", "")
    assert "procedural characters" in content_res.text

    # 4.3 Verify Security Invariant (P6 Path Traversal Prevention)
    forbidden_dots = client.get("/api/artifacts/content?path=../../etc/passwd")
    assert forbidden_dots.status_code == 400
    assert (
        "escapes workspace root" in forbidden_dots.json()["detail"]
        or "traversal" in forbidden_dots.json()["detail"].lower()
    )

    forbidden_abs = client.get("/api/artifacts/content?path=/etc/shadow")
    assert forbidden_abs.status_code == 400

    not_found = client.get("/api/artifacts/content?path=docs/missing_file.md")
    assert not_found.status_code == 404

    # 4.4 Verify Dynamic Knowledge Graph projection & triples
    # A. Session-filtered query
    kg_res = client.get(f"/api/knowledge-graph?session_id={session_id}")
    assert kg_res.status_code == 200
    kg_data = cast(dict[str, Any], kg_res.json())
    assert "triples" in kg_data
    assert "nodes" in kg_data
    assert "edges" in kg_data
    assert kg_data["summary"]["total_triples"] >= 1

    # Check relation triple with session provenance
    rel_triple = next((t for t in kg_data["triples"] if t["predicate"] == "specifies"), None)
    assert rel_triple is not None
    assert rel_triple["subject"] == "CardGameDesign"
    assert rel_triple["object"] == "FlameKnightCard"
    assert rel_triple["provenance"]["source_session"] == session_id
    assert rel_triple["provenance"]["confidence"] == 0.98
    assert rel_triple["provenance"]["origin"] == "derived"

    # B. Unfiltered query includes parent_type (is_a) and concept graph
    kg_all = client.get("/api/knowledge-graph")
    assert kg_all.status_code == 200
    all_data = cast(dict[str, Any], kg_all.json())
    assert all_data["summary"]["total_triples"] >= 2

    isa_triple = next((t for t in all_data["triples"] if t["predicate"] == "is_a"), None)
    assert isa_triple is not None
    assert isa_triple["subject"] == "CardGameDesign"
    assert isa_triple["object"] == "Specification"

    # Clean up bus
    await bus.stop()
