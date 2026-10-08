"""Unit tests for ontology CLI commands (list, teach, review, forget, validate)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from typer.testing import CliRunner

from uclone_x.agent.clone_builder import clone_ontology
from uclone_x.agent.persona_store import DEFAULT_PERSONA_NAME
from uclone_x.cli.main import app
from uclone_x.core.agent_home import AgentHome, default_agents_root, resolve_handle
from uclone_x.ontology.engine import OntologyEngine

runner = CliRunner()


def test_cli_ontology_list_empty() -> None:
    with TemporaryDirectory() as tmp_dir:
        res = runner.invoke(app, ["ontology", "list", "--dir", tmp_dir, "--agent", "empty_agent"])
        assert res.exit_code == 0
        assert "No ontology elements found" in res.stdout


def test_cli_ontology_teach_and_list() -> None:
    with TemporaryDirectory() as tmp_dir:
        # 1. Teach concept
        res_c = runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "--concept",
                "Microservice",
                "--attributes",
                "name:str,port:int",
                "--required",
                "name",
                "--dir",
                tmp_dir,
                "--agent",
                "test_agent",
            ],
        )
        assert res_c.exit_code == 0
        assert "Taught asserted element" in res_c.stdout

        # 2. Teach relation
        res_r = runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "--relation",
                "Microservice:calls:Database",
                "--dir",
                tmp_dir,
                "--agent",
                "test_agent",
            ],
        )
        assert res_r.exit_code == 0
        assert "Taught asserted relation" in res_r.stdout

        # 3. Teach axiom
        res_a = runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "--axiom",
                "PortBound:Microservice:port:8080",
                "--dir",
                tmp_dir,
                "--agent",
                "test_agent",
            ],
        )
        assert res_a.exit_code == 0
        assert "Taught asserted axiom" in res_a.stdout

        # 4. List all
        res_l = runner.invoke(
            app,
            ["ontology", "list", "--dir", tmp_dir, "--agent", "test_agent"],
        )
        assert res_l.exit_code == 0
        assert "Microservice" in res_l.stdout
        assert "calls" in res_l.stdout
        assert "PortBound" in res_l.stdout

        # 5. List with filter
        res_filtered = runner.invoke(
            app,
            ["ontology", "list", "--tier", "asserted", "--dir", tmp_dir, "--agent", "test_agent"],
        )
        assert res_filtered.exit_code == 0
        assert "Microservice" in res_filtered.stdout

        # 6. List with hyphenated filter
        res_hyphen = runner.invoke(
            app,
            [
                "ontology",
                "list",
                "--tier",
                "induced-candidate",
                "--dir",
                tmp_dir,
                "--agent",
                "test_agent",
            ],
        )
        assert res_hyphen.exit_code == 0
        assert "No ontology elements found" in res_hyphen.stdout

        # 7. List with invalid filter
        res_invalid_tier = runner.invoke(
            app,
            [
                "ontology",
                "list",
                "--tier",
                "invalid_tier",
                "--dir",
                tmp_dir,
                "--agent",
                "test_agent",
            ],
        )
        assert res_invalid_tier.exit_code == 1
        assert "Invalid tier" in res_invalid_tier.stdout

        # 8. List with arbitrary unrecognised tier
        res_arbitrary_tier = runner.invoke(
            app,
            [
                "ontology",
                "list",
                "--tier",
                "arbitrary_tier",
                "--dir",
                tmp_dir,
                "--agent",
                "test_agent",
            ],
        )
        assert res_arbitrary_tier.exit_code == 1
        assert "Invalid tier" in res_arbitrary_tier.stdout


def test_cli_ontology_review_and_promote() -> None:
    with TemporaryDirectory() as tmp_dir:
        # Seed an engine with candidates in YAML
        file_path = Path(tmp_dir) / "review_agent.yaml"
        engine_content = {
            "agent_id": "review_agent",
            "version": 1,
            "concepts": [
                {
                    "name": "CandidateWorker",
                    "tier": "induced_candidate",
                    "attributes": {"id": "str"},
                    "required_fields": ["id"],
                    "evidence": {
                        "observation_count": 2,
                        "first_seen": "2026-09-02T10:00:00Z",
                        "last_seen": "2026-09-02T12:00:00Z",
                    },
                }
            ],
            "relations": [],
            "axioms": [],
        }
        import yaml

        file_path.write_text(yaml.dump(engine_content), encoding="utf-8")

        # 1. Review list
        res_rev = runner.invoke(
            app,
            ["ontology", "review", "--dir", tmp_dir, "--agent", "review_agent"],
        )
        assert res_rev.exit_code == 0
        assert "CandidateWorker" in res_rev.stdout

        # 2. Promote without force (<5 obs) fails
        res_prom_fail = runner.invoke(
            app,
            [
                "ontology",
                "review",
                "--promote",
                "CandidateWorker",
                "--dir",
                tmp_dir,
                "--agent",
                "review_agent",
            ],
        )
        assert res_prom_fail.exit_code == 1
        assert "Promotion rejected" in res_prom_fail.stdout

        # 3. Promote with force succeeds
        res_prom = runner.invoke(
            app,
            [
                "ontology",
                "review",
                "--promote",
                "CandidateWorker",
                "--force",
                "--dir",
                tmp_dir,
                "--agent",
                "review_agent",
            ],
        )
        assert res_prom.exit_code == 0
        assert "Promoted candidate" in res_prom.stdout

        # 4. Demote term
        res_dem = runner.invoke(
            app,
            [
                "ontology",
                "review",
                "--demote",
                "CandidateWorker",
                "--dir",
                tmp_dir,
                "--agent",
                "review_agent",
            ],
        )
        assert res_dem.exit_code == 0
        assert "Demoted element" in res_dem.stdout


def test_cli_ontology_forget() -> None:
    with TemporaryDirectory() as tmp_dir:
        # Teach concept and dependent child
        runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "--concept",
                "ParentSvc",
                "--dir",
                tmp_dir,
                "--agent",
                "forget_agent",
            ],
        )
        runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "--concept",
                "ChildSvc",
                "--parent",
                "ParentSvc",
                "--dir",
                tmp_dir,
                "--agent",
                "forget_agent",
            ],
        )

        # 1. Retract blocked by dependent
        res_block = runner.invoke(
            app,
            ["ontology", "forget", "ParentSvc", "--dir", tmp_dir, "--agent", "forget_agent"],
        )
        assert res_block.exit_code == 1
        assert "Retraction blocked" in res_block.stdout

        # 2. Forced retract cascades
        res_force = runner.invoke(
            app,
            [
                "ontology",
                "forget",
                "ParentSvc",
                "--force",
                "--dir",
                tmp_dir,
                "--agent",
                "forget_agent",
            ],
        )
        assert res_force.exit_code == 0
        assert "Retracted ontology element(s)" in res_force.stdout


def test_cli_ontology_validate() -> None:
    with TemporaryDirectory() as tmp_dir:
        runner.invoke(
            app,
            [
                "ontology",
                "teach",
                "--concept",
                "Order",
                "--attributes",
                "order_id:str,amount:float",
                "--required",
                "order_id,amount",
                "--dir",
                tmp_dir,
                "--agent",
                "val_agent",
            ],
        )

        # 1. Valid data via --data
        valid_json = json.dumps({"order_id": "ord-123", "amount": 99.5})
        res_val = runner.invoke(
            app,
            [
                "ontology",
                "validate",
                "Order",
                "--data",
                valid_json,
                "--dir",
                tmp_dir,
                "--agent",
                "val_agent",
            ],
        )
        assert res_val.exit_code == 0
        assert "PASSED" in res_val.stdout

        # 2. Invalid data via --data (missing required field)
        invalid_json = json.dumps({"order_id": "ord-123"})
        res_inval = runner.invoke(
            app,
            [
                "ontology",
                "validate",
                "Order",
                "--data",
                invalid_json,
                "--dir",
                tmp_dir,
                "--agent",
                "val_agent",
            ],
        )
        assert res_inval.exit_code == 1
        assert "FAILED" in res_inval.stdout

        # 3. Valid data via --file
        file_path = Path(tmp_dir) / "valid_order.json"
        file_path.write_text(valid_json, encoding="utf-8")
        res_file = runner.invoke(
            app,
            [
                "ontology",
                "validate",
                "Order",
                "--file",
                str(file_path),
                "--dir",
                tmp_dir,
                "--agent",
                "val_agent",
            ],
        )
        assert res_file.exit_code == 0
        assert "PASSED" in res_file.stdout

        # 4. Bad JSON format
        res_bad_json = runner.invoke(
            app,
            [
                "ontology",
                "validate",
                "Order",
                "--data",
                "not-json",
                "--dir",
                tmp_dir,
                "--agent",
                "val_agent",
            ],
        )
        assert res_bad_json.exit_code == 1


def test_cli_ontology_list_with_hyphenated_candidate_filter() -> None:
    with TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "hyphen_agent.yaml"
        engine_content = {
            "agent_id": "hyphen_agent",
            "version": 1,
            "concepts": [
                {
                    "name": "HyphenConcept",
                    "tier": "induced-candidate",
                    "attributes": {"id": "str"},
                    "required_fields": ["id"],
                }
            ],
            "relations": [],
            "axioms": [],
        }
        import yaml

        file_path.write_text(yaml.dump(engine_content), encoding="utf-8")

        res = runner.invoke(
            app,
            [
                "ontology",
                "list",
                "--tier",
                "induced-candidate",
                "--dir",
                tmp_dir,
                "--agent",
                "hyphen_agent",
            ],
        )
        assert res.exit_code == 0
        assert "HyphenConcept" in res.stdout


def test_teach_without_dir_writes_the_default_clones_own_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ucx ontology teach` with no `--dir` writes the builtin clone's `ontology.yaml`, and
    the engine every head composes that clone with holds what was taught (#1817).

    Killed by: src/uclone_x/agent/clone_builder.py :: engine.load_from_yaml(path)
    Becomes: pass
    """
    monkeypatch.chdir(tmp_path)  # not a repository: no `ontology/` to import
    res = runner.invoke(app, ["ontology", "teach", "--axiom", "PortBound:Microservice:port:8080"])
    assert "Taught asserted axiom" in res.stdout, res.stdout

    clone_id = resolve_handle(DEFAULT_PERSONA_NAME)
    assert AgentHome.for_clone(clone_id).ontology_path.is_file()
    assert not (tmp_path / "ontology").exists()
    assert clone_ontology(clone_id).get_axiom("PortBound") is not None


def test_a_repository_ontology_file_is_merged_into_the_clone_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clone-data-scopes §3.8 step 5: `<repo>/ontology/<handle>.yaml` is merged into that
    clone's rules once per file digest; an axiom the clone holds differently is reported
    and the clone's own is kept.

    Killed by: src/uclone_x/agent/clone_builder.py :: if own_axiom is None:
    Becomes: if True:

    Killed by: src/uclone_x/agent/clone_builder.py :: if read_imports(folder).get(source_key) == digest:
    Becomes: if False:
    """
    monkeypatch.chdir(tmp_path)
    taught = runner.invoke(app, ["ontology", "teach", "--axiom", "Shared:Service:port:1"])
    assert "Taught asserted axiom" in taught.stdout, taught.stdout

    old = OntologyEngine()
    old.teach_axiom(name="Shared", subject_entity="Service", predicate="port", object_value="2")
    old.teach_axiom(name="FromRepo", subject_entity="Service", predicate="tier", object_value="db")
    old.save_to_yaml(tmp_path / "ontology" / f"{DEFAULT_PERSONA_NAME}.yaml")

    for _ in range(2):
        runner.invoke(app, ["ontology", "list"])

    clone_id = resolve_handle(DEFAULT_PERSONA_NAME)
    rules = clone_ontology(clone_id)
    assert rules.get_axiom("FromRepo") is not None
    shared = rules.get_axiom("Shared")
    assert shared is not None and shared.object_value == "1"

    log = "".join(p.read_text() for p in default_agents_root().glob(".migration-*.log"))
    imported = [line for line in log.splitlines() if "rule(s) from" in line]
    assert len(imported) == 1, log
    assert "axiom Shared" in imported[0]


_CORRUPT = "concepts: [CONTENT-TOKEN-2134 unclosed\n"


def test_a_corrupt_clone_ontology_file_starts_the_clone_with_no_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A hand-corrupted `agents/<id>/ontology.yaml` is logged and the clone's engine starts
    empty, as the import treats the same file; composing the clone never raises (#2134).
    The warning names the file and the error type, never the file's contents (#2136).

    Killed by: src/uclone_x/agent/clone_builder.py :: except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
    Becomes: except () as exc:

    Killed by: src/uclone_x/agent/clone_builder.py :: "%s was not loaded, so the clone has no rules: %s", path, type(exc).__name__
    Becomes: "%s was not loaded, so the clone has no rules: %s", path, exc
    """
    monkeypatch.chdir(tmp_path)
    taught = runner.invoke(app, ["ontology", "teach", "--axiom", "PortBound:Service:port:1"])
    assert "Taught asserted axiom" in taught.stdout, taught.stdout

    clone_id = resolve_handle(DEFAULT_PERSONA_NAME)
    path = AgentHome.for_clone(clone_id).ontology_path
    path.write_text(_CORRUPT, encoding="utf-8")

    with caplog.at_level("WARNING", logger="uclone_x.agent.clone_builder"):
        rules = clone_ontology(clone_id)

    assert isinstance(rules, OntologyEngine)
    assert rules.list_axioms() == [] and rules.list_concepts() == []
    assert str(path) in caplog.text and "ParserError" in caplog.text
    assert "CONTENT-TOKEN-2134" not in caplog.text  # the file's contents stay out of the log
    assert path.read_text(encoding="utf-8") == _CORRUPT  # never rewritten


def test_an_unknown_agent_is_refused_and_gets_no_clone_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ucx ontology --agent <name no clone carries>` is refused with a plain message and
    no clone directory is created for the name (#2134, #1817 review).

    Killed by: src/uclone_x/cli/commands/ontology.py :: agent_id = clone_id_of(agent_name)
    Becomes: agent_id = agent_name
    """
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["ontology", "list"])  # brings the clone store up first
    homes = sorted(p.name for p in default_agents_root().iterdir())

    for name in ("nobody", "../../etc"):
        res = runner.invoke(app, ["ontology", "list", "--agent", name])
        assert isinstance(res.exception, SystemExit) and res.exit_code == 1  # not a crash
        assert "✖" in res.stdout and f"'{name}'" in res.stdout, res.stdout

    assert sorted(p.name for p in default_agents_root().iterdir()) == homes
