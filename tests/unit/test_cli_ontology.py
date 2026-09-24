"""Unit tests for ontology CLI commands (list, teach, review, forget, validate)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from uclone_x.cli.main import app

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
