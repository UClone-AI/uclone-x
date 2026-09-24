"""Unit tests for ontology validation: subclass required fields inheritance and AST axiom evaluation (Issue #143)."""

from typing import Any

import pytest

from uclone_x.ontology.engine import (
    ExpressionEvaluationError,
    OntologyEngine,
    safe_eval_rule_expression,
)
from uclone_x.ontology.models import OntologyTier


class TestSubclassRequiredFieldsInheritance:
    """Tests that validate_entity enforces required fields across 3+ levels of inheritance."""

    def test_three_level_subclass_required_fields_inheritance(self) -> None:
        """Item(id) -> Task(priority) -> ReviewTask(reviewer). Missing top-level id must fail."""
        engine = OntologyEngine(agent_id="test_subclass")

        # Level 1: Base Item
        engine.teach_concept(
            name="Item",
            attributes={"id": "str", "name": "str"},
            required_fields=["id"],
        )

        # Level 2: Task extending Item
        engine.teach_concept(
            name="Task",
            parent_type="Item",
            attributes={"priority": "int", "description": "str"},
            required_fields=["priority"],
        )

        # Level 3: ReviewTask extending Task
        engine.teach_concept(
            name="ReviewTask",
            parent_type="Task",
            attributes={"reviewer": "str"},
            required_fields=["reviewer"],
        )

        # 1. Missing top-level 'id' -> must FAIL validation
        res_missing_id = engine.validate_entity(
            "ReviewTask",
            {"priority": 1, "reviewer": "alice", "description": "code review"},
        )
        assert res_missing_id.is_valid is False
        assert any("Missing required field 'id'" in err for err in res_missing_id.errors)

        # 2. Missing mid-level 'priority' -> must FAIL validation
        res_missing_priority = engine.validate_entity(
            "ReviewTask",
            {"id": "item-101", "reviewer": "alice"},
        )
        assert res_missing_priority.is_valid is False
        assert any(
            "Missing required field 'priority'" in err for err in res_missing_priority.errors
        )

        # 3. Missing leaf 'reviewer' -> must FAIL validation
        res_missing_reviewer = engine.validate_entity(
            "ReviewTask",
            {"id": "item-101", "priority": 1},
        )
        assert res_missing_reviewer.is_valid is False
        assert any(
            "Missing required field 'reviewer'" in err for err in res_missing_reviewer.errors
        )

        # 4. All required fields present -> must PASS validation
        res_valid = engine.validate_entity(
            "ReviewTask",
            {"id": "item-101", "priority": 1, "reviewer": "alice"},
        )
        assert res_valid.is_valid is True
        assert len(res_valid.errors) == 0
        assert res_valid.matched_tier == OntologyTier.ASSERTED

    def test_inherited_attribute_type_validation(self) -> None:
        """Inherited attributes from parent concepts must have their types validated."""
        engine = OntologyEngine()
        engine.teach_concept(
            name="BaseEntity",
            attributes={"count": "int"},
            required_fields=["count"],
        )
        engine.teach_concept(
            name="ChildEntity",
            parent_type="BaseEntity",
            attributes={"label": "str"},
            required_fields=["label"],
        )

        # 'count' is defined on BaseEntity as int; passing string should fail
        res = engine.validate_entity(
            "ChildEntity",
            {"count": "not_an_int", "label": "test"},
        )
        assert res.is_valid is False
        assert any("expected int" in err for err in res.errors)

    def test_parent_type_circular_reference_prevention(self) -> None:
        """Circular parent_type reference does not cause infinite recursion."""
        engine = OntologyEngine()
        engine.teach_concept(name="A", parent_type="B", required_fields=["a_field"])
        engine.teach_concept(name="B", parent_type="A", required_fields=["b_field"])

        res = engine.validate_entity("A", {"a_field": "val_a", "b_field": "val_b"})
        assert res.is_valid is True


class TestAxiomRuleExpressionEvaluation:
    """Tests for safe AST axiom rule expression evaluation in validate_entity."""

    def test_numeric_comparison_axiom(self) -> None:
        """Numeric comparison axioms like balance >= 0 reject negative balances."""
        engine = OntologyEngine()
        engine.teach_concept(
            name="Account",
            attributes={"account_id": "str", "balance": "float"},
            required_fields=["account_id", "balance"],
        )
        engine.teach_axiom(
            name="NonNegativeBalance",
            subject_entity="Account",
            rule_expression="balance >= 0",
            description="Account balance cannot be negative",
        )

        # 1. Negative balance fails validation
        res_fail = engine.validate_entity(
            "Account",
            {"account_id": "acc-001", "balance": -100.50},
        )
        assert res_fail.is_valid is False
        assert any("NonNegativeBalance" in err and "balance >= 0" in err for err in res_fail.errors)

        # 2. Zero balance passes
        res_zero = engine.validate_entity(
            "Account",
            {"account_id": "acc-001", "balance": 0.0},
        )
        assert res_zero.is_valid is True
        assert len(res_zero.errors) == 0

        # 3. Positive balance passes
        res_pos = engine.validate_entity(
            "Account",
            {"account_id": "acc-001", "balance": 500.0},
        )
        assert res_pos.is_valid is True

    def test_membership_and_logical_axiom(self) -> None:
        """Membership tests (status in ('open', 'closed')) and logical expressions."""
        engine = OntologyEngine()
        engine.teach_concept(
            name="Ticket",
            attributes={"ticket_id": "str", "status": "str", "priority": "int"},
            required_fields=["ticket_id", "status"],
        )
        engine.teach_axiom(
            name="ValidTicketStatus",
            subject_entity="Ticket",
            rule_expression="status in ('open', 'in_review', 'closed')",
        )
        engine.teach_axiom(
            name="PriorityRange",
            subject_entity="Ticket",
            rule_expression="priority >= 1 and priority <= 5",
        )

        # 1. Invalid status fails
        res_bad_status = engine.validate_entity(
            "Ticket",
            {"ticket_id": "t-1", "status": "unknown_status", "priority": 2},
        )
        assert res_bad_status.is_valid is False
        assert any("ValidTicketStatus" in err for err in res_bad_status.errors)

        # 2. Invalid priority fails
        res_bad_priority = engine.validate_entity(
            "Ticket",
            {"ticket_id": "t-1", "status": "open", "priority": 10},
        )
        assert res_bad_priority.is_valid is False
        assert any("PriorityRange" in err for err in res_bad_priority.errors)

        # 3. Valid ticket passes
        res_valid = engine.validate_entity(
            "Ticket",
            {"ticket_id": "t-1", "status": "in_review", "priority": 3},
        )
        assert res_valid.is_valid is True

    def test_subclass_inherits_ancestor_axioms(self) -> None:
        """Subclasses inherit axioms attached to ancestor concepts."""
        engine = OntologyEngine()
        engine.teach_concept(
            name="BaseResource",
            attributes={"resource_id": "str", "quota": "int"},
            required_fields=["resource_id", "quota"],
        )
        engine.teach_axiom(
            name="PositiveQuota",
            subject_entity="BaseResource",
            rule_expression="quota > 0",
        )

        engine.teach_concept(
            name="ComputeResource",
            parent_type="BaseResource",
            attributes={"cpu_cores": "int"},
            required_fields=["cpu_cores"],
        )
        engine.teach_axiom(
            name="MinimumCpu",
            subject_entity="ComputeResource",
            rule_expression="cpu_cores >= 2",
        )

        # Fails parent axiom (quota <= 0)
        res_quota_fail = engine.validate_entity(
            "ComputeResource",
            {"resource_id": "r-1", "quota": 0, "cpu_cores": 4},
        )
        assert res_quota_fail.is_valid is False
        assert any("PositiveQuota" in err for err in res_quota_fail.errors)

        # Fails child axiom (cpu_cores < 2)
        res_cpu_fail = engine.validate_entity(
            "ComputeResource",
            {"resource_id": "r-1", "quota": 10, "cpu_cores": 1},
        )
        assert res_cpu_fail.is_valid is False
        assert any("MinimumCpu" in err for err in res_cpu_fail.errors)

        # Passes both parent and child axioms
        res_ok = engine.validate_entity(
            "ComputeResource",
            {"resource_id": "r-1", "quota": 10, "cpu_cores": 4},
        )
        assert res_ok.is_valid is True

    def test_candidate_axiom_is_not_enforced(self) -> None:
        """Candidate tier axioms are non-enforcing in validate_entity."""
        engine = OntologyEngine()
        engine.teach_concept(name="Item", attributes={"val": "int"}, required_fields=["val"])
        engine.induce_axiom(
            name="CandidateRule",
            subject_entity="Item",
            rule_expression="val > 100",
        )

        res = engine.validate_entity("Item", {"val": 10})
        assert res.is_valid is True  # Candidate axiom must not fail validation


class TestSafeAstEvaluator:
    """Direct tests for safe_eval_rule_expression sandbox security and functionality."""

    def test_safe_operators_and_arithmetic(self) -> None:
        """Arithmetic and comparisons evaluate safely."""
        ctx: dict[str, Any] = {"x": 10, "y": 20, "z": 5}
        assert safe_eval_rule_expression("x + y == 30", ctx) is True
        assert safe_eval_rule_expression("x * 2 == y", ctx) is True
        assert safe_eval_rule_expression("y / z == 4.0", ctx) is True
        assert safe_eval_rule_expression("y // z == 4", ctx) is True
        assert safe_eval_rule_expression("x % 3 == 1", ctx) is True
        assert safe_eval_rule_expression("z ** 2 == 25", ctx) is True
        assert safe_eval_rule_expression("x < y and y > z", ctx) is True
        assert safe_eval_rule_expression("not (x > y)", ctx) is True
        assert safe_eval_rule_expression("-x == -10", ctx) is True
        assert safe_eval_rule_expression("+x == 10", ctx) is True

    def test_subscript_and_containers(self) -> None:
        """Subscript lookups into lists and dicts work properly."""
        ctx: dict[str, Any] = {
            "tags": ["prod", "db"],
            "meta": {"env": "production", "active": True},
        }
        assert safe_eval_rule_expression("tags[0] == 'prod'", ctx) is True
        assert safe_eval_rule_expression("'db' in tags", ctx) is True
        assert safe_eval_rule_expression("meta['env'] == 'production'", ctx) is True
        assert safe_eval_rule_expression("meta['active'] is True", ctx) is True

    def test_chained_comparisons(self) -> None:
        """Chained comparisons (e.g. 0 <= x <= 100) evaluate accurately."""
        ctx = {"score": 85}
        assert safe_eval_rule_expression("0 <= score <= 100", ctx) is True
        assert safe_eval_rule_expression("90 <= score <= 100", ctx) is False

    def test_rejects_function_and_method_calls(self) -> None:
        """AST evaluator strictly rejects function calls."""
        with pytest.raises(ExpressionEvaluationError, match="Disallowed or unsupported"):
            safe_eval_rule_expression("len(tags) > 0", {"tags": [1, 2]})

        with pytest.raises(ExpressionEvaluationError, match="Disallowed or unsupported"):
            safe_eval_rule_expression("print('hello')", {})

        with pytest.raises(ExpressionEvaluationError, match="Disallowed or unsupported"):
            safe_eval_expression = safe_eval_rule_expression
            safe_eval_expression("__import__('os').system('ls')", {})

    def test_rejects_attribute_access(self) -> None:
        """AST evaluator strictly rejects attribute traversal to prevent sandbox escape."""
        with pytest.raises(ExpressionEvaluationError, match="Disallowed or unsupported"):
            safe_eval_rule_expression("obj.__class__.__name__ == 'str'", {"obj": "text"})

    def test_division_by_zero_handling(self) -> None:
        """Division by zero raises ExpressionEvaluationError."""
        with pytest.raises(ExpressionEvaluationError, match="Division by zero"):
            safe_eval_rule_expression("x / 0 == 0", {"x": 10})

        with pytest.raises(ExpressionEvaluationError, match="Modulo by zero"):
            safe_eval_rule_expression("x % 0 == 0", {"x": 10})

    def test_missing_identifier_error(self) -> None:
        """Referencing an identifier not in payload raises ExpressionEvaluationError."""
        with pytest.raises(ExpressionEvaluationError, match="Identifier 'missing_var' not found"):
            safe_eval_rule_expression("missing_var > 0", {})

    def test_syntax_error_handling(self) -> None:
        """Invalid Python syntax raises ExpressionEvaluationError."""
        with pytest.raises(ExpressionEvaluationError, match="Invalid syntax"):
            safe_eval_rule_expression("balance >=", {"balance": 10})

    def test_empty_expression_returns_true(self) -> None:
        """Empty or blank expressions safely return True."""
        assert safe_eval_rule_expression("", {}) is True
        assert safe_eval_rule_expression("   ", {}) is True
