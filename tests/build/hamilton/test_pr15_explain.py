"""Tests for PR-15: Explain staleness.

Validates PlanEntry.explain_staleness() and explain_plan() functions.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.planner import (
    PlanEntry,
    StalenessExplanation,
)


class TestPlanEntryDepHashes:
    """Tests for PlanEntry dep_hashes fields."""

    @staticmethod
    def test_plan_entry_has_dep_hashes_field() -> None:
        """Verify PlanEntry has dep_hashes field."""
        entry = PlanEntry(
            target="b",
            node="t__b",
            module="graphs",
            status="compute",
            reason="hash_changed",
            input_hash="current",
            options_hash="opts",
            prior_input_hash="prior",
            dependencies=("a",),
            table_keys=(),
            dep_hashes={"a": "hash_a"},
        )
        if not entry.dep_hashes:
            pytest.fail("dep_hashes should be populated")

    @staticmethod
    def test_plan_entry_has_prior_dep_hashes_field() -> None:
        """Verify PlanEntry has prior_dep_hashes field."""
        entry = PlanEntry(
            target="b",
            node="t__b",
            module="graphs",
            status="compute",
            reason="hash_changed",
            input_hash="current",
            options_hash="opts",
            prior_input_hash="prior",
            dependencies=("a",),
            table_keys=(),
            dep_hashes={"a": "new_hash"},
            prior_dep_hashes={"a": "old_hash"},
        )
        if not entry.prior_dep_hashes:
            pytest.fail("prior_dep_hashes should be populated")

    @staticmethod
    def test_plan_entry_dep_hashes_default_empty() -> None:
        """Verify dep_hashes defaults to empty dict."""
        entry = PlanEntry(
            target="a",
            node="t__a",
            module="ingestion",
            status="compute",
            reason="no_manifest",
            input_hash=None,
            options_hash=None,
            prior_input_hash=None,
            dependencies=(),
            table_keys=(),
        )
        if entry.dep_hashes != {}:
            pytest.fail(f"dep_hashes should default to empty: {entry.dep_hashes}")
        if entry.prior_dep_hashes != {}:
            pytest.fail(f"prior_dep_hashes should default to empty: {entry.prior_dep_hashes}")


class TestStalenessExplanation:
    """Tests for StalenessExplanation dataclass."""

    @staticmethod
    def test_staleness_explanation_structure() -> None:
        """Verify StalenessExplanation has required fields."""
        explanation = StalenessExplanation(
            target="b",
            status="compute",
            reason="hash_changed",
            input_hash_current="current",
            input_hash_prior="prior",
            changed_deps=("a",),
            added_deps=(),
            removed_deps=(),
            dep_hashes={"a": "new"},
            prior_dep_hashes={"a": "old"},
        )
        if explanation.target != "b":
            pytest.fail("target not set correctly")
        if explanation.changed_deps != ("a",):
            pytest.fail("changed_deps not set correctly")

    @staticmethod
    def test_staleness_explanation_identifies_changed_deps() -> None:
        """Verify changed_deps correctly identifies changed dependencies."""
        explanation = StalenessExplanation(
            target="downstream",
            status="compute",
            reason="hash_changed",
            input_hash_current="new_hash",
            input_hash_prior="old_hash",
            changed_deps=("upstream1", "upstream2"),
            added_deps=(),
            removed_deps=(),
            dep_hashes={"upstream1": "v2", "upstream2": "v2"},
            prior_dep_hashes={"upstream1": "v1", "upstream2": "v1"},
        )
        if len(explanation.changed_deps) != 2:
            pytest.fail(f"Expected 2 changed deps, got {len(explanation.changed_deps)}")
        if "upstream1" not in explanation.changed_deps:
            pytest.fail("upstream1 should be in changed_deps")

    @staticmethod
    def test_staleness_explanation_identifies_added_deps() -> None:
        """Verify added_deps correctly identifies new dependencies."""
        explanation = StalenessExplanation(
            target="downstream",
            status="compute",
            reason="hash_changed",
            input_hash_current="new_hash",
            input_hash_prior="old_hash",
            changed_deps=(),
            added_deps=("new_dep",),
            removed_deps=(),
            dep_hashes={"existing": "v1", "new_dep": "v1"},
            prior_dep_hashes={"existing": "v1"},
        )
        if len(explanation.added_deps) != 1:
            pytest.fail(f"Expected 1 added dep, got {len(explanation.added_deps)}")
        if "new_dep" not in explanation.added_deps:
            pytest.fail("new_dep should be in added_deps")

    @staticmethod
    def test_staleness_explanation_identifies_removed_deps() -> None:
        """Verify removed_deps correctly identifies removed dependencies."""
        explanation = StalenessExplanation(
            target="downstream",
            status="compute",
            reason="hash_changed",
            input_hash_current="new_hash",
            input_hash_prior="old_hash",
            changed_deps=(),
            added_deps=(),
            removed_deps=("old_dep",),
            dep_hashes={"existing": "v1"},
            prior_dep_hashes={"existing": "v1", "old_dep": "v1"},
        )
        if len(explanation.removed_deps) != 1:
            pytest.fail(f"Expected 1 removed dep, got {len(explanation.removed_deps)}")
        if "old_dep" not in explanation.removed_deps:
            pytest.fail("old_dep should be in removed_deps")


class TestPlanEntryExplainStaleness:
    """Tests for PlanEntry.explain_staleness() method."""

    @staticmethod
    def test_explain_staleness_returns_explanation() -> None:
        """Verify explain_staleness returns StalenessExplanation."""
        entry = PlanEntry(
            target="b",
            node="t__b",
            module="graphs",
            status="compute",
            reason="hash_changed",
            input_hash="new_hash",
            options_hash="opts",
            prior_input_hash="old_hash",
            dependencies=("a",),
            table_keys=(),
            dep_hashes={"a": "v2"},
            prior_dep_hashes={"a": "v1"},
        )

        explanation = entry.explain_staleness()

        if not isinstance(explanation, StalenessExplanation):
            pytest.fail(f"Expected StalenessExplanation, got {type(explanation)}")
        if explanation.target != "b":
            pytest.fail("Explanation target incorrect")

    @staticmethod
    def test_explain_staleness_detects_changed_deps() -> None:
        """Verify explain_staleness correctly detects changed dependencies."""
        entry = PlanEntry(
            target="downstream",
            node="t__downstream",
            module="analytics",
            status="compute",
            reason="hash_changed",
            input_hash="new",
            options_hash="opts",
            prior_input_hash="old",
            dependencies=("a", "b"),
            table_keys=(),
            dep_hashes={"a": "v2", "b": "v1"},
            prior_dep_hashes={"a": "v1", "b": "v1"},
        )

        explanation = entry.explain_staleness()

        if "a" not in explanation.changed_deps:
            pytest.fail(f"'a' should be in changed_deps: {explanation.changed_deps}")
        if "b" in explanation.changed_deps:
            pytest.fail("'b' should not be in changed_deps (hash unchanged)")


class TestExplainPlan:
    """Tests for explain_plan() function."""

    @staticmethod
    def test_explain_plan_exists() -> None:
        """Verify explain_plan function is importable."""
        from codeintel.build.hamilton.planner import explain_plan

        if not callable(explain_plan):
            pytest.fail("explain_plan should be callable")

    @staticmethod
    def test_explain_plan_returns_explanations() -> None:
        """Verify explain_plan returns explanations for targets."""
        from codeintel.build.hamilton.planner import HamiltonBuildPlan, explain_plan

        entry = PlanEntry(
            target="a",
            node="t__a",
            module="ingestion",
            status="compute",
            reason="no_manifest",
            input_hash=None,
            options_hash=None,
            prior_input_hash=None,
            dependencies=(),
            table_keys=(),
        )

        plan = HamiltonBuildPlan(
            requested=("a",),
            closure=("a",),
            entries=(entry,),
        )

        explanations = explain_plan(plan)

        if len(explanations) != 1:
            pytest.fail(f"explain_plan should return 1 explanation, got {len(explanations)}")
        if explanations[0].target != "a":
            pytest.fail("Explanation target incorrect")
