"""Tests for PR-14: Graph exports (Mermaid and DOT).

Validates export_dag_mermaid and export_dag_dot functions.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.nodes.node_factory import clear_generated_module_cache
from codeintel.build.hamilton.observability import (
    export_dag_dot,
    export_dag_mermaid,
)


class TestMermaidExport:
    """Tests for Mermaid DAG export."""

    @staticmethod
    def test_export_dag_mermaid_returns_string() -> None:
        """Verify export_dag_mermaid returns a string."""
        clear_generated_module_cache()
        runtime = build_driver(mode="phase0")
        result = export_dag_mermaid(runtime, ["modules"])

        if not isinstance(result, str):
            pytest.fail(f"Expected string, got {type(result).__name__}")
        if not result:
            pytest.fail("Mermaid output should not be empty")

    @staticmethod
    def test_mermaid_output_starts_with_graph() -> None:
        """Verify Mermaid output starts with graph directive."""
        clear_generated_module_cache()
        runtime = build_driver(mode="phase0")
        result = export_dag_mermaid(runtime, ["modules"])

        if not result.strip().startswith(("graph", "flowchart")):
            pytest.fail(f"Mermaid should start with graph/flowchart: {result[:50]}")

    @staticmethod
    def test_mermaid_output_contains_target_nodes() -> None:
        """Verify Mermaid output contains target node references."""
        clear_generated_module_cache()
        runtime = build_driver(mode="phase0")
        result = export_dag_mermaid(runtime, ["modules"])

        if "modules" not in result.lower():
            pytest.fail("Mermaid output should reference modules target")


class TestDotExport:
    """Tests for Graphviz DOT export."""

    @staticmethod
    def test_export_dag_dot_returns_string() -> None:
        """Verify export_dag_dot returns a string."""
        clear_generated_module_cache()
        runtime = build_driver(mode="phase0")
        result = export_dag_dot(runtime, ["modules"])

        if not isinstance(result, str):
            pytest.fail(f"Expected string, got {type(result).__name__}")
        if not result:
            pytest.fail("DOT output should not be empty")

    @staticmethod
    def test_dot_output_is_valid_digraph() -> None:
        """Verify DOT output is a valid digraph structure."""
        clear_generated_module_cache()
        runtime = build_driver(mode="phase0")
        result = export_dag_dot(runtime, ["modules"])

        if not result.strip().startswith("digraph"):
            pytest.fail(f"DOT should start with digraph: {result[:50]}")

        if "{" not in result or "}" not in result:
            pytest.fail("DOT output should contain braces")

    @staticmethod
    def test_dot_output_contains_target_nodes() -> None:
        """Verify DOT output contains target node references."""
        clear_generated_module_cache()
        runtime = build_driver(mode="phase0")
        result = export_dag_dot(runtime, ["modules"])

        if "modules" not in result.lower():
            pytest.fail("DOT output should reference modules target")


class TestGraphExportConsistency:
    """Tests for consistency between export formats."""

    @staticmethod
    def test_mermaid_and_dot_have_same_nodes() -> None:
        """Verify Mermaid and DOT exports reference same targets."""
        clear_generated_module_cache()
        runtime = build_driver(mode="phase0")

        mermaid = export_dag_mermaid(runtime, ["modules", "scip"])
        dot = export_dag_dot(runtime, ["modules", "scip"])

        for target in ["modules", "scip"]:
            if target not in mermaid.lower():
                pytest.fail(f"Mermaid missing target: {target}")
            if target not in dot.lower():
                pytest.fail(f"DOT missing target: {target}")
