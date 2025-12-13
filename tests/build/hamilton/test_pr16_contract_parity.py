"""Tests for PR-16: Complete contract parity across all targets.

This module tests that:
1. All contract table_keys exist in SCHEMA_REGISTRY
2. Targets with plugins have contracts
3. Artifact templates are renderable
4. Real graph passes validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.contracts_validation import validate_contracts
from codeintel.build.registry import get_target_graph
from codeintel.config.datasets.contracts import get_table_schemas

if TYPE_CHECKING:
    from pathlib import Path


def test_all_contract_tables_in_schema_registry() -> None:
    """Verify every contract table_key exists in SCHEMA_REGISTRY."""
    graph = get_target_graph()
    table_schemas = get_table_schemas()

    missing_tables: list[tuple[str, str]] = [
        (target.name, table_key)
        for target in graph.all_targets
        for table_key in target.contract.table_keys
        if table_key not in table_schemas
    ]

    if missing_tables:
        table_list = "\n".join(f"  {target}: {table}" for target, table in missing_tables)
        pytest.fail(
            f"Found {len(missing_tables)} table_keys missing from SCHEMA_REGISTRY:\n{table_list}"
        )


def test_no_empty_contracts_for_output_targets() -> None:
    """Verify targets with plugins have non-empty contracts."""
    graph = get_target_graph()

    empty_contracts = [
        target.name
        for target in graph.all_targets
        if target.plugin and not target.contract.tables and not target.contract.artifacts
    ]

    if empty_contracts:
        contract_list = "\n".join([f"  {name}" for name in empty_contracts])
        pytest.fail(
            f"Found {len(empty_contracts)} targets with plugins but empty contracts:\n"
            f"{contract_list}"
        )


def test_artifact_templates_renderable(tmp_path: Path) -> None:
    """Verify artifact path templates render without KeyError."""
    graph = get_target_graph()
    errors = validate_contracts(graph, repo_root=tmp_path)

    template_errors = [e for e in errors if "invalid template" in e]
    if template_errors:
        error_list = "\n".join(f"  {err}" for err in template_errors)
        pytest.fail(f"Found {len(template_errors)} artifact template errors:\n{error_list}")


def test_validate_contracts_returns_empty(tmp_path: Path) -> None:
    """Verify real graph passes all validation checks."""
    graph = get_target_graph()
    errors = validate_contracts(graph, repo_root=tmp_path)

    if errors:
        error_list = "\n".join(f"  {err}" for err in errors)
        pytest.fail(f"Contract validation failed with {len(errors)} errors:\n{error_list}")


def test_key_targets_have_contracts() -> None:
    """Verify critical targets mentioned in plan have complete contracts."""
    graph = get_target_graph()

    # Targets explicitly mentioned in PR-16 plan
    critical_targets = {
        "modules": ["core.modules", "core.file_state", "core.repo_map"],
        "typing": ["analytics.typedness", "analytics.static_diagnostics"],
        "tests_ingest": ["analytics.test_catalog"],
        "coverage_ingest": ["analytics.coverage_lines"],
        "function_metrics": ["analytics.function_metrics", "analytics.function_types"],
        "risk_factors": ["analytics.goid_risk_factors"],
    }

    for target_name, expected_tables in critical_targets.items():
        target = graph.get(target_name)
        actual_tables = list(target.table_keys)

        # Check that all expected tables are present
        for table_key in expected_tables:
            if table_key not in actual_tables:
                pytest.fail(f"Target '{target_name}' missing expected table_key: '{table_key}'")


@pytest.mark.parametrize(
    "target_name",
    ["export_jsonl", "export_parquet"],
)
def test_export_targets_have_artifacts(target_name: str) -> None:
    """Verify export targets have artifact contracts."""
    graph = get_target_graph()
    target = graph.get(target_name)

    if not target.contract.artifacts:
        pytest.fail(f"Export target '{target_name}' should have artifact specs")
    if len(target.contract.artifacts) == 0:
        pytest.fail(f"Export target '{target_name}' should have at least one artifact spec")


def test_scip_target_has_artifacts() -> None:
    """Verify SCIP target has artifact contracts for index files."""
    graph = get_target_graph()
    scip = graph.get("scip")

    if not scip.contract.artifacts:
        pytest.fail("SCIP target should have artifact specs")
    artifact_names = {a.name for a in scip.contract.artifacts}
    if "scip_index" not in artifact_names:
        pytest.fail("SCIP target should have scip_index artifact spec")
    if "scip_json" not in artifact_names:
        pytest.fail("SCIP target should have scip_json artifact spec")
