"""Tests for DAG output inventory loading and validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent

import pytest

from codeintel.build.schemas import get_schema_provider
from codeintel.core.registry.service import DagOutputInventory, RegistryService
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)


def _write_inventory(path: Path, outputs_block: str) -> Path:
    content = f'version: 1\ngenerated_at: "2025-01-01"\noutputs:\n{outputs_block}'
    path.write_text(content, encoding="utf8")
    return path


def _single_output_block(*, materialization: str, extra: str = "") -> str:
    base = "\n".join(
        [
            '- target: "alpha"',
            '  domain: "analytics"',
            '  anchor: "t__alpha"',
            f'  materialization: "{materialization}"',
        ]
    )
    if extra:
        extra_block = indent(dedent(extra).strip("\n"), "  ")
        base = f"{base}\n{extra_block}"
    indented = indent(base, "  ")
    return f"{indented}\n"


def test_dag_output_inventory_loads_default() -> None:
    """Default inventory loads and includes pilot output."""
    inventory = RegistryService.load_dag_output_inventory()

    expect_equal(inventory.version, 1)
    expect_true(len(inventory.outputs) > 0)
    pilot = inventory.by_target("function_metrics")
    expect_true(pilot.pilot)


def test_dag_output_inventory_rejects_duplicate_targets(tmp_path: Path) -> None:
    """Duplicate targets in inventory should raise."""
    outputs = dedent(
        """
          - target: "alpha"
            domain: "analytics"
            anchor: "t__alpha"
            materialization: "table"
            table_keys:
              - "analytics.alpha"
          - target: "alpha"
            domain: "analytics"
            anchor: "t__alpha"
            materialization: "table"
            table_keys:
              - "analytics.alpha"
        """
    ).lstrip()
    path = _write_inventory(tmp_path / "inventory.yaml", outputs)

    with pytest.raises(ValueError, match="Duplicate output target"):
        DagOutputInventory.from_path(path)


def test_dag_output_inventory_rejects_invalid_materialization(tmp_path: Path) -> None:
    """Invalid materialization kinds should raise."""
    outputs = _single_output_block(materialization="blob")
    path = _write_inventory(tmp_path / "inventory.yaml", outputs)

    with pytest.raises(ValueError, match="Unsupported materialization"):
        DagOutputInventory.from_path(path)


def test_dag_output_inventory_requires_table_keys_for_tables(tmp_path: Path) -> None:
    """Table outputs must declare table_keys."""
    outputs = _single_output_block(materialization="table")
    path = _write_inventory(tmp_path / "inventory.yaml", outputs)

    with pytest.raises(ValueError, match="must define table_keys"):
        DagOutputInventory.from_path(path)


def test_dag_output_inventory_rejects_table_keys_for_artifacts(tmp_path: Path) -> None:
    """Artifact outputs must not declare table_keys."""
    extra = dedent(
        """
            table_keys:
              - "analytics.alpha"
        """
    )
    outputs = _single_output_block(materialization="artifact", extra=extra)
    path = _write_inventory(tmp_path / "inventory.yaml", outputs)

    with pytest.raises(ValueError, match="must not define table_keys"):
        DagOutputInventory.from_path(path)


def test_dag_output_inventory_contracts_default_to_table_keys(tmp_path: Path) -> None:
    """Missing contracts should default to table_keys."""
    extra = dedent(
        """
            table_keys:
              - "analytics.alpha"
        """
    )
    outputs = _single_output_block(materialization="table", extra=extra)
    path = _write_inventory(tmp_path / "inventory.yaml", outputs)

    inventory = DagOutputInventory.from_path(path)
    spec = inventory.by_target("alpha")

    expect_equal(spec.contracts, ("analytics.alpha",))


def test_dag_output_inventory_table_keys_have_schemas() -> None:
    """Inventory table_keys should resolve to schema provider entries."""
    inventory = RegistryService.load_dag_output_inventory()
    provider = get_schema_provider()

    missing = [
        table_key
        for spec in inventory.outputs
        for table_key in spec.table_keys
        if provider.get_table_schema(table_key) is None
    ]

    expect_true(not missing, message=f"Missing table schemas: {missing}")
