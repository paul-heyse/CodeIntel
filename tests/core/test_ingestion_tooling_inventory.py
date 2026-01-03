"""Tests for ingestion tooling inventory loading and validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent

import pytest

from codeintel.runtime.registry_service import IngestionToolInventory, RegistryService
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_true,
)


def _write_inventory(path: Path, tools_block: str) -> Path:
    content = f'version: 1\ngenerated_at: "2025-01-01"\ntools:\n{tools_block}'
    path.write_text(content, encoding="utf8")
    return path


def _single_tool_block(*, tool_name: str, kind: str, extra: str = "") -> str:
    base = "\n".join(
        [
            f'- tool_name: "{tool_name}"',
            f'  kind: "{kind}"',
        ]
    )
    if extra:
        extra_block = indent(dedent(extra).strip("\n"), "  ")
        base = f"{base}\n{extra_block}"
    indented = indent(base, "  ")
    return f"{indented}\n"


def test_ingestion_tooling_inventory_loads_default() -> None:
    """Default tooling inventory loads and includes scip-python."""
    inventory = RegistryService.load_ingestion_tooling_inventory()

    expect_equal(inventory.version, 1)
    expect_true(len(inventory.tools) > 0)
    scip_tool = inventory.by_tool_name("scip-python")
    expect_equal(scip_tool.tool_name, "scip-python")


def test_ingestion_tooling_inventory_rejects_duplicate_tools(tmp_path: Path) -> None:
    """Duplicate tool entries should raise."""
    tools = dedent(
        """
          - tool_name: "alpha"
            kind: "binary"
          - tool_name: "alpha"
            kind: "binary"
        """
    ).lstrip()
    path = _write_inventory(tmp_path / "tools.yaml", tools)

    with pytest.raises(ValueError, match="Duplicate tool"):
        IngestionToolInventory.from_path(path)


def test_ingestion_tooling_inventory_rejects_invalid_kind(tmp_path: Path) -> None:
    """Invalid tool kinds should raise."""
    tools = _single_tool_block(tool_name="alpha", kind="widget")
    path = _write_inventory(tmp_path / "tools.yaml", tools)

    with pytest.raises(ValueError, match="Unsupported tool kind"):
        IngestionToolInventory.from_path(path)
