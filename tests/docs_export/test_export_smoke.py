"""Smoke test for exporting datasets to Document Output."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from tests._helpers.fixtures import ProvisionedGateway, provision_docs_export_ready


@pytest.fixture
def docs_export_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision docs-export-ready gateway and ensure cleanup.

    Yields
    ------
    ProvisionedGateway
        Provisioned gateway seeded for docs export.
    """
    ctx = provision_docs_export_ready(tmp_path, repo="r", commit="c", file_backed=False)
    try:
        yield ctx
    finally:
        ctx.close()


def test_export_all_writes_expected_files(
    docs_export_gateway: ProvisionedGateway, tmp_path: Path
) -> None:
    """
    Seed a minimal DB and verify Parquet/JSONL exports are produced.

    This ensures the export mappings are usable end-to-end.

    Raises
    ------
    AssertionError
        If any expected export is missing after running both exporters.
    """
    output_dir = tmp_path / "Document Output"
    export_all_parquet(docs_export_gateway.gateway, output_dir)
    export_all_jsonl(docs_export_gateway.gateway, output_dir)

    expected_basenames = {
        "goids.parquet",
        "goid_crosswalk.parquet",
        "call_graph_nodes.parquet",
        "call_graph_edges.parquet",
        "cfg_blocks.parquet",
        "import_graph_edges.parquet",
        "docstrings.parquet",
        "function_metrics.parquet",
        "function_types.parquet",
        "coverage_functions.parquet",
        "test_catalog.parquet",
        "test_coverage_edges.parquet",
        "goid_risk_factors.parquet",
        "goids.jsonl",
        "goid_crosswalk.jsonl",
        "call_graph_nodes.jsonl",
        "call_graph_edges.jsonl",
        "cfg_blocks.jsonl",
        "import_graph_edges.jsonl",
        "docstrings.jsonl",
        "function_metrics.jsonl",
        "function_types.jsonl",
        "coverage_functions.jsonl",
        "test_catalog.jsonl",
        "test_coverage_edges.jsonl",
        "goid_risk_factors.jsonl",
        "repo_map.json",
        "index.json",
    }

    written = {p.name for p in output_dir.iterdir() if p.is_file()}

    missing = expected_basenames - written
    if missing:
        message = f"Expected exports missing: {sorted(missing)}"
        raise AssertionError(message)


def test_export_validation_passes_on_minimal_data(
    docs_export_gateway: ProvisionedGateway, tmp_path: Path
) -> None:
    """Ensure validation succeeds when provided with conforming exports."""
    output_dir = tmp_path / "Document Output"
    export_all_parquet(
        docs_export_gateway.gateway,
        output_dir,
        validate_exports=True,
        schemas=["function_profile"],
    )
    export_all_jsonl(
        docs_export_gateway.gateway,
        output_dir,
        validate_exports=True,
        schemas=["function_profile"],
    )
