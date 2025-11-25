"""Smoke test for exporting datasets to Document Output."""

from __future__ import annotations

from pathlib import Path

from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fixtures import seed_docs_export_minimal


def test_export_all_writes_expected_files(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """
    Seed a minimal DB and verify Parquet/JSONL exports are produced.

    This ensures the export mappings are usable end-to-end.

    Raises
    ------
    AssertionError
        If any expected export is missing after running both exporters.
    """
    seed_docs_export_minimal(fresh_gateway, repo="r", commit="c")

    output_dir = tmp_path / "Document Output"
    export_all_parquet(fresh_gateway, output_dir)
    export_all_jsonl(fresh_gateway, output_dir)

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
    fresh_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Ensure validation succeeds when provided with conforming exports."""
    seed_docs_export_minimal(fresh_gateway, repo="r", commit="c")

    output_dir = tmp_path / "Document Output"
    export_all_parquet(
        fresh_gateway,
        output_dir,
        validate_exports=True,
        schemas=["function_profile"],
    )
    export_all_jsonl(
        fresh_gateway,
        output_dir,
        validate_exports=True,
        schemas=["function_profile"],
    )
