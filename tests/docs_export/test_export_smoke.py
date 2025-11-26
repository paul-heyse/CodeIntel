"""Smoke test for exporting datasets to Document Output."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.docs_export.export_jsonl import export_all_jsonl, export_dataset_to_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet, export_dataset_to_parquet
from codeintel.storage.gateway import DatasetRegistry
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
        "datasets_manifest.json",
    }

    written = {p.name for p in output_dir.iterdir() if p.is_file()}

    missing = expected_basenames - written
    if missing:
        message = f"Expected exports missing: {sorted(missing)}"
        raise AssertionError(message)
    manifest = json.loads((output_dir / "datasets_manifest.json").read_text(encoding="utf-8"))
    dataset_entries = {entry["name"]: entry for entry in manifest.get("datasets", [])}
    if "function_metrics" not in dataset_entries:
        pytest.fail("function_metrics missing from dataset manifest")
    metrics_entry = dataset_entries["function_metrics"]
    if metrics_entry.get("jsonl") != "function_metrics.jsonl":
        pytest.fail(f"Unexpected manifest entry: {metrics_entry}")


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


def test_export_subset_by_dataset_name(
    docs_export_gateway: ProvisionedGateway, tmp_path: Path
) -> None:
    """Exports honor dataset-name selection using the registry."""
    output_dir = tmp_path / "Document Output"
    selected = ["function_metrics", "goids"]
    export_all_parquet(
        docs_export_gateway.gateway,
        output_dir,
        datasets=selected,
    )
    export_all_jsonl(
        docs_export_gateway.gateway,
        output_dir,
        datasets=selected,
    )

    written = {p.name for p in output_dir.iterdir() if p.is_file()}
    expected = {
        "function_metrics.parquet",
        "goids.parquet",
        "function_metrics.jsonl",
        "goids.jsonl",
        "repo_map.json",
        "index.json",
        "datasets_manifest.json",
    }
    if written != expected:
        message = f"Unexpected export set: missing {expected - written}, extra {written - expected}"
        pytest.fail(message)
    manifest = json.loads((output_dir / "datasets_manifest.json").read_text(encoding="utf-8"))
    selected_entries = {
        entry["name"] for entry in manifest.get("datasets", []) if entry.get("selected")
    }
    if set(selected) != selected_entries:
        pytest.fail(f"Manifest selected set mismatch: {selected_entries}")


def test_export_subset_validates_dataset_names(
    docs_export_gateway: ProvisionedGateway, tmp_path: Path
) -> None:
    """Dataset selection rejects unknown names."""
    output_dir = tmp_path / "Document Output"
    with pytest.raises(ValueError, match="Unknown dataset"):
        export_all_jsonl(
            docs_export_gateway.gateway,
            output_dir,
            datasets=["missing_dataset"],
        )


def test_export_helpers_resolve_dataset_names(
    docs_export_gateway: ProvisionedGateway, tmp_path: Path
) -> None:
    """Dataset-aware export helpers resolve registry names to filenames."""
    output_dir = tmp_path / "Document Output"
    jsonl_path = export_dataset_to_jsonl(
        docs_export_gateway.gateway, "function_metrics", output_dir
    )
    parquet_path = export_dataset_to_parquet(
        docs_export_gateway.gateway, "function_metrics", output_dir
    )
    if not jsonl_path.exists():
        message = f"JSONL export not written: {jsonl_path}"
        pytest.fail(message)
    if not parquet_path.exists():
        message = f"Parquet export not written: {parquet_path}"
        pytest.fail(message)
    if jsonl_path.name != "function_metrics.jsonl":
        message = f"Unexpected JSONL path: {jsonl_path.name}"
        pytest.fail(message)
    if parquet_path.name != "function_metrics.parquet":
        message = f"Unexpected Parquet path: {parquet_path.name}"
        pytest.fail(message)
    with pytest.raises(ValueError, match="Unknown dataset"):
        export_dataset_to_jsonl(docs_export_gateway.gateway, "missing_dataset", output_dir)
    export_all_jsonl(
        docs_export_gateway.gateway,
        output_dir,
        validate_exports=True,
        schemas=["function_profile"],
    )


def test_export_validation_runs_against_registry(
    docs_export_gateway: ProvisionedGateway, tmp_path: Path
) -> None:
    """Exports should validate the dataset registry before writing files."""
    output_dir = tmp_path / "Document Output"
    docs_export_gateway.gateway.datasets = DatasetRegistry(
        mapping={"broken": "missing.table"},
        tables=("broken",),
        views=(),
    )
    with pytest.raises(ValueError, match="missing tables/views"):
        export_all_jsonl(docs_export_gateway.gateway, output_dir)
