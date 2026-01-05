"""Smoke test for exporting datasets to the document output directory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.exports import (
    ExportCallOptions,
    export_all_jsonl,
    export_all_parquet,
    export_dataset_to_jsonl,
    export_dataset_to_parquet,
)
from codeintel.core.config.settings import ExportAuditSettings
from codeintel.core.gateway import BuildGateway
from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.storage.datasets import DatasetRegistry
from tests._helpers import TestContext, provision_docs_export_ready

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

EXPORT_SETTINGS = ExportAuditSettings()


def _build_gateway(ctx: TestContext) -> BuildGateway:
    return cast("BuildGateway", ctx.gateway)


@pytest.fixture
def docs_export_gateway(tmp_path: Path) -> Iterator[TestContext]:
    """Provision docs-export-ready gateway and ensure cleanup.

    Yields
    ------
    TestContext
        Provisioned context seeded for docs export.
    """
    provisioned = provision_docs_export_ready(tmp_path, repo="r", commit="c", file_backed=False)
    ctx = TestContext.from_provisioned(provisioned)
    try:
        yield ctx
    finally:
        ctx.close()


def test_export_all_writes_expected_files(docs_export_gateway: TestContext, tmp_path: Path) -> None:
    """
    Seed a minimal DB and verify Parquet/JSONL exports are produced.

    This ensures the export mappings are usable end-to-end.

    Raises
    ------
    AssertionError
        If any expected export is missing after running both exporters.
    """
    output_dir = tmp_path / "build" / "document_output"
    export_all_parquet(
        _build_gateway(docs_export_gateway),
        output_dir,
        settings=EXPORT_SETTINGS,
        options=ExportCallOptions(validate_exports=False),
    )
    export_all_jsonl(
        _build_gateway(docs_export_gateway),
        output_dir,
        settings=EXPORT_SETTINGS,
        options=ExportCallOptions(validate_exports=False),
    )

    expected_basenames = {
        "goids.parquet",
        "goid_crosswalk.parquet",
        "call_graph_nodes.parquet",
        "call_graph_edges.parquet",
        "cfg_blocks.parquet",
        "import_graph_edges.parquet",
        "docstrings.parquet",
        "function_types.parquet",
        "test_catalog.parquet",
        "goids.jsonl",
        "goid_crosswalk.jsonl",
        "call_graph_nodes.jsonl",
        "call_graph_edges.jsonl",
        "cfg_blocks.jsonl",
        "import_graph_edges.jsonl",
        "docstrings.jsonl",
        "function_types.jsonl",
        "test_catalog.jsonl",
        "datasets_manifest.json",
    }

    written = {p.name for p in output_dir.iterdir() if p.is_file()}

    missing = expected_basenames - written
    if missing:
        message = f"Expected exports missing: {sorted(missing)}"
        raise AssertionError(message)
    manifest = json.loads((output_dir / "datasets_manifest.json").read_text(encoding="utf-8"))
    dataset_entries = {entry["name"]: entry for entry in manifest.get("datasets", [])}
    if "function_types" not in dataset_entries:
        pytest.fail("function_types missing from dataset manifest")
    types_entry = dataset_entries["function_types"]
    if types_entry.get("jsonl") != "function_types.jsonl":
        pytest.fail(f"Unexpected manifest entry: {types_entry}")


def test_export_validation_passes_on_minimal_data(
    docs_export_gateway: TestContext, tmp_path: Path
) -> None:
    """Ensure validation succeeds when provided with conforming exports."""
    output_dir = tmp_path / "build" / "document_output"
    export_all_parquet(
        _build_gateway(docs_export_gateway),
        output_dir,
        settings=EXPORT_SETTINGS,
        options=ExportCallOptions(
            validate_exports=True,
            schemas=["analytics.function_types"],
        ),
    )


def test_export_subset_by_dataset_name(docs_export_gateway: TestContext, tmp_path: Path) -> None:
    """Exports honor dataset-name selection using the registry."""
    output_dir = tmp_path / "build" / "document_output"
    selected = ["function_types", "goids"]
    export_all_parquet(
        _build_gateway(docs_export_gateway),
        output_dir,
        settings=EXPORT_SETTINGS,
        options=ExportCallOptions(validate_exports=False, datasets=selected),
    )
    export_all_jsonl(
        _build_gateway(docs_export_gateway),
        output_dir,
        settings=EXPORT_SETTINGS,
        options=ExportCallOptions(validate_exports=False, datasets=selected),
    )

    written = {p.name for p in output_dir.iterdir() if p.is_file()}
    expected = {
        "function_types.parquet",
        "goids.parquet",
        "function_types.jsonl",
        "goids.jsonl",
        "function_types.parquet.manifest.json",
        "function_types.parquet.marker.json",
        "goids.parquet.manifest.json",
        "goids.parquet.marker.json",
        "function_types.jsonl.manifest.json",
        "function_types.jsonl.marker.json",
        "goids.jsonl.manifest.json",
        "goids.jsonl.marker.json",
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
    docs_export_gateway: TestContext, tmp_path: Path
) -> None:
    """Dataset selection rejects unknown names."""
    output_dir = tmp_path / "build" / "document_output"
    with pytest.raises(ValueError, match="Unknown dataset"):
        export_all_jsonl(
            _build_gateway(docs_export_gateway),
            output_dir,
            settings=EXPORT_SETTINGS,
            options=ExportCallOptions(validate_exports=False, datasets=["missing_dataset"]),
        )


def test_export_helpers_resolve_dataset_names(
    docs_export_gateway: TestContext, tmp_path: Path
) -> None:
    """Dataset-aware export helpers resolve registry names to filenames."""
    output_dir = tmp_path / "build" / "document_output"
    jsonl_path = export_dataset_to_jsonl(
        _build_gateway(docs_export_gateway),
        "function_types",
        output_dir,
        settings=EXPORT_SETTINGS,
    )
    parquet_path = export_dataset_to_parquet(
        _build_gateway(docs_export_gateway),
        "function_types",
        output_dir,
        settings=EXPORT_SETTINGS,
    )
    if not jsonl_path.exists():
        message = f"JSONL export not written: {jsonl_path}"
        pytest.fail(message)
    if not parquet_path.exists():
        message = f"Parquet export not written: {parquet_path}"
        pytest.fail(message)
    if jsonl_path.name != "function_types.jsonl":
        message = f"Unexpected JSONL path: {jsonl_path.name}"
        pytest.fail(message)
    if parquet_path.name != "function_types.parquet":
        message = f"Unexpected Parquet path: {parquet_path.name}"
        pytest.fail(message)
    with pytest.raises(ValueError, match="Unknown dataset"):
        export_dataset_to_jsonl(
            _build_gateway(docs_export_gateway),
            "missing_dataset",
            output_dir,
            settings=EXPORT_SETTINGS,
        )
    export_all_jsonl(
        _build_gateway(docs_export_gateway),
        output_dir,
        settings=EXPORT_SETTINGS,
        options=ExportCallOptions(
            validate_exports=True,
            schemas=["analytics.function_types"],
        ),
    )


def test_export_validation_runs_against_registry(
    docs_export_gateway: TestContext, tmp_path: Path
) -> None:
    """Exports should validate the dataset registry before writing files."""
    output_dir = tmp_path / "build" / "document_output"

    broken_contract = DatasetContract(
        table_key="missing.table",
        name="broken",
        schema=None,
        is_view=False,
    )
    docs_export_gateway.gateway.datasets = DatasetRegistry(
        by_name={"broken": broken_contract},
        by_table_key={"missing.table": broken_contract},
        jsonl_datasets={},
        parquet_datasets={},
    )
    with pytest.raises(ValueError, match="missing tables/views"):
        export_all_jsonl(
            _build_gateway(docs_export_gateway),
            output_dir,
            settings=EXPORT_SETTINGS,
            options=ExportCallOptions(validate_exports=False),
        )
