"""Tests for dataset scaffold helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from codeintel.storage.datasets.scaffold import ScaffoldOptions, scaffold_dataset
from tests._helpers.dataset_factories import sample_dataset_registry


def _base_opts(tmp_path: Path) -> ScaffoldOptions:
    return ScaffoldOptions(
        name="demo_dataset",
        table_key="analytics.demo_dataset",
        owner="team-data",
        freshness_sla="daily",
        retention_policy="90d",
        schema_version="1",
        stable_id="demo_dataset",
        validation_profile="strict",
        jsonl_filename="demo_dataset.jsonl",
        parquet_filename="demo_dataset.parquet",
        schema_id="demo_dataset",
        output_dir=tmp_path,
    )


def _ensure_no_registry_conflict(opts: ScaffoldOptions, registry: object) -> None:
    """Raise when scaffold options collide with existing registry entries.

    Raises
    ------
    ValueError
        If the name or table key already exists.
    """
    by_name = getattr(registry, "by_name", {})
    by_table = getattr(registry, "by_table_key", {})
    if opts.name in by_name:
        message = f"Dataset name already present in registry: {opts.name}"
        raise ValueError(message)
    if opts.table_key in by_table:
        message = f"Table key already present in registry: {opts.table_key}"
        raise ValueError(message)


def test_scaffold_writes_artifacts(tmp_path: Path) -> None:
    """Scaffold generation writes all expected artifacts."""
    result = scaffold_dataset(_base_opts(tmp_path))

    if not result.typed_dict.exists():
        pytest.fail("TypedDict stub was not created")
    if not result.json_schema.exists():
        pytest.fail("JSON Schema stub was not created")
    meta = json.loads(result.metadata.read_text(encoding="utf-8"))
    if meta.get("name") != "demo_dataset":
        pytest.fail("Metadata did not include dataset name")
    if meta.get("validation_profile") != "strict":
        pytest.fail("Metadata did not include validation profile")
    binding_content = result.row_binding.read_text(encoding="utf-8")
    if "demo_dataset" not in binding_content:
        pytest.fail("Row binding snippet did not reference dataset name")
    if result.bootstrap_snippet is not None:
        pytest.fail("Bootstrap snippet should not be written by default")


def test_scaffold_respects_dry_run(tmp_path: Path) -> None:
    """Dry-run should not create files."""
    opts = _base_opts(tmp_path)
    opts = replace(opts, dry_run=True)
    result = scaffold_dataset(opts)
    if result.typed_dict.exists() or result.metadata.exists():
        pytest.fail("Dry-run wrote files unexpectedly")


def test_scaffold_blocks_overwrite_without_flag(tmp_path: Path) -> None:
    """Existing targets should raise without --overwrite."""
    opts = _base_opts(tmp_path)
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    target = opts.output_dir / f"{opts.name}_rows.py"
    target.write_text("# existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scaffold_dataset(opts)


def test_scaffold_view_defaults_skip_exports(tmp_path: Path) -> None:
    """View scaffolds skip default export filenames."""
    opts = _base_opts(tmp_path)
    opts = replace(opts, is_view=True, jsonl_filename=None, parquet_filename=None)
    result = scaffold_dataset(opts)
    meta = json.loads(result.metadata.read_text(encoding="utf-8"))
    if meta.get("is_view") is not True:
        pytest.fail("View metadata flag missing")
    if meta.get("jsonl_filename") is not None or meta.get("parquet_filename") is not None:
        pytest.fail("View scaffold should not include export filenames by default")


def test_scaffold_emits_bootstrap_snippet_when_requested(tmp_path: Path) -> None:
    """Bootstrap snippet should be written when requested."""
    opts = _base_opts(tmp_path)
    opts = replace(opts, emit_bootstrap_snippet=True)
    result = scaffold_dataset(opts)
    if result.bootstrap_snippet is None or not result.bootstrap_snippet.exists():
        pytest.fail("Bootstrap snippet was not written")


def test_scaffold_registry_conflict_blocks_creation(tmp_path: Path) -> None:
    """Live registry clashes should fail fast when enabled."""
    opts = _base_opts(tmp_path)
    registry = sample_dataset_registry(tmp_path)
    opts = replace(
        opts,
        name="ast_nodes",
        table_key="core.ast_nodes",
        stable_id="ast_nodes",
    )
    with pytest.raises(ValueError, match="registry"):
        _ensure_no_registry_conflict(opts, registry)
