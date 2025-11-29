"""Tests for dataset scaffold helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from codeintel.cli.main import ScaffoldConfigError, build_scaffold_options
from codeintel.storage.datasets import Dataset, DatasetRegistry
from codeintel.storage.scaffold import ScaffoldOptions, scaffold_dataset


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
    opts = ScaffoldOptions(**{**opts.__dict__, "dry_run": True})
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
    opts = ScaffoldOptions(
        **{
            **opts.__dict__,
            "is_view": True,
            "jsonl_filename": None,
            "parquet_filename": None,
        }
    )
    result = scaffold_dataset(opts)
    meta = json.loads(result.metadata.read_text(encoding="utf-8"))
    if meta.get("is_view") is not True:
        pytest.fail("View metadata flag missing")
    if meta.get("jsonl_filename") is not None or meta.get("parquet_filename") is not None:
        pytest.fail("View scaffold should not include export filenames by default")


def test_scaffold_emits_bootstrap_snippet_when_requested(tmp_path: Path) -> None:
    """Bootstrap snippet should be written when requested."""
    opts = _base_opts(tmp_path)
    opts = ScaffoldOptions(**{**opts.__dict__, "emit_bootstrap_snippet": True})
    result = scaffold_dataset(opts)
    if result.bootstrap_snippet is None or not result.bootstrap_snippet.exists():
        pytest.fail("Bootstrap snippet was not written")


def test_scaffold_registry_conflict_blocks_creation(tmp_path: Path) -> None:
    """Live registry clashes should fail fast when enabled."""
    opts = _base_opts(tmp_path)
    existing = Dataset(
        table_key=opts.table_key,
        name=opts.name,
        schema=None,
        stable_id=opts.stable_id,
    )
    registry = DatasetRegistry(
        by_name={opts.name: existing},
        by_table_key={opts.table_key: existing},
        jsonl_datasets={},
        parquet_datasets={},
    )
    args = Namespace(
        name=opts.name,
        kind="table",
        table_key=opts.table_key,
        schema_id=opts.schema_id,
        stable_id=opts.stable_id,
        jsonl_filename=opts.jsonl_filename,
        parquet_filename=opts.parquet_filename,
        overwrite=False,
        specs_snapshot=tmp_path / "missing.json",
        owner=None,
        freshness_sla=None,
        retention_policy=None,
        schema_version=opts.schema_version,
        validation_profile=opts.validation_profile,
        output_dir=opts.output_dir,
        dry_run=False,
        emit_bootstrap_snippet=False,
    )
    with pytest.raises(ScaffoldConfigError):
        build_scaffold_options(args, registry=registry)
