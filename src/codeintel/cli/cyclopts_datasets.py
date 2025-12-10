"""Cyclopts wiring for dataset management commands.

This module wires Cyclopts command classes to unified ExecutionContext handlers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.datasets_handlers import (
    BootstrapSnippet,
    OverwritePolicy,
    SamplingStrictness,
    datasets_catalog_ctx,
    datasets_conformance_ctx,
    datasets_diff_ctx,
    datasets_generate_schemas_ctx,
    datasets_lint_ctx,
    datasets_list_ctx,
    datasets_scaffold_ctx,
    datasets_snapshot_ctx,
    datasets_validate_files_ctx,
)
from codeintel.cli.execution.adapter import CycloptsAdapter

datasets_ext_app = App(
    name="datasets",
    help="Extended dataset management commands.",
)


DocsFilterMode = Literal["include", "only", "exclude"]
ReadOnlyFilterMode = Literal["include", "only", "exclude"]


@datasets_ext_app.command(name="lint")
@dataclass
class LintCommand:
    """Validate dataset contract health."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sampling: Annotated[
        str,
        Parameter(
            name="--sampling",
            help="Sampling mode: enabled or disabled.",
        ),
    ] = "disabled"
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets lint command."""
        CycloptsAdapter("datasets.lint", datasets_lint_ctx)(self)


@datasets_ext_app.command(name="list")
@dataclass
class ListDatasetsCommand:
    """List datasets with capabilities and optional filters."""

    docs_view: Annotated[
        DocsFilterMode,
        Parameter(
            name="--docs-view",
            help='Docs view filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    read_only: Annotated[
        ReadOnlyFilterMode,
        Parameter(
            name="--read-only",
            help='Read-only filter: "include", "exclude", or "only".',
        ),
    ] = "include"
    max_description: Annotated[
        int,
        Parameter(
            name="--max-description",
            help="Maximum description length before truncation.",
        ),
    ] = 80
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets list command."""
        CycloptsAdapter("datasets.list", datasets_list_ctx)(self)


@datasets_ext_app.command(name="snapshot")
@dataclass
class SnapshotCommand:
    """Write current dataset specs to a JSON snapshot file."""

    output: Annotated[
        Path,
        Parameter(
            name="--output",
            help="Output file path for JSON dataset specs.",
        ),
    ] = Path("build/dataset_specs.json")
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets snapshot command."""
        CycloptsAdapter("datasets.snapshot", datasets_snapshot_ctx)(self)


@datasets_ext_app.command(name="diff")
@dataclass
class DiffCommand:
    """Diff current dataset specs against a baseline."""

    baseline: Annotated[
        Path | None,
        Parameter(
            name="--baseline",
            help="Path to JSON baseline from `codeintel datasets snapshot`.",
        ),
    ] = None
    output: Annotated[
        Path | None,
        Parameter(
            name="--output",
            help="Optional output file path for writing current specs.",
        ),
    ] = None
    against_ref: Annotated[
        str | None,
        Parameter(
            name="--against-ref",
            help="Git ref to diff against (e.g. HEAD~, main).",
        ),
    ] = None
    baseline_path: Annotated[
        Path,
        Parameter(
            name="--baseline-path",
            help="Path of the snapshot file inside the git ref.",
        ),
    ] = Path("build/dataset_specs.json")
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets diff command."""
        CycloptsAdapter("datasets.diff", datasets_diff_ctx)(self)


@datasets_ext_app.command(name="conformance")
@dataclass
class ConformanceCommand:
    """Run full dataset conformance checks."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sampling: Annotated[
        str,
        Parameter(
            name="--sampling",
            help="Sampling mode: enabled or disabled.",
        ),
    ] = "disabled"
    sample_size: Annotated[
        int,
        Parameter(
            name="--sample-size",
            help="Number of rows to sample when sampling is enabled.",
        ),
    ] = 100
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets conformance command."""
        CycloptsAdapter("datasets.conformance", datasets_conformance_ctx)(self)


@datasets_ext_app.command(name="generate-schemas")
@dataclass
class GenerateSchemasCommand:
    """Generate export JSON Schemas from TypedDict row models."""

    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Directory to write generated JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    datasets: Annotated[
        list[str] | None,
        Parameter(
            name="--dataset",
            help="Filter by dataset name (repeatable).",
        ),
    ] = None
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets generate-schemas command."""
        CycloptsAdapter("datasets.generate_schemas", datasets_generate_schemas_ctx)(self)


@datasets_ext_app.command(name="catalog")
@dataclass
class CatalogCommand:
    """Generate Markdown/HTML dataset catalog."""

    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Directory to write catalog artifacts (Markdown/HTML).",
        ),
    ] = Path("build/catalog")
    sample_rows_count: Annotated[
        int,
        Parameter(
            name="--sample-rows-count",
            help="Number of sample rows per dataset in the catalog.",
        ),
    ] = 3
    sample_rows_strict: Annotated[
        SamplingStrictness,
        Parameter(
            name="--sample-rows-strict",
            help="Sampling strictness: lenient or strict.",
        ),
    ] = SamplingStrictness.LENIENT
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets catalog command."""
        CycloptsAdapter("datasets.catalog", datasets_catalog_ctx)(self)


@datasets_ext_app.command(name="scaffold")
@dataclass
class ScaffoldCommand:
    """Create a new dataset scaffold."""

    name: Annotated[
        str,
        Parameter(
            help="Name of the dataset to scaffold (TypedDict / logical dataset name).",
            required=True,
        ),
    ] = ""
    kind: Annotated[
        str,
        Parameter(
            name="--kind",
            help='Kind of dataset: typically "table" or "view".',
        ),
    ] = "table"
    table_key: Annotated[
        str | None,
        Parameter(
            name="--table-key",
            help="Logical table key for the dataset.",
        ),
    ] = None
    owner: Annotated[
        str | None,
        Parameter(
            name="--owner",
            help="Owning team or contact identifier.",
        ),
    ] = None
    freshness_sla: Annotated[
        str | None,
        Parameter(
            name="--freshness-sla",
            help="Freshness SLA description (e.g. 1h, 1d).",
        ),
    ] = None
    retention_policy: Annotated[
        str | None,
        Parameter(
            name="--retention-policy",
            help="Retention policy summary for the dataset.",
        ),
    ] = None
    schema_version: Annotated[
        str,
        Parameter(
            name="--schema-version",
            help="Schema version tag for the export schema.",
        ),
    ] = "1"
    validation_profile: Annotated[
        str,
        Parameter(
            name="--validation-profile",
            help="Validation profile name (e.g. strict, permissive).",
        ),
    ] = "strict"
    schema_id: Annotated[
        str | None,
        Parameter(
            name="--schema-id",
            help="Explicit JSON Schema $id for the dataset.",
        ),
    ] = None
    jsonl_filename: Annotated[
        str | None,
        Parameter(
            name="--jsonl-filename",
            help="Filename for the JSONL export.",
        ),
    ] = None
    parquet_filename: Annotated[
        str | None,
        Parameter(
            name="--parquet-filename",
            help="Filename for the Parquet export.",
        ),
    ] = None
    stable_id: Annotated[
        str | None,
        Parameter(
            name="--stable-id",
            help="Stable identifier used for tracking.",
        ),
    ] = None
    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Directory to write scaffold files.",
        ),
    ] = Path("build/dataset_scaffolds")
    overwrite_policy: Annotated[
        OverwritePolicy,
        Parameter(
            name="--overwrite-policy",
            help="Overwrite policy when scaffold paths already exist.",
        ),
    ] = OverwritePolicy.ERROR
    specs_snapshot: Annotated[
        Path,
        Parameter(
            name="--specs-snapshot",
            help="Path to dataset specs snapshot used for bootstrap hints.",
        ),
    ] = Path("build/catalog/dataset_specs.json")
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Show scaffold plan without writing files.",
            negative=(),
        ),
    ] = False
    bootstrap: Annotated[
        BootstrapSnippet,
        Parameter(
            name="--bootstrap",
            help="Control emission of bootstrap snippets in metadata.",
        ),
    ] = BootstrapSnippet.SKIP
    registry_check: Annotated[
        str,
        Parameter(
            name="--registry-check",
            help="Registry check mode: enabled or disabled.",
        ),
    ] = "disabled"
    project_root: Annotated[
        Path | None,
        Parameter(
            name="--root",
            help="Project root directory.",
        ),
    ] = None
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets scaffold command."""
        CycloptsAdapter("datasets.scaffold", datasets_scaffold_ctx)(self)


@datasets_ext_app.command(name="validate-files")
@dataclass
class ValidateFilesCommand:
    """Validate exported JSONL/Parquet files against JSON Schemas."""

    schema: Annotated[
        str,
        Parameter(
            help="Schema name to validate against.",
            required=True,
        ),
    ] = ""
    files: Annotated[
        list[Path],
        Parameter(
            help="Files to validate (JSONL or Parquet).",
        ),
    ] = None  # type: ignore[assignment]
    schema_root: Annotated[
        Path | None,
        Parameter(
            name="--schema-root",
            help="Root directory for JSON Schemas.",
        ),
    ] = None
    validation: Annotated[
        str,
        Parameter(
            name="--validation",
            help="Validation mode: required or skip.",
        ),
    ] = "required"
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Show validation plan without executing.",
            negative=(),
        ),
    ] = False
    verbose: Annotated[
        int,
        Parameter(
            name=["-v", "--verbose"],
            help="Increase verbosity level.",
            count=True,
        ),
    ] = 0

    def __call__(self) -> None:
        """Execute the datasets validate-files command."""
        CycloptsAdapter("datasets.validate_files", datasets_validate_files_ctx)(self)


__all__ = ["datasets_ext_app"]
