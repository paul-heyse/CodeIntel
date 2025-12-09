"""Cyclopts-based implementation of the extended datasets CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import invoke_with_typer_translation
from codeintel.cli.commands import datasets as ds
from codeintel.cli.cyclopts_common import RuntimeCLI, runtime_cli_to_options

datasets_ext_app = App(
    name="datasets",
    help="Extended dataset management commands.",
)


@dataclass
class DatasetRuntimeCli:
    """Runtime selection shared by all datasets commands."""

    runtime: Annotated[RuntimeCLI, Parameter(name="*")] = field(default_factory=RuntimeCLI)


def _runtime_from_cli(cli: DatasetRuntimeCli) -> ds.RuntimeOptions:
    options = runtime_cli_to_options(cli.runtime)
    project = ds.ProjectSelection(
        project_root=options.project_root,
        repo=options.repo,
        commit=options.commit,
        repo_root=options.repo_root,
    )
    build = ds.BuildSelection(
        db_path=options.db_path,
        build_dir=options.build_dir,
    )
    return ds.RuntimeOptions(project=project, build=build)


def _run(
    handler: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> None:
    """Invoke a Typer-era handler and normalize exits into ``SystemExit``."""
    invoke_with_typer_translation(handler, *args, **kwargs)


@dataclass
class LintCliOptions:
    """Options for ``codeintel datasets lint``."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sample_rows: Annotated[
        bool,
        Parameter(
            name="--sample-rows",
            help="Request row sampling (SamplingMode.ENABLED).",
            negative=(),
        ),
    ] = False


@datasets_ext_app.command(name="lint")
def lint(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    options: Annotated[LintCliOptions, Parameter(name="*")] | None = None,
) -> None:
    """Validate dataset contract health."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    selected_options = options or LintCliOptions()
    lint_opts = ds.LintOptions(
        schema_dir=selected_options.schema_dir,
        sampling=(
            ds.SamplingMode.ENABLED if selected_options.sample_rows else ds.SamplingMode.DISABLED
        ),
    )
    _run(ds.datasets_lint_handler, runtime_opts, lint_opts, runtime_cfg.runtime.verbose)


DocsFilterMode = Literal["include", "only", "exclude"]
ReadOnlyFilterMode = Literal["include", "only", "exclude"]


@dataclass
class ListCliFilters:
    """Filters for ``codeintel datasets list``."""

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


@datasets_ext_app.command(name="list")
def list_datasets(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    filters: Annotated[ListCliFilters, Parameter(name="*")] | None = None,
) -> None:
    """List datasets with capabilities and optional filters."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    selected_filters = filters or ListCliFilters()
    filter_opts = ds.ListFilters(
        docs_view=selected_filters.docs_view,
        read_only=selected_filters.read_only,
        max_description=selected_filters.max_description,
    )
    _run(ds.datasets_list_handler, runtime_opts, filter_opts, runtime_cfg.runtime.verbose)


@dataclass
class SnapshotCliOptions:
    """Options for ``codeintel datasets snapshot``."""

    output: Annotated[
        Path,
        Parameter(
            name="--output",
            help="Output file path for JSON dataset specs.",
        ),
    ] = Path("build/dataset_specs.json")


@datasets_ext_app.command(name="snapshot")
def snapshot(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    options: Annotated[SnapshotCliOptions, Parameter(name="*")] | None = None,
) -> None:
    """Write current dataset specs to a JSON snapshot file."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    selected_options = options or SnapshotCliOptions()
    _run(
        ds.datasets_snapshot_handler,
        runtime_opts,
        selected_options.output,
        runtime_cfg.runtime.verbose,
    )


@dataclass
class DiffCliOptions:
    """Options for ``codeintel datasets diff``."""

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


@datasets_ext_app.command(name="diff")
def diff(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    options: Annotated[DiffCliOptions, Parameter(name="*")] | None = None,
) -> None:
    """Diff current dataset specs against a baseline."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    selected_options = options or DiffCliOptions()
    diff_opts = ds.DiffOptions(
        baseline=selected_options.baseline,
        output=selected_options.output,
        against_ref=selected_options.against_ref,
        baseline_path=selected_options.baseline_path,
    )
    _run(ds.datasets_diff_handler, runtime_opts, diff_opts, runtime_cfg.runtime.verbose)


@dataclass
class ConformanceCliOptions:
    """Options for ``codeintel datasets conformance``."""

    schema_dir: Annotated[
        Path,
        Parameter(
            name="--schema-dir",
            help="Directory containing export JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")
    sample_rows: Annotated[
        bool,
        Parameter(
            name="--sample-rows",
            help="Enable row sampling against JSON Schemas.",
            negative=(),
        ),
    ] = False
    sample_size: Annotated[
        int,
        Parameter(
            name="--sample-size",
            help="Number of rows to sample when sampling is enabled.",
        ),
    ] = 50


@datasets_ext_app.command(name="conformance")
def conformance(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    options: Annotated[ConformanceCliOptions, Parameter(name="*")] | None = None,
) -> None:
    """Run full dataset conformance checks."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    selected_options = options or ConformanceCliOptions()
    conf_opts = ds.ConformanceOptions(
        schema_dir=selected_options.schema_dir,
        sampling=(
            ds.SamplingMode.ENABLED if selected_options.sample_rows else ds.SamplingMode.DISABLED
        ),
        sample_size=selected_options.sample_size,
    )
    _run(ds.datasets_conformance_handler, runtime_opts, conf_opts, runtime_cfg.runtime.verbose)


@dataclass
class ExportCliOptions:
    """Shared export configuration for generate-schemas."""

    validation: Annotated[
        ds.ExportValidationMode,
        Parameter(
            name="--validation",
            help="Validation strategy for exports.",
        ),
    ] = ds.ExportValidationMode.REQUIRED
    macro_requirement: Annotated[
        ds.MacroRequirement,
        Parameter(
            name="--macro-requirement",
            help="Macro requirement policy for exports.",
        ),
    ] = ds.MacroRequirement.REQUIRE_NORMALIZED
    schemas: Annotated[
        list[str] | None,
        Parameter(
            name="--schema",
            help="Filter by export schema ID (repeatable).",
        ),
    ] = None
    datasets: Annotated[
        list[str] | None,
        Parameter(
            name="--dataset",
            help="Filter by dataset name (repeatable).",
        ),
    ] = None
    output_format: Annotated[
        ds.OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format for command metadata (text or json).",
        ),
    ] = ds.OutputFormat.TEXT
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            help="Plan schema generation without writing files.",
            negative=(),
        ),
    ] = False


@dataclass
class GenerateSchemasCliOptions:
    """Options for ``codeintel datasets generate-schemas``."""

    output_dir: Annotated[
        Path,
        Parameter(
            name="--output-dir",
            help="Directory to write generated JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")


def _export_from_cli(cfg: ExportCliOptions) -> ds.DatasetExportOptions:
    return ds.DatasetExportOptions(
        validation=cfg.validation,
        macro_requirement=cfg.macro_requirement,
        schemas=cfg.schemas,
        datasets=cfg.datasets,
        output_format=cfg.output_format,
        run_mode=ds.DryRunMode.DRY_RUN if cfg.dry_run else ds.DryRunMode.EXECUTE,
    )


@datasets_ext_app.command(name="generate-schemas")
def generate_schemas(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    export: Annotated[ExportCliOptions, Parameter(name="*")] | None = None,
    schema: Annotated[GenerateSchemasCliOptions, Parameter(name="*")] | None = None,
) -> None:
    """Generate export JSON Schemas from TypedDict row models."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    export_opts = _export_from_cli(export or ExportCliOptions())
    schema_cfg = schema or GenerateSchemasCliOptions()
    schema_opts = ds.GenerateSchemasOptions(output_dir=schema_cfg.output_dir)
    _run(
        ds.datasets_generate_schemas_handler,
        runtime_opts,
        export_opts,
        schema_opts,
        runtime_cfg.runtime.verbose,
    )


@dataclass
class CatalogCliOptions:
    """Options for ``codeintel datasets catalog``."""

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
        ds.SamplingStrictness,
        Parameter(
            name="--sample-rows-strict",
            help="Sampling strictness: lenient or strict.",
        ),
    ] = ds.SamplingStrictness.LENIENT


@datasets_ext_app.command(name="catalog")
def catalog(
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    options: Annotated[CatalogCliOptions, Parameter(name="*")] | None = None,
) -> None:
    """Generate Markdown/HTML dataset catalog."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    selected_options = options or CatalogCliOptions()
    catalog_opts = ds.CatalogOptions(
        output_dir=selected_options.output_dir,
        sample_rows_count=selected_options.sample_rows_count,
        sample_rows_strict=selected_options.sample_rows_strict,
    )
    _run(ds.datasets_catalog_handler, runtime_opts, catalog_opts, runtime_cfg.runtime.verbose)


@dataclass
class ScaffoldCliOptions:
    """Options for ``codeintel datasets scaffold``."""

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
        ds.OverwritePolicy,
        Parameter(
            name="--overwrite-policy",
            help="Overwrite policy when scaffold paths already exist.",
        ),
    ] = ds.OverwritePolicy.ERROR
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
        ds.BootstrapSnippet,
        Parameter(
            name="--bootstrap",
            help="Control emission of bootstrap snippets in metadata.",
        ),
    ] = ds.BootstrapSnippet.SKIP
    registry_check: Annotated[
        bool,
        Parameter(
            name=["--registry-check", "--check-registry"],
            help="Check existing dataset registry for conflicts.",
            negative=(),
        ),
    ] = False


def _scaffold_options_from_cli(cfg: ScaffoldCliOptions) -> ds.ScaffoldCliOptions:
    metadata = ds.ScaffoldMetadataOptions(
        kind=cfg.kind,
        table_key=cfg.table_key,
        owner=cfg.owner,
        freshness_sla=cfg.freshness_sla,
        retention_policy=cfg.retention_policy,
    )
    schema = ds.ScaffoldSchemaOptions(
        schema_version=cfg.schema_version,
        validation_profile=cfg.validation_profile,
        schema_id=cfg.schema_id,
    )
    files = ds.ScaffoldFileOptions(
        jsonl_filename=cfg.jsonl_filename,
        parquet_filename=cfg.parquet_filename,
        stable_id=cfg.stable_id,
    )
    dataset_opts = ds.DatasetScaffoldOptions(
        output_dir=cfg.output_dir,
        overwrite_policy=cfg.overwrite_policy,
    )
    io_opts = ds.ScaffoldIOOptions(
        specs_snapshot=cfg.specs_snapshot,
        scaffold=dataset_opts,
    )
    behavior = ds.ScaffoldBehaviorOptions(
        run_mode=ds.DryRunMode.DRY_RUN if cfg.dry_run else ds.DryRunMode.EXECUTE,
        bootstrap=cfg.bootstrap,
        registry_check=ds.RegistryCheck.ENABLED
        if cfg.registry_check
        else ds.RegistryCheck.DISABLED,
    )
    return ds.ScaffoldCliOptions(
        metadata=metadata,
        schema=schema,
        files=files,
        io=io_opts,
        behavior=behavior,
    )


@datasets_ext_app.command(name="scaffold")
def scaffold(
    name: Annotated[
        str,
        Parameter(
            help="Name of the dataset to scaffold (TypedDict / logical dataset name).",
        ),
    ],
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] | None = None,
    options: Annotated[ScaffoldCliOptions, Parameter(name="*")] | None = None,
) -> None:
    """Create a new dataset scaffold."""
    runtime_cfg = runtime or DatasetRuntimeCli()
    runtime_opts = _runtime_from_cli(runtime_cfg)
    scaffold_opts = _scaffold_options_from_cli(options or ScaffoldCliOptions())
    _run(
        ds.datasets_scaffold_handler,
        name,
        runtime_opts,
        scaffold_opts,
        runtime_cfg.runtime.verbose,
    )


__all__ = ["datasets_ext_app"]
