"""Cyclopts-based implementation of the extended datasets CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from codeintel.cli.cli_errors import run_handler
from codeintel.cli.cyclopts_common import (
    ExistingDir,
    ExistingPath,
    OutputPath,
    RuntimeCLI,
    RuntimeParam,
    runtime_cli_to_options,
)
from codeintel.cli.datasets_handlers import (
    BootstrapSnippet,
    BuildSelection,
    CatalogOptions,
    ConformanceOptions,
    DatasetExportOptions,
    DatasetScaffoldOptions,
    DiffOptions,
    DryRunMode,
    ExportValidationMode,
    GenerateSchemasOptions,
    LintOptions,
    ListFilters,
    MacroRequirement,
    OutputFormat,
    OverwritePolicy,
    ProjectSelection,
    RegistryCheck,
    RuntimeOptions,
    SamplingMode,
    SamplingStrictness,
    ScaffoldBehaviorOptions,
    ScaffoldFileOptions,
    ScaffoldIOOptions,
    ScaffoldMetadataOptions,
    ScaffoldSchemaOptions,
    datasets_catalog_handler,
    datasets_conformance_handler,
    datasets_diff_handler,
    datasets_generate_schemas_handler,
    datasets_lint_handler,
    datasets_list_handler,
    datasets_scaffold_handler,
    datasets_snapshot_handler,
)
from codeintel.cli.datasets_handlers import (
    ScaffoldCliOptions as ScaffoldCliOptionsHandler,
)

datasets_ext_app = App(
    name="datasets",
    help="Extended dataset management commands.",
)


@dataclass
class DatasetRuntimeCli:
    """Runtime selection shared by all datasets commands."""

    runtime: RuntimeParam = field(default_factory=RuntimeCLI)


def _runtime_from_cli(cli: DatasetRuntimeCli) -> RuntimeOptions:
    options = runtime_cli_to_options(cli.runtime)
    project = ProjectSelection(
        project_root=options.project_root,
        repo=options.repo,
        commit=options.commit,
        repo_root=options.repo_root,
    )
    build = BuildSelection(
        db_path=options.db_path,
        build_dir=options.build_dir,
    )
    return RuntimeOptions(project=project, build=build)


@dataclass
class LintCliOptions:
    """Options for ``codeintel datasets lint``."""

    schema_dir: Annotated[
        ExistingDir,
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
@dataclass
class LintCommand:
    """Validate dataset contract health."""

    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    options: Annotated[LintCliOptions, Parameter(name="*")] = field(default_factory=LintCliOptions)

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        lint_opts = LintOptions(
            schema_dir=self.options.schema_dir,
            sampling=(SamplingMode.ENABLED if self.options.sample_rows else SamplingMode.DISABLED),
        )
        run_handler(datasets_lint_handler, runtime_opts, lint_opts, self.runtime.runtime.verbose)


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
@dataclass
class ListDatasetsCommand:
    """List datasets with capabilities and optional filters."""

    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    filters: Annotated[ListCliFilters, Parameter(name="*")] = field(default_factory=ListCliFilters)

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        filter_opts = ListFilters(
            docs_view=self.filters.docs_view,
            read_only=self.filters.read_only,
            max_description=self.filters.max_description,
        )
        run_handler(datasets_list_handler, runtime_opts, filter_opts, self.runtime.runtime.verbose)


@dataclass
class SnapshotCliOptions:
    """Options for ``codeintel datasets snapshot``."""

    output: Annotated[
        OutputPath,
        Parameter(
            name="--output",
            help="Output file path for JSON dataset specs.",
        ),
    ] = Path("build/dataset_specs.json")


@datasets_ext_app.command(name="snapshot")
@dataclass
class SnapshotCommand:
    """Write current dataset specs to a JSON snapshot file."""

    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    options: Annotated[SnapshotCliOptions, Parameter(name="*")] = field(
        default_factory=SnapshotCliOptions
    )

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        run_handler(
            datasets_snapshot_handler,
            runtime_opts,
            self.options.output,
            self.runtime.runtime.verbose,
        )


@dataclass
class DiffCliOptions:
    """Options for ``codeintel datasets diff``."""

    baseline: Annotated[
        ExistingPath | None,
        Parameter(
            name="--baseline",
            help="Path to JSON baseline from `codeintel datasets snapshot`.",
        ),
    ] = None
    output: Annotated[
        OutputPath | None,
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
        OutputPath,
        Parameter(
            name="--baseline-path",
            help="Path of the snapshot file inside the git ref.",
        ),
    ] = Path("build/dataset_specs.json")


@datasets_ext_app.command(name="diff")
@dataclass
class DiffCommand:
    """Diff current dataset specs against a baseline."""

    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    options: Annotated[DiffCliOptions, Parameter(name="*")] = field(default_factory=DiffCliOptions)

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        diff_opts = DiffOptions(
            baseline=self.options.baseline,
            output=self.options.output,
            against_ref=self.options.against_ref,
            baseline_path=self.options.baseline_path,
        )
        run_handler(datasets_diff_handler, runtime_opts, diff_opts, self.runtime.runtime.verbose)


@dataclass
class ConformanceCliOptions:
    """Options for ``codeintel datasets conformance``."""

    schema_dir: Annotated[
        ExistingDir,
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
@dataclass
class ConformanceCommand:
    """Run full dataset conformance checks."""

    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    options: Annotated[ConformanceCliOptions, Parameter(name="*")] = field(
        default_factory=ConformanceCliOptions
    )

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        conf_opts = ConformanceOptions(
            schema_dir=self.options.schema_dir,
            sampling=(SamplingMode.ENABLED if self.options.sample_rows else SamplingMode.DISABLED),
            sample_size=self.options.sample_size,
        )
        run_handler(
            datasets_conformance_handler, runtime_opts, conf_opts, self.runtime.runtime.verbose
        )


@dataclass
class ExportCliOptions:
    """Shared export configuration for generate-schemas."""

    validation: Annotated[
        ExportValidationMode,
        Parameter(
            name="--validation",
            help="Validation strategy for exports.",
        ),
    ] = ExportValidationMode.REQUIRED
    macro_requirement: Annotated[
        MacroRequirement,
        Parameter(
            name="--macro-requirement",
            help="Macro requirement policy for exports.",
        ),
    ] = MacroRequirement.REQUIRE_NORMALIZED
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
        OutputFormat,
        Parameter(
            name="--output-format",
            help="Output format for command metadata (text or json).",
        ),
    ] = OutputFormat.TEXT
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
        OutputPath,
        Parameter(
            name="--output-dir",
            help="Directory to write generated JSON Schemas.",
        ),
    ] = Path("src/codeintel/config/schemas/export")


def _export_from_cli(cfg: ExportCliOptions) -> DatasetExportOptions:
    return DatasetExportOptions(
        validation=cfg.validation,
        macro_requirement=cfg.macro_requirement,
        schemas=cfg.schemas,
        datasets=cfg.datasets,
        output_format=cfg.output_format,
        run_mode=DryRunMode.DRY_RUN if cfg.dry_run else DryRunMode.EXECUTE,
    )


@datasets_ext_app.command(name="generate-schemas")
@dataclass
class GenerateSchemasCommand:
    """Generate export JSON Schemas from TypedDict row models."""

    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    export: Annotated[ExportCliOptions, Parameter(name="*")] = field(
        default_factory=ExportCliOptions
    )
    schema: Annotated[GenerateSchemasCliOptions, Parameter(name="*")] = field(
        default_factory=GenerateSchemasCliOptions
    )

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        export_opts = _export_from_cli(self.export)
        schema_opts = GenerateSchemasOptions(output_dir=self.schema.output_dir)
        run_handler(
            datasets_generate_schemas_handler,
            runtime_opts,
            export_opts,
            schema_opts,
            self.runtime.runtime.verbose,
        )


@dataclass
class CatalogCliOptions:
    """Options for ``codeintel datasets catalog``."""

    output_dir: Annotated[
        OutputPath,
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


@datasets_ext_app.command(name="catalog")
@dataclass
class CatalogCommand:
    """Generate Markdown/HTML dataset catalog."""

    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    options: Annotated[CatalogCliOptions, Parameter(name="*")] = field(
        default_factory=CatalogCliOptions
    )

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        catalog_opts = CatalogOptions(
            output_dir=self.options.output_dir,
            sample_rows_count=self.options.sample_rows_count,
            sample_rows_strict=self.options.sample_rows_strict,
        )
        run_handler(
            datasets_catalog_handler, runtime_opts, catalog_opts, self.runtime.runtime.verbose
        )


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
        OutputPath,
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
        ExistingPath,
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
        bool,
        Parameter(
            name=["--registry-check", "--check-registry"],
            help="Check existing dataset registry for conflicts.",
            negative=(),
        ),
    ] = False


def _scaffold_options_from_cli(cfg: ScaffoldCliOptions) -> ScaffoldCliOptionsHandler:
    metadata = ScaffoldMetadataOptions(
        kind=cfg.kind,
        table_key=cfg.table_key,
        owner=cfg.owner,
        freshness_sla=cfg.freshness_sla,
        retention_policy=cfg.retention_policy,
    )
    schema = ScaffoldSchemaOptions(
        schema_version=cfg.schema_version,
        validation_profile=cfg.validation_profile,
        schema_id=cfg.schema_id,
    )
    files = ScaffoldFileOptions(
        jsonl_filename=cfg.jsonl_filename,
        parquet_filename=cfg.parquet_filename,
        stable_id=cfg.stable_id,
    )
    dataset_opts = DatasetScaffoldOptions(
        output_dir=cfg.output_dir,
        overwrite_policy=cfg.overwrite_policy,
    )
    io_opts = ScaffoldIOOptions(
        specs_snapshot=cfg.specs_snapshot,
        scaffold=dataset_opts,
    )
    behavior = ScaffoldBehaviorOptions(
        run_mode=DryRunMode.DRY_RUN if cfg.dry_run else DryRunMode.EXECUTE,
        bootstrap=cfg.bootstrap,
        registry_check=RegistryCheck.ENABLED if cfg.registry_check else RegistryCheck.DISABLED,
    )
    return ScaffoldCliOptionsHandler(
        metadata=metadata,
        schema=schema,
        files=files,
        io=io_opts,
        behavior=behavior,
    )


@datasets_ext_app.command(name="scaffold")
@dataclass
class ScaffoldCommand:
    """Create a new dataset scaffold."""

    name: Annotated[
        str,
        Parameter(
            help="Name of the dataset to scaffold (TypedDict / logical dataset name).",
        ),
    ] = ""
    runtime: Annotated[DatasetRuntimeCli, Parameter(name="*")] = field(
        default_factory=DatasetRuntimeCli
    )
    options: Annotated[ScaffoldCliOptions, Parameter(name="*")] = field(
        default_factory=ScaffoldCliOptions
    )

    def __call__(self) -> None:
        runtime_opts = _runtime_from_cli(self.runtime)
        scaffold_opts = _scaffold_options_from_cli(self.options)
        run_handler(
            datasets_scaffold_handler,
            self.name,
            runtime_opts,
            scaffold_opts,
            self.runtime.runtime.verbose,
        )


__all__ = ["datasets_ext_app"]
