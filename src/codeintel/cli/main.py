"""CLI entrypoint for CodeIntel pipeline and document exports."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol, cast

from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions, build_graph_runtime
from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.config import ConfigBuilder
from codeintel.config.models import CliConfigOptions, CliPathsInput, CodeIntelConfig, RepoConfig
from codeintel.config.parser_types import FunctionParserKind
from codeintel.config.primitives import GraphBackendConfig, GraphFeatureFlags, SnapshotRef
from codeintel.graphs.nx_backend import maybe_enable_nx_gpu
from codeintel.ingestion.source_scanner import (
    default_code_profile,
    default_config_profile,
    profile_from_env,
)
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import (
    ExportOptions,
    ExportRunner,
    run_validated_exports,
)
from codeintel.pipeline.orchestration.prefect_flow import ExportArgs, export_docs_flow
from codeintel.pipeline.orchestration.steps import REGISTRY, StepPhase
from codeintel.serving.http.datasets import validate_dataset_registry
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.errors import ExportError, log_problem, problem
from codeintel.storage.catalog import (
    build_catalog,
    write_html_catalog,
    write_markdown_catalog,
)
from codeintel.storage.conformance import run_conformance
from codeintel.storage.contract_validation import collect_contract_issues
from codeintel.storage.datasets import DatasetRegistry, list_dataset_specs, load_dataset_registry
from codeintel.storage.gateway import (
    DuckDBError,
    StorageConfig,
    StorageGateway,
    build_snapshot_gateway_resolver,
    open_gateway,
)
from codeintel.storage.metadata_bootstrap import (
    _assert_macro_coverage,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)
from codeintel.storage.scaffold import ScaffoldOptions, scaffold_dataset
from codeintel.storage.schema_generation import generate_export_schemas

LOG = logging.getLogger("codeintel.cli")


class GatewayFactory(Protocol):
    """Factory for creating gateways with optional read-only mode."""

    def __call__(self, cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
        """Create a gateway."""
        ...


CommandHandler = Callable[..., int]


# ---------------------------------------------------------------------------
# Argument parsing / logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbosity: int) -> None:
    """
    Configure logging based on -v/--verbose count.

    0 -> WARNING, 1 -> INFO, 2+ -> DEBUG.
    """
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _add_common_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(),
        help="Path to the repository root (default: current directory)",
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Repository slug (used in GOIDs and exports, e.g. 'my-org/my-repo')",
    )
    p.add_argument(
        "--commit",
        required=True,
        help="Commit SHA for this analysis run (embedded into GOIDs)",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=Path("build/db/codeintel_prefect.duckdb"),
        help="Path to the DuckDB database (default: build/db/codeintel_prefect.duckdb)",
    )
    p.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Build directory (default: build/)",
    )
    p.add_argument(
        "--document-output-dir",
        type=Path,
        default=None,
        help="Override Document Output/ directory (default: <repo-root>/Document Output)",
    )


def _add_graph_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--nx-gpu",
        action="store_true",
        help="Prefer GPU backend for NetworkX (nx-cugraph) when available.",
    )
    parser.add_argument(
        "--nx-backend",
        choices=["auto", "cpu", "nx-cugraph"],
        default="auto",
        help="NetworkX backend selection (default: auto).",
    )
    parser.add_argument(
        "--nx-gpu-strict",
        action="store_true",
        help="Fail instead of falling back to CPU if GPU backend cannot be enabled.",
    )


def _add_subsystem_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register subsystem-related subcommands."""
    p_subsys = subparsers.add_parser(
        "subsystem",
        help="Subsystem exploration helpers (docs views, read-only caches)",
    )
    subsys_sub = p_subsys.add_subparsers(dest="subcommand", required=True)

    p_subsys_list = subsys_sub.add_parser(
        "list",
        help="List inferred subsystems with role/risk metadata (cached docs view).",
    )
    _add_common_repo_args(p_subsys_list)
    p_subsys_list.add_argument("--role", help="Filter subsystems by role tag")
    p_subsys_list.add_argument("--q", help="Search substring on name/description")
    p_subsys_list.add_argument("--limit", type=int, default=None, help="Limit subsystem count")
    p_subsys_list.set_defaults(func=_cmd_subsystem_list)

    p_subsys_show = subsys_sub.add_parser(
        "show",
        help="Show subsystem detail and modules (docs view, read-only).",
    )
    _add_common_repo_args(p_subsys_show)
    p_subsys_show.add_argument("--subsystem-id", required=True, help="Subsystem identifier")
    p_subsys_show.set_defaults(func=_cmd_subsystem_show)

    p_subsys_profiles = subsys_sub.add_parser(
        "profiles",
        help="List subsystem profiles (docs.v_subsystem_profile, read-only docs view).",
    )
    _add_common_repo_args(p_subsys_profiles)
    p_subsys_profiles.add_argument(
        "--limit", type=int, default=None, help="Limit subsystem profile rows"
    )
    p_subsys_profiles.set_defaults(func=_cmd_subsystem_profiles)

    p_subsys_coverage = subsys_sub.add_parser(
        "coverage",
        help="List subsystem coverage rollups (docs.v_subsystem_coverage, read-only docs view).",
    )
    _add_common_repo_args(p_subsys_coverage)
    p_subsys_coverage.add_argument(
        "--limit", type=int, default=None, help="Limit subsystem coverage rows"
    )
    p_subsys_coverage.set_defaults(func=_cmd_subsystem_coverage)

    p_subsys_modules = subsys_sub.add_parser(
        "module-memberships",
        help="List subsystem memberships for a module (docs view, read-only).",
    )
    _add_common_repo_args(p_subsys_modules)
    p_subsys_modules.add_argument("--module", required=True, help="Module name (e.g., pkg.mod)")
    p_subsys_modules.set_defaults(func=_cmd_subsystem_modules)


def _add_pipeline_run_subparser(
    pipeline_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_run = pipeline_sub.add_parser(
        "run",
        help="Run the full pipeline via Prefect (targets are ignored; runs export_docs_flow).",
    )
    _add_common_repo_args(p_run)
    p_run.add_argument(
        "--target",
        action="append",
        default=None,
        help=(
            "Name of a pipeline step to run (can be specified multiple times). "
            "Defaults to 'export_docs' if omitted."
        ),
    )
    p_run.add_argument(
        "--skip-scip",
        action="store_true",
        help="Skip SCIP ingestion (sets CODEINTEL_SKIP_SCIP=true).",
    )
    p_run.add_argument(
        "--function-fail-on-missing-spans",
        action="store_true",
        help="Fail pipeline when function spans are missing or files cannot be parsed.",
    )
    p_run.add_argument(
        "--function-parser",
        choices=[kind.value for kind in FunctionParserKind],
        help="Optional parser selector for function analytics (e.g., 'python').",
    )
    p_run.add_argument(
        "--history-commit",
        dest="history_commits",
        action="append",
        help="Commit SHA to include in history_timeseries (can be repeated).",
    )
    p_run.add_argument(
        "--history-db-dir",
        type=Path,
        default=Path("build/db"),
        help="Directory containing per-commit DuckDB snapshots for history_timeseries.",
    )
    p_run.add_argument(
        "--export-dataset",
        dest="export_datasets",
        action="append",
        help="Dataset name to export during docs export step (can be repeated).",
    )
    p_run.add_argument(
        "--export-validation-profile",
        choices=["strict", "lenient"],
        help="Override validation profile for exports (default: dataset contract default).",
    )
    p_run.add_argument(
        "--force-full-export",
        action="store_true",
        help="Force re-export even when incremental markers match.",
    )
    _add_graph_backend_args(p_run)
    p_run.set_defaults(func=_cmd_pipeline_run)


def _add_pipeline_list_steps_subparser(
    pipeline_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_list_steps = pipeline_sub.add_parser(
        "list-steps",
        help="List all available pipeline steps with descriptions.",
    )
    p_list_steps.add_argument(
        "--phase",
        choices=["ingestion", "graphs", "analytics", "export"],
        help="Filter steps by phase.",
    )
    p_list_steps.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output as JSON for machine consumption.",
    )
    p_list_steps.set_defaults(func=_cmd_pipeline_list_steps)


def _add_pipeline_deps_subparser(
    pipeline_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_deps = pipeline_sub.add_parser(
        "deps",
        help="Show dependency tree for a pipeline step.",
    )
    p_deps.add_argument(
        "step_name",
        help="Name of the step to show dependencies for.",
    )
    p_deps.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Output as JSON for machine consumption.",
    )
    p_deps.set_defaults(func=_cmd_pipeline_deps)


def _register_pipeline_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    pipeline_parser = subparsers.add_parser("pipeline", help="Run enrichment pipeline")
    pipeline_sub = pipeline_parser.add_subparsers(dest="subcommand", required=True)
    _add_pipeline_run_subparser(pipeline_sub)
    _add_pipeline_list_steps_subparser(pipeline_sub)
    _add_pipeline_deps_subparser(pipeline_sub)


def _add_docs_export_subparser(
    docs_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_export = docs_sub.add_parser(
        "export",
        help="Export Parquet + JSONL datasets from DuckDB into Document Output/",
    )
    _add_common_repo_args(p_export)
    p_export.add_argument(
        "--validate",
        dest="validate_exports",
        action="store_true",
        help="Validate exported datasets against JSON Schema definitions.",
    )
    p_export.add_argument(
        "--schema",
        dest="schemas",
        action="append",
        help="Schema name to validate (can be repeated). Defaults to the standard export set.",
    )
    p_export.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Dataset name to export (can be repeated). Defaults to all mapped datasets.",
    )
    p_export.add_argument(
        "--require-normalized-macros",
        action="store_true",
        help="Fail if any requested dataset lacks a normalized macro.",
    )
    _add_graph_backend_args(p_export)
    p_export.set_defaults(func=cmd_docs_export)


def _register_docs_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_docs = subparsers.add_parser("docs", help="Document Output helpers")
    docs_sub = p_docs.add_subparsers(dest="subcommand", required=True)
    _add_docs_export_subparser(docs_sub)


def _register_storage_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_storage = subparsers.add_parser("storage", help="Storage validation helpers")
    storage_sub = p_storage.add_subparsers(dest="subcommand", required=True)
    p_validate_macros = storage_sub.add_parser(
        "validate-macros",
        help="Validate macro registry hashes and normalized macro schemas.",
    )
    p_validate_macros.add_argument(
        "--db-path",
        type=Path,
        default=Path("build/db/codeintel_prefect.duckdb"),
        help="Path to the DuckDB database to validate.",
    )
    p_validate_macros.add_argument(
        "--require-ingest-macros",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any ingest macros are missing (default: True).",
    )
    p_validate_macros.set_defaults(func=_cmd_storage_validate_macros)


def _register_history_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_history = subparsers.add_parser(
        "history-timeseries",
        help="Aggregate analytics.history_timeseries across commits.",
    )
    p_history.add_argument(
        "--repo-root",
        type=Path,
        default=Path(),
        help="Path to the repository root (default: current directory)",
    )
    p_history.add_argument(
        "--repo",
        required=True,
        help="Repository slug (e.g., 'my-org/my-repo')",
    )
    p_history.add_argument(
        "--commits",
        nargs="+",
        required=True,
        help="Commits to include in the timeseries (latest first).",
    )
    p_history.add_argument(
        "--db-dir",
        type=Path,
        default=Path("build/db"),
        help="Directory with per-commit DuckDB snapshots (codeintel-<commit>.duckdb).",
    )
    p_history.add_argument(
        "--output-db",
        type=Path,
        default=Path("build/db/history.duckdb"),
        help="Destination DuckDB for history_timeseries (will be created if missing).",
    )
    p_history.add_argument(
        "--entity-kind",
        choices=["function", "module", "both"],
        default="function",
        help="Entity kind to include in the history aggregation.",
    )
    p_history.add_argument(
        "--max-entities",
        type=int,
        default=500,
        help="Maximum entities to track (top-N by selection strategy).",
    )
    p_history.add_argument(
        "--selection-strategy",
        default="risk_score",
        help="Selection strategy for picking entities (default: risk_score).",
    )
    p_history.set_defaults(func=_cmd_history_timeseries)


def _register_ide_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p_ide = subparsers.add_parser("ide", help="IDE helper commands")
    ide_sub = p_ide.add_subparsers(dest="subcommand", required=True)

    p_ide_hints = ide_sub.add_parser(
        "hints",
        help="Emit IDE hints (module + subsystem context) for a relative file path.",
    )
    _add_common_repo_args(p_ide_hints)
    p_ide_hints.add_argument(
        "--rel-path",
        required=True,
        help="File path relative to repo root (e.g., pkg/module.py)",
    )
    p_ide_hints.set_defaults(func=_cmd_ide_hints)


def _register_scaffold_parser(
    ds_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = ds_sub.add_parser(
        "scaffold",
        help="Create a new dataset scaffold (TypedDict, schema, bindings, metadata).",
    )
    parser.add_argument("name", help="Logical dataset name (e.g., my_dataset).")
    parser.add_argument(
        "--kind",
        choices=["table", "view"],
        default="table",
        help="Dataset kind; views skip default export filenames (default: table).",
    )
    parser.add_argument(
        "--table-key",
        type=str,
        default=None,
        help="Fully qualified table key (default: analytics.<name> or docs.<name> for views).",
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=None,
        help="Owner/team for the dataset (optional).",
    )
    parser.add_argument(
        "--freshness-sla",
        type=str,
        default=None,
        help="Freshness expectation (e.g., daily, hourly).",
    )
    parser.add_argument(
        "--retention-policy",
        type=str,
        default=None,
        help="Retention policy string (e.g., 90d).",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default="1",
        help="Schema version identifier (default: 1).",
    )
    parser.add_argument(
        "--validation-profile",
        choices=["strict", "lenient"],
        default="strict",
        help="Validation profile to seed in metadata (default: strict).",
    )
    parser.add_argument(
        "--schema-id",
        type=str,
        default=None,
        help="JSON Schema identifier to create (default: <name>).",
    )
    parser.add_argument(
        "--jsonl-filename",
        type=str,
        default=None,
        help="Default JSONL filename (default: <name>.jsonl).",
    )
    parser.add_argument(
        "--parquet-filename",
        type=str,
        default=None,
        help="Default Parquet filename (default: <name>.parquet).",
    )
    parser.add_argument(
        "--stable-id",
        type=str,
        default=None,
        help="Stable identifier for contract diffs (default: <name>).",
    )
    parser.add_argument(
        "--specs-snapshot",
        type=Path,
        default=Path("build/catalog/dataset_specs.json"),
        help="Optional dataset specs snapshot to check for name/stable_id clashes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/dataset_scaffolds"),
        help="Directory to write scaffold artifacts (default: build/dataset_scaffolds).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the scaffold without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files.",
    )
    parser.add_argument(
        "--emit-bootstrap-snippet",
        action="store_true",
        help="Write a combined bootstrap snippet for metadata and bindings.",
    )
    parser.add_argument(
        "--check-registry",
        action="store_true",
        help="Validate against the live registry to catch name/stable_id/table clashes.",
    )
    parser.set_defaults(func=_cmd_datasets_scaffold)


def _register_dataset_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register dataset contract utility commands."""
    p_ds = subparsers.add_parser("datasets", help="Dataset contract utilities")
    ds_sub = p_ds.add_subparsers(dest="subcommand", required=True)

    p_lint = ds_sub.add_parser("lint", help="Validate dataset contract health")
    _add_common_repo_args(p_lint)
    p_lint.add_argument(
        "--schema-dir",
        type=Path,
        default=Path("src/codeintel/config/schemas/export"),
        help="Directory containing export JSON Schemas (default: src/codeintel/config/schemas/export)",
    )
    p_lint.add_argument(
        "--sample-rows",
        action="store_true",
        help="Validate a small sample of rows against JSON Schemas when available.",
    )
    p_lint.set_defaults(func=_cmd_datasets_lint)

    p_list = ds_sub.add_parser(
        "list",
        help="List datasets with capabilities and optional docs/read-only filters.",
    )
    _add_common_repo_args(p_list)
    p_list.add_argument(
        "--docs-view",
        choices=["include", "exclude", "only"],
        default="include",
        help="Filter docs.* views (default: include).",
    )
    p_list.add_argument(
        "--read-only",
        choices=["include", "exclude", "only"],
        default="include",
        help="Filter read-only datasets (views and docs) (default: include).",
    )
    p_list.add_argument(
        "--max-description",
        type=int,
        default=80,
        help="Maximum description length before truncation (default: 80).",
    )
    p_list.set_defaults(func=_cmd_datasets_list)

    p_diff = ds_sub.add_parser(
        "diff", help="Diff current dataset specs against a baseline JSON file"
    )
    _add_common_repo_args(p_diff)
    p_diff.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to a JSON baseline produced by `codeintel datasets snapshot` (optional when using --against-ref).",
    )
    p_diff.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the current specs snapshot.",
    )
    p_diff.add_argument(
        "--against-ref",
        type=str,
        default=None,
        help="Git ref to load a baseline snapshot from (with --baseline-path).",
    )
    p_diff.add_argument(
        "--baseline-path",
        type=Path,
        default=Path("build/dataset_specs.json"),
        help="Path (relative to repo root) of the snapshot inside the git ref (default: build/dataset_specs.json).",
    )
    p_diff.set_defaults(func=_cmd_datasets_diff)

    p_snapshot = ds_sub.add_parser(
        "snapshot", help="Write the current dataset specs to a JSON file"
    )
    _add_common_repo_args(p_snapshot)
    p_snapshot.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the snapshot JSON.",
    )
    p_snapshot.set_defaults(func=_cmd_datasets_snapshot)

    p_conf = ds_sub.add_parser(
        "conformance", help="Run full dataset conformance checks (optionally sample rows)"
    )
    _add_common_repo_args(p_conf)
    p_conf.add_argument(
        "--schema-dir",
        type=Path,
        default=Path("src/codeintel/config/schemas/export"),
        help="Directory containing export JSON Schemas (default: src/codeintel/config/schemas/export)",
    )
    p_conf.add_argument(
        "--sample-rows",
        action="store_true",
        help="Validate a sample of rows against JSON Schemas when available.",
    )
    p_conf.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of rows to sample per dataset when --sample-rows is set (default: 50).",
    )
    p_conf.set_defaults(func=_cmd_datasets_conformance)

    p_codegen = ds_sub.add_parser(
        "generate-schemas",
        help="Generate export JSON Schemas from TypedDict row models into a directory",
    )
    _add_common_repo_args(p_codegen)
    p_codegen.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/codeintel/config/schemas/export"),
        help="Directory to write generated schemas (default: src/codeintel/config/schemas/export)",
    )
    p_codegen.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset names to generate; defaults to all datasets with row bindings.",
    )
    p_codegen.set_defaults(func=_cmd_datasets_generate_schemas)

    p_catalog = ds_sub.add_parser(
        "catalog",
        help="Generate a Markdown/HTML dataset catalog from the registry.",
    )
    _add_common_repo_args(p_catalog)
    p_catalog.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/catalog"),
        help="Directory to write catalog artifacts (default: build/catalog).",
    )
    p_catalog.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of sample rows to include per dataset (default: 3, use 0 to skip).",
    )
    p_catalog.add_argument(
        "--sample-rows-strict",
        action="store_true",
        help="Fail if sampling cannot be performed instead of silently skipping.",
    )
    p_catalog.set_defaults(func=_cmd_datasets_catalog)

    _register_scaffold_parser(ds_sub)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeintel",
        description="CodeIntel metadata pipeline and export CLI",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_pipeline_commands(subparsers)
    _register_docs_commands(subparsers)
    _register_storage_commands(subparsers)
    _register_history_commands(subparsers)
    _register_ide_commands(subparsers)
    _add_subsystem_subparser(subparsers)
    _register_dataset_commands(subparsers)

    return parser


def make_parser() -> argparse.ArgumentParser:
    """
    Public helper to construct the CLI parser (for tests/tools).

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all subcommands registered.
    """
    return _make_parser()


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _build_config_from_args(args: argparse.Namespace) -> CodeIntelConfig:
    graph_backend = _build_graph_backend_config(args)
    graph_features = _build_graph_feature_flags_from_env()
    LOG.info(
        "cli.runtime.config repo=%s commit=%s backend=%s use_gpu=%s features=%s filters=%s",
        args.repo,
        args.commit,
        graph_backend.backend,
        graph_backend.use_gpu,
        graph_features,
        "n/a",
    )
    paths_cfg = CliPathsInput(
        repo_root=args.repo_root,
        build_dir=args.build_dir,
        db_path=args.db_path,
        document_output_dir=args.document_output_dir,
    )
    repo_cfg = RepoConfig(repo=args.repo, commit=args.commit)
    return CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=CliConfigOptions(graph_backend=graph_backend, graph_features=graph_features),
    )


def _build_graph_backend_config(args: argparse.Namespace) -> GraphBackendConfig:
    backend_value = str(getattr(args, "nx_backend", "auto"))
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    if backend_value == "cpu":
        backend = "cpu"
    elif backend_value == "nx-cugraph":
        backend = "nx-cugraph"
    return GraphBackendConfig(
        use_gpu=bool(getattr(args, "nx_gpu", False)),
        backend=backend,
        strict=bool(getattr(args, "nx_gpu_strict", False)),
    )


def _parse_env_flag(value: str | None, *, default: bool | None = None) -> bool | None:
    """
    Parse a boolean-ish environment string.

    Returns
    -------
    bool | None
        Parsed boolean value or the provided default when parsing fails.
    """
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _build_graph_feature_flags_from_env() -> GraphFeatureFlags:
    """
    Construct GraphFeatureFlags from CODEINTEL_* environment variables.

    Returns
    -------
    GraphFeatureFlags
        Feature flags derived from environment variables.
    """
    eager = (
        _parse_env_flag(os.environ.get("CODEINTEL_GRAPH_EAGER"))
        if "CODEINTEL_GRAPH_EAGER" in os.environ
        else None
    )
    community_limit = (
        int(os.environ["CODEINTEL_GRAPH_COMMUNITY_LIMIT"])
        if "CODEINTEL_GRAPH_COMMUNITY_LIMIT" in os.environ
        else None
    )
    validation_strict = (
        _parse_env_flag(os.environ.get("CODEINTEL_GRAPH_VALIDATION_STRICT"))
        if "CODEINTEL_GRAPH_VALIDATION_STRICT" in os.environ
        else None
    )
    return GraphFeatureFlags(
        eager_hydration=eager,
        community_detection_limit=community_limit,
        validation_strict=validation_strict,
    )


def _open_gateway(cfg: CodeIntelConfig, *, read_only: bool) -> StorageGateway:
    """
    Open a StorageGateway and ensure schemas/views exist for write mode.

    Parameters
    ----------
    cfg:
        CodeIntel configuration holding paths.
    read_only:
        Whether to open the database read-only.

    Returns
    -------
    StorageGateway
        Gateway bound to the configured DuckDB database.
    """
    cfg.paths.db_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = (
        StorageConfig.for_readonly(cfg.paths.db_path)
        if read_only
        else StorageConfig.for_ingest(cfg.paths.db_path)
    )
    gateway_cfg = replace(
        base_cfg,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
    )
    return open_gateway(gateway_cfg)


def _build_runtime(cfg: CodeIntelConfig, gateway: StorageGateway) -> GraphRuntime:
    """
    Construct a GraphRuntime for CLI commands.

    Returns
    -------
    GraphRuntime
        Runtime bound to the CLI snapshot and backend settings.
    """
    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    return build_graph_runtime(
        gateway,
        GraphRuntimeOptions(
            snapshot=snapshot,
            backend=cfg.graph_backend,
            features=cfg.graph_features,
        ),
    )


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    maybe_enable_nx_gpu(cfg.graph_backend)
    repo_root = cfg.paths.repo_root
    db_path = cfg.paths.db_path
    build_dir = cfg.paths.build_dir
    output_dir = cfg.paths.document_output_dir
    if output_dir is not None:
        os.environ.setdefault("CODEINTEL_OUTPUT_DIR", str(output_dir))

    if args.skip_scip:
        os.environ["CODEINTEL_SKIP_SCIP"] = "true"

    targets = list(args.target) if args.target else None
    LOG.info(
        "Running Prefect export_docs_flow for repo=%s commit=%s targets=%s",
        cfg.repo.repo,
        cfg.repo.commit,
        targets,
    )

    export_docs_flow(
        args=ExportArgs(
            repo_root=repo_root,
            repo=cfg.repo.repo,
            commit=cfg.repo.commit,
            db_path=db_path,
            build_dir=build_dir,
            tools=cfg.tools,
            code_profile=profile_from_env(default_code_profile(repo_root)),
            config_profile=profile_from_env(default_config_profile(repo_root)),
            function_fail_on_missing_spans=args.function_fail_on_missing_spans,
            function_parser=FunctionParserKind(args.function_parser)
            if args.function_parser
            else None,
            history_commits=tuple(args.history_commits) if args.history_commits else None,
            history_db_dir=args.history_db_dir,
            graph_backend=cfg.graph_backend,
            export_datasets=tuple(args.export_datasets) if args.export_datasets else None,
            export_validation_profile=getattr(args, "export_validation_profile", None),
            force_full_export=bool(getattr(args, "force_full_export", False)),
        ),
        targets=targets,
    )
    return 0


def _cmd_pipeline_list_steps(args: argparse.Namespace) -> int:
    """
    List all available pipeline steps with descriptions.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    phase_filter = getattr(args, "phase", None)
    output_json = getattr(args, "output_json", False)

    if phase_filter:
        phase = StepPhase(phase_filter)
        steps = REGISTRY.list_by_phase(phase)
    else:
        steps = REGISTRY.list_all()

    if output_json:
        data = [
            {
                "name": meta.name,
                "description": meta.description,
                "phase": meta.phase.value,
                "deps": list(meta.deps),
            }
            for meta in steps
        ]
        sys.stdout.write(json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        for meta in steps:
            deps_str = ", ".join(meta.deps) if meta.deps else "(none)"
            sys.stdout.write(f"{meta.name} [{meta.phase.value}]\n")
            sys.stdout.write(f"  {meta.description}\n")
            sys.stdout.write(f"  deps: {deps_str}\n")
            sys.stdout.write("\n")
    return 0


def _cmd_pipeline_deps(args: argparse.Namespace) -> int:
    """
    Show dependency tree for a pipeline step.

    Returns
    -------
    int
        Exit code (0 on success, 1 if step not found).
    """
    step_name = args.step_name
    output_json = getattr(args, "output_json", False)

    if step_name not in REGISTRY:
        LOG.error("Unknown step: %s", step_name)
        return 1

    # Get all transitive dependencies
    expanded = REGISTRY.expand_with_deps([step_name])
    expanded.discard(step_name)  # Remove the step itself from deps

    # Get direct deps
    direct_deps = tuple(REGISTRY.get_deps(step_name))

    if output_json:
        data = {
            "step": step_name,
            "direct_deps": list(direct_deps),
            "transitive_deps": sorted(expanded),
        }
        sys.stdout.write(json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        step = REGISTRY[step_name]
        sys.stdout.write(f"Step: {step_name}\n")
        sys.stdout.write(f"Description: {step.description}\n")
        sys.stdout.write(f"Phase: {step.phase.value}\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"Direct dependencies ({len(direct_deps)}):\n")
        if direct_deps:
            for dep in direct_deps:
                sys.stdout.write(f"  - {dep}\n")
        else:
            sys.stdout.write("  (none)\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"All transitive dependencies ({len(expanded)}):\n")
        if expanded:
            for dep in sorted(expanded):
                sys.stdout.write(f"  - {dep}\n")
        else:
            sys.stdout.write("  (none)\n")
    return 0


def _cmd_ide_hints(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.get_file_hints(rel_path=args.rel_path)
    if not response.found or not response.hints:
        LOG.error("No IDE hints found for %s", args.rel_path)
        return 1

    payload = {
        "rel_path": args.rel_path,
        "hints": [hint.model_dump() for hint in response.hints],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_list(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.list_subsystems(
        limit=args.limit,
        role=args.role,
        q=args.q,
    )
    payload = {
        "subsystems": [row.model_dump() for row in response.subsystems],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_show(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.get_subsystem_modules(subsystem_id=args.subsystem_id)
    if not response.found or response.subsystem is None:
        LOG.error("Subsystem not found: %s", args.subsystem_id)
        return 1
    payload = {
        "subsystem": response.subsystem.model_dump(),
        "modules": [row.model_dump() for row in response.modules],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_profiles(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.service.list_subsystem_profiles(limit=args.limit)
    payload = {
        "profiles": [row.model_dump() for row in response.profiles],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_coverage(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.service.list_subsystem_coverage(limit=args.limit)
    payload = {
        "coverage": [row.model_dump() for row in response.coverage],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_subsystem_modules(args: argparse.Namespace) -> int:
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    runtime = _build_runtime(cfg, gateway)
    engine = runtime.engine
    backend = DuckDBBackend(
        gateway=gateway,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        query_engine=engine,
    )
    response = backend.get_module_subsystems(module=args.module)
    payload = {
        "found": response.found,
        "memberships": [row.model_dump() for row in response.memberships],
        "meta": response.meta.model_dump(),
    }
    sys.stdout.write(json.dumps(payload))
    sys.stdout.write("\n")
    return 0


def _cmd_storage_validate_macros(args: argparse.Namespace) -> int:
    """
    Validate macro registry hashes and normalized macro schemas for a database.

    Returns
    -------
    int
        Exit code indicating success (0) or failure (non-zero).
    """
    cfg = StorageConfig.for_ingest(args.db_path)
    gateway = open_gateway(cfg)
    missing_ingest: list[str] = []
    error: RuntimeError | None = None
    try:
        _assert_macro_coverage()
        validate_macro_registry(gateway.con)
        validate_dataset_schema_registry(gateway.con)
        validate_normalized_macro_schemas(gateway.con)
        missing_ingest, present_ingest = ingest_macro_coverage(gateway.con)
        if missing_ingest:
            LOG.warning("Missing ingest macros: %s", ", ".join(missing_ingest))
        LOG.debug("Present ingest macros: %s", ", ".join(present_ingest))
        if args.require_ingest_macros and missing_ingest:
            message = ", ".join(missing_ingest)
            error = RuntimeError(f"Ingest macros missing: {message}")
    except RuntimeError as exc:
        error = exc
    if error is not None:
        LOG.error("Macro validation failed", exc_info=error)
        gateway.close()
        return 1
    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        LOG.info(
            "dataset_rows-only datasets (no normalized macro): %s",
            ", ".join(dataset_rows_list),
        )
    gateway.close()
    LOG.info("Macro validation passed.")
    return 0


def cmd_docs_export(
    args: argparse.Namespace,
    *,
    validator: Callable[[StorageGateway], None] = validate_dataset_registry,
    export_runner: ExportRunner = run_validated_exports,
    gateway_factory: GatewayFactory | None = None,
) -> int:
    """
    Export only, assuming the pipeline has already populated the DuckDB db.

    This is what `scripts/generate_documents.sh` should call:

        codeintel docs export --repo <slug> --commit <sha>

    Returns
    -------
    int
        Exit code (0 on success).

    Raises
    ------
    RuntimeError
        If document_output_dir could not be resolved.
    """
    cfg = _build_config_from_args(args)
    maybe_enable_nx_gpu(cfg.graph_backend)
    gateway = (gateway_factory or _open_gateway)(cfg, read_only=True)

    out_dir = cfg.paths.document_output_dir
    if out_dir is None:
        message = "document_output_dir was not resolved"
        raise RuntimeError(message)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Exporting Parquet + JSONL datasets into %s", out_dir)
    schemas = list(args.schemas) if getattr(args, "schemas", None) else None
    datasets = list(args.datasets) if getattr(args, "datasets", None) else None
    try:
        export_runner(
            gateway=gateway,
            output_dir=out_dir,
            options=ExportOptions(
                export=ExportCallOptions(
                    validate_exports=bool(getattr(args, "validate_exports", False)),
                    schemas=schemas,
                    datasets=datasets,
                    require_normalized_macros=bool(
                        getattr(args, "require_normalized_macros", False)
                    ),
                ),
                validator=validator,
            ),
        )
    except ExportError as exc:
        log_problem(LOG, exc.problem_detail)
        return 1

    LOG.info("Export complete.")
    return 0


def _cmd_history_timeseries(args: argparse.Namespace) -> int:
    """
    Compute analytics.history_timeseries across multiple commits.

    Returns
    -------
    int
        Exit code indicating success (0) or failure.
    """
    _setup_logging(args.verbose)
    runner = ToolRunner(cache_dir=args.repo_root / "build" / ".tool_cache")
    builder = ConfigBuilder.from_snapshot(
        repo=args.repo,
        commit=args.commits[0] if args.commits else "",
        repo_root=args.repo_root,
    )
    cfg = builder.history_timeseries(
        commits=tuple(args.commits),
        entity_kind=args.entity_kind,
        max_entities=args.max_entities,
        selection_strategy=args.selection_strategy,
    )

    storage_cfg = StorageConfig.for_ingest(args.output_db)
    gateway = open_gateway(storage_cfg)
    snapshot_resolver = build_snapshot_gateway_resolver(
        db_dir=args.db_dir,
        repo=args.repo,
        primary_gateway=gateway,
    )
    try:
        compute_history_timeseries_gateways(
            gateway,
            cfg,
            snapshot_resolver,
            runner=runner,
        )
    except FileNotFoundError:
        LOG.exception("Missing snapshot database for history_timeseries")
        return 1
    except DuckDBError:  # pragma: no cover - surfaced to caller
        LOG.exception("Failed to compute history_timeseries")
        return 1
    LOG.info(
        "history_timeseries written to %s for %d commits",
        args.output_db,
        len(args.commits),
    )
    return 0


def _format_capabilities(caps: dict[str, bool]) -> str:
    """
    Return a compact capability label string.

    Returns
    -------
    str
        Comma-separated capability labels or "-" when empty.
    """
    labels: list[str] = []
    if caps.get("docs_view"):
        labels.append("docs")
    if caps.get("read_only"):
        labels.append("ro")
    if caps.get("can_validate"):
        labels.append("validate")
    if caps.get("can_export_jsonl"):
        labels.append("jsonl")
    if caps.get("can_export_parquet"):
        labels.append("parquet")
    if caps.get("has_row_binding"):
        labels.append("binding")
    if caps.get("is_view") and not caps.get("docs_view"):
        labels.append("view")
    return ",".join(labels) if labels else "-"


ELLIPSIS_LEN = 3


def _caps_match(
    caps: dict[str, bool],
    *,
    docs_view_filter: str,
    read_only_filter: str,
) -> bool:
    """
    Apply docs/read-only filters to capability flags.

    Returns
    -------
    bool
        True when the dataset matches both filter settings.
    """
    is_docs = bool(caps.get("docs_view"))
    is_read_only = bool(caps.get("read_only"))
    docs_ok = (docs_view_filter != "only" or is_docs) and (
        docs_view_filter != "exclude" or not is_docs
    )
    read_only_ok = (read_only_filter != "only" or is_read_only) and (
        read_only_filter != "exclude" or not is_read_only
    )
    return docs_ok and read_only_ok


def _truncate(text: str, limit: int) -> str:
    """
    Truncate a string to a maximum length with an ellipsis.

    Returns
    -------
    str
        Original text when within the limit, otherwise a trimmed string.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= ELLIPSIS_LEN:
        return text[:limit]
    return text[: limit - ELLIPSIS_LEN] + "..."


def _cmd_datasets_list(args: argparse.Namespace) -> int:
    """
    List datasets with capabilities, optionally filtering docs and read-only entries.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)
    max_desc = int(args.max_description)
    rows: list[tuple[str, str, str, str, str]] = []
    for name, ds in sorted(registry.by_name.items()):
        caps = ds.capabilities()
        if not _caps_match(
            caps,
            docs_view_filter=args.docs_view,
            read_only_filter=args.read_only,
        ):
            continue
        rows.append(
            (
                name,
                ds.table_key,
                ds.family or "",
                _format_capabilities(caps),
                _truncate(ds.description or "", max_desc),
            )
        )

    if not rows:
        sys.stdout.write("No datasets matched the requested filters.\n")
        return 0

    headers = ("name", "table", "family", "caps", "description")
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _fmt(row: tuple[str, ...]) -> str:
        parts: list[str] = []
        for idx, value in enumerate(row):
            if idx == len(row) - 1:
                parts.append(value)
            else:
                parts.append(value.ljust(widths[idx]))
        return "  ".join(parts)

    sys.stdout.write(_fmt(headers) + "\n")
    for row in rows:
        sys.stdout.write(_fmt(row) + "\n")
    return 0


def _cmd_datasets_lint(args: argparse.Namespace) -> int:
    """
    Validate dataset contract for the configured database.

    Returns
    -------
    int
        Exit code (0 on success, 1 on failures).
    """
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    issues = collect_contract_issues(gateway.con, schema_base_dir=args.schema_dir)
    if args.sample_rows:
        # Row-sampling handled by conformance runner; keep lint fast by omitting here.
        sys.stderr.write("Row sampling requested; run `codeintel datasets conformance` instead.\n")
        return 2
    if issues:
        for issue in issues:
            sys.stderr.write(f"{issue}\n")
        return 1
    sys.stdout.write("Dataset contract validation passed.\n")
    return 0


def _cmd_datasets_snapshot(args: argparse.Namespace) -> int:
    """
    Write the current dataset specs to a JSON snapshot file.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    specs = list_dataset_specs(load_dataset_registry(gateway.con))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2), encoding="utf-8")
    sys.stdout.write(f"Wrote dataset specs to {args.output}\n")
    return 0


def _cmd_datasets_conformance(args: argparse.Namespace) -> int:
    """
    Run conformance checks including contract validation and optional row sampling.

    Returns
    -------
    int
        Exit code (0 on success, 1 on issues, 2 on invalid input).
    """
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    try:
        report = run_conformance(
            gateway.con,
            schema_base_dir=args.schema_dir,
            sample_rows=bool(args.sample_rows),
            sample_size=int(args.sample_size),
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Conformance run failed: {exc}\n")
        return 2
    if not report.ok:
        for issue in report.issues:
            prefix = issue.dataset or "global"
            sys.stderr.write(f"[{prefix}] {issue.message}\n")
        return 1
    sys.stdout.write("Dataset conformance passed.\n")
    return 0


def _cmd_datasets_generate_schemas(args: argparse.Namespace) -> int:
    """
    Generate JSON Schemas from TypedDict row models for export validation.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)
    include = set(args.datasets) if args.datasets else None
    written = generate_export_schemas(
        registry,
        output_dir=args.output_dir,
        include_datasets=include,
    )
    if not written:
        sys.stdout.write("No schemas generated (no matching datasets with row bindings).\n")
        return 0
    sys.stdout.write(f"Wrote {len(written)} schemas to {args.output_dir}\n")
    return 0


def _cmd_datasets_catalog(args: argparse.Namespace) -> int:
    """
    Generate a Markdown/HTML catalog from the dataset registry.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    db_path: Path = args.db_path
    warnings_seen: set[str] = set()

    def _warn(msg: str) -> None:
        if msg in warnings_seen:
            return
        warnings_seen.add(msg)
        sys.stderr.write(msg + "\n")

    if not db_path.exists():
        if args.sample_rows_strict:
            sys.stderr.write(f"Database not found at {db_path}; cannot generate catalog.\n")
            return 1
        _warn(f"Database not found at {db_path}; writing empty catalog artifacts.")
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        md = write_markdown_catalog(output_dir, [])
        html = write_html_catalog(output_dir, [])
        sys.stdout.write(f"Wrote catalog: {md}, {html}\n")
        return 0

    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    registry = load_dataset_registry(gateway.con)
    try:
        entries = build_catalog(
            registry,
            con=gateway.con,
            sample_rows=int(args.sample_rows),
            sample_rows_strict=bool(args.sample_rows_strict),
            warn=_warn,
        )
    except Exception as exc:  # noqa: BLE001 - surface sampling failures when strict
        sys.stderr.write(f"Failed to generate catalog samples: {exc}\n")
        return 1
    output_dir = args.output_dir
    md = write_markdown_catalog(output_dir, entries)
    html = write_html_catalog(output_dir, entries)
    sys.stdout.write(f"Wrote catalog: {md}, {html}\n")
    return 0


def run_datasets_catalog(args: argparse.Namespace) -> int:
    """
    Public wrapper to generate the catalog (primarily for tests/tools).

    Returns
    -------
    int
        Exit code (0 on success).
    """
    return _cmd_datasets_catalog(args)


class ScaffoldConfigError(Exception):
    """Configuration error while building scaffold options."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def build_scaffold_options(
    args: argparse.Namespace, *, registry: DatasetRegistry | None = None
) -> ScaffoldOptions:
    """
    Construct scaffold options from CLI arguments with guardrails.

    Returns
    -------
    ScaffoldOptions
        Validated scaffold options derived from CLI arguments.

    Raises
    ------
    ScaffoldConfigError
        When validation fails.
    """
    name = args.name
    kind = str(getattr(args, "kind", "table"))
    table_key = args.table_key or f"{'docs' if kind == 'view' else 'analytics'}.{name}"
    schema_id = args.schema_id or name
    stable_id = args.stable_id or name
    jsonl_filename = args.jsonl_filename or (None if kind == "view" else f"{name}.jsonl")
    parquet_filename = args.parquet_filename or (None if kind == "view" else f"{name}.parquet")
    existing_schema = Path("src/codeintel/config/schemas/export") / f"{schema_id}.json"
    overwrite = bool(args.overwrite)
    if existing_schema.exists() and not overwrite:
        message = f"Schema already exists: {existing_schema}"
        raise ScaffoldConfigError(message, exit_code=1)
    specs_snapshot: Path = args.specs_snapshot
    if specs_snapshot.exists() and not overwrite:
        try:
            specs = json.loads(specs_snapshot.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            message = f"Failed to parse specs snapshot {specs_snapshot}: {exc}"
            raise ScaffoldConfigError(message, exit_code=2) from exc
        names = {str(spec.get("name")) for spec in specs}
        stable_ids = {str(spec.get("stable_id")) for spec in specs if "stable_id" in spec}
        if name in names:
            message = f"Dataset name already present in snapshot: {name}"
            raise ScaffoldConfigError(message, 1)
        if stable_id in stable_ids:
            message = f"Stable ID already present in snapshot: {stable_id}"
            raise ScaffoldConfigError(message, exit_code=1)

    if registry is not None:
        if name in registry.by_name:
            message = f"Dataset name already present in registry: {name}"
            raise ScaffoldConfigError(message, exit_code=1)
        if stable_id in {ds.stable_id for ds in registry.by_name.values() if ds.stable_id}:
            message = f"Stable ID already present in registry: {stable_id}"
            raise ScaffoldConfigError(message, exit_code=1)
        if table_key in registry.by_table_key:
            message = f"Table key already present in registry: {table_key}"
            raise ScaffoldConfigError(message, exit_code=1)
    return ScaffoldOptions(
        name=name,
        table_key=table_key,
        owner=args.owner,
        freshness_sla=args.freshness_sla,
        retention_policy=args.retention_policy,
        schema_version=args.schema_version,
        stable_id=stable_id,
        validation_profile=cast("Literal['strict', 'lenient']", args.validation_profile),
        jsonl_filename=jsonl_filename,
        parquet_filename=parquet_filename,
        schema_id=schema_id,
        output_dir=args.output_dir,
        is_view=kind == "view",
        overwrite=overwrite,
        dry_run=bool(args.dry_run),
        emit_bootstrap_snippet=bool(args.emit_bootstrap_snippet),
    )


def _cmd_datasets_scaffold(args: argparse.Namespace) -> int:
    """
    Scaffold a new dataset contract skeleton.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    registry: DatasetRegistry | None = None
    if getattr(args, "check_registry", False):
        cfg = _build_config_from_args(args)
        gateway = _open_gateway(cfg, read_only=True)
        registry = load_dataset_registry(gateway.con)
    try:
        opts = build_scaffold_options(args, registry=registry)
    except ScaffoldConfigError as exc:
        sys.stderr.write(f"{exc}\n")
        return exc.exit_code
    result = scaffold_dataset(opts)
    sys.stdout.write(
        "Scaffold plan:\n"
        f"  TypedDict: {result.typed_dict}\n"
        f"  Row binding snippet: {result.row_binding}\n"
        f"  JSON Schema: {result.json_schema}\n"
        f"  Metadata: {result.metadata}\n"
        f"  Bootstrap snippet: {result.bootstrap_snippet}\n"
    )
    if opts.dry_run:
        sys.stdout.write("Dry-run only; no files were written.\n")
    return 0


def _cmd_datasets_diff(args: argparse.Namespace) -> int:
    """
    Diff current dataset specs against a baseline snapshot.

    Returns
    -------
    int
        Exit code (0 on no differences, 1 when differences found or errors occurred).
    """
    cfg = _build_config_from_args(args)
    gateway = _open_gateway(cfg, read_only=True)
    current_specs = list_dataset_specs(load_dataset_registry(gateway.con))
    baseline_specs: list[dict[str, object]] = []
    if args.against_ref:
        baseline_specs = _load_specs_from_ref(
            repo_root=cfg.paths.repo_root,
            ref=args.against_ref,
            snapshot_path=args.baseline_path,
        )
    elif args.baseline is not None:
        if not args.baseline.exists():
            sys.stderr.write(f"Baseline file not found: {args.baseline}\n")
            return 1
        baseline_specs = json.loads(args.baseline.read_text(encoding="utf-8"))
    else:
        sys.stderr.write("Provide either --baseline or --against-ref\n")
        return 2
    added, removed, changed = _diff_specs(current_specs, baseline_specs)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(current_specs, indent=2), encoding="utf-8")

    if not (added or removed or changed):
        sys.stdout.write("No dataset spec differences detected.\n")
        return 0

    if added:
        sys.stdout.write(f"Added datasets: {', '.join(added)}\n")
    if removed:
        sys.stdout.write(f"Removed datasets: {', '.join(removed)}\n")
    if changed:
        sys.stdout.write(f"Changed datasets: {', '.join(changed)}\n")
    return 1


def _load_specs_from_ref(
    *, repo_root: Path, ref: str, snapshot_path: Path
) -> list[dict[str, object]]:
    """
    Load dataset specs snapshot from a git ref and path.

    Parameters
    ----------
    repo_root
        Repository root for running git commands.
    ref
        Git reference (commit SHA, branch, or tag).
    snapshot_path
        Path to the snapshot file inside the repository at the ref.

    Returns
    -------
    list[dict[str, object]]
        Parsed dataset specs JSON content from the referenced snapshot.

    Raises
    ------
    RuntimeError
        When the snapshot cannot be loaded from the provided ref/path.
    """
    target = f"{ref}:{snapshot_path.as_posix()}"
    runner = ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    result = runner.run("git", ["show", target], cwd=repo_root)
    if result.returncode != 0:
        message = f"Failed to load snapshot from {target}: {result.stderr.strip()}"
        raise RuntimeError(message)
    return json.loads(result.stdout)


def _diff_specs(
    current_specs: list[dict[str, object]],
    baseline_specs: list[dict[str, object]],
) -> tuple[list[str], list[str], list[str]]:
    """
    Compute added/removed/changed dataset names between two spec sets.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Added, removed, and changed dataset names.
    """
    baseline_by_name: dict[str, dict[str, object]] = {
        str(spec.get("name")): spec for spec in baseline_specs
    }
    current_by_name: dict[str, dict[str, object]] = {
        str(spec.get("name")): spec for spec in current_specs
    }
    added = sorted(set(current_by_name) - set(baseline_by_name))
    removed = sorted(set(baseline_by_name) - set(current_by_name))
    changed = sorted(
        name
        for name in current_by_name
        if name in baseline_by_name and current_by_name[name] != baseline_by_name[name]
    )
    return added, removed, changed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    """
    CLI entrypoint for the CodeIntel pipeline and export commands.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to sys.argv).

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _make_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    _setup_logging(args.verbose)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        func: CommandHandler = args.func
        return int(func(args))
    except Exception as exc:  # noqa: BLE001 pragma: no cover - error path
        pd = problem(
            code="cli.failure",
            title="CLI command failed",
            detail=str(exc),
            extras={"command": args.command},
        )
        log_problem(LOG, pd)
        return 1


if __name__ == "__main__":
    sys.exit(main())
