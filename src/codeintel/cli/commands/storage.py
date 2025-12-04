"""Storage validation commands for the CodeIntel CLI.

This module provides Typer commands for validating storage metadata,
macro registries, and schema integrity.

Commands
--------
- **validate-macros**: Validate macro registry hashes and normalized schemas
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from codeintel.cli.commands._common import VerboseOpt, setup_logging
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.metadata import (
    _assert_macro_coverage,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)

LOG = logging.getLogger(__name__)

storage_app = typer.Typer(
    name="storage",
    help="Storage validation utilities.",
    no_args_is_help=True,
)


# -----------------------------------------------------------------------------
# Option Type Aliases
# -----------------------------------------------------------------------------

DbPathArg = Annotated[
    Path,
    typer.Option(
        "--db-path",
        help="Path to the DuckDB database to validate.",
    ),
]

RequireIngestMacrosOpt = Annotated[
    bool,
    typer.Option(
        "--require-ingest-macros/--no-require-ingest-macros",
        help="Fail if any ingest macros are missing (default: enabled).",
    ),
]


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------


@storage_app.command("validate-macros")
def storage_validate_macros(
    db_path: DbPathArg = Path("build/db/codeintel_prefect.duckdb"),
    require_ingest_macros: RequireIngestMacrosOpt = True,
    verbose: VerboseOpt = 0,
) -> None:
    """Validate macro registry hashes and normalized macro schemas.

    Checks that all registered macros have valid hashes and that the
    normalized schemas are consistent with the raw macro definitions.

    Examples
    --------
    .. code-block:: bash

        # Validate default database
        codeintel storage validate-macros

        # Validate specific database
        codeintel storage validate-macros --db-path build/db/my.duckdb

        # Allow missing ingest macros
        codeintel storage validate-macros --no-require-ingest-macros
    """
    setup_logging(verbose)

    cfg = StorageConfig.for_ingest(db_path)
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
        if require_ingest_macros and missing_ingest:
            message = ", ".join(missing_ingest)
            error = RuntimeError(f"Ingest macros missing: {message}")
    except RuntimeError as exc:
        error = exc

    if error is not None:
        LOG.error("Macro validation failed", exc_info=error)
        gateway.close()
        typer.secho(f"Macro validation failed: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        LOG.info(
            "dataset_rows-only datasets (no normalized macro): %s",
            ", ".join(dataset_rows_list),
        )

    gateway.close()
    LOG.info("Macro validation passed.")
    typer.secho("Macro validation passed.", fg=typer.colors.GREEN)


# -----------------------------------------------------------------------------
# generate-macros command
# -----------------------------------------------------------------------------


@storage_app.command("generate-macros")
def storage_generate_macros(
    tables: Annotated[
        list[str] | None,
        typer.Argument(help="Table keys to render (defaults to all with schemas)."),
    ] = None,
    verbose: VerboseOpt = 0,
) -> None:
    """Generate normalized macro DDL for dataset tables.

    Renders DuckDB macro definitions for the specified tables (or all tables
    with schemas) to stdout. Useful for authoring and updating normalized macros.

    Examples
    --------
    .. code-block:: bash

        # Generate macros for all tables
        codeintel storage generate-macros

        # Generate macro for specific table
        codeintel storage generate-macros core.functions

        # Generate multiple tables
        codeintel storage generate-macros core.functions analytics.metrics
    """
    import sys

    from codeintel.config.datasets import get_dataset_contracts_by_table_key
    from codeintel.storage.macros.generation import render_macro

    setup_logging(verbose)

    def _iter_tables(selected: list[str] | None) -> list[str]:
        if selected:
            return list(selected)
        return sorted(
            table_key
            for table_key, contract in get_dataset_contracts_by_table_key().items()
            if contract.schema is not None and not contract.is_view
        )

    table_keys = _iter_tables(tables)
    if not table_keys:
        typer.secho("No tables found with schemas.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=0)

    for table_key in table_keys:
        try:
            macro = render_macro(table_key)
            sys.stdout.write(macro.ddl)
            sys.stdout.write("\n\n")
        except KeyError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc


# -----------------------------------------------------------------------------
# profile-views command
# -----------------------------------------------------------------------------

OutputDirOpt = Annotated[
    Path,
    typer.Option(
        "--output-dir",
        help="Directory to write profiling artifacts.",
    ),
]

AnalyzeOpt = Annotated[
    bool,
    typer.Option(
        "--analyze/--no-analyze",
        help="Use EXPLAIN ANALYZE instead of EXPLAIN.",
    ),
]


@storage_app.command("profile-views")
def storage_profile_views(
    db_path: DbPathArg = Path("build/db/codeintel.duckdb"),
    output_dir: OutputDirOpt = Path("build/profiling"),
    analyze: AnalyzeOpt = False,
    verbose: VerboseOpt = 0,
) -> None:
    """Generate EXPLAIN plans for docs views.

    Profiles the query plans for docs.v_subsystem_profile and
    docs.v_subsystem_coverage views, writing artifacts to the output directory.

    Examples
    --------
    .. code-block:: bash

        # Profile default database
        codeintel storage profile-views

        # Profile with EXPLAIN ANALYZE
        codeintel storage profile-views --analyze

        # Custom paths
        codeintel storage profile-views --db-path my.duckdb --output-dir profiles/
    """
    from codeintel.storage.helpers.profiling import run_profile

    setup_logging(verbose)

    try:
        run_profile(db_path=db_path, output_dir=output_dir, analyze=analyze)
        typer.secho(f"Profiling artifacts written to {output_dir}", fg=typer.colors.GREEN)
    except FileNotFoundError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(f"Profiling failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["storage_app"]
