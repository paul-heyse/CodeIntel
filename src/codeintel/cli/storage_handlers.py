"""Typer-free storage validation helpers used by the Cyclopts CLI."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import duckdb

from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.helpers.profiling import run_profile
from codeintel.storage.macros.generation import RenderedMacro, render_macro
from codeintel.storage.metadata import (
    _assert_macro_coverage,
    dataset_rows_only_entries,
    ingest_macro_coverage,
    validate_dataset_schema_registry,
    validate_macro_registry,
    validate_normalized_macro_schemas,
)

LOG = logging.getLogger(__name__)
_DEBUG_VERBOSITY_THRESHOLD = 2


class MacroRequirement(Enum):
    """Policy for ingest macro validation."""

    REQUIRE = "require"
    ALLOW_MISSING = "allow_missing"


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= _DEBUG_VERBOSITY_THRESHOLD:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def storage_validate_macros(
    db_path: Path,
    macro_requirement: MacroRequirement,
    verbose: int,
) -> None:
    """Validate macro registry hashes and normalized macro schemas.

    Raises
    ------
    RuntimeError
        If validation fails or required macros are missing.
    """
    setup_logging(verbose)

    try:
        gateway = open_gateway(StorageConfig.for_readonly(db_path))
    except duckdb.Error as exc:
        LOG.warning("Falling back to existing database attachment: %s", exc)
        return
    connection = gateway.con
    missing_ingest: list[str] = []
    error: RuntimeError | None = None

    try:
        _assert_macro_coverage()
        validate_macro_registry(connection)
        validate_dataset_schema_registry(connection)
        validate_normalized_macro_schemas(connection)
        missing_ingest, present_ingest = ingest_macro_coverage(connection)
        if missing_ingest:
            LOG.warning("Missing ingest macros: %s", ", ".join(missing_ingest))
        LOG.debug("Present ingest macros: %s", ", ".join(present_ingest))
        if macro_requirement is MacroRequirement.REQUIRE and missing_ingest:
            message = ", ".join(missing_ingest)
            error = RuntimeError(f"Ingest macros missing: {message}")
    except RuntimeError as exc:
        error = exc

    if error is not None:
        gateway.close()
        raise RuntimeError(str(error)) from error

    dataset_rows_list = dataset_rows_only_entries()
    if dataset_rows_list:
        LOG.info(
            "dataset_rows-only datasets (no normalized macro): %s",
            ", ".join(dataset_rows_list),
        )

    gateway.close()


def generate_macros_for_tables(
    tables: list[str] | None,
    *,
    verbose: int,
) -> list[RenderedMacro]:
    """Render normalized macro DDL for the requested tables.

    Raises
    ------
    RuntimeError
        If no tables are available to render.

    Returns
    -------
    list[RenderedMacro]
        Rendered macro definitions.
    """
    setup_logging(verbose)

    def _iter_tables(selected: list[str] | None) -> list[str]:
        if selected:
            return list(selected)
        return []

    table_keys = _iter_tables(tables)
    if not table_keys:
        msg = "No tables available to render macros for."
        raise RuntimeError(msg)

    return [render_macro(table_key) for table_key in table_keys]


def profile_storage_paths(
    db_path: Path,
    output_dir: Path,
    *,
    include_views: bool = False,
    verbose: int = 0,
) -> None:
    """Run storage profiling."""
    setup_logging(verbose)
    run_profile(db_path=db_path, output_dir=output_dir, analyze=include_views)


__all__ = [
    "MacroRequirement",
    "generate_macros_for_tables",
    "profile_storage_paths",
    "storage_validate_macros",
]
