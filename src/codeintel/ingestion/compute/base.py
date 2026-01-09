"""Base types for ingestion compute layer.

This module defines common types used by all ingestion compute modules,
analogous to base types in graphs/compute/.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.arrowdsl import ExecutionPlan, PipelineRunOptions, run_pipeline
from codeintel.core.columnar.conversion import empty_table_from_schema
from codeintel.core.columnar.execution_context import ExecutionContext, resolve_execution_context
from codeintel.core.columnar.finalize_ops import (
    FinalizeDedupe,
    FinalizeMode,
    FinalizeResult,
    finalize_spec_for_table,
)
from codeintel.core.columnar.ordering import OrderingSpec
from codeintel.core.columnar.run_manifest import RunManifestOptions
from codeintel.core.columnar.streaming import ScanTelemetry
from codeintel.core.query_results import records_from_arrow_table
from codeintel.core.schemas.service import get_schema_service
from codeintel.ingestion.compute import queryspecs as ingest_queryspecs

build_ingest_query_spec = ingest_queryspecs.build_ingest_query_spec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.ingestion.infrastructure.py_frontend import PyFrontend
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.tools import IngestToolPort


class BaseExtractStep:
    """Base class for module extraction steps with port injection.

    Provides shared initialization and helper methods for steps that:

    - Iterate over Python modules and read source

    Parameters
    ----------
    discovery
        Discovery port for reading module source.
    frontend
        Optional shared frontend cache for source and AST reuse.
    """

    _discovery: ModuleDiscoveryPort
    _frontend: PyFrontend | None

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        frontend: PyFrontend | None = None,
    ) -> None:
        """Initialize the step with discovery ports.

        Parameters
        ----------
        discovery
            Discovery port for reading module source.
        frontend
            Optional shared frontend cache for source and AST reuse.
        """
        self._discovery = discovery
        self._frontend = frontend

    def _iter_python_sources(
        self, modules: Sequence[ModuleRecord]
    ) -> Iterator[tuple[ModuleRecord, str]]:
        """Yield (module, source) pairs for Python files with readable source.

        Parameters
        ----------
        modules
            Sequence of module records to iterate.

        Yields
        ------
        tuple[ModuleRecord, str]
            Module record and its source code for each readable Python file.
        """
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            if self._frontend is not None:
                source = self._frontend.get_source_text(module)
            else:
                source = self._discovery.read_module_source(module)
            if source is not None:
                yield module, source


class BaseToolIngestStep:
    """Base class for ingestion steps requiring tool execution.

    Provides shared initialization for steps that need tool ports.

    Parameters
    ----------
    tools
        Tool port for running external tools.
    """

    _tools: IngestToolPort

    def __init__(
        self,
        tools: IngestToolPort,
    ) -> None:
        """Initialize the step with tool ports.

        Parameters
        ----------
        tools
            Tool port for running external tools.
        """
        self._tools = tools


@dataclass(frozen=True, slots=True)
class FinalizeArrowRequest:
    """Inputs for finalizing Arrow data against contracts."""

    mode: FinalizeMode = "tolerant"
    ctx: ExecutionContext | None = None
    manifest_dir: Path | None = None
    manifest_options: RunManifestOptions | None = None
    scan_telemetry: ScanTelemetry | None = None


def finalize_arrow_tables(
    tables: Mapping[str, pa.Table],
    *,
    request: FinalizeArrowRequest,
) -> tuple[dict[str, pa.Table], list[str]]:
    """Finalize Arrow tables against their contracts in tolerant mode.

    Returns
    -------
    tuple[dict[str, pyarrow.Table], list[str]]
        Finalized tables keyed by table_key plus warning messages.
    """
    finalized: dict[str, pa.Table] = {}
    warnings: list[str] = []
    resolved_ctx = resolve_execution_context(request.ctx)
    ordering = OrderingSpec.implicit(reason="ingest table")
    for table_key, table in tables.items():
        spec = finalize_spec_for_table(
            table_key,
            mode=request.mode,
            dedupe=FinalizeDedupe(enabled=False),
            emit_artifacts=True,
        )
        plan = ExecutionPlan.from_table(table, ordering=ordering)
        try:
            result = run_pipeline(
                plan=plan,
                finalize=spec,
                options=PipelineRunOptions(
                    ctx=resolved_ctx,
                    manifest_dir=request.manifest_dir,
                    manifest_options=request.manifest_options,
                    scan_telemetry=request.scan_telemetry,
                ),
            )
        except ValueError as exc:
            warnings.append(f"{table_key}: {exc}")
            finalized[table_key] = table
            continue
        finalized[table_key] = result.good
        warnings.extend(_finalize_warnings(table_key, result))
    return finalized, warnings


def finalize_arrow_readers(
    readers: Mapping[str, pa.RecordBatchReader],
    *,
    request: FinalizeArrowRequest,
) -> tuple[dict[str, pa.Table], list[str]]:
    """Finalize Arrow readers against their contracts in tolerant mode.

    Returns
    -------
    tuple[dict[str, pyarrow.Table], list[str]]
        Finalized tables keyed by table_key plus warning messages.
    """
    finalized: dict[str, pa.Table] = {}
    warnings: list[str] = []
    resolved_ctx = resolve_execution_context(request.ctx)
    ordering = OrderingSpec.implicit(reason="ingest reader")
    for table_key, reader in readers.items():
        spec = finalize_spec_for_table(
            table_key,
            mode=request.mode,
            dedupe=FinalizeDedupe(enabled=False),
            emit_artifacts=True,
        )
        plan = ExecutionPlan.from_reader(reader, ordering=ordering)
        try:
            result = run_pipeline(
                plan=plan,
                finalize=spec,
                options=PipelineRunOptions(
                    ctx=resolved_ctx,
                    manifest_dir=request.manifest_dir,
                    manifest_options=request.manifest_options,
                    scan_telemetry=request.scan_telemetry,
                ),
            )
        except ValueError as exc:
            warnings.append(f"{table_key}: {exc}")
            finalized[table_key] = empty_table_from_schema(reader.schema)
            continue
        finalized[table_key] = result.good
        warnings.extend(_finalize_warnings(table_key, result))
    return finalized, warnings


def build_typed_extras(
    table_key: str,
    extras: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Build a typed extras struct aligned to the table schema.

    Returns
    -------
    dict[str, object] | None
        Typed extras mapping when the schema defines a struct.
    """
    if not extras:
        return None
    arrow_schema = get_schema_service().get_arrow_schema(table_key)
    if arrow_schema is None or "extras" not in arrow_schema.names:
        return dict(extras)
    extras_field = arrow_schema.field("extras")
    if not pa.types.is_struct(extras_field.type):
        return dict(extras)
    typed: dict[str, object] = {}
    for field in extras_field.type:
        value = extras.get(field.name)
        if (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)) and isinstance(
            value, (tuple, set)
        ):
            value = list(value)
        typed[field.name] = value
    return typed


def _finalize_warnings(table_key: str, result: FinalizeResult) -> list[str]:
    warnings: list[str] = []
    if result.stats.num_rows:
        for row in records_from_arrow_table(result.stats):
            code = row.get("error_code")
            count = row.get("count")
            if isinstance(code, str):
                warnings.append(f"{table_key}: finalize error {code}: {count} rows")
            else:
                warnings.append(f"{table_key}: finalize error {row}")
    if result.alignment.num_rows:
        records = records_from_arrow_table(result.alignment)
        if records:
            row = records[0]
            warnings.append(
                f"{table_key}: finalize alignment missing={row.get('missing_columns')} "
                f"extra={row.get('extra_columns')} coerced={row.get('coerced_columns')}"
            )
    return warnings


__all__ = [
    "BaseExtractStep",
    "BaseToolIngestStep",
    "build_ingest_query_spec",
    "build_typed_extras",
    "finalize_arrow_readers",
    "finalize_arrow_tables",
]
