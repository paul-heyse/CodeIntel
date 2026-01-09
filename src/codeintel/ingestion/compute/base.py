"""Base types for ingestion compute layer.

This module defines common types used by all ingestion compute modules,
analogous to base types in graphs/compute/.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.expr_vocab import E, Expression
from codeintel.core.columnar.finalize_ops import (
    FinalizeDedupe,
    FinalizeMode,
    FinalizeResult,
    FinalizeSpec,
    finalize_table,
)
from codeintel.core.columnar.plan_ops import build_query_plan, materialize_plan
from codeintel.core.columnar.queryspec import ProjectionSpec, QuerySpec
from codeintel.core.query_results import records_from_arrow_table
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.ingestion.infrastructure.py_frontend import PyFrontend
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
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


def persist_arrow_tables(
    storage: IngestStoragePort | None,
    tables: Mapping[str, pa.Table | pa.RecordBatchReader],
    *,
    scope: str | None = None,
) -> None:
    """Persist Arrow tables when a storage port is provided."""
    if storage is None:
        return
    for table_key, payload in tables.items():
        if isinstance(payload, pa.RecordBatchReader):
            storage.write_reader(table_key, payload, scope=scope)
            continue
        if payload.num_rows == 0:
            continue
        storage.write_table(table_key, payload, scope=scope)


def finalize_arrow_tables(
    tables: Mapping[str, pa.Table],
    *,
    mode: FinalizeMode = "tolerant",
) -> tuple[dict[str, pa.Table], list[str]]:
    """Finalize Arrow tables against their contracts in tolerant mode.

    Returns
    -------
    tuple[dict[str, pyarrow.Table], list[str]]
        Finalized tables keyed by table_key plus warning messages.
    """
    finalized: dict[str, pa.Table] = {}
    warnings: list[str] = []
    for table_key, table in tables.items():
        spec = FinalizeSpec(
            table_key=table_key,
            mode=mode,
            required_non_null=_required_non_null_columns(table_key),
            dedupe=FinalizeDedupe(enabled=False),
            emit_artifacts=True,
        )
        try:
            result = finalize_table(table, spec=spec)
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
    mode: FinalizeMode = "tolerant",
) -> tuple[dict[str, pa.Table], list[str]]:
    """Finalize Arrow readers against their contracts in tolerant mode.

    Returns
    -------
    tuple[dict[str, pyarrow.Table], list[str]]
        Finalized tables keyed by table_key plus warning messages.
    """
    finalized: dict[str, pa.Table] = {}
    warnings: list[str] = []
    for table_key, reader in readers.items():
        spec = FinalizeSpec(
            table_key=table_key,
            mode=mode,
            required_non_null=_required_non_null_columns(table_key),
            dedupe=FinalizeDedupe(enabled=False),
            emit_artifacts=True,
        )
        table = reader_to_table(reader)
        table = _apply_ingest_query_plan(table, table_key=table_key)
        try:
            result = finalize_table(table, spec=spec)
        except ValueError as exc:
            warnings.append(f"{table_key}: {exc}")
            finalized[table_key] = table
            continue
        finalized[table_key] = result.good
        warnings.extend(_finalize_warnings(table_key, result))
    return finalized, warnings


def _apply_ingest_query_plan(table: pa.Table, *, table_key: str) -> pa.Table:
    spec = build_ingest_query_spec(table_key)
    try:
        dataset = ds.dataset(table)
        plan = build_query_plan(dataset, spec=spec)
        return materialize_plan(plan, use_threads=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return table


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


def build_ingest_query_spec(
    table_key: str,
    *,
    columns: Sequence[str] | None = None,
    repo: str | None = None,
    commit: str | None = None,
    rel_path: str | None = None,
) -> QuerySpec:
    """Build an ingestion-friendly QuerySpec for repo/commit/rel_path scoping.

    Returns
    -------
    QuerySpec
        Query specification with optional repo/commit/path filtering.
    """
    resolved_columns = _resolve_query_columns(table_key, columns)
    predicate = _ingest_scope_predicate(
        column_names=set(resolved_columns),
        repo=repo,
        commit=commit,
        rel_path=rel_path,
    )
    projection = ProjectionSpec(base_cols=tuple(resolved_columns))
    return QuerySpec(
        predicate=predicate,
        pushdown_predicate=predicate,
        projection=projection,
    )


def _resolve_query_columns(table_key: str, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    schema = get_schema_service().get_table_schema(table_key)
    if schema is None:
        return []
    return list(schema.column_names())


def _ingest_scope_predicate(
    *,
    column_names: set[str],
    repo: str | None,
    commit: str | None,
    rel_path: str | None,
) -> Expression | None:
    exprs: list[Expression] = []
    if repo is not None and "repo" in column_names:
        exprs.append(E.field("repo") == E.scalar(repo))
    if commit is not None and "commit" in column_names:
        exprs.append(E.field("commit") == E.scalar(commit))
    if rel_path is not None and "rel_path" in column_names:
        exprs.append(E.field("rel_path") == E.scalar(rel_path))
    if not exprs:
        return None
    return E.and_(*exprs)


def _required_non_null_columns(table_key: str) -> tuple[str, ...]:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return ()
    if schema is None:
        return ()
    return tuple(column.name for column in schema.columns if not column.nullable)


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
    "persist_arrow_tables",
]
