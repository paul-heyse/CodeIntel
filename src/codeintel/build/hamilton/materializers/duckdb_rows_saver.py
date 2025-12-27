"""DuckDB data saver for row-oriented materialization.

This module implements a Hamilton ``DataSaver`` that persists a sequence of
row tuples into DuckDB for a specific snapshot. It is used by targets that
compute structured row data (rather than DuckDB relations) and want DAG-visible
I/O via Hamilton's saver nodes.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast, get_args, get_origin

from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.build.hamilton.materializers.write_policy import resolve_materialize_options
from codeintel.build.schemas.column_resolution import DeferredColumns, resolve_columns
from codeintel.core.execution.materialization import (
    failed_table_result,
    succeeded_table_result,
)
from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas import SchemaService
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
)


def _schema_service() -> SchemaService:
    get_service = cast(
        "Callable[[], SchemaService]",
        lazy_getattr("codeintel.build.schemas.service", "get_schema_service"),
    )
    return get_service()


@dataclass(frozen=True)
class _RowsMaterializationRequest:
    """Bundle parameters for row materialization to satisfy strict style gates."""

    table_key: str
    rows: tuple[tuple[object, ...], ...]
    columns: tuple[str, ...]
    validate: bool
    options: MaterializeOptions


@dataclass(frozen=True)
class DuckDBRowsSaver(DataSaver):
    """Persist row tuples to DuckDB for a specific snapshot.

    This adapter:
    - Resolves target metadata from the DAG catalog.
    - Writes rows for the current snapshot using ``Warehouse``.
    - Returns metadata convertible to a MaterializationResult describing the write outcome.
    """

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    table_key: str
    columns: tuple[str, ...] | DeferredColumns
    output_role: Literal["contract", "internal"] | None = None

    @classmethod
    def name(cls) -> str:
        """Return a stable name for this saver adapter.

        Returns
        -------
        str
            Adapter name used by Hamilton for saver metadata.
        """
        return "codeintel.duckdb_rows"

    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return types this saver can persist.

        Returns
        -------
        list[type]
            Types that this saver can write to DuckDB.
        """
        return [tuple, list]

    @classmethod
    def applies_to(cls, type_: type) -> bool:
        """Return True when this saver can handle the Hamilton node output type.

        Parameters
        ----------
        type_
            Hamilton node output type.

        Returns
        -------
        bool
            True when the saver can persist the output type.
        """
        origin = get_origin(type_)
        if origin in {tuple, list}:
            return True

        if origin in {types.UnionType, typing.Union}:
            bases = {get_origin(arg) or arg for arg in get_args(type_)}
            if bases.issubset({tuple, list, type(None)}):
                return True

        return super().applies_to(type_)

    def save_data(self, data: object) -> dict[str, object]:
        """Save the provided rows and return metadata describing the write.

        Parameters
        ----------
        data
            Data value produced by the upstream compute node. Must be a sequence
            of row tuples (or None to indicate no output).

        Returns
        -------
        dict[str, object]
            Metadata describing the write and materialization outcome.

        Raises
        ------
        ValueError
            If the provided data does not contain row tuples.
        """
        start = perf_counter()
        input_hash: str | None = None
        result: MaterializationResult | None = None

        try:
            prepared = resolve_materialization_context(
                env=self.env,
                catalog=self.catalog,
                target_name=self.target_name,
            )
            if isinstance(prepared, MaterializationContextError):
                result = failed_table_result(
                    table_key=self.table_key,
                    duration_ms=duration_ms(start),
                    input_hash=prepared.input_hash or "",
                    error=prepared.message,
                )
            else:
                context = prepared
                input_hash = context.input_hash
                if data is None:
                    result = failed_table_result(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        error="Expected row data but received None",
                    )
                else:
                    rows = _coerce_rows(data)
                    warehouse = self.env.warehouse
                    resolved_columns = resolve_columns(
                        self.columns,
                        schema_service=_schema_service(),
                    )
                    if not resolved_columns:
                        msg = f"Missing column order for {self.table_key}"
                        raise ValueError(msg)
                    options = resolve_materialize_options(
                        env=self.env,
                        target_name=self.target_name,
                        table_key=self.table_key,
                        input_hash=input_hash,
                        column_names=resolved_columns,
                    )
                    row_count = _materialize_rows(
                        warehouse,
                        _RowsMaterializationRequest(
                            table_key=self.table_key,
                            rows=rows,
                            columns=resolved_columns,
                            validate=self.env.validate_outputs,
                            options=options,
                        ),
                    )

                    result = succeeded_table_result(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        row_count=row_count,
                    )

        except _RECOVERABLE_EXCEPTIONS as exc:
            result = failed_table_result(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=str(exc),
            )

        if result is None:
            result = failed_table_result(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown row materialization failure",
            )

        return result.to_mapping()


def _coerce_rows(data: object) -> tuple[tuple[object, ...], ...]:
    """Coerce a Hamilton node output into an immutable row tuple sequence.

    Parameters
    ----------
    data
        Value produced by an upstream node.

    Returns
    -------
    tuple[tuple[object, ...], ...]
        Rows as an immutable tuple of row tuples.

    Raises
    ------
    TypeError
        If the provided data is not a sequence of row tuples.
    """
    if isinstance(data, tuple):
        # Common case: tuple[tuple[...], ...]
        if not data:
            return ()
        if isinstance(data[0], tuple):
            return cast("tuple[tuple[object, ...], ...]", data)

    if isinstance(data, list):
        return tuple(_coerce_rows(tuple(data)))

    if isinstance(data, Sequence):
        rows_list: list[tuple[object, ...]] = []
        for row in data:
            if not isinstance(row, tuple):
                msg = f"Expected rows to be tuples, got {type(row).__name__}"
                raise TypeError(msg)
            rows_list.append(row)
        return tuple(rows_list)

    msg = f"Expected rows to be a sequence, got {type(data).__name__}"
    raise TypeError(msg)


def _materialize_rows(warehouse: Warehouse, request: _RowsMaterializationRequest) -> int:
    result = warehouse.materialize_rows(
        request.table_key, request.rows, columns=request.columns, options=request.options
    )
    return result.rows_written or 0


__all__ = ["DuckDBRowsSaver"]
