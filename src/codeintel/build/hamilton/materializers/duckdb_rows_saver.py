"""DuckDB data saver for row-oriented materialization.

This module implements a Hamilton ``DataSaver`` that persists a sequence of
row tuples into DuckDB for a specific snapshot. It is used by targets that
compute structured row data (rather than Ibis expressions) and want DAG-visible
I/O via Hamilton's saver nodes.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast, get_args, get_origin

import pandas as pd
from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    manifest_row_count,
    resolve_materialization_context,
)
from codeintel.build.hamilton.materializers.metadata import DuckDBMaterializationMetadata
from codeintel.build.hamilton.materializers.write_policy import resolve_materialize_options
from codeintel.build.hashing import InputHashOptions
from codeintel.build.schemas.column_resolution import DeferredColumns, resolve_columns
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.targets import TargetGraph
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

SaveStatus = Literal["succeeded", "skipped", "failed"]

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
)


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
    - Computes the target input hash (manifest key) from the graph + env.
    - Applies manifest-based skip (authoritative for artifact writes).
    - Writes rows for the current snapshot using ``Warehouse``.
    - Optionally validates the produced dataframe against Pandera schema.
    - Returns a metadata dict (as required by Hamilton's DataSaver API).
    """

    env: BuildEnv
    graph: TargetGraph
    target_name: str
    table_key: str
    columns: tuple[str, ...] | DeferredColumns
    hash_options: InputHashOptions | None = None
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

    def save_data(self, data: object) -> MaterializationMetadata:
        """Save the provided rows and return metadata describing the write.

        Parameters
        ----------
        data
            Data value produced by the upstream compute node. Must be a sequence
            of row tuples (or None to indicate no output).

        Returns
        -------
        MaterializationMetadata
            Metadata describing the write, including status and input hash for
            manifest-based incremental builds.

        Raises
        ------
        ValueError
            If the provided data does not contain row tuples.
        """
        start = perf_counter()
        input_hash: str | None = None
        result: MaterializationMetadata | None = None

        try:
            prepared = resolve_materialization_context(
                env=self.env,
                graph=self.graph,
                target_name=self.target_name,
                hash_options=self.hash_options,
            )
            if isinstance(prepared, MaterializationContextError):
                result = _failed(
                    table_key=self.table_key,
                    duration_ms=duration_ms(start),
                    input_hash=prepared.input_hash or "",
                    error=prepared.message,
                )
            else:
                context = prepared
                input_hash = context.input_hash
                if context.should_skip:
                    result = _skipped(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash,
                        row_count=manifest_row_count(self.env, target_name=self.target_name),
                    )
                elif data is None:
                    result = _skipped(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash,
                        row_count=None,
                    )
                else:
                    rows = _coerce_rows(data)
                    # Validate contract if strict mode is enabled
                    ContractEnforcer.validate_table_write(self.table_key)

                    warehouse = self.env.warehouse
                    resolved_columns = resolve_columns(
                        self.columns,
                        schema_service=get_schema_service(),
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

                    result = _succeeded(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash,
                        row_count=row_count,
                    )

        except _RECOVERABLE_EXCEPTIONS as exc:
            result = _failed(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=str(exc),
            )

        if result is None:
            return _failed(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown row materialization failure",
            )

        return result


def _succeeded(
    *, table_key: str, duration_ms: float, input_hash: str, row_count: int
) -> MaterializationMetadata:
    return DuckDBMaterializationMetadata(
        status="succeeded",
        table_key=table_key,
        row_count=row_count,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=None,
    ).to_dict()


def _skipped(
    *,
    table_key: str,
    duration_ms: float,
    input_hash: str,
    row_count: int | None,
) -> MaterializationMetadata:
    return DuckDBMaterializationMetadata(
        status="skipped",
        table_key=table_key,
        row_count=row_count,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=None,
    ).to_dict()


def _failed(
    *, table_key: str, duration_ms: float, input_hash: str, error: str
) -> MaterializationMetadata:
    return DuckDBMaterializationMetadata(
        status="failed",
        table_key=table_key,
        row_count=None,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=error,
    ).to_dict()


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
    if not request.validate:
        result = warehouse.materialize_rows(
            request.table_key, request.rows, columns=request.columns, options=request.options
        )
        return result.rows_written or 0

    schema = get_pandera_schema(request.table_key)
    if schema is None:
        result = warehouse.materialize_rows(
            request.table_key, request.rows, columns=request.columns, options=request.options
        )
        return result.rows_written or 0

    frame = pd.DataFrame.from_records(request.rows, columns=list(request.columns))
    validated = schema.validate(frame, lazy=False)
    result = warehouse.materialize_dataframe(
        request.table_key,
        validated,
        options=request.options,
    )
    return result.rows_written or 0


__all__ = [
    "DuckDBRowsSaver",
    "SaveStatus",
]
