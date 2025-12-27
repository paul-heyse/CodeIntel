"""DuckDB data savers for Hamilton materialization.

This module implements Hamilton ``DataSaver`` adapters that persist computed
node outputs into DuckDB, making I/O visible in the DAG.
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast, get_args, get_origin

import ibis.expr.types as ir
import pandas as pd
from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.build.hamilton.materializers.write_policy import resolve_materialize_options
from codeintel.core.execution.materialization import (
    failed_table_result,
    succeeded_table_result,
)
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
)


@dataclass(frozen=True)
class DuckDBIbisTableSaver(DataSaver):
    """Persist an Ibis table expression to DuckDB for a specific snapshot.

    This adapter:
    - Resolves target metadata from the DAG catalog.
    - Writes the table for the current snapshot using ``Warehouse``.
    - Optionally validates the output against Pandera schema when enabled.
    - Returns metadata convertible to a MaterializationResult describing the write outcome.

    Notes
    -----
    Hamilton materializer nodes should not raise exceptions. This saver catches
    errors and returns ``status="failed"`` with an error message so downstream
    target nodes can return a ``TargetRunRecord`` rather than crashing the run.
    """

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    table_key: str
    output_role: Literal["contract", "internal"] | None = None

    @classmethod
    def name(cls) -> str:
        """Return a stable name for this saver adapter.

        Returns
        -------
        str
            Adapter name used by Hamilton for materializer metadata.
        """
        return "codeintel.duckdb_table"

    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return types this saver can persist.

        Returns
        -------
        list[type]
            Types that this saver can write to DuckDB.
        """
        return [ir.Table]

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
        if origin in {types.UnionType, typing.Union}:
            args = set(get_args(type_))
            if ir.Table in args and args.issubset({ir.Table, type(None)}):
                return True
        return super().applies_to(type_)

    def save_data(self, data: object) -> dict[str, object]:
        """Save the provided data and return metadata describing the write.

        Parameters
        ----------
        data
            Data value produced by the upstream compute node.

        Returns
        -------
        dict[str, object]
            Metadata describing the write, including status, row count, and
            input hash for manifest-based incremental builds.
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
                        error="Expected table data but received None",
                    )
                elif not isinstance(data, ir.Table):
                    result = failed_table_result(
                        table_key=self.table_key,
                        duration_ms=duration_ms(start),
                        input_hash=input_hash or "",
                        error=f"Expected ir.Table, got {type(data).__name__}",
                    )
                else:
                    # Validate contract if strict mode is enabled
                    ContractEnforcer.validate_table_write(self.table_key)

                    warehouse = self.env.warehouse
                    options = resolve_materialize_options(
                        env=self.env,
                        target_name=self.target_name,
                        table_key=self.table_key,
                        input_hash=input_hash,
                        column_names=None,
                    )
                    row_count = _materialize_table(
                        warehouse,
                        table_key=self.table_key,
                        table=data,
                        validate=self.env.validate_outputs,
                        options=options,
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
                error="Unknown materialization failure",
            )
        return result.to_mapping()


def _materialize_table(
    warehouse: Warehouse,
    *,
    table_key: str,
    table: ir.Table,
    validate: bool,
    options: MaterializeOptions,
) -> int:
    if not validate:
        result = warehouse.materialize_ibis(table_key, table, options=options)
        return result.rows_written or 0

    schema = get_pandera_schema(table_key)
    if schema is None:
        result = warehouse.materialize_ibis(table_key, table, options=options)
        return result.rows_written or 0

    df = cast("pd.DataFrame", table.execute())
    validated = schema.validate(df, lazy=False)
    result = warehouse.materialize_dataframe(table_key, validated, options=options)
    return result.rows_written or 0


__all__ = ["DuckDBIbisTableSaver"]
