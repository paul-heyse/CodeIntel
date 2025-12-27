"""DuckDB relation saver for Hamilton materialization."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, get_args, get_origin

import duckdb
import polars as pl
import pyarrow as pa
from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.build.hamilton.materializers.write_policy import resolve_materialize_options
from codeintel.build.tabular.duckdb_relation import register_ephemeral
from codeintel.core.execution.materialization import failed_table_result, succeeded_table_result
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
)

_TABULAR_TYPES: tuple[type, ...] = (
    duckdb.DuckDBPyRelation,
    pa.Table,
    pa.RecordBatchReader,
    pl.DataFrame,
    pl.LazyFrame,
)


@dataclass(frozen=True)
class DuckDBRelationSaver(DataSaver):
    """Persist relation-like tabular outputs to DuckDB.

    This saver:
    - Resolves target metadata from the DAG catalog.
    - Coerces tabular inputs to DuckDB relations.
    - Writes the relation for the current snapshot using ``Warehouse``.
    - Returns metadata convertible to a ``MaterializationResult`` describing the outcome.
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
            Adapter name used by Hamilton for saver metadata.
        """
        return "codeintel.duckdb_relation"

    @classmethod
    def applicable_types(cls) -> list[type]:
        """Return types this saver can persist.

        Returns
        -------
        list[type]
            Types that this saver can write to DuckDB.
        """
        return list(_TABULAR_TYPES)

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
            if args.issubset(set(_TABULAR_TYPES) | {type(None)}):
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
        temp_name: str | None = None

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
                        error="Expected relation data but received None",
                    )
                else:
                    relation, temp_name = _coerce_relation(
                        self.env,
                        data=data,
                        table_key=self.table_key,
                    )
                    ContractEnforcer.validate_table_write(self.table_key)
                    options = resolve_materialize_options(
                        env=self.env,
                        target_name=self.target_name,
                        table_key=self.table_key,
                        input_hash=input_hash,
                        column_names=tuple(relation.columns),
                    )
                    row_count = _materialize_relation(
                        self.env.warehouse,
                        table_key=self.table_key,
                        relation=relation,
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
        finally:
            if temp_name is not None:
                self.env.gateway.unregister(temp_name)

        if result is None:
            result = failed_table_result(
                table_key=self.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown materialization failure",
            )
        return result.to_mapping()


def _coerce_relation(
    env: BuildEnv,
    *,
    data: object,
    table_key: str,
) -> tuple[duckdb.DuckDBPyRelation, str | None]:
    """Coerce a tabular input into a DuckDB relation.

    Parameters
    ----------
    env
        Build environment providing the DuckDB gateway.
    data
        Tabular input object to coerce.
    table_key
        Table key used for temp name prefixing.

    Returns
    -------
    tuple[duckdb.DuckDBPyRelation, str | None]
        DuckDB relation plus optional registered temp name.

    Raises
    ------
    TypeError
        If the input type is not supported.
    """
    if isinstance(data, duckdb.DuckDBPyRelation):
        return data, None
    if isinstance(data, _TABULAR_TYPES):
        temp_name = register_ephemeral(env.gateway.con, data, prefix=table_key)
        return env.gateway.con.table(temp_name), temp_name
    msg = f"Unsupported relation input type: {type(data).__name__}"
    raise TypeError(msg)


def _materialize_relation(
    warehouse: Warehouse,
    *,
    table_key: str,
    relation: duckdb.DuckDBPyRelation,
    options: MaterializeOptions,
) -> int:
    result = warehouse.materialize_table(table_key, relation, options=options)
    return result.rows_written or 0


__all__ = ["DuckDBRelationSaver"]
