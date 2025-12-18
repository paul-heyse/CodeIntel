"""DuckDB data savers for Hamilton materialization.

This module implements Hamilton ``DataSaver`` adapters that persist computed
node outputs into DuckDB, making I/O visible in the DAG.
"""

from __future__ import annotations

import time
import types
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, get_args, get_origin

import ibis.expr.types as ir
from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.metadata import DuckDBMaterializationMetadata
from codeintel.build.hamilton.run_records import should_skip_native_target
from codeintel.build.hashing import compute_input_hash
from codeintel.build.targets import TargetGraph
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

if TYPE_CHECKING:
    import pandas as pd

SaveStatus = Literal["succeeded", "skipped", "failed"]


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
    - Computes the target input hash (manifest key) from the graph + env.
    - Applies manifest-based skip (authoritative for artifact writes).
    - Writes the table for the current snapshot using ``Warehouse``.
    - Optionally validates the output against Pandera schema when enabled.
    - Returns a metadata dict (as required by Hamilton's DataSaver API).

    Notes
    -----
    Hamilton materializer nodes should not raise exceptions. This saver catches
    errors and returns ``status="failed"`` with an error message so downstream
    target nodes can return a ``TargetRunRecord`` rather than crashing the run.
    """

    env: BuildEnv
    graph: TargetGraph
    target_name: str
    table_key: str

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

    def save_data(self, data: object) -> dict[str, Any]:
        """Save the provided data and return metadata describing the write.

        Parameters
        ----------
        data
            Data value produced by the upstream compute node.

        Returns
        -------
        dict[str, Any]
            Metadata describing the write, including status, row count, and
            input hash for manifest-based incremental builds.
        """
        start = time.perf_counter()
        input_hash: str | None = None
        result: dict[str, Any] | None = None
        try:
            if not isinstance(self.env, BuildEnv):
                result = _failed(
                    table_key=self.table_key,
                    duration_ms=_duration_ms(start),
                    input_hash="",
                    error=f"Expected env to be BuildEnv, got {type(self.env).__name__}",
                )
            elif not isinstance(self.graph, TargetGraph):
                result = _failed(
                    table_key=self.table_key,
                    duration_ms=_duration_ms(start),
                    input_hash="",
                    error=f"Expected graph to be TargetGraph, got {type(self.graph).__name__}",
                )
            else:
                target = self.graph.get(self.target_name)
                if target is None:
                    result = _failed(
                        table_key=self.table_key,
                        duration_ms=_duration_ms(start),
                        input_hash="",
                        error=f"Target not found in graph: {self.target_name}",
                    )
                else:
                    input_hash = compute_input_hash(
                        target=target,
                        snapshot=self.env.snapshot,
                        gateway=self.env.gateway,
                        options_hash=None,
                        manifests=self.env.manifest_index,
                    )

                    if should_skip_native_target(self.env, target, input_hash):
                        result = _skipped(
                            table_key=self.table_key,
                            duration_ms=_duration_ms(start),
                            input_hash=input_hash,
                            row_count=_manifest_row_count(self.env, target_name=self.target_name),
                        )
                    elif data is None:
                        result = _skipped(
                            table_key=self.table_key,
                            duration_ms=_duration_ms(start),
                            input_hash=input_hash,
                            row_count=None,
                        )
                    elif not isinstance(data, ir.Table):
                        result = _failed(
                            table_key=self.table_key,
                            duration_ms=_duration_ms(start),
                            input_hash=input_hash,
                            error=f"Expected ir.Table, got {type(data).__name__}",
                        )
                    else:
                        # Validate contract if strict mode is enabled
                        ContractEnforcer.validate_table_write(self.table_key)

                        warehouse = Warehouse(self.env.gateway)
                        row_count = _materialize_table(
                            warehouse,
                            table_key=self.table_key,
                            table=data,
                            validate=self.env.validate_outputs,
                            options=MaterializeOptions(
                                snapshot=self.env.snapshot,
                                mode="replace",
                                owner_target=self.target_name,
                                input_hash=input_hash,
                            ),
                        )

                        result = _succeeded(
                            table_key=self.table_key,
                            duration_ms=_duration_ms(start),
                            input_hash=input_hash,
                            row_count=row_count,
                        )

        except _RECOVERABLE_EXCEPTIONS as exc:
            result = _failed(
                table_key=self.table_key,
                duration_ms=_duration_ms(start),
                input_hash=input_hash or "",
                error=str(exc),
            )
        if result is None:
            return _failed(
                table_key=self.table_key,
                duration_ms=_duration_ms(start),
                input_hash=input_hash or "",
                error="Unknown materialization failure",
            )
        return result


def _duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _succeeded(
    *, table_key: str, duration_ms: float, input_hash: str, row_count: int
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    return DuckDBMaterializationMetadata(
        status="skipped",
        table_key=table_key,
        row_count=row_count,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=None,
    ).to_dict()


def _failed(*, table_key: str, duration_ms: float, input_hash: str, error: str) -> dict[str, Any]:
    return DuckDBMaterializationMetadata(
        status="failed",
        table_key=table_key,
        row_count=None,
        duration_ms=duration_ms,
        input_hash=input_hash,
        error=error,
    ).to_dict()


def _manifest_row_count(env: BuildEnv, *, target_name: str) -> int | None:
    index = env.manifest_index
    if index is None:
        return None
    manifest = index.get(target_name)
    if manifest is None:
        return None
    return manifest.row_count


def _materialize_table(
    warehouse: Warehouse,
    *,
    table_key: str,
    table: ir.Table,
    validate: bool,
    options: MaterializeOptions,
) -> int:
    if not validate:
        result = warehouse.materialize_table(table_key, table, options=options)
        return result.rows_written or 0

    schema = get_pandera_schema(table_key)
    if schema is None:
        result = warehouse.materialize_table(table_key, table, options=options)
        return result.rows_written or 0

    snapshot = options.snapshot
    if snapshot is None:
        msg = "Validated DuckDB materialization requires a snapshot"
        raise ValueError(msg)

    df = cast("pd.DataFrame", table.execute())
    validated = schema.validate(df, lazy=False)
    warehouse.delete_for_snapshot(table_key, snapshot=snapshot)
    result = warehouse.materialize_dataframe(
        table_key,
        validated,
        options=MaterializeOptions(
            snapshot=snapshot,
            mode="append",
            owner_target=options.owner_target,
            input_hash=options.input_hash,
        ),
    )
    return result.rows_written or 0


__all__ = [
    "DuckDBIbisTableSaver",
    "SaveStatus",
]
