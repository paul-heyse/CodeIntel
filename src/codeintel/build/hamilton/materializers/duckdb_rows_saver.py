"""DuckDB data saver for row-oriented materialization.

This module implements a Hamilton ``DataSaver`` that persists a sequence of
row tuples into DuckDB for a specific snapshot. It is used by targets that
compute structured row data (rather than Ibis expressions) and want DAG-visible
I/O via Hamilton's saver nodes.
"""

from __future__ import annotations

import time
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast, get_args, get_origin

import pandas as pd
from hamilton.io.data_adapters import DataSaver

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materialize_options import materialize_options
from codeintel.build.hamilton.materializers.metadata import DuckDBMaterializationMetadata
from codeintel.build.hamilton.run_records import options_hash_for_target, should_skip_native_target
from codeintel.build.hashing import compute_input_hash
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
    columns: tuple[str, ...]

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
        """
        start = time.perf_counter()
        input_hash: str | None = None
        result: MaterializationMetadata | None = None

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
                    options_hash = options_hash_for_target(self.env, self.target_name)
                    input_hash = compute_input_hash(
                        target=target,
                        snapshot=self.env.snapshot,
                        gateway=self.env.gateway,
                        options_hash=options_hash,
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
                    else:
                        rows = _coerce_rows(data)
                        # Validate contract if strict mode is enabled
                        ContractEnforcer.validate_table_write(self.table_key)

                        warehouse = self.env.warehouse
                        row_count = _materialize_rows(
                            warehouse,
                            _RowsMaterializationRequest(
                                table_key=self.table_key,
                                rows=rows,
                                columns=self.columns,
                                validate=self.env.validate_outputs,
                                options=materialize_options(
                                    self.env,
                                    owner_target=self.target_name,
                                    mode="replace",
                                    input_hash=input_hash,
                                ),
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
                error="Unknown row materialization failure",
            )

        return result


def _duration_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


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


def _manifest_row_count(env: BuildEnv, *, target_name: str) -> int | None:
    index = env.manifest_index
    if index is None:
        return None
    manifest = index.get(target_name)
    if manifest is None:
        return None
    return manifest.row_count


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

    snapshot = request.options.snapshot
    if snapshot is None:
        msg = "Validated DuckDB row materialization requires a snapshot"
        raise ValueError(msg)

    frame = pd.DataFrame.from_records(request.rows, columns=list(request.columns))
    validated = schema.validate(frame, lazy=False)
    warehouse.delete_for_snapshot(request.table_key, snapshot=snapshot)
    result = warehouse.materialize_dataframe(
        request.table_key,
        validated,
        options=MaterializeOptions(
            snapshot=snapshot,
            mode="append",
            owner_target=request.options.owner_target,
            input_hash=request.options.input_hash,
        ),
    )
    return result.rows_written or 0


__all__ = [
    "DuckDBRowsSaver",
    "SaveStatus",
]
