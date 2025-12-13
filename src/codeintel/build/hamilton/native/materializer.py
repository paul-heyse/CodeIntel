"""DuckDB materializer for native Hamilton targets.

This module provides utilities for materializing Ibis expressions to DuckDB
tables with proper snapshot isolation (repo + commit filtering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.storage.tracking.asset_tracking import AssetRecord

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class MaterializationContext:
    """Context for materializing Ibis tables.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Snapshot identity (repo + commit) for isolation.
    validate
        When True, validate produced outputs against Pandera schemas.
    owner_target
        Optional target name that produced these assets (for asset catalog).
    input_hash
        Optional input hash from manifest (for asset catalog).
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    validate: bool = False
    owner_target: str | None = None
    input_hash: str | None = None


def materialize_table(
    ctx: MaterializationContext,
    table_key: str,
    expr: ir.Table,
) -> DatasetRef:
    """Materialize an Ibis expression to a DuckDB table for this snapshot.

    This function writes the expression to DuckDB, replacing any existing
    data for this repo/commit combination to ensure snapshot isolation.

    Parameters
    ----------
    ctx
        Materialization context with gateway and snapshot info.
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    expr
        Ibis table expression to materialize.

    Returns
    -------
    DatasetRef
        Reference to the materialized dataset with row count.

    Raises
    ------
    ValueError
        If validation is enabled and the Pandera schema validation fails.

    Examples
    --------
    >>> from codeintel.config.primitives import SnapshotRef
    >>> snapshot = SnapshotRef(repo="example", commit="abc123")
    >>> # Assume gateway and expr are set up
    >>> ctx = MaterializationContext(gateway=gateway, snapshot=snapshot)
    >>> ref = materialize_table(ctx, "analytics.my_table", expr)
    >>> ref.table_key
    'analytics.my_table'
    >>> ref.repo
    'example'
    """
    # Validate contract if strict mode is enabled
    ContractEnforcer.validate_table_write(table_key)

    ctx.gateway.policy.delete_for_snapshot(
        table_key,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )

    schema = get_pandera_schema(table_key) if ctx.validate else None

    if schema is not None:
        df = pd.DataFrame(expr.execute())
        try:
            validated = schema.validate(df, lazy=False)
        except Exception as exc:
            msg = f"Schema validation failed for {table_key}: {exc}"
            raise ValueError(msg) from exc

        ctx.gateway.ibis.write(table_key, validated)
        row_count = len(validated)
    else:
        count_value = expr.count().execute()
        if isinstance(count_value, pd.DataFrame):
            row_count = int(count_value.iloc[0, 0])
        elif isinstance(count_value, pd.Series):
            row_count = int(count_value.iloc[0])
        else:
            row_count = int(count_value)
        ctx.gateway.ibis.write(table_key, expr)

    # Record in asset catalog if owner_target provided
    if ctx.owner_target is not None:
        ctx.gateway.assets.record_asset(
            AssetRecord(
                asset_key=table_key,
                asset_type="table",
                repo=ctx.snapshot.repo,
                commit=ctx.snapshot.commit,
                owner_target=ctx.owner_target,
                row_count=row_count,
                input_hash=ctx.input_hash,
            )
        )

    return DatasetRef(
        table_key=table_key,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        row_count=row_count,
    )


def materialize_tables(
    ctx: MaterializationContext,
    tables: dict[str, ir.Table],
) -> tuple[DatasetRef, ...]:
    """Materialize multiple Ibis expressions to DuckDB tables.

    Parameters
    ----------
    ctx
        Materialization context with gateway and snapshot info.
    tables
        Mapping of table_key to Ibis expression.

    Returns
    -------
    tuple[DatasetRef, ...]
        Tuple of references to the materialized datasets.

    Examples
    --------
    >>> tables_dict = {
    ...     "analytics.table_a": expr_a,
    ...     "analytics.table_b": expr_b,
    ... }
    >>> ctx = MaterializationContext(gateway=gateway, snapshot=snapshot)
    >>> refs = materialize_tables(ctx, tables_dict)
    >>> len(refs)
    2
    """
    refs: list[DatasetRef] = []
    for table_key, expr in tables.items():
        ref = materialize_table(
            ctx,
            table_key,
            expr,
        )
        refs.append(ref)
    return tuple(refs)


__all__ = [
    "MaterializationContext",
    "materialize_table",
    "materialize_tables",
]
