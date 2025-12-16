"""DuckDB materializer for native Hamilton targets.

This module provides utilities for materializing Ibis expressions to DuckDB
tables with proper snapshot isolation (repo + commit filtering).

The MaterializationContext is compatible with BuildContext from the unified
context hierarchy, enabling seamless integration with the consolidated
build system. For new code, prefer using BuildContext directly.

Example
-------
>>> ctx = BuildContext(
...     gateway=gateway,
...     snapshot=snapshot,
...     paths=paths,
...     validate_schemas=True,
... )
>>> ref = materialize_table(ctx, "analytics.my_table", expr)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

from codeintel.build.hamilton.contracts.enforcement import ContractEnforcer
from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.storage.warehouse import MaterializeOptions, Warehouse

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import ibis.expr.types as ir

    from codeintel.build.context_base import BuildContext
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway.protocol import StorageGateway


@runtime_checkable
class MaterializationContextProtocol(Protocol):
    """Protocol for contexts that support materialization.

    Both BuildContext and MaterializationContext implement this protocol,
    enabling them to be used interchangeably for materialization.
    """

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway."""
        ...

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference."""
        ...


class SchemaValidatorProtocol(Protocol):
    """Protocol for Pandera-like schema validators."""

    def validate(self, frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
        """Validate and return the input frame."""
        ...


@dataclass(frozen=True)
class MaterializationContext:
    """Context for materializing Ibis tables.

    .. deprecated::
        Use :class:`~codeintel.build.context_base.BuildContext` instead.
        This class is retained for backward compatibility.

    This context is compatible with BuildContext from the unified hierarchy.
    Both can be used interchangeably for materialization operations.

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

    @property
    def validate_schemas(self) -> bool:
        """Return whether schema validation is enabled.

        Returns
        -------
        bool
            True when schema validation should be enforced during materialization.
        """
        return self.validate

    @classmethod
    def from_build_context(
        cls,
        ctx: BuildContext,
        *,
        validate: bool | None = None,
        owner_target: str | None = None,
        input_hash: str | None = None,
    ) -> MaterializationContext:
        """Create MaterializationContext from a BuildContext.

        .. deprecated::
            Prefer using BuildContext directly for materialization.

        Parameters
        ----------
        ctx
            BuildContext with gateway and snapshot.
        validate
            Whether to validate against Pandera schemas.
            Defaults to ctx.validate_schemas if not provided.
        owner_target
            Target name that produced these assets.
            Defaults to ctx.owner_target if not provided.
        input_hash
            Input hash for asset catalog.
            Defaults to ctx.input_hash if not provided.

        Returns
        -------
        MaterializationContext
            New materialization context.
        """
        return cls(
            gateway=ctx.gateway,
            snapshot=ctx.snapshot,
            validate=validate if validate is not None else ctx.validate_schemas,
            owner_target=owner_target if owner_target is not None else ctx.owner_target,
            input_hash=input_hash if input_hash is not None else ctx.input_hash,
        )


def materialize_table(
    ctx: BuildContext | MaterializationContext,
    table_key: str,
    expr: ir.Table,
    *,
    schema_resolver: Callable[[str], SchemaValidatorProtocol | None] | None = None,
) -> DatasetRef:
    """Materialize an Ibis expression to a DuckDB table for this snapshot.

    This function writes the expression to DuckDB, replacing any existing
    data for this repo/commit combination to ensure snapshot isolation.

    Accepts either BuildContext or MaterializationContext for backward
    compatibility. For new code, prefer using BuildContext directly.

    Parameters
    ----------
    ctx
        Context with gateway and snapshot info (BuildContext or
        MaterializationContext).
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    expr
        Ibis table expression to materialize.
    schema_resolver
        Optional callable that resolves a schema validator for ``table_key``.
        Defaults to :func:`codeintel.build.hamilton.contracts.pandera_hook.get_pandera_schema`.

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
    >>> from codeintel.build.context_base import BuildContext
    >>> ctx = BuildContext(
    ...     gateway=gateway,
    ...     snapshot=snapshot,
    ...     paths=paths,
    ...     validate_schemas=True,
    ... )
    >>> ref = materialize_table(ctx, "analytics.my_table", expr)
    >>> ref.table_key
    'analytics.my_table'
    """
    # Validate contract if strict mode is enabled
    ContractEnforcer.validate_table_write(table_key)

    # Extract common fields from either context type
    gateway = ctx.gateway
    snapshot = ctx.snapshot
    validate = ctx.validate_schemas
    owner_target = ctx.owner_target
    input_hash = ctx.input_hash

    warehouse = Warehouse(gateway)

    active_schema_resolver = schema_resolver or get_pandera_schema
    schema = active_schema_resolver(table_key) if validate else None

    if schema is not None:
        warehouse.delete_for_snapshot(table_key, snapshot=snapshot)
        df = pd.DataFrame(expr.execute())
        try:
            validated = schema.validate(df, lazy=False)
        except Exception as exc:
            msg = f"Schema validation failed for {table_key}: {exc}"
            raise ValueError(msg) from exc

        result = warehouse.materialize_dataframe(
            table_key,
            validated,
            options=MaterializeOptions(
                snapshot=snapshot,
                mode="append",
                owner_target=owner_target,
                input_hash=input_hash,
            ),
        )
        row_count = result.rows_written or 0
    else:
        result = warehouse.materialize_table(
            table_key,
            expr,
            options=MaterializeOptions(
                snapshot=snapshot,
                mode="replace",
                owner_target=owner_target,
                input_hash=input_hash,
            ),
        )
        row_count = result.rows_written or 0

    metadata: dict[str, object] = {}
    if result.schema_hash is not None:
        metadata["schema_hash"] = result.schema_hash
    if result.schema_version is not None:
        metadata["schema_version"] = result.schema_version
    if result.profiling_artifact is not None:
        metadata["profiling_artifact"] = result.profiling_artifact

    return DatasetRef(
        table_key=table_key,
        repo=snapshot.repo,
        commit=snapshot.commit,
        row_count=row_count,
        metadata=metadata,
    )


def materialize_tables(
    ctx: BuildContext | MaterializationContext,
    tables: dict[str, ir.Table],
) -> tuple[DatasetRef, ...]:
    """Materialize multiple Ibis expressions to DuckDB tables.

    Accepts either BuildContext or MaterializationContext for backward
    compatibility. For new code, prefer using BuildContext directly.

    Parameters
    ----------
    ctx
        Context with gateway and snapshot info (BuildContext or
        MaterializationContext).
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
    >>> ctx = BuildContext(gateway=gateway, snapshot=snapshot, paths=paths)
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


def materialize_rows(
    ctx: BuildContext | MaterializationContext,
    table_key: str,
    rows: Sequence[tuple[object, ...]],
    columns: Sequence[str],
) -> DatasetRef:
    """Materialize row tuples to a DuckDB table for this snapshot.

    This function writes row tuples to DuckDB, replacing any existing
    data for this repo/commit combination to ensure snapshot isolation.
    Use this for analytics computations that produce row data rather than
    Ibis expressions.

    Accepts either BuildContext or MaterializationContext for backward
    compatibility. For new code, prefer using BuildContext directly.

    Parameters
    ----------
    ctx
        Context with gateway and snapshot info (BuildContext or
        MaterializationContext).
    table_key
        Fully-qualified table name (e.g., "analytics.cfg_function_metrics").
    rows
        Sequence of row tuples to write. Each tuple must have values in the
        same order as the columns parameter.
    columns
        Column names matching the row tuple positions.

    Returns
    -------
    DatasetRef
        Reference to the materialized dataset with row count.

    Notes
    -----
    Unlike ``materialize_table``, this function does not support Pandera
    schema validation. The rows are written directly to the database.

    Examples
    --------
    >>> rows = [
    ...     ("goid1", "repo", "commit", 100),
    ...     ("goid2", "repo", "commit", 200),
    ... ]
    >>> columns = ["goid", "repo", "commit", "loc"]
    >>> ref = materialize_rows(ctx, "analytics.my_table", rows, columns)
    >>> ref.row_count
    2
    """
    # Validate contract if strict mode is enabled
    ContractEnforcer.validate_table_write(table_key)

    # Extract common fields from either context type
    gateway = ctx.gateway
    snapshot = ctx.snapshot
    owner_target = ctx.owner_target
    input_hash = ctx.input_hash

    warehouse = Warehouse(gateway)
    result = warehouse.materialize_rows(
        table_key,
        rows,
        columns=columns,
        options=MaterializeOptions(
            snapshot=snapshot,
            mode="replace",
            owner_target=owner_target,
            input_hash=input_hash,
        ),
    )
    row_count = result.rows_written or 0

    metadata: dict[str, object] = {}
    if result.schema_hash is not None:
        metadata["schema_hash"] = result.schema_hash
    if result.schema_version is not None:
        metadata["schema_version"] = result.schema_version
    if result.profiling_artifact is not None:
        metadata["profiling_artifact"] = result.profiling_artifact

    return DatasetRef(
        table_key=table_key,
        repo=snapshot.repo,
        commit=snapshot.commit,
        row_count=row_count,
        metadata=metadata,
    )


__all__ = [
    "MaterializationContext",
    "MaterializationContextProtocol",
    "materialize_rows",
    "materialize_table",
    "materialize_tables",
]
