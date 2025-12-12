"""Type-safe dataset references for Hamilton DAG.

DatasetRef provides a lightweight reference to a DuckDB table that can
flow through the Hamilton DAG. The actual data is not materialized until
explicitly requested via the IO adapters.

Design Principles
-----------------
1. DatasetRef is a frozen dataclass for immutability and hashability.
2. References carry metadata but not data - data flows through IbisGateway.
3. Helper functions bridge target execution results to DatasetRef instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetRef:
    """Reference to a dataset in the build DAG.

    This is a lightweight handle that identifies a table without loading data.
    Used to establish lineage relationships in the Hamilton DAG.

    Attributes
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    schema_version
        Optional schema version for compatibility tracking.
    row_count
        Optional row count if known from prior computation.
    source_target
        Target that produced this dataset (for lineage).
    metadata
        Additional metadata for observability and debugging.

    Examples
    --------
    >>> ref = DatasetRef(
    ...     table_key="analytics.function_metrics",
    ...     source_target="function_metrics",
    ...     row_count=1500,
    ... )
    >>> ref.schema_name
    'analytics'
    >>> ref.table_name
    'function_metrics'
    """

    table_key: str
    schema_version: str | None = None
    row_count: int | None = None
    source_target: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def schema_name(self) -> str:
        """Extract schema name from table key.

        Returns
        -------
        str
            Schema portion of the table key, or "main" if unqualified.

        Examples
        --------
        >>> DatasetRef(table_key="analytics.metrics").schema_name
        'analytics'
        >>> DatasetRef(table_key="simple_table").schema_name
        'main'
        """
        parts = self.table_key.split(".", 1)
        return parts[0] if len(parts) > 1 else "main"

    @property
    def table_name(self) -> str:
        """Extract table name from table key.

        Returns
        -------
        str
            Table name portion of the table key.

        Examples
        --------
        >>> DatasetRef(table_key="analytics.metrics").table_name
        'metrics'
        >>> DatasetRef(table_key="simple_table").table_name
        'simple_table'
        """
        parts = self.table_key.split(".", 1)
        return parts[1] if len(parts) > 1 else parts[0]

    def with_row_count(self, count: int) -> DatasetRef:
        """Return a new ref with updated row count.

        Parameters
        ----------
        count
            New row count value.

        Returns
        -------
        DatasetRef
            New instance with updated row count.

        Examples
        --------
        >>> ref = DatasetRef(table_key="test.table")
        >>> updated = ref.with_row_count(100)
        >>> updated.row_count
        100
        """
        return DatasetRef(
            table_key=self.table_key,
            schema_version=self.schema_version,
            row_count=count,
            source_target=self.source_target,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: object) -> DatasetRef:
        """Return a new ref with additional metadata.

        Parameters
        ----------
        key
            Metadata key.
        value
            Metadata value.

        Returns
        -------
        DatasetRef
            New instance with updated metadata.

        Examples
        --------
        >>> ref = DatasetRef(table_key="test.table")
        >>> updated = ref.with_metadata("computed_at", "2024-01-01")
        >>> updated.metadata["computed_at"]
        '2024-01-01'
        """
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return DatasetRef(
            table_key=self.table_key,
            schema_version=self.schema_version,
            row_count=self.row_count,
            source_target=self.source_target,
            metadata=new_metadata,
        )


def refs_from_target_result(
    target_name: str,
    table_keys: tuple[str, ...],
    row_counts: dict[str, int] | None = None,
) -> dict[str, DatasetRef]:
    """Create DatasetRef instances from a target execution result.

    Parameters
    ----------
    target_name
        Name of the target that produced these datasets.
    table_keys
        Table keys produced by the target.
    row_counts
        Optional mapping of table key to row count.

    Returns
    -------
    dict[str, DatasetRef]
        Mapping of table key to DatasetRef.

    Examples
    --------
    >>> refs = refs_from_target_result(
    ...     target_name="function_metrics",
    ...     table_keys=("analytics.function_metrics",),
    ...     row_counts={"analytics.function_metrics": 1500},
    ... )
    >>> refs["analytics.function_metrics"].row_count
    1500
    """
    counts = row_counts or {}
    return {
        key: DatasetRef(
            table_key=key,
            source_target=target_name,
            row_count=counts.get(key),
        )
        for key in table_keys
    }


def refs_to_tuple(refs: dict[str, DatasetRef]) -> tuple[DatasetRef, ...]:
    """Convert a dict of DatasetRef to an immutable tuple.

    Parameters
    ----------
    refs
        Mapping of table keys to DatasetRef instances.

    Returns
    -------
    tuple[DatasetRef, ...]
        Immutable tuple of DatasetRef instances.

    Examples
    --------
    >>> refs = {"t1": DatasetRef(table_key="t1"), "t2": DatasetRef(table_key="t2")}
    >>> tup = refs_to_tuple(refs)
    >>> len(tup)
    2
    """
    return tuple(refs.values())


__all__ = [
    "DatasetRef",
    "refs_from_target_result",
    "refs_to_tuple",
]
