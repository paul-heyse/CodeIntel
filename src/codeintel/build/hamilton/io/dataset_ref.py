"""Type-safe dataset references for Hamilton DAG.

DatasetRef provides a lightweight reference to a dataset table that can
flow through the Hamilton DAG. The actual data is not materialized until
explicitly requested via dataset-backed loaders.

Design Principles
-----------------
1. DatasetRef is a NamedTuple for immutability.
2. References carry metadata but not data - data flows through dataset loaders.
3. Helper functions bridge target execution results to DatasetRef instances.
4. repo/commit fields enable snapshot identity for lineage tracking.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.config.primitives import SnapshotRef


_EMPTY_METADATA: Mapping[str, object] = MappingProxyType({})


class DatasetRef(NamedTuple):
    """Reference to a dataset in the build DAG.

    This is a lightweight handle that identifies a table without loading data.
    Used to establish lineage relationships in the Hamilton DAG.

    Attributes
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    repo
        Repository slug for snapshot identity (e.g., "org/repo").
    commit
        Commit SHA for snapshot identity.
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
    ...     repo="org/repo",
    ...     commit="abc123",
    ...     source_target="function_metrics",
    ...     row_count=1500,
    ... )
    >>> ref.schema_name
    'analytics'
    >>> ref.table_name
    'function_metrics'
    """

    table_key: str
    repo: str = ""
    commit: str = ""
    schema_version: str | None = None
    row_count: int | None = None
    source_target: str | None = None
    metadata: Mapping[str, object] = _EMPTY_METADATA

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
        if "." not in self.table_key:
            return "main"
        schema, _ = split_table_key(self.table_key)
        return schema

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
        if "." not in self.table_key:
            return self.table_key
        _, table = split_table_key(self.table_key)
        return table

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
        >>> ref = DatasetRef(table_key="test.table", repo="org/repo", commit="abc")
        >>> updated = ref.with_row_count(100)
        >>> updated.row_count
        100
        """
        return DatasetRef(
            table_key=self.table_key,
            repo=self.repo,
            commit=self.commit,
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
        >>> ref = DatasetRef(table_key="test.table", repo="org/repo", commit="abc")
        >>> updated = ref.with_metadata("computed_at", "2024-01-01")
        >>> updated.metadata["computed_at"]
        '2024-01-01'
        """
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return DatasetRef(
            table_key=self.table_key,
            repo=self.repo,
            commit=self.commit,
            schema_version=self.schema_version,
            row_count=self.row_count,
            source_target=self.source_target,
            metadata=new_metadata,
        )


def refs_from_target_result(
    target_name: str,
    table_keys: tuple[str, ...],
    row_counts: dict[str, int] | None = None,
    *,
    snapshot: SnapshotRef | None = None,
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
    snapshot
        Optional snapshot reference for repo/commit identity.

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
    repo = snapshot.repo if snapshot else ""
    commit = snapshot.commit if snapshot else ""
    return {
        key: DatasetRef(
            table_key=key,
            repo=repo,
            commit=commit,
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
