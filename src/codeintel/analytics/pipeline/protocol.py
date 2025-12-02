"""Protocol definitions for the dataset pipeline system.

This module defines the core protocols and data structures for
defining and computing typed datasets.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.analytics.pipeline.contracts import DatasetContract
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

@dataclass(frozen=True)
class TableSchema:
    """Schema definition for a database table.

    Attributes
    ----------
    name
        Fully qualified table name (e.g., "analytics.function_metrics").
    columns
        Tuple of column definitions (name, type, nullable).
    primary_key
        Columns forming the primary key.
    indexes
        Additional index definitions.
    """

    name: str
    columns: tuple[tuple[str, str, bool], ...] = ()
    primary_key: tuple[str, ...] = ()
    indexes: tuple[tuple[str, ...], ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return just the column names.

        Returns
        -------
        tuple[str, ...]
            Column names in order.
        """
        return tuple(c[0] for c in self.columns)


@dataclass(frozen=True)
class DatasetSpec[RowT]:
    """Specification for a typed dataset.

    A dataset spec defines the metadata, schema, dependencies, and
    contracts for a dataset that can be computed by the pipeline.

    Type Parameters
    ---------------
    RowT
        The row type this dataset produces.

    Attributes
    ----------
    name
        Unique identifier for the dataset.
    description
        Human-readable description.
    row_type
        Python type for dataset rows.
    schema
        Database table schema.
    inputs
        Datasets this computation depends on.
    outputs
        Tables this computation writes to.
    contract
        Validation contract for output data.
    version
        Semantic version of the dataset spec.
    tags
        Classification tags.
    """

    name: str
    description: str = ""
    row_type: type[RowT] | None = None
    schema: TableSchema | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    contract: DatasetContract | None = None
    version: str = "1.0.0"
    tags: tuple[str, ...] = ()

    @property
    def primary_output(self) -> str:
        """Return the primary output table name.

        Returns
        -------
        str
            First output table name, or the dataset name if no outputs.
        """
        return self.outputs[0] if self.outputs else self.name


@dataclass
class PipelineContext:
    """Context for dataset computation.

    Provides access to storage, configuration, and other resources
    needed during dataset computation.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    run_id
        Unique identifier for this pipeline run.
    timestamp
        Timestamp when the run started.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def repo(self) -> str:
        """Return the repository identifier."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return the commit identifier."""
        return self.snapshot.commit


@runtime_checkable
class DatasetComputation[RowT_co](Protocol):
    """Protocol for dataset computation implementations.

    A dataset computation defines how to produce rows for a dataset
    given its input dependencies.

    Type Parameters
    ---------------
    RowT_co
        The row type this computation produces (covariant).
    """

    @property
    def spec(self) -> DatasetSpec[RowT_co]:
        """Return the dataset specification.

        Returns
        -------
        DatasetSpec[RowT_co]
            Specification for this dataset.
        """
        ...

    def compute(
        self,
        ctx: PipelineContext,
        inputs: dict[str, Any],
    ) -> Iterator[RowT_co]:
        """Compute dataset rows.

        Parameters
        ----------
        ctx
            Pipeline execution context.
        inputs
            Mapping of input dataset names to their loaded data.

        Yields
        ------
        RowT_co
            Computed rows for this dataset.
        """
        ...


@dataclass(frozen=True)
class DatasetResult[RowT]:
    """Result of computing a dataset.

    Attributes
    ----------
    spec
        The dataset specification.
    row_count
        Number of rows computed.
    duration_ms
        Computation time in milliseconds.
    success
        Whether computation succeeded.
    error
        Error message if failed.
    """

    spec: DatasetSpec[RowT]
    row_count: int = 0
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


__all__ = [
    "DatasetComputation",
    "DatasetResult",
    "DatasetSpec",
    "PipelineContext",
    "TableSchema",
]
