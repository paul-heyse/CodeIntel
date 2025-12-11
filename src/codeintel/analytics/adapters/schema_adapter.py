"""Schema-aware adapter base classes.

This module provides adapter base classes that integrate with the unified
schema registry for automatic validation before persistence.

Example
-------
>>> class MyMetricsAdapter(SchemaAwareBatchAdapter[MetricsRow]):
...     table_key: ClassVar[str] = "analytics.my_metrics"
...
...     def persist(self, rows: Sequence[MetricsRow]) -> int:
...         df = pd.DataFrame(rows)
...         validated_df = self.validate_dataframe(df)
...         return self._insert_validated(validated_df)
"""

from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

if TYPE_CHECKING:
    from codeintel.config.datasets.schema import DatasetSchema
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

__all__ = [
    "SchemaAwareBatchAdapter",
    "SchemaValidationMixin",
]


class SchemaValidationMixin:
    """Mixin providing schema validation capabilities.

    Add this mixin to any adapter class to gain schema-based validation.
    The class must define a `table_key` class variable specifying which
    dataset schema to use.

    Attributes
    ----------
    table_key
        Class variable defining the fully qualified table name
        (e.g., "analytics.function_metrics").
    """

    table_key: ClassVar[str]

    def get_schema(self) -> DatasetSchema:
        """Return the DatasetSchema for this adapter's table.

        Returns
        -------
        DatasetSchema
            The schema registered for this adapter's table_key.
        """
        return SCHEMA_REGISTRY.require(self.table_key)

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate a DataFrame against this adapter's schema.

        Parameters
        ----------
        df
            DataFrame to validate.

        Returns
        -------
        pd.DataFrame
            Validated (and possibly coerced) DataFrame.
        """
        schema = self.get_schema()
        log.debug("Validating %d rows against schema %s", len(df), self.table_key)
        return schema.validate(df)

    def try_validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to validate a DataFrame, logging warnings on failure.

        This is a lenient validation that logs errors but does not raise.
        Use this for backward compatibility during migration.

        Parameters
        ----------
        df
            DataFrame to validate.

        Returns
        -------
        pd.DataFrame
            Original DataFrame (unchanged, even if validation fails).
        """
        try:
            return self.validate_dataframe(df)
        except (SchemaError, SchemaErrors) as exc:
            log.warning(
                "Schema validation failed for %s: %s (proceeding with unvalidated data)",
                self.table_key,
                exc,
            )
            return df
        except KeyError as exc:
            log.warning(
                "Schema not found for %s: %s (proceeding with unvalidated data)",
                self.table_key,
                exc,
            )
            return df


class SchemaAwareBatchAdapter[RowT](BatchAdapter[RowT], SchemaValidationMixin, ABC):
    """Batch adapter with integrated schema validation.

    This adapter extends BatchAdapter with schema validation capabilities.
    Subclasses can use `validate_dataframe()` to validate data before
    persistence.

    Type Parameters
    ---------------
    RowT
        The row type this adapter works with.

    Example
    -------
    >>> class MetricsAdapter(SchemaAwareBatchAdapter[MetricsRow]):
    ...     table_key: ClassVar[str] = "analytics.function_metrics"
    ...
    ...     @property
    ...     def table_name(self) -> str:
    ...         return self.table_key
    ...
    ...     def persist(self, rows: Sequence[MetricsRow]) -> int:
    ...         df = pd.DataFrame(rows)
    ...         validated_df = self.validate_dataframe(df)
    ...         return self._do_insert(validated_df)
    """

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
    ) -> None:
        """Initialize the schema-aware adapter.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        snapshot
            Repository snapshot reference.
        """
        super().__init__(gateway, snapshot)

    @property
    def table_name(self) -> str:
        """Return the target table name from table_key.

        Returns
        -------
        str
            Fully qualified table name.
        """
        return type(self).table_key

    def persist_validated(
        self,
        rows: Sequence[RowT],
        *,
        delete_before: bool = True,
        strict: bool = True,
    ) -> int:
        """Persist rows with schema validation.

        Parameters
        ----------
        rows
            Rows to persist.
        delete_before
            Whether to delete existing rows before inserting.
        strict
            If True, raise on validation failure. If False, log and proceed.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if delete_before:
            self._delete_existing()

        if not rows:
            return 0

        df = pd.DataFrame(rows)

        if strict:
            self.validate_dataframe(df)
        else:
            self.try_validate_dataframe(df)

        return self.persist(rows)
