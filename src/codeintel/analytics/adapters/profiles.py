"""Adapters for profile analytics persistence.

This module provides adapters for persisting function, file, and module
profiles to DuckDB.

All profile adapters include schema validation via SchemaValidationMixin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.analytics.adapters.schema_adapter import SchemaValidationMixin
from codeintel.config.datasets import load_columns_by_table, serialize_row
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import datetime

    import pandas as pd

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class FunctionProfileAdapter(BatchAdapter[dict[str, Any]], SchemaValidationMixin):
    """Adapter for analytics.function_profile table.

    Handle persisting aggregated function profile data.

    Includes schema validation via SchemaValidationMixin.
    """

    table_key: ClassVar[str] = "analytics.function_profile"

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        timestamp
            Optional timestamp for created_at field.
        """
        super().__init__(gateway, snapshot)
        self._timestamp = timestamp

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return type(self).table_key

    def load(self) -> Iterator[dict[str, Any]]:
        """Raise NotImplementedError as profiles are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "FunctionProfileAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[dict[str, Any]]) -> int:
        """Persist function profile rows.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        columns = load_columns_by_table().get(self.table_name, [])
        tuple_rows = [serialize_row(row, columns) for row in rows]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(self.table_name, tuple_rows, columns=columns)

        log.info(
            "Persisted %d function profile rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)

    def persist_with_validation(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = False,
    ) -> int:
        """Persist a DataFrame with schema validation.

        Parameters
        ----------
        df
            DataFrame to validate and persist.
        strict
            If True, raise on validation failure. If False, log and proceed.

        Returns
        -------
        int
            Number of rows persisted.
        """
        validated_df = self.validate_dataframe(df) if strict else self.try_validate_dataframe(df)
        rows: list[dict[str, Any]] = validated_df.to_dict(orient="records")
        return self.persist(rows)


class FileProfileAdapter(BatchAdapter[dict[str, Any]], SchemaValidationMixin):
    """Adapter for analytics.file_profile table.

    Handle persisting aggregated file profile data.

    Includes schema validation via SchemaValidationMixin.
    """

    table_key: ClassVar[str] = "analytics.file_profile"

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        timestamp
            Optional timestamp for created_at field.
        """
        super().__init__(gateway, snapshot)
        self._timestamp = timestamp

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return type(self).table_key

    def load(self) -> Iterator[dict[str, Any]]:
        """Raise NotImplementedError as profiles are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "FileProfileAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[dict[str, Any]]) -> int:
        """Persist file profile rows.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        columns = load_columns_by_table().get(self.table_name, [])
        tuple_rows = [serialize_row(row, columns) for row in rows]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(self.table_name, tuple_rows, columns=columns)

        log.info(
            "Persisted %d file profile rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)

    def persist_with_validation(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = False,
    ) -> int:
        """Persist a DataFrame with schema validation.

        Parameters
        ----------
        df
            DataFrame to validate and persist.
        strict
            If True, raise on validation failure. If False, log and proceed.

        Returns
        -------
        int
            Number of rows persisted.
        """
        validated_df = self.validate_dataframe(df) if strict else self.try_validate_dataframe(df)
        rows: list[dict[str, Any]] = validated_df.to_dict(orient="records")
        return self.persist(rows)


class ModuleProfileAdapter(BatchAdapter[dict[str, Any]], SchemaValidationMixin):
    """Adapter for analytics.module_profile table.

    Handle persisting aggregated module profile data.

    Includes schema validation via SchemaValidationMixin.
    """

    table_key: ClassVar[str] = "analytics.module_profile"

    def __init__(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Initialize the adapter.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        snapshot
            Repository snapshot reference.
        timestamp
            Optional timestamp for created_at field.
        """
        super().__init__(gateway, snapshot)
        self._timestamp = timestamp

    @property
    def table_name(self) -> str:
        """Return the target table name."""
        return type(self).table_key

    def load(self) -> Iterator[dict[str, Any]]:
        """Raise NotImplementedError as profiles are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "ModuleProfileAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[dict[str, Any]]) -> int:
        """Persist module profile rows.

        Parameters
        ----------
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        columns = load_columns_by_table().get(self.table_name, [])
        tuple_rows = [serialize_row(row, columns) for row in rows]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(self.table_name, tuple_rows, columns=columns)

        log.info(
            "Persisted %d module profile rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)

    def persist_with_validation(
        self,
        df: pd.DataFrame,
        *,
        strict: bool = False,
    ) -> int:
        """Persist a DataFrame with schema validation.

        Parameters
        ----------
        df
            DataFrame to validate and persist.
        strict
            If True, raise on validation failure. If False, log and proceed.

        Returns
        -------
        int
            Number of rows persisted.
        """
        validated_df = self.validate_dataframe(df) if strict else self.try_validate_dataframe(df)
        rows: list[dict[str, Any]] = validated_df.to_dict(orient="records")
        return self.persist(rows)


__all__ = [
    "FileProfileAdapter",
    "FunctionProfileAdapter",
    "ModuleProfileAdapter",
]
