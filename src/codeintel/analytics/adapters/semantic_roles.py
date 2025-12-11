"""Adapters for semantic roles analytics persistence.

This module provides adapters for persisting semantic role classification
results to DuckDB.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.analytics.adapters.base import BatchAdapter
from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

FUNCTION_COLUMNS: tuple[str, ...] = (
    "repo",
    "commit",
    "function_goid_h128",
    "role",
    "framework",
    "role_confidence",
    "role_sources_json",
    "created_at",
)
MODULE_COLUMNS: tuple[str, ...] = (
    "repo",
    "commit",
    "module",
    "role",
    "role_confidence",
    "role_sources_json",
    "created_at",
)
LEGACY_FUNCTION_TUPLE_LEN = 5
LEGACY_MODULE_TUPLE_LEN = 5


@dataclass(frozen=True)
class FunctionSemanticRoleRow:
    """Normalized row for analytics.semantic_roles_functions."""

    repo: str
    commit: str
    function_goid_h128: int
    role: str | None
    framework: str | None
    role_confidence: float | None
    role_sources_json: str
    created_at: str

    def to_tuple(self) -> tuple[object, ...]:
        """Return row values in storage column order.

        Returns
        -------
        tuple[object, ...]
            Row values matching FUNCTION_COLUMNS.
        """
        return (
            self.repo,
            self.commit,
            float(self.function_goid_h128),
            self.role,
            self.framework,
            self.role_confidence,
            self.role_sources_json,
            self.created_at,
        )


@dataclass(frozen=True)
class ModuleSemanticRoleRow:
    """Normalized row for analytics.semantic_roles_modules."""

    repo: str
    commit: str
    module: str
    role: str | None
    role_confidence: float | None
    role_sources_json: str
    created_at: str

    def to_tuple(self) -> tuple[object, ...]:
        """Return row values in storage column order.

        Returns
        -------
        tuple[object, ...]
            Row values matching MODULE_COLUMNS.
        """
        return (
            self.repo,
            self.commit,
            self.module,
            self.role,
            self.role_confidence,
            self.role_sources_json,
            self.created_at,
        )


def _timestamp_str(timestamp: datetime | None) -> str:
    return (timestamp or datetime.now(tz=UTC)).isoformat()


def _coerce_int(value: object, field: str) -> int:
    if not isinstance(value, (int, float, str)):
        message = f"{field} must be int-convertible"
        raise TypeError(message)
    try:
        return int(value)
    except Exception as exc:
        message = f"{field} must be int-convertible"
        raise ValueError(message) from exc


def _coerce_optional_float(value: object | None, field: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        message = f"{field} must be float-convertible"
        raise TypeError(message)
    try:
        return float(value)
    except Exception as exc:
        message = f"{field} must be float-convertible"
        raise ValueError(message) from exc


def _normalize_function_row(
    raw: FunctionSemanticRoleRow | Mapping[str, Any] | Sequence[object],
    repo: str,
    commit: str,
    created_at: str,
) -> FunctionSemanticRoleRow:
    if isinstance(raw, FunctionSemanticRoleRow):
        return FunctionSemanticRoleRow(
            repo=repo,
            commit=commit,
            function_goid_h128=raw.function_goid_h128,
            role=raw.role,
            framework=raw.framework,
            role_confidence=raw.role_confidence,
            role_sources_json=raw.role_sources_json,
            created_at=raw.created_at,
        )

    if isinstance(raw, Mapping):
        function_goid_h128 = raw.get("function_goid_h128")
        if function_goid_h128 is None:
            message = "function_goid_h128 is required for semantic role rows"
            raise ValueError(message)
        return FunctionSemanticRoleRow(
            repo=repo,
            commit=commit,
            function_goid_h128=_coerce_int(function_goid_h128, "function_goid_h128"),
            role=raw.get("role"),
            framework=raw.get("framework"),
            role_confidence=_coerce_optional_float(
                raw.get("role_confidence"),
                "role_confidence",
            ),
            role_sources_json=str(raw.get("role_sources_json", "[]")),
            created_at=str(raw.get("created_at", created_at)),
        )

    values = tuple(raw)
    if len(values) == LEGACY_FUNCTION_TUPLE_LEN:
        _, _, function_goid_h128, role, role_confidence = values
        framework = None
        role_sources_json = "[]"
        created = created_at
    elif len(values) == len(FUNCTION_COLUMNS):
        (
            _,
            _,
            function_goid_h128,
            role,
            framework,
            role_confidence,
            role_sources_json,
            created,
        ) = values
    else:
        message = (
            f"Expected {len(FUNCTION_COLUMNS)} values for function role row "
            f"or legacy {LEGACY_FUNCTION_TUPLE_LEN}-tuple, got {len(values)}"
        )
        raise ValueError(message)

    return FunctionSemanticRoleRow(
        repo=repo,
        commit=commit,
        function_goid_h128=_coerce_int(function_goid_h128, "function_goid_h128"),
        role=str(role) if role is not None else None,
        framework=str(framework) if framework is not None else None,
        role_confidence=_coerce_optional_float(role_confidence, "role_confidence"),
        role_sources_json=str(role_sources_json) if role_sources_json is not None else "[]",
        created_at=str(created) if created is not None else created_at,
    )


def _normalize_module_row(
    raw: ModuleSemanticRoleRow | Mapping[str, Any] | Sequence[object],
    repo: str,
    commit: str,
    created_at: str,
) -> ModuleSemanticRoleRow:
    if isinstance(raw, ModuleSemanticRoleRow):
        return ModuleSemanticRoleRow(
            repo=repo,
            commit=commit,
            module=raw.module,
            role=raw.role,
            role_confidence=raw.role_confidence,
            role_sources_json=raw.role_sources_json,
            created_at=raw.created_at,
        )

    if isinstance(raw, Mapping):
        module = raw.get("module")
        if module is None:
            message = "module is required for semantic role rows"
            raise ValueError(message)
        return ModuleSemanticRoleRow(
            repo=repo,
            commit=commit,
            module=str(module),
            role=raw.get("role"),
            role_confidence=_coerce_optional_float(
                raw.get("role_confidence"),
                "role_confidence",
            ),
            role_sources_json=str(raw.get("role_sources_json", "[]")),
            created_at=str(raw.get("created_at", created_at)),
        )

    values = tuple(raw)
    if len(values) == LEGACY_MODULE_TUPLE_LEN:
        _, _, module, role, role_confidence = values
        role_sources_json = "[]"
        created = created_at
    elif len(values) == len(MODULE_COLUMNS):
        (
            _,
            _,
            module,
            role,
            role_confidence,
            role_sources_json,
            created,
        ) = values
    else:
        message = (
            f"Expected {len(MODULE_COLUMNS)} values for module role row "
            f"or legacy {LEGACY_MODULE_TUPLE_LEN}-tuple, got {len(values)}"
        )
        raise ValueError(message)

    return ModuleSemanticRoleRow(
        repo=repo,
        commit=commit,
        module=str(module),
        role=str(role) if role is not None else None,
        role_confidence=_coerce_optional_float(role_confidence, "role_confidence"),
        role_sources_json=str(role_sources_json) if role_sources_json is not None else "[]",
        created_at=str(created) if created is not None else created_at,
    )


class SemanticRolesFunctionsAdapter(BatchAdapter[tuple[object, ...]]):
    """Adapter for analytics.semantic_roles_functions table.

    Handle persisting function semantic role classifications.
    """

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
        return "analytics.semantic_roles_functions"

    def load(self) -> Iterator[tuple[object, ...]]:
        """Raise NotImplementedError as roles are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "SemanticRolesFunctionsAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[tuple[object, ...]]) -> int:
        """Persist function semantic role rows.

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

        normalized_timestamp = _timestamp_str(self._timestamp)
        normalized_rows = [
            _normalize_function_row(
                row,
                repo=self.repo,
                commit=self.commit,
                created_at=normalized_timestamp,
            ).to_tuple()
            for row in rows
        ]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(
            self.table_name,
            normalized_rows,
            columns=FUNCTION_COLUMNS,
        )

        log.info(
            "Persisted %d function semantic role rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


class SemanticRolesModulesAdapter(BatchAdapter[tuple[object, ...]]):
    """Adapter for analytics.semantic_roles_modules table.

    Handle persisting module semantic role classifications.
    """

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
        return "analytics.semantic_roles_modules"

    def load(self) -> Iterator[tuple[object, ...]]:
        """Raise NotImplementedError as roles are computed not loaded.

        Raises
        ------
        NotImplementedError
            This adapter is write-only.
        """
        message = "SemanticRolesModulesAdapter does not support loading"
        raise NotImplementedError(message)

    def persist(self, rows: Sequence[tuple[object, ...]]) -> int:
        """Persist module semantic role rows.

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

        normalized_timestamp = _timestamp_str(self._timestamp)
        normalized_rows = [
            _normalize_module_row(
                row,
                repo=self.repo,
                commit=self.commit,
                created_at=normalized_timestamp,
            ).to_tuple()
            for row in rows
        ]

        backend = DuckDBPolicyBackend(self._gateway)
        backend.delete_for_snapshot(self.table_name, repo=self.repo, commit=self.commit)
        backend.bulk_insert(
            self.table_name,
            normalized_rows,
            columns=MODULE_COLUMNS,
        )

        log.info(
            "Persisted %d module semantic role rows for %s@%s",
            len(rows),
            self.repo,
            self.commit,
        )
        return len(rows)


__all__ = [
    "SemanticRolesFunctionsAdapter",
    "SemanticRolesModulesAdapter",
]
