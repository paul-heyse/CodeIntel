"""Recording gateway wrappers for testing.

This module provides gateway wrappers that record SQL executions for test
assertions. Use these for verifying database interactions in tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway import DuckDBConnection, DuckDBRelation, StorageGateway


class RecordingConnection:
    """Connection wrapper that records executions and delegates to real connection.

    This wrapper intercepts SQL executions and records them while forwarding
    to the underlying connection.
    """

    def __init__(
        self,
        real_con: DuckDBConnection,
        executions: list[tuple[str, list[object]]],
    ) -> None:
        self._real_con = real_con
        self._executions = executions

    def execute(self, sql: str, params: Sequence[object] | None = None) -> RecordingConnection:
        """Record and forward SQL execution.

        Parameters
        ----------
        sql
            SQL query to execute.
        params
            Query parameters.

        Returns
        -------
        RecordingConnection
            Self for chaining.
        """
        self._executions.append((sql, list(params or [])))
        self._real_con.execute(sql, params)
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Fetch all results from the underlying connection.

        Returns
        -------
        list[tuple[Any, ...]]
            All rows from the query.
        """
        return self._real_con.fetchall()

    def __getattr__(self, item: str) -> object:
        """Delegate unknown attributes to the real connection.

        Returns
        -------
        object
            Attribute from the underlying connection.
        """
        return getattr(self._real_con, item)


class ConnectionRecordingGateway:
    """Gateway wrapper that records con.execute() calls.

    This wrapper intercepts calls through the `con` property to record
    SQL executions that go through `gateway.con.execute()`.

    Use this when testing code that accesses `gateway.con.execute()` directly
    rather than using `gateway.execute()`.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        self._gateway = gateway
        self.executions: list[tuple[str, list[object]]] = []
        self.analytics = gateway.analytics
        self.build = gateway.build
        self.config = gateway.config
        self.core = gateway.core
        self.datasets = gateway.datasets
        self.docs = gateway.docs
        self.graph = gateway.graph
        self.ibis = gateway.ibis
        self.runs = gateway.runs
        self._recording_con = RecordingConnection(gateway.con, self.executions)

    @property
    def con(self) -> DuckDBConnection:
        """Return the recording connection wrapper.

        Returns
        -------
        DuckDBConnection
            Recording connection that tracks executions.
        """
        return cast("DuckDBConnection", self._recording_con)

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection:
        """Delegate SQL execution to the underlying gateway.

        Returns
        -------
        DuckDBConnection
            Result of the delegated execution.
        """
        return self._gateway.execute(sql, params)

    def table(self, name: str) -> DuckDBRelation:
        """Return a relation from the underlying gateway.

        Returns
        -------
        DuckDBRelation
            Relation retrieved from the gateway.
        """
        return self._gateway.table(name)

    def close(self) -> None:
        """Close the underlying gateway."""
        self._gateway.close()

    def __getattr__(self, item: str) -> object:
        """Delegate unknown attributes to the real gateway.

        Returns
        -------
        object
            Attribute from the underlying gateway.
        """
        return getattr(self._gateway, item)


class _FailingConnectionProxy:
    """Connection proxy that fails on execute."""

    def __init__(self, error_message: str) -> None:
        self._error_message = error_message

    def execute(self, sql: str, params: object = None) -> _FailingConnectionProxy:
        """Raise RuntimeError to simulate database failure.

        Raises
        ------
        RuntimeError
            Always raises to simulate database failure.
        """
        _ = sql, params
        raise RuntimeError(self._error_message)


class FailingGateway:
    """Gateway that raises on execute for testing error recovery.

    This is a proper test double that implements the gateway interface
    but raises RuntimeError on execute to simulate database failures.
    """

    def __init__(self, gateway: StorageGateway, error_message: str = "db down") -> None:
        self._gateway = gateway
        self._error_message = error_message
        self.records: list[tuple[str, tuple[object, ...]]] = []
        self.analytics = gateway.analytics
        self.build = gateway.build
        self.config = gateway.config
        self.core = gateway.core
        self.datasets = gateway.datasets
        self.docs = gateway.docs
        self.graph = gateway.graph
        self.ibis = gateway.ibis
        self.runs = gateway.runs

    @property
    def con(self) -> DuckDBConnection:
        """Return a failing connection proxy.

        Returns
        -------
        DuckDBConnection
            A proxy that fails on execute.
        """
        return cast("DuckDBConnection", _FailingConnectionProxy(self._error_message))

    def execute(self, sql: str, params: Sequence[object] | None = None) -> DuckDBConnection:
        """Record and raise on SQL execution.

        Raises
        ------
        RuntimeError
            Always raises to simulate database failure.
        """
        self.records.append((sql, tuple(params or ())))
        raise RuntimeError(self._error_message)

    def table(self, name: str) -> DuckDBRelation:
        """Raise when attempting to access a table.

        Raises
        ------
        RuntimeError
            Always raises to simulate database failure.
        """
        _ = name
        raise RuntimeError(self._error_message)

    def close(self) -> None:
        """Close the underlying gateway."""
        self._gateway.close()


__all__ = [
    "ConnectionRecordingGateway",
    "FailingGateway",
    "RecordingConnection",
]
