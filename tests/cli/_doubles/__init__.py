"""Test doubles for CLI testing.

These are protocol-compliant implementations that can be injected
during tests without monkeypatching. Following the testing charter,
these doubles implement the same interfaces as production components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FakeStorageGateway:
    """Test double for StorageGateway.

    Provides in-memory storage that can be pre-populated
    with test data and inspected after test execution.

    Parameters
    ----------
    tables
        Pre-populated table data.
    queries_executed
        Record of queries executed (for assertions).
    """

    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    queries_executed: list[str] = field(default_factory=list)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a query against fake storage.

        Parameters
        ----------
        sql
            SQL query string.

        Returns
        -------
        list[dict[str, Any]]
            Query results.
        """
        self.queries_executed.append(sql)
        # Simple table name extraction for basic queries
        for table_name, data in self.tables.items():
            if table_name in sql:
                return data
        return []

    def insert(self, table: str, rows: list[dict[str, Any]]) -> int:
        """Insert rows into fake storage.

        Parameters
        ----------
        table
            Table name.
        rows
            Rows to insert.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if table not in self.tables:
            self.tables[table] = []
        self.tables[table].extend(rows)
        return len(rows)


@dataclass
class FakeOperationCatalog:
    """Test double for operation catalog.

    Parameters
    ----------
    operations
        Pre-registered operations.
    """

    operations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_operation(self, op_id: str) -> dict[str, Any] | None:
        """Get operation by ID.

        Parameters
        ----------
        op_id
            Operation identifier.

        Returns
        -------
        dict[str, Any] | None
            Operation metadata or None.
        """
        return self.operations.get(op_id)

    def list_operations(self) -> list[dict[str, Any]]:
        """List all operations.

        Returns
        -------
        list[dict[str, Any]]
            All registered operations.
        """
        return list(self.operations.values())

    def invoke(self, op_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Invoke an operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Operation result.

        Raises
        ------
        KeyError
            If operation not found.
        """
        if op_id not in self.operations:
            msg = f"Operation not found: {op_id}"
            raise KeyError(msg)
        # Return a mock result
        return {"op_id": op_id, "params": params, "status": "success"}


@dataclass
class FakeConsole:
    """Test double for rich Console.

    Captures all output for assertion.

    Parameters
    ----------
    output
        Captured output lines.
    """

    output: list[str] = field(default_factory=list)

    def print(self, *args: object, **kwargs: object) -> None:
        """Capture print output.

        Parameters
        ----------
        *args
            Print arguments.
        **kwargs
            Print keyword arguments (unused).
        """
        _ = kwargs  # Unused
        self.output.append(" ".join(str(arg) for arg in args))

    def clear(self) -> None:
        """Clear captured output."""
        self.output.clear()


@dataclass
class FakeFileSystem:
    """Test double for file system operations.

    Parameters
    ----------
    files
        Pre-populated file contents.
    directories
        Pre-existing directories.
    """

    files: dict[str, str] = field(default_factory=dict)
    directories: set[str] = field(default_factory=set)

    def read_text(self, path: Path) -> str:
        """Read file content.

        Parameters
        ----------
        path
            File path.

        Returns
        -------
        str
            File content.

        Raises
        ------
        FileNotFoundError
            If file doesn't exist.
        """
        key = str(path)
        if key not in self.files:
            msg = f"No such file: {path}"
            raise FileNotFoundError(msg)
        return self.files[key]

    def write_text(self, path: Path, content: str) -> None:
        """Write file content.

        Parameters
        ----------
        path
            File path.
        content
            Content to write.
        """
        self.files[str(path)] = content

    def exists(self, path: Path) -> bool:
        """Check if path exists.

        Parameters
        ----------
        path
            Path to check.

        Returns
        -------
        bool
            True if exists.
        """
        key = str(path)
        return key in self.files or key in self.directories

    def is_dir(self, path: Path) -> bool:
        """Check if path is a directory.

        Parameters
        ----------
        path
            Path to check.

        Returns
        -------
        bool
            True if directory.
        """
        return str(path) in self.directories


__all__ = [
    "FakeConsole",
    "FakeFileSystem",
    "FakeOperationCatalog",
    "FakeStorageGateway",
]
