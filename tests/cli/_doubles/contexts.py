"""Test context builders for CLI testing.

Provides fluent builders for constructing test scenarios
with the appropriate doubles injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from tests.cli._doubles import (
    FakeConsole,
    FakeFileSystem,
    FakeOperationCatalog,
    FakeStorageGateway,
)


@dataclass
class CliTestContext:
    """Context for CLI handler tests.

    Provides all dependencies needed by handlers,
    allowing tests to inject test doubles.

    Parameters
    ----------
    storage
        Storage gateway (real or fake).
    operations
        Operation catalog (real or fake).
    console
        Console for output (real or fake).
    filesystem
        File system adapter (real or fake).
    """

    storage: FakeStorageGateway = field(default_factory=FakeStorageGateway)
    operations: FakeOperationCatalog = field(default_factory=FakeOperationCatalog)
    console: FakeConsole = field(default_factory=FakeConsole)
    filesystem: FakeFileSystem = field(default_factory=FakeFileSystem)


class CliTestContextBuilder:
    """Fluent builder for CliTestContext.

    Example
    -------
    >>> ctx = (
    ...     CliTestContextBuilder()
    ...     .with_operation("analyze.functions", summary="Analyze functions")
    ...     .with_table("analytics.functions", [{"name": "foo", "loc": 10}])
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        """Initialize builder with empty context."""
        self._storage = FakeStorageGateway()
        self._operations = FakeOperationCatalog()
        self._console = FakeConsole()
        self._filesystem = FakeFileSystem()

    def with_operation(
        self,
        op_id: str,
        *,
        summary: str = "",
        tags: list[str] | None = None,
    ) -> Self:
        """Add an operation to the catalog.

        Parameters
        ----------
        op_id
            Operation identifier.
        summary
            Operation summary.
        tags
            Operation tags.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._operations.operations[op_id] = {
            "id": op_id,
            "summary": summary,
            "tags": tags or [],
        }
        return self

    def with_table(
        self,
        table_key: str,
        rows: list[dict[str, Any]],
    ) -> Self:
        """Add table data to storage.

        Parameters
        ----------
        table_key
            Table key.
        rows
            Table rows.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._storage.tables[table_key] = rows
        return self

    def with_file(self, path: str, content: str) -> Self:
        """Add a file to the filesystem.

        Parameters
        ----------
        path
            File path.
        content
            File content.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._filesystem.files[path] = content
        return self

    def with_directory(self, path: str) -> Self:
        """Add a directory to the filesystem.

        Parameters
        ----------
        path
            Directory path.

        Returns
        -------
        Self
            Builder for chaining.
        """
        self._filesystem.directories.add(path)
        return self

    def build(self) -> CliTestContext:
        """Build the test context.

        Returns
        -------
        CliTestContext
            Configured test context.
        """
        return CliTestContext(
            storage=self._storage,
            operations=self._operations,
            console=self._console,
            filesystem=self._filesystem,
        )


__all__ = [
    "CliTestContext",
    "CliTestContextBuilder",
]
