"""Base classes for plugin test harnesses.

This module provides shared infrastructure for both analytics and ingestion
plugin test harnesses. The base classes extract common patterns to eliminate
code duplication while allowing domain-specific customization.

The design uses protocols and generics to support different result types
while providing consistent assertion APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable
from uuid import uuid4

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


# =============================================================================
# Result Protocol - Common interface for plugin results
# =============================================================================


@runtime_checkable
class ResultLike(Protocol):
    """Protocol for plugin results supporting common assertions.

    Both PluginResult and IngestPluginResult implement this protocol,
    enabling shared assertion logic.
    """

    @property
    def success(self) -> bool:
        """Return whether execution succeeded."""
        ...

    @property
    def error(self) -> str | None:
        """Return error message if any."""
        ...

    @property
    def row_counts(self) -> Mapping[str, int] | None:
        """Return row counts per table."""
        ...


# =============================================================================
# Base Result Assertions
# =============================================================================


@dataclass
class BaseResultAssertions[TResult: ResultLike](ABC):
    """Base class for fluent result assertions.

    Provide common assertion methods that work with any ResultLike object.
    Subclasses can add domain-specific assertions.

    Attributes
    ----------
    _result : TResult
        The result being asserted on.
    _message_prefix : str
        Optional prefix for assertion error messages.
    """

    _result: TResult
    _message_prefix: str = ""

    def with_message(self, prefix: str) -> Self:
        """Set a prefix for assertion messages.

        Parameters
        ----------
        prefix
            Prefix to add to assertion messages.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._message_prefix = prefix
        return self

    def succeeded(self) -> Self:
        """Assert that execution succeeded.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If execution failed.
        """
        if not self._result.success:
            msg = f"{self._message_prefix}Expected success but got failure: {self._result.error}"
            raise AssertionError(msg.strip())
        return self

    def failed(self) -> Self:
        """Assert that execution failed.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If execution succeeded.
        """
        if self._result.success:
            msg = f"{self._message_prefix}Expected failure but got success"
            raise AssertionError(msg.strip())
        return self

    def has_error(self, containing: str | None = None) -> Self:
        """Assert that there is an error message.

        Parameters
        ----------
        containing
            Optional substring the error must contain.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If no error or substring not found.
        """
        if self._result.error is None:
            msg = f"{self._message_prefix}Expected error but got none"
            raise AssertionError(msg.strip())

        if containing is not None and containing not in self._result.error:
            msg = (
                f"{self._message_prefix}Expected error containing '{containing}' "
                f"but got: {self._result.error}"
            )
            raise AssertionError(msg.strip())

        return self

    def has_no_error(self) -> Self:
        """Assert that there is no error message.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If there is an error.
        """
        if self._result.error is not None:
            msg = f"{self._message_prefix}Expected no error but got: {self._result.error}"
            raise AssertionError(msg.strip())
        return self

    def has_row_count(
        self,
        table: str,
        *,
        min_rows: int | None = None,
        max_rows: int | None = None,
        exact: int | None = None,
    ) -> Self:
        """Assert row count for a table.

        Parameters
        ----------
        table
            Table name to check.
        min_rows
            Minimum expected rows.
        max_rows
            Maximum expected rows.
        exact
            Exact expected row count.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        AssertionError
            If row count doesn't match expectations.
        """
        row_counts = self._result.row_counts or {}
        actual = row_counts.get(table, 0)

        if exact is not None:
            if actual != exact:
                msg = f"{self._message_prefix}Expected {table} to have {exact} rows, got {actual}"
                raise AssertionError(msg.strip())
            return self

        if min_rows is not None and actual < min_rows:
            msg = (
                f"{self._message_prefix}Expected {table} to have at least "
                f"{min_rows} rows, got {actual}"
            )
            raise AssertionError(msg.strip())

        if max_rows is not None and actual > max_rows:
            msg = (
                f"{self._message_prefix}Expected {table} to have at most "
                f"{max_rows} rows, got {actual}"
            )
            raise AssertionError(msg.strip())

        return self


# =============================================================================
# Base Test Harness
# =============================================================================


@dataclass
class BaseTestHarness[TPlugin, TContext](ABC):
    """Base class for fluent plugin test harnesses.

    Provide common builder methods for configuring test context.
    Subclasses implement domain-specific context building.

    Attributes
    ----------
    _plugin : TPlugin
        The plugin being tested.
    _gateway : StorageGateway | None
        Storage gateway for database access.
    _repo : str
        Repository identifier.
    _commit : str
        Commit identifier.
    _repo_root : Path | None
        Repository root path.
    _scratch_data : dict[str, object]
        Pre-populated scratch store data.
    _run_id : str
        Unique run identifier.
    """

    _plugin: TPlugin
    _gateway: StorageGateway | None = None
    _repo: str = "test-repo"
    _commit: str = "test-commit"
    _repo_root: Path | None = None
    _scratch_data: dict[str, object] = field(default_factory=dict)
    _run_id: str = field(default_factory=lambda: uuid4().hex)

    def with_gateway(self, gateway: StorageGateway) -> Self:
        """Set the storage gateway.

        Parameters
        ----------
        gateway
            Storage gateway for database access.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._gateway = gateway
        return self

    def with_snapshot(
        self,
        repo: str,
        commit: str,
        repo_root: Path | None = None,
    ) -> Self:
        """Set the repository snapshot.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Optional repository root path.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._repo = repo
        self._commit = commit
        if repo_root is not None:
            self._repo_root = repo_root
        return self

    def with_scratch(self, key: str, value: object) -> Self:
        """Pre-populate scratch store with data.

        Useful for testing plugins that consume data from upstream plugins.

        Parameters
        ----------
        key
            Scratch key.
        value
            Value to store.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._scratch_data[key] = value
        return self

    def with_run_id(self, run_id: str) -> Self:
        """Set the run identifier.

        Parameters
        ----------
        run_id
            Unique run identifier.

        Returns
        -------
        Self
            Self for chaining.
        """
        self._run_id = run_id
        return self

    @abstractmethod
    def build_context(self) -> TContext:
        """Build the execution context for testing.

        Returns
        -------
        TContext
            Configured execution context.

        Raises
        ------
        ValueError
            If required fields are not set.
        """
        ...


__all__ = [
    "BaseResultAssertions",
    "BaseTestHarness",
    "ResultLike",
]
