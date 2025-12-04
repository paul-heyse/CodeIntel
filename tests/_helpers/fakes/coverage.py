"""Fake coverage classes for testing.

This module provides fake coverage data and loader implementations for tests
that need deterministic coverage behavior without running real coverage tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from coverage import Coverage

    from codeintel.config import TestCoverageStepConfig


class FakeCoverageData:
    """Lightweight coverage data implementing measured_files/contexts_by_lineno."""

    def __init__(self, contexts_by_file: dict[str, dict[int, set[str]]]) -> None:
        self._contexts_by_file = contexts_by_file

    def measured_files(self) -> list[str]:
        """
        Return measured file paths.

        Returns
        -------
        list[str]
            File paths observed in coverage data.
        """
        return list(self._contexts_by_file.keys())

    def contexts_by_lineno(self, filename: str) -> dict[int, set[str]]:
        """
        Return contexts keyed by line number for a file.

        Parameters
        ----------
        filename
            File path to resolve contexts for.

        Returns
        -------
        dict[int, set[str]]
            Mapping of line numbers to context identifiers.
        """
        return self._contexts_by_file.get(filename, {})


class FakeCoverage:
    """Coverage shim providing deterministic statements/contexts."""

    def __init__(
        self,
        statements: dict[str, list[int]],
        contexts: dict[str, dict[int, set[str]]],
    ) -> None:
        self._statements = statements
        self._contexts = contexts

    def analysis2(self, filename: str) -> tuple[str, list[int], list[int], list[int], list[int]]:
        """
        Analyze a file and return statement information.

        Parameters
        ----------
        filename
            File path to analyze.

        Returns
        -------
        tuple[str, list[int], list[int], list[int], list[int]]
            Tuple of (filename, statements, excluded, missing, executed).
        """
        stmts = self._statements.get(filename, [])
        return filename, stmts, [], [], stmts

    def get_data(self) -> FakeCoverageData:
        """
        Return deterministic coverage data wrapper.

        Returns
        -------
        FakeCoverageData
            Coverage data exposing measured files and contexts.
        """
        return FakeCoverageData(self._contexts)


class CoverageLoader(Protocol):
    """Protocol for injecting coverage loaders."""

    def __call__(self, cfg: TestCoverageStepConfig | object) -> Coverage:
        """Return a Coverage-compatible object."""
        raise NotImplementedError


__all__ = [
    "CoverageLoader",
    "FakeCoverage",
    "FakeCoverageData",
]
