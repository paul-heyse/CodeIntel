"""Fake coverage classes for testing.

This module provides fake coverage data and loader implementations for tests
that need deterministic coverage behavior without running real coverage tools.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from coverage import Coverage

    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


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

    def __call__(self, snapshot: SnapshotRef | object) -> Coverage:
        """Return a Coverage-compatible object."""
        raise NotImplementedError


def build_fake_coverage_from_gateway(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> FakeCoverage:
    """
    Build a FakeCoverage object from seeded DuckDB tables.

    Prefers analytics.coverage_lines when present; falls back to
    analytics.coverage_functions for statement ranges.

    Returns
    -------
    FakeCoverage
        Coverage-compatible shim backed by seeded tables.
    """
    statements: dict[str, list[int]] = defaultdict(list)
    contexts: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    rows = gateway.con.execute(
        """
        SELECT rel_path, line, is_executable, is_covered
        FROM analytics.coverage_lines
        WHERE repo = ? AND commit = ?
        ORDER BY rel_path, line
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchall()

    if rows:
        for rel_path, line, is_exec, is_cov in rows:
            if is_exec:
                statements[rel_path].append(int(line))
            if is_cov:
                contexts[rel_path][int(line)].add("test")
    else:
        func_rows = gateway.con.execute(
            """
            SELECT rel_path, start_line, executable_lines, covered_lines
            FROM analytics.coverage_functions
            WHERE repo = ? AND commit = ?
            ORDER BY rel_path, start_line
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchall()
        for rel_path, start, executable, covered in func_rows:
            exec_lines = list(range(int(start), int(start) + int(executable)))
            statements[rel_path].extend(exec_lines)
            covered_lines = exec_lines[: int(covered)]
            for line in covered_lines:
                contexts[rel_path][line].add("test")

    return FakeCoverage(
        statements=dict(statements), contexts={k: dict(v) for k, v in contexts.items()}
    )


__all__ = [
    "CoverageLoader",
    "FakeCoverage",
    "FakeCoverageData",
    "build_fake_coverage_from_gateway",
]
