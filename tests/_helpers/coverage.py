"""Helper utilities for seeding coverage-related test data.

CoveragePack is the canonical way to seed coverage tables. Direct insert
helpers are retained for legacy tests; prefer applying CoveragePack via
`seed_coverage_pack` and loading coverage via `build_fake_coverage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from tests._helpers.context import TestContext
from tests._helpers.fakes.coverage import FakeCoverage, build_fake_coverage_from_gateway
from tests._helpers.seeds.coverage import COVERAGE_PACK, CoveragePack

if TYPE_CHECKING:
    from coverage import Coverage
    from duckdb import DuckDBPyConnection


@dataclass(frozen=True)
class GoidSeedData:
    """Data for inserting a GOID row."""

    urn: str
    rel_path: str
    kind: str
    qualname: str
    goid_h128: int
    start_line: int
    end_line: int | None
    language: str = "python"


@dataclass(frozen=True)
class CoverageLineSeedData:
    """Data for inserting a single coverage line."""

    rel_path: str
    line: int
    is_executable: bool
    is_covered: bool


@dataclass(frozen=True)
class CoverageRangeSeedData:
    """Range specification for seeding many coverage lines."""

    rel_path: str
    start: int
    end: int
    is_executable: bool = True
    is_covered: bool = True


def seed_goid(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: GoidSeedData,
) -> None:
    """Insert a GOID row into core.goids for the given snapshot."""
    con.execute(
        """
        INSERT INTO core.goids (
            urn, repo, commit, rel_path, language, kind, qualname, goid_h128,
            start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
        """,
        [
            data.urn,
            snapshot.repo,
            snapshot.commit,
            data.rel_path,
            data.language,
            data.kind,
            data.qualname,
            data.goid_h128,
            data.start_line,
            data.end_line,
        ],
    )


def seed_coverage_line(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: CoverageLineSeedData,
) -> None:
    """Insert a coverage line into analytics.coverage_lines."""
    hits = 1 if data.is_covered else 0
    context_count = 0
    con.execute(
        """
        INSERT INTO analytics.coverage_lines (
            repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW())
        """,
        [
            snapshot.repo,
            snapshot.commit,
            data.rel_path,
            data.line,
            data.is_executable,
            data.is_covered,
            hits,
            context_count,
        ],
    )


def seed_coverage_lines_range(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: CoverageRangeSeedData,
) -> None:
    """Insert coverage lines for a half-open line range [start, end)."""
    for line in range(data.start, data.end):
        seed_coverage_line(
            con,
            snapshot,
            CoverageLineSeedData(
                rel_path=data.rel_path,
                line=line,
                is_executable=data.is_executable,
                is_covered=data.is_covered,
            ),
        )


def seed_coverage_pack(ctx: TestContext, pack: CoveragePack | None = None) -> None:
    """Apply the canonical coverage seed pack to a TestContext."""
    (pack or COVERAGE_PACK).apply(ctx)


def build_fake_coverage(ctx: TestContext) -> FakeCoverage:
    """Load coverage data from the context gateway into a Coverage-compatible object.

    Returns
    -------
    FakeCoverage
        Coverage-compatible shim backed by seeded tables.
    """
    return build_fake_coverage_from_gateway(ctx.gateway, ctx.snapshot)
