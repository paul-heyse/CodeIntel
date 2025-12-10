"""Helper utilities for seeding coverage-related test data.

CoveragePack is the canonical way to seed coverage tables. Direct insert
helpers are retained for legacy tests; prefer applying CoveragePack via
`seed_coverage_pack` and loading coverage via `build_fake_coverage`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from tests._helpers.builders import (
    CoverageFunctionRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    insert_rows,
)
from tests._helpers.context import TestContext
from tests._helpers.fakes.coverage import FakeCoverage, build_fake_coverage_from_gateway
from tests._helpers.seeds.coverage import COVERAGE_PACK, CoveragePack

if TYPE_CHECKING:
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


def synthesize_coverage_edges(
    ctx: TestContext,
    edges: list[tuple[str, int]],
    *,
    status: str = "passed",
    test_meta: Mapping[str, Mapping[str, object]] | None = None,
    edge_meta: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Create minimal test_catalog + coverage_edges + coverage_functions rows.

    Raises
    ------
    ValueError
        If the TestContext is missing a gateway.
    RuntimeError
        If the database fails to return a current timestamp.
    """
    gateway = ctx.gateway
    if gateway is None:
        msg = "TestContext.gateway must be populated to synthesize coverage edges"
        raise ValueError(msg)
    row = gateway.con.execute("SELECT NOW()").fetchone()
    if row is None:
        msg = "SELECT NOW() returned no rows"
        raise RuntimeError(msg)
    now = row[0]
    tests = {}
    edge_rows: list[TestCoverageEdgeRow] = []
    coverage_rows: list[CoverageFunctionRow] = []
    coverage_counts: dict[int, int] = {}
    first_span: dict[int, tuple[str, str]] = {}
    for test_id, goid in edges:
        rel_path = test_id.split("::")[0]
        qualname = test_id.split("::")[-1]
        urn = f"urn:{ctx.repo}:{ctx.commit}:{rel_path}#{qualname}"
        meta = (edge_meta or {}).get(test_id, {})
        coverage_counts[goid] = coverage_counts.get(goid, 0) + 1
        first_span.setdefault(goid, (rel_path, qualname))
        covered_lines = int(meta.get("covered_lines", 1))
        executable_lines = int(meta.get("executable_lines", 1))
        ratio = float(meta.get("coverage_ratio", covered_lines / executable_lines))
        last_status = str(meta.get("last_status", status))
        edge_rows.append(
            TestCoverageEdgeRow(
                test_id=test_id,
                test_goid_h128=None,
                function_goid_h128=goid,
                urn=urn,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=rel_path,
                qualname=qualname,
                covered_lines=covered_lines,
                executable_lines=executable_lines,
                coverage_ratio=ratio,
                last_status=last_status,
                created_at=now,
            )
        )
        test_meta_payload = (test_meta or {}).get(test_id, {})
        tests.setdefault(
            test_id,
            TestCatalogRow(
                test_id=test_id,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=rel_path,
                qualname=qualname,
                status=str(test_meta_payload.get("status", status)),
                kind=str(test_meta_payload.get("kind", "unit")),
                duration_ms=int(test_meta_payload.get("duration_ms", 0)),
                markers=test_meta_payload.get("markers", "[]"),
                parametrized=bool(test_meta_payload.get("parametrized", False)),
                flaky=bool(test_meta_payload.get("flaky", False)),
                created_at=now,
            ),
        )

    for goid, count in coverage_counts.items():
        rel_path, qualname = first_span[goid]
        coverage_rows.append(
            CoverageFunctionRow(
                function_goid_h128=goid,
                urn=f"urn:{ctx.repo}:{ctx.commit}:{rel_path}#{qualname}",
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=rel_path,
                language="python",
                kind="function",
                qualname=qualname,
                start_line=1,
                end_line=1,
                executable_lines=count,
                covered_lines=count,
                coverage_ratio=1.0,
                tested=True,
                untested_reason=None,
                created_at=now,
            )
        )

    insert_rows(ctx.gateway, tests.values())
    insert_rows(ctx.gateway, edge_rows)
    insert_rows(ctx.gateway, coverage_rows)
