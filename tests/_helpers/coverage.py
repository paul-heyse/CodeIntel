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


def _coerce_int(value: object, *, default: int) -> int:
    """Safely coerce arbitrary values to int with fallback.

    Returns
    -------
    int
        Coerced integer or provided default.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: object, *, default: float | None) -> float | None:
    """Safely coerce arbitrary values to float with fallback.

    Returns
    -------
    float | None
        Coerced float or provided default.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


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


@dataclass(frozen=True)
class CoverageEdgeMeta:
    """Typed coverage edge metadata with defaults."""

    covered_lines: int = 1
    executable_lines: int = 1
    coverage_ratio: float | None = None
    last_status: str | None = None

    @classmethod
    def from_mapping(cls, meta: Mapping[str, object] | None) -> CoverageEdgeMeta:
        """Coerce loose mapping to typed coverage edge metadata.

        Returns
        -------
        CoverageEdgeMeta
            Parsed metadata with defaults applied.
        """
        if meta is None:
            return cls()
        covered_lines = _coerce_int(meta.get("covered_lines"), default=1)
        executable_lines = _coerce_int(meta.get("executable_lines"), default=1)
        ratio = _coerce_float(meta.get("coverage_ratio"), default=None)
        last_status = meta.get("last_status")
        return cls(
            covered_lines=covered_lines,
            executable_lines=executable_lines,
            coverage_ratio=ratio,
            last_status=str(last_status) if last_status is not None else None,
        )

    def resolved_ratio(self) -> float:
        """Return a safe coverage ratio, computing from lines when absent.

        Returns
        -------
        float
            Coverage ratio computed from available fields.
        """
        if self.coverage_ratio is not None:
            return self.coverage_ratio
        if self.executable_lines <= 0:
            return 0.0
        return self.covered_lines / self.executable_lines


@dataclass(frozen=True)
class TestMeta:
    """Typed test catalog metadata with defaults."""

    status: str = "passed"
    kind: str = "unit"
    duration_ms: int = 0
    markers: str = "[]"
    parametrized: bool = False
    flaky: bool = False

    @classmethod
    def from_mapping(cls, meta: Mapping[str, object] | None, *, status: str) -> TestMeta:
        """Coerce loose mapping to typed test metadata.

        Returns
        -------
        TestMeta
            Parsed test metadata with defaults applied.
        """
        if meta is None:
            return cls(status=status)
        markers_raw = meta.get("markers", "[]")
        return cls(
            status=str(meta.get("status", status)),
            kind=str(meta.get("kind", "unit")),
            duration_ms=_coerce_int(meta.get("duration_ms"), default=0),
            markers=str(markers_raw),
            parametrized=bool(meta.get("parametrized", False)),
            flaky=bool(meta.get("flaky", False)),
        )


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
        meta = CoverageEdgeMeta.from_mapping((edge_meta or {}).get(test_id))
        coverage_counts[goid] = coverage_counts.get(goid, 0) + 1
        first_span.setdefault(goid, (rel_path, qualname))
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
                covered_lines=meta.covered_lines,
                executable_lines=meta.executable_lines,
                coverage_ratio=meta.resolved_ratio(),
                last_status=meta.last_status or status,
                created_at=now,
            )
        )
        test_meta_payload = TestMeta.from_mapping((test_meta or {}).get(test_id), status=status)
        tests.setdefault(
            test_id,
            TestCatalogRow(
                test_id=test_id,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=rel_path,
                qualname=qualname,
                status=test_meta_payload.status,
                kind=test_meta_payload.kind,
                duration_ms=test_meta_payload.duration_ms,
                markers=test_meta_payload.markers,
                parametrized=test_meta_payload.parametrized,
                flaky=test_meta_payload.flaky,
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
