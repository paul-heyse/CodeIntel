"""Unified coverage fixtures for tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.fixtures.repos import (
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
    MOD_UTIL_PATH,
)
from tests._helpers.fixtures.rows import (
    CoverageFunctionRow,
    CoverageLineRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    insert_rows,
)
from tests._helpers.seeds.core import CORE_PACK

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests._helpers.context import SeedPack, TestContext


class FakeCoverageData:
    """Lightweight coverage data implementing measured_files/contexts_by_lineno."""

    def __init__(self, contexts_by_file: dict[str, dict[int, set[str]]]) -> None:
        self._contexts_by_file = contexts_by_file

    def measured_files(self) -> list[str]:
        """Return measured file paths.

        Returns
        -------
        list[str]
            File paths with coverage data.
        """
        return list(self._contexts_by_file.keys())

    def contexts_by_lineno(self, filename: str) -> dict[int, set[str]]:
        """Return contexts keyed by line number for a file.

        Returns
        -------
        dict[int, set[str]]
            Contexts keyed by line number.
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
        """Analyze a file and return statement information.

        Returns
        -------
        tuple[str, list[int], list[int], list[int], list[int]]
            Tuple containing filename and statement metadata.
        """
        stmts = self._statements.get(filename, [])
        return filename, stmts, [], [], stmts

    def get_data(self) -> FakeCoverageData:
        """Return deterministic coverage data wrapper.

        Returns
        -------
        FakeCoverageData
            Coverage data accessor.
        """
        return FakeCoverageData(self._contexts)


@dataclass(frozen=True)
class CoverageFixtureSpec:
    """Options for seeding coverage fixtures."""

    include_catalog: bool = True
    include_edges: bool = True
    include_functions: bool = True
    include_lines: bool = True
    passing_ratio: float = 0.75
    edge_meta: Mapping[str, object] | None = None
    test_meta: Mapping[str, object] | None = None
    line_spans: Mapping[int, tuple[str, int, int]] | None = None
    line_coverage: Mapping[int, float] | None = None


class CoverageFixtureFactory:
    """Factory for seeding coverage fixtures and fake coverage objects."""

    @staticmethod
    def seed(ctx: TestContext, spec: CoverageFixtureSpec) -> None:
        """Seed coverage-related tables in the provided TestContext."""
        if spec.include_catalog:
            _seed_test_catalog(ctx, spec)
        if spec.include_edges:
            _seed_coverage_edges(ctx, spec)
        if spec.include_functions:
            _seed_coverage_functions(ctx)
        if spec.include_lines:
            _seed_coverage_lines(ctx, spec)

    @staticmethod
    def build_fake_coverage(ctx: TestContext) -> FakeCoverage:
        """Build FakeCoverage backed by analytics coverage tables.

        Returns
        -------
        FakeCoverage
            Coverage shim built from analytics tables.
        """
        statements: dict[str, list[int]] = {}
        contexts: dict[str, dict[int, set[str]]] = {}

        rows = ctx.gateway.con.execute(
            """
            SELECT rel_path, line, is_executable, is_covered
            FROM analytics.coverage_lines
            WHERE repo = ? AND commit = ?
            ORDER BY rel_path, line
            """,
            [ctx.repo, ctx.commit],
        ).fetchall()

        if rows:
            for rel_path, line, _is_exec, is_cov in rows:
                rel_path_str = str(rel_path)
                statements.setdefault(rel_path_str, []).append(int(line))
                if is_cov:
                    contexts.setdefault(rel_path_str, {}).setdefault(int(line), set()).add("test")
        else:
            func_rows = ctx.gateway.con.execute(
                """
                SELECT rel_path, start_line, executable_lines, covered_lines
                FROM analytics.coverage_functions
                WHERE repo = ? AND commit = ?
                ORDER BY rel_path, start_line
                """,
                [ctx.repo, ctx.commit],
            ).fetchall()
            for rel_path, start, executable, covered in func_rows:
                rel_path_str = str(rel_path)
                exec_lines = list(range(int(start), int(start) + int(executable)))
                statements.setdefault(rel_path_str, []).extend(exec_lines)
                covered_lines = exec_lines[: int(covered)]
                for line in covered_lines:
                    contexts.setdefault(rel_path_str, {}).setdefault(line, set()).add("test")

        return FakeCoverage(statements=statements, contexts=contexts)


@dataclass
class CoveragePack:
    """Seed pack for test coverage data."""

    name: str = "coverage"
    include_catalog: bool = True
    include_edges: bool = True
    include_functions: bool = True
    passing_ratio: float = 0.75

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one."""
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply coverage seeds to the test context."""
        spec = CoverageFixtureSpec(
            include_catalog=self.include_catalog,
            include_edges=self.include_edges,
            include_functions=self.include_functions,
            include_lines=False,
            passing_ratio=self.passing_ratio,
        )
        CoverageFixtureFactory.seed(ctx, spec)


COVERAGE_PACK = CoveragePack()

TEST_A = "tests/test_mod_a.py::test_func_a"
TEST_B = "tests/test_mod_b.py::test_func_b"
TEST_C = "tests/test_mod_c.py::test_func_c"
TEST_HELPER = "tests/test_util.py::test_helper"


@dataclass(frozen=True)
class CoverageEdgeMeta:
    """Typed coverage edge metadata with defaults."""

    covered_lines: int = 1
    executable_lines: int = 1
    coverage_ratio: float | None = None
    last_status: str | None = None

    @classmethod
    def from_mapping(cls, meta: Mapping[str, object] | None) -> CoverageEdgeMeta:
        """Build a CoverageEdgeMeta from a mapping.

        Returns
        -------
        CoverageEdgeMeta
            Normalized metadata with defaults applied.
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
        """Return a resolved coverage ratio.

        Returns
        -------
        float
            Coverage ratio computed from metadata.
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
        """Build a TestMeta from a mapping and status override.

        Returns
        -------
        TestMeta
            Normalized metadata with defaults applied.
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


def _coerce_int(value: object, *, default: int) -> int:
    """Coerce a value to int using a default fallback.

    Returns
    -------
    int
        Coerced integer value.
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
    """Coerce a value to float using a default fallback.

    Returns
    -------
    float | None
        Coerced float value.
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


def _seed_test_catalog(ctx: TestContext, spec: CoverageFixtureSpec) -> None:
    """Seed analytics.test_catalog rows for coverage tests."""
    now = datetime.now(tz=UTC)
    test_statuses = ["passed", "passed", "passed", "failed"]
    passing_count = int(len(test_statuses) * spec.passing_ratio)
    for idx in range(passing_count):
        test_statuses[idx] = "passed"

    rows = [
        TestCatalogRow(
            test_id=TEST_A,
            repo=ctx.repo,
            commit=ctx.commit,
            rel_path="tests/test_mod_a.py",
            qualname="test_func_a",
            status=test_statuses[0],
            kind="unit",
            duration_ms=_coerce_int((spec.test_meta or {}).get("duration_ms"), default=150),
            markers="[]",
            parametrized=False,
            flaky=False,
            created_at=now,
        ),
        TestCatalogRow(
            test_id=TEST_B,
            repo=ctx.repo,
            commit=ctx.commit,
            rel_path="tests/test_mod_b.py",
            qualname="test_func_b",
            status=test_statuses[1],
            kind="unit",
            duration_ms=_coerce_int((spec.test_meta or {}).get("duration_ms"), default=200),
            markers='["slow"]',
            parametrized=False,
            flaky=False,
            created_at=now,
        ),
        TestCatalogRow(
            test_id=TEST_C,
            repo=ctx.repo,
            commit=ctx.commit,
            rel_path="tests/test_mod_c.py",
            qualname="test_func_c",
            status=test_statuses[2],
            kind="integration",
            duration_ms=_coerce_int((spec.test_meta or {}).get("duration_ms"), default=500),
            markers="[]",
            parametrized=True,
            flaky=False,
            created_at=now,
        ),
        TestCatalogRow(
            test_id=TEST_HELPER,
            repo=ctx.repo,
            commit=ctx.commit,
            rel_path="tests/test_util.py",
            qualname="test_helper",
            status=test_statuses[3],
            kind="unit",
            duration_ms=_coerce_int((spec.test_meta or {}).get("duration_ms"), default=50),
            markers="[]",
            parametrized=False,
            flaky=True,
            created_at=now,
        ),
    ]
    insert_rows(ctx.gateway, rows)


def _seed_coverage_edges(ctx: TestContext, spec: CoverageFixtureSpec) -> None:
    """Seed analytics.test_coverage_edges rows for coverage tests."""
    now = datetime.now(tz=UTC)
    meta = spec.edge_meta or {}
    covered_lines = _coerce_int(meta.get("covered_lines"), default=1)
    executable_lines = _coerce_int(meta.get("executable_lines"), default=1)
    ratio = _coerce_float(meta.get("coverage_ratio"), default=None)
    last_status = str(meta.get("last_status", "passed"))
    if ratio is None:
        ratio = covered_lines / executable_lines if executable_lines else 0.0

    test_id = TEST_A
    rel_path = "tests/test_mod_a.py"
    qualname = "test_func_a"
    urn = f"urn:{ctx.repo}:{ctx.commit}:{rel_path}#{qualname}"
    rows = [
        TestCoverageEdgeRow(
            test_id=test_id,
            test_goid_h128=100,
            function_goid_h128=GOID_FUNC_A,
            urn=urn,
            repo=ctx.repo,
            commit=ctx.commit,
            rel_path=rel_path,
            qualname=qualname,
            coverage_ratio=ratio,
            covered_lines=covered_lines,
            executable_lines=executable_lines,
            last_status=last_status,
            created_at=now,
        )
    ]
    insert_rows(ctx.gateway, rows)


def _seed_coverage_functions(ctx: TestContext) -> None:
    """Seed analytics.coverage_functions rows for coverage tests."""
    now = datetime.now(tz=UTC)
    default_language = "python"
    default_kind = "function"
    rows = [
        CoverageFunctionRow(
            repo=ctx.repo,
            commit=ctx.commit,
            function_goid_h128=GOID_FUNC_A,
            urn=f"urn:{ctx.repo}:{ctx.commit}:{MOD_A_PATH}#func_a",
            rel_path=MOD_A_PATH,
            language=default_language,
            kind=default_kind,
            qualname="func_a",
            start_line=1,
            end_line=1,
            executable_lines=3,
            covered_lines=3,
            coverage_ratio=1.0,
            tested=True,
            untested_reason=None,
            created_at=now,
        ),
        CoverageFunctionRow(
            repo=ctx.repo,
            commit=ctx.commit,
            function_goid_h128=GOID_FUNC_B,
            urn=f"urn:{ctx.repo}:{ctx.commit}:{MOD_B_PATH}#func_b",
            rel_path=MOD_B_PATH,
            language=default_language,
            kind=default_kind,
            qualname="func_b",
            start_line=1,
            end_line=1,
            executable_lines=4,
            covered_lines=2,
            coverage_ratio=0.5,
            tested=True,
            untested_reason=None,
            created_at=now,
        ),
        CoverageFunctionRow(
            repo=ctx.repo,
            commit=ctx.commit,
            function_goid_h128=GOID_FUNC_C,
            urn=f"urn:{ctx.repo}:{ctx.commit}:{MOD_C_PATH}#func_c",
            rel_path=MOD_C_PATH,
            language=default_language,
            kind=default_kind,
            qualname="func_c",
            start_line=1,
            end_line=1,
            executable_lines=2,
            covered_lines=0,
            coverage_ratio=0.0,
            tested=False,
            untested_reason="missing",
            created_at=now,
        ),
        CoverageFunctionRow(
            repo=ctx.repo,
            commit=ctx.commit,
            function_goid_h128=GOID_HELPER,
            urn=f"urn:{ctx.repo}:{ctx.commit}:{MOD_UTIL_PATH}#helper",
            rel_path=MOD_UTIL_PATH,
            language=default_language,
            kind=default_kind,
            qualname="helper",
            start_line=1,
            end_line=1,
            executable_lines=1,
            covered_lines=1,
            coverage_ratio=1.0,
            tested=True,
            untested_reason=None,
            created_at=now,
        ),
    ]
    insert_rows(ctx.gateway, rows)


def _seed_coverage_lines(ctx: TestContext, spec: CoverageFixtureSpec) -> None:
    """Seed analytics.coverage_lines rows for coverage tests."""
    now = datetime.now(tz=UTC)
    default_spans = {
        GOID_FUNC_A: (MOD_A_PATH, 1, 2),
        GOID_FUNC_B: (MOD_B_PATH, 1, 1),
    }
    default_coverage = {
        GOID_FUNC_A: 1.0,
        GOID_FUNC_B: 0.0,
    }
    line_spans = spec.line_spans or default_spans
    line_coverage = spec.line_coverage or default_coverage

    rows: list[CoverageLineRow] = []
    for goid, (rel_path, start_line, end_line) in line_spans.items():
        ratio = max(0.0, min(1.0, float(line_coverage.get(goid, 0.0))))
        total_lines = max(0, end_line - start_line + 1)
        covered_count = int(total_lines * ratio)
        for idx, line in enumerate(range(start_line, end_line + 1)):
            is_covered = idx < covered_count
            rows.append(
                CoverageLineRow(
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=rel_path,
                    line=line,
                    is_executable=True,
                    is_covered=is_covered,
                    hits=1 if is_covered else 0,
                    context_count=0,
                    created_at=now,
                )
            )
    insert_rows(ctx.gateway, rows)


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
    RuntimeError
        If the database fails to return a timestamp.
    """
    row = ctx.gateway.con.execute("SELECT NOW()").fetchone()
    if row is None:
        msg = "SELECT NOW() returned no rows"
        raise RuntimeError(msg)
    now = row[0]
    tests: dict[str, TestCatalogRow] = {}
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


__all__ = [
    "COVERAGE_PACK",
    "CoverageEdgeMeta",
    "CoverageFixtureFactory",
    "CoverageFixtureSpec",
    "CoveragePack",
    "TestMeta",
    "synthesize_coverage_edges",
]
