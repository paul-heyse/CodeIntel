"""Build per-test profiles and behavioral coverage tags."""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.analytics.ast_features.extract import build_import_map, io_flags_from_call
from codeintel.build.analytics.ast_features.patterns import DEFAULT_PATTERNS, AstFeaturePatterns
from codeintel.build.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.build.analytics.testing.coverage.inputs import (
    FunctionCoverageEntry,
    SubsystemCoverageEntry,
    TestGraphMetrics,
)
from codeintel.build.analytics.testing.profiles.rows import (
    TestProfileInputs,
    build_test_profile_context,
    build_test_profile_rows,
)
from codeintel.build.analytics.testing.profiles.types import (
    IoFlags,
    TestAstInfo,
    TestProfileOptions,
    TestRecord,
)
from codeintel.build.analytics.utilities.ast import resolve_call_target
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.paths import path_to_module
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.query_results import (
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_str,
    coerce_str,
    iter_tuples_from_arrow_reader,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsTestProfileRow as ProfileRowModel,
    )
    from codeintel.storage.gateway import DuckDBConnection, StorageGateway

log = logging.getLogger(__name__)

PRIMARY_COVERAGE_THRESHOLD = 0.4


EMPTY_FUNCTION_COVERAGE_ENTRY = FunctionCoverageEntry(functions=[], count=0, primary=[])
EMPTY_SUBSYSTEM_ENTRY = SubsystemCoverageEntry(
    subsystems=[],
    count=0,
    primary_subsystem_id=None,
    max_risk_score=0.0,
)
EMPTY_TEST_METRICS = TestGraphMetrics(
    degree=None,
    weighted_degree=None,
    proj_degree=None,
    proj_weight=None,
    proj_clustering=None,
    proj_betweenness=None,
)


def _normalize_io_entry(value: object) -> dict[str, list[str]]:
    if isinstance(value, Mapping):
        libs = value.get("libs", [])
        funcs = value.get("funcs", [])
    else:
        libs = []
        funcs = []
    return {
        "libs": [str(item) for item in libs] if isinstance(libs, Iterable) else [],
        "funcs": [str(item) for item in funcs] if isinstance(funcs, Iterable) else [],
    }


def _patterns_from_io_spec(
    io_spec: Mapping[str, object] | None,
) -> AstFeaturePatterns:
    base_spec = io_spec or DEFAULT_PATTERNS.io_spec
    normalized_spec = {key: _normalize_io_entry(value) for key, value in base_spec.items()}
    return AstFeaturePatterns(
        io_spec=normalized_spec,
        concurrency_libs=set(DEFAULT_PATTERNS.concurrency_libs),
        http_client_libs=set(DEFAULT_PATTERNS.http_client_libs),
        http_server_libs=set(DEFAULT_PATTERNS.http_server_libs),
        db_libs=set(DEFAULT_PATTERNS.db_libs),
        message_libs=set(DEFAULT_PATTERNS.message_libs),
    )


@dataclass(frozen=True)
class BehavioralProfile:
    """Existing behavioral signals pulled from test_profile."""

    functions_covered: list[dict[str, object]]
    subsystems_covered: list[dict[str, object]]
    assert_count: int
    raise_count: int
    markers: list[str]


@dataclass(frozen=True)
class TestProfileBuildResult:
    """Result from test profile computation.

    Attributes
    ----------
    rows
        Profile row models ready for insertion, or None if no tests found.
    """

    rows: list[ProfileRowModel] | None


def build_test_profile_result(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    options: TestProfileOptions | None = None,
) -> TestProfileBuildResult:
    """Compute test profile rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/DuckDBRelationSaver.

    Parameters
    ----------
    gateway
        Storage gateway bound to the target DuckDB database.
    snapshot
        Snapshot reference with repo, commit, and repo_root.
    options
        Optional test profile configuration options.

    Returns
    -------
    TestProfileBuildResult
        Container with profile row models.
    """
    opts = options or TestProfileOptions()
    con = gateway.con
    tests: list[TestRecord] = _load_test_records(con, snapshot.repo, snapshot.commit)
    if not tests:
        log.info("No tests found for %s@%s; skipping test_profile", snapshot.repo, snapshot.commit)
        return TestProfileBuildResult(rows=None)

    functions_covered = _load_functions_covered(con, snapshot.repo, snapshot.commit)
    subsystems_covered = _load_subsystems_covered(con, snapshot.repo, snapshot.commit)
    tg_metrics = _load_test_graph_metrics(con, snapshot.repo, snapshot.commit)
    io_spec_raw = opts.io_spec if isinstance(opts.io_spec, dict) else None
    patterns = _patterns_from_io_spec(io_spec_raw)
    ast_info = _build_test_ast_index(snapshot.repo_root, tests, patterns)
    inputs = TestProfileInputs(
        functions_covered=functions_covered,
        subsystems_covered=subsystems_covered,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
    ctx = build_test_profile_context(
        snapshot=snapshot,
        inputs=inputs,
        options=opts,
    )
    rows = build_test_profile_rows(tests, ctx)
    return TestProfileBuildResult(rows=rows)


def infer_behavior_tags(
    *,
    name: str,
    markers: Iterable[str],
    io_flags: IoFlags,
    ast_info: TestAstInfo,
) -> list[str]:
    """Infer behavior coverage tags from names, markers, IO flags, and AST hints.

    Parameters
    ----------
    name
        Qualified test name or pytest nodeid.
    markers
        Iterable of marker strings associated with the test.
    io_flags
        IO usage flags inferred from AST analysis.
    ast_info
        AST-derived metrics including pytest.raises and boundary checks.

    Returns
    -------
    list[str]
        Sorted list of behavior tags.
    """
    lower_name = name.lower()
    lower_markers = [marker.lower() for marker in markers]
    tags: set[str] = set()
    tags.update(_tags_from_name(lower_name))
    tags.update(_tags_from_markers(lower_markers))
    tags.update(_tags_from_io_flags(io_flags))
    tags.update(_tags_from_ast_info(ast_info))
    return sorted(tags)


def _tags_from_name(lower_name: str) -> set[str]:
    tags: set[str] = set()
    keyword_map = {
        "happy_path": ("happy", "ok", "success"),
        "error_paths": ("error", "fail", "invalid", "exception"),
        "edge_cases": ("edge", "boundary", "corner"),
        "concurrency": ("concurrent", "parallel", "thread", "async", "race"),
    }
    for tag, keywords in keyword_map.items():
        if any(keyword in lower_name for keyword in keywords):
            tags.add(tag)
    return tags


def _tags_from_markers(lower_markers: Iterable[str]) -> set[str]:
    tags: set[str] = set()
    markers_set = set(lower_markers)
    if "xfail" in markers_set:
        tags.add("known_bug")
    if {"integration", "e2e"} & markers_set:
        tags.add("integration_scenario")
    if "slow" in markers_set:
        tags.add("io_heavy")
    if markers_set.intersection({"network", "api", "http"}):
        tags.add("network_interaction")
    if markers_set.intersection({"db", "database"}):
        tags.add("db_interaction")
    return tags


def _tags_from_io_flags(io_flags: IoFlags) -> set[str]:
    tags: set[str] = set()
    if io_flags.uses_network:
        tags.add("network_interaction")
    if io_flags.uses_db:
        tags.add("db_interaction")
    if io_flags.uses_filesystem:
        tags.add("filesystem_interaction")
    if io_flags.uses_subprocess:
        tags.add("process_interaction")
    if io_flags.io_bound:
        tags.add("io_heavy")
    return tags


def _tags_from_ast_info(ast_info: TestAstInfo) -> set[str]:
    tags: set[str] = set()
    if ast_info.uses_pytest_raises:
        tags.add("error_paths")
    if ast_info.uses_concurrency_lib:
        tags.add("concurrency")
    if ast_info.has_boundary_asserts:
        tags.add("edge_cases")
    return tags


def _normalize_markers(markers: list[str] | None) -> list[str]:
    if markers is None:
        return []
    return [str(marker) for marker in markers]


def _load_test_records(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> list[TestRecord]:
    reader = con.execute(
        """
        SELECT
            t.test_id,
            t.test_goid_h128,
            t.urn,
            t.rel_path,
            m.module,
            COALESCE(t.qualname, g.qualname),
            COALESCE(g.language, 'python'),
            t.kind,
            t.status,
            t.duration_ms,
            t.markers,
            t.flaky,
            g.start_line,
            g.end_line
        FROM analytics.test_catalog t
        LEFT JOIN core.goids g
          ON g.goid_h128 = t.test_goid_h128
         AND g.repo = t.repo
         AND g.commit = t.commit
        LEFT JOIN core.modules m
          ON m.repo = t.repo
         AND m.commit = t.commit
         AND m.path = t.rel_path
        WHERE t.repo = ? AND t.commit = ?
        """,
        [repo, commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    records: list[TestRecord] = []
    for (
        test_id,
        goid,
        urn,
        rel_path,
        module,
        qualname,
        language,
        kind,
        status,
        duration_ms,
        markers,
        flaky,
        start_line,
        end_line,
    ) in iter_tuples_from_arrow_reader(reader):
        rel_path_text = coerce_str(rel_path, ctx="test_catalog.rel_path")
        module_name = coerce_optional_str(module, ctx="test_catalog.module") or path_to_module(
            rel_path_text
        )
        records.append(
            TestRecord(
                test_id=coerce_str(test_id, ctx="test_catalog.test_id"),
                test_goid_h128=normalize_decimal_id(goid),
                urn=coerce_optional_str(urn, ctx="test_catalog.urn"),
                rel_path=rel_path_text,
                module=module_name,
                qualname=coerce_optional_str(qualname, ctx="test_catalog.qualname"),
                language=coerce_optional_str(language, ctx="test_catalog.language"),
                kind=coerce_optional_str(kind, ctx="test_catalog.kind"),
                status=coerce_optional_str(status, ctx="test_catalog.status"),
                duration_ms=coerce_optional_float(duration_ms, ctx="test_catalog.duration_ms"),
                markers=_normalize_markers(markers if isinstance(markers, list) else None),
                flaky=bool(flaky) if flaky is not None else None,
                start_line=coerce_optional_int(start_line, ctx="test_catalog.start_line"),
                end_line=coerce_optional_int(end_line, ctx="test_catalog.end_line"),
            )
        )
    return records


def load_functions_covered(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, FunctionCoverageEntry]:
    """Load per-test function coverage entries.

    Returns
    -------
    dict[str, FunctionCoverageEntry]
        Coverage entries keyed by ``test_id``.
    """
    return _load_functions_covered(con, repo, commit)


def load_subsystems_covered(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, SubsystemCoverageEntry]:
    """Load per-test subsystem coverage entries.

    Returns
    -------
    dict[str, SubsystemCoverageEntry]
        Subsystem coverage entries keyed by ``test_id``.
    """
    return _load_subsystems_covered(con, repo, commit)


def load_test_graph_metrics_public(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, TestGraphMetrics]:
    """Load test graph metrics rows.

    Returns
    -------
    dict[str, TestGraphMetrics]
        Metrics keyed by ``test_id``.
    """
    return _load_test_graph_metrics(con, repo, commit)


def load_test_records_public(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> list[TestRecord]:
    """Load test records from catalog.

    Returns
    -------
    list[TestRecord]
        Test records for the snapshot.
    """
    return _load_test_records(con, repo, commit)


def _load_functions_covered(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, FunctionCoverageEntry]:
    reader = con.execute(
        """
        WITH per_edge AS (
            SELECT
                test_id,
                function_goid_h128,
                SUM(covered_lines) AS covered_lines,
                SUM(executable_lines) AS executable_lines
            FROM analytics.test_coverage_edges
            WHERE repo = ? AND commit = ?
            GROUP BY test_id, function_goid_h128
        ),
        per_test_totals AS (
            SELECT
                test_id,
                SUM(covered_lines) AS total_covered_lines
            FROM per_edge
            GROUP BY test_id
        )
        SELECT
            pe.test_id,
            pe.function_goid_h128,
            pe.covered_lines * 1.0 / NULLIF(pe.executable_lines, 0) AS coverage_ratio,
            pe.covered_lines * 1.0 / NULLIF(pt.total_covered_lines, 0) AS coverage_share,
            g.urn,
            m.module,
            g.qualname,
            g.rel_path
        FROM per_edge pe
        JOIN per_test_totals pt USING (test_id)
        LEFT JOIN core.goids g
          ON g.goid_h128 = pe.function_goid_h128
         AND g.repo = ?
         AND g.commit = ?
        LEFT JOIN core.modules m
          ON m.repo = g.repo
         AND m.commit = g.commit
         AND m.path = g.rel_path
        """,
        [repo, commit, repo, commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)

    result: dict[str, FunctionCoverageEntry] = {}
    for (
        test_id,
        function_goid_h128,
        coverage_ratio,
        coverage_share,
        urn,
        module,
        qualname,
        rel_path,
    ) in iter_tuples_from_arrow_reader(reader):
        rel_path_text = coerce_str(rel_path, ctx="test_coverage_edges.rel_path")
        module_name = coerce_optional_str(
            module, ctx="test_coverage_edges.module"
        ) or path_to_module(rel_path_text)
        test_key = coerce_str(test_id, ctx="test_coverage_edges.test_id")
        entry = result.get(test_key)
        if entry is None:
            entry = FunctionCoverageEntry(functions=[], count=0, primary=[])
            result[test_key] = entry
        functions = list(entry.functions)
        primary = list(entry.primary)
        goid_value = normalize_decimal_id(function_goid_h128)
        ratio_value = coerce_optional_float(coverage_ratio, ctx="coverage_ratio")
        share_value = coerce_optional_float(coverage_share, ctx="coverage_share")
        functions.append(
            {
                "function_goid_h128": goid_value,
                "urn": coerce_optional_str(urn, ctx="test_coverage_edges.urn"),
                "module": module_name,
                "qualname": coerce_optional_str(qualname, ctx="test_coverage_edges.qualname"),
                "rel_path": rel_path_text,
                "coverage_ratio": ratio_value,
                "coverage_share": share_value,
            }
        )
        if (
            goid_value is not None
            and share_value is not None
            and share_value >= PRIMARY_COVERAGE_THRESHOLD
        ):
            primary.append(goid_value)
        result[test_key] = FunctionCoverageEntry(
            functions=functions,
            count=len(functions),
            primary=primary,
        )
    return result


def _load_subsystems_covered(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, SubsystemCoverageEntry]:
    reader = con.execute(
        """
        WITH per_edge AS (
            SELECT
                e.test_id,
                sm.subsystem_id,
                SUM(e.covered_lines) AS covered_lines,
                SUM(e.executable_lines) AS executable_lines
            FROM analytics.test_coverage_edges e
            JOIN core.goids g
              ON g.goid_h128 = e.function_goid_h128
             AND g.repo = e.repo
             AND g.commit = e.commit
            JOIN core.modules m
              ON m.repo = g.repo
             AND m.commit = g.commit
             AND m.path = g.rel_path
            JOIN analytics.subsystem_modules sm
              ON sm.module = m.module
             AND sm.repo = e.repo
             AND sm.commit = e.commit
            WHERE e.repo = ? AND e.commit = ?
            GROUP BY e.test_id, sm.subsystem_id
        ),
        per_test_totals AS (
            SELECT
                test_id,
                SUM(covered_lines) AS total_covered_lines
            FROM per_edge
            GROUP BY test_id
        )
        SELECT
            pe.test_id,
            pe.subsystem_id,
            pe.covered_lines * 1.0 / NULLIF(pt.total_covered_lines, 0) AS coverage_share,
            s.name,
            s.max_risk_score
        FROM per_edge pe
        JOIN per_test_totals pt USING (test_id)
        LEFT JOIN analytics.subsystems s
          ON s.subsystem_id = pe.subsystem_id
         AND s.repo = ?
         AND s.commit = ?
        """,
        [repo, commit, repo, commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)

    result: dict[str, SubsystemCoverageEntry] = {}
    for (
        test_id,
        subsystem_id,
        coverage_share,
        name,
        max_risk_score,
    ) in iter_tuples_from_arrow_reader(reader):
        test_key = coerce_str(test_id, ctx="subsystem_coverage.test_id")
        entry = result.get(test_key) or SubsystemCoverageEntry(
            subsystems=[],
            count=0,
            primary_subsystem_id=None,
            max_risk_score=0.0,
        )
        share = coerce_optional_float(coverage_share, ctx="coverage_share") or 0.0
        subsystem_id_text = coerce_str(subsystem_id, ctx="subsystem_coverage.subsystem_id")
        subsystem_name = coerce_optional_str(name, ctx="subsystems.name") or ""
        subsystems = list(entry.subsystems)
        subsystems.append(
            {
                "subsystem_id": subsystem_id_text,
                "name": subsystem_name,
                "coverage_share": share,
            }
        )
        primary_subsystem_id = entry.primary_subsystem_id
        primary_share = share if primary_subsystem_id == subsystem_id_text else -1.0
        if primary_subsystem_id is None or share > primary_share:
            primary_subsystem_id = subsystem_id_text
        result[test_key] = SubsystemCoverageEntry(
            subsystems=subsystems,
            count=len(subsystems),
            primary_subsystem_id=primary_subsystem_id,
            max_risk_score=max(
                entry.max_risk_score or 0.0,
                coerce_optional_float(max_risk_score, ctx="max_risk_score") or 0.0,
            ),
        )
    return result


def _load_test_graph_metrics(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, TestGraphMetrics]:
    reader = con.execute(
        """
        SELECT
            test_id,
            degree,
            weighted_degree,
            proj_degree,
            proj_weight,
            proj_clustering,
            proj_betweenness
        FROM analytics.test_graph_metrics_tests
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    metrics: dict[str, TestGraphMetrics] = {}
    for (
        test_id,
        degree,
        weighted_degree,
        proj_degree,
        proj_weight,
        proj_clustering,
        proj_betweenness,
    ) in iter_tuples_from_arrow_reader(reader):
        test_id_text = coerce_str(test_id, ctx="test_graph_metrics.test_id")
        metrics[test_id_text] = TestGraphMetrics(
            degree=coerce_optional_int(degree, ctx="test_graph_metrics.degree"),
            weighted_degree=coerce_optional_float(
                weighted_degree,
                ctx="test_graph_metrics.weighted_degree",
            ),
            proj_degree=coerce_optional_int(proj_degree, ctx="test_graph_metrics.proj_degree"),
            proj_weight=coerce_optional_float(proj_weight, ctx="test_graph_metrics.proj_weight"),
            proj_clustering=coerce_optional_float(
                proj_clustering,
                ctx="test_graph_metrics.proj_clustering",
            ),
            proj_betweenness=coerce_optional_float(
                proj_betweenness,
                ctx="test_graph_metrics.proj_betweenness",
            ),
        )
    return metrics


def load_test_profile_context(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, dict[str, object]]:
    """Load test profile context.

    Returns
    -------
    dict[str, dict[str, object]]
        Profile context keyed by ``test_id``.
    """
    reader = con.execute(
        """
        SELECT
            test_id,
            markers,
            functions_covered,
            subsystems_covered,
            assert_count,
            raise_count,
            status
        FROM analytics.test_profile
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    ctx: dict[str, dict[str, object]] = {}
    for (
        test_id,
        markers,
        functions_covered,
        subsystems_covered,
        assert_count,
        raise_count,
        status,
    ) in iter_tuples_from_arrow_reader(reader):
        test_id_text = coerce_str(test_id, ctx="test_profile.test_id")
        ctx[test_id_text] = {
            "markers": markers,
            "functions_covered": functions_covered or [],
            "subsystems_covered": subsystems_covered or [],
            "assert_count": coerce_optional_int(assert_count, ctx="test_profile.assert_count") or 0,
            "raise_count": coerce_optional_int(raise_count, ctx="test_profile.raise_count") or 0,
            "status": coerce_optional_str(status, ctx="test_profile.status"),
        }
    return ctx


def build_test_ast_index_for_tests(
    repo_root: Path,
    tests: Iterable[TestRecord],
) -> dict[str, TestAstInfo]:
    """Build AST span index for tests using default IO heuristics.

    This helper is intended for unit tests that need the same parsing behavior
    as production.

    Returns
    -------
    dict[str, TestAstInfo]
        Mapping from test IDs to AST-derived metrics.
    """
    return _build_test_ast_index(repo_root, tests, DEFAULT_PATTERNS)


def _build_test_ast_index(
    repo_root: Path,
    tests: Iterable[TestRecord],
    patterns: AstFeaturePatterns,
) -> dict[str, TestAstInfo]:
    tests_by_path: dict[str, list[TestRecord]] = {}
    for test in tests:
        tests_by_path.setdefault(test.rel_path, []).append(test)

    info_by_id: dict[str, TestAstInfo] = {}
    for rel_path, path_tests in tests_by_path.items():
        ast_results = _analyze_file(
            repo_root / rel_path,
            path_tests,
            patterns,
        )
        info_by_id.update(ast_results)
    return info_by_id


def build_test_ast_index(
    repo_root: Path,
    tests: Iterable[TestRecord],
    patterns: AstFeaturePatterns,
) -> dict[str, TestAstInfo]:
    """Build the AST index for test spans.

    Returns
    -------
    dict[str, TestAstInfo]
        AST-derived info keyed by ``test_id``.
    """
    return _build_test_ast_index(repo_root, tests, patterns)


@dataclass
class SpanState:
    """Mutable AST-derived flags for a test span."""

    assert_count: int = 0
    raise_count: int = 0
    uses_pytest_raises: bool = False
    uses_concurrency: bool = False
    has_boundary_asserts: bool = False
    uses_fixtures: bool = False
    io_flags: IoFlags = field(default_factory=IoFlags)


@dataclass(frozen=True)
class SpanConfig:
    """Configuration describing the span and import resolution for a test."""

    import_map: dict[str, str]
    start_line: int
    end_line: int
    patterns: AstFeaturePatterns


def _analyze_file(
    path: Path,
    tests: Iterable[TestRecord],
    patterns: AstFeaturePatterns,
) -> dict[str, TestAstInfo]:
    parsed = parse_python_module(path)
    if parsed is None:
        return {test.test_id: TestAstInfo() for test in tests}
    _, tree = parsed
    import_map = build_import_map(tree)
    info: dict[str, TestAstInfo] = {}
    for test in tests:
        if test.start_line is None:
            info[test.test_id] = TestAstInfo()
            continue
        config = SpanConfig(
            import_map=import_map,
            start_line=test.start_line,
            end_line=test.end_line or test.start_line,
            patterns=patterns,
        )
        info[test.test_id] = _analyze_span(tree, config)
    return info


def _analyze_span(
    tree: ast.AST,
    config: SpanConfig,
) -> TestAstInfo:
    state = SpanState()
    for node in ast.walk(tree):
        if not _node_in_span(node, config):
            continue
        _update_span_state(node, config, state)

    return TestAstInfo(
        assert_count=state.assert_count,
        raise_count=state.raise_count,
        uses_pytest_raises=state.uses_pytest_raises,
        uses_concurrency_lib=state.uses_concurrency,
        has_boundary_asserts=state.has_boundary_asserts,
        uses_fixtures=state.uses_fixtures,
        io_flags=state.io_flags,
    )


def _node_in_span(node: ast.AST, config: SpanConfig) -> bool:
    lineno = getattr(node, "lineno", None)
    return lineno is not None and config.start_line <= lineno <= config.end_line


def _update_span_state(
    node: ast.AST,
    config: SpanConfig,
    state: SpanState,
) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        state.uses_fixtures = state.uses_fixtures or _uses_fixtures(node)
    if isinstance(node, ast.Assert):
        state.assert_count += 1
        state.has_boundary_asserts = state.has_boundary_asserts or _is_boundary_assert(node)
    if isinstance(node, ast.Raise):
        state.raise_count += 1
    if isinstance(node, (ast.With, ast.AsyncWith)) and _with_uses_pytest_raises(node):
        state.uses_pytest_raises = True
    if isinstance(node, ast.Call):
        if _is_pytest_raises(node.func):
            state.uses_pytest_raises = True
        state.io_flags = io_flags_from_call(
            node,
            config.import_map,
            state.io_flags,
            patterns=config.patterns,
        )
        state.uses_concurrency = state.uses_concurrency or _uses_concurrency(node, config)


def _uses_fixtures(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args = [arg.arg for arg in node.args.args if arg.arg not in {"self", "cls"}]
    return bool(args)


def _is_boundary_assert(node: ast.Assert) -> bool:
    if not isinstance(node.test, ast.Compare):
        return False
    return any(isinstance(op, (ast.LtE, ast.GtE, ast.Lt, ast.Gt)) for op in node.test.ops)


def _with_uses_pytest_raises(node: ast.With | ast.AsyncWith) -> bool:
    return any(_is_pytest_raises(item.context_expr) for item in node.items)


def _is_pytest_raises(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.AST):
        return False
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id == "pytest" and node.attr == "raises"
    return False


def _uses_concurrency(node: ast.Call, config: SpanConfig) -> bool:
    target = resolve_call_target(node.func, config.import_map)
    if target.library is None:
        return False
    library_root = target.library.split(".", maxsplit=1)[0]
    return library_root in config.patterns.concurrency_libs


__all__ = [
    "EMPTY_FUNCTION_COVERAGE_ENTRY",
    "EMPTY_SUBSYSTEM_ENTRY",
    "EMPTY_TEST_METRICS",
    "PRIMARY_COVERAGE_THRESHOLD",
    "BehavioralProfile",
    "FunctionCoverageEntry",
    "SpanConfig",
    "SpanState",
    "SubsystemCoverageEntry",
    "TestGraphMetrics",
    "TestProfileBuildResult",
    "build_test_ast_index",
    "build_test_ast_index_for_tests",
    "build_test_profile_result",
    "compute_flakiness_score",
    "compute_importance_score",
    "infer_behavior_tags",
    "load_functions_covered",
    "load_subsystems_covered",
    "load_test_graph_metrics_public",
    "load_test_profile_context",
    "load_test_records_public",
]
