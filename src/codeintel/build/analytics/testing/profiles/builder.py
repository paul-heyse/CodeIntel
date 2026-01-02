"""Build per-test profiles and behavioral coverage tags."""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.ast_features.extract import build_import_map, io_flags_from_call
from codeintel.build.analytics.ast_features.patterns import DEFAULT_PATTERNS, AstFeaturePatterns
from codeintel.build.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.build.analytics.testing.profiles.rows import (
    TestProfileInputs,
    build_test_profile_context,
    build_test_profile_rows,
)
from codeintel.build.analytics.testing.profiles.types import (
    IoFlags,
    TestAstInfo,
    TestGraphMetrics,
    TestProfileOptions,
    TestRecord,
)
from codeintel.build.analytics.utilities.ast import resolve_call_target
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.paths import path_to_module
from codeintel.core.query_results import (
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_str,
    coerce_str,
)
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.analytics.testing.profiles.types import (
        FunctionCoverageEntryProtocol,
        SubsystemCoverageEntryProtocol,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsTestProfileRow as ProfileRowModel,
    )

log = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class TestProfileFrameInputs:
    """Frame inputs required to build test profiles."""

    test_catalog_frame: pl.DataFrame | None = None
    goids_frame: pl.DataFrame | None = None
    modules_frame: pl.DataFrame | None = None
    subsystem_modules_frame: pl.DataFrame | None = None
    subsystems_frame: pl.DataFrame | None = None
    test_graph_metrics_frame: pl.DataFrame | None = None
    options: TestProfileOptions | None = None


def build_test_profile_result(
    snapshot: SnapshotRef,
    inputs: TestProfileFrameInputs,
) -> TestProfileBuildResult:
    """Compute test profile rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/ArrowDatasetSaver.

    Parameters
    ----------
    snapshot
        Snapshot reference with repo, commit, and repo_root.
    inputs
        Bundled inputs for test profile computation.

    Returns
    -------
    TestProfileBuildResult
        Container with profile row models.
    """
    opts = inputs.options or TestProfileOptions()
    tests: list[TestRecord] = _load_test_records_from_frames(
        inputs.test_catalog_frame,
        inputs.goids_frame,
        inputs.modules_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    if not tests:
        log.info("No tests found for %s@%s; skipping test_profile", snapshot.repo, snapshot.commit)
        return TestProfileBuildResult(rows=None)

    functions_covered: dict[str, FunctionCoverageEntryProtocol] = {}
    subsystems_covered: dict[str, SubsystemCoverageEntryProtocol] = {}
    tg_metrics = _load_test_graph_metrics_from_frame(
        inputs.test_graph_metrics_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    io_spec_raw = opts.io_spec if isinstance(opts.io_spec, dict) else None
    patterns = _patterns_from_io_spec(io_spec_raw)
    ast_info = _build_test_ast_index(snapshot.repo_root, tests, patterns)
    profile_inputs = TestProfileInputs(
        functions_covered=functions_covered,
        subsystems_covered=subsystems_covered,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
    ctx = build_test_profile_context(
        snapshot=snapshot,
        inputs=profile_inputs,
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


def _filter_frame_by_snapshot(frame: pl.DataFrame, *, repo: str, commit: str) -> pl.DataFrame:
    filtered = frame
    if "repo" in filtered.columns:
        filtered = filtered.filter(pl.col("repo") == repo)
    if "commit" in filtered.columns:
        filtered = filtered.filter(pl.col("commit") == commit)
    return filtered


def _load_test_records_from_frames(
    test_catalog_frame: pl.DataFrame | None,
    goids_frame: pl.DataFrame | None,
    modules_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> list[TestRecord]:
    if test_catalog_frame is None or test_catalog_frame.is_empty():
        return []
    filtered = _filter_frame_by_snapshot(test_catalog_frame, repo=repo, commit=commit)
    module_by_path: dict[str, str] = {}
    if modules_frame is not None and not modules_frame.is_empty():
        modules_filtered = _filter_frame_by_snapshot(modules_frame, repo=repo, commit=commit)
        for row in modules_filtered.iter_rows(named=True):
            path = row.get("path")
            module = row.get("module")
            if isinstance(path, str) and module is not None:
                module_by_path[path] = str(module)
    goid_meta: dict[int, dict[str, object]] = {}
    if goids_frame is not None and not goids_frame.is_empty():
        goids_filtered = _filter_frame_by_snapshot(goids_frame, repo=repo, commit=commit)
        for row in goids_filtered.iter_rows(named=True):
            goid = normalize_decimal_id(row.get("goid_h128"))
            if goid is None:
                continue
            goid_meta[goid] = {
                "qualname": row.get("qualname"),
                "language": row.get("language"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
            }
    records: list[TestRecord] = []
    for row in filtered.iter_rows(named=True):
        rel_path_text = coerce_str(row.get("rel_path"), ctx="test_catalog.rel_path")
        module_name = coerce_optional_str(
            module_by_path.get(rel_path_text),
            ctx="test_catalog.module",
        ) or path_to_module(rel_path_text)
        goid_value = normalize_decimal_id(row.get("test_goid_h128"))
        goid_info = goid_meta.get(goid_value, {}) if goid_value is not None else {}
        qualname_value = coerce_optional_str(
            row.get("qualname") or goid_info.get("qualname"),
            ctx="test_catalog.qualname",
        )
        language_value = (
            coerce_optional_str(
                row.get("language") or goid_info.get("language"),
                ctx="test_catalog.language",
            )
            or "python"
        )
        records.append(
            TestRecord(
                test_id=coerce_str(row.get("test_id"), ctx="test_catalog.test_id"),
                test_goid_h128=goid_value,
                urn=coerce_optional_str(row.get("urn"), ctx="test_catalog.urn"),
                rel_path=rel_path_text,
                module=module_name,
                qualname=qualname_value,
                language=language_value,
                kind=coerce_optional_str(row.get("kind"), ctx="test_catalog.kind"),
                status=coerce_optional_str(row.get("status"), ctx="test_catalog.status"),
                duration_ms=coerce_optional_float(
                    row.get("duration_ms"),
                    ctx="test_catalog.duration_ms",
                ),
                markers=_normalize_markers(
                    row.get("markers") if isinstance(row.get("markers"), list) else None
                ),
                flaky=bool(row.get("flaky")) if row.get("flaky") is not None else None,
                start_line=coerce_optional_int(
                    goid_info.get("start_line"),
                    ctx="test_catalog.start_line",
                ),
                end_line=coerce_optional_int(
                    goid_info.get("end_line"),
                    ctx="test_catalog.end_line",
                ),
            )
        )
    return records


def load_test_graph_metrics_public(
    test_graph_metrics_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> dict[str, TestGraphMetrics]:
    """Load test graph metrics rows from frames.

    Returns
    -------
    dict[str, TestGraphMetrics]
        Metrics keyed by ``test_id``.
    """
    return _load_test_graph_metrics_from_frame(
        test_graph_metrics_frame,
        repo=repo,
        commit=commit,
    )


def load_test_records_public(
    test_catalog_frame: pl.DataFrame | None,
    goids_frame: pl.DataFrame | None,
    modules_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> list[TestRecord]:
    """Load test records from tabular inputs.

    Returns
    -------
    list[TestRecord]
        Parsed test records for the snapshot.
    """
    return _load_test_records_from_frames(
        test_catalog_frame,
        goids_frame,
        modules_frame,
        repo=repo,
        commit=commit,
    )


def _load_test_graph_metrics_from_frame(
    frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> dict[str, TestGraphMetrics]:
    if frame is None or frame.is_empty():
        return {}
    filtered = _filter_frame_by_snapshot(frame, repo=repo, commit=commit)
    metrics: dict[str, TestGraphMetrics] = {}
    for row in filtered.iter_rows(named=True):
        test_id_text = coerce_str(row.get("test_id"), ctx="test_graph_metrics.test_id")
        metrics[test_id_text] = TestGraphMetrics(
            degree=coerce_optional_int(
                row.get("degree"),
                ctx="test_graph_metrics.degree",
            ),
            weighted_degree=coerce_optional_float(
                row.get("weighted_degree"),
                ctx="test_graph_metrics.weighted_degree",
            ),
            proj_degree=coerce_optional_int(
                row.get("proj_degree"),
                ctx="test_graph_metrics.proj_degree",
            ),
            proj_weight=coerce_optional_float(
                row.get("proj_weight"),
                ctx="test_graph_metrics.proj_weight",
            ),
            proj_clustering=coerce_optional_float(
                row.get("proj_clustering"),
                ctx="test_graph_metrics.proj_clustering",
            ),
            proj_betweenness=coerce_optional_float(
                row.get("proj_betweenness"),
                ctx="test_graph_metrics.proj_betweenness",
            ),
        )
    return metrics


def load_test_profile_context(
    test_profile_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> dict[str, dict[str, object]]:
    """Load test profile context from tabular inputs.

    Returns
    -------
    dict[str, dict[str, object]]
        Test profile context keyed by test ID.
    """
    ctx: dict[str, dict[str, object]] = {}
    if test_profile_frame is None or test_profile_frame.is_empty():
        return ctx
    filtered = _filter_frame_by_snapshot(test_profile_frame, repo=repo, commit=commit)
    for row in filtered.iter_rows(named=True):
        test_id = row.get("test_id")
        if test_id is None:
            continue
        test_id_text = coerce_str(test_id, ctx="test_profile.test_id")
        ctx[test_id_text] = {
            "markers": row.get("markers"),
            "functions_covered": row.get("functions_covered") or [],
            "subsystems_covered": row.get("subsystems_covered") or [],
            "assert_count": coerce_optional_int(
                row.get("assert_count"),
                ctx="test_profile.assert_count",
            )
            or 0,
            "raise_count": coerce_optional_int(
                row.get("raise_count"),
                ctx="test_profile.raise_count",
            )
            or 0,
            "status": coerce_optional_str(row.get("status"), ctx="test_profile.status"),
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
    "BehavioralProfile",
    "SpanConfig",
    "SpanState",
    "TestGraphMetrics",
    "TestProfileBuildResult",
    "build_test_ast_index",
    "build_test_ast_index_for_tests",
    "build_test_profile_result",
    "compute_flakiness_score",
    "compute_importance_score",
    "infer_behavior_tags",
    "load_test_graph_metrics_public",
    "load_test_profile_context",
    "load_test_records_public",
]
