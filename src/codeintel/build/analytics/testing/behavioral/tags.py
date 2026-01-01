"""Behavioral tagging helpers for test analytics."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.ast_features.extract import build_import_map, io_flags_from_call
from codeintel.build.analytics.ast_features.model import IoFlags
from codeintel.build.analytics.ast_features.patterns import DEFAULT_PATTERNS
from codeintel.build.analytics.testing.profiles.types import (
    BehavioralContext,
    BehavioralCoverageOptions,
    BehavioralLLMRequest,
    BehavioralLLMRunner,
    TestAstInfo,
    TestRecord,
)
from codeintel.build.analytics.utilities.ast import resolve_call_target
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import (
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_str,
    coerce_str,
)
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from codeintel.build.analytics.ast_features.patterns import AstFeaturePatterns
    from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True)
class _LLMInputs:
    """Typed inputs forwarded to the LLM request builder."""

    markers: list[str]
    functions_covered: list[dict[str, object]]
    subsystems_covered: list[dict[str, object]]
    assert_count: int
    raise_count: int


@dataclass(frozen=True)
class BehaviorRowHooks:
    """Optional hooks to override behavioral row inputs for testing."""

    load_tests: (
        Callable[
            [pl.DataFrame | None, pl.DataFrame | None, pl.DataFrame | None, SnapshotRef],
            list[TestRecord],
        ]
        | None
    ) = None
    build_ast: (
        Callable[
            [Path, Iterable[TestRecord], AstFeaturePatterns],
            dict[str, TestAstInfo],
        ]
        | None
    ) = None
    load_profile_ctx: (
        Callable[[pl.DataFrame | None, SnapshotRef], Mapping[str, dict[str, object]]] | None
    ) = None
    row_builder: Callable[[TestRecord, BehavioralContext], tuple[object, ...]] | None = None


@dataclass(frozen=True)
class BehavioralRowInputs:
    """Inputs required to build behavioral coverage rows."""

    test_catalog_frame: pl.DataFrame | None = None
    goids_frame: pl.DataFrame | None = None
    modules_frame: pl.DataFrame | None = None
    test_profile_frame: pl.DataFrame | None = None
    options: BehavioralCoverageOptions | None = None
    llm_runner: BehavioralLLMRunner | None = None
    hooks: BehaviorRowHooks | None = None


def _default_load_test_records(
    test_catalog_frame: pl.DataFrame | None,
    goids_frame: pl.DataFrame | None,
    modules_frame: pl.DataFrame | None,
    snapshot: SnapshotRef,
) -> list[TestRecord]:
    """Load test records from tabular inputs.

    Returns
    -------
    list[TestRecord]
        Parsed test records for the snapshot.
    """
    if test_catalog_frame is None or test_catalog_frame.is_empty():
        return []
    filtered = _filter_frame_by_snapshot(test_catalog_frame, snapshot)
    module_by_path: dict[str, str] = {}
    if modules_frame is not None and not modules_frame.is_empty():
        modules_filtered = _filter_frame_by_snapshot(modules_frame, snapshot)
        for row in modules_filtered.iter_rows(named=True):
            path = row.get("path")
            module = row.get("module")
            if isinstance(path, str) and module is not None:
                module_by_path[path] = str(module)
    goid_meta: dict[int, dict[str, object]] = {}
    if goids_frame is not None and not goids_frame.is_empty():
        goids_filtered = _filter_frame_by_snapshot(goids_frame, snapshot)
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
        test_id = coerce_str(row.get("test_id"), ctx="test_catalog.test_id")
        test_goid = normalize_decimal_id(row.get("test_goid_h128"))
        rel_path = coerce_str(row.get("rel_path"), ctx="test_catalog.rel_path")
        module = module_by_path.get(rel_path)
        goid_info = goid_meta.get(test_goid, {}) if test_goid is not None else {}
        qualname = coerce_optional_str(
            row.get("qualname") or goid_info.get("qualname"),
            ctx="test_catalog.qualname",
        )
        language = (
            coerce_optional_str(
                row.get("language") or goid_info.get("language"),
                ctx="test_catalog.language",
            )
            or "python"
        )
        records.append(
            TestRecord(
                test_id=test_id,
                test_goid_h128=test_goid,
                urn=coerce_optional_str(row.get("urn"), ctx="test_catalog.urn"),
                rel_path=rel_path,
                module=coerce_optional_str(module, ctx="test_catalog.module"),
                qualname=qualname,
                language=language,
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


def build_behavior_rows(
    snapshot: SnapshotRef,
    inputs: BehavioralRowInputs,
) -> list[tuple[object, ...]]:
    """Build behavioral coverage rows for insertion.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    inputs
        Bundled inputs for behavioral coverage classification.

    Returns
    -------
    list[tuple[object, ...]]
        Rows aligned with ``analytics.behavioral_coverage`` column order.
    """
    opts = inputs.options or BehavioralCoverageOptions()
    load_tests_fn = inputs.hooks.load_tests if inputs.hooks is not None else None
    if load_tests_fn is None:
        load_tests_fn = _default_load_test_records
    tests = load_tests_fn(
        inputs.test_catalog_frame,
        inputs.goids_frame,
        inputs.modules_frame,
        snapshot,
    )
    if not tests:
        return []

    ast_builder = inputs.hooks.build_ast if inputs.hooks is not None else None
    if ast_builder is None:
        ast_builder = build_test_ast_index
    ast_info = ast_builder(snapshot.repo_root, tests, DEFAULT_PATTERNS)
    profile_loader = inputs.hooks.load_profile_ctx if inputs.hooks is not None else None
    if profile_loader is None:
        profile_loader = load_behavioral_context
    profile_ctx = profile_loader(inputs.test_profile_frame, snapshot)
    llm_runner = inputs.llm_runner or BehavioralLLMRunner()
    behavior_ctx = BehavioralContext(
        snapshot=snapshot,
        options=opts,
        ast_info=ast_info,
        profile_ctx=profile_ctx,
        now=datetime.now(tz=UTC),
        llm_runner=llm_runner,
    )
    row_fn = inputs.hooks.row_builder if inputs.hooks is not None else None
    if row_fn is None:
        row_fn = _build_behavior_row
    return [row_fn(test, behavior_ctx) for test in tests]


def infer_behavior_tags(
    *,
    name: str,
    markers: Iterable[str],
    io_flags: IoFlags,
    ast_info: TestAstInfo,
) -> list[str]:
    """Infer behavior tags from name, markers, IO flags, and AST info.

    Returns
    -------
    list[str]
        Sorted list of inferred behavior tags.
    """
    lower_name = name.lower()
    lower_markers = [marker.lower() for marker in markers]
    tags: set[str] = set()
    tags.update(_tags_from_name(lower_name))
    tags.update(_tags_from_markers(lower_markers))
    tags.update(_tags_from_io_flags(io_flags))
    tags.update(_tags_from_ast_info(ast_info))
    return sorted(tags)


def load_behavioral_context(
    test_profile_frame: pl.DataFrame | None,
    snapshot: SnapshotRef,
) -> Mapping[str, dict[str, object]]:
    """Load behavioral profile context from analytics.test_profile.

    Returns
    -------
    Mapping[str, dict[str, object]]
        Behavioral context keyed by test ID.
    """
    ctx: dict[str, dict[str, object]] = {}
    if test_profile_frame is None or test_profile_frame.is_empty():
        return ctx
    filtered = _filter_frame_by_snapshot(test_profile_frame, snapshot)
    for row in filtered.iter_rows(named=True):
        test_id = row.get("test_id")
        if test_id is None:
            continue
        ctx[str(test_id)] = {
            "markers": row.get("markers"),
            "functions_covered": row.get("functions_covered") or [],
            "subsystems_covered": row.get("subsystems_covered") or [],
            "assert_count": _coerce_int(row.get("assert_count")) or 0,
            "raise_count": _coerce_int(row.get("raise_count")) or 0,
            "status": row.get("status"),
        }
    return ctx


def build_test_ast_index(
    repo_root: Path,
    tests: Iterable[TestRecord],
    patterns: AstFeaturePatterns,
) -> dict[str, TestAstInfo]:
    """Build AST span index for tests using the configured IO heuristics.

    Returns
    -------
    dict[str, TestAstInfo]
        AST-derived info keyed by ``test_id``.
    """
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


def _build_behavior_row(test: TestRecord, ctx: BehavioralContext) -> tuple[object, ...]:
    profile = ctx.profile_ctx.get(test.test_id, {})
    markers_value = profile.get("markers")
    markers = _normalize_markers(markers_value if isinstance(markers_value, list) else test.markers)
    functions_covered = _as_dict_list(profile.get("functions_covered"))
    subsystems_covered = _as_dict_list(profile.get("subsystems_covered"))
    assert_count = _coerce_int(profile.get("assert_count"))
    raise_count = _coerce_int(profile.get("raise_count"))
    ast_details = ctx.ast_info.get(test.test_id, TestAstInfo())
    tags = infer_behavior_tags(
        name=test.qualname or test.test_id,
        markers=markers,
        io_flags=ast_details.io_flags,
        ast_info=ast_details,
    )
    tag_source = "heuristic"
    llm_model = None
    llm_run_id = None
    if ctx.options.enable_llm and ctx.llm_runner is not None:
        llm_inputs = _LLMInputs(
            markers=markers,
            functions_covered=functions_covered,
            subsystems_covered=subsystems_covered,
            assert_count=assert_count if assert_count is not None else ast_details.assert_count,
            raise_count=raise_count if raise_count is not None else ast_details.raise_count,
        )
        llm_result = ctx.llm_runner(
            _build_llm_request(snapshot=ctx.snapshot, test=test, profile=llm_inputs)
        )
        llm_tags = set(llm_result.tags)
        if llm_tags:
            tag_source = "mixed" if tags else "llm"
            tags = sorted(set(tags).union(llm_tags))
            llm_model = llm_result.model or ctx.options.llm_model
            llm_run_id = llm_result.run_id
    return (
        ctx.snapshot.repo,
        ctx.snapshot.commit,
        test.test_id,
        test.test_goid_h128,
        test.rel_path,
        test.qualname or test.test_id,
        tags,
        tag_source,
        ctx.options.heuristic_version,
        llm_model or ctx.options.llm_model,
        llm_run_id,
        ctx.now,
    )


def _build_llm_request(
    *,
    snapshot: SnapshotRef,
    test: TestRecord,
    profile: _LLMInputs,
) -> BehavioralLLMRequest:
    source = _load_source(snapshot.repo_root, test.rel_path)
    return BehavioralLLMRequest(
        repo=snapshot.repo,
        commit=snapshot.commit,
        test_id=test.test_id,
        rel_path=test.rel_path,
        qualname=test.qualname or test.test_id,
        markers=profile.markers,
        functions_covered=profile.functions_covered,
        subsystems_covered=profile.subsystems_covered,
        assert_count=profile.assert_count,
        raise_count=profile.raise_count,
        status=test.status,
        source=source,
    )


def _load_source(repo_root: Path, rel_path: str) -> str | None:
    path = repo_root / rel_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _normalize_markers(markers: list[str] | None) -> list[str]:
    if markers is None:
        return []
    return [str(marker) for marker in markers]


def _coerce_int(value: object | None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _filter_frame_by_snapshot(frame: pl.DataFrame, snapshot: SnapshotRef) -> pl.DataFrame:
    filtered = frame
    if "repo" in filtered.columns:
        filtered = filtered.filter(pl.col("repo") == snapshot.repo)
    if "commit" in filtered.columns:
        filtered = filtered.filter(pl.col("commit") == snapshot.commit)
    return filtered


def _as_dict_list(value: object | None) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [entry for entry in value if isinstance(entry, dict)]


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
