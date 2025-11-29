"""Behavioral tagging helpers for test analytics."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from codeintel.analytics.tests_profiles.coverage_inputs import load_test_records
from codeintel.analytics.tests_profiles.types import (
    BehavioralContext,
    BehavioralLLMRequest,
    BehavioralLLMRunner,
    IoFlags,
    TestAstInfo,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig
from codeintel.ingestion.ast_utils import parse_python_module
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

DEFAULT_IO_SPEC: dict[str, dict[str, list[str]]] = {
    "network": {
        "libs": ["requests", "httpx", "urllib3", "aiohttp", "socket", "boto3", "paramiko"],
        "funcs": ["get", "post", "put", "delete", "request", "send"],
    },
    "db": {
        "libs": ["sqlalchemy", "psycopg2", "asyncpg", "pymysql", "pymongo", "redis"],
        "funcs": ["execute", "session", "commit", "query"],
    },
    "filesystem": {
        "libs": ["pathlib", "os", "shutil"],
        "funcs": ["open", "unlink", "remove", "rmtree", "rename"],
    },
    "subprocess": {
        "libs": ["subprocess"],
        "funcs": ["run", "popen", "call", "check_call"],
    },
}

CONCURRENCY_LIBS: set[str] = {
    "asyncio",
    "anyio",
    "trio",
    "threading",
    "concurrent",
    "multiprocessing",
}


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
        Callable[[DuckDBConnection, BehavioralCoverageStepConfig], list[TestRecord]] | None
    ) = None
    build_ast: (
        Callable[
            [Path, Iterable[TestRecord], dict[str, dict[str, list[str]]], set[str]],
            dict[str, TestAstInfo],
        ]
        | None
    ) = None
    load_profile_ctx: (
        Callable[[DuckDBConnection, BehavioralCoverageStepConfig], Mapping[str, dict[str, object]]]
        | None
    ) = None
    row_builder: Callable[[TestRecord, BehavioralContext], tuple[object, ...]] | None = None


def build_behavior_rows(
    gateway: StorageGateway,
    cfg: BehavioralCoverageStepConfig,
    *,
    llm_runner: BehavioralLLMRunner | None = None,
    hooks: BehaviorRowHooks | None = None,
) -> list[tuple[object, ...]]:
    """
    Build behavioral coverage rows for insertion.

    Returns
    -------
    list[tuple[object, ...]]
        Rows aligned with ``analytics.behavioral_coverage`` column order.
    """
    con = gateway.con
    ensure_schema(con, "analytics.behavioral_coverage")
    load_tests_fn = hooks.load_tests if hooks is not None else None
    if load_tests_fn is None:
        load_tests_fn = load_test_records
    tests = load_tests_fn(con, cfg)
    if not tests:
        return []

    ast_builder = hooks.build_ast if hooks is not None else None
    if ast_builder is None:
        ast_builder = build_test_ast_index
    ast_info = ast_builder(cfg.repo_root, tests, DEFAULT_IO_SPEC, CONCURRENCY_LIBS)
    profile_loader = hooks.load_profile_ctx if hooks is not None else None
    if profile_loader is None:
        profile_loader = load_behavioral_context
    profile_ctx = profile_loader(con, cfg)
    behavior_ctx = BehavioralContext(
        cfg=cfg,
        ast_info=ast_info,
        profile_ctx=profile_ctx,
        now=datetime.now(tz=UTC),
        llm_runner=llm_runner,
    )
    row_fn = hooks.row_builder if hooks is not None else None
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
    """
    Infer behavior tags from name, markers, IO flags, and AST info.

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
    con: DuckDBConnection,
    cfg: BehavioralCoverageStepConfig,
) -> Mapping[str, dict[str, object]]:
    """
    Load behavioral profile context from analytics.test_profile.

    Returns
    -------
    Mapping[str, dict[str, object]]
        Context keyed by ``test_id``.

    Raises
    ------
    RuntimeError
        If the provided connection does not support execute/fetchall.
    """
    execute = getattr(con, "execute", None)
    fetchall = getattr(con, "fetchall", None)
    if execute is None or fetchall is None or not callable(execute) or not callable(fetchall):
        message = "DuckDB connection is required to load behavioral context."
        raise RuntimeError(message)
    execute_fn = cast("Callable[[str, list[object] | None], DuckDBConnection]", execute)
    fetchall_fn = cast("Callable[[], list[tuple[object, ...]]]", fetchall)

    execute_fn(
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
        [cfg.repo, cfg.commit],
    )
    rows = fetchall_fn()
    ctx: dict[str, dict[str, object]] = {}
    for (
        test_id,
        markers,
        functions_covered,
        subsystems_covered,
        assert_count,
        raise_count,
        status,
    ) in rows:
        ctx[str(test_id)] = {
            "markers": markers,
            "functions_covered": functions_covered or [],
            "subsystems_covered": subsystems_covered or [],
            "assert_count": _coerce_int(assert_count) or 0,
            "raise_count": _coerce_int(raise_count) or 0,
            "status": status,
        }
    return ctx


def build_test_ast_index(
    repo_root: Path,
    tests: Iterable[TestRecord],
    io_spec: dict[str, dict[str, list[str]]],
    concurrency_libs: set[str],
) -> dict[str, TestAstInfo]:
    """
    Build AST span index for tests using the configured IO heuristics.

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
            io_spec,
            concurrency_libs,
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
    if ctx.cfg.enable_llm and ctx.llm_runner is not None:
        llm_inputs = _LLMInputs(
            markers=markers,
            functions_covered=functions_covered,
            subsystems_covered=subsystems_covered,
            assert_count=assert_count if assert_count is not None else ast_details.assert_count,
            raise_count=raise_count if raise_count is not None else ast_details.raise_count,
        )
        llm_result = ctx.llm_runner(_build_llm_request(cfg=ctx.cfg, test=test, profile=llm_inputs))
        llm_tags = set(llm_result.tags)
        if llm_tags:
            tag_source = "mixed" if tags else "llm"
            tags = sorted(set(tags).union(llm_tags))
            llm_model = llm_result.model or ctx.cfg.llm_model
            llm_run_id = llm_result.run_id
    return (
        ctx.cfg.repo,
        ctx.cfg.commit,
        test.test_id,
        test.test_goid_h128,
        test.rel_path,
        test.qualname or test.test_id,
        tags,
        tag_source,
        ctx.cfg.heuristic_version,
        llm_model or ctx.cfg.llm_model,
        llm_run_id,
        ctx.now,
    )


def _build_llm_request(
    *,
    cfg: BehavioralCoverageStepConfig,
    test: TestRecord,
    profile: _LLMInputs,
) -> BehavioralLLMRequest:
    source = _load_source(cfg.repo_root, test.rel_path)
    return BehavioralLLMRequest(
        repo=cfg.repo,
        commit=cfg.commit,
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
    io_spec: dict[str, dict[str, list[str]]]
    concurrency_libs: set[str]


def _analyze_file(
    path: Path,
    tests: Iterable[TestRecord],
    io_spec: dict[str, dict[str, list[str]]],
    concurrency_libs: set[str],
) -> dict[str, TestAstInfo]:
    parsed = parse_python_module(path)
    if parsed is None:
        return {test.test_id: TestAstInfo() for test in tests}
    _, tree = parsed
    import_map = _build_import_map(tree)
    info: dict[str, TestAstInfo] = {}
    for test in tests:
        if test.start_line is None:
            info[test.test_id] = TestAstInfo()
            continue
        config = SpanConfig(
            import_map=import_map,
            start_line=test.start_line,
            end_line=test.end_line or test.start_line,
            io_spec=io_spec,
            concurrency_libs=concurrency_libs,
        )
        info[test.test_id] = _analyze_span(tree, config)
    return info


def _build_import_map(tree: ast.AST) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                mapping[alias.asname or alias.name] = root
        if isinstance(node, ast.ImportFrom):
            module = node.module.split(".", maxsplit=1)[0] if node.module else ""
            for alias in node.names:
                mapping[alias.asname or alias.name] = module
    return mapping


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
        state.io_flags = _update_io_flags(node, config, state.io_flags)
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
    root_name, _ = _call_root_and_attr(node.func)
    if root_name is None:
        return False
    module = config.import_map.get(root_name, root_name)
    return module in config.concurrency_libs


def _call_root_and_attr(func: ast.AST) -> tuple[str | None, str | None]:
    if isinstance(func, ast.Name):
        return func.id, func.id
    if isinstance(func, ast.Attribute):
        attr = func.attr
        value = func.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name):
            return value.id, attr
    return None, None


def _update_io_flags(
    node: ast.Call,
    config: SpanConfig,
    existing: IoFlags,
) -> IoFlags:
    root_name, attr = _call_root_and_attr(node.func)
    if root_name is None:
        return existing
    module = config.import_map.get(root_name, root_name)
    module_root = module.split(".", maxsplit=1)[0]
    attr_lower = attr.lower() if attr is not None else None

    uses_network = existing.uses_network
    uses_db = existing.uses_db
    uses_filesystem = existing.uses_filesystem
    uses_subprocess = existing.uses_subprocess

    network_spec = config.io_spec["network"]
    db_spec = config.io_spec["db"]
    filesystem_spec = config.io_spec["filesystem"]
    subprocess_spec = config.io_spec["subprocess"]

    if module_root in network_spec["libs"] or (
        attr_lower is not None and attr_lower in network_spec["funcs"]
    ):
        uses_network = True
    if module_root in db_spec["libs"] or (
        attr_lower is not None and attr_lower in db_spec["funcs"]
    ):
        uses_db = True
    if module_root in filesystem_spec["libs"] or (
        attr_lower is not None and attr_lower in filesystem_spec["funcs"]
    ):
        uses_filesystem = True
    if module_root in subprocess_spec["libs"] or (
        attr_lower is not None and attr_lower in subprocess_spec["funcs"]
    ):
        uses_subprocess = True

    return IoFlags(
        uses_network=uses_network,
        uses_db=uses_db,
        uses_filesystem=uses_filesystem,
        uses_subprocess=uses_subprocess,
    )
