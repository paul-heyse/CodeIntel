"""Build per-test profiles and behavioral coverage tags."""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.ingestion.ast_utils import parse_python_module
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.utils.paths import relpath_to_module

log = logging.getLogger(__name__)

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

PRIMARY_COVERAGE_THRESHOLD = 0.4


@dataclass(frozen=True)
class IoFlags:
    """Flags describing IO usage within a test."""

    uses_network: bool = False
    uses_db: bool = False
    uses_filesystem: bool = False
    uses_subprocess: bool = False

    @property
    def io_bound(self) -> bool:
        """Return True when any IO flag is set."""
        return self.uses_network or self.uses_db or self.uses_filesystem or self.uses_subprocess


@dataclass(frozen=True)
class TestAstInfo:
    """AST-derived metrics for a single test span."""

    __test__ = False

    assert_count: int = 0
    raise_count: int = 0
    uses_pytest_raises: bool = False
    uses_concurrency_lib: bool = False
    has_boundary_asserts: bool = False
    uses_fixtures: bool = False
    io_flags: IoFlags = IoFlags()


@dataclass(frozen=True)
class TestRecord:
    """Identity and span information for a test."""

    __test__ = False

    test_id: str
    test_goid_h128: int | None
    urn: str | None
    rel_path: str
    module: str | None
    qualname: str | None
    language: str | None
    kind: str | None
    status: str | None
    duration_ms: float | None
    markers: list[str]
    flaky: bool | None
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True)
class TestProfileContext:
    """Shared inputs for building test_profile rows."""

    cfg: TestProfileStepConfig
    now: datetime
    max_function_count: int
    max_weighted_degree: float
    max_subsystem_risk: float
    functions_covered: dict[str, FunctionCoverageEntry]
    subsystems_covered: dict[str, SubsystemCoverageEntry]
    tg_metrics: dict[str, TestGraphMetrics]
    ast_info: dict[str, TestAstInfo]


@dataclass(frozen=True)
class ImportanceInputs:
    """Inputs required to compute test importance."""

    functions_covered_count: int
    weighted_degree: float | None
    max_function_count: int
    max_weighted_degree: float
    subsystem_risk: float | None
    max_subsystem_risk: float


@dataclass(frozen=True)
class FunctionCoverageEntry:
    """Coverage details for a test-to-function mapping."""

    functions: list[dict[str, object]]
    count: int
    primary: list[int]


@dataclass(frozen=True)
class SubsystemCoverageEntry:
    """Coverage details for subsystems touched by a test."""

    subsystems: list[dict[str, object]]
    count: int
    primary_subsystem_id: str | None
    max_risk_score: float | None


@dataclass(frozen=True)
class TestGraphMetrics:
    """Graph metrics for a single test node."""

    degree: int | None
    weighted_degree: float | None
    proj_degree: int | None
    proj_weight: float | None
    proj_clustering: float | None
    proj_betweenness: float | None


@dataclass(frozen=True)
class BehavioralLLMRequest:
    """Payload sent to an LLM classifier for behavioral coverage."""

    repo: str
    commit: str
    test_id: str
    rel_path: str
    qualname: str
    markers: list[str]
    functions_covered: list[dict[str, object]]
    subsystems_covered: list[dict[str, object]]
    assert_count: int
    raise_count: int
    status: str | None
    source: str | None


@dataclass(frozen=True)
class BehavioralProfile:
    """Existing behavioral signals pulled from test_profile."""

    functions_covered: list[dict[str, object]]
    subsystems_covered: list[dict[str, object]]
    assert_count: int
    raise_count: int
    markers: list[str]


@dataclass(frozen=True)
class BehavioralLLMResult:
    """LLM classification result for behavioral coverage."""

    tags: list[str]
    model: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class BehavioralContext:
    """Context for behavioral coverage tagging."""

    cfg: BehavioralCoverageStepConfig
    ast_info: dict[str, TestAstInfo]
    profile_ctx: dict[str, dict[str, object]]
    now: datetime
    llm_runner: BehavioralLLMRunner | None


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


def build_test_profile(gateway: StorageGateway, cfg: TestProfileStepConfig) -> None:
    """
    Populate analytics.test_profile for a repo snapshot.

    Parameters
    ----------
    gateway :
        Storage gateway bound to the target DuckDB database.
    cfg :
        Configuration containing repo identity and parsing options.
    """
    con = gateway.con
    ensure_schema(con, "analytics.test_profile")
    tests = _load_test_records(con, cfg.repo, cfg.commit)
    if not tests:
        log.info("No tests found for %s@%s; skipping test_profile", cfg.repo, cfg.commit)
        return

    con.execute(
        "DELETE FROM analytics.test_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    ctx = _build_profile_context(con=con, cfg=cfg, tests=tests)
    rows = [_build_test_profile_row(test, ctx) for test in tests]

    con.executemany(
        """
        INSERT INTO analytics.test_profile (
            repo,
            commit,
            test_id,
            test_goid_h128,
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
            last_run_at,
            functions_covered,
            functions_covered_count,
            primary_function_goids,
            subsystems_covered,
            subsystems_covered_count,
            primary_subsystem_id,
            assert_count,
            raise_count,
            uses_parametrize,
            uses_fixtures,
            io_bound,
            uses_network,
            uses_db,
            uses_filesystem,
            uses_subprocess,
            flakiness_score,
            importance_score,
            notes,
            tg_degree,
            tg_weighted_degree,
            tg_proj_degree,
            tg_proj_weight,
            tg_proj_clustering,
            tg_proj_betweenness,
            created_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )
    log.info("test_profile populated: %d rows for %s@%s", len(rows), cfg.repo, cfg.commit)


def _build_profile_context(
    *,
    con: DuckDBConnection,
    cfg: TestProfileStepConfig,
    tests: list[TestRecord],
) -> TestProfileContext:
    now = datetime.now(tz=UTC)
    io_spec_raw = cfg.io_spec if isinstance(cfg.io_spec, dict) else None
    io_spec: dict[str, dict[str, list[str]]] = (
        cast("dict[str, dict[str, list[str]]]", io_spec_raw)
        if io_spec_raw is not None
        else DEFAULT_IO_SPEC
    )
    functions_covered = _load_functions_covered(con, cfg.repo, cfg.commit)
    subsystems_covered = _load_subsystems_covered(con, cfg.repo, cfg.commit)
    tg_metrics = _load_test_graph_metrics(con, cfg.repo, cfg.commit)
    ast_info = _build_test_ast_index(cfg.repo_root, tests, io_spec, CONCURRENCY_LIBS)
    max_function_count = max((entry.count for entry in functions_covered.values()), default=0)
    max_weighted_degree = max(
        (metrics.weighted_degree or 0.0 for metrics in tg_metrics.values()), default=0.0
    )
    max_subsystem_risk = max(
        (entry.max_risk_score or 0.0 for entry in subsystems_covered.values()),
        default=0.0,
    )
    return TestProfileContext(
        cfg=cfg,
        now=now,
        max_function_count=max_function_count,
        max_weighted_degree=max_weighted_degree,
        max_subsystem_risk=max_subsystem_risk,
        functions_covered=functions_covered,
        subsystems_covered=subsystems_covered,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )


def _build_test_profile_row(test: TestRecord, ctx: TestProfileContext) -> tuple[object, ...]:
    markers = _normalize_markers(test.markers)
    ast_details = ctx.ast_info.get(test.test_id, TestAstInfo())
    cov_entry = ctx.functions_covered.get(test.test_id, EMPTY_FUNCTION_COVERAGE_ENTRY)
    subs_entry = ctx.subsystems_covered.get(test.test_id, EMPTY_SUBSYSTEM_ENTRY)
    tg_entry = ctx.tg_metrics.get(test.test_id, EMPTY_TEST_METRICS)
    uses_parametrize = _uses_parametrize(test, markers)
    uses_fixtures = ast_details.uses_fixtures or _markers_use_fixtures(markers)
    flakiness = compute_flakiness_score(
        status=test.status,
        markers=markers,
        duration_ms=test.duration_ms,
        io_flags=ast_details.io_flags,
        slow_test_threshold_ms=ctx.cfg.slow_test_threshold_ms,
    )
    importance_inputs = ImportanceInputs(
        functions_covered_count=cov_entry.count,
        weighted_degree=tg_entry.weighted_degree,
        max_function_count=ctx.max_function_count,
        max_weighted_degree=ctx.max_weighted_degree,
        subsystem_risk=subs_entry.max_risk_score,
        max_subsystem_risk=ctx.max_subsystem_risk,
    )
    importance = compute_importance_score(importance_inputs)
    return (
        ctx.cfg.repo,
        ctx.cfg.commit,
        test.test_id,
        test.test_goid_h128,
        test.urn,
        test.rel_path,
        test.module,
        test.qualname,
        test.language or "python",
        test.kind,
        test.status,
        test.duration_ms,
        markers,
        test.flaky,
        ctx.now,
        cov_entry.functions,
        cov_entry.count,
        cov_entry.primary,
        subs_entry.subsystems,
        subs_entry.count,
        subs_entry.primary_subsystem_id,
        ast_details.assert_count,
        ast_details.raise_count,
        uses_parametrize,
        uses_fixtures,
        ast_details.io_flags.io_bound,
        ast_details.io_flags.uses_network,
        ast_details.io_flags.uses_db,
        ast_details.io_flags.uses_filesystem,
        ast_details.io_flags.uses_subprocess,
        flakiness,
        importance,
        None,
        tg_entry.degree,
        tg_entry.weighted_degree,
        tg_entry.proj_degree,
        tg_entry.proj_weight,
        tg_entry.proj_clustering,
        tg_entry.proj_betweenness,
        ctx.now,
    )


BehavioralLLMRunner = Callable[[BehavioralLLMRequest], BehavioralLLMResult]


def build_behavioral_coverage(
    gateway: StorageGateway,
    cfg: BehavioralCoverageStepConfig,
    llm_runner: BehavioralLLMRunner | None = None,
) -> None:
    """
    Populate analytics.behavioral_coverage using heuristic tags.

    Parameters
    ----------
    gateway :
        Storage gateway bound to the target DuckDB database.
    cfg :
        Configuration containing repo identity for tagging.
    llm_runner :
        Optional callable that returns LLM-derived behavior tags; when absent, only heuristic
        tagging runs.
    """
    con = gateway.con
    ensure_schema(con, "analytics.behavioral_coverage")
    tests = _load_test_records(con, cfg.repo, cfg.commit)
    if not tests:
        log.info("No tests found for %s@%s; skipping behavioral tags", cfg.repo, cfg.commit)
        return

    ast_info = _build_test_ast_index(cfg.repo_root, tests, DEFAULT_IO_SPEC, CONCURRENCY_LIBS)
    profile_ctx = _load_test_profile_context(con, cfg.repo, cfg.commit)
    con.execute(
        "DELETE FROM analytics.behavioral_coverage WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    behavior_ctx = BehavioralContext(
        cfg=cfg,
        ast_info=ast_info,
        profile_ctx=profile_ctx,
        now=datetime.now(tz=UTC),
        llm_runner=llm_runner,
    )
    rows = [_build_behavior_row(test, behavior_ctx) for test in tests]

    con.executemany(
        """
        INSERT INTO analytics.behavioral_coverage (
            repo,
            commit,
            test_id,
            test_goid_h128,
            rel_path,
            qualname,
            behavior_tags,
            tag_source,
            heuristic_version,
            llm_model,
            llm_run_id,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


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
        profile_inputs = BehavioralProfile(
            functions_covered=functions_covered,
            subsystems_covered=subsystems_covered,
            assert_count=assert_count if assert_count is not None else ast_details.assert_count,
            raise_count=raise_count if raise_count is not None else ast_details.raise_count,
            markers=markers,
        )
        llm_result = ctx.llm_runner(_build_llm_request(ctx.cfg, test, profile_inputs))
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


def compute_flakiness_score(
    *,
    status: str | None,
    markers: Iterable[str],
    duration_ms: float | None,
    io_flags: IoFlags,
    slow_test_threshold_ms: float,
) -> float:
    """
    Derive a heuristic flakiness score in the range [0.0, 1.0].

    Parameters
    ----------
    status :
        Last known pytest status for the test.
    markers :
        Iterable of marker strings associated with the test.
    duration_ms :
        Execution duration in milliseconds when available.
    io_flags :
        Flags capturing whether the test uses network, DB, filesystem, or subprocess resources.
    slow_test_threshold_ms :
        Threshold used to treat a test as slow.

    Returns
    -------
    float
        Score between 0.0 and 1.0 capturing flakiness risk.
    """
    score = 0.0
    markers_lower = [marker.lower() for marker in markers]
    if any("flaky" in marker for marker in markers_lower):
        score += 0.6
    if status is not None and status.lower() in {"xfail", "xpass"}:
        score += 0.2
    if io_flags.uses_network:
        score += 0.15
    if io_flags.uses_db or io_flags.uses_subprocess:
        score += 0.1
    if io_flags.uses_filesystem:
        score += 0.05
    if duration_ms is not None and duration_ms > slow_test_threshold_ms:
        score += 0.1
    return min(score, 1.0)


def compute_importance_score(inputs: ImportanceInputs) -> float | None:
    """
    Estimate relative importance using coverage breadth and graph metrics.

    Parameters
    ----------
    inputs :
        Structured inputs containing coverage breadth, graph weights, and subsystem risk.

    Returns
    -------
    float | None
        Normalized importance in [0, 1] or None when no signals are present.
    """
    scores: list[float] = []
    if inputs.functions_covered_count > 0 and inputs.max_function_count > 0:
        scores.append(inputs.functions_covered_count / inputs.max_function_count)
    if inputs.weighted_degree is not None and inputs.max_weighted_degree > 0:
        scores.append(inputs.weighted_degree / inputs.max_weighted_degree)
    if inputs.subsystem_risk is not None and inputs.max_subsystem_risk > 0:
        scores.append(inputs.subsystem_risk / inputs.max_subsystem_risk)
    if not scores:
        return None
    return min(sum(scores) / len(scores), 1.0)


def infer_behavior_tags(
    *,
    name: str,
    markers: Iterable[str],
    io_flags: IoFlags,
    ast_info: TestAstInfo,
) -> list[str]:
    """
    Infer behavior coverage tags from names, markers, IO flags, and AST hints.

    Parameters
    ----------
    name :
        Qualified test name or pytest nodeid.
    markers :
        Iterable of marker strings associated with the test.
    io_flags :
        IO usage flags inferred from AST analysis.
    ast_info :
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


def _build_llm_request(
    cfg: BehavioralCoverageStepConfig,
    test: TestRecord,
    profile: BehavioralProfile,
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


def _uses_parametrize(test: TestRecord, markers: Iterable[str]) -> bool:
    markers_lower = [marker.lower() for marker in markers]
    if test.kind == "parametrized_case":
        return True
    if any("parametrize" in marker for marker in markers_lower):
        return True
    qual = test.qualname or ""
    return "[" in qual and "]" in qual


def _markers_use_fixtures(markers: Iterable[str]) -> bool:
    return any("usefixtures" in marker.lower() for marker in markers)


def _load_test_records(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> list[TestRecord]:
    rows = con.execute(
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
    ).fetchall()
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
    ) in rows:
        module_name = str(module) if module is not None else relpath_to_module(str(rel_path))
        records.append(
            TestRecord(
                test_id=str(test_id),
                test_goid_h128=int(goid) if goid is not None else None,
                urn=str(urn) if urn is not None else None,
                rel_path=str(rel_path),
                module=module_name,
                qualname=str(qualname) if qualname is not None else None,
                language=str(language) if language is not None else None,
                kind=str(kind) if kind is not None else None,
                status=str(status) if status is not None else None,
                duration_ms=float(duration_ms) if duration_ms is not None else None,
                markers=_normalize_markers(markers),
                flaky=bool(flaky) if flaky is not None else None,
                start_line=int(start_line) if start_line is not None else None,
                end_line=int(end_line) if end_line is not None else None,
            )
        )
    return records


def _load_functions_covered(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, FunctionCoverageEntry]:
    rows = con.execute(
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
    ).fetchall()

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
    ) in rows:
        module_name = module if module is not None else relpath_to_module(str(rel_path))
        test_key = str(test_id)
        entry = result.get(test_key)
        if entry is None:
            entry = FunctionCoverageEntry(functions=[], count=0, primary=[])
            result[test_key] = entry
        functions = list(entry.functions)
        primary = list(entry.primary)
        functions.append(
            {
                "function_goid_h128": (
                    int(function_goid_h128) if function_goid_h128 is not None else None
                ),
                "urn": urn,
                "module": module_name,
                "qualname": qualname,
                "rel_path": rel_path,
                "coverage_ratio": float(coverage_ratio) if coverage_ratio is not None else None,
                "coverage_share": float(coverage_share) if coverage_share is not None else None,
            }
        )
        if (
            function_goid_h128 is not None
            and coverage_share is not None
            and float(coverage_share) >= PRIMARY_COVERAGE_THRESHOLD
        ):
            primary.append(int(function_goid_h128))
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
    rows = con.execute(
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
    ).fetchall()

    result: dict[str, SubsystemCoverageEntry] = {}
    for test_id, subsystem_id, coverage_share, name, max_risk_score in rows:
        test_key = str(test_id)
        entry = result.get(test_key) or SubsystemCoverageEntry(
            subsystems=[],
            count=0,
            primary_subsystem_id=None,
            max_risk_score=0.0,
        )
        share = float(coverage_share) if coverage_share is not None else 0.0
        subsystems = list(entry.subsystems)
        subsystems.append({"subsystem_id": subsystem_id, "name": name, "coverage_share": share})
        primary_subsystem_id = entry.primary_subsystem_id
        primary_share = share if primary_subsystem_id == subsystem_id else -1.0
        if primary_subsystem_id is None or share > primary_share:
            primary_subsystem_id = subsystem_id
        result[test_key] = SubsystemCoverageEntry(
            subsystems=subsystems,
            count=len(subsystems),
            primary_subsystem_id=primary_subsystem_id,
            max_risk_score=max(entry.max_risk_score or 0.0, max_risk_score or 0.0),
        )
    return result


def _load_test_graph_metrics(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, TestGraphMetrics]:
    rows = con.execute(
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
    ).fetchall()
    metrics: dict[str, TestGraphMetrics] = {}
    for (
        test_id,
        degree,
        weighted_degree,
        proj_degree,
        proj_weight,
        proj_clustering,
        proj_betweenness,
    ) in rows:
        metrics[str(test_id)] = TestGraphMetrics(
            degree=int(degree) if degree is not None else None,
            weighted_degree=float(weighted_degree) if weighted_degree is not None else None,
            proj_degree=int(proj_degree) if proj_degree is not None else None,
            proj_weight=float(proj_weight) if proj_weight is not None else None,
            proj_clustering=float(proj_clustering) if proj_clustering is not None else None,
            proj_betweenness=float(proj_betweenness) if proj_betweenness is not None else None,
        )
    return metrics


def _load_test_profile_context(
    con: DuckDBConnection,
    repo: str,
    commit: str,
) -> dict[str, dict[str, object]]:
    rows = con.execute(
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
    ).fetchall()
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
            "assert_count": int(assert_count) if assert_count is not None else 0,
            "raise_count": int(raise_count) if raise_count is not None else 0,
            "status": status,
        }
    return ctx


def build_test_ast_index_for_tests(
    repo_root: Path,
    tests: Iterable[TestRecord],
) -> dict[str, TestAstInfo]:
    """
    Build AST span index for tests using default IO heuristics.

    This helper is intended for unit tests that need the same parsing behavior as production.

    Returns
    -------
    dict[str, TestAstInfo]
        Mapping from test IDs to AST-derived metrics.
    """
    return _build_test_ast_index(repo_root, tests, DEFAULT_IO_SPEC, CONCURRENCY_LIBS)


def _build_test_ast_index(
    repo_root: Path,
    tests: Iterable[TestRecord],
    io_spec: dict[str, dict[str, list[str]]],
    concurrency_libs: set[str],
) -> dict[str, TestAstInfo]:
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
