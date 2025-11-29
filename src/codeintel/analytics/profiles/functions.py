"""Function profile recipe helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from codeintel.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.analytics.profiles.types import (
    CoverageSummary,
    FunctionBaseInfo,
    FunctionContractView,
    FunctionDocView,
    FunctionEffectsView,
    FunctionGraphFeatures,
    FunctionHistoryView,
    FunctionProfileInputs,
    FunctionRiskView,
    FunctionRoleView,
)
from codeintel.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
    int_or_default,
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.config import ProfilesAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import (
    FUNCTION_PROFILE_COLUMNS,
    FunctionProfileRowModel,
    function_profile_row_to_tuple,
)
from codeintel.storage.sql_helpers import ensure_schema

log = logging.getLogger(__name__)

SLOW_TEST_THRESHOLD_MS = 1000.0


@dataclass(frozen=True)
class FunctionProfileViews:
    """Container for per-function views used to assemble profiles."""

    base_by_func: Mapping[int, FunctionBaseInfo]
    risk_by_func: Mapping[int, FunctionRiskView]
    coverage_by_func: Mapping[int, CoverageSummary]
    graph_by_func: Mapping[int, FunctionGraphFeatures]
    effects_by_func: Mapping[int, FunctionEffectsView]
    contracts_by_func: Mapping[int, FunctionContractView]
    roles_by_func: Mapping[int, FunctionRoleView]
    docs_by_func: Mapping[int, FunctionDocView]
    history_by_func: Mapping[int, FunctionHistoryView]


def compute_function_profile_inputs(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    slow_test_threshold_ms: float = SLOW_TEST_THRESHOLD_MS,
) -> FunctionProfileInputs:
    """
    Normalize snapshot inputs for function profile computations.

    The returned object is intentionally lightweight; heavy lifting happens in
    downstream helpers.

    Returns
    -------
    FunctionProfileInputs
        Snapshot handle used by downstream helpers.
    """
    return FunctionProfileInputs(
        con=gateway.con,
        repo=cfg.repo,
        commit=cfg.commit,
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=slow_test_threshold_ms,
    )


def load_function_base_info(
    inputs: FunctionProfileInputs,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> Mapping[int, FunctionBaseInfo]:
    """Load static per-function metadata from metrics and typedness tables.

    Returns
    -------
    Mapping[int, FunctionBaseInfo]
        Mapping keyed by function GOID with static attributes.

    Raises
    ------
    ValueError
        If an unexpected module table name is provided.
    """
    con = inputs.con
    sql_core = """
        SELECT
            rf.function_goid_h128,
            rf.urn,
            rf.repo,
            rf.commit,
            rf.rel_path,
            m.module,
            rf.language,
            rf.kind,
            rf.qualname,
            fm.start_line,
            fm.end_line,
            rf.loc,
            rf.logical_loc,
            rf.cyclomatic_complexity,
            rf.complexity_bucket,
            fm.param_count,
            fm.positional_params,
            fm.keyword_only_params AS keyword_params,
            fm.has_varargs AS vararg,
            fm.has_varkw AS kwarg,
            fm.max_nesting_depth,
            fm.stmt_count,
            fm.decorator_count,
            fm.has_docstring,
            ft.total_params,
            ft.annotated_params,
            ft.return_type,
            ft.param_types,
            ft.fully_typed,
            ft.partial_typed,
            ft.untyped,
            rf.typedness_bucket,
            rf.typedness_source,
            rf.file_typed_ratio,
            rf.static_error_count,
            rf.has_static_errors
        FROM analytics.goid_risk_factors AS rf
        LEFT JOIN analytics.function_metrics AS fm
          ON rf.function_goid_h128 = fm.function_goid_h128
         AND rf.repo = fm.repo
         AND rf.commit = fm.commit
        LEFT JOIN analytics.function_types AS ft
          ON rf.function_goid_h128 = ft.function_goid_h128
         AND rf.repo = ft.repo
         AND rf.commit = ft.commit
        LEFT JOIN core.modules AS m
          ON m.path = rf.rel_path
         AND (m.repo IS NULL OR m.repo = rf.repo)
         AND (m.commit IS NULL OR m.commit = rf.commit)
        WHERE rf.repo = ? AND rf.commit = ?
        """
    sql_catalog = sql_core.replace("core.modules", CATALOG_MODULE_TABLE)
    if module_table == DEFAULT_MODULE_TABLE:
        sql = sql_core
    elif module_table == CATALOG_MODULE_TABLE:
        sql = sql_catalog
    else:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    rows = con.execute(sql, [inputs.repo, inputs.commit]).fetchall()
    columns = [
        "function_goid_h128",
        "urn",
        "repo",
        "commit",
        "rel_path",
        "module",
        "language",
        "kind",
        "qualname",
        "start_line",
        "end_line",
        "loc",
        "logical_loc",
        "cyclomatic_complexity",
        "complexity_bucket",
        "param_count",
        "positional_params",
        "keyword_params",
        "vararg",
        "kwarg",
        "max_nesting_depth",
        "stmt_count",
        "decorator_count",
        "has_docstring",
        "total_params",
        "annotated_params",
        "return_type",
        "param_types",
        "fully_typed",
        "partial_typed",
        "untyped",
        "typedness_bucket",
        "typedness_source",
        "file_typed_ratio",
        "static_error_count",
        "has_static_errors",
    ]

    result: dict[int, FunctionBaseInfo] = {}
    for row in rows:
        record = dict(zip(columns, row, strict=False))
        goid_int = int(record["function_goid_h128"])
        result[goid_int] = FunctionBaseInfo(
            function_goid_h128=goid_int,
            urn=optional_str(record["urn"]),
            repo=str(record["repo"]),
            commit=str(record["commit"]),
            rel_path=str(record["rel_path"]),
            module=optional_str(record["module"]),
            language=optional_str(record["language"]),
            kind=optional_str(record["kind"]),
            qualname=optional_str(record["qualname"]),
            start_line=optional_int(record["start_line"]),
            end_line=optional_int(record["end_line"]),
            loc=int_or_default(record["loc"]),
            logical_loc=int_or_default(record["logical_loc"]),
            cyclomatic_complexity=int_or_default(record["cyclomatic_complexity"]),
            complexity_bucket=optional_str(record["complexity_bucket"]),
            param_count=int_or_default(record["param_count"]),
            positional_params=int_or_default(record["positional_params"]),
            keyword_params=int_or_default(record["keyword_params"]),
            vararg=bool(record["vararg"]),
            kwarg=bool(record["kwarg"]),
            max_nesting_depth=optional_int(record["max_nesting_depth"]),
            stmt_count=optional_int(record["stmt_count"]),
            decorator_count=optional_int(record["decorator_count"]),
            has_docstring=bool(record["has_docstring"]),
            total_params=int_or_default(record["total_params"]),
            annotated_params=int_or_default(record["annotated_params"]),
            return_type=optional_str(record["return_type"]),
            param_types=record["param_types"],
            fully_typed=bool(record["fully_typed"]),
            partial_typed=bool(record["partial_typed"]),
            untyped=bool(record["untyped"]),
            typedness_bucket=optional_str(record["typedness_bucket"]),
            typedness_source=optional_str(record["typedness_source"]),
            file_typed_ratio=optional_float(record["file_typed_ratio"]),
            static_error_count=int_or_default(record["static_error_count"]),
            has_static_errors=bool(record["has_static_errors"]),
        )
    return result


def join_function_risk(inputs: FunctionProfileInputs) -> Mapping[int, FunctionRiskView]:
    """
    Collect risk scores, levels, and ownership metadata.

    Returns
    -------
    Mapping[int, FunctionRiskView]
        Mapping keyed by function GOID.
    """
    rows = inputs.con.execute(
        """
        SELECT
            function_goid_h128,
            risk_score,
            risk_level,
            hotspot_score,
            tags,
            owners
        FROM analytics.goid_risk_factors
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()
    result: dict[int, FunctionRiskView] = {}
    for (
        function_goid_h128,
        risk_score,
        risk_level,
        hotspot_score,
        tags,
        owners,
    ) in rows:
        goid = int(function_goid_h128)
        result[goid] = FunctionRiskView(
            function_goid_h128=goid,
            risk_score=float(risk_score or 0.0),
            risk_level=str(risk_level) if risk_level is not None else None,
            hotspot_score=float(hotspot_score) if hotspot_score is not None else None,
            tags=tags if tags is not None else "[]",
            owners=owners if owners is not None else "[]",
        )
    return result


def join_function_coverage(inputs: FunctionProfileInputs) -> Mapping[int, CoverageSummary]:
    """
    Aggregate coverage and test metrics per function.

    Returns
    -------
    Mapping[int, CoverageSummary]
        Mapping keyed by function GOID.
    """
    con = inputs.con
    rows = con.execute(
        """
        WITH t_stats AS (
            SELECT
                e.function_goid_h128,
                COUNT(DISTINCT e.test_id) AS tests_touching,
                COUNT(DISTINCT CASE WHEN tc.status IN ('failed', 'error') THEN e.test_id END)
                    AS failing_tests,
                COUNT(DISTINCT CASE WHEN tc.duration_ms > ? THEN e.test_id END) AS slow_tests,
                COUNT(DISTINCT CASE WHEN tc.flaky THEN e.test_id END) AS flaky_tests,
                MODE() WITHIN GROUP (ORDER BY tc.status) AS dominant_test_status
            FROM analytics.test_coverage_edges AS e
            LEFT JOIN analytics.test_catalog AS tc
              ON e.test_id = tc.test_id
             AND e.repo = tc.repo
             AND e.commit = tc.commit
            WHERE e.repo = ? AND e.commit = ?
            GROUP BY e.function_goid_h128
        )
        SELECT
            rf.function_goid_h128,
            rf.executable_lines,
            rf.covered_lines,
            rf.coverage_ratio,
            rf.tested,
            cf.untested_reason,
            COALESCE(t_stats.tests_touching, 0) AS tests_touching,
            COALESCE(t_stats.failing_tests, 0) AS failing_tests,
            COALESCE(t_stats.slow_tests, 0) AS slow_tests,
            COALESCE(t_stats.flaky_tests, 0) AS flaky_tests,
            rf.last_test_status,
            COALESCE(t_stats.dominant_test_status, rf.last_test_status) AS dominant_test_status
        FROM analytics.goid_risk_factors AS rf
        LEFT JOIN analytics.coverage_functions AS cf
          ON rf.function_goid_h128 = cf.function_goid_h128
         AND rf.repo = cf.repo
         AND rf.commit = cf.commit
        LEFT JOIN t_stats
          ON rf.function_goid_h128 = t_stats.function_goid_h128
        WHERE rf.repo = ? AND rf.commit = ?
        """,
        [
            inputs.slow_test_threshold_ms,
            inputs.repo,
            inputs.commit,
            inputs.repo,
            inputs.commit,
        ],
    ).fetchall()
    result: dict[int, CoverageSummary] = {}
    for (
        function_goid_h128,
        executable_lines,
        covered_lines,
        coverage_ratio,
        tested,
        untested_reason,
        tests_touching,
        failing_tests,
        slow_tests,
        flaky_tests,
        last_test_status,
        dominant_test_status,
    ) in rows:
        goid = int(function_goid_h128)
        result[goid] = CoverageSummary(
            function_goid_h128=goid,
            executable_lines=int(executable_lines or 0),
            covered_lines=int(covered_lines or 0),
            coverage_ratio=float(coverage_ratio) if coverage_ratio is not None else None,
            tested=bool(tested),
            untested_reason=str(untested_reason) if untested_reason is not None else None,
            tests_touching=int(tests_touching or 0),
            failing_tests=int(failing_tests or 0),
            slow_tests=int(slow_tests or 0),
            flaky_tests=int(flaky_tests or 0),
            last_test_status=str(last_test_status) if last_test_status is not None else None,
            dominant_test_status=(
                str(dominant_test_status) if dominant_test_status is not None else None
            ),
        )
    return result


def join_function_effects(inputs: FunctionProfileInputs) -> Mapping[int, FunctionEffectsView]:
    """
    Collect effect summaries from analytics.function_effects.

    Returns
    -------
    Mapping[int, FunctionEffectsView]
        Mapping keyed by function GOID.
    """
    rows = inputs.con.execute(
        """
        SELECT
            function_goid_h128,
            is_pure,
            uses_io,
            touches_db,
            uses_time,
            uses_randomness,
            modifies_globals,
            modifies_closure,
            spawns_threads_or_tasks,
            has_transitive_effects,
            purity_confidence
        FROM analytics.function_effects
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()
    result: dict[int, FunctionEffectsView] = {}
    for (
        function_goid_h128,
        is_pure,
        uses_io,
        touches_db,
        uses_time,
        uses_randomness,
        modifies_globals,
        modifies_closure,
        spawns_threads_or_tasks,
        has_transitive_effects,
        purity_confidence,
    ) in rows:
        goid = int(function_goid_h128)
        result[goid] = FunctionEffectsView(
            function_goid_h128=goid,
            is_pure=bool(is_pure),
            uses_io=bool(uses_io),
            touches_db=bool(touches_db),
            uses_time=bool(uses_time),
            uses_randomness=bool(uses_randomness),
            modifies_globals=bool(modifies_globals),
            modifies_closure=bool(modifies_closure),
            spawns_threads_or_tasks=bool(spawns_threads_or_tasks),
            has_transitive_effects=bool(has_transitive_effects),
            purity_confidence=float(purity_confidence) if purity_confidence is not None else None,
        )
    return result


def join_function_contracts(inputs: FunctionProfileInputs) -> Mapping[int, FunctionContractView]:
    """
    Collect contract metadata from analytics.function_contracts.

    Returns
    -------
    Mapping[int, FunctionContractView]
        Mapping keyed by function GOID.
    """
    rows = inputs.con.execute(
        """
        SELECT
            function_goid_h128,
            param_nullability_json,
            return_nullability,
            COALESCE(json_array_length(preconditions_json), 0) > 0 AS has_preconditions,
            COALESCE(json_array_length(postconditions_json), 0) > 0 AS has_postconditions,
            COALESCE(json_array_length(raises_json), 0) > 0 AS has_raises,
            contract_confidence
        FROM analytics.function_contracts
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()
    result: dict[int, FunctionContractView] = {}
    for (
        function_goid_h128,
        param_nullability_json,
        return_nullability,
        has_preconditions,
        has_postconditions,
        has_raises,
        contract_confidence,
    ) in rows:
        goid = int(function_goid_h128)
        result[goid] = FunctionContractView(
            function_goid_h128=goid,
            param_nullability_json=param_nullability_json,
            return_nullability=str(return_nullability) if return_nullability is not None else None,
            has_preconditions=bool(has_preconditions),
            has_postconditions=bool(has_postconditions),
            has_raises=bool(has_raises),
            contract_confidence=float(contract_confidence)
            if contract_confidence is not None
            else None,
        )
    return result


def join_function_roles(inputs: FunctionProfileInputs) -> Mapping[int, FunctionRoleView]:
    """
    Collect semantic roles per function.

    Returns
    -------
    Mapping[int, FunctionRoleView]
        Mapping keyed by function GOID.
    """
    rows = inputs.con.execute(
        """
        SELECT
            function_goid_h128,
            role,
            framework,
            role_confidence,
            role_sources_json
        FROM analytics.semantic_roles_functions
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()
    result: dict[int, FunctionRoleView] = {}
    for function_goid_h128, role, framework, role_confidence, role_sources_json in rows:
        goid = int(function_goid_h128)
        result[goid] = FunctionRoleView(
            function_goid_h128=goid,
            role=str(role) if role is not None else None,
            framework=str(framework) if framework is not None else None,
            role_confidence=float(role_confidence) if role_confidence is not None else None,
            role_sources_json=role_sources_json if role_sources_json is not None else "[]",
        )
    return result


def join_function_docs(inputs: FunctionProfileInputs) -> Mapping[int, FunctionDocView]:
    """
    Collect docstring surfaces per function.

    Returns
    -------
    Mapping[int, FunctionDocView]
        Mapping keyed by function GOID.
    """
    rows = inputs.con.execute(
        """
        SELECT
            rf.function_goid_h128,
            doc.short_desc AS doc_short,
            doc.long_desc AS doc_long,
            doc.params AS doc_params,
            doc.returns AS doc_returns
        FROM analytics.goid_risk_factors AS rf
        LEFT JOIN core.docstrings AS doc
          ON doc.repo = rf.repo
         AND doc.commit = rf.commit
         AND doc.rel_path = rf.rel_path
         AND doc.qualname = rf.qualname
         AND doc.kind = rf.kind
        WHERE rf.repo = ? AND rf.commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()
    result: dict[int, FunctionDocView] = {}
    for function_goid_h128, doc_short, doc_long, doc_params, doc_returns in rows:
        goid = int(function_goid_h128)
        result[goid] = FunctionDocView(
            function_goid_h128=goid,
            doc_short=str(doc_short) if doc_short is not None else None,
            doc_long=str(doc_long) if doc_long is not None else None,
            doc_params=doc_params,
            doc_returns=doc_returns,
        )
    return result


def join_function_history(inputs: FunctionProfileInputs) -> Mapping[int, FunctionHistoryView]:
    """
    Collect function history records.

    Returns
    -------
    Mapping[int, FunctionHistoryView]
        Mapping keyed by function GOID.
    """
    rows = inputs.con.execute(
        """
        SELECT
            function_goid_h128,
            created_in_commit,
            created_at,
            last_modified_commit,
            last_modified_at,
            age_days,
            commit_count,
            author_count,
            lines_added,
            lines_deleted,
            churn_score,
            stability_bucket
        FROM analytics.function_history
        WHERE repo = ? AND commit = ?
        """,
        [inputs.repo, inputs.commit],
    ).fetchall()
    result: dict[int, FunctionHistoryView] = {}
    for (
        function_goid_h128,
        created_in_commit,
        created_at_history,
        last_modified_commit,
        last_modified_at,
        age_days,
        commit_count,
        author_count,
        lines_added,
        lines_deleted,
        churn_score,
        stability_bucket,
    ) in rows:
        goid = int(function_goid_h128)
        result[goid] = FunctionHistoryView(
            function_goid_h128=goid,
            created_in_commit=str(created_in_commit) if created_in_commit is not None else None,
            created_at_history=created_at_history,
            last_modified_commit=str(last_modified_commit)
            if last_modified_commit is not None
            else None,
            last_modified_at=last_modified_at,
            age_days=int(age_days) if age_days is not None else None,
            commit_count=int(commit_count or 0),
            author_count=int(author_count or 0),
            lines_added=int(lines_added or 0),
            lines_deleted=int(lines_deleted or 0),
            churn_score=float(churn_score) if churn_score is not None else None,
            stability_bucket=str(stability_bucket) if stability_bucket is not None else None,
        )
    return result


def build_function_profile_rows(
    inputs: FunctionProfileInputs,
    views: FunctionProfileViews,
) -> Iterable[FunctionProfileRowModel]:
    """
    Assemble FunctionProfileRowModel values from per-concern mappings.

    Yields
    ------
    FunctionProfileRowModel
        Row models ready for insertion into ``analytics.function_profile``.
    """
    for goid, base in views.base_by_func.items():
        risk = views.risk_by_func.get(goid)
        coverage = views.coverage_by_func.get(goid)
        graph = views.graph_by_func.get(goid)
        effects = views.effects_by_func.get(goid)
        contract = views.contracts_by_func.get(goid)
        role = views.roles_by_func.get(goid)
        doc = views.docs_by_func.get(goid)
        history = views.history_by_func.get(goid)

        coverage_ratio = coverage.coverage_ratio if coverage is not None else None
        risk_component_coverage = (
            (1.0 - coverage_ratio) * 0.4 if coverage_ratio is not None else 0.4
        )
        risk_component_complexity = 0.0
        if base.complexity_bucket == "high":
            risk_component_complexity = 0.4
        elif base.complexity_bucket == "medium":
            risk_component_complexity = 0.2
        risk_component_static = 0.2 if base.has_static_errors else 0.0
        risk_component_hotspot = (
            0.1
            if risk is not None
            and risk.hotspot_score is not None
            and float(risk.hotspot_score) > 0.0
            else 0.0
        )

        row: FunctionProfileRowModel = {
            "function_goid_h128": goid,
            "urn": base.urn,
            "repo": base.repo,
            "commit": base.commit,
            "rel_path": base.rel_path,
            "module": base.module,
            "language": base.language,
            "kind": base.kind,
            "qualname": base.qualname,
            "start_line": base.start_line,
            "end_line": base.end_line,
            "loc": base.loc,
            "logical_loc": base.logical_loc,
            "cyclomatic_complexity": base.cyclomatic_complexity,
            "complexity_bucket": base.complexity_bucket,
            "param_count": base.param_count,
            "positional_params": base.positional_params,
            "keyword_params": base.keyword_params,
            "vararg": base.vararg,
            "kwarg": base.kwarg,
            "max_nesting_depth": base.max_nesting_depth,
            "stmt_count": base.stmt_count,
            "decorator_count": base.decorator_count,
            "has_docstring": base.has_docstring,
            "total_params": base.total_params,
            "annotated_params": base.annotated_params,
            "return_type": base.return_type,
            "param_types": base.param_types,
            "fully_typed": base.fully_typed,
            "partial_typed": base.partial_typed,
            "untyped": base.untyped,
            "typedness_bucket": base.typedness_bucket,
            "typedness_source": base.typedness_source,
            "file_typed_ratio": base.file_typed_ratio,
            "static_error_count": base.static_error_count,
            "has_static_errors": base.has_static_errors,
            "executable_lines": coverage.executable_lines if coverage is not None else 0,
            "covered_lines": coverage.covered_lines if coverage is not None else 0,
            "coverage_ratio": coverage_ratio,
            "tested": coverage.tested if coverage is not None else False,
            "untested_reason": coverage.untested_reason if coverage is not None else None,
            "tests_touching": coverage.tests_touching if coverage is not None else 0,
            "failing_tests": coverage.failing_tests if coverage is not None else 0,
            "slow_tests": coverage.slow_tests if coverage is not None else 0,
            "flaky_tests": coverage.flaky_tests if coverage is not None else 0,
            "last_test_status": coverage.last_test_status if coverage is not None else None,
            "dominant_test_status": (
                coverage.dominant_test_status if coverage is not None else None
            ),
            "slow_test_threshold_ms": inputs.slow_test_threshold_ms,
            "created_in_commit": history.created_in_commit if history is not None else None,
            "created_at_history": history.created_at_history if history is not None else None,
            "last_modified_commit": (history.last_modified_commit if history is not None else None),
            "last_modified_at": history.last_modified_at if history is not None else None,
            "age_days": history.age_days if history is not None else None,
            "commit_count": history.commit_count if history is not None else 0,
            "author_count": history.author_count if history is not None else 0,
            "lines_added": history.lines_added if history is not None else 0,
            "lines_deleted": history.lines_deleted if history is not None else 0,
            "churn_score": history.churn_score if history is not None else 0.0,
            "stability_bucket": history.stability_bucket if history is not None else "unknown",
            "call_fan_in": graph.call_fan_in if graph is not None else 0,
            "call_fan_out": graph.call_fan_out if graph is not None else 0,
            "call_edge_in_count": graph.call_edge_in_count if graph is not None else 0,
            "call_edge_out_count": graph.call_edge_out_count if graph is not None else 0,
            "call_is_leaf": graph.call_is_leaf if graph is not None else False,
            "call_is_entrypoint": graph.call_is_entrypoint if graph is not None else False,
            "call_is_public": graph.call_is_public if graph is not None else False,
            "risk_score": risk.risk_score if risk is not None else 0.0,
            "risk_level": risk.risk_level if risk is not None else None,
            "risk_component_coverage": risk_component_coverage,
            "risk_component_complexity": risk_component_complexity,
            "risk_component_static": risk_component_static,
            "risk_component_hotspot": risk_component_hotspot,
            "is_pure": effects.is_pure if effects is not None else False,
            "uses_io": effects.uses_io if effects is not None else False,
            "touches_db": effects.touches_db if effects is not None else False,
            "uses_time": effects.uses_time if effects is not None else False,
            "uses_randomness": effects.uses_randomness if effects is not None else False,
            "modifies_globals": effects.modifies_globals if effects is not None else False,
            "modifies_closure": effects.modifies_closure if effects is not None else False,
            "spawns_threads_or_tasks": (
                effects.spawns_threads_or_tasks if effects is not None else False
            ),
            "has_transitive_effects": (
                effects.has_transitive_effects if effects is not None else False
            ),
            "purity_confidence": effects.purity_confidence if effects is not None else None,
            "param_nullability_json": (
                contract.param_nullability_json if contract is not None else None
            ),
            "return_nullability": contract.return_nullability if contract is not None else None,
            "has_preconditions": contract.has_preconditions if contract is not None else False,
            "has_postconditions": contract.has_postconditions if contract is not None else False,
            "has_raises": contract.has_raises if contract is not None else False,
            "contract_confidence": (contract.contract_confidence if contract is not None else None),
            "role": role.role if role is not None else None,
            "framework": role.framework if role is not None else None,
            "role_confidence": role.role_confidence if role is not None else None,
            "role_sources_json": role.role_sources_json if role is not None else None,
            "tags": risk.tags if risk is not None else "[]",
            "owners": risk.owners if risk is not None else "[]",
            "doc_short": doc.doc_short if doc is not None else None,
            "doc_long": doc.doc_long if doc is not None else None,
            "doc_params": doc.doc_params if doc is not None else None,
            "doc_returns": doc.doc_returns if doc is not None else None,
            "created_at": inputs.created_at,
        }

        yield row


def write_function_profile_rows(
    gateway: StorageGateway,
    rows: Iterable[FunctionProfileRowModel],
) -> int:
    """
    Insert rows into analytics.function_profile.

    Returns
    -------
    int
        Number of rows inserted.
    """
    rows = list(rows)
    if not rows:
        return 0

    repo = rows[0]["repo"]
    commit = rows[0]["commit"]
    con = gateway.con
    ensure_schema(con, "analytics.function_profile")
    con.execute(
        "DELETE FROM analytics.function_profile WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    tuples = [function_profile_row_to_tuple(row) for row in rows]
    column_list = ",\n            ".join(FUNCTION_PROFILE_COLUMNS)
    placeholders = ", ".join("?" for _ in FUNCTION_PROFILE_COLUMNS)
    con.executemany(
        f"""
        INSERT INTO analytics.function_profile (
            {column_list}
        ) VALUES ({placeholders})
        """,
        tuples,
    )
    return len(tuples)


def build_function_profile_recipe(
    gateway: StorageGateway,
    cfg: ProfilesAnalyticsStepConfig,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """
    Compute and persist analytics.function_profile rows.

    Returns
    -------
    int
        Number of rows inserted.
    """
    inputs = compute_function_profile_inputs(gateway, cfg)
    views = FunctionProfileViews(
        base_by_func=load_function_base_info(inputs, module_table=module_table),
        risk_by_func=join_function_risk(inputs),
        coverage_by_func=join_function_coverage(inputs),
        graph_by_func=summarize_graph_for_function_profile(inputs),
        effects_by_func=join_function_effects(inputs),
        contracts_by_func=join_function_contracts(inputs),
        roles_by_func=join_function_roles(inputs),
        docs_by_func=join_function_docs(inputs),
        history_by_func=join_function_history(inputs),
    )
    rows = build_function_profile_rows(inputs, views=views)
    return write_function_profile_rows(gateway, rows)
