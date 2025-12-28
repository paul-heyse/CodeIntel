"""Function profile recipe helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.analytics.profiles.types import (
    CoverageSummary,
    FunctionBaseInfo,
    FunctionContractView,
    FunctionDocView,
    FunctionEffectsView,
    FunctionHistoryView,
    FunctionProfileInputs,
    FunctionRiskView,
    FunctionRoleView,
)
from codeintel.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
)
from codeintel.analytics.profiles.writer_guard import (
    PolicyWriterConfig,
    write_rows_via_policy_backend,
)
from codeintel.analytics.utilities.type_coercion import (
    int_or_default,
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.query_results import records_from_relation
from codeintel.storage.snapshot_scoping import maybe_scope_by_repo_commit

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.analytics.profiles.types import FunctionGraphFeatures
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFunctionProfileRow as FunctionProfileRowModel,
    )
    from codeintel.storage.duckdb_types import DuckDBRelation
    from codeintel.storage.gateway import StorageGateway

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


@dataclass(frozen=True)
class _FunctionBaseTables:
    """Filtered tables needed to assemble function base info."""

    metrics: DuckDBRelation
    types: DuckDBRelation
    modules: DuckDBRelation
    typedness: DuckDBRelation
    diagnostics: DuckDBRelation


def _load_function_base_tables(
    inputs: FunctionProfileInputs,
    module_table: str,
) -> _FunctionBaseTables | None:
    gw = inputs.gateway
    repo = inputs.repo
    commit = inputs.commit

    try:
        metrics = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.function_metrics"),
            repo=repo,
            commit=commit,
        )
        types = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.function_types"),
            repo=repo,
            commit=commit,
        )
        modules = maybe_scope_by_repo_commit(
            gw.relation_from_table_key(module_table),
            repo=repo,
            commit=commit,
        )
        typedness = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.typedness"),
            repo=repo,
            commit=commit,
        )
        diagnostics = maybe_scope_by_repo_commit(
            gw.relation_from_table_key("analytics.static_diagnostics"),
            repo=repo,
            commit=commit,
        )
    except DuckDBError as exc:
        log.warning("function_profile: failed to access base tables: %s", exc)
        return None

    return _FunctionBaseTables(
        metrics=metrics,
        types=types,
        modules=modules,
        typedness=typedness,
        diagnostics=diagnostics,
    )


def compute_function_profile_inputs(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    slow_test_threshold_ms: float = SLOW_TEST_THRESHOLD_MS,
) -> FunctionProfileInputs:
    """
    Normalize snapshot inputs for function profile computations.

    The returned object is intentionally lightweight; heavy lifting happens in
    downstream helpers.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.
    slow_test_threshold_ms
        Threshold for slow tests in milliseconds.

    Returns
    -------
    FunctionProfileInputs
        Snapshot handle used by downstream helpers.
    """
    return FunctionProfileInputs(
        gateway=gateway,
        con=gateway.con,
        repo=snapshot.repo,
        commit=snapshot.commit,
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
    if module_table not in {DEFAULT_MODULE_TABLE, CATALOG_MODULE_TABLE}:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    tables = _load_function_base_tables(inputs, module_table)
    if tables is None:
        return {}

    try:
        types_rel = tables.types.select(
            "function_goid_h128",
            "repo",
            "commit",
            "total_params",
            "annotated_params",
            "return_type",
            "param_types",
            "fully_typed",
            "partial_typed",
            "untyped",
            "typedness_bucket",
            "typedness_source",
        )
        joined = tables.metrics.join(
            types_rel,
            ["function_goid_h128", "repo", "commit"],
            how="left",
        )
        modules_rel = tables.modules.select(
            "repo",
            "commit",
            "path as rel_path",
            "module",
        )
        joined = joined.join(modules_rel, ["repo", "commit", "rel_path"], how="left")
        typedness_rel = tables.typedness.select(
            "repo",
            "commit",
            "path as rel_path",
            "annotation_ratio",
        )
        joined = joined.join(typedness_rel, ["repo", "commit", "rel_path"], how="left")
        joined = joined.join(tables.diagnostics, ["repo", "commit", "rel_path"], how="left")
        selected = joined.select(
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
            "keyword_only_params as keyword_params",
            "has_varargs as vararg",
            "has_varkw as kwarg",
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
            "cast(json_extract(annotation_ratio, '$.params') as double) as file_typed_ratio",
            "total_errors as static_error_count",
            "has_errors as has_static_errors",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load base info: %s", exc)
        return {}

    result: dict[int, FunctionBaseInfo] = {}
    for record in records:
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
    try:
        fm = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.function_metrics"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        rf = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.goid_risk_factors"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        modules = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key(DEFAULT_MODULE_TABLE),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        hotspots = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.hotspots"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        joined = fm.join(
            rf,
            ["function_goid_h128", "repo", "commit"],
            how="left",
        )
        modules_rel = modules.select(
            "repo",
            "commit",
            "path as rel_path",
            "tags",
            "owners",
        )
        joined = joined.join(modules_rel, ["repo", "commit", "rel_path"], how="left")
        hotspots_rel = hotspots.select(
            "rel_path",
            "score as hotspot_score",
        )
        joined = joined.join(hotspots_rel, ["rel_path"], how="left")
        selected = joined.select(
            "function_goid_h128",
            "risk_score",
            "risk_level",
            "hotspot_score",
            "tags",
            "owners",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load risk factors: %s", exc)
        return {}
    result: dict[int, FunctionRiskView] = {}
    for record in records:
        goid = int(record["function_goid_h128"])
        result[goid] = FunctionRiskView(
            function_goid_h128=goid,
            risk_score=float(record["risk_score"] or 0.0),
            risk_level=optional_str(record["risk_level"]),
            hotspot_score=(
                float(record["hotspot_score"]) if record["hotspot_score"] is not None else None
            ),
            tags=record["tags"] if record["tags"] is not None else "[]",
            owners=record["owners"] if record["owners"] is not None else "[]",
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
    repo = inputs.repo
    commit = inputs.commit

    try:
        edges, catalog = _load_test_tables(inputs.gateway, repo, commit)
        t_stats, status_top = _compute_test_stats(
            edges, catalog, slow_threshold_ms=inputs.slow_test_threshold_ms
        )

        cf = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.coverage_functions"),
            repo=repo,
            commit=commit,
        )
        joined = (
            cf.join(t_stats, ["function_goid_h128"], how="left")
            .join(status_top, ["function_goid_h128"], how="left")
        )
        selected = joined.select(
            "function_goid_h128",
            "executable_lines",
            "covered_lines",
            "coverage_ratio",
            "tested",
            "untested_reason",
            "coalesce(tests_touching, 0) as tests_touching",
            "coalesce(failing_tests, 0) as failing_tests",
            "coalesce(slow_tests, 0) as slow_tests",
            "coalesce(flaky_tests, 0) as flaky_tests",
            "coalesce(dominant_test_status, 'untested') as last_test_status",
            "dominant_test_status",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load coverage: %s", exc)
        return {}
    result: dict[int, CoverageSummary] = {}
    for record in records:
        goid = int(record["function_goid_h128"])
        result[goid] = CoverageSummary(
            function_goid_h128=goid,
            executable_lines=int(record["executable_lines"] or 0),
            covered_lines=int(record["covered_lines"] or 0),
            coverage_ratio=(
                float(record["coverage_ratio"]) if record["coverage_ratio"] is not None else None
            ),
            tested=bool(record["tested"]),
            untested_reason=optional_str(record["untested_reason"]),
            tests_touching=int(record["tests_touching"] or 0),
            failing_tests=int(record["failing_tests"] or 0),
            slow_tests=int(record["slow_tests"] or 0),
            flaky_tests=int(record["flaky_tests"] or 0),
            last_test_status=optional_str(record["last_test_status"]),
            dominant_test_status=(
                optional_str(record["dominant_test_status"])
            ),
        )
    return result


def _load_test_tables(
    gateway: StorageGateway, repo: str, commit: str
) -> tuple[DuckDBRelation, DuckDBRelation]:
    """Load coverage edge and catalog tables with repo/commit filtering.

    Returns
    -------
    tuple[DuckDBRelation, DuckDBRelation]
        Filtered coverage edge table and catalog table.
    """
    edges = maybe_scope_by_repo_commit(
        gateway.relation_from_table_key("analytics.test_coverage_edges"),
        repo=repo,
        commit=commit,
    )
    catalog = maybe_scope_by_repo_commit(
        gateway.relation_from_table_key("analytics.test_catalog"),
        repo=repo,
        commit=commit,
    )
    return edges, catalog


def _compute_test_stats(
    edges: DuckDBRelation,
    catalog: DuckDBRelation,
    *,
    slow_threshold_ms: float,
) -> tuple[DuckDBRelation, DuckDBRelation]:
    """Compute per-function test stats and dominant status.

    Returns
    -------
    tuple[DuckDBRelation, DuckDBRelation]
        Aggregated coverage stats and status summaries.
    """
    joined = edges.join(
        catalog,
        ["test_id", "repo", "commit"],
        how="left",
    )

    slow_threshold = float(slow_threshold_ms)
    t_stats = joined.group_by("function_goid_h128").aggregate(
        [
            "count(distinct test_id) as tests_touching",
            (
                "count(distinct case when status in ('failed', 'error') "
                "then test_id end) as failing_tests"
            ),
            (
                "count(distinct case when duration_ms > "
                f"{slow_threshold} then test_id end) as slow_tests"
            ),
            "count(distinct case when flaky then test_id end) as flaky_tests",
        ]
    )

    status_counts = joined.group_by("function_goid_h128", "status").aggregate(
        "count(test_id) as status_count"
    )
    status_ranked = status_counts.select(
        "*",
        (
            "row_number() over (partition by function_goid_h128 "
            "order by status_count desc, status) as status_rank"
        ),
    )
    status_top = status_ranked.filter("status_rank = 1").select(
        "function_goid_h128",
        "status as dominant_test_status",
    )
    return t_stats, status_top


def join_function_effects(inputs: FunctionProfileInputs) -> Mapping[int, FunctionEffectsView]:
    """
    Collect effect summaries from analytics.function_effects.

    Returns
    -------
    Mapping[int, FunctionEffectsView]
        Mapping keyed by function GOID.
    """
    try:
        effects = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.function_effects"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        selected = effects.select(
            "function_goid_h128",
            "is_pure",
            "uses_io",
            "touches_db",
            "uses_time",
            "uses_randomness",
            "modifies_globals",
            "modifies_closure",
            "spawns_threads_or_tasks",
            "has_transitive_effects",
            "purity_confidence",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load effects: %s", exc)
        return {}
    result: dict[int, FunctionEffectsView] = {}
    for record in records:
        goid = int(record["function_goid_h128"])
        result[goid] = FunctionEffectsView(
            function_goid_h128=goid,
            is_pure=bool(record["is_pure"]),
            uses_io=bool(record["uses_io"]),
            touches_db=bool(record["touches_db"]),
            uses_time=bool(record["uses_time"]),
            uses_randomness=bool(record["uses_randomness"]),
            modifies_globals=bool(record["modifies_globals"]),
            modifies_closure=bool(record["modifies_closure"]),
            spawns_threads_or_tasks=bool(record["spawns_threads_or_tasks"]),
            has_transitive_effects=bool(record["has_transitive_effects"]),
            purity_confidence=(
                float(record["purity_confidence"])
                if record["purity_confidence"] is not None
                else None
            ),
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
    try:
        contracts = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.function_contracts"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        selected = contracts.select(
            "function_goid_h128",
            "param_nullability_json",
            "return_nullability",
            "coalesce(cast(preconditions_json as varchar), '') <> '' as has_preconditions",
            "coalesce(cast(postconditions_json as varchar), '') <> '' as has_postconditions",
            "coalesce(cast(raises_json as varchar), '') <> '' as has_raises",
            "contract_confidence",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load contracts: %s", exc)
        return {}
    result: dict[int, FunctionContractView] = {}
    for record in records:
        goid = int(record["function_goid_h128"])
        result[goid] = FunctionContractView(
            function_goid_h128=goid,
            param_nullability_json=record["param_nullability_json"],
            return_nullability=optional_str(record["return_nullability"]),
            has_preconditions=bool(record["has_preconditions"]),
            has_postconditions=bool(record["has_postconditions"]),
            has_raises=bool(record["has_raises"]),
            contract_confidence=(
                float(record["contract_confidence"])
                if record["contract_confidence"] is not None
                else None
            ),
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
    try:
        roles = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.semantic_roles_functions"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        selected = roles.select(
            "function_goid_h128",
            "role",
            "framework",
            "role_confidence",
            "role_sources_json",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load roles: %s", exc)
        return {}
    result: dict[int, FunctionRoleView] = {}
    for record in records:
        goid = int(record["function_goid_h128"])
        result[goid] = FunctionRoleView(
            function_goid_h128=goid,
            role=optional_str(record["role"]),
            framework=optional_str(record["framework"]),
            role_confidence=(
                float(record["role_confidence"])
                if record["role_confidence"] is not None
                else None
            ),
            role_sources_json=(
                record["role_sources_json"] if record["role_sources_json"] is not None else "[]"
            ),
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
    try:
        fm = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.function_metrics"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        docs = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("core.docstrings"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        joined = fm.join(
            docs,
            ["repo", "commit", "rel_path", "qualname", "kind"],
            how="left",
        )
        selected = joined.select(
            "function_goid_h128",
            "short_desc as doc_short",
            "long_desc as doc_long",
            "params as doc_params",
            "returns as doc_returns",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load docs: %s", exc)
        return {}
    result: dict[int, FunctionDocView] = {}
    for record in records:
        goid = int(record["function_goid_h128"])
        result[goid] = FunctionDocView(
            function_goid_h128=goid,
            doc_short=optional_str(record["doc_short"]),
            doc_long=optional_str(record["doc_long"]),
            doc_params=record["doc_params"],
            doc_returns=record["doc_returns"],
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
    try:
        history = maybe_scope_by_repo_commit(
            inputs.gateway.relation_from_table_key("analytics.function_history"),
            repo=inputs.repo,
            commit=inputs.commit,
        )
        selected = history.select(
            "function_goid_h128",
            "created_in_commit",
            "created_at",
            "last_modified_commit",
            "last_modified_at",
            "age_days",
            "commit_count",
            "author_count",
            "lines_added",
            "lines_deleted",
            "churn_score",
            "stability_bucket",
        )
        records = records_from_relation(selected)
    except DuckDBError as exc:
        log.warning("function_profile: failed to load history: %s", exc)
        return {}
    result: dict[int, FunctionHistoryView] = {}
    for record in records:
        goid = int(record["function_goid_h128"])
        result[goid] = FunctionHistoryView(
            function_goid_h128=goid,
            created_in_commit=optional_str(record["created_in_commit"]),
            created_at_history=record["created_at"],
            last_modified_commit=optional_str(record["last_modified_commit"]),
            last_modified_at=record["last_modified_at"],
            age_days=optional_int(record["age_days"]),
            commit_count=int(record["commit_count"] or 0),
            author_count=int(record["author_count"] or 0),
            lines_added=int(record["lines_added"] or 0),
            lines_deleted=int(record["lines_deleted"] or 0),
            churn_score=(
                float(record["churn_score"]) if record["churn_score"] is not None else None
            ),
            stability_bucket=optional_str(record["stability_bucket"]),
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


def build_function_profile_recipe(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> int:
    """Compute and persist analytics.function_profile rows.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository and commit identifiers.
    module_table
        Name of the module table to use.

    Returns
    -------
    int
        Number of rows inserted.
    """
    inputs = compute_function_profile_inputs(gateway, snapshot)
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
    rows = list(build_function_profile_rows(inputs, views=views))
    if not rows:
        return 0

    config = PolicyWriterConfig(
        table_key="analytics.function_profile",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return write_rows_via_policy_backend(gateway, rows=rows, config=config)
