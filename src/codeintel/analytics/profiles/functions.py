"""Function profile recipe helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import ibis

from codeintel.analytics.compute.ibis_utils import zero_if_null
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
from codeintel.core.ibis_typing import (
    and_predicates,
    cast_dtype,
    col_count,
    col_nunique,
    filter_by,
    get_column,
    gt,
    ibis_bool,
    isin_values,
    ne,
    window_over,
)
from codeintel.storage.gateway import DuckDBError, ibis_facade

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import ibis.expr.types as it

    from codeintel.analytics.profiles.types import (
        FunctionGraphFeatures,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFunctionProfileRow as FunctionProfileRowModel,
    )
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

    metrics: it.Table
    types: it.Table
    modules: it.Table
    typedness: it.Table
    diagnostics: it.Table


def _load_function_base_tables(
    inputs: FunctionProfileInputs,
    module_table: str,
) -> _FunctionBaseTables | None:
    gw = inputs.gateway
    repo = inputs.repo
    commit = inputs.commit

    try:
        metrics_table = ibis_facade.table(gw, "analytics.function_metrics")
        metrics = filter_by(
            metrics_table,
            metrics_table.repo == repo,
            metrics_table.commit == commit,
        )
        types_table = ibis_facade.table(gw, "analytics.function_types")
        types = filter_by(
            types_table,
            types_table.repo == repo,
            types_table.commit == commit,
        )
        modules = ibis_facade.table(gw, module_table)
        typedness_table = ibis_facade.table(gw, "analytics.typedness")
        typedness = filter_by(
            typedness_table,
            typedness_table.repo == repo,
            typedness_table.commit == commit,
        )
        diagnostics_table = ibis_facade.table(gw, "analytics.static_diagnostics")
        diagnostics = filter_by(
            diagnostics_table,
            diagnostics_table.repo == repo,
            diagnostics_table.commit == commit,
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

    joined = (
        tables.metrics.left_join(
            tables.types,
            predicates=[
                (tables.metrics.function_goid_h128, tables.types.function_goid_h128),
                (tables.metrics.repo, tables.types.repo),
                (tables.metrics.commit, tables.types.commit),
            ],
        )
        .left_join(
            tables.modules,
            predicates=[
                and_predicates(
                    tables.modules.path == tables.metrics.rel_path,
                    (tables.modules.repo.isnull()) | (tables.modules.repo == tables.metrics.repo),
                    (tables.modules.commit.isnull())
                    | (tables.modules.commit == tables.metrics.commit),
                )
            ],
        )
        .left_join(
            tables.typedness,
            predicates=[
                (tables.typedness.path, tables.metrics.rel_path),
                (tables.typedness.repo, tables.metrics.repo),
                (tables.typedness.commit, tables.metrics.commit),
            ],
        )
        .left_join(
            tables.diagnostics,
            predicates=[
                (tables.diagnostics.rel_path, tables.metrics.rel_path),
                (tables.diagnostics.repo, tables.metrics.repo),
                (tables.diagnostics.commit, tables.metrics.commit),
            ],
        )
    )

    try:
        df = joined.select(
            tables.metrics.function_goid_h128,
            tables.metrics.urn,
            tables.metrics.repo,
            tables.metrics.commit,
            tables.metrics.rel_path,
            tables.modules.module,
            tables.metrics.language,
            tables.metrics.kind,
            tables.metrics.qualname,
            tables.metrics.start_line,
            tables.metrics.end_line,
            tables.metrics.loc,
            tables.metrics.logical_loc,
            tables.metrics.cyclomatic_complexity,
            tables.metrics.complexity_bucket,
            tables.metrics.param_count,
            tables.metrics.positional_params,
            tables.metrics.keyword_only_params.name("keyword_params"),
            tables.metrics.has_varargs.name("vararg"),
            tables.metrics.has_varkw.name("kwarg"),
            tables.metrics.max_nesting_depth,
            tables.metrics.stmt_count,
            tables.metrics.decorator_count,
            tables.metrics.has_docstring,
            tables.types.total_params,
            tables.types.annotated_params,
            tables.types.return_type,
            tables.types.param_types,
            tables.types.fully_typed,
            tables.types.partial_typed,
            tables.types.untyped,
            tables.types.typedness_bucket,
            tables.types.typedness_source,
            tables.typedness.annotation_ratio.name("file_typed_ratio"),
            tables.diagnostics.total_errors.name("static_error_count"),
            tables.diagnostics.has_errors.name("has_static_errors"),
        ).execute()
    except DuckDBError as exc:
        log.warning("function_profile: failed to load base info: %s", exc)
        return {}

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
    for row in df.itertuples(index=False, name=None):
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
    try:
        fm_table = ibis_facade.table(inputs.gateway, "analytics.function_metrics")
        fm = filter_by(fm_table, fm_table.repo == inputs.repo, fm_table.commit == inputs.commit)
        rf_table = ibis_facade.table(inputs.gateway, "analytics.goid_risk_factors")
        rf = filter_by(rf_table, rf_table.repo == inputs.repo, rf_table.commit == inputs.commit)
        modules = ibis_facade.table(inputs.gateway, DEFAULT_MODULE_TABLE)
        hotspots = ibis_facade.table(inputs.gateway, "analytics.hotspots")
        df = (
            fm.left_join(
                rf,
                predicates=[
                    (fm.function_goid_h128, rf.function_goid_h128),
                    (fm.repo, rf.repo),
                    (fm.commit, rf.commit),
                ],
            )
            .left_join(
                modules,
                predicates=[
                    and_predicates(
                        modules.path == fm.rel_path,
                        (modules.repo.isnull()) | (modules.repo == fm.repo),
                        (modules.commit.isnull()) | (modules.commit == fm.commit),
                    )
                ],
            )
            .left_join(
                hotspots,
                predicates=[(hotspots.rel_path, fm.rel_path)],
            )
            .select(
                fm.function_goid_h128,
                rf.risk_score,
                rf.risk_level,
                hotspots.score.name("hotspot_score"),
                modules.tags,
                modules.owners,
            )
            .execute()
        )
    except DuckDBError as exc:
        log.warning("function_profile: failed to load risk factors: %s", exc)
        return {}
    result: dict[int, FunctionRiskView] = {}
    for function_goid_h128, risk_score, risk_level, hotspot_score, tags, owners in df.itertuples(
        index=False, name=None
    ):
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
    repo = inputs.repo
    commit = inputs.commit

    try:
        edges, catalog = _load_test_tables(inputs.gateway, repo, commit)
        t_stats, status_top = _compute_test_stats(
            edges, catalog, slow_threshold_ms=inputs.slow_test_threshold_ms
        )

        cf = filter_by(
            ibis_facade.table(inputs.gateway, "analytics.coverage_functions"),
            ibis_facade.table(inputs.gateway, "analytics.coverage_functions").repo == repo,
            ibis_facade.table(inputs.gateway, "analytics.coverage_functions").commit == commit,
        )

        df = (
            cf.left_join(
                t_stats,
                predicates=[(cf.function_goid_h128, t_stats.function_goid_h128)],
            )
            .left_join(
                status_top,
                predicates=[(cf.function_goid_h128, status_top.function_goid_h128)],
            )
            .select(
                cf.function_goid_h128,
                cf.executable_lines,
                cf.covered_lines,
                cf.coverage_ratio,
                cf.tested,
                cf.untested_reason,
                zero_if_null(t_stats.tests_touching).name("tests_touching"),
                zero_if_null(t_stats.failing_tests).name("failing_tests"),
                zero_if_null(t_stats.slow_tests).name("slow_tests"),
                zero_if_null(t_stats.flaky_tests).name("flaky_tests"),
                ibis.coalesce(
                    status_top.dominant_test_status,
                    ibis.literal("untested"),
                ).name("last_test_status"),
                status_top.dominant_test_status.name("dominant_test_status"),
            )
            .execute()
        )
    except DuckDBError as exc:
        log.warning("function_profile: failed to load coverage: %s", exc)
        return {}
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
    ) in df.itertuples(index=False, name=None):
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


def _load_test_tables(gateway: StorageGateway, repo: str, commit: str) -> tuple[it.Table, it.Table]:
    """Load coverage edge and catalog tables with repo/commit filtering.

    Returns
    -------
    tuple[it.Table, it.Table]
        Filtered coverage edge table and catalog table.
    """
    edges_table = ibis_facade.table(gateway, "analytics.test_coverage_edges")
    edges = filter_by(edges_table, edges_table.repo == repo, edges_table.commit == commit)

    catalog_table = ibis_facade.table(gateway, "analytics.test_catalog")
    catalog = filter_by(
        catalog_table,
        catalog_table.repo == repo,
        catalog_table.commit == commit,
    )
    return edges, catalog


def _compute_test_stats(
    edges: it.Table,
    catalog: it.Table,
    *,
    slow_threshold_ms: float,
) -> tuple[it.Table, it.Table]:
    """Compute per-function test stats and dominant status.

    Returns
    -------
    tuple[it.Table, it.Table]
        Aggregated coverage stats and status summaries.
    """
    failing_predicate = isin_values(catalog.status, ["failed", "error"])
    slow_predicate = gt(cast("it.Value", catalog.duration_ms), slow_threshold_ms)
    flaky_predicate = ibis_bool(catalog.flaky)

    joined = edges.left_join(
        catalog,
        predicates=[
            (edges.test_id, catalog.test_id),
            (edges.repo, catalog.repo),
            (edges.commit, catalog.commit),
        ],
    )

    t_stats = joined.group_by(joined.function_goid_h128).aggregate(
        tests_touching=col_nunique(joined.test_id),
        failing_tests=col_nunique(ibis.ifelse(failing_predicate, joined.test_id, ibis.null())),
        slow_tests=col_nunique(ibis.ifelse(slow_predicate, joined.test_id, ibis.null())),
        flaky_tests=col_nunique(ibis.ifelse(flaky_predicate, joined.test_id, ibis.null())),
    )

    status_counts = joined.group_by(joined.function_goid_h128, catalog.status).aggregate(
        status_count=col_count(joined.test_id)
    )
    status_window = window_over(
        partition_by=[status_counts.function_goid_h128],
        order_by=[ibis.desc(status_counts.status_count), status_counts.status],
    )
    status_ranked = status_counts.mutate(rank=ibis.row_number().over(status_window))
    status_top = filter_by(status_ranked, status_ranked.rank == 0).select(
        status_counts.function_goid_h128, status_counts.status.name("dominant_test_status")
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
        effects_table = ibis_facade.table(inputs.gateway, "analytics.function_effects")
        effects = filter_by(
            effects_table,
            effects_table.repo == inputs.repo,
            effects_table.commit == inputs.commit,
        )
        df = effects.select(
            effects.function_goid_h128,
            effects.is_pure,
            effects.uses_io,
            effects.touches_db,
            effects.uses_time,
            effects.uses_randomness,
            effects.modifies_globals,
            effects.modifies_closure,
            effects.spawns_threads_or_tasks,
            effects.has_transitive_effects,
            effects.purity_confidence,
        ).execute()
    except DuckDBError as exc:
        log.warning("function_profile: failed to load effects: %s", exc)
        return {}
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
    ) in df.itertuples(index=False, name=None):
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
    try:
        contracts_table = ibis_facade.table(inputs.gateway, "analytics.function_contracts")
        contracts = filter_by(
            contracts_table,
            contracts_table.repo == inputs.repo,
            contracts_table.commit == inputs.commit,
        )
        preconditions = cast_dtype(get_column(contracts, "preconditions_json"), "string")
        postconditions = cast_dtype(get_column(contracts, "postconditions_json"), "string")
        raises_json = cast_dtype(get_column(contracts, "raises_json"), "string")
        df = contracts.select(
            contracts.function_goid_h128,
            contracts.param_nullability_json,
            contracts.return_nullability,
            ne(preconditions, "").name("has_preconditions"),
            ne(postconditions, "").name("has_postconditions"),
            ne(raises_json, "").name("has_raises"),
            contracts.contract_confidence,
        ).execute()
    except DuckDBError as exc:
        log.warning("function_profile: failed to load contracts: %s", exc)
        return {}
    result: dict[int, FunctionContractView] = {}
    for (
        function_goid_h128,
        param_nullability_json,
        return_nullability,
        has_preconditions,
        has_postconditions,
        has_raises,
        contract_confidence,
    ) in df.itertuples(index=False, name=None):
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
    try:
        roles_table = ibis_facade.table(inputs.gateway, "analytics.semantic_roles_functions")
        roles = filter_by(
            roles_table,
            roles_table.repo == inputs.repo,
            roles_table.commit == inputs.commit,
        )
        df = roles.select(
            roles.function_goid_h128,
            roles.role,
            roles.framework,
            roles.role_confidence,
            roles.role_sources_json,
        ).execute()
    except DuckDBError as exc:
        log.warning("function_profile: failed to load roles: %s", exc)
        return {}
    result: dict[int, FunctionRoleView] = {}
    for function_goid_h128, role, framework, role_confidence, role_sources_json in df.itertuples(
        index=False, name=None
    ):
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
    try:
        fm_table = ibis_facade.table(inputs.gateway, "analytics.function_metrics")
        fm = filter_by(fm_table, fm_table.repo == inputs.repo, fm_table.commit == inputs.commit)
        docs = ibis_facade.table(inputs.gateway, "core.docstrings")
        df = (
            (
                fm.left_join(
                    docs,
                    predicates=[
                        and_predicates(
                            docs.repo == fm.repo,
                            docs.commit == fm.commit,
                            docs.rel_path == fm.rel_path,
                            docs.qualname == fm.qualname,
                            docs.kind == fm.kind,
                        )
                    ],
                )
            )
            .select(
                fm.function_goid_h128,
                docs.short_desc.name("doc_short"),
                docs.long_desc.name("doc_long"),
                docs.params.name("doc_params"),
                docs.returns.name("doc_returns"),
            )
            .execute()
        )
    except DuckDBError as exc:
        log.warning("function_profile: failed to load docs: %s", exc)
        return {}
    result: dict[int, FunctionDocView] = {}
    for function_goid_h128, doc_short, doc_long, doc_params, doc_returns in df.itertuples(
        index=False, name=None
    ):
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
    try:
        history_table = ibis_facade.table(inputs.gateway, "analytics.function_history")
        history = filter_by(
            history_table,
            history_table.repo == inputs.repo,
            history_table.commit == inputs.commit,
        )
        df = history.select(
            history.function_goid_h128,
            history.created_in_commit,
            history.created_at,
            history.last_modified_commit,
            history.last_modified_at,
            history.age_days,
            history.commit_count,
            history.author_count,
            history.lines_added,
            history.lines_deleted,
            history.churn_score,
            history.stability_bucket,
        ).execute()
    except DuckDBError as exc:
        log.warning("function_profile: failed to load history: %s", exc)
        return {}
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
    ) in df.itertuples(index=False, name=None):
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
