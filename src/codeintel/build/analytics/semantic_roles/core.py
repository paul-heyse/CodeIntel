"""Heuristic semantic role classification for functions and modules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import msgspec
import pyarrow as pa

from codeintel.build.analytics.compute.semantic_roles import (
    FunctionContext,
    ModuleRecord,
    RoleArtifacts,
    classify_function_role,
    classify_modules,
)
from codeintel.build.analytics.compute.semantic_roles.classification import decorator_names
from codeintel.build.analytics.utilities.snapshot import (
    SnapshotContext,
    require_columns,
    snapshot_table,
)
from codeintel.core.columnar.conversion import table_to_reader
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
)
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.plan_kernels import GroupedRollupSpec, grouped_rollup_table
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.paths import normalize_path
from codeintel.core.query_results import coerce_optional_int, coerce_optional_str, coerce_str

if TYPE_CHECKING:
    from codeintel.build.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef


@dataclass(frozen=True)
class SemanticRolesResult:
    """Result from semantic roles computation.

    Attributes
    ----------
    function_rows
        Rows for analytics.semantic_roles_functions table.
    module_rows
        Rows for analytics.semantic_roles_modules table.
    """

    function_rows: list[tuple[object, ...]]
    module_rows: list[tuple[object, ...]]


@dataclass(frozen=True)
class SemanticRoleInputs:
    """Inputs required to compute semantic roles."""

    module_by_path: dict[str, str]
    ast_map: dict[int, FunctionAst]
    features_map: dict[int, FunctionAstFeatures]
    goids_frame: pa.Table | None = None
    function_effects_frame: pa.Table | None = None
    function_contracts_frame: pa.Table | None = None
    graph_metrics_frame: pa.Table | None = None
    modules_frame: pa.Table | None = None
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


def build_semantic_roles_rows(
    snapshot: SnapshotRef,
    inputs: SemanticRoleInputs,
) -> SemanticRolesResult:
    """
    Build semantic role rows without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/ArrowDatasetSaver.

    Parameters
    ----------
    snapshot
        Repository and commit identifiers.
    inputs
        Bundled inputs for semantic role computation.

    Returns
    -------
    SemanticRolesResult
        Container with function and module rows.
    """
    module_meta = _module_meta_from_frame(
        inputs.modules_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    function_rows = _function_rows_from_frame(
        inputs.goids_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    effects = _effects_from_frame(
        inputs.function_effects_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    contracts = _contracts_from_frame(
        inputs.function_contracts_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    graph_metrics = _graph_metrics_from_frame(
        inputs.graph_metrics_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )

    artifacts = RoleArtifacts(
        module_by_path=inputs.module_by_path,
        module_meta=module_meta,
        ast_map=inputs.ast_map,
        effects=effects,
        contracts=contracts,
        graph_metrics=graph_metrics,
        features=inputs.features_map,
    )

    now = datetime.now(tz=UTC)
    fn_rows, roles_by_module = _build_function_role_rows(
        function_rows=function_rows,
        artifacts=artifacts,
        repo=snapshot.repo,
        commit=snapshot.commit,
        now=now,
    )

    module_rows = classify_modules(
        module_meta=module_meta,
        roles_by_module=roles_by_module,
        repo=snapshot.repo,
        commit=snapshot.commit,
        now=now,
    )

    return SemanticRolesResult(
        function_rows=fn_rows,
        module_rows=module_rows,
    )


def _build_function_role_rows(
    *,
    function_rows: list[tuple[int, str, str, int | None]],
    artifacts: RoleArtifacts,
    repo: str,
    commit: str,
    now: datetime,
) -> tuple[list[tuple[object, ...]], dict[str, list[tuple[str, float]]]]:
    fn_rows: list[tuple[object, ...]] = []
    roles_by_module: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for goid, rel_path, qualname, loc in function_rows:
        normalized_path = normalize_path(rel_path)
        module = artifacts.module_by_path.get(normalized_path)
        module_record = artifacts.module_meta.get(module or "")
        module_tags: list[str] = module_record.tags if module_record else []

        ast_info = artifacts.ast_map.get(goid)
        decorators = decorator_names(ast_info.node.decorator_list) if ast_info else []

        context = FunctionContext(
            goid=goid,
            rel_path=normalized_path,
            qualname=qualname,
            decorators=decorators,
            effects=artifacts.effects.get(goid, {}),
            contracts=artifacts.contracts.get(goid, {}),
            module_tags=module_tags,
            module_name=module,
            graph=artifacts.graph_metrics.get(goid, {}),
            loc=loc,
            features=artifacts.features.get(goid),
        )

        role, confidence, framework, role_sources = classify_function_role(context)

        fn_rows.append(
            (
                repo,
                commit,
                goid,
                role,
                framework,
                confidence,
                {"role_sources": role_sources},
                now,
            )
        )
        if module:
            roles_by_module[module].append((role, confidence))

    return fn_rows, roles_by_module


def _function_rows_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> list[tuple[int, str, str, int | None]]:
    if frame is None or frame.num_rows == 0:
        return []
    column_names = set(frame.column_names)
    if "function_goid_h128" in column_names:
        goid_column = "function_goid_h128"
    elif "goid_h128" in column_names:
        goid_column = "goid_h128"
    else:
        return []
    table = _function_worklist_table(
        frame,
        repo=repo,
        commit=commit,
        goid_column=goid_column,
        ctx=ctx,
    )
    if table.num_rows == 0:
        return []
    result: list[tuple[int, str, str, int | None]] = []
    columns = [goid_column, "rel_path", "qualname", "start_line", "end_line"]
    for values in iter_tuples(table_to_reader(table), columns=columns):
        goid_raw, rel_path, qualname, start_line_raw, end_line_raw = values
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        start_line = coerce_optional_int(start_line_raw, ctx="core.goids.start_line")
        end_line = coerce_optional_int(end_line_raw, ctx="core.goids.end_line")
        loc = end_line - start_line + 1 if start_line is not None and end_line is not None else None
        result.append(
            (
                goid,
                coerce_str(rel_path, ctx="core.goids.rel_path"),
                coerce_str(qualname, ctx="core.goids.qualname"),
                loc,
            )
        )
    return result


def _function_worklist_table(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
    goid_column: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    required = {goid_column, "rel_path", "qualname", "start_line", "end_line"}
    missing = [name for name in required if name not in frame.column_names]
    if missing:
        msg = f"Missing function worklist columns: {missing}"
        raise ValueError(msg)
    scoped = _scoped_table(
        frame,
        repo=repo,
        commit=commit,
        columns=(goid_column, "rel_path", "qualname", "start_line", "end_line"),
        ctx=ctx,
    )
    if scoped.num_rows == 0:
        return scoped
    return grouped_rollup_table(
        scoped,
        spec=GroupedRollupSpec(
            keys=(goid_column,),
            aggregates=[
                ("rel_path", "min", None, "rel_path"),
                ("qualname", "min", None, "qualname"),
                ("start_line", "min", None, "start_line"),
                ("end_line", "max", None, "end_line"),
            ],
        ),
        ctx=resolve_columnar_context(ctx),
    )


def _effects_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[int, dict[str, object]]:
    if frame is None or frame.num_rows == 0:
        return {}
    scoped = _scoped_table(
        frame,
        repo=repo,
        commit=commit,
        columns=(
            "function_goid_h128",
            "touches_db",
            "uses_io",
            "uses_time",
            "uses_randomness",
            "modifies_globals",
            "modifies_closure",
            "spawns_threads_or_tasks",
        ),
        ctx=ctx,
    )
    if scoped.num_rows == 0:
        return {}
    filtered = grouped_rollup_table(
        scoped,
        spec=GroupedRollupSpec(
            keys=("function_goid_h128",),
            aggregates=[
                ("touches_db", "max", None, "touches_db"),
                ("uses_io", "max", None, "uses_io"),
                ("uses_time", "max", None, "uses_time"),
                ("uses_randomness", "max", None, "uses_randomness"),
                ("modifies_globals", "max", None, "modifies_globals"),
                ("modifies_closure", "max", None, "modifies_closure"),
                ("spawns_threads_or_tasks", "max", None, "spawns_threads_or_tasks"),
            ],
        ),
        ctx=resolve_columnar_context(ctx),
    )
    if filtered.num_rows == 0:
        return {}
    mapping: dict[int, dict[str, object]] = {}
    columns = [
        "function_goid_h128",
        "touches_db",
        "uses_io",
        "uses_time",
        "uses_randomness",
        "modifies_globals",
        "modifies_closure",
        "spawns_threads_or_tasks",
    ]
    for values in iter_tuples(table_to_reader(filtered), columns=columns):
        (
            goid_raw,
            touches_db,
            uses_io,
            uses_time,
            uses_randomness,
            modifies_globals,
            modifies_closure,
            spawns_threads_or_tasks,
        ) = values
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        mapping[goid] = {
            "touches_db": bool(touches_db),
            "uses_io": bool(uses_io),
            "uses_time": bool(uses_time),
            "uses_randomness": bool(uses_randomness),
            "modifies_globals": bool(modifies_globals),
            "modifies_closure": bool(modifies_closure),
            "spawns_threads_or_tasks": bool(spawns_threads_or_tasks),
        }
    return mapping


def _contracts_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[int, dict[str, object]]:
    if frame is None or frame.num_rows == 0:
        return {}
    table = _scoped_table(
        frame,
        repo=repo,
        commit=commit,
        columns=("function_goid_h128", "extras"),
        ctx=ctx,
    )
    if table.num_rows == 0:
        return {}
    mapping: dict[int, dict[str, object]] = {}
    for goid_raw, extras in iter_tuples(
        table_to_reader(table),
        columns=["function_goid_h128", "extras"],
    ):
        if isinstance(extras, Mapping):
            preconditions = extras.get("preconditions")
            raises = extras.get("raises")
            param_nullability = extras.get("param_nullability")
        else:
            preconditions = None
            raises = None
            param_nullability = None
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        mapping[goid] = {
            "preconditions": _coerce_json(preconditions) or [],
            "raises": _coerce_json(raises) or [],
            "param_nullability": _coerce_json(param_nullability) or {},
        }
    return mapping


def _graph_metrics_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[int, dict[str, int]]:
    if frame is None or frame.num_rows == 0:
        return {}
    scoped = _scoped_table(
        frame,
        repo=repo,
        commit=commit,
        columns=("function_goid_h128", "call_fan_in", "call_fan_out"),
        ctx=ctx,
    )
    if scoped.num_rows == 0:
        return {}
    filtered = grouped_rollup_table(
        scoped,
        spec=GroupedRollupSpec(
            keys=("function_goid_h128",),
            aggregates=[
                ("call_fan_in", "max", None, "call_fan_in"),
                ("call_fan_out", "max", None, "call_fan_out"),
            ],
        ),
        ctx=resolve_columnar_context(ctx),
    )
    if filtered.num_rows == 0:
        return {}
    mapping: dict[int, dict[str, int]] = {}
    for goid_raw, call_fan_in, call_fan_out in iter_tuples(
        table_to_reader(filtered),
        columns=["function_goid_h128", "call_fan_in", "call_fan_out"],
    ):
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        mapping[goid] = {
            "call_fan_in": coerce_optional_int(call_fan_in, ctx="call_fan_in") or 0,
            "call_fan_out": coerce_optional_int(call_fan_out, ctx="call_fan_out") or 0,
        }
    return mapping


def _module_meta_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[str, ModuleRecord]:
    if frame is None or frame.num_rows == 0:
        return {}
    table = _scoped_table(
        frame,
        repo=repo,
        commit=commit,
        columns=("module", "path", "tags"),
        ctx=ctx,
    )
    if table.num_rows == 0:
        return {}
    meta: dict[str, ModuleRecord] = {}
    for module, path, tags in iter_tuples(
        table_to_reader(table),
        columns=["module", "path", "tags"],
    ):
        path_value = coerce_optional_str(path, ctx="core.modules.path")
        normalized_path = normalize_path(path_value) if path_value else ""
        normalized_tags = _normalize_tags(tags)
        meta[coerce_str(module, ctx="core.modules.module")] = ModuleRecord(
            path=normalized_path,
            tags=normalized_tags,
        )
    return meta


def _scoped_table(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
    columns: Sequence[str],
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> pa.Table:
    require_columns(frame, ("repo", "commit"))
    return snapshot_table(
        frame,
        columns=columns,
        context=SnapshotContext(repo=repo, commit=commit, ctx=ctx),
    )


def _coerce_json(value: object) -> object:
    if isinstance(value, str):
        try:
            return msgspec.json.decode(value)
        except msgspec.DecodeError:
            return value
    return value


def _normalize_tags(raw: object) -> list[str]:
    tags_obj = _coerce_json(raw)
    if tags_obj is None:
        return []
    if isinstance(tags_obj, list):
        return [str(tag) for tag in tags_obj if tag is not None]
    return [str(tags_obj)]
