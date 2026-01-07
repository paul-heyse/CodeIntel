"""Heuristic semantic role classification for functions and modules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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
from codeintel.build.analytics.utilities.ast import RowDecoder
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_masks import FilterExprContext
from codeintel.core.data_models.ids import normalize_decimal_id
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
    )
    function_rows = _function_rows_from_frame(
        inputs.goids_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    effects = _effects_from_frame(
        inputs.function_effects_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    contracts = _contracts_from_frame(
        inputs.function_contracts_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    graph_metrics = _graph_metrics_from_frame(
        inputs.graph_metrics_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
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
                role_sources,
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
    filtered = _filter_table_by_snapshot(
        frame,
        repo=repo,
        commit=commit,
        columns=[goid_column, "rel_path", "qualname", "start_line", "end_line"],
    )
    if filtered.num_rows == 0:
        return []
    result: list[tuple[int, str, str, int | None]] = []
    for row in iter_rows(
        filtered,
        [goid_column, "rel_path", "qualname", "start_line", "end_line"],
    ):
        goid_raw = row.get(goid_column)
        rel_path = row.get("rel_path")
        qualname = row.get("qualname")
        start_line_raw = row.get("start_line")
        end_line_raw = row.get("end_line")
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


def _effects_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> dict[int, dict[str, object]]:
    if frame is None or frame.num_rows == 0:
        return {}
    filtered = _filter_table_by_snapshot(
        frame,
        repo=repo,
        commit=commit,
        columns=[
            "function_goid_h128",
            "touches_db",
            "uses_io",
            "uses_time",
            "uses_randomness",
            "modifies_globals",
            "modifies_closure",
            "spawns_threads_or_tasks",
        ],
    )
    if filtered.num_rows == 0:
        return {}
    mapping: dict[int, dict[str, object]] = {}
    for row in iter_rows(
        filtered,
        [
            "function_goid_h128",
            "touches_db",
            "uses_io",
            "uses_time",
            "uses_randomness",
            "modifies_globals",
            "modifies_closure",
            "spawns_threads_or_tasks",
        ],
    ):
        goid_raw = row.get("function_goid_h128")
        touches_db = row.get("touches_db")
        uses_io = row.get("uses_io")
        uses_time = row.get("uses_time")
        uses_randomness = row.get("uses_randomness")
        modifies_globals = row.get("modifies_globals")
        modifies_closure = row.get("modifies_closure")
        spawns_threads_or_tasks = row.get("spawns_threads_or_tasks")
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
) -> dict[int, dict[str, object]]:
    if frame is None or frame.num_rows == 0:
        return {}
    decoder = RowDecoder(
        columns=(
            "preconditions_json",
            "raises_json",
            "param_nullability_json",
        ),
    )
    filtered = _filter_table_by_snapshot(
        frame,
        repo=repo,
        commit=commit,
        columns=[
            "function_goid_h128",
            "preconditions_json",
            "raises_json",
            "param_nullability_json",
        ],
    )
    if filtered.num_rows == 0:
        return {}
    mapping: dict[int, dict[str, object]] = {}
    for row in iter_rows(
        filtered,
        [
            "function_goid_h128",
            "preconditions_json",
            "raises_json",
            "param_nullability_json",
        ],
    ):
        decoded = decoder.decode(row)
        goid_raw = decoded.get("function_goid_h128")
        preconditions = decoded.get("preconditions_json")
        raises = decoded.get("raises_json")
        param_nullability = decoded.get("param_nullability_json")
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
) -> dict[int, dict[str, int]]:
    if frame is None or frame.num_rows == 0:
        return {}
    filtered = _filter_table_by_snapshot(
        frame,
        repo=repo,
        commit=commit,
        columns=["function_goid_h128", "call_fan_in", "call_fan_out"],
    )
    if filtered.num_rows == 0:
        return {}
    mapping: dict[int, dict[str, int]] = {}
    for row in iter_rows(
        filtered,
        ["function_goid_h128", "call_fan_in", "call_fan_out"],
    ):
        goid_raw = row.get("function_goid_h128")
        call_fan_in = row.get("call_fan_in")
        call_fan_out = row.get("call_fan_out")
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
) -> dict[str, ModuleRecord]:
    if frame is None or frame.num_rows == 0:
        return {}
    decoder = RowDecoder(columns=("tags",))
    filtered = _filter_table_by_snapshot(
        frame,
        repo=repo,
        commit=commit,
        columns=["module", "path", "tags"],
    )
    if filtered.num_rows == 0:
        return {}
    meta: dict[str, ModuleRecord] = {}
    for row in iter_rows(filtered, ["module", "path", "tags"]):
        decoded = decoder.decode(row)
        module = decoded.get("module")
        path = decoded.get("path")
        tags = decoded.get("tags")
        path_value = coerce_optional_str(path, ctx="core.modules.path")
        normalized_path = normalize_path(path_value) if path_value else ""
        normalized_tags = _normalize_tags(tags)
        meta[coerce_str(module, ctx="core.modules.module")] = ModuleRecord(
            path=normalized_path,
            tags=normalized_tags,
        )
    return meta


def _filter_table_by_snapshot(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
    columns: Sequence[str] | None = None,
) -> pa.Table:
    missing = [name for name in ("repo", "commit") if name not in frame.column_names]
    if missing:
        msg = f"Missing snapshot columns: {missing}"
        raise ValueError(msg)
    context = FilterExprContext(repo=repo, commit=commit)
    filtered = context.apply(frame)
    if columns is None:
        return filtered
    return filtered.select(list(columns))


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
