"""GOID targets built from AST and module inventory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl
import pyarrow as pa

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_relation_table,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_helper
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.types import TabularInput
from codeintel.core.columnar import to_lazyframe
from codeintel.core.columnar.tabular_adapter import PolarsExecutionOptions, collect_lazyframe
from codeintel.core.schemas.contracts import arrow_contract_for_table_schema
from codeintel.graphs.compute.goid import (
    GoidDescriptor,
    compute_goid_result,
    determine_kind,
)

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, TabularInput, pa.Table)

GOIDS_TARGET_NAME = "goids"
GOIDS_TABLE_KEY = "core.goids"
GOID_CROSSWALK_TABLE_KEY = "core.goid_crosswalk"
GOIDS_TABLE_KEYS = (GOIDS_TABLE_KEY, GOID_CROSSWALK_TABLE_KEY)

GOIDS_SAVE_CONTEXT = SaverContext(domain="graphs", target=GOIDS_TARGET_NAME)

_ALLOWED_NODE_TYPES = {
    "Module",
    "ClassDef",
    "FunctionDef",
    "AsyncFunctionDef",
}


@dataclass(frozen=True, slots=True)
class _GoidsFrames:
    goids: pa.Table
    crosswalk: pa.Table


def _coerce_line(value: object, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _table_from_rows(table_key: str, rows: list[dict[str, object]]) -> pa.Table:
    schema = get_schema_service().require_table_schema(table_key)
    arrow_schema = arrow_contract_for_table_schema(table_schema=schema)
    if not rows:
        return pa.Table.from_batches([], schema=arrow_schema)
    return pa.Table.from_pylist(rows, schema=arrow_schema)


def _build_goid_frames(
    q__core__ast_nodes: TabularInput,
    q__core__modules: TabularInput,
) -> _GoidsFrames:
    ast_frame = to_lazyframe(q__core__ast_nodes).select(
        pl.col("path").alias("rel_path"),
        "node_type",
        "qualname",
        "parent_qualname",
        pl.col("lineno").alias("start_line"),
        pl.col("end_lineno").alias("end_line"),
    )
    module_frame = to_lazyframe(q__core__modules).select(
        pl.col("path").alias("rel_path"),
        pl.col("module").alias("module_name"),
        "repo",
        "commit",
        "language",
    )
    joined = collect_lazyframe(
        ast_frame.join(module_frame, on="rel_path", how="inner"),
        options=PolarsExecutionOptions(),
    )

    now = datetime.now(UTC)
    goid_rows: list[dict[str, object]] = []
    crosswalk_rows: list[dict[str, object]] = []
    seen_goids: set[int] = set()
    seen_crosswalk: set[tuple[str, str, str]] = set()

    for row in joined.iter_rows(named=True):
        node_type = row.get("node_type")
        if not isinstance(node_type, str) or node_type not in _ALLOWED_NODE_TYPES:
            continue
        qualname = row.get("qualname")
        if not isinstance(qualname, str) or not qualname:
            continue
        module_name = row.get("module_name")
        if not isinstance(module_name, str) or not module_name:
            continue
        repo = row.get("repo")
        commit = row.get("commit")
        language = row.get("language")
        if not all(isinstance(value, str) and value for value in (repo, commit, language)):
            continue
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str) or not rel_path:
            continue
        parent_qualname = row.get("parent_qualname")
        parent_value = parent_qualname if isinstance(parent_qualname, str) else None
        start_line = _coerce_line(row.get("start_line"), default=1)
        end_line = _coerce_line(row.get("end_line"), default=start_line)

        kind = determine_kind(node_type, parent_value, rel_path, module_name)
        descriptor = GoidDescriptor(
            repo=repo,
            commit=commit,
            language=language,
            rel_path=rel_path,
            kind=kind,
            qualname=qualname,
            start_line=start_line,
            end_line=end_line,
        )
        result = compute_goid_result(descriptor)
        if result.goid_h128 in seen_goids:
            continue
        seen_goids.add(result.goid_h128)
        crosswalk_key = (repo, commit, result.urn)
        if crosswalk_key in seen_crosswalk:
            continue
        seen_crosswalk.add(crosswalk_key)

        goid_rows.append(
            {
                "goid_h128": result.goid_h128,
                "urn": result.urn,
                "repo": repo,
                "commit": commit,
                "rel_path": rel_path,
                "language": language,
                "kind": kind,
                "qualname": qualname,
                "start_line": start_line,
                "end_line": end_line,
                "created_at": now,
            }
        )
        crosswalk_rows.append(
            {
                "repo": repo,
                "commit": commit,
                "goid": result.urn,
                "lang": language,
                "module_path": module_name,
                "file_path": rel_path,
                "start_line": start_line,
                "end_line": end_line,
                "scip_symbol": None,
                "ast_qualname": qualname,
                "cst_node_id": None,
                "chunk_id": None,
                "symbol_id": None,
                "updated_at": now,
            }
        )

    return _GoidsFrames(
        goids=_table_from_rows(GOIDS_TABLE_KEY, goid_rows),
        crosswalk=_table_from_rows(GOID_CROSSWALK_TABLE_KEY, crosswalk_rows),
    )


@tag_helper(domain="graphs", target=GOIDS_TARGET_NAME)
def goids__frames(
    q__core__ast_nodes: TabularInput,
    q__core__modules: TabularInput,
) -> _GoidsFrames:
    """Build GOID and crosswalk tables from AST nodes."""
    return _build_goid_frames(
        q__core__ast_nodes=q__core__ast_nodes,
        q__core__modules=q__core__modules,
    )


@save_relation_table(
    context=GOIDS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=GOIDS_TABLE_KEY),
)
def goids__table(goids__frames: _GoidsFrames) -> pa.Table:
    """Persist computed GOID rows."""
    return goids__frames.goids


@save_relation_table(
    context=GOIDS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=GOID_CROSSWALK_TABLE_KEY),
)
def goids__crosswalk_table(goids__frames: _GoidsFrames) -> pa.Table:
    """Persist computed GOID crosswalk rows."""
    return goids__frames.crosswalk


goids__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=GOIDS_TARGET_NAME,
    table_keys=GOIDS_TABLE_KEYS,
    node_name="goids__table_materializations",
)


@codeintel_target(domain="graphs", target=GOIDS_TARGET_NAME)
def t__goids(
    env: BuildEnv,
    catalog: DagCatalog,
    goids__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize goids target run record."""
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=GOIDS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=goids__table_materializations,
    )


__all__ = [
    "GOID_CROSSWALK_TABLE_KEY",
    "GOIDS_TABLE_KEY",
    "GOIDS_TARGET_NAME",
    "goids__crosswalk_table",
    "goids__frames",
    "goids__table",
    "goids__table_materializations",
    "t__goids",
]
