"""GOID targets built from AST and module inventory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

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
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


@dataclass(frozen=True, slots=True)
class _ParsedGoidRow:
    repo: str
    commit: str
    language: str
    rel_path: str
    module_name: str
    node_type: str
    qualname: str
    parent_value: str | None
    start_line: int
    end_line: int


def _parse_goid_row(row: Mapping[str, object]) -> _ParsedGoidRow | None:
    node_type = row.get("node_type")
    if not isinstance(node_type, str) or node_type not in _ALLOWED_NODE_TYPES:
        return None
    qualname = row.get("qualname")
    if not isinstance(qualname, str) or not qualname:
        return None
    module_name = row.get("module_name")
    if not isinstance(module_name, str) or not module_name:
        return None
    repo = row.get("repo")
    commit = row.get("commit")
    language = row.get("language")
    if not (
        isinstance(repo, str)
        and repo
        and isinstance(commit, str)
        and commit
        and isinstance(language, str)
        and language
    ):
        return None
    rel_path = row.get("rel_path")
    if not isinstance(rel_path, str) or not rel_path:
        return None
    parent_qualname = row.get("parent_qualname")
    parent_value = parent_qualname if isinstance(parent_qualname, str) else None
    start_line = _coerce_line(row.get("start_line"), default=1)
    end_line = _coerce_line(row.get("end_line"), default=start_line)
    return _ParsedGoidRow(
        repo=repo,
        commit=commit,
        language=language,
        rel_path=rel_path,
        module_name=module_name,
        node_type=node_type,
        qualname=qualname,
        parent_value=parent_value,
        start_line=start_line,
        end_line=end_line,
    )


def _table_from_rows(table_key: str, rows: list[dict[str, object]]) -> pa.Table:
    schema = get_schema_service().require_table_schema(table_key)
    arrow_schema = arrow_contract_for_table_schema(table_schema=schema)
    if not rows:
        return pa.Table.from_batches([], schema=arrow_schema)
    return pa.Table.from_pylist(rows, schema=arrow_schema)


def _joined_goid_inputs(
    q__core__ast_nodes: TabularInput,
    q__core__modules: TabularInput,
) -> pl.DataFrame:
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
    return collect_lazyframe(
        ast_frame.join(module_frame, on="rel_path", how="inner"),
        options=PolarsExecutionOptions(),
    )


def _collect_goid_rows(
    joined: pl.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    now = datetime.now(UTC)
    goid_rows: list[dict[str, object]] = []
    crosswalk_rows: list[dict[str, object]] = []
    seen_goids: set[int] = set()
    seen_crosswalk: set[tuple[str, str, str]] = set()

    for row in joined.iter_rows(named=True):
        row_mapping = cast("Mapping[str, object]", row)
        parsed = _parse_goid_row(row_mapping)
        if parsed is None:
            continue
        kind = determine_kind(
            parsed.node_type,
            parsed.parent_value,
            parsed.rel_path,
            parsed.module_name,
        )
        descriptor = GoidDescriptor(
            repo=parsed.repo,
            commit=parsed.commit,
            language=parsed.language,
            rel_path=parsed.rel_path,
            kind=kind,
            qualname=parsed.qualname,
            start_line=parsed.start_line,
            end_line=parsed.end_line,
        )
        result = compute_goid_result(descriptor)
        if result.goid_h128 in seen_goids:
            continue
        seen_goids.add(result.goid_h128)
        crosswalk_key = (parsed.repo, parsed.commit, result.urn)
        if crosswalk_key in seen_crosswalk:
            continue
        seen_crosswalk.add(crosswalk_key)

        goid_rows.append(
            {
                "goid_h128": result.goid_h128,
                "urn": result.urn,
                "repo": parsed.repo,
                "commit": parsed.commit,
                "rel_path": parsed.rel_path,
                "language": parsed.language,
                "kind": kind,
                "qualname": parsed.qualname,
                "start_line": parsed.start_line,
                "end_line": parsed.end_line,
                "created_at": now,
            }
        )
        crosswalk_rows.append(
            {
                "repo": parsed.repo,
                "commit": parsed.commit,
                "goid": result.urn,
                "lang": parsed.language,
                "module_path": parsed.module_name,
                "file_path": parsed.rel_path,
                "start_line": parsed.start_line,
                "end_line": parsed.end_line,
                "scip_symbol": None,
                "ast_qualname": parsed.qualname,
                "cst_node_id": None,
                "chunk_id": None,
                "symbol_id": None,
                "updated_at": now,
            }
        )

    return goid_rows, crosswalk_rows


def _build_goid_frames(
    q__core__ast_nodes: TabularInput,
    q__core__modules: TabularInput,
) -> _GoidsFrames:
    joined = _joined_goid_inputs(
        q__core__ast_nodes=q__core__ast_nodes,
        q__core__modules=q__core__modules,
    )
    goid_rows, crosswalk_rows = _collect_goid_rows(joined)
    return _GoidsFrames(
        goids=_table_from_rows(GOIDS_TABLE_KEY, goid_rows),
        crosswalk=_table_from_rows(GOID_CROSSWALK_TABLE_KEY, crosswalk_rows),
    )


@tag_helper(domain="graphs", target=GOIDS_TARGET_NAME)
def goids__frames(
    q__core__ast_nodes: TabularInput,
    q__core__modules: TabularInput,
) -> _GoidsFrames:
    """Build GOID and crosswalk tables from AST nodes.

    Returns
    -------
    _GoidsFrames
        GOID and crosswalk Arrow tables.
    """
    return _build_goid_frames(
        q__core__ast_nodes=q__core__ast_nodes,
        q__core__modules=q__core__modules,
    )


@save_relation_table(
    context=GOIDS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=GOIDS_TABLE_KEY),
)
def goids__table(goids__frames: _GoidsFrames) -> pa.Table:
    """Persist computed GOID rows.

    Returns
    -------
    pyarrow.Table
        GOID table output.
    """
    return goids__frames.goids


@save_relation_table(
    context=GOIDS_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=GOID_CROSSWALK_TABLE_KEY),
)
def goids__crosswalk_table(goids__frames: _GoidsFrames) -> pa.Table:
    """Persist computed GOID crosswalk rows.

    Returns
    -------
    pyarrow.Table
        GOID crosswalk output.
    """
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
    """Finalize goids target run record.

    Returns
    -------
    TargetRunRecord
        Run record for GOID materializations.
    """
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
    "GOIDS_TABLE_KEY",
    "GOIDS_TARGET_NAME",
    "GOID_CROSSWALK_TABLE_KEY",
    "goids__crosswalk_table",
    "goids__frames",
    "goids__table",
    "goids__table_materializations",
    "t__goids",
]
