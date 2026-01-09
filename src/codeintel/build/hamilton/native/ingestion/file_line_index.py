"""Materialize per-file line byte offsets for span normalization."""

from __future__ import annotations

import io
import logging
import sys
import tokenize
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpecOptions,
    TableTargetContext,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.ingestion_normalize import scoped_table_for_ingest
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.execution_context import (
    ExecutionContext,
    resolve_columnar_context,
    resolve_execution_context,
)
from codeintel.core.columnar.expr_vocab import E
from codeintel.core.columnar.plan_builder import build_grouped_rollup_plan, build_table_plan
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FILE_LINE_INDEX_TARGET_NAME = "file_line_index"
FILE_LINE_INDEX_TABLE_KEY = "core.file_line_index"


def _canonical_sort_keys_for_table(
    table_key: str,
    columns: Sequence[str],
) -> list[tuple[str, str]] | None:
    schema = get_schema_service().get_table_schema(table_key)
    keys = resolve_canonical_sort_keys(schema)
    if not keys:
        return None
    available = set(columns)
    return [(key, "ascending") for key in keys if key in available]


def _plan_from_table(table: pa.Table, *, table_key: str) -> Plan:
    plan = Plan.table(table)
    sort_keys = _canonical_sort_keys_for_table(table_key, table.column_names)
    if sort_keys:
        return plan.order_by(sort_keys=sort_keys)
    return plan


def _resolve_module_paths(
    modules_table: pa.Table,
    *,
    execution_ctx: ExecutionContext,
) -> dict[str, str | None]:
    if modules_table.num_rows == 0 or "path" not in modules_table.column_names:
        return {}
    columns = ["path"]
    if "language" in modules_table.column_names:
        columns.append("language")
    project = {name: E.field(name) for name in columns}
    plan = build_table_plan(table=modules_table).project(project)
    plan = plan.filter(E.and_(E.is_valid("path"), E.field("path") != E.scalar("")))
    if "language" in columns:
        plan = build_grouped_rollup_plan(
            plan,
            keys=("path",),
            aggregates=(("language", "min", None, "language"),),
            order_by=(("path", "ascending"),),
        )
    else:
        plan = build_grouped_rollup_plan(
            plan,
            keys=("path",),
            aggregates=(),
            order_by=(("path", "ascending"),),
        )
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    paths: dict[str, str | None] = {}
    for batch in reader:
        for row in iter_rows(batch):
            rel_path = row.get("path")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            language = row.get("language")
            paths[rel_path] = language if isinstance(language, str) else None
    return paths


def _resolve_file_path(repo_root: Path, rel_path: str) -> Path:
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _detect_encoding(
    *,
    path: Path,
    data: bytes,
    language: str | None,
) -> str:
    if language == "python" or path.suffix == ".py":
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        except (SyntaxError, UnicodeDecodeError, LookupError):
            return "utf-8"
        else:
            return encoding
    return "utf-8"


def _line_rows_for_bytes(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    data: bytes,
    encoding: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    for line_index, line_bytes in enumerate(data.splitlines(keepends=True)):
        end_offset = offset + len(line_bytes)
        rows.append(
            {
                "repo": repo,
                "commit": commit,
                "rel_path": rel_path,
                "line": line_index,
                "start_byte": offset,
                "end_byte": end_offset,
                "encoding": encoding,
            }
        )
        offset = end_offset
    return rows


def file_line_index__base(
    env: BuildEnv,
    q__core__modules: InferableTabularInput,
) -> Plan:
    """Build core.file_line_index rows from repository files.

    Returns
    -------
    Plan
        Plan for line index rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    modules_table = scoped_table_for_ingest(
        q__core__modules,
        table_key="core.modules",
        scope=scope,
        columns=None,
        require_scope_columns=True,
    )
    if modules_table.num_rows == 0:
        return _plan_from_table(
            empty_table_for_table(FILE_LINE_INDEX_TABLE_KEY),
            table_key=FILE_LINE_INDEX_TABLE_KEY,
        )

    repo_root = Path(env.snapshot.repo_root)
    execution_ctx = resolve_columnar_context(env.execution_context)
    resolved_ctx = resolve_execution_context(execution_ctx)
    path_languages = _resolve_module_paths(modules_table, execution_ctx=resolved_ctx)
    if not path_languages:
        return _plan_from_table(
            empty_table_for_table(FILE_LINE_INDEX_TABLE_KEY),
            table_key=FILE_LINE_INDEX_TABLE_KEY,
        )

    rows: list[dict[str, object]] = []
    for rel_path, language in sorted(path_languages.items()):
        file_path = _resolve_file_path(repo_root, rel_path)
        if not file_path.is_file():
            log.warning("Missing file for line index: %s", file_path)
            continue
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            log.warning("Failed to read %s: %s", file_path, exc)
            continue
        encoding = _detect_encoding(path=file_path, data=data, language=language)
        rows.extend(
            _line_rows_for_bytes(
                repo=env.repo,
                commit=env.commit,
                rel_path=rel_path,
                data=data,
                encoding=encoding,
            )
        )

    table, _ = table_for_rows(FILE_LINE_INDEX_TABLE_KEY, rows)
    return _plan_from_table(table, table_key=FILE_LINE_INDEX_TABLE_KEY)


_MODULE = sys.modules[__name__]
_FILE_LINE_INDEX_TABLE_TARGET_SPEC = TableTargetContext.build_dataset_table_spec(
    context=TableTargetContext(
        domain="ingestion",
        target_name=FILE_LINE_INDEX_TARGET_NAME,
        table_key=FILE_LINE_INDEX_TABLE_KEY,
        base_node="file_line_index__base",
        input_type=InferableTabularInput,
    ),
    save_options=DatasetSaveSpecOptions(
        partition_columns=("repo", "commit"),
        ingest_finalize=True,
    ),
)
attach_table_target_template(_MODULE, spec=_FILE_LINE_INDEX_TABLE_TARGET_SPEC)
file_line_index__table = _MODULE.file_line_index__table
file_line_index__table_materializations = _MODULE.file_line_index__table_materializations
t__file_line_index = _MODULE.t__file_line_index


__all__ = [
    "FILE_LINE_INDEX_TABLE_KEY",
    "FILE_LINE_INDEX_TARGET_NAME",
    "file_line_index__base",
    "file_line_index__table",
    "file_line_index__table_materializations",
    "t__file_line_index",
]
