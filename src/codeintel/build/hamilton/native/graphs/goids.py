"""GOID table targets built with inferable tabular nodes."""

from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.build.graphs.compute.goid import (
    GoidDescriptor,
    build_crosswalk_row,
    build_goid_row,
    compute_goid_result,
    determine_kind,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.compute_filters import (
    filter_goid_ast_nodes,
    filter_modules_with_language,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    MultiTableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.rows import GoidCrosswalkRow, GoidRow
from codeintel.core.schemas.row_models import columns_for_table_key
from codeintel.core.spans import normalize_line_span

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

GOIDS_TARGET_NAME = "goids"
GOIDS_TABLE_KEY = "core.goids"
GOID_CROSSWALK_TABLE_KEY = "core.goid_crosswalk"

LOG = logging.getLogger(__name__)


def _columns_for_table(table_key: str) -> tuple[str, ...]:
    columns = columns_for_table_key(table_key)
    if not columns:
        msg = f"No schema columns registered for {table_key}"
        raise ValueError(msg)
    return tuple(columns)


GOIDS_COLUMNS = _columns_for_table(GOIDS_TABLE_KEY)
GOID_CROSSWALK_COLUMNS = _columns_for_table(GOID_CROSSWALK_TABLE_KEY)


def _partitioned_save_spec(table_key: str) -> DatasetSaveSpec:
    return DatasetSaveSpec(
        table_key=table_key,
        partition_columns=("repo", "commit"),
    )


@dataclass(frozen=True, slots=True)
class _GoidsInputs:
    modules: pa.Table
    ast_nodes: pa.Table


@dataclass(frozen=True, slots=True)
class _ResolvedDescriptor:
    descriptor: GoidDescriptor
    module_name: str


@dataclass(frozen=True, slots=True)
class _GoidsAnalysis:
    goid_rows: tuple[GoidRow, ...]
    crosswalk_rows: tuple[GoidCrosswalkRow, ...]


def _rows_to_reader(
    rows: tuple[GoidRow | GoidCrosswalkRow, ...],
    table_key: str,
) -> InferableTabularInput:
    if not rows:
        return empty_table_for_table(table_key)
    reader, _ = table_for_rows(
        table_key,
        (dataclasses.asdict(row) for row in rows),
    )
    return reader


def _module_frame(modules_table: pa.Table) -> list[dict[str, str]]:
    if modules_table.num_rows == 0:
        return []
    required = {"path", "module", "language"}
    if not required.issubset(set(modules_table.column_names)):
        return []
    modules_table = filter_modules_with_language(modules_table)
    rows: list[dict[str, str]] = []
    for row in iter_rows(modules_table):
        path = row.get("path")
        module = row.get("module")
        language = row.get("language")
        if not isinstance(path, str) or not path:
            continue
        if not isinstance(module, str) or not module:
            continue
        if not isinstance(language, str) or not language:
            continue
        rows.append({"path": path, "module": module, "language": language})
    return rows


def _resolve_qualname(
    *,
    node_type: str,
    module_name: str,
    name: object,
    qualname: object,
    parent_qualname: object,
) -> str | None:
    if isinstance(qualname, str) and qualname:
        return qualname
    if node_type == "Module":
        return module_name
    if not isinstance(name, str) or not name:
        return None
    if isinstance(parent_qualname, str) and parent_qualname:
        return f"{parent_qualname}.{name}"
    return f"{module_name}.{name}"


def _resolve_start_line(node_type: str, start_line: object) -> int | None:
    if isinstance(start_line, int):
        return start_line
    if node_type == "Module":
        return 0
    return None


_ALLOWED_NODE_TYPES = frozenset({"Module", "ClassDef", "FunctionDef", "AsyncFunctionDef"})


@dataclass(frozen=True, slots=True)
class _DescriptorContext:
    repo: str
    commit: str


@dataclass(frozen=True, slots=True)
class _DescriptorValues:
    node_type: str | None
    path: str | None
    module_name: str | None
    language: str | None
    name: object
    qualname: object
    parent_qualname: object
    lineno: object
    end_lineno: object


def _descriptor_from_values(
    *,
    values: _DescriptorValues,
    context: _DescriptorContext,
) -> tuple[_ResolvedDescriptor, tuple[str, str, str, int]] | None:
    if values.node_type is None or values.node_type not in _ALLOWED_NODE_TYPES:
        return None
    if values.path is None or values.module_name is None or values.language is None:
        return None
    start_line = _resolve_start_line(values.node_type, values.lineno)
    if start_line is None:
        return None
    _, resolved_end = normalize_line_span(
        start_line,
        values.end_lineno if isinstance(values.end_lineno, int) else None,
    )
    qualname_value = _resolve_qualname(
        node_type=values.node_type,
        module_name=values.module_name,
        name=values.name,
        qualname=values.qualname,
        parent_qualname=values.parent_qualname,
    )
    if qualname_value is None:
        return None
    parent_value = (
        values.parent_qualname if isinstance(values.parent_qualname, str) else values.module_name
    )
    kind = determine_kind(values.node_type, parent_value, values.path, values.module_name)
    descriptor = GoidDescriptor(
        repo=context.repo,
        commit=context.commit,
        language=values.language,
        rel_path=values.path,
        kind=kind,
        qualname=qualname_value,
        start_line=start_line,
        end_line=resolved_end,
    )
    dedupe_key = (values.path, values.node_type, qualname_value, start_line)
    return (
        _ResolvedDescriptor(descriptor=descriptor, module_name=values.module_name),
        dedupe_key,
    )


def _joined_ast_nodes(
    ast_nodes_table: pa.Table,
    modules: list[dict[str, str]],
) -> list[dict[str, object]]:
    if ast_nodes_table.num_rows == 0 or not modules:
        return []
    required = {
        "path",
        "node_type",
        "name",
        "qualname",
        "parent_qualname",
        "lineno",
        "end_lineno",
    }
    if not required.issubset(set(ast_nodes_table.column_names)):
        return []
    module_by_path = {row["path"]: row for row in modules}
    joined: list[dict[str, object]] = []
    total = 0
    matched = 0
    filtered = filter_goid_ast_nodes(ast_nodes_table)
    for row in iter_rows(filtered):
        node_type = row.get("node_type")
        if node_type not in _ALLOWED_NODE_TYPES:
            continue
        total += 1
        path = row.get("path")
        if not isinstance(path, str):
            continue
        module_row = module_by_path.get(path)
        if module_row is None:
            continue
        matched += 1
        joined.append({**row, **module_row})
    if total:
        LOG.info(
            "goids join coverage ast_nodes_to_modules matched=%d total=%d",
            matched,
            total,
        )
    return joined


def _collect_descriptors(
    *,
    joined_nodes: list[dict[str, object]],
    repo: str,
    commit: str,
) -> list[_ResolvedDescriptor]:
    descriptors: list[_ResolvedDescriptor] = []
    seen: set[tuple[str, str, str, int]] = set()
    if not joined_nodes:
        return descriptors
    for row in joined_nodes:
        node_type = row.get("node_type")
        path = row.get("path")
        module_name = row.get("module")
        language = row.get("language")
        name = row.get("name")
        qualname = row.get("qualname")
        parent_qualname = row.get("parent_qualname")
        lineno = row.get("lineno")
        end_lineno = row.get("end_lineno")
        result = _descriptor_from_values(
            values=_DescriptorValues(
                node_type=str(node_type) if node_type is not None else None,
                path=str(path) if path is not None else None,
                module_name=str(module_name) if module_name is not None else None,
                language=str(language) if language is not None else None,
                name=name,
                qualname=qualname,
                parent_qualname=parent_qualname,
                lineno=lineno,
                end_lineno=end_lineno,
            ),
            context=_DescriptorContext(repo=repo, commit=commit),
        )
        if result is None:
            continue
        resolved, dedupe_key = result
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        descriptors.append(resolved)
    return descriptors


def goids_inputs(
    q__core__modules: InferableTabularInput,
    q__core__ast_nodes: InferableTabularInput,
) -> _GoidsInputs:
    """Collect GOID inputs from inferred tabular nodes.

    Returns
    -------
    _GoidsInputs
        Collected frames for GOID computation.
    """
    modules_table = tabular_to_scoped_table(
        q__core__modules,
        columns=["path", "module", "language"],
        scope=None,
        require_scope_columns=False,
    )
    ast_nodes_table = tabular_to_scoped_table(
        q__core__ast_nodes,
        columns=[
            "path",
            "node_type",
            "name",
            "qualname",
            "parent_qualname",
            "lineno",
            "end_lineno",
        ],
        scope=None,
        require_scope_columns=False,
    )
    return _GoidsInputs(modules=modules_table, ast_nodes=ast_nodes_table)


def goids_analysis(env: BuildEnv, goids_inputs: _GoidsInputs) -> _GoidsAnalysis:
    """Compute GOID and crosswalk rows from AST nodes.

    Parameters
    ----------
    env
        Build environment for repo/commit context.
    goids_inputs
        Collected core modules and AST nodes frames.

    Returns
    -------
    _GoidsAnalysis
        Container with GOID and crosswalk row tuples.
    """
    if goids_inputs.modules.num_rows == 0 or goids_inputs.ast_nodes.num_rows == 0:
        return _GoidsAnalysis(goid_rows=(), crosswalk_rows=())

    modules = _module_frame(goids_inputs.modules)
    if not modules:
        return _GoidsAnalysis(goid_rows=(), crosswalk_rows=())

    joined_nodes = _joined_ast_nodes(goids_inputs.ast_nodes, modules)
    descriptors = _collect_descriptors(
        joined_nodes=joined_nodes,
        repo=env.repo,
        commit=env.commit,
    )
    if not descriptors:
        return _GoidsAnalysis(goid_rows=(), crosswalk_rows=())

    now = datetime.now(UTC)
    goid_rows: list[GoidRow] = []
    crosswalk_rows: list[GoidCrosswalkRow] = []
    for resolved in descriptors:
        result = compute_goid_result(resolved.descriptor)
        goid_rows.append(
            build_goid_row(
                resolved.descriptor,
                result.goid_h128,
                result.urn,
                now,
            )
        )
        crosswalk_rows.append(
            build_crosswalk_row(
                resolved.descriptor,
                result.urn,
                resolved.module_name,
                now,
            )
        )
    return _GoidsAnalysis(
        goid_rows=tuple(goid_rows),
        crosswalk_rows=tuple(crosswalk_rows),
    )


def goids__base(goids_analysis: _GoidsAnalysis) -> InferableTabularInput:
    """Build core.goids rows from the analysis payload.

    Returns
    -------
    InferableTabularInput
        Arrow reader of core.goids rows.
    """
    return _rows_to_reader(goids_analysis.goid_rows, GOIDS_TABLE_KEY)


def goid_crosswalk__base(goids_analysis: _GoidsAnalysis) -> InferableTabularInput:
    """Build core.goid_crosswalk rows from the analysis payload.

    Returns
    -------
    InferableTabularInput
        Arrow reader of core.goid_crosswalk rows.
    """
    return _rows_to_reader(goids_analysis.crosswalk_rows, GOID_CROSSWALK_TABLE_KEY)


_MODULE = sys.modules[__name__]
_GOIDS_TABLE_CONTEXTS = (
    TableTargetTableContext(
        table_key=GOIDS_TABLE_KEY,
        base_node="goids__base",
        node_name="goids__table",
    ),
    TableTargetTableContext(
        table_key=GOID_CROSSWALK_TABLE_KEY,
        base_node="goid_crosswalk__base",
        node_name="goid_crosswalk__table",
    ),
)
_GOIDS_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="ingestion",
        target_name=GOIDS_TARGET_NAME,
        tables=(),
        table_materializations_node="goids__table_materializations",
        anchor_node_name="t__goids",
        save_spec_factory=_partitioned_save_spec,
        default_input_type=InferableTabularInput,
    ),
    table_contexts=_GOIDS_TABLE_CONTEXTS,
)
attach_table_target_template(_MODULE, spec=_GOIDS_TABLE_TARGET_SPEC)
goids__table = _MODULE.goids__table
goid_crosswalk__table = _MODULE.goid_crosswalk__table
goids__table_materializations = _MODULE.goids__table_materializations
t__goids = _MODULE.t__goids


__all__ = [
    "GOIDS_TABLE_KEY",
    "GOIDS_TARGET_NAME",
    "GOID_CROSSWALK_TABLE_KEY",
    "goid_crosswalk__base",
    "goid_crosswalk__table",
    "goids__base",
    "goids__table",
    "goids__table_materializations",
    "t__goids",
]
