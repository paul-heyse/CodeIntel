"""GOID table targets built with inferable tabular nodes."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from codeintel.build.graphs.compute.goid import (
    GoidDescriptor,
    build_crosswalk_row,
    build_goid_row,
    compute_goid_result,
    determine_kind,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.tabular.frames import empty_frame_for_table
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.data_models.rows import GoidCrosswalkRow, GoidRow
from codeintel.core.spans import normalize_line_span

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

GOIDS_TARGET_NAME = "goids"
GOIDS_TABLE_KEY = "core.goids"
GOID_CROSSWALK_TABLE_KEY = "core.goid_crosswalk"

GOIDS_COLUMNS = (
    "goid_h128",
    "urn",
    "repo",
    "commit",
    "rel_path",
    "language",
    "kind",
    "qualname",
    "start_line",
    "end_line",
    "created_at",
)
GOID_CROSSWALK_COLUMNS = (
    "repo",
    "commit",
    "goid",
    "lang",
    "module_path",
    "file_path",
    "start_line",
    "end_line",
    "scip_symbol",
    "ast_qualname",
    "cst_node_id",
    "chunk_id",
    "symbol_id",
    "updated_at",
)


@dataclass(frozen=True, slots=True)
class _GoidsInputs:
    modules: pl.DataFrame
    ast_nodes: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _ResolvedDescriptor:
    descriptor: GoidDescriptor
    module_name: str


@dataclass(frozen=True, slots=True)
class _GoidsAnalysis:
    goid_rows: tuple[GoidRow, ...]
    crosswalk_rows: tuple[GoidCrosswalkRow, ...]


def _rows_to_frame(
    rows: tuple[GoidRow | GoidCrosswalkRow, ...],
    columns: tuple[str, ...],
    table_key: str,
) -> pl.LazyFrame:
    if not rows:
        return empty_frame_for_table(table_key)
    frame = pl.DataFrame([dataclasses.asdict(row) for row in rows], orient="row")
    return frame.lazy().select(list(columns))


def _module_lookup(
    modules_frame: pl.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    module_by_path: dict[str, str] = {}
    language_by_path: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        language = row.get("language")
        if not isinstance(path, str) or not path:
            continue
        if not isinstance(module, str) or not module:
            continue
        if not isinstance(language, str) or not language:
            continue
        module_by_path[path] = module
        language_by_path[path] = language
    return module_by_path, language_by_path


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


def _descriptor_from_row(
    *,
    row: dict[str, object],
    module_by_path: dict[str, str],
    language_by_path: dict[str, str],
    repo: str,
    commit: str,
) -> tuple[_ResolvedDescriptor, tuple[str, str, str, int]] | None:
    node_type = row.get("node_type")
    if not isinstance(node_type, str) or node_type not in _ALLOWED_NODE_TYPES:
        return None
    path = row.get("path")
    if not isinstance(path, str):
        return None
    module_name = module_by_path.get(path)
    language = language_by_path.get(path)
    if module_name is None or language is None:
        return None
    start_line = _resolve_start_line(node_type, row.get("lineno"))
    if start_line is None:
        return None
    end_line = row.get("end_lineno")
    _, resolved_end = normalize_line_span(
        start_line,
        end_line if isinstance(end_line, int) else None,
    )
    qualname = _resolve_qualname(
        node_type=node_type,
        module_name=module_name,
        name=row.get("name"),
        qualname=row.get("qualname"),
        parent_qualname=row.get("parent_qualname"),
    )
    if qualname is None:
        return None
    parent_qualname = row.get("parent_qualname")
    parent_value = parent_qualname if isinstance(parent_qualname, str) else module_name
    kind = determine_kind(node_type, parent_value, path, module_name)
    descriptor = GoidDescriptor(
        repo=repo,
        commit=commit,
        language=language,
        rel_path=path,
        kind=kind,
        qualname=qualname,
        start_line=start_line,
        end_line=resolved_end,
    )
    dedupe_key = (path, node_type, qualname, start_line)
    return _ResolvedDescriptor(descriptor=descriptor, module_name=module_name), dedupe_key


def _collect_descriptors(
    *,
    ast_nodes_frame: pl.DataFrame,
    module_by_path: dict[str, str],
    language_by_path: dict[str, str],
    repo: str,
    commit: str,
) -> list[_ResolvedDescriptor]:
    descriptors: list[_ResolvedDescriptor] = []
    seen: set[tuple[str, str, str, int]] = set()

    for row in ast_nodes_frame.iter_rows(named=True):
        result = _descriptor_from_row(
            row=row,
            module_by_path=module_by_path,
            language_by_path=language_by_path,
            repo=repo,
            commit=commit,
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
    return _GoidsInputs(
        modules=tabular_to_lazyframe(q__core__modules).collect(),
        ast_nodes=tabular_to_lazyframe(q__core__ast_nodes).collect(),
    )


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
    if goids_inputs.modules.is_empty() or goids_inputs.ast_nodes.is_empty():
        return _GoidsAnalysis(goid_rows=(), crosswalk_rows=())

    module_by_path, language_by_path = _module_lookup(goids_inputs.modules)
    if not module_by_path:
        return _GoidsAnalysis(goid_rows=(), crosswalk_rows=())

    descriptors = _collect_descriptors(
        ast_nodes_frame=goids_inputs.ast_nodes,
        module_by_path=module_by_path,
        language_by_path=language_by_path,
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


def goids__base(goids_analysis: _GoidsAnalysis) -> pl.LazyFrame:
    """Build core.goids rows from the analysis payload.

    Returns
    -------
    polars.LazyFrame
        Lazy frame of core.goids rows.
    """
    return _rows_to_frame(goids_analysis.goid_rows, GOIDS_COLUMNS, GOIDS_TABLE_KEY)


def goid_crosswalk__base(goids_analysis: _GoidsAnalysis) -> pl.LazyFrame:
    """Build core.goid_crosswalk rows from the analysis payload.

    Returns
    -------
    polars.LazyFrame
        Lazy frame of core.goid_crosswalk rows.
    """
    return _rows_to_frame(
        goids_analysis.crosswalk_rows,
        GOID_CROSSWALK_COLUMNS,
        GOID_CROSSWALK_TABLE_KEY,
    )


_MODULE = sys.modules[__name__]
_GOIDS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=GOIDS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=GOIDS_TABLE_KEY,
            base_node="goids__base",
            save_spec=DatasetSaveSpec(
                table_key=GOIDS_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="goids__table",
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=GOID_CROSSWALK_TABLE_KEY,
            base_node="goid_crosswalk__base",
            save_spec=DatasetSaveSpec(
                table_key=GOID_CROSSWALK_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="goid_crosswalk__table",
            input_type=pl.LazyFrame,
        ),
    ),
    table_materializations_node="goids__table_materializations",
    anchor_node_name="t__goids",
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
