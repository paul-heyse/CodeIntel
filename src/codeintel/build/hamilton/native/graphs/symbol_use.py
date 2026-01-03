"""Symbol use relation sources for graph targets."""

from __future__ import annotations

import dataclasses
import sys

import polars as pl
from intervaltree import IntervalTree

from codeintel.build.graphs.compute.symbols import (
    SymbolOccurrence,
    SymbolUseEdge,
    build_use_edges,
    edges_to_rows,
    parse_symbol_roles,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_reader_for_table, record_batch_reader_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.spans import normalize_line_span, to_half_open_span

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SYMBOL_USES_TARGET_NAME = "symbol_uses"
SYMBOL_USE_EDGES_TABLE_KEY = "graph.symbol_use_edges"


def _module_by_path(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    if modules_frame.is_empty():
        return module_by_path
    if not {"path", "module"}.issubset(set(modules_frame.columns)):
        return module_by_path
    data = modules_frame.select(["path", "module"]).to_dict(as_series=False)
    module_by_path.update(
        {
            path: module
            for path, module in zip(data["path"], data["module"], strict=True)
            if isinstance(path, str) and isinstance(module, str)
        }
    )
    return module_by_path


def _goid_spans_by_path(
    goids_frame: pl.DataFrame,
) -> dict[str, IntervalTree]:
    spans_by_path: dict[str, IntervalTree] = {}
    if goids_frame.is_empty() or "rel_path" not in goids_frame.columns:
        return spans_by_path
    data = goids_frame.select(["rel_path", "goid_h128", "start_line", "end_line"]).to_dict(
        as_series=False
    )
    for rel_path, goid_raw, start_line, end_line in zip(
        data["rel_path"],
        data["goid_h128"],
        data["start_line"],
        data["end_line"],
        strict=True,
    ):
        if not isinstance(rel_path, str):
            continue
        goid_value = normalize_decimal_id(goid_raw)
        if goid_value is None:
            continue
        if not isinstance(start_line, int):
            continue
        _, resolved_end = normalize_line_span(
            start_line,
            end_line if isinstance(end_line, int) else None,
        )
        span_start, span_end = to_half_open_span(start_line, resolved_end)
        tree = spans_by_path.get(rel_path)
        if tree is None:
            tree = IntervalTree()
            spans_by_path[rel_path] = tree
        tree.addi(span_start, span_end, int(goid_value))
    return spans_by_path


def _match_goid(tree: IntervalTree | None, line: int) -> int | None:
    if tree is None:
        return None
    span_start, span_end = to_half_open_span(line, line)
    matches = tree.overlap(span_start, span_end)
    if not matches:
        return None
    best = min(matches, key=lambda item: (item.end - item.begin, item.begin, int(item.data)))
    return int(best.data)


def _symbol_occurrences(occurrences_frame: pl.DataFrame) -> list[SymbolOccurrence]:
    occurrences: list[SymbolOccurrence] = []
    if occurrences_frame.is_empty():
        return occurrences
    required = {"symbol", "rel_path", "start_line"}
    if not required.issubset(set(occurrences_frame.columns)):
        return occurrences
    columns = ["symbol", "rel_path", "start_line"]
    if "roles" in occurrences_frame.columns:
        columns.append("roles")
    data = occurrences_frame.select(columns).to_dict(as_series=False)
    roles_values = data.get("roles")
    for idx, (symbol, rel_path, start_line) in enumerate(
        zip(data["symbol"], data["rel_path"], data["start_line"], strict=True)
    ):
        if not isinstance(symbol, str) or not isinstance(rel_path, str):
            continue
        if not isinstance(start_line, int):
            continue
        roles = parse_symbol_roles(roles_values[idx] if roles_values is not None else None)
        occurrences.append(
            SymbolOccurrence(
                symbol=symbol,
                rel_path=rel_path,
                line=start_line,
                roles=roles,
            )
        )
    return occurrences


def _definitions_by_symbol(
    occurrences: list[SymbolOccurrence],
) -> dict[str, tuple[str, int]]:
    definitions: dict[str, tuple[str, int]] = {}
    for occ in occurrences:
        if not occ.is_definition:
            continue
        if occ.symbol in definitions:
            continue
        definitions[occ.symbol] = (occ.rel_path, occ.line)
    return definitions


def _reference_lines_by_symbol_path(
    occurrences: list[SymbolOccurrence],
) -> dict[tuple[str, str], int]:
    use_lines: dict[tuple[str, str], int] = {}
    for occ in occurrences:
        if not occ.is_reference:
            continue
        key = (occ.symbol, occ.rel_path)
        existing = use_lines.get(key)
        if existing is None or occ.line < existing:
            use_lines[key] = occ.line
    return use_lines


def _attach_goids(
    edges: list[SymbolUseEdge],
    def_map: dict[str, tuple[str, int]],
    use_lines_by_symbol_path: dict[tuple[str, str], int],
    goid_spans_by_path: dict[str, IntervalTree],
) -> list[SymbolUseEdge]:
    updated: list[SymbolUseEdge] = []
    for edge in edges:
        def_info = def_map.get(edge.symbol)
        def_goid: int | None = None
        if def_info is not None:
            def_tree = goid_spans_by_path.get(def_info[0])
            def_goid = _match_goid(def_tree, def_info[1])
        use_tree = goid_spans_by_path.get(edge.use_path)
        use_line = use_lines_by_symbol_path.get((edge.symbol, edge.use_path))
        use_goid = _match_goid(use_tree, use_line) if use_line is not None else None
        updated.append(
            SymbolUseEdge(
                symbol=edge.symbol,
                def_path=edge.def_path,
                use_path=edge.use_path,
                same_file=edge.same_file,
                same_module=edge.same_module,
                def_goid=def_goid,
                use_goid=use_goid,
            )
        )
    return updated


def symbol_use_edges_compute(
    q__core__scip_occurrences: InferableTabularInput,
    q__core__modules: InferableTabularInput,
    q__core__goids: InferableTabularInput,
) -> InferableTabularInput:
    """Build symbol use edges from SCIP occurrences and GOID spans.

    Returns
    -------
    InferableTabularInput
        Tabular input for computed symbol use edges.
    """
    occurrences_frame = (
        tabular_to_lazyframe(q__core__scip_occurrences)
        .select(["symbol", "rel_path", "start_line", "roles"])
        .collect()
    )
    if occurrences_frame.is_empty():
        return empty_reader_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    modules_frame = tabular_to_lazyframe(q__core__modules).select(["path", "module"]).collect()
    goids_frame = (
        tabular_to_lazyframe(q__core__goids)
        .select(["rel_path", "goid_h128", "start_line", "end_line"])
        .collect()
    )
    occurrences = _symbol_occurrences(occurrences_frame)
    if not occurrences:
        return empty_reader_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    module_by_path = _module_by_path(modules_frame)
    def_info_by_symbol = _definitions_by_symbol(occurrences)
    def_path_by_symbol = {symbol: info[0] for symbol, info in def_info_by_symbol.items()}
    edges = build_use_edges(occurrences, def_path_by_symbol, module_by_path)
    if not edges:
        return empty_reader_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    use_lines_by_symbol_path = _reference_lines_by_symbol_path(occurrences)
    goid_spans_by_path = _goid_spans_by_path(goids_frame)
    edges = _attach_goids(edges, def_info_by_symbol, use_lines_by_symbol_path, goid_spans_by_path)
    rows = (dataclasses.asdict(row) for row in edges_to_rows(edges))
    reader, _ = record_batch_reader_for_rows(SYMBOL_USE_EDGES_TABLE_KEY, rows)
    return reader


def symbol_use_edges_existing(env: BuildEnv) -> InferableTabularInput:
    """Load symbol use edges from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing symbol use edges.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=SYMBOL_USE_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def symbol_use_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for symbol use edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for symbol use edges.
    """
    _ = env
    return empty_reader_for_table(SYMBOL_USE_EDGES_TABLE_KEY)


_MODULE = sys.modules[__name__]
_SYMBOL_USES_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=SYMBOL_USES_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SYMBOL_USE_EDGES_TABLE_KEY,
            base_node="symbol_use_edges",
            save_spec=DatasetSaveSpec(table_key=SYMBOL_USE_EDGES_TABLE_KEY),
            node_name="symbol_use_edges__table",
        ),
    ),
    table_materializations_node="symbol_uses__table_materializations",
    anchor_node_name="t__symbol_uses",
)
attach_table_target_template(_MODULE, spec=_SYMBOL_USES_TABLE_TARGET_SPEC)
symbol_use_edges__table = _MODULE.symbol_use_edges__table
symbol_uses__table_materializations = _MODULE.symbol_uses__table_materializations
t__symbol_uses = _MODULE.t__symbol_uses


__all__ = [
    "SYMBOL_USES_TARGET_NAME",
    "SYMBOL_USE_EDGES_TABLE_KEY",
    "symbol_use_edges__table",
    "symbol_use_edges_compute",
    "symbol_use_edges_empty",
    "symbol_use_edges_existing",
    "symbol_uses__table_materializations",
    "t__symbol_uses",
]
