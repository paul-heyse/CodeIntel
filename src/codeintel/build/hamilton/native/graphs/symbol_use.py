"""Symbol use relation sources for graph targets."""

from __future__ import annotations

import dataclasses

import polars as pl

from codeintel.build.graphs.compute.symbols import (
    SymbolOccurrence,
    SymbolUseEdge,
    build_use_edges,
    edges_to_rows,
    parse_symbol_roles,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import empty_frame_for_table
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_dataset,
)
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_lazyframe
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.core.data_models.ids import normalize_decimal_id

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SYMBOL_USES_TARGET_NAME = "symbol_uses"
SYMBOL_USE_EDGES_TABLE_KEY = "graph.symbol_use_edges"

SYMBOL_USES_SAVE_CONTEXT = SaverContext(domain="graphs", target=SYMBOL_USES_TARGET_NAME)


def _module_by_path(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_by_path[path] = module
    return module_by_path


def _goid_spans_by_path(
    goids_frame: pl.DataFrame,
) -> dict[str, list[tuple[int, int, int]]]:
    spans_by_path: dict[str, list[tuple[int, int, int]]] = {}
    for row in goids_frame.iter_rows(named=True):
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        goid_value = normalize_decimal_id(row.get("goid_h128"))
        if goid_value is None:
            continue
        start_line = row.get("start_line")
        if not isinstance(start_line, int):
            continue
        end_line = row.get("end_line")
        resolved_end = end_line if isinstance(end_line, int) else start_line
        spans_by_path.setdefault(rel_path, []).append((start_line, resolved_end, int(goid_value)))
    for spans in spans_by_path.values():
        spans.sort(key=lambda span: (span[1] - span[0], span[0], span[2]))
    return spans_by_path


def _match_goid(spans: list[tuple[int, int, int]], line: int) -> int | None:
    for start, end, goid in spans:
        if start <= line <= end:
            return goid
    return None


def _symbol_occurrences(occurrences_frame: pl.DataFrame) -> list[SymbolOccurrence]:
    occurrences: list[SymbolOccurrence] = []
    for row in occurrences_frame.iter_rows(named=True):
        symbol = row.get("symbol")
        rel_path = row.get("rel_path")
        start_line = row.get("start_line")
        if not isinstance(symbol, str) or not isinstance(rel_path, str):
            continue
        if not isinstance(start_line, int):
            continue
        roles = parse_symbol_roles(row.get("roles"))
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
    goid_spans_by_path: dict[str, list[tuple[int, int, int]]],
) -> list[SymbolUseEdge]:
    updated: list[SymbolUseEdge] = []
    for edge in edges:
        def_info = def_map.get(edge.symbol)
        def_goid: int | None = None
        if def_info is not None:
            def_spans = goid_spans_by_path.get(def_info[0], [])
            def_goid = _match_goid(def_spans, def_info[1])
        use_spans = goid_spans_by_path.get(edge.use_path, [])
        use_line = use_lines_by_symbol_path.get((edge.symbol, edge.use_path))
        use_goid = _match_goid(use_spans, use_line) if use_line is not None else None
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
) -> TabularFrame:
    """Build symbol use edges from SCIP occurrences and GOID spans.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for computed symbol use edges.
    """
    occurrences_frame = tabular_to_lazyframe(q__core__scip_occurrences).collect()
    if occurrences_frame.is_empty():
        return empty_frame_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    modules_frame = tabular_to_lazyframe(q__core__modules).collect()
    goids_frame = tabular_to_lazyframe(q__core__goids).collect()
    occurrences = _symbol_occurrences(occurrences_frame)
    if not occurrences:
        return empty_frame_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    module_by_path = _module_by_path(modules_frame)
    def_info_by_symbol = _definitions_by_symbol(occurrences)
    def_path_by_symbol = {symbol: info[0] for symbol, info in def_info_by_symbol.items()}
    edges = build_use_edges(occurrences, def_path_by_symbol, module_by_path)
    if not edges:
        return empty_frame_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    use_lines_by_symbol_path = _reference_lines_by_symbol_path(occurrences)
    goid_spans_by_path = _goid_spans_by_path(goids_frame)
    edges = _attach_goids(edges, def_info_by_symbol, use_lines_by_symbol_path, goid_spans_by_path)
    rows = edges_to_rows(edges)
    frame = pl.DataFrame([dataclasses.asdict(row) for row in rows], orient="row")
    return frame.lazy().select(
        [
            "symbol",
            "def_path",
            "use_path",
            "same_file",
            "same_module",
            "def_goid_h128",
            "use_goid_h128",
        ]
    )


def symbol_use_edges_existing(env: BuildEnv) -> TabularFrame:
    """Load symbol use edges from the dataset snapshot.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for existing symbol use edges.
    """
    return load_snapshot_lazyframe(
        env=env,
        table_key=SYMBOL_USE_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def symbol_use_edges_empty(env: BuildEnv) -> TabularFrame:
    """Return an empty frame for symbol use edges.

    Returns
    -------
    polars.LazyFrame
        Empty LazyFrame for symbol use edges.
    """
    _ = env
    return empty_frame_for_table(SYMBOL_USE_EDGES_TABLE_KEY)


@save_dataset(
    context=SYMBOL_USES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SYMBOL_USE_EDGES_TABLE_KEY),
)
@tag_dataset(domain="graphs", target=SYMBOL_USES_TARGET_NAME, table_key=SYMBOL_USE_EDGES_TABLE_KEY)
def symbol_use_edges__table(symbol_use_edges: pl.LazyFrame) -> pl.LazyFrame:
    """Persist symbol use edges.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for symbol use edges.
    """
    return symbol_use_edges


symbol_uses__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=SYMBOL_USES_TARGET_NAME,
    table_keys=(SYMBOL_USE_EDGES_TABLE_KEY,),
    node_name="symbol_uses__table_materializations",
)


@codeintel_target(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
def t__symbol_uses(
    env: BuildEnv,
    catalog: DagCatalog,
    symbol_uses__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize symbol_uses target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the symbol_uses target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SYMBOL_USES_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=symbol_uses__table_materializations,
    )


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
