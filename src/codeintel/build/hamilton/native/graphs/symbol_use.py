"""Symbol use relation sources for graph targets."""

from __future__ import annotations

import dataclasses
import sys

import pyarrow as pa

from codeintel.build.graphs.assembly import tabular_to_table
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
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import table_to_reader
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_reader
from codeintel.build.tabular.plan_ops import Plan, materialize_plan
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.intervals.span_resolver import SpanResolver

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SYMBOL_USES_TARGET_NAME = "symbol_uses"
SYMBOL_USE_EDGES_TABLE_KEY = "graph.symbol_use_edges"
SYMBOL_USE_EDGES_SORT_KEYS: tuple[SortKey, ...] = (
    ("repo", "ascending"),
    ("commit", "ascending"),
    ("symbol", "ascending"),
    ("def_path", "ascending"),
    ("use_path", "ascending"),
)


def _module_by_path(modules_table: pa.Table) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    if modules_table.num_rows == 0:
        return module_by_path
    if not {"path", "module"}.issubset(set(modules_table.column_names)):
        return module_by_path
    filtered = _python_modules_table(modules_table)
    for row in iter_rows(filtered):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_by_path[path] = module
    return module_by_path


def _goid_resolver(
    goids_table: pa.Table,
) -> SpanResolver[int]:
    resolver = SpanResolver.for_lines(path_normalizer=lambda value: value)
    if goids_table.num_rows == 0 or "rel_path" not in goids_table.column_names:
        return resolver
    filtered = _filtered_goids_table(goids_table)
    for row in iter_rows(filtered):
        rel_path = row.get("rel_path")
        goid_raw = row.get("goid_h128")
        start_line = row.get("start_line")
        end_line = row.get("end_line")
        if not isinstance(rel_path, str):
            continue
        goid_value = normalize_decimal_id(goid_raw)
        if goid_value is None:
            continue
        if not isinstance(start_line, int):
            continue
        resolver.add_span(
            rel_path,
            start_line,
            end_line if isinstance(end_line, int) else None,
            int(goid_value),
        )
    return resolver


def _match_goid(resolver: SpanResolver[int], rel_path: str, line: int) -> int | None:
    match = resolver.resolve(rel_path, line, line)
    if match.match_kind == "NONE":
        return None
    if match.payload is None:
        return None
    return int(match.payload)


def _symbol_occurrences(occurrences_table: pa.Table) -> list[SymbolOccurrence]:
    occurrences: list[SymbolOccurrence] = []
    if occurrences_table.num_rows == 0:
        return occurrences
    required = {"symbol", "rel_path", "start_line"}
    if not required.issubset(set(occurrences_table.column_names)):
        return occurrences
    filtered = _filtered_occurrences_table(occurrences_table)
    for row in iter_rows(filtered):
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
    goid_resolver: SpanResolver[int],
) -> list[SymbolUseEdge]:
    updated: list[SymbolUseEdge] = []
    for edge in edges:
        def_info = def_map.get(edge.symbol)
        def_goid: int | None = None
        if def_info is not None:
            def_goid = _match_goid(goid_resolver, def_info[0], def_info[1])
        use_line = use_lines_by_symbol_path.get((edge.symbol, edge.use_path))
        use_goid = (
            _match_goid(goid_resolver, edge.use_path, use_line) if use_line is not None else None
        )
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


def _symbol_use_tables(
    q__core__scip_occurrences: InferableTabularInput,
    q__core__modules: InferableTabularInput,
    q__core__goids: InferableTabularInput,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    occurrences_table = tabular_to_table(q__core__scip_occurrences).select(
        ["symbol", "rel_path", "start_line", "roles"]
    )
    modules_table = tabular_to_table(q__core__modules).select(["path", "module"])
    goids_table = tabular_to_table(q__core__goids).select(
        ["rel_path", "goid_h128", "start_line", "end_line"]
    )
    return occurrences_table, modules_table, goids_table


def _definition_maps(
    occurrences: list[SymbolOccurrence],
) -> tuple[dict[str, tuple[str, int]], dict[str, str]]:
    def_info_by_symbol = _definitions_by_symbol(occurrences)
    def_path_by_symbol = {symbol: info[0] for symbol, info in def_info_by_symbol.items()}
    return def_info_by_symbol, def_path_by_symbol


def symbol_use_edges_compute(
    env: BuildEnv,
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
    occurrences_table, modules_table, goids_table = _symbol_use_tables(
        q__core__scip_occurrences,
        q__core__modules,
        q__core__goids,
    )
    if occurrences_table.num_rows == 0:
        return empty_table_for_table(SYMBOL_USE_EDGES_TABLE_KEY)
    occurrences = _symbol_occurrences(occurrences_table)
    if not occurrences:
        return empty_table_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    module_by_path = _module_by_path(modules_table)
    def_info_by_symbol, def_path_by_symbol = _definition_maps(occurrences)
    edges = build_use_edges(occurrences, def_path_by_symbol, module_by_path)
    if not edges:
        return empty_table_for_table(SYMBOL_USE_EDGES_TABLE_KEY)

    use_lines_by_symbol_path = _reference_lines_by_symbol_path(occurrences)
    goid_resolver = _goid_resolver(goids_table)
    edges = _attach_goids(edges, def_info_by_symbol, use_lines_by_symbol_path, goid_resolver)
    rows = (dataclasses.asdict(row) for row in edges_to_rows(edges, env.repo, env.commit))
    table, _ = table_for_rows(SYMBOL_USE_EDGES_TABLE_KEY, rows)
    reader = table_to_reader(table, batch_size=None)
    result = finalize_reader(
        reader,
        spec=FinalizeSpec(
            table_key=SYMBOL_USE_EDGES_TABLE_KEY,
            mode="strict",
            order_by=SYMBOL_USE_EDGES_SORT_KEYS,
            target_name=SYMBOL_USES_TARGET_NAME,
        ),
    )
    return result.good


def _filtered_occurrences_table(occurrences_table: pa.Table) -> pa.Table:
    required = {"symbol", "rel_path", "start_line"}
    if occurrences_table.num_rows == 0 or not required.issubset(
        set(occurrences_table.column_names)
    ):
        return occurrences_table
    plan = Plan.table(occurrences_table).project(
        {
            "symbol": E.cast(E.field("symbol"), "string"),
            "rel_path": E.cast(E.field("rel_path"), "string"),
            "start_line": E.field("start_line"),
            "roles": E.field("roles"),
        }
    )
    plan = plan.filter(
        E.and_(
            _non_empty_expr("symbol"),
            _non_empty_expr("rel_path"),
            E.is_valid("start_line"),
        )
    )
    return materialize_plan(plan, use_threads=True)


def _filtered_goids_table(goids_table: pa.Table) -> pa.Table:
    required = {"rel_path", "goid_h128", "start_line"}
    if goids_table.num_rows == 0 or not required.issubset(set(goids_table.column_names)):
        return goids_table
    projection = {
        "rel_path": E.cast(E.field("rel_path"), "string"),
        "goid_h128": E.field("goid_h128"),
        "start_line": E.field("start_line"),
        "end_line": E.field("end_line"),
    }
    plan = Plan.table(goids_table).project(projection)
    plan = plan.filter(
        E.and_(
            _non_empty_expr("rel_path"),
            E.is_valid("goid_h128"),
            E.is_valid("start_line"),
        )
    )
    return materialize_plan(plan, use_threads=True)


def _python_modules_table(modules_table: pa.Table) -> pa.Table:
    required = {"path", "module"}
    if modules_table.num_rows == 0 or not required.issubset(set(modules_table.column_names)):
        return modules_table
    projection = {
        "path": E.cast(E.field("path"), "string"),
        "module": E.cast(E.field("module"), "string"),
    }
    if "language" in modules_table.column_names:
        projection["language"] = E.cast(E.field("language"), "string")
    plan = Plan.table(modules_table).project(projection)
    exprs: list[Expression] = [_non_empty_expr("path"), _non_empty_expr("module")]
    if "language" in projection:
        exprs.append(_python_language_expr())
    plan = plan.filter(E.and_(*exprs))
    return materialize_plan(plan, use_threads=True)


def _python_language_expr() -> Expression:
    return E.or_(E.is_null("language"), E.field("language") == E.scalar("python"))


def _non_empty_expr(name: str) -> Expression:
    return E.and_(E.is_valid(name), E.field(name) != E.scalar(""))


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
    return empty_table_for_table(SYMBOL_USE_EDGES_TABLE_KEY)


_MODULE = sys.modules[__name__]
_SYMBOL_USES_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext(
        domain="graphs",
        target_name=SYMBOL_USES_TARGET_NAME,
        table_key=SYMBOL_USE_EDGES_TABLE_KEY,
        base_node="symbol_use_edges",
        node_name="symbol_use_edges__table",
        table_materializations_node="symbol_uses__table_materializations",
    )
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
