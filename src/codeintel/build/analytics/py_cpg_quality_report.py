"""Python CPG quality report helpers for post-run analytics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.expr_vocab import E
from codeintel.build.tabular.plan_ops import HashJoinSpec, Plan, materialize_plan
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_int

PY_CPG_QUALITY_REPORT_TABLE_KEY = "analytics.py_cpg_quality_report"


@dataclass(frozen=True, slots=True)
class PyCpgQualityInputs:
    """Scoped tables required to compute Python CPG quality metrics."""

    instructions: pa.Table
    scopes: pa.Table
    blocks: pa.Table
    inspect_objects: pa.Table
    cfg_edges: pa.Table
    defuse_events: pa.Table
    cpg_nodes: pa.Table
    cpg_edges: pa.Table


@dataclass(frozen=True, slots=True)
class _AnchorRate:
    total: int
    anchored: int

    @property
    def rate(self) -> float | None:
        if self.total == 0:
            return None
        return self.anchored / self.total


@dataclass(frozen=True, slots=True)
class _CfgReachability:
    total_blocks: int
    reachable_blocks: int

    @property
    def rate(self) -> float | None:
        if self.total_blocks == 0:
            return None
        return self.reachable_blocks / self.total_blocks


@dataclass(frozen=True, slots=True)
class _DefuseCoverage:
    event_count: int
    edge_count: int

    @property
    def rate(self) -> float | None:
        if self.event_count == 0:
            return None
        return self.edge_count / self.event_count


@dataclass(frozen=True, slots=True)
class _CpgEdgeScan:
    defuse_edge_count: int
    inspect_anchor_ids: set[int]
    symbol_edge_count: int
    external_symbol_edge_count: int
    binding_resolution_edge_count: int
    binding_unresolved_edge_count: int

    @property
    def external_symbol_rate(self) -> float | None:
        return _rate(self.external_symbol_edge_count, self.symbol_edge_count)

    @property
    def binding_unresolved_rate(self) -> float | None:
        return _rate(self.binding_unresolved_edge_count, self.binding_resolution_edge_count)


def build_py_cpg_quality_report_rows(
    *,
    repo: str,
    commit: str,
    run_id: str,
    inputs: PyCpgQualityInputs,
) -> list[dict[str, object]]:
    """Build run-level Python CPG quality metrics from scoped tables.

    Returns
    -------
    list[dict[str, object]]
        Single-row payload for analytics.py_cpg_quality_report.
    """
    instruction_rate = _anchor_rate(inputs.instructions, anchor_column="span_start_byte")
    symtable_rate = _anchor_rate(inputs.scopes, anchor_column="anchor_ast_node_id")
    cfg_rate = _cfg_reachability(inputs.blocks, inputs.cfg_edges)
    defuse_event_count = _defuse_event_count(inputs.defuse_events)
    edge_scan = _scan_cpg_edges(inputs.cpg_edges, node_table=inputs.cpg_nodes)
    defuse_rate = _DefuseCoverage(
        event_count=defuse_event_count,
        edge_count=edge_scan.defuse_edge_count,
    )
    inspect_total = _count_rows(inputs.inspect_objects)
    inspect_rate = _AnchorRate(total=inspect_total, anchored=len(edge_scan.inspect_anchor_ids))

    return [
        {
            "repo": repo,
            "commit": commit,
            "run_id": run_id,
            "instruction_count": instruction_rate.total,
            "instruction_anchored_count": instruction_rate.anchored,
            "instruction_anchor_rate": instruction_rate.rate,
            "sym_scope_count": symtable_rate.total,
            "sym_scope_anchored_count": symtable_rate.anchored,
            "sym_scope_anchor_rate": symtable_rate.rate,
            "cfg_block_count": cfg_rate.total_blocks,
            "cfg_reachable_block_count": cfg_rate.reachable_blocks,
            "cfg_reachability_rate": cfg_rate.rate,
            "defuse_event_count": defuse_rate.event_count,
            "defuse_edge_count": defuse_rate.edge_count,
            "defuse_resolution_rate": defuse_rate.rate,
            "inspect_object_count": inspect_rate.total,
            "inspect_anchored_count": inspect_rate.anchored,
            "inspect_anchor_rate": inspect_rate.rate,
            "symbol_edge_count": edge_scan.symbol_edge_count,
            "external_symbol_edge_count": edge_scan.external_symbol_edge_count,
            "external_symbol_edge_rate": edge_scan.external_symbol_rate,
            "binding_resolution_edge_count": edge_scan.binding_resolution_edge_count,
            "binding_unresolved_edge_count": edge_scan.binding_unresolved_edge_count,
            "binding_unresolved_edge_rate": edge_scan.binding_unresolved_rate,
            "created_at": datetime.now(UTC),
        }
    ]


def _reader_rows(table: pa.Table) -> Iterable[dict[str, object]]:
    for batch in table.to_batches():
        if batch.num_rows > 0:
            yield from iter_rows(batch)


def _reader_has_columns(table: pa.Table, columns: Iterable[str]) -> bool:
    return set(columns).issubset(table.schema.names)


def _count_rows(table: pa.Table) -> int:
    return table.num_rows


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _anchor_rate(
    table: pa.Table,
    *,
    anchor_column: str,
) -> _AnchorRate:
    total = table.num_rows
    if anchor_column not in table.schema.names:
        return _AnchorRate(total=total, anchored=0)
    plan = Plan.table(table)
    plan = plan.project(
        {
            "row_marker": E.scalar(1),
            "anchor_value": E.field(anchor_column),
        }
    )
    plan = plan.aggregate(
        keys=[],
        aggregates=[
            ("row_marker", "count", None, "row_count"),
            ("anchor_value", "count", None, "anchored_count"),
        ],
    )
    aggregated = materialize_plan(plan, use_threads=True)
    row = next(iter_rows(aggregated, ("row_count", "anchored_count")), None)
    if row is None:
        return _AnchorRate(total=0, anchored=0)
    total_value = row.get("row_count")
    anchored_value = row.get("anchored_count")
    return _AnchorRate(
        total=coerce_int(total_value, ctx="row_count") if total_value is not None else 0,
        anchored=(
            coerce_int(anchored_value, ctx="anchored_count") if anchored_value is not None else 0
        ),
    )


def _cfg_reachability(
    blocks_reader: pa.Table,
    edges_reader: pa.Table,
) -> _CfgReachability:
    blocks_by_unit = _blocks_by_unit(blocks_reader)
    edges_by_unit = _edges_by_unit(edges_reader)
    total_blocks = 0
    reachable_blocks = 0
    for code_unit_id, blocks in blocks_by_unit.items():
        total_blocks += len(blocks)
        entry_block = _entry_block_id(blocks)
        if entry_block is None:
            continue
        adjacency = _adjacency_for_unit(edges_by_unit.get(code_unit_id, []))
        reachable_blocks += _reachable_count(entry_block, adjacency)
    return _CfgReachability(total_blocks=total_blocks, reachable_blocks=reachable_blocks)


def _blocks_by_unit(
    blocks_reader: pa.Table,
) -> dict[str, list[tuple[str, int | None, int | None]]]:
    if not _reader_has_columns(
        blocks_reader,
        ("code_unit_id", "block_id", "start_offset", "first_instr_index"),
    ):
        return {}
    blocks_by_unit: dict[str, list[tuple[str, int | None, int | None]]] = {}
    for row in _reader_rows(blocks_reader):
        code_unit_id = row.get("code_unit_id")
        block_id = row.get("block_id")
        if not isinstance(code_unit_id, str) or not isinstance(block_id, str):
            continue
        start_offset = row.get("start_offset")
        first_instr_index = row.get("first_instr_index")
        first_index = first_instr_index if isinstance(first_instr_index, int) else None
        blocks_by_unit.setdefault(code_unit_id, []).append(
            (
                block_id,
                start_offset if isinstance(start_offset, int) else None,
                first_index,
            )
        )
    return blocks_by_unit


def _edges_by_unit(edges_reader: pa.Table) -> dict[str, list[tuple[str, str]]]:
    if not _reader_has_columns(
        edges_reader,
        ("code_unit_id", "src_block_id", "dst_block_id"),
    ):
        return {}
    edges_by_unit: dict[str, list[tuple[str, str]]] = {}
    for row in _reader_rows(edges_reader):
        code_unit_id = row.get("code_unit_id")
        src_block = row.get("src_block_id")
        dst_block = row.get("dst_block_id")
        if not isinstance(code_unit_id, str) or not isinstance(src_block, str):
            continue
        if not isinstance(dst_block, str):
            continue
        edges_by_unit.setdefault(code_unit_id, []).append((src_block, dst_block))
    return edges_by_unit


def _entry_block_id(blocks: list[tuple[str, int | None, int | None]]) -> str | None:
    if not blocks:
        return None
    entry_block, *_ = min(
        blocks,
        key=lambda item: (
            item[1] if item[1] is not None else 1_000_000_000,
            item[2] if isinstance(item[2], int) else 1_000_000_000,
        ),
    )
    return entry_block


def _adjacency_for_unit(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for src, dst in edges:
        adjacency.setdefault(src, set()).add(dst)
    return adjacency


def _reachable_count(entry_block: str, adjacency: dict[str, set[str]]) -> int:
    seen: set[str] = set()
    stack = [entry_block]
    while stack:
        block_id = stack.pop()
        if block_id in seen:
            continue
        seen.add(block_id)
        stack.extend(sorted(adjacency.get(block_id, set())))
    return len(seen)


def _defuse_event_count(reader: pa.Table) -> int:
    if not _reader_has_columns(reader, ("event_kind", "space")):
        return 0
    plan = Plan.table(reader)
    plan = plan.project({"event_kind": E.field("event_kind"), "space": E.field("space")})
    plan = plan.filter(
        E.and_(
            E.in_("event_kind", ["DEF", "USE"]),
            E.in_("space", ["local", "free", "global"]),
        )
    )
    return _count_from_plan(plan, count_column="event_kind")


def _scan_cpg_edges(
    reader: pa.Table,
    *,
    node_table: pa.Table | None = None,
) -> _CpgEdgeScan:
    has_edge_kind = "edge_kind" in reader.schema.names
    if not has_edge_kind:
        return _CpgEdgeScan(0, set(), 0, 0, 0, 0)
    has_src = "src_cpg_node_id" in reader.schema.names
    has_dst = "dst_cpg_node_id" in reader.schema.names
    has_layer = "edge_layer" in reader.schema.names
    defuse_edge_count = _defuse_edge_count(reader)
    inspect_anchor_ids = _inspect_anchor_ids(reader, has_src=has_src)
    symbol_edge_count, external_symbol_edge_count = _symbol_edge_counts(
        reader,
        has_dst=has_dst,
        has_layer=has_layer,
        node_table=node_table,
    )
    binding_resolution_edge_count, binding_unresolved_edge_count = _binding_edge_counts(
        reader,
        has_src=has_src,
        has_dst=has_dst,
        node_table=node_table,
    )
    return _CpgEdgeScan(
        defuse_edge_count=defuse_edge_count,
        inspect_anchor_ids=inspect_anchor_ids,
        symbol_edge_count=symbol_edge_count,
        external_symbol_edge_count=external_symbol_edge_count,
        binding_resolution_edge_count=binding_resolution_edge_count,
        binding_unresolved_edge_count=binding_unresolved_edge_count,
    )


def _defuse_edge_count(reader: pa.Table) -> int:
    if "extras_kv" not in reader.schema.names or "edge_kind" not in reader.schema.names:
        return 0
    space_expr = E.field(("extras_kv", "space"))
    plan = Plan.table(reader)
    plan = plan.project({"edge_kind": E.field("edge_kind"), "space": space_expr})
    plan = plan.filter(
        E.and_(
            E.in_("edge_kind", ["DEFINES_BINDING", "USES_BINDING"]),
            E.in_(space_expr, ["local", "free", "global"]),
        )
    )
    return _count_from_plan(plan, count_column="edge_kind")


def _inspect_anchor_ids(reader: pa.Table, *, has_src: bool) -> set[int]:
    if not has_src:
        return set()
    plan = Plan.table(reader)
    plan = plan.filter(
        E.and_(
            E.field("edge_kind") == E.scalar("INSPECT_ANCHORS_AST"),
            E.is_valid("src_cpg_node_id"),
        )
    )
    plan = plan.project({"src_id": E.field("src_cpg_node_id")})
    plan = plan.aggregate(keys=[E.field("src_id")], aggregates=[])
    table = materialize_plan(plan, use_threads=True)
    anchor_ids: set[int] = set()
    for row in iter_rows(table, ("src_id",)):
        src_id = normalize_decimal_id(row.get("src_id"))
        if src_id is not None:
            anchor_ids.add(src_id)
    return anchor_ids


def _symbol_edge_counts(
    reader: pa.Table,
    *,
    has_dst: bool,
    has_layer: bool,
    node_table: pa.Table | None,
) -> tuple[int, int]:
    if node_table is None or node_table.num_rows == 0:
        return 0, 0
    if not has_dst or not has_layer:
        return 0, 0
    if not _reader_has_columns(node_table, ("cpg_node_id", "node_kind")):
        return 0, 0
    edges_plan = Plan.table(reader)
    edges_plan = edges_plan.filter(
        E.and_(
            E.field("edge_layer") == E.scalar("SYMBOL"),
            E.is_valid("dst_cpg_node_id"),
        )
    )
    edges_plan = edges_plan.project({"dst_id": E.field("dst_cpg_node_id")})

    nodes_plan = Plan.table(node_table)
    nodes_plan = nodes_plan.filter(
        E.and_(
            E.is_valid("cpg_node_id"),
            E.in_("node_kind", ["SCIP_SYMBOL", "SCIP_SYMBOL_EXTERNAL"]),
        )
    )
    nodes_plan = nodes_plan.project(
        {
            "dst_id": E.field("cpg_node_id"),
            "node_kind": E.field("node_kind"),
        }
    )
    joined = edges_plan.hash_join(
        right=nodes_plan,
        spec=HashJoinSpec(
            left_keys=["dst_id"],
            right_keys=["dst_id"],
            how="inner",
            left_output=["dst_id"],
            right_output=["node_kind"],
        ),
    )
    joined = joined.aggregate(
        keys=[E.field("node_kind")],
        aggregates=[("node_kind", "count", None, "edge_count")],
    )
    aggregated = materialize_plan(joined, use_threads=True)
    symbol_edge_count = 0
    external_symbol_edge_count = 0
    for row in iter_rows(aggregated, ("node_kind", "edge_count")):
        kind = row.get("node_kind")
        count_value = row.get("edge_count")
        if kind is None or count_value is None:
            continue
        count = coerce_int(count_value, ctx="symbol_edge_count")
        kind_text = str(kind)
        if kind_text in {"SCIP_SYMBOL", "SCIP_SYMBOL_EXTERNAL"}:
            symbol_edge_count += count
        if kind_text == "SCIP_SYMBOL_EXTERNAL":
            external_symbol_edge_count += count
    return symbol_edge_count, external_symbol_edge_count


def _binding_edge_counts(
    reader: pa.Table,
    *,
    has_src: bool,
    has_dst: bool,
    node_table: pa.Table | None,
) -> tuple[int, int]:
    if node_table is None or node_table.num_rows == 0:
        return 0, 0
    if not has_src or not has_dst:
        return 0, 0
    if not _reader_has_columns(node_table, ("cpg_node_id", "node_kind")):
        return 0, 0
    edges_plan = Plan.table(reader)
    edges_plan = edges_plan.filter(
        E.and_(
            E.field("edge_kind") == E.scalar("RESOLVES_TO"),
            E.is_valid("src_cpg_node_id"),
            E.is_valid("dst_cpg_node_id"),
        )
    )
    edges_plan = edges_plan.project(
        {
            "src_id": E.field("src_cpg_node_id"),
            "dst_id": E.field("dst_cpg_node_id"),
        }
    )

    src_nodes = Plan.table(node_table)
    src_nodes = src_nodes.filter(
        E.and_(
            E.is_valid("cpg_node_id"),
            E.field("node_kind") == E.scalar("BINDING"),
        )
    )
    src_nodes = src_nodes.project({"src_id": E.field("cpg_node_id")})

    joined_src = edges_plan.hash_join(
        right=src_nodes,
        spec=HashJoinSpec(
            left_keys=["src_id"],
            right_keys=["src_id"],
            how="inner",
            left_output=["src_id", "dst_id"],
            right_output=[],
        ),
    )
    binding_resolution_edge_count = _count_from_plan(joined_src, count_column="src_id")

    dst_nodes = Plan.table(node_table)
    dst_nodes = dst_nodes.filter(
        E.and_(
            E.is_valid("cpg_node_id"),
            E.field("node_kind") == E.scalar("BINDING_UNRESOLVED"),
        )
    )
    dst_nodes = dst_nodes.project({"dst_id": E.field("cpg_node_id")})

    joined_dst = joined_src.hash_join(
        right=dst_nodes,
        spec=HashJoinSpec(
            left_keys=["dst_id"],
            right_keys=["dst_id"],
            how="inner",
            left_output=["dst_id"],
            right_output=[],
        ),
    )
    binding_unresolved_edge_count = _count_from_plan(joined_dst, count_column="dst_id")
    return binding_resolution_edge_count, binding_unresolved_edge_count


def _count_from_plan(plan: Plan, *, count_column: str) -> int:
    aggregated = plan.aggregate(
        keys=[],
        aggregates=[(count_column, "count", None, "row_count")],
    )
    table = materialize_plan(aggregated, use_threads=True)
    return _count_from_table(table, column="row_count")


def _count_from_table(table: pa.Table, *, column: str) -> int:
    row = next(iter_rows(table, (column,)), None)
    if row is None:
        return 0
    value = row.get(column)
    return coerce_int(value, ctx=column) if value is not None else 0


__all__ = [
    "PY_CPG_QUALITY_REPORT_TABLE_KEY",
    "PyCpgQualityInputs",
    "build_py_cpg_quality_report_rows",
]
