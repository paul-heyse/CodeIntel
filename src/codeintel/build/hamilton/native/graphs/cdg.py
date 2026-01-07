"""Control dependence graph sources for graph targets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.tabular.arrow_ops import (
    align_table_to_contract,
    dedupe_table_for_table,
    emit_alignment_report,
    iter_rows,
)
from codeintel.build.tabular.compute_helpers import safe_filter
from codeintel.build.tabular.compute_masks import (
    and_kleene,
    is_valid_expr,
    is_valid_mask,
    non_empty_string_expr,
    non_empty_string_mask,
)
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table

CDG_EDGES_TABLE_KEY = "graph.cdg_edges"
CDG_TARGET_NAME = "cdg"

_EXPR_TYPE = getattr(pc, "Expression", None)


@dataclass(frozen=True)
class _CdgEdgeRow:
    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    via_succ_block_id: str
    edge_kind: str
    via_edge_kind: str | None


@dataclass(frozen=True, slots=True)
class _CdgEdgeContext:
    function_goid: int
    post: dict[int, int]
    inv_idx: dict[int, int]
    block_id_by_idx: dict[int, str]
    edge_kind_by_pair: dict[tuple[int, int], str | None]
    exit_id: int
    synthetic_exit: int | None


class _PostdominatorConvergenceError(RuntimeError):
    """Raised when postdominator computation fails to converge."""

    def __init__(self) -> None:
        super().__init__("Postdominator computation did not converge.")


def _bit_iter(mask: int) -> Iterable[int]:
    while mask:
        lsb = mask & -mask
        index = lsb.bit_length() - 1
        yield index
        mask ^= lsb


def _compute_postdom_bitsets(
    node_ids: list[int],
    succ: dict[int, list[int]],
    exit_id: int,
) -> tuple[dict[int, int], dict[int, int]]:
    idx: dict[int, int] = {node_id: i for i, node_id in enumerate(node_ids)}
    total = len(node_ids)
    all_mask = (1 << total) - 1
    exit_bit = 1 << idx[exit_id]

    post: dict[int, int] = dict.fromkeys(node_ids, all_mask)
    post[exit_id] = exit_bit

    for _ in range(1000):
        changed = False
        for node_id in node_ids:
            if node_id == exit_id:
                continue
            successors = succ.get(node_id)
            if not successors:
                successors = [exit_id]
            intersection = all_mask
            for succ_id in successors:
                intersection &= post[succ_id]
            new_mask = (1 << idx[node_id]) | intersection
            if new_mask != post[node_id]:
                post[node_id] = new_mask
                changed = True
        if not changed:
            break
    else:
        raise _PostdominatorConvergenceError

    return post, idx


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _block_indexes(blocks: Sequence[Mapping[str, object]]) -> tuple[dict[str, int], dict[int, str]]:
    block_idx_by_id: dict[str, int] = {}
    block_id_by_idx: dict[int, str] = {}
    if not blocks:
        return block_idx_by_id, block_id_by_idx
    for row in blocks:
        block_id = row.get("block_id")
        block_idx = _coerce_int(row.get("block_idx"))
        if not isinstance(block_id, str) or block_idx is None:
            continue
        block_idx_by_id[block_id] = block_idx
        block_id_by_idx[block_idx] = block_id
    return block_idx_by_id, block_id_by_idx


def _edge_indexes(
    edges: Sequence[Mapping[str, object]],
    block_idx_by_id: dict[str, int],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str | None]]:
    edges_idx: list[tuple[int, int]] = []
    edge_kind_by_pair: dict[tuple[int, int], str | None] = {}
    if not edges:
        return edges_idx, edge_kind_by_pair
    for row in edges:
        src_id = row.get("src_block_id")
        dst_id = row.get("dst_block_id")
        if not isinstance(src_id, str) or not isinstance(dst_id, str):
            continue
        src_idx = block_idx_by_id.get(src_id)
        dst_idx = block_idx_by_id.get(dst_id)
        if src_idx is None or dst_idx is None:
            continue
        edges_idx.append((src_idx, dst_idx))
        edge_kind_raw = row.get("edge_kind")
        edge_kind_by_pair[src_idx, dst_idx] = (
            edge_kind_raw if isinstance(edge_kind_raw, str) else None
        )
    return edges_idx, edge_kind_by_pair


def _build_successors(
    edges_idx: list[tuple[int, int]],
    node_ids: list[int],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    out_degree: dict[int, int] = dict.fromkeys(node_ids, 0)
    succ: dict[int, list[int]] = {}
    for src_idx, dst_idx in edges_idx:
        out_degree[src_idx] = out_degree.get(src_idx, 0) + 1
        succ.setdefault(src_idx, []).append(dst_idx)
    return out_degree, succ


def _ensure_exit_node(
    exits: list[int],
    node_ids: list[int],
    succ: dict[int, list[int]],
    edges_idx: list[tuple[int, int]],
    edge_kind_by_pair: dict[tuple[int, int], str | None],
) -> tuple[int, int | None]:
    exit_id = exits[0]
    synthetic_exit = None
    if len(exits) > 1:
        synthetic_exit = max(node_ids) + 1
        exit_id = synthetic_exit
        node_ids.append(synthetic_exit)
        for exit_node in exits:
            succ.setdefault(exit_node, []).append(synthetic_exit)
            edges_idx.append((exit_node, synthetic_exit))
            edge_kind_by_pair[exit_node, synthetic_exit] = None
    return exit_id, synthetic_exit


def _cdg_edge_rows_for_pair(
    context: _CdgEdgeContext,
    *,
    src_idx: int,
    dst_idx: int,
) -> list[_CdgEdgeRow]:
    if context.synthetic_exit is not None and dst_idx == context.synthetic_exit:
        return []
    diff = context.post[dst_idx] & (~context.post[src_idx])
    if diff == 0:
        return []
    rows: list[_CdgEdgeRow] = []
    for bit_index in _bit_iter(diff):
        controlled_idx = context.inv_idx[bit_index]
        if controlled_idx == context.exit_id:
            continue
        src_block_id = context.block_id_by_idx.get(src_idx)
        dst_block_id = context.block_id_by_idx.get(controlled_idx)
        via_block_id = context.block_id_by_idx.get(dst_idx)
        if src_block_id is None or dst_block_id is None or via_block_id is None:
            continue
        rows.append(
            _CdgEdgeRow(
                function_goid_h128=context.function_goid,
                src_block_id=src_block_id,
                dst_block_id=dst_block_id,
                via_succ_block_id=via_block_id,
                edge_kind="CDG",
                via_edge_kind=context.edge_kind_by_pair.get((src_idx, dst_idx)),
            )
        )
    return rows


def _cdg_edges_for_function(
    function_goid: int,
    blocks: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
) -> list[_CdgEdgeRow]:
    block_idx_by_id, block_id_by_idx = _block_indexes(blocks)
    if not block_idx_by_id:
        return []

    edges_idx, edge_kind_by_pair = _edge_indexes(edges, block_idx_by_id)
    if not edges_idx:
        return []

    node_ids = list(block_id_by_idx)
    out_degree, succ = _build_successors(edges_idx, node_ids)
    exits = [node_id for node_id in node_ids if out_degree.get(node_id, 0) == 0]
    if not exits:
        return []

    exit_id, synthetic_exit = _ensure_exit_node(
        exits,
        node_ids,
        succ,
        edges_idx,
        edge_kind_by_pair,
    )
    post, idx = _compute_postdom_bitsets(node_ids, succ, exit_id)
    inv_idx = {value: key for key, value in idx.items()}
    context = _CdgEdgeContext(
        function_goid=function_goid,
        post=post,
        inv_idx=inv_idx,
        block_id_by_idx=block_id_by_idx,
        edge_kind_by_pair=edge_kind_by_pair,
        exit_id=exit_id,
        synthetic_exit=synthetic_exit,
    )

    rows: list[_CdgEdgeRow] = []
    for src_idx, dst_idx in edges_idx:
        rows.extend(_cdg_edge_rows_for_pair(context, src_idx=src_idx, dst_idx=dst_idx))
    return rows


def _prefilter_cdg_blocks(blocks_table: pa.Table) -> pa.Table:
    if blocks_table.num_rows == 0:
        return blocks_table
    required = {"function_goid_h128", "block_id", "block_idx"}
    if not required.issubset(set(blocks_table.column_names)):
        return blocks_table
    try:
        if _EXPR_TYPE is not None:
            expr = (
                is_valid_expr("function_goid_h128")
                & is_valid_expr("block_id")
                & is_valid_expr("block_idx")
            )
            return safe_filter(blocks_table, expr)
        goid_mask = is_valid_mask(blocks_table.column("function_goid_h128"))
        block_id_mask = is_valid_mask(blocks_table.column("block_id"))
        block_idx_mask = is_valid_mask(blocks_table.column("block_idx"))
        mask = and_kleene(goid_mask, block_id_mask)
        mask = and_kleene(mask, block_idx_mask)
        return safe_filter(blocks_table, mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return blocks_table


def _prefilter_cdg_edges(edges_table: pa.Table) -> pa.Table:
    if edges_table.num_rows == 0:
        return edges_table
    required = {"function_goid_h128", "edge_kind"}
    if not required.issubset(set(edges_table.column_names)):
        return edges_table
    try:
        if _EXPR_TYPE is not None:
            expr = is_valid_expr("function_goid_h128") & non_empty_string_expr("edge_kind")
            return safe_filter(edges_table, expr)
        goid_mask = is_valid_mask(edges_table.column("function_goid_h128"))
        kind_mask = non_empty_string_mask(edges_table.column("edge_kind"))
        mask = and_kleene(goid_mask, kind_mask)
        return safe_filter(edges_table, mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return edges_table


def cdg_edges(
    q__graph__cfg_blocks: InferableTabularInput,
    q__graph__cfg_edges: InferableTabularInput,
) -> InferableTabularInput:
    """Build control dependence edges from CFG blocks/edges.

    Returns
    -------
    InferableTabularInput
        Arrow reader for graph.cdg_edges.
    """
    blocks_table = tabular_to_scoped_table(
        q__graph__cfg_blocks,
        columns=["function_goid_h128", "block_id", "block_idx"],
        scope=None,
        require_scope_columns=False,
    )
    edges_table = tabular_to_scoped_table(
        q__graph__cfg_edges,
        columns=["function_goid_h128", "src_block_id", "dst_block_id", "edge_kind"],
        scope=None,
        require_scope_columns=False,
    )
    blocks_table = _prefilter_cdg_blocks(blocks_table)
    edges_table = _prefilter_cdg_edges(edges_table)
    if blocks_table.num_rows == 0 or edges_table.num_rows == 0:
        return empty_table_for_table(CDG_EDGES_TABLE_KEY)

    blocks_by_goid: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in iter_rows(blocks_table):
        function_goid = _coerce_int(row.get("function_goid_h128"))
        if function_goid is None:
            continue
        blocks_by_goid[function_goid].append(row)
    edges_by_goid: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in iter_rows(edges_table):
        function_goid = _coerce_int(row.get("function_goid_h128"))
        if function_goid is None:
            continue
        edges_by_goid[function_goid].append(row)

    rows: list[dict[str, object]] = []
    for function_goid, blocks in blocks_by_goid.items():
        edges = edges_by_goid.get(function_goid)
        if not edges:
            continue
        rows.extend(
            {
                "function_goid_h128": row.function_goid_h128,
                "src_block_id": row.src_block_id,
                "dst_block_id": row.dst_block_id,
                "via_succ_block_id": row.via_succ_block_id,
                "edge_kind": row.edge_kind,
                "via_edge_kind": row.via_edge_kind,
            }
            for row in _cdg_edges_for_function(function_goid, blocks, edges)
        )

    if not rows:
        return empty_table_for_table(CDG_EDGES_TABLE_KEY)

    table = pa.Table.from_pylist(rows).select(
        [
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "via_succ_block_id",
            "edge_kind",
            "via_edge_kind",
        ]
    )
    table = dedupe_table_for_table(CDG_EDGES_TABLE_KEY, table)
    return align_table_to_contract(
        CDG_EDGES_TABLE_KEY,
        table,
        target_name=CDG_TARGET_NAME,
        reporter=emit_alignment_report,
    )


__all__ = [
    "CDG_EDGES_TABLE_KEY",
    "cdg_edges",
]
