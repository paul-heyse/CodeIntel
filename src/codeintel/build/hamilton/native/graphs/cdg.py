"""Control dependence graph sources for graph targets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from codeintel.build.tabular.conversion import tabular_to_frame
from codeintel.build.tabular.frames import dedupe_frame_for_table, empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput

CDG_EDGES_TABLE_KEY = "graph.cdg_edges"


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


def _block_indexes(blocks: pl.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    block_idx_by_id: dict[str, int] = {}
    block_id_by_idx: dict[int, str] = {}
    for row in blocks.iter_rows(named=True):
        block_id = row.get("block_id")
        block_idx = row.get("block_idx")
        if not isinstance(block_id, str) or not isinstance(block_idx, int):
            continue
        block_idx_by_id[block_id] = block_idx
        block_id_by_idx[block_idx] = block_id
    return block_idx_by_id, block_id_by_idx


def _edge_indexes(
    edges: pl.DataFrame, block_idx_by_id: dict[str, int]
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str | None]]:
    edges_idx: list[tuple[int, int]] = []
    edge_kind_by_pair: dict[tuple[int, int], str | None] = {}
    for row in edges.iter_rows(named=True):
        src_id = row.get("src_block_id")
        dst_id = row.get("dst_block_id")
        if not isinstance(src_id, str) or not isinstance(dst_id, str):
            continue
        src_idx = block_idx_by_id.get(src_id)
        dst_idx = block_idx_by_id.get(dst_id)
        if src_idx is None or dst_idx is None:
            continue
        edges_idx.append((src_idx, dst_idx))
        edge_kind_by_pair[src_idx, dst_idx] = row.get("edge_kind")
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
    blocks: pl.DataFrame,
    edges: pl.DataFrame,
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


def cdg_edges(
    q__graph__cfg_blocks: InferableTabularInput,
    q__graph__cfg_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build control dependence edges from CFG blocks/edges.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for graph.cdg_edges.
    """
    blocks_frame = tabular_to_frame(q__graph__cfg_blocks)
    edges_frame = tabular_to_frame(q__graph__cfg_edges)
    if blocks_frame.is_empty() or edges_frame.is_empty():
        return empty_frame_for_table(CDG_EDGES_TABLE_KEY)

    rows: list[dict[str, object]] = []
    for function_goid in blocks_frame.get_column("function_goid_h128").unique().to_list():
        if not isinstance(function_goid, int):
            continue
        blocks = blocks_frame.filter(pl.col("function_goid_h128") == function_goid)
        edges = edges_frame.filter(pl.col("function_goid_h128") == function_goid)
        if blocks.is_empty() or edges.is_empty():
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
        return empty_frame_for_table(CDG_EDGES_TABLE_KEY)

    frame = pl.DataFrame(rows)
    frame = dedupe_frame_for_table(frame.lazy(), table_key=CDG_EDGES_TABLE_KEY)
    return frame.select(
        [
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "via_succ_block_id",
            "edge_kind",
            "via_edge_kind",
        ]
    )


__all__ = [
    "CDG_EDGES_TABLE_KEY",
    "cdg_edges",
]
