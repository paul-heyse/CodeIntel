"""Control dependence graph sources for graph targets."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pyarrow as pa

from codeintel.build.hamilton.native.graphs.filter_helpers import plan_filter_or_fallback
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.compute_masks import is_valid_expr, non_empty_string_expr
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.finalize_ops import finalize_spec_for_table, finalize_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id

CDG_EDGES_TABLE_KEY = "graph.cdg_edges"
CDG_TARGET_NAME = "cdg"
LOG = logging.getLogger(__name__)
_ASCENDING: Literal["ascending"] = "ascending"


@dataclass(frozen=True)
class _CdgEdgeRow:
    repo: str
    commit: str
    function_goid_h128: int
    src_block_id: str
    dst_block_id: str
    via_succ_block_id: str
    edge_kind: str
    via_edge_kind: str | None


@dataclass(frozen=True, slots=True)
class _CdgEdgeContext:
    repo: str
    commit: str
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


def _coerce_goid(value: object) -> int | None:
    return normalize_decimal_id(value)


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str) and value:
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


def _resolve_repo_commit(
    edges: Sequence[Mapping[str, object]],
    blocks: Sequence[Mapping[str, object]],
) -> tuple[str, str] | None:
    for row in edges:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        if repo is not None and commit is not None:
            return repo, commit
    for row in blocks:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        if repo is not None and commit is not None:
            return repo, commit
    return None


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
                repo=context.repo,
                commit=context.commit,
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
    repo_commit = _resolve_repo_commit(edges, blocks)
    if repo_commit is None:
        return []
    context = _CdgEdgeContext(
        repo=repo_commit[0],
        commit=repo_commit[1],
        function_goid=function_goid,
        post=post,
        inv_idx={value: key for key, value in idx.items()},
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

    expr = (
        is_valid_expr("function_goid_h128") & is_valid_expr("block_id") & is_valid_expr("block_idx")
    )
    return plan_filter_or_fallback(blocks_table, expr)


def _prefilter_cdg_edges(edges_table: pa.Table) -> pa.Table:
    if edges_table.num_rows == 0:
        return edges_table
    required = {"function_goid_h128", "edge_kind"}
    if not required.issubset(set(edges_table.column_names)):
        return edges_table

    expr = is_valid_expr("function_goid_h128") & non_empty_string_expr("edge_kind")
    return plan_filter_or_fallback(edges_table, expr)


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
        columns=["repo", "commit", "function_goid_h128", "block_id", "block_idx"],
        scope=None,
        require_scope_columns=False,
    )
    edges_table = tabular_to_scoped_table(
        q__graph__cfg_edges,
        columns=[
            "repo",
            "commit",
            "function_goid_h128",
            "src_block_id",
            "dst_block_id",
            "edge_kind",
        ],
        scope=None,
        require_scope_columns=False,
    )
    blocks_table = _prefilter_cdg_blocks(blocks_table)
    edges_table = _prefilter_cdg_edges(edges_table)
    if blocks_table.num_rows == 0 or edges_table.num_rows == 0:
        return empty_table_for_table(CDG_EDGES_TABLE_KEY)

    missing_goids: Counter[str] = Counter()
    blocks_by_goid: dict[int, list[dict[str, object]]] = defaultdict(list)
    block_columns = ("repo", "commit", "function_goid_h128", "block_id", "block_idx")
    for values in iter_tuples(blocks_table.to_reader(), columns=block_columns):
        row = dict(zip(block_columns, values, strict=False))
        function_goid = _coerce_goid(row.get("function_goid_h128"))
        if function_goid is None:
            missing_goids["blocks_missing_goid"] += 1
            continue
        blocks_by_goid[function_goid].append(row)
    edges_by_goid: dict[int, list[dict[str, object]]] = defaultdict(list)
    edge_columns = (
        "repo",
        "commit",
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "edge_kind",
    )
    for values in iter_tuples(edges_table.to_reader(), columns=edge_columns):
        row = dict(zip(edge_columns, values, strict=False))
        function_goid = _coerce_goid(row.get("function_goid_h128"))
        if function_goid is None:
            missing_goids["edges_missing_goid"] += 1
            continue
        edges_by_goid[function_goid].append(row)
    if missing_goids:
        LOG.info(
            "cdg_edges dropped rows missing function_goid_h128 blocks=%d edges=%d",
            missing_goids.get("blocks_missing_goid", 0),
            missing_goids.get("edges_missing_goid", 0),
        )

    rows: list[dict[str, object]] = []
    for function_goid, blocks in blocks_by_goid.items():
        edges = edges_by_goid.get(function_goid)
        if not edges:
            continue
        rows.extend(
            {
                "repo": row.repo,
                "commit": row.commit,
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

    table, _ = table_for_rows(CDG_EDGES_TABLE_KEY, rows)
    result = finalize_table(
        table,
        spec=finalize_spec_for_table(
            CDG_EDGES_TABLE_KEY,
            mode="strict",
            key_fields=_key_fields_for_table(CDG_EDGES_TABLE_KEY),
            order_by=_order_by_for_table(CDG_EDGES_TABLE_KEY),
            target_name=CDG_TARGET_NAME,
        ),
    )
    return result.good


def _key_fields_for_table(table_key: str) -> tuple[str, ...]:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except (KeyError, RuntimeError, TypeError):
        return ()
    if schema is None or not schema.primary_key:
        return ()
    return tuple(schema.primary_key)


def _order_by_for_table(table_key: str) -> tuple[SortKey, ...]:
    key_fields = _key_fields_for_table(table_key)
    if not key_fields:
        return ()
    return tuple((field, _ASCENDING) for field in key_fields)


__all__ = [
    "CDG_EDGES_TABLE_KEY",
    "cdg_edges",
]
