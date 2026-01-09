"""CFG/DFG analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.build.analytics.cfg_dfg.cfg_core import CfgInputs, cfg_rows_for_fn
from codeintel.build.analytics.cfg_dfg.compute import CfgMetricsResult, DfgMetricsResult
from codeintel.build.analytics.cfg_dfg.dfg_core import (
    DfgInputs,
    build_dfg_context,
    dfg_block_rows,
    dfg_ext_row,
    dfg_fn_row,
)
from codeintel.build.analytics.cfg_dfg.helpers import parse_block_idx
from codeintel.build.analytics.graphs.constants import (
    MAX_CFG_CENTRALITY_SAMPLE,
    MAX_CFG_EIGEN_SAMPLE,
    MAX_DFG_CENTRALITY_SAMPLE,
)
from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.graphs.runtime.context import (
    GraphContext,
    GraphContextSpec,
    resolve_graph_context,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.finalize_helpers import finalize_analytics_rows
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    MultiTableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.data_models.ids import normalize_decimal_id

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

CFG_DFG_METRICS_TARGET_NAME = "cfg_dfg_metrics"

CFG_FUNCTION_METRICS_TABLE_KEY = "analytics.cfg_function_metrics"
CFG_BLOCK_METRICS_TABLE_KEY = "analytics.cfg_block_metrics"
CFG_FUNCTION_METRICS_EXT_TABLE_KEY = "analytics.cfg_function_metrics_ext"
DFG_FUNCTION_METRICS_TABLE_KEY = "analytics.dfg_function_metrics"
DFG_BLOCK_METRICS_TABLE_KEY = "analytics.dfg_block_metrics"
DFG_FUNCTION_METRICS_EXT_TABLE_KEY = "analytics.dfg_function_metrics_ext"


CFG_FUNCTION_METRICS_CONTRACT = contract_ref_for_table(
    table_key=CFG_FUNCTION_METRICS_TABLE_KEY,
    target_name=CFG_DFG_METRICS_TARGET_NAME,
    input_name="cfg_function_metrics__base",
    required_cols=(),
    clip_column=None,
)
CFG_BLOCK_METRICS_CONTRACT = contract_ref_for_table(
    table_key=CFG_BLOCK_METRICS_TABLE_KEY,
    target_name=CFG_DFG_METRICS_TARGET_NAME,
    input_name="cfg_block_metrics__base",
    required_cols=(),
    clip_column=None,
)
CFG_FUNCTION_METRICS_EXT_CONTRACT = contract_ref_for_table(
    table_key=CFG_FUNCTION_METRICS_EXT_TABLE_KEY,
    target_name=CFG_DFG_METRICS_TARGET_NAME,
    input_name="cfg_function_metrics_ext__base",
    required_cols=(),
    clip_column=None,
)
DFG_FUNCTION_METRICS_CONTRACT = contract_ref_for_table(
    table_key=DFG_FUNCTION_METRICS_TABLE_KEY,
    target_name=CFG_DFG_METRICS_TARGET_NAME,
    input_name="dfg_function_metrics__base",
    required_cols=(),
    clip_column=None,
)
DFG_BLOCK_METRICS_CONTRACT = contract_ref_for_table(
    table_key=DFG_BLOCK_METRICS_TABLE_KEY,
    target_name=CFG_DFG_METRICS_TARGET_NAME,
    input_name="dfg_block_metrics__base",
    required_cols=(),
    clip_column=None,
)
DFG_FUNCTION_METRICS_EXT_CONTRACT = contract_ref_for_table(
    table_key=DFG_FUNCTION_METRICS_EXT_TABLE_KEY,
    target_name=CFG_DFG_METRICS_TARGET_NAME,
    input_name="dfg_function_metrics_ext__base",
    required_cols=(),
    clip_column=None,
)


@dataclass(frozen=True, slots=True)
class _CfgDfgMetricsAnalysis:
    cfg: CfgMetricsResult
    dfg: DfgMetricsResult


@dataclass(frozen=True, slots=True)
class _CfgDfgMetricsInputs:
    cfg_blocks: pa.Table
    cfg_edges: pa.Table
    dfg_edges: pa.Table
    goids: pa.Table
    modules: pa.Table


def _rows_to_reader(
    rows: tuple[tuple[object, ...], ...],
    table_key: str,
) -> pa.Table:
    if not rows:
        return empty_table_for_table(table_key)
    return finalize_analytics_rows(table_key, rows)


def _module_by_path(modules_frame: pa.Table) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    for row in iter_rows(modules_frame):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_by_path[path] = module
    return module_by_path


def _function_metadata(
    goids_frame: pa.Table,
    modules_frame: pa.Table,
) -> dict[int, tuple[str, str | None, str | None]]:
    module_by_path = _module_by_path(modules_frame)
    metadata: dict[int, tuple[str, str | None, str | None]] = {}
    for row in iter_rows(goids_frame):
        if row.get("kind") not in {"function", "method"}:
            continue
        goid_raw = row.get("goid_h128")
        goid = normalize_decimal_id(goid_raw)
        if goid is None:
            continue
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        qualname = row.get("qualname")
        module = module_by_path.get(rel_path)
        metadata[int(goid)] = (rel_path, module, qualname if isinstance(qualname, str) else None)
    return metadata


def _cfg_blocks_by_fn(
    cfg_blocks_frame: pa.Table,
) -> dict[int, list[tuple[int, str, int, int]]]:
    blocks_by_fn: dict[int, list[tuple[int, str, int, int]]] = {}
    for row in iter_rows(cfg_blocks_frame):
        fn = normalize_decimal_id(row.get("function_goid_h128"))
        block_idx = normalize_decimal_id(row.get("block_idx"))
        if fn is None or block_idx is None:
            continue
        kind = row.get("kind")
        in_deg = normalize_decimal_id(row.get("in_degree")) or 0
        out_deg = normalize_decimal_id(row.get("out_degree")) or 0
        blocks_by_fn.setdefault(int(fn), []).append(
            (int(block_idx), str(kind), int(in_deg), int(out_deg))
        )
    return blocks_by_fn


def _coerce_block_id(value: object) -> str | int | None:
    if isinstance(value, (str, int)):
        return value
    return None


def _cfg_edges_by_fn(
    cfg_edges_frame: pa.Table,
) -> dict[int, list[tuple[int, int, str]]]:
    edges_by_fn: dict[int, list[tuple[int, int, str]]] = {}
    for row in iter_rows(cfg_edges_frame):
        fn = normalize_decimal_id(row.get("function_goid_h128"))
        if fn is None:
            continue
        src_idx = parse_block_idx(_coerce_block_id(row.get("src_block_id")))
        dst_idx = parse_block_idx(_coerce_block_id(row.get("dst_block_id")))
        if src_idx is None or dst_idx is None:
            continue
        edge_kind = row.get("edge_kind")
        edges_by_fn.setdefault(int(fn), []).append(
            (src_idx, dst_idx, str(edge_kind) if edge_kind is not None else "unknown")
        )
    return edges_by_fn


def _dfg_edges_by_fn(
    dfg_edges_frame: pa.Table,
) -> dict[int, list[tuple[int, int, str, str, bool, str]]]:
    edges_by_fn: dict[int, list[tuple[int, int, str, str, bool, str]]] = {}
    for row in iter_rows(dfg_edges_frame):
        fn = normalize_decimal_id(row.get("function_goid_h128"))
        if fn is None:
            continue
        src_idx = parse_block_idx(_coerce_block_id(row.get("src_block_id")))
        dst_idx = parse_block_idx(_coerce_block_id(row.get("dst_block_id")))
        if src_idx is None or dst_idx is None:
            continue
        src_var = row.get("src_var")
        dst_var = row.get("dst_var")
        if not isinstance(src_var, str) or not isinstance(dst_var, str):
            continue
        use_kind = row.get("use_kind")
        edges_by_fn.setdefault(int(fn), []).append(
            (
                src_idx,
                dst_idx,
                src_var,
                dst_var,
                bool(row.get("via_phi")),
                str(use_kind) if use_kind is not None else "unknown",
            )
        )
    return edges_by_fn


def _graph_context(
    env: BuildEnv,
    now: datetime,
    *,
    betweenness_cap: int,
    eigen_cap: int,
) -> GraphContext:
    return resolve_graph_context(
        GraphContextSpec(
            repo=env.repo,
            commit=env.commit,
            use_gpu=False,
            now=now,
            betweenness_cap=betweenness_cap,
            eigen_cap=eigen_cap,
        )
    )


def _build_cfg_metrics(
    env: BuildEnv,
    metadata: dict[int, tuple[str, str | None, str | None]],
    cfg_blocks_frame: pa.Table,
    cfg_edges_frame: pa.Table,
    graph_ctx: GraphContext,
) -> CfgMetricsResult:
    fn_rows: list[tuple[object, ...]] = []
    ext_rows: list[tuple[object, ...]] = []
    block_rows: list[tuple[object, ...]] = []
    cfg_inputs = CfgInputs(
        repo=env.repo,
        commit=env.commit,
        blocks_by_fn=_cfg_blocks_by_fn(cfg_blocks_frame),
        edges_by_fn=_cfg_edges_by_fn(cfg_edges_frame),
        now=graph_ctx.resolved_now(),
        graph_ctx=graph_ctx,
    )
    for fn_goid, meta in metadata.items():
        rows = cfg_rows_for_fn(fn_goid=fn_goid, meta=meta, inputs=cfg_inputs)
        if rows is None:
            continue
        fn_rows.append(rows.fn_row)
        ext_rows.append(rows.ext_row)
        block_rows.extend(rows.block_rows)
    return CfgMetricsResult(
        fn_rows=tuple(fn_rows),
        block_rows=tuple(block_rows),
        ext_rows=tuple(ext_rows),
    )


def _build_dfg_metrics(
    env: BuildEnv,
    metadata: dict[int, tuple[str, str | None, str | None]],
    dfg_edges_frame: pa.Table,
    graph_ctx: GraphContext,
) -> DfgMetricsResult:
    fn_rows: list[tuple[object, ...]] = []
    ext_rows: list[tuple[object, ...]] = []
    block_rows: list[tuple[object, ...]] = []
    dfg_edges_by_fn = _dfg_edges_by_fn(dfg_edges_frame)
    for fn_goid, meta in metadata.items():
        ctx = build_dfg_context(
            DfgInputs(
                fn_goid=fn_goid,
                meta=meta,
                edges=dfg_edges_by_fn.get(fn_goid, []),
                repo=env.repo,
                commit=env.commit,
                now=graph_ctx.resolved_now(),
                graph_ctx=graph_ctx,
            )
        )
        if ctx is None:
            continue
        fn_rows.append(dfg_fn_row(ctx))
        ext_rows.append(dfg_ext_row(ctx))
        block_rows.extend(dfg_block_rows(ctx))
    return DfgMetricsResult(
        fn_rows=tuple(fn_rows),
        block_rows=tuple(block_rows),
        ext_rows=tuple(ext_rows),
    )


def cfg_dfg_metrics_inputs(
    q__graph__cfg_blocks: InferableTabularInput,
    q__graph__cfg_edges: InferableTabularInput,
    q__graph__dfg_edges: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> _CfgDfgMetricsInputs:
    """Collect CFG/DFG metrics inputs from inferred tabular nodes.

    Returns
    -------
    _CfgDfgMetricsInputs
        Collected frames for CFG/DFG metrics computation.
    """
    return _CfgDfgMetricsInputs(
        cfg_blocks=tabular_to_scoped_table(
            q__graph__cfg_blocks,
            columns=None,
            scope=None,
            require_scope_columns=True,
        ),
        cfg_edges=tabular_to_scoped_table(
            q__graph__cfg_edges,
            columns=None,
            scope=None,
            require_scope_columns=True,
        ),
        dfg_edges=tabular_to_scoped_table(
            q__graph__dfg_edges,
            columns=None,
            scope=None,
            require_scope_columns=True,
        ),
        goids=tabular_to_scoped_table(
            q__core__goids,
            columns=None,
            scope=None,
            require_scope_columns=True,
        ),
        modules=tabular_to_scoped_table(
            q__core__modules,
            columns=None,
            scope=None,
            require_scope_columns=True,
        ),
    )


def cfg_dfg_metrics_analysis(
    env: BuildEnv,
    cfg_dfg_metrics_inputs: _CfgDfgMetricsInputs,
) -> _CfgDfgMetricsAnalysis:
    """Compute CFG/DFG metrics rows using graph tables and function metadata.

    Parameters
    ----------
    env
        Build environment with repo/commit metadata.
    cfg_dfg_metrics_inputs
        Collected CFG/DFG inputs from upstream graph and core tables.

    Returns
    -------
    _CfgDfgMetricsAnalysis
        CFG/DFG metrics payloads for downstream table nodes.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    goids = tabular_to_scoped_table(
        cfg_dfg_metrics_inputs.goids,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    modules = tabular_to_scoped_table(
        cfg_dfg_metrics_inputs.modules,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    cfg_blocks = tabular_to_scoped_table(
        cfg_dfg_metrics_inputs.cfg_blocks,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    cfg_edges = tabular_to_scoped_table(
        cfg_dfg_metrics_inputs.cfg_edges,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    dfg_edges = tabular_to_scoped_table(
        cfg_dfg_metrics_inputs.dfg_edges,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    metadata = _function_metadata(
        goids,
        modules,
    )
    now = datetime.now(UTC)
    cfg_ctx = _graph_context(
        env,
        now,
        betweenness_cap=MAX_CFG_CENTRALITY_SAMPLE,
        eigen_cap=MAX_CFG_EIGEN_SAMPLE,
    )
    dfg_ctx = _graph_context(
        env,
        now,
        betweenness_cap=MAX_DFG_CENTRALITY_SAMPLE,
        eigen_cap=MAX_CFG_EIGEN_SAMPLE,
    )
    cfg = _build_cfg_metrics(
        env,
        metadata,
        cfg_blocks,
        cfg_edges,
        cfg_ctx,
    )
    dfg = _build_dfg_metrics(
        env,
        metadata,
        dfg_edges,
        dfg_ctx,
    )
    return _CfgDfgMetricsAnalysis(cfg=cfg, dfg=dfg)


def cfg_function_metrics__base(
    cfg_dfg_metrics_analysis: _CfgDfgMetricsAnalysis,
) -> pa.Table:
    """Build CFG function metrics rows from the analysis payload.

    Returns
    -------
    pa.Table
        Reader of CFG function metrics rows.
    """
    return _rows_to_reader(
        cfg_dfg_metrics_analysis.cfg.fn_rows,
        CFG_FUNCTION_METRICS_TABLE_KEY,
    )


def cfg_block_metrics__base(
    cfg_dfg_metrics_analysis: _CfgDfgMetricsAnalysis,
) -> pa.Table:
    """Build CFG block metrics rows from the analysis payload.

    Returns
    -------
    pa.Table
        Reader of CFG block metrics rows.
    """
    return _rows_to_reader(
        cfg_dfg_metrics_analysis.cfg.block_rows,
        CFG_BLOCK_METRICS_TABLE_KEY,
    )


def cfg_function_metrics_ext__base(
    cfg_dfg_metrics_analysis: _CfgDfgMetricsAnalysis,
) -> pa.Table:
    """Build CFG function metrics ext rows from the analysis payload.

    Returns
    -------
    pa.Table
        Reader of CFG function metrics ext rows.
    """
    return _rows_to_reader(
        cfg_dfg_metrics_analysis.cfg.ext_rows,
        CFG_FUNCTION_METRICS_EXT_TABLE_KEY,
    )


def dfg_function_metrics__base(
    cfg_dfg_metrics_analysis: _CfgDfgMetricsAnalysis,
) -> pa.Table:
    """Build DFG function metrics rows from the analysis payload.

    Returns
    -------
    pa.Table
        Reader of DFG function metrics rows.
    """
    return _rows_to_reader(
        cfg_dfg_metrics_analysis.dfg.fn_rows,
        DFG_FUNCTION_METRICS_TABLE_KEY,
    )


def dfg_block_metrics__base(
    cfg_dfg_metrics_analysis: _CfgDfgMetricsAnalysis,
) -> pa.Table:
    """Build DFG block metrics rows from the analysis payload.

    Returns
    -------
    pa.Table
        Reader of DFG block metrics rows.
    """
    return _rows_to_reader(
        cfg_dfg_metrics_analysis.dfg.block_rows,
        DFG_BLOCK_METRICS_TABLE_KEY,
    )


def dfg_function_metrics_ext__base(
    cfg_dfg_metrics_analysis: _CfgDfgMetricsAnalysis,
) -> pa.Table:
    """Build DFG function metrics ext rows from the analysis payload.

    Returns
    -------
    pa.Table
        Reader of DFG function metrics ext rows.
    """
    return _rows_to_reader(
        cfg_dfg_metrics_analysis.dfg.ext_rows,
        DFG_FUNCTION_METRICS_EXT_TABLE_KEY,
    )


def _cfg_dfg_save_spec(table_key: str) -> DatasetSaveSpec:
    return DatasetSaveSpec(
        table_key=table_key,
        partition_columns=("repo", "commit"),
    )


_MODULE = sys.modules[__name__]
_CFG_DFG_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=CFG_DFG_METRICS_TARGET_NAME,
        tables=(),
        table_materializations_node="cfg_dfg_metrics__table_materializations",
        anchor_node_name="t__cfg_dfg_metrics",
        save_spec_factory=_cfg_dfg_save_spec,
        default_input_type=pa.Table,
    ),
    table_contexts=(
        TableTargetTableContext.from_contract_ref(
            contract_ref=CFG_FUNCTION_METRICS_CONTRACT,
            node_name="cfg_function_metrics__table",
        ),
        TableTargetTableContext.from_contract_ref(
            contract_ref=CFG_BLOCK_METRICS_CONTRACT,
            node_name="cfg_block_metrics__table",
        ),
        TableTargetTableContext.from_contract_ref(
            contract_ref=CFG_FUNCTION_METRICS_EXT_CONTRACT,
            node_name="cfg_function_metrics_ext__table",
        ),
        TableTargetTableContext.from_contract_ref(
            contract_ref=DFG_FUNCTION_METRICS_CONTRACT,
            node_name="dfg_function_metrics__table",
        ),
        TableTargetTableContext.from_contract_ref(
            contract_ref=DFG_BLOCK_METRICS_CONTRACT,
            node_name="dfg_block_metrics__table",
        ),
        TableTargetTableContext.from_contract_ref(
            contract_ref=DFG_FUNCTION_METRICS_EXT_CONTRACT,
            node_name="dfg_function_metrics_ext__table",
        ),
    ),
)
attach_table_target_template(_MODULE, spec=_CFG_DFG_TABLE_TARGET_SPEC)
cfg_function_metrics__table = _MODULE.cfg_function_metrics__table
cfg_block_metrics__table = _MODULE.cfg_block_metrics__table
cfg_function_metrics_ext__table = _MODULE.cfg_function_metrics_ext__table
dfg_function_metrics__table = _MODULE.dfg_function_metrics__table
dfg_block_metrics__table = _MODULE.dfg_block_metrics__table
dfg_function_metrics_ext__table = _MODULE.dfg_function_metrics_ext__table
cfg_dfg_metrics__table_materializations = _MODULE.cfg_dfg_metrics__table_materializations
t__cfg_dfg_metrics = _MODULE.t__cfg_dfg_metrics


__all__ = [
    "cfg_block_metrics__base",
    "cfg_block_metrics__table",
    "cfg_dfg_metrics__table_materializations",
    "cfg_dfg_metrics_inputs",
    "cfg_function_metrics__base",
    "cfg_function_metrics__table",
    "cfg_function_metrics_ext__base",
    "cfg_function_metrics_ext__table",
    "dfg_block_metrics__base",
    "dfg_block_metrics__table",
    "dfg_function_metrics__base",
    "dfg_function_metrics__table",
    "dfg_function_metrics_ext__base",
    "dfg_function_metrics_ext__table",
    "t__cfg_dfg_metrics",
]
