"""Subsystem graph metrics built with inferable tabular nodes."""

from __future__ import annotations

import sys

import polars as pl

from codeintel.build.analytics.graphs.graph_metrics import (
    build_graph_metric_filters_from_sets,
    build_import_graph_from_rows,
)
from codeintel.build.analytics.graphs.subsystem_graph_metrics import (
    SubsystemGraphMetricInputs,
    build_subsystem_graph_metrics_rows,
)
from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import rows_to_frame
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEM_GRAPH_METRICS_TARGET_NAME = "subsystem_graph_metrics"
SUBSYSTEM_GRAPH_METRICS_TABLE_KEY = "analytics.subsystem_graph_metrics"
SUBSYSTEM_GRAPH_METRICS_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_graph_metrics__base",
)

SUBSYSTEM_GRAPH_METRICS_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "import_in_degree",
    "import_out_degree",
    "import_pagerank",
    "import_betweenness",
    "import_closeness",
    "import_layer",
    "created_at",
)


def _graph_runtime_options(env: BuildEnv) -> GraphRuntimeOptions:
    if env.execution_context is None:
        return GraphRuntimeOptions(snapshot=env.snapshot)
    return GraphRuntimeOptions(
        snapshot=env.snapshot,
        backend=env.execution_context.graph_backend,
        features=env.execution_context.graph_features,
    )


def _collect_rows(
    value: InferableTabularInput,
    columns: tuple[str, ...],
    *,
    repo: str | None,
    commit: str | None,
) -> list[dict[str, object]]:
    frame = tabular_to_lazyframe(value)
    available = set(frame.columns)
    if repo is not None and "repo" in available:
        frame = frame.filter(pl.col("repo") == repo)
    if commit is not None and "commit" in available:
        frame = frame.filter(pl.col("commit") == commit)
    return frame.select(list(columns)).collect().to_dicts()


def subsystem_graph_metrics__base(
    env: BuildEnv,
    q__analytics__subsystem_modules: InferableTabularInput,
    q__graph__import_graph_edges: InferableTabularInput,
    q__graph__import_modules: InferableTabularInput,
) -> pl.LazyFrame:
    """Build subsystem graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem graph metrics rows.
    """
    membership_rows = _collect_rows(
        q__analytics__subsystem_modules,
        ("repo", "commit", "subsystem_id", "module"),
        repo=env.repo,
        commit=env.commit,
    )
    import_edge_rows = _collect_rows(
        q__graph__import_graph_edges,
        ("src_module", "dst_module", "module_layer"),
        repo=env.repo,
        commit=env.commit,
    )
    import_module_rows = _collect_rows(
        q__graph__import_modules,
        ("module", "scc_id", "component_size", "layer"),
        repo=env.repo,
        commit=env.commit,
    )
    import_graph = build_import_graph_from_rows(import_edge_rows, import_module_rows)
    subsystem_ids = {
        str(row["subsystem_id"]) for row in membership_rows if row.get("subsystem_id") is not None
    }
    module_names = {str(row["module"]) for row in membership_rows if row.get("module") is not None}
    filters = build_graph_metric_filters_from_sets(
        modules=module_names,
        subsystems=subsystem_ids,
    )
    runtime_options = _graph_runtime_options(env)
    rows = build_subsystem_graph_metrics_rows(
        SubsystemGraphMetricInputs(
            repo=env.repo,
            commit=env.commit,
            import_graph=import_graph,
            membership_rows=membership_rows,
            runtime=runtime_options,
            filters=filters,
        )
    )
    return rows_to_frame(
        SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,
        rows,
        columns=SUBSYSTEM_GRAPH_METRICS_COLUMNS,
    )


_MODULE = sys.modules[__name__]
_SUBSYSTEM_GRAPH_METRICS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,
            base_node="subsystem_graph_metrics__base",
            contract=SUBSYSTEM_GRAPH_METRICS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SUBSYSTEM_GRAPH_METRICS_TABLE_KEY),
            node_name="subsystem_graph_metrics__table",
        ),
    ),
    table_materializations_node="subsystem_graph_metrics__table_materializations",
    anchor_node_name="t__subsystem_graph_metrics",
)
attach_table_target_template(_MODULE, spec=_SUBSYSTEM_GRAPH_METRICS_TABLE_TARGET_SPEC)
subsystem_graph_metrics__table = _MODULE.subsystem_graph_metrics__table
subsystem_graph_metrics__table_materializations = _MODULE.subsystem_graph_metrics__table_materializations
t__subsystem_graph_metrics = _MODULE.t__subsystem_graph_metrics


__all__ = [
    "subsystem_graph_metrics__base",
    "subsystem_graph_metrics__table",
    "t__subsystem_graph_metrics",
]
