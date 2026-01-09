"""Subsystem graph metrics built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pyarrow as pa
from hamilton.function_modifiers import cache

from codeintel.build.analytics.graphs.graph_metrics import (
    build_graph_metric_filters_from_sets,
    build_import_graph_from_tables,
)
from codeintel.build.analytics.graphs.subsystem_graph_metrics import (
    SubsystemGraphMetricInputs,
    build_subsystem_graph_metrics_rows,
)
from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.graphs.runtime import GraphRuntimeOptions, graph_runtime_options_from_env
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.finalize_helpers import finalize_analytics_rows
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.scoping import collect_scoped_rows
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEM_GRAPH_METRICS_TARGET_NAME = "subsystem_graph_metrics"
SUBSYSTEM_GRAPH_METRICS_TABLE_KEY = "analytics.subsystem_graph_metrics"
SUBSYSTEM_GRAPH_METRICS_CONTRACT = contract_ref_for_table(
    table_key=SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,
    target_name=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
    input_name="subsystem_graph_metrics__base",
    required_cols=(),
    clip_column=None,
)


@cache(behavior="ignore")
def _graph_runtime_options(env: BuildEnv) -> GraphRuntimeOptions:
    return graph_runtime_options_from_env(env)


def subsystem_graph_metrics__base(
    env: BuildEnv,
    q__analytics__subsystem_modules: InferableTabularInput,
    q__graph__import_graph_edges: InferableTabularInput,
    q__graph__import_modules: InferableTabularInput,
) -> pa.Table:
    """Build subsystem graph metrics rows.

    Returns
    -------
    pa.Table
        Reader containing subsystem graph metrics rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    membership_rows = collect_scoped_rows(
        q__analytics__subsystem_modules,
        ("repo", "commit", "subsystem_id", "module"),
        scope=scope,
    )
    import_edges_table = tabular_to_scoped_table(
        q__graph__import_graph_edges,
        columns=("src_module", "dst_module", "module_layer"),
        scope=scope,
        require_scope_columns=True,
    )
    import_modules_table = tabular_to_scoped_table(
        q__graph__import_modules,
        columns=("module", "scc_id", "component_size", "layer"),
        scope=scope,
        require_scope_columns=True,
    )
    import_graph = build_import_graph_from_tables(import_edges_table, import_modules_table)
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
    if not rows:
        return empty_table_for_table(SUBSYSTEM_GRAPH_METRICS_TABLE_KEY)
    return finalize_analytics_rows(SUBSYSTEM_GRAPH_METRICS_TABLE_KEY, rows)


_MODULE = sys.modules[__name__]
_SUBSYSTEM_GRAPH_METRICS_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=SUBSYSTEM_GRAPH_METRICS_CONTRACT,
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_SUBSYSTEM_GRAPH_METRICS_TABLE_TARGET_SPEC)
subsystem_graph_metrics__table = _MODULE.subsystem_graph_metrics__table
subsystem_graph_metrics__table_materializations = (
    _MODULE.subsystem_graph_metrics__table_materializations
)
t__subsystem_graph_metrics = _MODULE.t__subsystem_graph_metrics


__all__ = [
    "subsystem_graph_metrics__base",
    "subsystem_graph_metrics__table",
    "t__subsystem_graph_metrics",
]
