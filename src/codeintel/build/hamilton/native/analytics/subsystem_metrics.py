"""Subsystem graph metrics built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.graphs.subsystem_graph_metrics import (
    build_subsystem_graph_metrics_rows,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import rows_to_frame
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEM_GRAPH_METRICS_TARGET_NAME = "subsystem_graph_metrics"
SUBSYSTEM_GRAPH_METRICS_TABLE_KEY = "analytics.subsystem_graph_metrics"
SUBSYSTEM_GRAPH_METRICS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
)
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


def subsystem_graph_metrics__base(
    env: BuildEnv,
    _q__analytics__subsystem_modules: InferableTabularInput,
    _q__analytics__graph_metrics_modules: InferableTabularInput,
    _q__graph__import_graph_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build subsystem graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem graph metrics rows.
    """
    rows = build_subsystem_graph_metrics_rows(
        env.gateway,
        repo=env.repo,
        commit=env.commit,
    )
    return rows_to_frame(
        SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,
        rows,
        columns=SUBSYSTEM_GRAPH_METRICS_COLUMNS,
    )


@save_dataset(
    context=SUBSYSTEM_GRAPH_METRICS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SUBSYSTEM_GRAPH_METRICS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
    table_key=SUBSYSTEM_GRAPH_METRICS_TABLE_KEY,
)
@table_contract(SUBSYSTEM_GRAPH_METRICS_CONTRACT)
def subsystem_graph_metrics__table(
    subsystem_graph_metrics__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist subsystem graph metrics rows.

    Returns
    -------
    pl.LazyFrame
        Persisted subsystem graph metrics frame.
    """
    return subsystem_graph_metrics__base


@codeintel_target(domain="analytics", target=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME)
def t__subsystem_graph_metrics(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__subsystem_graph_metrics: MaterializationResult,
) -> TargetRunRecord:
    """Finalize subsystem_graph_metrics target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the subsystem_graph_metrics target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SUBSYSTEM_GRAPH_METRICS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            SUBSYSTEM_GRAPH_METRICS_TABLE_KEY: m__analytics__subsystem_graph_metrics,
        },
    )


__all__ = [
    "subsystem_graph_metrics__base",
    "subsystem_graph_metrics__table",
    "t__subsystem_graph_metrics",
]
