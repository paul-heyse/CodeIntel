"""Graph table targets built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.call_graph import (
    CALL_GRAPH_EDGES_TABLE_KEY,
    CALL_GRAPH_NODES_TABLE_KEY,
)
from codeintel.build.hamilton.native.graphs.cfg_dfg import (
    CFG_BLOCKS_TABLE_KEY,
    CFG_EDGES_TABLE_KEY,
    DFG_EDGES_TABLE_KEY,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    IMPORT_GRAPH_EDGES_TABLE_KEY,
    IMPORT_MODULES_TABLE_KEY,
)
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
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, pl.LazyFrame)

CALL_GRAPH_TARGET_NAME = "call_graph"
IMPORT_GRAPH_TARGET_NAME = "import_graph"
CFG_TARGET_NAME = "cfg"
DFG_TARGET_NAME = "dfg"

CALL_GRAPH_TABLE_KEYS = (CALL_GRAPH_NODES_TABLE_KEY, CALL_GRAPH_EDGES_TABLE_KEY)
IMPORT_GRAPH_TABLE_KEYS = (IMPORT_MODULES_TABLE_KEY, IMPORT_GRAPH_EDGES_TABLE_KEY)
CFG_TABLE_KEYS = (CFG_BLOCKS_TABLE_KEY, CFG_EDGES_TABLE_KEY)
DFG_TABLE_KEYS = (DFG_EDGES_TABLE_KEY,)

CALL_GRAPH_SAVE_CONTEXT = SaverContext(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
IMPORT_GRAPH_SAVE_CONTEXT = SaverContext(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
CFG_SAVE_CONTEXT = SaverContext(domain="graphs", target=CFG_TARGET_NAME)
DFG_SAVE_CONTEXT = SaverContext(domain="graphs", target=DFG_TARGET_NAME)


@save_dataset(
    context=CALL_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CALL_GRAPH_NODES_TABLE_KEY),
)
@tag_dataset(domain="graphs", target=CALL_GRAPH_TARGET_NAME, table_key=CALL_GRAPH_NODES_TABLE_KEY)
def call_graph__nodes_table(call_graph_nodes: pl.LazyFrame) -> pl.LazyFrame:
    """Persist call graph nodes.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for call graph nodes.
    """
    return call_graph_nodes


@save_dataset(
    context=CALL_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(
        table_key=CALL_GRAPH_EDGES_TABLE_KEY,
        partition_columns=("repo", "commit"),
    ),
)
@tag_dataset(domain="graphs", target=CALL_GRAPH_TARGET_NAME, table_key=CALL_GRAPH_EDGES_TABLE_KEY)
def call_graph__edges_table(call_graph_edges: pl.LazyFrame) -> pl.LazyFrame:
    """Persist call graph edges.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for call graph edges.
    """
    return call_graph_edges


@save_dataset(
    context=IMPORT_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(
        table_key=IMPORT_MODULES_TABLE_KEY,
        partition_columns=("repo", "commit"),
    ),
)
@tag_dataset(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME, table_key=IMPORT_MODULES_TABLE_KEY)
def import_graph__modules_table(import_modules: pl.LazyFrame) -> pl.LazyFrame:
    """Persist import modules.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for import modules.
    """
    return import_modules


@save_dataset(
    context=IMPORT_GRAPH_SAVE_CONTEXT,
    spec=DatasetSaveSpec(
        table_key=IMPORT_GRAPH_EDGES_TABLE_KEY,
        partition_columns=("repo", "commit"),
    ),
)
@tag_dataset(
    domain="graphs",
    target=IMPORT_GRAPH_TARGET_NAME,
    table_key=IMPORT_GRAPH_EDGES_TABLE_KEY,
)
def import_graph__edges_table(import_graph_edges: pl.LazyFrame) -> pl.LazyFrame:
    """Persist import graph edges.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for import graph edges.
    """
    return import_graph_edges


@save_dataset(
    context=CFG_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CFG_BLOCKS_TABLE_KEY),
)
@tag_dataset(domain="graphs", target=CFG_TARGET_NAME, table_key=CFG_BLOCKS_TABLE_KEY)
def cfg__blocks_table(cfg_blocks: pl.LazyFrame) -> pl.LazyFrame:
    """Persist CFG blocks.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for CFG blocks.
    """
    return cfg_blocks


@save_dataset(
    context=CFG_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=CFG_EDGES_TABLE_KEY),
)
@tag_dataset(domain="graphs", target=CFG_TARGET_NAME, table_key=CFG_EDGES_TABLE_KEY)
def cfg__edges_table(cfg_edges: pl.LazyFrame) -> pl.LazyFrame:
    """Persist CFG edges.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for CFG edges.
    """
    return cfg_edges


@save_dataset(
    context=DFG_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=DFG_EDGES_TABLE_KEY),
)
@tag_dataset(domain="graphs", target=DFG_TARGET_NAME, table_key=DFG_EDGES_TABLE_KEY)
def dfg__edges_table(dfg_edges: pl.LazyFrame) -> pl.LazyFrame:
    """Persist DFG edges.

    Returns
    -------
    polars.LazyFrame
        Lazy frame to materialize for DFG edges.
    """
    return dfg_edges


call_graph__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=CALL_GRAPH_TARGET_NAME,
    table_keys=CALL_GRAPH_TABLE_KEYS,
    node_name="call_graph__table_materializations",
)

import_graph__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=IMPORT_GRAPH_TARGET_NAME,
    table_keys=IMPORT_GRAPH_TABLE_KEYS,
    node_name="import_graph__table_materializations",
)

cfg__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=CFG_TARGET_NAME,
    table_keys=CFG_TABLE_KEYS,
    node_name="cfg__table_materializations",
)

dfg__table_materializations = make_table_materializations_collector(
    domain="graphs",
    target=DFG_TARGET_NAME,
    table_keys=DFG_TABLE_KEYS,
    node_name="dfg__table_materializations",
)


@codeintel_target(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
def t__call_graph(
    env: BuildEnv,
    catalog: DagCatalog,
    call_graph__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize call_graph target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the call_graph target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=CALL_GRAPH_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=call_graph__table_materializations,
    )


@codeintel_target(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def t__import_graph(
    env: BuildEnv,
    catalog: DagCatalog,
    import_graph__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize import_graph target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the import_graph target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=IMPORT_GRAPH_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=import_graph__table_materializations,
    )


@codeintel_target(domain="graphs", target=CFG_TARGET_NAME)
def t__cfg(
    env: BuildEnv,
    catalog: DagCatalog,
    cfg__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize cfg target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the cfg target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=CFG_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=cfg__table_materializations,
    )


@codeintel_target(domain="graphs", target=DFG_TARGET_NAME)
def t__dfg(
    env: BuildEnv,
    catalog: DagCatalog,
    dfg__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize dfg target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the dfg target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=DFG_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=dfg__table_materializations,
    )


__all__ = [
    "CALL_GRAPH_TARGET_NAME",
    "CFG_TARGET_NAME",
    "DFG_TARGET_NAME",
    "IMPORT_GRAPH_TARGET_NAME",
    "call_graph__edges_table",
    "call_graph__nodes_table",
    "call_graph__table_materializations",
    "cfg__blocks_table",
    "cfg__edges_table",
    "cfg__table_materializations",
    "dfg__edges_table",
    "dfg__table_materializations",
    "import_graph__edges_table",
    "import_graph__modules_table",
    "import_graph__table_materializations",
    "t__call_graph",
    "t__cfg",
    "t__dfg",
    "t__import_graph",
]
