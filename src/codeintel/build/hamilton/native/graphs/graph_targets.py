"""Graph table targets built with inferable tabular nodes."""

from __future__ import annotations

import sys

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.call_graph import (
    CALL_GRAPH_EDGES_TABLE_KEY,
    CALL_GRAPH_NODES_TABLE_KEY,
)
from codeintel.build.hamilton.native.graphs.call_wiring import (
    CALL_WIRING_TARGET_NAME,
    CPG_ARG_TO_PARAM_EDGES_TABLE_KEY,
    CPG_CALL_EDGES_TABLE_KEY,
    CPG_CALL_TARGETS_TABLE_KEY,
    CPG_RET_TO_CALL_EDGES_TABLE_KEY,
)
from codeintel.build.hamilton.native.graphs.cdg import CDG_EDGES_TABLE_KEY
from codeintel.build.hamilton.native.graphs.cfg_dfg import (
    CFG_BLOCKS_TABLE_KEY,
    CFG_EDGES_TABLE_KEY,
    DFG_EDGES_TABLE_KEY,
)
from codeintel.build.hamilton.native.graphs.cpg import (
    CPG_EDGES_TABLE_KEY,
    CPG_NODES_TABLE_KEY,
    CPG_TARGET_NAME,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    IMPORT_GRAPH_EDGES_TABLE_KEY,
    IMPORT_MODULES_TABLE_KEY,
)
from codeintel.build.hamilton.native.graphs.pdg import PDG_EDGES_TABLE_KEY
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

CALL_GRAPH_TARGET_NAME = "call_graph"
IMPORT_GRAPH_TARGET_NAME = "import_graph"
CFG_TARGET_NAME = "cfg"
DFG_TARGET_NAME = "dfg"
CDG_TARGET_NAME = "cdg"
PDG_TARGET_NAME = "pdg"
CALL_GRAPH_TABLE_KEYS = (CALL_GRAPH_NODES_TABLE_KEY, CALL_GRAPH_EDGES_TABLE_KEY)
IMPORT_GRAPH_TABLE_KEYS = (IMPORT_MODULES_TABLE_KEY, IMPORT_GRAPH_EDGES_TABLE_KEY)


_MODULE = sys.modules[__name__]
_CALL_GRAPH_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=CALL_GRAPH_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=CALL_GRAPH_NODES_TABLE_KEY,
            base_node="call_graph_nodes",
            save_spec=DatasetSaveSpec(table_key=CALL_GRAPH_NODES_TABLE_KEY),
            node_name="call_graph__nodes_table",
        ),
        TableTargetTableSpec(
            table_key=CALL_GRAPH_EDGES_TABLE_KEY,
            base_node="call_graph_edges",
            save_spec=DatasetSaveSpec(
                table_key=CALL_GRAPH_EDGES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="call_graph__edges_table",
        ),
    ),
    table_materializations_node="call_graph__table_materializations",
    anchor_node_name="t__call_graph",
)
attach_table_target_template(_MODULE, spec=_CALL_GRAPH_TABLE_TARGET_SPEC)
call_graph__nodes_table = _MODULE.call_graph__nodes_table
call_graph__edges_table = _MODULE.call_graph__edges_table
call_graph__table_materializations = _MODULE.call_graph__table_materializations
t__call_graph = _MODULE.t__call_graph

_IMPORT_GRAPH_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=IMPORT_GRAPH_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=IMPORT_MODULES_TABLE_KEY,
            base_node="import_modules",
            save_spec=DatasetSaveSpec(
                table_key=IMPORT_MODULES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="import_graph__modules_table",
        ),
        TableTargetTableSpec(
            table_key=IMPORT_GRAPH_EDGES_TABLE_KEY,
            base_node="import_graph_edges",
            save_spec=DatasetSaveSpec(
                table_key=IMPORT_GRAPH_EDGES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="import_graph__edges_table",
        ),
    ),
    table_materializations_node="import_graph__table_materializations",
    anchor_node_name="t__import_graph",
)
attach_table_target_template(_MODULE, spec=_IMPORT_GRAPH_TABLE_TARGET_SPEC)
import_graph__modules_table = _MODULE.import_graph__modules_table
import_graph__edges_table = _MODULE.import_graph__edges_table
import_graph__table_materializations = _MODULE.import_graph__table_materializations
t__import_graph = _MODULE.t__import_graph

_CFG_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=CFG_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=CFG_BLOCKS_TABLE_KEY,
            base_node="cfg_blocks",
            save_spec=DatasetSaveSpec(table_key=CFG_BLOCKS_TABLE_KEY),
            node_name="cfg__blocks_table",
        ),
        TableTargetTableSpec(
            table_key=CFG_EDGES_TABLE_KEY,
            base_node="cfg_edges",
            save_spec=DatasetSaveSpec(table_key=CFG_EDGES_TABLE_KEY),
            node_name="cfg__edges_table",
        ),
    ),
    table_materializations_node="cfg__table_materializations",
    anchor_node_name="t__cfg",
)
attach_table_target_template(_MODULE, spec=_CFG_TABLE_TARGET_SPEC)
cfg__blocks_table = _MODULE.cfg__blocks_table
cfg__edges_table = _MODULE.cfg__edges_table
cfg__table_materializations = _MODULE.cfg__table_materializations
t__cfg = _MODULE.t__cfg

_DFG_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=DFG_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=DFG_EDGES_TABLE_KEY,
            base_node="dfg_edges",
            save_spec=DatasetSaveSpec(table_key=DFG_EDGES_TABLE_KEY),
            node_name="dfg__edges_table",
        ),
    ),
    table_materializations_node="dfg__table_materializations",
    anchor_node_name="t__dfg",
)
attach_table_target_template(_MODULE, spec=_DFG_TABLE_TARGET_SPEC)
dfg__edges_table = _MODULE.dfg__edges_table
dfg__table_materializations = _MODULE.dfg__table_materializations
t__dfg = _MODULE.t__dfg

_CDG_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=CDG_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=CDG_EDGES_TABLE_KEY,
            base_node="cdg_edges",
            save_spec=DatasetSaveSpec(table_key=CDG_EDGES_TABLE_KEY),
            node_name="cdg__edges_table",
        ),
    ),
    table_materializations_node="cdg__table_materializations",
    anchor_node_name="t__cdg",
)
attach_table_target_template(_MODULE, spec=_CDG_TABLE_TARGET_SPEC)
cdg__edges_table = _MODULE.cdg__edges_table
cdg__table_materializations = _MODULE.cdg__table_materializations
t__cdg = _MODULE.t__cdg

_PDG_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=PDG_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=PDG_EDGES_TABLE_KEY,
            base_node="pdg_edges",
            save_spec=DatasetSaveSpec(table_key=PDG_EDGES_TABLE_KEY),
            node_name="pdg__edges_table",
        ),
    ),
    table_materializations_node="pdg__table_materializations",
    anchor_node_name="t__pdg",
)
attach_table_target_template(_MODULE, spec=_PDG_TABLE_TARGET_SPEC)
pdg__edges_table = _MODULE.pdg__edges_table
pdg__table_materializations = _MODULE.pdg__table_materializations
t__pdg = _MODULE.t__pdg

_CALL_WIRING_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=CALL_WIRING_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=CPG_CALL_TARGETS_TABLE_KEY,
            base_node="cpg_call_targets",
            save_spec=DatasetSaveSpec(
                table_key=CPG_CALL_TARGETS_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="call_wiring__call_targets_table",
        ),
        TableTargetTableSpec(
            table_key=CPG_CALL_EDGES_TABLE_KEY,
            base_node="cpg_edges_calls",
            save_spec=DatasetSaveSpec(
                table_key=CPG_CALL_EDGES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="call_wiring__edges_calls_table",
        ),
        TableTargetTableSpec(
            table_key=CPG_ARG_TO_PARAM_EDGES_TABLE_KEY,
            base_node="cpg_edges_arg_to_param",
            save_spec=DatasetSaveSpec(
                table_key=CPG_ARG_TO_PARAM_EDGES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="call_wiring__edges_arg_to_param_table",
        ),
        TableTargetTableSpec(
            table_key=CPG_RET_TO_CALL_EDGES_TABLE_KEY,
            base_node="cpg_edges_ret_to_call",
            save_spec=DatasetSaveSpec(
                table_key=CPG_RET_TO_CALL_EDGES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="call_wiring__edges_ret_to_call_table",
        ),
    ),
    table_materializations_node="call_wiring__table_materializations",
    anchor_node_name="t__call_wiring",
)
attach_table_target_template(_MODULE, spec=_CALL_WIRING_TABLE_TARGET_SPEC)
call_wiring__call_targets_table = _MODULE.call_wiring__call_targets_table
call_wiring__edges_calls_table = _MODULE.call_wiring__edges_calls_table
call_wiring__edges_arg_to_param_table = _MODULE.call_wiring__edges_arg_to_param_table
call_wiring__edges_ret_to_call_table = _MODULE.call_wiring__edges_ret_to_call_table
call_wiring__table_materializations = _MODULE.call_wiring__table_materializations
t__call_wiring = _MODULE.t__call_wiring

_CPG_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="graphs",
    target_name=CPG_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=CPG_NODES_TABLE_KEY,
            base_node="cpg_nodes",
            save_spec=DatasetSaveSpec(
                table_key=CPG_NODES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="cpg__nodes_table",
        ),
        TableTargetTableSpec(
            table_key=CPG_EDGES_TABLE_KEY,
            base_node="cpg_edges",
            save_spec=DatasetSaveSpec(
                table_key=CPG_EDGES_TABLE_KEY,
                partition_columns=("repo", "commit"),
            ),
            node_name="cpg__edges_table",
        ),
    ),
    table_materializations_node="cpg__table_materializations",
    anchor_node_name="t__cpg",
)
attach_table_target_template(_MODULE, spec=_CPG_TABLE_TARGET_SPEC)
cpg__nodes_table = _MODULE.cpg__nodes_table
cpg__edges_table = _MODULE.cpg__edges_table
cpg__table_materializations = _MODULE.cpg__table_materializations
t__cpg = _MODULE.t__cpg


__all__ = [
    "CALL_GRAPH_TABLE_KEYS",
    "CALL_GRAPH_TARGET_NAME",
    "CALL_WIRING_TARGET_NAME",
    "CDG_TARGET_NAME",
    "CFG_TARGET_NAME",
    "CPG_TARGET_NAME",
    "DFG_TARGET_NAME",
    "IMPORT_GRAPH_TABLE_KEYS",
    "IMPORT_GRAPH_TARGET_NAME",
    "PDG_TARGET_NAME",
    "call_graph__edges_table",
    "call_graph__nodes_table",
    "call_graph__table_materializations",
    "call_wiring__call_targets_table",
    "call_wiring__edges_arg_to_param_table",
    "call_wiring__edges_calls_table",
    "call_wiring__edges_ret_to_call_table",
    "call_wiring__table_materializations",
    "cdg__edges_table",
    "cdg__table_materializations",
    "cfg__blocks_table",
    "cfg__edges_table",
    "cfg__table_materializations",
    "cpg__edges_table",
    "cpg__nodes_table",
    "cpg__table_materializations",
    "dfg__edges_table",
    "dfg__table_materializations",
    "import_graph__edges_table",
    "import_graph__modules_table",
    "import_graph__table_materializations",
    "pdg__edges_table",
    "pdg__table_materializations",
    "t__call_graph",
    "t__call_wiring",
    "t__cdg",
    "t__cfg",
    "t__cpg",
    "t__dfg",
    "t__import_graph",
    "t__pdg",
]
