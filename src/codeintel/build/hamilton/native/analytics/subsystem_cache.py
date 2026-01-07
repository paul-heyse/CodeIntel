"""Subsystem cache tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from contextlib import suppress

import pyarrow as pa

from codeintel.build.analytics.subsystems.cache import build_subsystem_profile_cache_frame
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.arrow_ops import align_table_to_contract
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEM_CACHES_TARGET_NAME = "subsystem_caches"
SUBSYSTEM_PROFILE_CACHE_TABLE_KEY = "analytics.subsystem_profile_cache"
SUBSYSTEM_PROFILE_CACHE_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_PROFILE_CACHE_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_profile_cache__base",
)


def subsystem_profile_cache__base(
    env: BuildEnv,
    q__analytics__subsystems: InferableTabularInput,
    q__analytics__subsystem_graph_metrics: InferableTabularInput,
) -> pa.Table:
    """Build cached subsystem profile rows.

    Returns
    -------
    pa.Table
        Table containing subsystem profile cache rows.
    """
    subsystems_frame = tabular_to_arrow_table(q__analytics__subsystems)
    metrics_frame = tabular_to_arrow_table(q__analytics__subsystem_graph_metrics)
    table = build_subsystem_profile_cache_frame(
        env.snapshot,
        subsystems_frame=subsystems_frame,
        subsystem_graph_metrics_frame=metrics_frame,
    )
    with suppress(KeyError, RuntimeError, ValueError):
        table = align_table_to_contract(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY, table)
    return table


_MODULE = sys.modules[__name__]
_SUBSYSTEM_CACHES_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract(
        contract=SUBSYSTEM_PROFILE_CACHE_CONTRACT,
        input_type=pa.Table,
        node_name="subsystem_profile_cache__table",
    )
)
attach_table_target_template(_MODULE, spec=_SUBSYSTEM_CACHES_TABLE_TARGET_SPEC)
subsystem_profile_cache__table = _MODULE.subsystem_profile_cache__table
subsystem_caches__table_materializations = _MODULE.subsystem_caches__table_materializations
t__subsystem_caches = _MODULE.t__subsystem_caches


__all__ = [
    "subsystem_caches__table_materializations",
    "subsystem_profile_cache__base",
    "subsystem_profile_cache__table",
    "t__subsystem_caches",
]
