"""Subsystem agreement checks built with inferable tabular nodes."""

from __future__ import annotations

import sys

import polars as pl

from codeintel.build.analytics.graphs.subsystem_agreement import (
    build_subsystem_agreement_frame,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
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

SUBSYSTEM_AGREEMENT_TARGET_NAME = "subsystem_agreement"
SUBSYSTEM_AGREEMENT_TABLE_KEY = "analytics.subsystem_agreement"
SUBSYSTEM_AGREEMENT_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_agreement__base",
)


def subsystem_agreement__base(
    env: BuildEnv,
    q__analytics__subsystem_modules: InferableTabularInput,
    q__analytics__graph_metrics_modules_ext: InferableTabularInput,
) -> pl.LazyFrame:
    """Build subsystem agreement rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem agreement rows.
    """
    subsystem_frame = tabular_to_lazyframe(q__analytics__subsystem_modules).select(
        ["repo", "commit", "module", "subsystem_id"]
    )
    metrics_frame = tabular_to_lazyframe(q__analytics__graph_metrics_modules_ext).select(
        ["repo", "commit", "module", "import_community_id"]
    )
    return build_subsystem_agreement_frame(
        repo=env.repo,
        commit=env.commit,
        subsystem_modules=subsystem_frame,
        graph_metrics_modules=metrics_frame,
    )


_MODULE = sys.modules[__name__]
_SUBSYSTEM_AGREEMENT_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY,
            base_node="subsystem_agreement__base",
            contract=SUBSYSTEM_AGREEMENT_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY),
            node_name="subsystem_agreement__table",
        ),
    ),
    table_materializations_node="subsystem_agreement__table_materializations",
    anchor_node_name="t__subsystem_agreement",
)
attach_table_target_template(_MODULE, spec=_SUBSYSTEM_AGREEMENT_TABLE_TARGET_SPEC)
subsystem_agreement__table = _MODULE.subsystem_agreement__table
subsystem_agreement__table_materializations = _MODULE.subsystem_agreement__table_materializations
t__subsystem_agreement = _MODULE.t__subsystem_agreement


__all__ = [
    "subsystem_agreement__base",
    "subsystem_agreement__table",
    "t__subsystem_agreement",
]
