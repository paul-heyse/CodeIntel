"""Subsystem agreement checks built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pyarrow as pa

from codeintel.build.analytics.graphs.subsystem_agreement import (
    SubsystemAgreementInputs,
    build_subsystem_agreement_rows,
)
from codeintel.build.contracts.registry import require_contract
from codeintel.build.contracts.types import ContractOverrides
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEM_AGREEMENT_TARGET_NAME = "subsystem_agreement"
SUBSYSTEM_AGREEMENT_TABLE_KEY = "analytics.subsystem_agreement"
SUBSYSTEM_AGREEMENT_CONTRACT = require_contract(
    table_key=SUBSYSTEM_AGREEMENT_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEM_AGREEMENT_TARGET_NAME,
    overrides=ContractOverrides(
        input_name="subsystem_agreement__base",
        required_cols=(),
        clip_column=None,
    ),
)


def subsystem_agreement__base(
    env: BuildEnv,
    q__analytics__subsystem_modules: InferableTabularInput,
    q__analytics__graph_metrics_modules_ext: InferableTabularInput,
) -> pa.Table:
    """Build subsystem agreement rows.

    Returns
    -------
    pa.Table
        Reader containing subsystem agreement rows.
    """
    subsystem_table = tabular_to_arrow_table(q__analytics__subsystem_modules).select(
        ["repo", "commit", "module", "subsystem_id"]
    )
    metrics_table = tabular_to_arrow_table(q__analytics__graph_metrics_modules_ext).select(
        ["repo", "commit", "module", "import_community_id"]
    )
    inputs = SubsystemAgreementInputs(
        repo=env.repo,
        commit=env.commit,
        subsystem_module_rows=list(iter_rows(subsystem_table)),
        graph_metrics_module_rows=list(iter_rows(metrics_table)),
    )
    rows = build_subsystem_agreement_rows(inputs)
    if not rows:
        return empty_table_for_table(SUBSYSTEM_AGREEMENT_TABLE_KEY)
    reader, _ = table_for_rows(SUBSYSTEM_AGREEMENT_TABLE_KEY, rows)
    return reader


_MODULE = sys.modules[__name__]
_SUBSYSTEM_AGREEMENT_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract(
        contract=SUBSYSTEM_AGREEMENT_CONTRACT,
        input_type=pa.Table,
    )
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
