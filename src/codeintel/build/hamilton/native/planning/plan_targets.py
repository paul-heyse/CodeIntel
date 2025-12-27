"""Planning target anchors."""

from __future__ import annotations

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.core.schemas.tables.ci_plan_entries import CI_PLAN_ENTRIES_TABLE_KEY

PLAN_DOMAIN = "ops"
CI_PLAN_TARGET_NAME = "ci_plan"
CI_PLAN_JSON_ARTIFACT = "ci.plan.json"
CI_PLAN_EXPLAIN_ARTIFACT = "ci.plan.explain.md"


@codeintel_target(
    domain=PLAN_DOMAIN,
    target=CI_PLAN_TARGET_NAME,
    spec=TargetSpecDescriptor(spec_version="1"),
)
def t__ci_plan(
    env: BuildEnv,
    catalog: DagCatalog,
    m__artifact__ci__plan__json: MaterializationResult,
    m__artifact__ci__plan__explain__md: MaterializationResult,
    m__ci__plan_entries: MaterializationResult,
) -> TargetRunRecord:
    """Persist planning artifacts and emit a target record.

    Returns
    -------
    TargetRunRecord
        Run record for the ci_plan target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=CI_PLAN_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations={
            CI_PLAN_JSON_ARTIFACT: m__artifact__ci__plan__json,
            CI_PLAN_EXPLAIN_ARTIFACT: m__artifact__ci__plan__explain__md,
        },
        table_materializations={
            CI_PLAN_ENTRIES_TABLE_KEY: m__ci__plan_entries,
        },
    )


__all__ = [
    "CI_PLAN_EXPLAIN_ARTIFACT",
    "CI_PLAN_JSON_ARTIFACT",
    "CI_PLAN_TARGET_NAME",
    "PLAN_DOMAIN",
    "t__ci_plan",
]
