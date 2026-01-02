"""Graph validation table built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.validation.runner import (
    GraphValidationRunRequest,
    run_graph_validations_with_runner,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import (
    empty_frame_for_table,
    rows_to_frame,
)
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
from codeintel.core.validation.reporters import GraphValidationReporter

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

GRAPH_VALIDATION_TARGET_NAME = "graph_validation"
GRAPH_VALIDATION_TABLE_KEY = "analytics.graph_validation"
GRAPH_VALIDATION_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=GRAPH_VALIDATION_TARGET_NAME,
)
GRAPH_VALIDATION_CONTRACT = TableContractSpec(
    table_key=GRAPH_VALIDATION_TABLE_KEY,
    domain="analytics",
    target=GRAPH_VALIDATION_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="graph_validation__base",
)


def _report_findings(reporter: GraphValidationReporter, findings: list[dict[str, object]]) -> None:
    for finding in findings:
        graph_name = str(finding.get("check_name") or "graph_validation")
        entity_ref = finding.get("path") or finding.get("entity_id") or finding.get("graph_name")
        entity_id = str(entity_ref) if entity_ref is not None else graph_name
        issue = str(finding.get("issue") or finding.get("severity") or graph_name)
        severity = str(finding.get("severity") or "info")
        rel_path = finding.get("path")
        detail = str(finding.get("detail") or "")
        metadata = finding.get("context")
        extras = {
            "severity": severity,
            "rel_path": str(rel_path) if rel_path is not None else None,
            "metadata": metadata,
        }
        reporter.record(
            graph_name=graph_name,
            entity_id=entity_id,
            issue=issue,
            detail=detail,
            extras=extras,
        )


def graph_validation__base(
    env: BuildEnv,
) -> pl.LazyFrame:
    """Build graph validation rows from validation findings.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing graph validation rows.
    """
    runtime = GraphRuntimeOptions(
        snapshot=env.snapshot,
        dataset_root_dir=env.paths.dataset_root_dir,
    )
    request = GraphValidationRunRequest(
        snapshot=env.snapshot,
        runtime=runtime,
        dataset_root_dir=env.paths.dataset_root_dir,
    )
    report = run_graph_validations_with_runner(request=request)
    if not report.findings:
        return empty_frame_for_table(GRAPH_VALIDATION_TABLE_KEY)
    reporter = GraphValidationReporter(repo=env.repo, commit=env.commit)
    _report_findings(reporter, report.findings)
    rows = reporter.to_rows()
    return rows_to_frame(GRAPH_VALIDATION_TABLE_KEY, rows)


@save_dataset(
    context=GRAPH_VALIDATION_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=GRAPH_VALIDATION_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=GRAPH_VALIDATION_TARGET_NAME,
    table_key=GRAPH_VALIDATION_TABLE_KEY,
)
@table_contract(GRAPH_VALIDATION_CONTRACT)
def graph_validation__table(graph_validation__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist graph validation rows.

    Returns
    -------
    pl.LazyFrame
        Persisted graph validation frame.
    """
    return graph_validation__base


@codeintel_target(domain="analytics", target=GRAPH_VALIDATION_TARGET_NAME)
def t__graph_validation(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__graph_validation: MaterializationResult,
) -> TargetRunRecord:
    """Finalize graph validation target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the graph validation target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=GRAPH_VALIDATION_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            GRAPH_VALIDATION_TABLE_KEY: m__analytics__graph_validation,
        },
    )


__all__ = [
    "graph_validation__base",
    "graph_validation__table",
    "t__graph_validation",
]
