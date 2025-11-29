"""Export pipeline step."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.pipeline.export.export_jsonl import ExportCallOptions, export_all_jsonl
from codeintel.pipeline.export.export_parquet import export_all_parquet
from codeintel.pipeline.orchestration.core import (
    PipelineContext,
    PipelineStep,
    StepPhase,
    _log_step,
)
from codeintel.serving.http.datasets import validate_dataset_registry
from codeintel.storage.views import create_all_views

log = logging.getLogger(__name__)


@dataclass
class ExportDocsStep:
    """Export all Parquet + JSONL datasets into Document Output/."""

    name: str = "export_docs"
    description: str = "Export all Parquet and JSONL datasets into Document Output/."
    phase: StepPhase = StepPhase.EXPORT
    deps: Sequence[str] = (
        "repo_scan",
        "scip_ingest",
        "cst_extract",
        "ast_extract",
        "coverage_ingest",
        "tests_ingest",
        "typing_ingest",
        "docstrings_ingest",
        "config_ingest",
        "function_metrics",
        "function_effects",
        "function_contracts",
        "data_models",
        "data_model_usage",
        "config_data_flow",
        "coverage_functions",
        "test_coverage_edges",
        "hotspots",
        "function_history",
        "risk_factors",
        "graph_metrics",
        "subsystems",
        "semantic_roles",
        "entrypoints",
        "external_dependencies",
        "test_profile",
        "behavioral_coverage",
        "profiles",
        "history_timeseries",
        "callgraph",
        "cfg",
        "import_graph",
        "symbol_uses",
        "graph_validation",
    )

    def run(self, ctx: PipelineContext) -> None:
        """Create views and export Parquet/JSONL artifacts."""
        _log_step(self.name)
        con = ctx.gateway.con
        create_all_views(con)
        validate_dataset_registry(ctx.gateway)
        datasets = list(ctx.export_datasets) if ctx.export_datasets is not None else None
        export_opts = ExportCallOptions(
            validate_exports=False,
            datasets=datasets,
            validation_profile=ctx.export_validation_profile,
            force_full_export=ctx.force_full_export,
        )
        export_all_parquet(
            ctx.gateway,
            ctx.document_output_dir,
            options=export_opts,
        )
        export_all_jsonl(
            ctx.gateway,
            ctx.document_output_dir,
            options=export_opts,
        )
        log.info("Document Output refreshed at %s", ctx.document_output_dir)


EXPORT_STEPS: dict[str, PipelineStep] = {
    "export_docs": ExportDocsStep(),
}


__all__ = ["EXPORT_STEPS", "ExportDocsStep"]
