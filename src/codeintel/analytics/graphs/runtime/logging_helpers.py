"""Shared logging helpers for graph runtime plugin execution."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.graphs.runtime.model import GraphPluginRunRecord
from codeintel.config.steps_graphs import GraphRunScope

log = logging.getLogger(__name__)


def log_plugin_finish(
    record: GraphPluginRunRecord,
    runtime: GraphRuntime,
    scope: GraphRunScope,
    *,
    features: Sequence[str] | None = None,
) -> None:
    """Emit a structured finish log for a plugin run."""
    log.info(
        "graph_runtime.plugin.finish name=%s stage=%s status=%s duration_ms=%.2f attempts=%d",
        record.name,
        record.stage,
        record.status,
        record.duration_ms,
        record.attempts,
        extra={
            "metric": "graph_runtime",
            "op": record.name,
            "duration_ms": record.duration_ms,
            "use_gpu": runtime.use_gpu,
            "features": features or runtime.options.features,
            "plugin_status": record.status,
            "plugin_started_at": record.started_at.isoformat(),
            "plugin_ended_at": record.ended_at.isoformat(),
            "plugin_stage": record.stage,
            "plugin_attempts": record.attempts,
            "plugin_timeout_ms": record.timeout_ms,
            "plugin_severity": record.severity,
            "plugin_contracts": [c.status for c in record.contracts],
            "graph_run_id": record.run_id,
            "scope": scope,
        },
    )
