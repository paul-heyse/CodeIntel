"""Pre-built table specifications for common CLI outputs.

This module provides standard table specs for consistent formatting
across commands that output similar data.
"""

from __future__ import annotations

from codeintel.cli.rendering.table import ColumnSpec, TableSpec

OPERATIONS_TABLE = TableSpec(
    columns=(
        ColumnSpec("id", "Operation ID", style="cyan"),
        ColumnSpec("summary", "Summary"),
        ColumnSpec("tags", "Tags", style="dim"),
    ),
    title="Available Operations",
)


DATASETS_TABLE = TableSpec(
    columns=(
        ColumnSpec("table_key", "Table", style="cyan"),
        ColumnSpec("name", "Name"),
        ColumnSpec("row_count", "Rows", justify="right"),
        ColumnSpec("description", "Description", style="dim"),
    ),
    title="Datasets",
)


BUILD_TARGETS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Target", style="cyan"),
        ColumnSpec("module", "Module"),
        ColumnSpec("status", "Status"),
        ColumnSpec("duration", "Duration", justify="right"),
    ),
    title="Build Targets",
)


PLUGINS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Plugin", style="cyan"),
        ColumnSpec("version", "Version"),
        ColumnSpec("status", "Status"),
        ColumnSpec("capabilities", "Capabilities", style="dim"),
    ),
    title="Installed Plugins",
)


JOBS_TABLE = TableSpec(
    columns=(
        ColumnSpec("job_id", "Job ID", style="cyan"),
        ColumnSpec("operation", "Operation"),
        ColumnSpec("status", "Status"),
        ColumnSpec("created_at", "Created", style="dim"),
    ),
    title="Jobs",
)


HEALTH_TABLE = TableSpec(
    columns=(
        ColumnSpec("component", "Component", style="cyan"),
        ColumnSpec("status", "Status"),
        ColumnSpec("message", "Message"),
        ColumnSpec("latency_ms", "Latency", justify="right"),
    ),
    title="Health Status",
)


SUBSYSTEMS_TABLE = TableSpec(
    columns=(
        ColumnSpec("name", "Subsystem", style="cyan"),
        ColumnSpec("type", "Type"),
        ColumnSpec("count", "Count", justify="right"),
        ColumnSpec("description", "Description", style="dim"),
    ),
    title="Subsystems",
)


__all__ = [
    "BUILD_TARGETS_TABLE",
    "DATASETS_TABLE",
    "HEALTH_TABLE",
    "JOBS_TABLE",
    "OPERATIONS_TABLE",
    "PLUGINS_TABLE",
    "SUBSYSTEMS_TABLE",
]
