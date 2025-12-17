"""Native Hamilton target support.

This package contains pure Hamilton implementations of build targets,
replacing the plugin-based wrapper approach with explicit compute + materialize nodes.
"""

from __future__ import annotations

from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.ibis_helpers import (
    filter_for_snapshot,
    filter_tables_for_snapshot,
    select_snapshot_columns,
)
from codeintel.build.hamilton.run_records import (
    NativeRunInfo,
    create_run_record,
    save_manifest,
    should_skip_native_target,
)

__all__ = [
    "NativeRunInfo",
    "NativeTargetExecutor",
    "create_run_record",
    "filter_for_snapshot",
    "filter_tables_for_snapshot",
    "save_manifest",
    "select_snapshot_columns",
    "should_skip_native_target",
]
