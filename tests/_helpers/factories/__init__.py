"""Factory functions for creating test data structures.

This module provides factory functions for creating test data including:
- Blank profile rows for analytics contract tests
- Config builders for snapshots and runtime options
"""

from __future__ import annotations

from tests._helpers.factories.config_factories import (
    GraphRuntimeOptionsKwargs,
    make_graph_runtime_options,
)
from tests._helpers.factories.step_config_factories import make_snapshot
from tests._helpers.fixtures.rows import (
    blank_behavioral_coverage_row,
    blank_file_profile_row,
    blank_function_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
    sample_file_profile_rows,
    sample_function_profile_rows,
    sample_module_profile_rows,
    sample_test_profile_rows,
)

__all__ = [
    "GraphRuntimeOptionsKwargs",
    "blank_behavioral_coverage_row",
    "blank_file_profile_row",
    "blank_function_profile_row",
    "blank_module_profile_row",
    "blank_test_profile_row",
    "make_graph_runtime_options",
    "make_snapshot",
    "sample_file_profile_rows",
    "sample_function_profile_rows",
    "sample_module_profile_rows",
    "sample_test_profile_rows",
]
