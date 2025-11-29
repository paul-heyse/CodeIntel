"""Guards to ensure function_profile row serialization matches schema."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.storage.rows import (
    FUNCTION_PROFILE_COLUMNS,
    FunctionProfileRowModel,
    function_profile_row_to_tuple,
)


def test_function_profile_tuple_length_matches_columns() -> None:
    """function_profile_row_to_tuple must align with column list length."""
    row: FunctionProfileRowModel = {
        key: None
        for key in FUNCTION_PROFILE_COLUMNS  # type: ignore[arg-type]
    }
    # Populate required non-null fields with minimal placeholders.
    row["function_goid_h128"] = 1
    row["repo"] = "r"
    row["commit"] = "c"
    row["rel_path"] = "p.py"
    row["loc"] = row["logical_loc"] = row["cyclomatic_complexity"] = 1
    row["param_count"] = row["positional_params"] = row["keyword_params"] = 0
    row["vararg"] = row["kwarg"] = False
    row["total_params"] = row["annotated_params"] = 0
    row["fully_typed"] = row["partial_typed"] = row["untyped"] = False
    row["static_error_count"] = 0
    row["has_static_errors"] = False
    row["executable_lines"] = row["covered_lines"] = 0
    row["tested"] = False
    row["tests_touching"] = row["failing_tests"] = row["slow_tests"] = row["flaky_tests"] = 0
    row["slow_test_threshold_ms"] = 0.0
    row["commit_count"] = row["author_count"] = row["lines_added"] = row["lines_deleted"] = 0
    row["call_fan_in"] = row["call_fan_out"] = row["call_edge_in_count"] = row[
        "call_edge_out_count"
    ] = 0
    row["call_is_leaf"] = row["call_is_entrypoint"] = row["call_is_public"] = False
    row["risk_score"] = row["risk_component_coverage"] = row["risk_component_complexity"] = row[
        "risk_component_static"
    ] = row["risk_component_hotspot"] = 0.0
    row["is_pure"] = row["uses_io"] = row["touches_db"] = row["uses_time"] = False
    row["uses_randomness"] = row["modifies_globals"] = row["modifies_closure"] = False
    row["spawns_threads_or_tasks"] = row["has_transitive_effects"] = False
    row["has_preconditions"] = row["has_postconditions"] = row["has_raises"] = False
    row["role_sources_json"] = row["param_types"] = row["param_nullability_json"] = {}
    row["tags"] = row["owners"] = row["doc_params"] = row["doc_returns"] = {}
    row["created_at"] = datetime.now(tz=UTC)

    values = function_profile_row_to_tuple(row)
    assert len(values) == len(FUNCTION_PROFILE_COLUMNS)
