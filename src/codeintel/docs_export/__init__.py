"""Doc export utilities for emitting CodeIntel analytics as JSONL or Parquet artifacts."""

DEFAULT_VALIDATION_SCHEMAS: list[str] = [
    "function_profile",
    "file_profile",
    "module_profile",
    "call_graph_edges",
    "symbol_use_edges",
    "test_coverage_edges",
    "test_profile",
    "behavioral_coverage",
]
