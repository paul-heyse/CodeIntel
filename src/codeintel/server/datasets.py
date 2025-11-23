"""Dataset registry helpers shared by FastAPI and MCP backends."""

from __future__ import annotations

from collections import OrderedDict
from typing import Literal

from codeintel.config.schemas.tables import TABLE_SCHEMAS

DOCS_VIEWS = {
    "v_function_summary": "docs.v_function_summary",
    "v_call_graph_enriched": "docs.v_call_graph_enriched",
    "v_test_to_function": "docs.v_test_to_function",
    "v_file_summary": "docs.v_file_summary",
    "v_function_profile": "docs.v_function_profile",
    "v_file_profile": "docs.v_file_profile",
    "v_module_profile": "docs.v_module_profile",
}


def _dataset_name(table_key: str) -> str:
    """
    Derive a stable dataset name from a fully qualified table key.

    Examples
    --------
    "core.ast_nodes" -> "ast_nodes"
    "analytics.function_metrics" -> "function_metrics"

    Returns
    -------
    str
        Dataset-safe name derived from the table key.
    """
    _, name = table_key.split(".", maxsplit=1)
    return name


PREVIEW_COLUMN_COUNT = 5


def build_dataset_registry(
    *, include_docs_views: Literal["include", "exclude"] = "include"
) -> dict[str, str]:
    """
    Build a deterministic dataset registry from the DuckDB schema registry.

    Parameters
    ----------
    include_docs_views:
        When set to ``"include"``, include docs.* views in addition to base tables.

    Returns
    -------
    dict[str, str]
        Mapping from dataset name to fully qualified table/view name.
    """
    registry: OrderedDict[str, str] = OrderedDict()
    for table_key in sorted(TABLE_SCHEMAS):
        name = _dataset_name(table_key)
        if name not in registry:
            registry[name] = table_key
    if include_docs_views == "include":
        for name, table in DOCS_VIEWS.items():
            if name not in registry:
                registry[name] = table
    return dict(registry)


def describe_dataset(name: str, table: str) -> str:
    """
    Produce a human-friendly description for a dataset/table.

    Parameters
    ----------
    name:
        Registry name for the dataset (ignored when schema metadata exists).
    table:
        Fully qualified table or view name.

    Returns
    -------
    str
        Description string including a column preview when available.
    """
    schema = TABLE_SCHEMAS.get(table)
    if schema is None:
        return f"DuckDB table/view {table}"
    column_names = ", ".join(col.name for col in schema.columns[:PREVIEW_COLUMN_COUNT])
    extra = "" if len(schema.columns) <= PREVIEW_COLUMN_COUNT else "..."
    return f"{name}: {table} ({column_names}{extra})"
