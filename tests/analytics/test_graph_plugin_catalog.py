"""Catalog generation coverage for graph metric plugins."""

from __future__ import annotations

from typing import Any, cast

import pytest

from codeintel.analytics.graphs.catalog import (
    build_plugin_catalog,
    render_plugin_catalog_markdown,
)


def test_build_plugin_catalog_includes_expected_fields() -> None:
    """Catalog JSON should expose core metadata fields for each plugin."""
    catalog = build_plugin_catalog()
    plugins = cast("dict[str, dict[str, Any]]", catalog.get("plugins", {}))
    if not plugins:
        message = "Catalog should include at least one plugin entry"
        pytest.fail(message)
    first_meta = next(iter(plugins.values()))
    required = (
        "name",
        "description",
        "stage",
        "severity",
        "enabled_by_default",
        "depends_on",
        "provides",
        "requires",
        "resource_hints",
        "options_model",
        "options_default",
        "version_hash",
        "contract_checkers",
        "scope_aware",
        "supported_scopes",
        "requires_isolation",
        "isolation_kind",
        "config_schema_ref",
        "row_count_tables",
        "cache_populates",
        "cache_consumes",
    )
    missing = tuple(field for field in required if field not in first_meta)
    if missing:
        message = f"Catalog entries missing required fields: {missing}"
        pytest.fail(message)


def test_render_plugin_catalog_markdown_contains_examples() -> None:
    """Markdown render should include plugin names and plan/manifest examples."""
    catalog = build_plugin_catalog()
    plugins = cast("dict[str, dict[str, Any]]", catalog.get("plugins", {}))
    if not plugins:
        message = "Catalog should include at least one plugin for markdown render"
        pytest.fail(message)
    name = next(iter(plugins))
    markdown = render_plugin_catalog_markdown(catalog)
    if name not in markdown:
        message = "Plugin name should appear in markdown output"
        pytest.fail(message)
    if "Plan Output Examples" not in markdown or "Manifest excerpts" not in markdown:
        message = "Markdown output should include plan and manifest example sections"
        pytest.fail(message)
