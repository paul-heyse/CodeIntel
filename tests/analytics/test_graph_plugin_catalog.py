"""Catalog generation coverage for analytics plugins.

.. deprecated::
    The plugin catalog is deprecated. All plugins have been migrated to
    native Hamilton modules. Use ``codeintel.build.hamilton.native`` instead.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from codeintel.analytics.graphs.plugin_catalog import (
    build_plugin_catalog,
    render_plugin_catalog_markdown,
)


@pytest.mark.skip(reason="Plugin catalog deprecated - all plugins migrated to Hamilton native")
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
        "version",
        "version_hash",
        "stage",
        "enabled_by_default",
        "depends_on",
        "provides",
        "requires",
    )
    missing = tuple(field for field in required if field not in first_meta)
    if missing:
        message = f"Catalog entries missing required fields: {missing}"
        pytest.fail(message)


@pytest.mark.skip(reason="Plugin catalog deprecated - all plugins migrated to Hamilton native")
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
