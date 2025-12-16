"""PR-86: view discovery no longer relies on a global registry module."""

from __future__ import annotations

from importlib.util import find_spec

from codeintel.storage.views import ibis_views
from codeintel.storage.views.discovery import discover_view_builders
from tests._helpers.assertions.expectation_assertions import expect_true


def test_ibis_view_registry_module_removed() -> None:
    """The legacy registry module should not exist after migration."""
    expect_true(find_spec("codeintel.storage.views.ibis_registry") is None)


def test_view_builders_discoverable_via_tags() -> None:
    """Both semantic and non-semantic views should be discoverable from tags."""
    builders = discover_view_builders(modules=(ibis_views,))
    table_keys = {b.table_key for b in builders}
    expect_true("docs.v_function_summary" in table_keys)
    expect_true("analytics.v_function_summary" in table_keys)
