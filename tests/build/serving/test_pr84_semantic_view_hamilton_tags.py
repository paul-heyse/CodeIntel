"""PR-84: semantic view metadata is discoverable via Hamilton tags."""

from __future__ import annotations

from typing import cast

from hamilton import driver

from codeintel.build.hamilton import tags as ht
from codeintel.storage.views import ibis_views
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


def test_semantic_view_decorator_applies_hamilton_tags() -> None:
    """semantic_view should apply `output_kind=semantic_view` and required tag fields."""
    dr = driver.Driver({}, ibis_views)
    nodes = dr.list_available_variables(
        tag_filter={ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW},
    )
    expect_true(len(nodes) > 0)

    # Assert at least one known semantic view is tagged as expected.
    match = next((n for n in nodes if n.tags.get(ht.TAG_SEMANTIC_ID) == "function.summary"), None)
    expect_true(match is not None)
    match_node = cast("driver.Variable", match)
    expect_equal(match_node.tags.get(ht.TAG_TABLE_KEY), "docs.v_function_summary")
