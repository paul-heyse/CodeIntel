"""PR-84: semantic view metadata is discoverable via Hamilton tags."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from hamilton.driver import Driver

from codeintel.build.hamilton.native.export.serving_artifacts import (
    SERVING_ARTIFACT_BUILDSPEC,
    SERVING_ARTIFACT_DATASET_MANIFEST_PATHS,
    SERVING_ARTIFACT_SCHEMA_MANIFEST,
    SERVING_ARTIFACT_SEMANTIC_REGISTRY,
    SERVING_ARTIFACTS_TARGET_NAME,
)
from codeintel.core.hamilton import tags as ht
from codeintel.storage.views import sqlglot_views
from tests._helpers.assertions import (
    assert_record_has_artifacts,
    assert_target_ok,
    expect_equal,
    expect_true,
)
from tests._helpers.harnesses.serving_harness import ServingTargetHarness

if TYPE_CHECKING:
    from hamilton.driver import HamiltonNode


def test_semantic_view_decorator_applies_hamilton_tags() -> None:
    """semantic_view should apply `output_kind=semantic_view` and required tag fields."""
    dr = Driver({}, sqlglot_views)
    nodes = dr.list_available_variables(
        tag_filter={ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW},
    )
    expect_true(len(nodes) > 0)

    # Assert at least one known semantic view is tagged as expected.
    match = next((n for n in nodes if n.tags.get(ht.TAG_SEMANTIC_ID) == "function.summary"), None)
    expect_true(match is not None)
    match_node = cast("HamiltonNode", match)
    expect_equal(match_node.tags.get(ht.TAG_TABLE_KEY), "docs.v_function_summary")


def test_serving_harness_emits_semantic_artifacts(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Serving harness should emit semantic registry artifacts."""
    records = serving_target_harness.run_targets([SERVING_ARTIFACTS_TARGET_NAME])
    record = records[SERVING_ARTIFACTS_TARGET_NAME]
    assert_target_ok(record)
    assert_record_has_artifacts(
        record,
        (
            SERVING_ARTIFACT_SEMANTIC_REGISTRY,
            SERVING_ARTIFACT_SCHEMA_MANIFEST,
            SERVING_ARTIFACT_BUILDSPEC,
            SERVING_ARTIFACT_DATASET_MANIFEST_PATHS,
        ),
    )
