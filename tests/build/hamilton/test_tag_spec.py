"""Tests for TagSpec validation helpers."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.tag_spec import (
    NodeType,
    TagSpec,
    tag_spec_from_tags,
    validate_tag_spec,
)


def test_tag_spec_requires_table_key_for_loader() -> None:
    """Require table_key for loader TagSpec validation."""
    spec = TagSpec.for_loader_query(
        domain="analytics",
        target="hotspots",
        table_key="core.modules",
    )
    validate_tag_spec(spec)

    bad_spec = TagSpec(
        node_type=NodeType.LOADER_QUERY,
        domain="analytics",
        target="hotspots",
    )
    with pytest.raises(ValueError, match="table_key"):
        validate_tag_spec(bad_spec)


def test_tag_spec_from_tags_parses_extra_tags() -> None:
    """Parse extra tags into TagSpec from raw tags."""
    tags = {
        "node_type": NodeType.COMPUTE.value,
        "domain": "analytics",
        "target": "hotspots",
        "output_kind": "view",
    }
    spec = tag_spec_from_tags(tags)
    assert spec is not None
    assert spec.node_type is NodeType.COMPUTE
    assert spec.extra_tags["output_kind"] == "view"


def test_tag_spec_rejects_primary_tag_overrides() -> None:
    """Reject attempts to override primary tags via extra_tags."""
    spec = TagSpec.for_compute(
        domain="analytics",
        target="hotspots",
        extra_tags={"node_type": NodeType.COMPUTE.value},
    )
    with pytest.raises(ValueError, match="primary tag node_type"):
        spec.to_tags()
