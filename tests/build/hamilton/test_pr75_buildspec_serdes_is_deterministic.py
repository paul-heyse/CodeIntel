"""Tests for PR-75: BuildSpec primitives + deterministic JSON + hashing."""

from __future__ import annotations

import pytest

from codeintel.build.spec import (
    ArtifactOutSpec,
    BuildSpec,
    DatasetSpec,
    TargetSpec,
    buildspec_from_json,
    buildspec_to_json,
)


def test_buildspec_serdes_is_deterministic() -> None:
    """Serialize twice and require byte-identical output."""
    spec = BuildSpec(
        spec_version=1,
        targets=(
            TargetSpec(
                name="b",
                domain="analytics",
                impl_kind="native",
                deps=("z", "a"),
                outputs=("analytics.zz", "analytics.aa"),
                artifacts=(ArtifactOutSpec(name="out_b", kind="jsonl"),),
            ),
            TargetSpec(
                name="a",
                domain="core",
                impl_kind="native",
                deps=(),
                outputs=("core.modules",),
                artifacts=(),
            ),
        ),
        datasets=(
            DatasetSpec(table_key="core.modules", schema_hash="h1", columns=("module", "path")),
            DatasetSpec(table_key="analytics.aa", schema_hash="h2", columns=None),
        ),
    )

    out1 = buildspec_to_json(spec, indent=2)
    out2 = buildspec_to_json(spec, indent=2)

    if out1 != out2:
        pytest.fail("BuildSpec JSON output is not deterministic across two serializations")


def test_buildspec_roundtrip_preserves_hash() -> None:
    """Roundtrip JSON and require hash-stable serialization."""
    spec = BuildSpec(
        spec_version=1,
        targets=(
            TargetSpec(
                name="x",
                domain="graphs",
                impl_kind="native",
                deps=("a", "b"),
                outputs=("graph.call_graph_edges",),
                artifacts=(),
            ),
        ),
        datasets=(DatasetSpec(table_key="graph.call_graph_edges", schema_hash="deadbeef"),),
    )

    text = buildspec_to_json(spec, indent=2)
    parsed = buildspec_from_json(text)

    out_again = buildspec_to_json(parsed, indent=2)
    if text != out_again:
        pytest.fail("BuildSpec JSON roundtrip did not preserve stable output")
