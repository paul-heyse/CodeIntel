"""Validate DataSaver nodes have canonical tags for IO introspection."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.core.hamilton import tags as ht


def _get_saver_tags(tags: object) -> dict[str, object] | None:
    if not isinstance(tags, dict):
        return None
    if tags.get("hamilton.data_saver") is not True:
        return None
    return tags


def _saver_tag_errors(node_name: str, tags: dict[str, object]) -> list[str]:
    target = tags.get(ht.TAG_TARGET)
    if not isinstance(target, str) or not target:
        return [f"{node_name}: missing target tag"]

    output_role = tags.get("output_role")
    if output_role not in {"contract", "internal"}:
        return [f"{node_name}: missing/invalid output_role tag"]

    if output_role != "contract":
        return []

    table_key = tags.get(ht.TAG_TABLE_KEY)
    artifact = tags.get(ht.TAG_ARTIFACT)
    has_table = isinstance(table_key, str) and bool(table_key)
    has_artifact = isinstance(artifact, str) and bool(artifact)

    errors: list[str] = []
    if not (has_table or has_artifact):
        errors.append(f"{node_name}: missing table_key/artifact tag")
        return errors

    if has_artifact:
        path_template = tags.get(ht.TAG_ARTIFACT_PATH_TEMPLATE)
        if not isinstance(path_template, str) or not path_template:
            errors.append(f"{node_name}: missing artifact_path_template tag")

    return errors


def test_saver_nodes_have_canonical_tags() -> None:
    """Ensure DataSaver nodes expose canonical tags for inventory/contract checks."""
    runtime = build_driver()

    missing: list[str] = []
    for node_name, node in runtime.dr.graph.nodes.items():
        tags = _get_saver_tags(node.tags)
        if tags is None:
            continue
        missing.extend(_saver_tag_errors(node_name, tags))

    if missing:
        pytest.fail("Saver nodes missing canonical tags:\n" + "\n".join(sorted(missing)))
