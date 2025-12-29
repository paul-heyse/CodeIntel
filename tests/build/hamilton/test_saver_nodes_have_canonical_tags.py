"""Validate DataSaver nodes have canonical tags for IO introspection."""

from __future__ import annotations

import pytest

from codeintel.core.hamilton import tags as ht
from codeintel.runtime.runtime_bundle import RuntimeBundle


def _variable_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(variable)


def _output_kind_matches(value: object, expected: str) -> bool:
    if isinstance(value, str):
        return value == expected
    if isinstance(value, list):
        return expected in value
    return False


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

    if has_table and not _output_kind_matches(tags.get(ht.TAG_OUTPUT_KIND), ht.OUTPUT_KIND_TABLE):
        errors.append(f"{node_name}: missing output_kind=table tag")

    if has_artifact:
        path_template = tags.get(ht.TAG_ARTIFACT_PATH_TEMPLATE)
        if not isinstance(path_template, str) or not path_template:
            errors.append(f"{node_name}: missing artifact_path_template tag")

    materialization = tags.get(ht.TAG_MATERIALIZATION)
    if not isinstance(materialization, str) or not materialization:
        errors.append(f"{node_name}: missing materialization tag")

    materialized_name = tags.get(ht.TAG_MATERIALIZED_NAME)
    if not isinstance(materialized_name, str) or not materialized_name:
        errors.append(f"{node_name}: missing materialized_name tag")

    return errors


def test_saver_nodes_have_canonical_tags(hamilton_runtime: RuntimeBundle) -> None:
    """Ensure DataSaver nodes expose canonical tags for inventory/contract checks."""
    missing: list[str] = []
    variables = hamilton_runtime.tag_query.query({"hamilton.data_saver": True})
    for variable in variables:
        tags = getattr(variable, "tags", None)
        if not isinstance(tags, dict):
            continue
        missing.extend(_saver_tag_errors(_variable_name(variable), tags))

    if missing:
        pytest.fail("Saver nodes missing canonical tags:\n" + "\n".join(sorted(missing)))
