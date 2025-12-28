"""Canonical tag filter builders for Hamilton tag queries."""

from __future__ import annotations

from codeintel.core.hamilton import tags as ht


def tf_datasets(*, table_key: str | None = None) -> dict[str, object]:
    """Return a tag filter for dataset nodes.

    Returns
    -------
    dict[str, object]
        Tag filter matching dataset nodes.
    """
    tag_filter: dict[str, object] = {ht.TAG_NODE_TYPE: ht.NODE_TYPE_DATASET}
    if table_key is not None:
        tag_filter[ht.TAG_TABLE_KEY] = table_key
    return tag_filter


def tf_artifacts(*, artifact: str | None = None) -> dict[str, object]:
    """Return a tag filter for artifact nodes.

    Returns
    -------
    dict[str, object]
        Tag filter matching artifact nodes.
    """
    tag_filter: dict[str, object] = {ht.TAG_NODE_TYPE: ht.NODE_TYPE_ARTIFACT}
    if artifact is not None:
        tag_filter[ht.TAG_ARTIFACT] = artifact
    return tag_filter


def tf_semantic_views() -> dict[str, object]:
    """Return a tag filter for semantic view nodes.

    Returns
    -------
    dict[str, object]
        Tag filter matching semantic view nodes.
    """
    return {
        ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
        ht.TAG_MCP_VISIBLE: "1",
    }


def tf_schema_tables(*, table_key: str | None = None) -> dict[str, object]:
    """Return a tag filter for schema table outputs.

    Returns
    -------
    dict[str, object]
        Tag filter matching contract table saver nodes.
    """
    tag_filter: dict[str, object] = {
        "hamilton.data_saver": True,
        "output_role": "contract",
        ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_TABLE,
    }
    if table_key is not None:
        tag_filter[ht.TAG_TABLE_KEY] = table_key
    return tag_filter


def tf_savers(*, role: str | None = None, sink: str | None = None) -> dict[str, object]:
    """Return a tag filter for data saver nodes.

    Returns
    -------
    dict[str, object]
        Tag filter matching data saver nodes.
    """
    tag_filter: dict[str, object] = {"hamilton.data_saver": True}
    if role is not None:
        tag_filter["output_role"] = role
    if sink is not None:
        tag_filter["hamilton.data_saver.sink"] = sink
    return tag_filter


__all__ = ["tf_artifacts", "tf_datasets", "tf_savers", "tf_schema_tables", "tf_semantic_views"]
