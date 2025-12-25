"""Typed tag specifications for Hamilton build nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, cast

from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from collections.abc import Mapping


TagValue = str | list[str]
TagKey = Literal[
    "domain",
    "target",
    "table_key",
    "artifact",
    "artifact_path_template",
    "node_type",
    "output_kind",
    "semantic_id",
    "entity",
    "grain",
    "mcp_visible",
    "tools",
    "target_resources",
    "target_execution",
    "target_parameters",
    "target_estimated_duration_ms",
    "target_spec_version",
]

_PRIMARY_TAG_KEYS: set[TagKey] = {
    cast("TagKey", ht.TAG_DOMAIN),
    cast("TagKey", ht.TAG_TARGET),
    cast("TagKey", ht.TAG_TABLE_KEY),
    cast("TagKey", ht.TAG_ARTIFACT),
    cast("TagKey", ht.TAG_NODE_TYPE),
}


class NodeType(str, Enum):
    """Canonical node type values for Hamilton tags."""

    LOADER_QUERY = ht.NODE_TYPE_LOADER_QUERY
    LOADER_DATAFRAME = ht.NODE_TYPE_LOADER_DATAFRAME
    DATASET = ht.NODE_TYPE_DATASET
    COMPUTE = ht.NODE_TYPE_COMPUTE
    MATERIALIZE = ht.NODE_TYPE_MATERIALIZE
    ARTIFACT = ht.NODE_TYPE_ARTIFACT
    TOOL = ht.NODE_TYPE_TOOL
    HELPER = ht.NODE_TYPE_HELPER


@dataclass(frozen=True, slots=True)
class TagSpec:
    """Typed tagging specification for Hamilton nodes."""

    node_type: NodeType
    domain: str
    target: str | None = None
    table_key: str | None = None
    artifact_name: str | None = None
    extra_tags: Mapping[TagKey, TagValue] = field(default_factory=dict)

    @classmethod
    def for_compute(
        cls,
        *,
        domain: str,
        target: str | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.COMPUTE,
            domain=domain,
            target=target,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    @classmethod
    def for_materialize(
        cls,
        *,
        domain: str,
        target: str | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.MATERIALIZE,
            domain=domain,
            target=target,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    @classmethod
    def for_dataset(
        cls,
        *,
        domain: str,
        target: str | None = None,
        table_key: str,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.DATASET,
            domain=domain,
            target=target,
            table_key=table_key,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    @classmethod
    def for_artifact(
        cls,
        *,
        domain: str,
        target: str | None = None,
        artifact_name: str,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.ARTIFACT,
            domain=domain,
            target=target,
            artifact_name=artifact_name,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    @classmethod
    def for_tool(
        cls,
        *,
        domain: str,
        target: str | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.TOOL,
            domain=domain,
            target=target,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    @classmethod
    def for_helper(
        cls,
        *,
        domain: str,
        target: str | None = None,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.HELPER,
            domain=domain,
            target=target,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    @classmethod
    def for_loader_query(
        cls,
        *,
        domain: str,
        target: str | None = None,
        table_key: str,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.LOADER_QUERY,
            domain=domain,
            target=target,
            table_key=table_key,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    @classmethod
    def for_loader_dataframe(
        cls,
        *,
        domain: str,
        target: str | None = None,
        table_key: str,
        extra_tags: Mapping[TagKey, TagValue] | None = None,
    ) -> TagSpec:
        return cls(
            node_type=NodeType.LOADER_DATAFRAME,
            domain=domain,
            target=target,
            table_key=table_key,
            extra_tags=_copy_extra_tags(extra_tags),
        )

    def with_extra_tags(self, extra_tags: Mapping[TagKey, TagValue] | None) -> TagSpec:
        """Return a new TagSpec with additional tags merged in."""
        if not extra_tags:
            return self
        merged = dict(self.extra_tags)
        merged.update(extra_tags)
        return TagSpec(
            node_type=self.node_type,
            domain=self.domain,
            target=self.target,
            table_key=self.table_key,
            artifact_name=self.artifact_name,
            extra_tags=merged,
        )

    def to_tags(self) -> dict[TagKey, TagValue]:
        """Render TagSpec into a Hamilton tag mapping."""
        validate_tag_spec(self)
        tags: dict[TagKey, TagValue] = {
            cast("TagKey", ht.TAG_NODE_TYPE): self.node_type.value,
            cast("TagKey", ht.TAG_DOMAIN): self.domain,
        }
        if self.target is not None:
            tags[cast("TagKey", ht.TAG_TARGET)] = self.target
        if self.table_key is not None:
            tags[cast("TagKey", ht.TAG_TABLE_KEY)] = self.table_key
        if self.artifact_name is not None:
            tags[cast("TagKey", ht.TAG_ARTIFACT)] = self.artifact_name

        for key, value in self.extra_tags.items():
            if key in _PRIMARY_TAG_KEYS:
                msg = f"extra_tags cannot override primary tag {key}"
                raise ValueError(msg)
            tags[key] = value

        return tags


def validate_tag_spec(tag_spec: TagSpec) -> None:
    """Validate TagSpec for required fields and tag consistency."""
    if not tag_spec.domain:
        msg = "TagSpec.domain is required"
        raise ValueError(msg)

    required_fields = _required_fields(tag_spec.node_type)
    for field_name in required_fields:
        value = getattr(tag_spec, field_name)
        if not value:
            msg = f"TagSpec.{field_name} is required for {tag_spec.node_type.value}"
            raise ValueError(msg)


def tag_spec_from_tags(tags: Mapping[str, TagValue]) -> TagSpec | None:
    """Build a TagSpec from a raw tag mapping."""
    node_type_value = tags.get(ht.TAG_NODE_TYPE)
    if not isinstance(node_type_value, str):
        return None
    try:
        node_type = NodeType(node_type_value)
    except ValueError:
        return None

    domain = tags.get(ht.TAG_DOMAIN)
    if not isinstance(domain, str):
        return None

    target_value = tags.get(ht.TAG_TARGET)
    target = target_value if isinstance(target_value, str) else None
    table_value = tags.get(ht.TAG_TABLE_KEY)
    table_key = table_value if isinstance(table_value, str) else None
    artifact_value = tags.get(ht.TAG_ARTIFACT)
    artifact_name = artifact_value if isinstance(artifact_value, str) else None

    extra_tags: dict[TagKey, TagValue] = {}
    for raw_key, raw_value in tags.items():
        if raw_key in _PRIMARY_TAG_KEYS:
            continue
        if raw_key in _tag_key_set():
            extra_tags[cast("TagKey", raw_key)] = raw_value

    return TagSpec(
        node_type=node_type,
        domain=domain,
        target=target,
        table_key=table_key,
        artifact_name=artifact_name,
        extra_tags=extra_tags,
    )


def _required_fields(node_type: NodeType) -> tuple[str, ...]:
    if node_type in {NodeType.LOADER_QUERY, NodeType.LOADER_DATAFRAME, NodeType.DATASET}:
        return ("table_key",)
    if node_type is NodeType.ARTIFACT:
        return ("artifact_name",)
    return ()


def _copy_extra_tags(extra_tags: Mapping[TagKey, TagValue] | None) -> dict[TagKey, TagValue]:
    if not extra_tags:
        return {}
    return dict(extra_tags)


def _tag_key_set() -> set[str]:
    return {
        ht.TAG_DOMAIN,
        ht.TAG_TARGET,
        ht.TAG_TABLE_KEY,
        ht.TAG_ARTIFACT,
        ht.TAG_ARTIFACT_PATH_TEMPLATE,
        ht.TAG_NODE_TYPE,
        ht.TAG_OUTPUT_KIND,
        ht.TAG_SEMANTIC_ID,
        ht.TAG_ENTITY,
        ht.TAG_GRAIN,
        ht.TAG_MCP_VISIBLE,
        ht.TAG_TOOLS,
        ht.TAG_TARGET_RESOURCES,
        ht.TAG_TARGET_EXECUTION,
        ht.TAG_TARGET_PARAMETERS,
        ht.TAG_TARGET_ESTIMATED_DURATION_MS,
        ht.TAG_TARGET_SPEC_VERSION,
    }


__all__ = [
    "NodeType",
    "TagKey",
    "TagSpec",
    "TagValue",
    "tag_spec_from_tags",
    "validate_tag_spec",
]
