"""Compile semantic registry artifacts from Hamilton tag metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.semantic_tags import (
    TAG_SEMANTIC_ENTITY,
    TAG_SEMANTIC_GRAIN,
    TAG_SEMANTIC_KIND,
)
from codeintel.core.hamilton.tag_filters import tf_semantic_views
from codeintel.serving.semantic_compile import (
    CompiledSemanticRegistry,
    compile_semantic_registry_from_views,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Protocol

    from codeintel.core.schemas.provider import SchemaProvider

    class TagQueryLike(Protocol):
        def query(self, tag_filter: Mapping[str, object]) -> Iterable[object]: ...


@dataclass(frozen=True, slots=True)
class SemanticTagIssue:
    """Single semantic tag validation issue."""

    node: str
    message: str
    tag: str | None = None


class SemanticTagValidationError(ValueError):
    """Raised when semantic tag validation fails."""

    def __init__(self, issues: Iterable[SemanticTagIssue]) -> None:
        issues_tuple = tuple(issues)
        summary = ", ".join(
            f"{issue.node}:{issue.tag}" if issue.tag else issue.node for issue in issues_tuple
        )
        super().__init__(f"Semantic tag validation failed: {summary}")
        self.issues = issues_tuple


def compile_semantic_registry(
    *,
    tag_query: TagQueryLike,
    schema_provider: SchemaProvider,
    version: str = "v1",
) -> CompiledSemanticRegistry:
    """Compile semantic registry from Hamilton tag metadata.

    Parameters
    ----------
    tag_query
        Tag query helper bound to the Hamilton driver.
    schema_provider
        Schema provider for resolving table schemas.
    version
        Semantic registry version string.

    Returns
    -------
    CompiledSemanticRegistry
        Compiled registry with semantic view entries.

    Raises
    ------
    SemanticTagValidationError
        When semantic tag requirements are not satisfied.
    """
    variables = _semantic_variables(tag_query)
    issues: list[SemanticTagIssue] = []
    view_tags: dict[str, dict[str, str]] = {}

    for variable in variables:
        tags = _tags_from_variable(variable)
        if not tags:
            continue
        if _tag_str_value(tags, ht.TAG_MCP_VISIBLE) == "0":
            continue
        if not _is_semantic_output(tags):
            continue
        name = _variable_name(variable)
        issues.extend(_semantic_tag_issues(name=name, tags=tags))
        view_tags[name] = {str(k): _stringify_tag_value(v) for k, v in tags.items()}

    if issues:
        raise SemanticTagValidationError(issues)

    return compile_semantic_registry_from_views(
        schema_provider=schema_provider,
        view_tags=view_tags,
        version=version,
    )


def _semantic_variables(tag_query: TagQueryLike) -> tuple[object, ...]:
    seen: set[str] = set()
    collected: list[object] = []
    for variable in tag_query.query(tf_semantic_views()):
        key = _variable_key(variable)
        if key in seen:
            continue
        seen.add(key)
        collected.append(variable)
    for variable in tag_query.query({ht.TAG_LAYER: "semantic"}):
        key = _variable_key(variable)
        if key in seen:
            continue
        seen.add(key)
        collected.append(variable)
    return tuple(collected)


def _variable_key(variable: object) -> str:
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(id(variable))


def _variable_name(variable: object) -> str:
    name = getattr(variable, "name", None)
    return str(name) if name is not None else "unknown"


def _tags_from_variable(variable: object) -> Mapping[str, object] | None:
    tags = getattr(variable, "tags", None)
    if isinstance(tags, Mapping):
        return tags
    return None


def _stringify_tag_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, float, int)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    return str(value)


def _tag_str_value(tags: Mapping[str, object], key: str) -> str | None:
    value = tags.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _semantic_tag_value(tags: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = _tag_str_value(tags, key)
        if value is not None:
            return value
    return None


def _semantic_kind(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (ht.TAG_KIND, TAG_SEMANTIC_KIND))


def _semantic_entity(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (ht.TAG_ENTITY, TAG_SEMANTIC_ENTITY))


def _semantic_grain(tags: Mapping[str, object]) -> str | None:
    return _semantic_tag_value(tags, (ht.TAG_GRAIN, TAG_SEMANTIC_GRAIN))


def _is_semantic_output(tags: Mapping[str, object]) -> bool:
    output_kind = _tag_str_value(tags, ht.TAG_OUTPUT_KIND)
    layer = _tag_str_value(tags, ht.TAG_LAYER)
    return output_kind == ht.OUTPUT_KIND_SEMANTIC_VIEW or layer == "semantic"


def _missing_semantic_tag_issue(
    *,
    name: str,
    tag: str,
    message: str,
) -> SemanticTagIssue:
    return SemanticTagIssue(node=name, tag=tag, message=message)


def _semantic_layer_issue(
    *,
    name: str,
    tags: Mapping[str, object],
) -> SemanticTagIssue | None:
    layer = _tag_str_value(tags, ht.TAG_LAYER)
    if layer == "semantic":
        return None
    detail = "missing layer tag" if layer is None else f"layer tag must be 'semantic' (got {layer})"
    return _missing_semantic_tag_issue(
        name=name,
        tag=ht.TAG_LAYER,
        message=f"Semantic output {detail}",
    )


def _semantic_required_tag_issues(
    *,
    name: str,
    tags: Mapping[str, object],
    kind: str | None,
) -> list[SemanticTagIssue]:
    required = (
        (
            ht.TAG_SEMANTIC_ID,
            _tag_str_value(tags, ht.TAG_SEMANTIC_ID),
            "Semantic output missing semantic_id tag",
        ),
        (ht.TAG_KIND, kind, "Semantic output missing kind tag"),
        (ht.TAG_ENTITY, _semantic_entity(tags), "Semantic output missing entity tag"),
        (ht.TAG_GRAIN, _semantic_grain(tags), "Semantic output missing grain tag"),
        (
            ht.TAG_VERSION,
            _tag_str_value(tags, ht.TAG_VERSION),
            "Semantic output missing version tag",
        ),
    )
    issues: list[SemanticTagIssue] = []
    for tag, value, message in required:
        if value is None:
            issues.append(_missing_semantic_tag_issue(name=name, tag=tag, message=message))
    return issues


def _semantic_table_tag_issues(
    *,
    name: str,
    tags: Mapping[str, object],
    kind: str | None,
) -> list[SemanticTagIssue]:
    if kind != "table":
        return []
    required = (
        (ht.TAG_SCHEMA_REF, "Semantic table missing schema_ref tag"),
        (ht.TAG_ENTITY_KEYS, "Semantic table missing entity_keys tag"),
        (ht.TAG_JOIN_KEYS, "Semantic table missing join_keys tag"),
    )
    issues: list[SemanticTagIssue] = []
    for tag, message in required:
        if _tag_str_value(tags, tag) is None:
            issues.append(_missing_semantic_tag_issue(name=name, tag=tag, message=message))
    return issues


def _semantic_tag_issues(
    *,
    name: str,
    tags: Mapping[str, object],
) -> list[SemanticTagIssue]:
    issues: list[SemanticTagIssue] = []
    layer_issue = _semantic_layer_issue(name=name, tags=tags)
    if layer_issue is not None:
        issues.append(layer_issue)
    kind = _semantic_kind(tags)
    issues.extend(_semantic_required_tag_issues(name=name, tags=tags, kind=kind))
    issues.extend(_semantic_table_tag_issues(name=name, tags=tags, kind=kind))
    return issues


__all__ = [
    "CompiledSemanticRegistry",
    "SemanticTagIssue",
    "SemanticTagValidationError",
    "compile_semantic_registry",
]
