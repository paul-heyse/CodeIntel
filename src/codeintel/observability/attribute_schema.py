"""Attribute schema registry and normalization helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from codeintel.observability.attribute_sanitizer import (
    SpanAttributeValue,
    coerce_attribute_value,
)
from codeintel.observability.policy import AttributeBudget, ObservabilityPolicy
from codeintel.observability.semconv_keys import (
    BUILD_COMMIT,
    BUILD_DECISION_TRACE_ARTIFACT,
    BUILD_DURATION_MS,
    BUILD_REPO,
    BUILD_RUN_ID,
    BUILD_SCHEMA_INFERENCE_ERRORS_COUNT,
    BUILD_TARGETS,
    BUILD_VALIDATION_ISSUE_COUNT,
    BUILD_VALIDATION_MODE,
    CLI_ARG_COUNT,
    CLI_ARG_NAMES,
    CLI_COMMAND,
    CLI_DURATION_MS,
    CLI_ERROR_TYPE,
    CLI_EXIT_CODE,
    CLI_INVOCATION_ID,
    CLI_IS_PARSE_ERROR,
    CLI_PARSE_DURATION_MS,
    CODEINTEL_ACTOR,
    CODEINTEL_COMMIT,
    CODEINTEL_COMPONENT,
    CODEINTEL_CORRELATION_ID,
    CODEINTEL_DB_STATEMENT_SHA256,
    CODEINTEL_DOMAIN,
    CODEINTEL_ENDPOINT,
    CODEINTEL_HEALTH_CHECK,
    CODEINTEL_OPERATION,
    CODEINTEL_OUTPUT_FORMAT,
    CODEINTEL_QUERY_BATCH_SIZE,
    CODEINTEL_QUERY_ENDPOINT,
    CODEINTEL_QUERY_ENGINE,
    CODEINTEL_QUERY_ENGINE_PREFERENCE,
    CODEINTEL_QUERY_HASH,
    CODEINTEL_QUERY_ROW_COUNT,
    CODEINTEL_QUERY_SCAN_BYTES,
    CODEINTEL_QUERY_SCAN_FILES,
    CODEINTEL_QUERY_SCAN_ROWS,
    CODEINTEL_QUERY_SCHEMA_HASH,
    CODEINTEL_QUERY_TRUNCATED,
    CODEINTEL_QUERY_VIEW_ID,
    CODEINTEL_REPO,
    CODEINTEL_RUN_ID,
    CODEINTEL_STORAGE_READ_ONLY,
    CODEINTEL_SUCCESS,
    DB_NAMESPACE,
    DB_QUERY_PARAMETER_PREFIX,
    DB_QUERY_SUMMARY,
    DB_QUERY_TEXT,
    DB_STATEMENT,
    DB_SYSTEM_NAME,
    HTTP_METHOD,
    HTTP_ROUTE,
    MCP_METHOD,
    MCP_TOOL_NAME,
    SCIP_COMMIT,
    SCIP_DURATION_MS,
    SCIP_ERROR,
    SCIP_MODE,
    SCIP_REPO,
    SCIP_RUN_ID,
    SCIP_STATUS,
    SHUTDOWN_ACTIVE_THREAD_NAMES,
    SHUTDOWN_ACTIVE_THREADS_COUNT,
    SHUTDOWN_ERROR_MESSAGE,
    SHUTDOWN_ERROR_TYPE,
    SHUTDOWN_PENDING_TASK_SAMPLES,
    SHUTDOWN_PENDING_TASKS_COUNT,
    SHUTDOWN_STATUS,
    SHUTDOWN_SUBPROCESS_COUNT,
    SHUTDOWN_SUBPROCESS_SAMPLES,
    TELEMETRY_ACTION,
    TELEMETRY_FLUSH_MS,
    TELEMETRY_FLUSH_OK,
    TELEMETRY_INSTRUMENTATION_NAME,
    TELEMETRY_INSTRUMENTATION_STATUS,
)


class CardinalityTier(StrEnum):
    """Cardinality classification for attribute keys."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class AttributeSchema:
    """Schema describing an attribute key and its allowed values."""

    key: str
    value_types: tuple[type, ...]
    cardinality: CardinalityTier = CardinalityTier.LOW
    max_length: int | None = None
    match_prefix: bool = False
    allow_sequence: bool = False

    def matches_key(self, key: str) -> bool:
        """Return True when this schema matches the supplied key.

        Returns
        -------
        bool
            True when the schema matches the key.
        """
        if self.match_prefix:
            return key.startswith(self.key)
        return key == self.key

    def is_value_allowed(self, value: SpanAttributeValue) -> bool:
        """Return True when the value matches the schema.

        Returns
        -------
        bool
            True when the value is allowed by the schema.
        """
        if self.allow_sequence:
            if isinstance(value, str):
                return False
            if not isinstance(value, Sequence):
                return False
            return all(isinstance(item, self.value_types) for item in value)
        return isinstance(value, self.value_types)


@dataclass(frozen=True, slots=True)
class SchemaOptions:
    """Optional schema configuration overrides for attribute keys."""

    cardinality: CardinalityTier = CardinalityTier.LOW
    max_length: int | None = None
    match_prefix: bool = False
    allow_sequence: bool = False


@dataclass(frozen=True, slots=True)
class AttributeRegistry:
    """Registry of known attribute schemas."""

    schemas: tuple[AttributeSchema, ...]
    _exact: Mapping[str, AttributeSchema] = field(init=False, repr=False)
    _prefix: tuple[AttributeSchema, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Populate the exact and prefix schema caches."""
        exact: dict[str, AttributeSchema] = {}
        prefix: list[AttributeSchema] = []
        for schema in self.schemas:
            if schema.match_prefix:
                prefix.append(schema)
            else:
                exact[schema.key] = schema
        object.__setattr__(self, "_exact", exact)
        object.__setattr__(self, "_prefix", tuple(prefix))

    def resolve(self, key: str) -> AttributeSchema | None:
        """Return the schema for a key, if registered.

        Returns
        -------
        AttributeSchema | None
            Schema for the key, or None when not registered.
        """
        schema = self._exact.get(key)
        if schema is not None:
            return schema
        for prefix in self._prefix:
            if prefix.matches_key(key):
                return prefix
        return None

    def keys(self) -> tuple[str, ...]:
        """Return a stable list of exact keys in the registry.

        Returns
        -------
        tuple[str, ...]
            Sorted exact keys in the registry.
        """
        return tuple(sorted(self._exact.keys()))

    def iter_prefixes(self) -> Iterator[str]:
        """Yield prefix keys registered for dynamic attributes.

        Yields
        ------
        str
            Registered prefix key.
        """
        for schema in self._prefix:
            yield schema.key


@dataclass(frozen=True, slots=True)
class AttributeNormalizer:
    """Normalize attributes with schema enforcement and budget limits."""

    registry: AttributeRegistry
    budget: AttributeBudget

    def normalize(
        self,
        attributes: Mapping[str, object],
        *,
        allowed_keys: frozenset[str] | None = None,
        allowed_prefixes: Sequence[str] | None = None,
        allow_unknown: bool = False,
    ) -> dict[str, SpanAttributeValue]:
        """Normalize a mapping of attributes into safe span attributes.

        Returns
        -------
        dict[str, SpanAttributeValue]
            Normalized span attributes.
        """
        if not attributes:
            return {}

        allowed = allowed_keys or frozenset()
        prefixes = tuple(allowed_prefixes or ())
        list_limit = self.budget.max_list_len
        default_str_limit = self.budget.max_str_len

        shaped: dict[str, SpanAttributeValue] = {}
        for key, value in attributes.items():
            if allowed or prefixes:
                allowlist_match = bool(allowed) and key in allowed
                prefix_match = bool(prefixes) and any(key.startswith(prefix) for prefix in prefixes)
                if not (allowlist_match or prefix_match):
                    continue

            schema = self.registry.resolve(key)
            if schema is None and not allow_unknown:
                continue

            max_str_len = default_str_limit
            if schema is not None and schema.max_length is not None:
                if max_str_len is None:
                    max_str_len = schema.max_length
                else:
                    max_str_len = min(max_str_len, schema.max_length)

            attr_value = coerce_attribute_value(
                value,
                max_list_len=list_limit,
                max_str_len=max_str_len,
            )
            if attr_value is None:
                continue
            if schema is not None and not schema.is_value_allowed(attr_value):
                continue
            shaped[key] = attr_value
        return shaped


def _schema(
    key: str,
    value_type: type | tuple[type, ...],
    options: SchemaOptions | None = None,
) -> AttributeSchema:
    resolved = options or SchemaOptions()
    value_types = value_type if isinstance(value_type, tuple) else (value_type,)
    return AttributeSchema(
        key=key,
        value_types=value_types,
        cardinality=resolved.cardinality,
        max_length=resolved.max_length,
        match_prefix=resolved.match_prefix,
        allow_sequence=resolved.allow_sequence,
    )


def default_attribute_registry() -> AttributeRegistry:
    """Return the default attribute registry for CodeIntel observability.

    Returns
    -------
    AttributeRegistry
        Registry populated with default attribute schemas.
    """
    medium = SchemaOptions(cardinality=CardinalityTier.MEDIUM)
    medium_512 = SchemaOptions(cardinality=CardinalityTier.MEDIUM, max_length=512)
    high_4096 = SchemaOptions(cardinality=CardinalityTier.HIGH, max_length=4096)
    high_prefix = SchemaOptions(cardinality=CardinalityTier.HIGH, match_prefix=True)
    high_sequence = SchemaOptions(cardinality=CardinalityTier.HIGH, allow_sequence=True)
    route_max = SchemaOptions(max_length=256)
    tool_name_max = SchemaOptions(max_length=128)

    schemas = (
        _schema(CODEINTEL_COMPONENT, str),
        _schema(CODEINTEL_OPERATION, str),
        _schema(CODEINTEL_SUCCESS, bool),
        _schema(CODEINTEL_ENDPOINT, str),
        _schema(CODEINTEL_OUTPUT_FORMAT, str),
        _schema(CODEINTEL_HEALTH_CHECK, bool),
        _schema(CODEINTEL_CORRELATION_ID, str, medium),
        _schema(CODEINTEL_RUN_ID, str, medium),
        _schema(CODEINTEL_DOMAIN, str),
        _schema(CODEINTEL_REPO, str),
        _schema(CODEINTEL_COMMIT, str),
        _schema(CODEINTEL_ACTOR, str, medium),
        _schema(CODEINTEL_QUERY_ENDPOINT, str),
        _schema(CODEINTEL_QUERY_ENGINE, str),
        _schema(CODEINTEL_QUERY_ENGINE_PREFERENCE, str),
        _schema(CODEINTEL_QUERY_BATCH_SIZE, int),
        _schema(CODEINTEL_QUERY_SCAN_ROWS, int),
        _schema(CODEINTEL_QUERY_SCAN_FILES, int),
        _schema(CODEINTEL_QUERY_SCAN_BYTES, int),
        _schema(CODEINTEL_QUERY_ROW_COUNT, int),
        _schema(CODEINTEL_QUERY_TRUNCATED, bool),
        _schema(CODEINTEL_QUERY_VIEW_ID, str, medium),
        _schema(CODEINTEL_QUERY_HASH, str),
        _schema(CODEINTEL_QUERY_SCHEMA_HASH, str),
        _schema(HTTP_METHOD, str),
        _schema(HTTP_ROUTE, str, route_max),
        _schema(MCP_METHOD, str),
        _schema(MCP_TOOL_NAME, str, tool_name_max),
        _schema(DB_SYSTEM_NAME, str),
        _schema(DB_NAMESPACE, str),
        _schema(DB_STATEMENT, str, medium_512),
        _schema(DB_QUERY_SUMMARY, str, medium_512),
        _schema(DB_QUERY_TEXT, str, high_4096),
        _schema(DB_QUERY_PARAMETER_PREFIX, (str, bool, int, float), high_prefix),
        _schema(CODEINTEL_DB_STATEMENT_SHA256, str),
        _schema(BUILD_RUN_ID, str),
        _schema(BUILD_REPO, str),
        _schema(BUILD_COMMIT, str),
        _schema(BUILD_TARGETS, str, medium),
        _schema(BUILD_DURATION_MS, (float, int)),
        _schema(BUILD_DECISION_TRACE_ARTIFACT, str, medium),
        _schema(BUILD_VALIDATION_MODE, str),
        _schema(BUILD_VALIDATION_ISSUE_COUNT, int),
        _schema(BUILD_SCHEMA_INFERENCE_ERRORS_COUNT, int),
        _schema(CLI_INVOCATION_ID, str),
        _schema(CLI_COMMAND, str, medium),
        _schema(CLI_EXIT_CODE, int),
        _schema(CLI_IS_PARSE_ERROR, bool),
        _schema(CLI_ERROR_TYPE, str),
        _schema(CLI_ARG_COUNT, int),
        _schema(CLI_ARG_NAMES, str, high_sequence),
        _schema(CLI_DURATION_MS, (float, int)),
        _schema(CLI_PARSE_DURATION_MS, (float, int)),
        _schema(SHUTDOWN_STATUS, str),
        _schema(SHUTDOWN_PENDING_TASKS_COUNT, int),
        _schema(SHUTDOWN_ACTIVE_THREADS_COUNT, int),
        _schema(SHUTDOWN_SUBPROCESS_COUNT, int),
        _schema(SHUTDOWN_PENDING_TASK_SAMPLES, str, high_sequence),
        _schema(SHUTDOWN_ACTIVE_THREAD_NAMES, str, high_sequence),
        _schema(SHUTDOWN_SUBPROCESS_SAMPLES, str, high_sequence),
        _schema(SHUTDOWN_ERROR_TYPE, str, medium),
        _schema(SHUTDOWN_ERROR_MESSAGE, str, medium_512),
        _schema(TELEMETRY_FLUSH_OK, bool),
        _schema(TELEMETRY_FLUSH_MS, (float, int)),
        _schema(TELEMETRY_ACTION, str),
        _schema(TELEMETRY_INSTRUMENTATION_NAME, str, medium),
        _schema(TELEMETRY_INSTRUMENTATION_STATUS, str),
        _schema(CODEINTEL_STORAGE_READ_ONLY, bool),
        _schema(SCIP_RUN_ID, str),
        _schema(SCIP_REPO, str),
        _schema(SCIP_COMMIT, str),
        _schema(SCIP_MODE, str),
        _schema(SCIP_STATUS, str),
        _schema(SCIP_ERROR, str, medium),
        _schema(SCIP_DURATION_MS, (float, int)),
    )
    return AttributeRegistry(schemas=schemas)


def build_attribute_normalizer(policy: ObservabilityPolicy) -> AttributeNormalizer:
    """Return an attribute normalizer for the supplied policy.

    Returns
    -------
    AttributeNormalizer
        Normalizer configured with the policy budget and registry.
    """
    return AttributeNormalizer(
        registry=default_attribute_registry(),
        budget=policy.budget,
    )


__all__ = [
    "AttributeNormalizer",
    "AttributeRegistry",
    "AttributeSchema",
    "CardinalityTier",
    "build_attribute_normalizer",
    "default_attribute_registry",
]
