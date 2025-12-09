"""Introspection utilities for CLI operations.

Provide runtime discovery of operations, their schemas, and examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from codeintel.cli.execution import OperationSpec
from codeintel.cli.operation_registry import get_operation_registry


@dataclass(frozen=True)
class OperationInfo:
    """Detailed information about an operation.

    Parameters
    ----------
    operation_id
        Unique identifier.
    category
        Operation category.
    description
        Human-readable description.
    parameters
        Parameter specifications.
    examples
        Usage examples.
    requires_progress
        Whether progress is shown.
    retryable
        Whether operation is retryable.
    """

    operation_id: str
    category: str
    description: str
    parameters: list[dict[str, object]]
    examples: list[str]
    requires_progress: bool
    retryable: bool

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return asdict(self)


def get_operation_info(operation_id: str) -> OperationInfo | None:
    """Get detailed information about an operation.

    Parameters
    ----------
    operation_id
        Operation identifier.

    Returns
    -------
    OperationInfo | None
        Operation information or None if not found.
    """
    registry = get_operation_registry()
    spec = registry.get(operation_id)

    if spec is None:
        return None

    parameters = _extract_parameters(spec)
    examples = _generate_examples(spec)

    return OperationInfo(
        operation_id=spec.operation_id,
        category=spec.category.value,
        description=spec.description,
        parameters=parameters,
        examples=examples,
        requires_progress=spec.requires_progress,
        retryable=spec.retryable,
    )


def get_operation_schema(operation_id: str) -> dict[str, object] | None:
    """Get JSON Schema for an operation's parameters.

    Parameters
    ----------
    operation_id
        Operation identifier.

    Returns
    -------
    dict[str, object] | None
        JSON Schema or None if not found.
    """
    registry = get_operation_registry()
    spec = registry.get(operation_id)

    if spec is None or spec.param_schema is None:
        return None

    return _schema_to_json_schema(spec.param_schema)


def list_operations_by_category() -> dict[str, list[str]]:
    """List operations grouped by category.

    Returns
    -------
    dict[str, list[str]]
        Operations grouped by category name.
    """
    registry = get_operation_registry()
    result: dict[str, list[str]] = {}

    for spec in registry.list_operations():
        category = spec.category.value
        if category not in result:
            result[category] = []
        result[category].append(spec.operation_id)

    return result


def search_operations(query: str) -> list[OperationInfo]:
    """Search operations by ID or description.

    Parameters
    ----------
    query
        Search query.

    Returns
    -------
    list[OperationInfo]
        Matching operations.
    """
    registry = get_operation_registry()
    query_lower = query.lower()
    results = []

    for spec in registry.list_operations():
        match_id = query_lower in spec.operation_id.lower()
        match_desc = query_lower in spec.description.lower()
        if match_id or match_desc:
            info = get_operation_info(spec.operation_id)
            if info:
                results.append(info)

    return results


def list_all_operations() -> list[OperationInfo]:
    """List all registered operations.

    Returns
    -------
    list[OperationInfo]
        All operation info.
    """
    registry = get_operation_registry()
    results = []

    for spec in registry.list_operations():
        info = get_operation_info(spec.operation_id)
        if info:
            results.append(info)

    return results


def _extract_parameters(spec: OperationSpec[Any]) -> list[dict[str, object]]:
    """Extract parameter information from spec.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    list[dict[str, object]]
        Parameter information.
    """
    if spec.param_schema is None:
        return []

    params: list[dict[str, object]] = []
    for name, validator in spec.param_schema.validators.items():
        validator_name = type(validator).__name__
        param_info: dict[str, object] = {
            "name": name,
            "type": validator_name.replace("Validator", "").lower(),
            "required": True,  # Default to required
        }
        params.append(param_info)

    return params


def _generate_examples(spec: OperationSpec[Any]) -> list[str]:
    """Generate usage examples for an operation.

    Parameters
    ----------
    spec
        Operation specification.

    Returns
    -------
    list[str]
        Example command lines.
    """
    base = f"codeintel op call {spec.operation_id}"
    examples = [base]

    if spec.param_schema:
        param_parts = [f"--{name}=VALUE" for name in spec.param_schema.validators]
        if param_parts:
            param_str = " ".join(param_parts)
            examples.append(f"{base} {param_str}")

    return examples


def _schema_to_json_schema(schema: object) -> dict[str, object]:
    """Convert validation schema to JSON Schema.

    Parameters
    ----------
    schema
        Validation schema.

    Returns
    -------
    dict[str, object]
        JSON Schema.
    """
    # Basic conversion - can be extended for full schema support
    properties: dict[str, dict[str, object]] = {}
    required: list[str] = []

    validators = getattr(schema, "validators", None)
    if validators is not None and isinstance(validators, dict):
        for name, validator in validators.items():
            prop: dict[str, object] = {"type": "string"}

            validator_name = type(validator).__name__
            if "Int" in validator_name:
                prop["type"] = "integer"
            elif "Path" in validator_name:
                prop["type"] = "string"
                prop["format"] = "path"

            properties[name] = prop
            required.append(name)

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required,
    }


__all__ = [
    "OperationInfo",
    "get_operation_info",
    "get_operation_schema",
    "list_all_operations",
    "list_operations_by_category",
    "search_operations",
]
