"""Validate serving operation contracts across protocols, services, and backends.

This script reflects component signatures, validates them against the
DatasetSchema-backed contract, and checks for missing operations.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable

from codeintel.serving.contracts.operation_contract_reflection import (
    ComponentSpec,
    build_operation_contract_dataframe,
    validate_operation_contracts,
)
from codeintel.serving.contracts.operation_contracts_dataset import TRANSPORT_KINDS
from codeintel.serving.mcp.backend import DuckDBBackend, HttpBackend
from codeintel.serving.operations import iter_operations
from codeintel.serving.services.query_service import HttpQueryService, LocalQueryService
from codeintel.serving.types import QueryBackendProtocol, QueryServiceProtocol


def _component_specs() -> list[ComponentSpec]:
    """Return the reflected component specifications."""
    return [
        ComponentSpec(component=QueryServiceProtocol, transport="protocol_service"),
        ComponentSpec(component=QueryBackendProtocol, transport="protocol_backend"),
        ComponentSpec(component=LocalQueryService, transport="service"),
        ComponentSpec(component=HttpQueryService, transport="service"),
        ComponentSpec(component=DuckDBBackend, transport="backend"),
        ComponentSpec(component=HttpBackend, transport="backend"),
    ]


def _expected_methods() -> set[str]:
    """Return the set of backend method names from the operation catalog."""
    ignore = {"graph_plugin_plan", "health"}
    return {spec.backend_method for spec in iter_operations() if spec.backend_method not in ignore}


def _find_missing_methods(
    df_components: Iterable[tuple[str, set[str]]],
    expected: set[str],
) -> dict[str, set[str]]:
    """Return mapping from component label to missing method names."""
    missing: dict[str, set[str]] = {}
    for component, methods in df_components:
        missing_methods = expected - methods
        if missing_methods:
            missing[component] = missing_methods
    return missing


def _validate_missing(df_grouped: dict[str, set[str]], expected: set[str]) -> dict[str, set[str]]:
    """Check missing methods for each component and return the diff."""
    return _find_missing_methods(df_grouped.items(), expected)


def run() -> int:
    """Execute the contract validation and missing-method check."""
    components = _component_specs()
    df = build_operation_contract_dataframe(components)
    validated = validate_operation_contracts(df)

    grouped_methods: dict[str, set[str]] = defaultdict(set)
    for _, row in validated.iterrows():
        grouped_methods[row["component"]].add(str(row["method"]))

    expected = _expected_methods()
    missing = _validate_missing(grouped_methods, expected)
    if missing:
        sys.stderr.write("Operation contract validation failed; missing methods detected:\n")
        for component, methods in sorted(missing.items()):
            formatted = ", ".join(sorted(methods))
            sys.stderr.write(f"- {component}: missing [{formatted}]\n")
        return 1

    # Ensure transports are expected
    invalid_transports = validated[
        ~validated["transport"].isin(TRANSPORT_KINDS)
    ].transport.unique()
    if len(invalid_transports) > 0:
        sys.stderr.write(
            f"Unexpected transport kinds found: {', '.join(sorted(map(str, invalid_transports)))}\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run())
