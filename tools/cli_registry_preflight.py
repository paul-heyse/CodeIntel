"""Preflight check for CLI registry canonicalization."""

from __future__ import annotations

import sys

from codeintel.cli.execution.registry import get_registry
from codeintel.cli.introspection.registry_inventory import build_registry_inventory


def _format_list(items: list[str]) -> str:
    if not items:
        return ""
    return "\n".join(f"  - {item}" for item in items)


def main() -> int:
    """Run registry preflight checks.

    Returns
    -------
    int
        Exit code (0 when checks pass).
    """
    registry = get_registry()
    inventory = build_registry_inventory(registry)
    errors: list[str] = []

    conflicting = [
        candidate.alias_id for candidate in inventory.alias_candidates if candidate.canonical_exists
    ]
    if conflicting:
        errors.append(
            "Conflicting operation IDs detected (legacy prefix with canonical duplicate):\n"
            + _format_list(conflicting)
        )

    op_ids = {spec.operation_id for spec in inventory.operations}
    missing_aliases = [
        f"Alias target not registered: {alias.alias_id} -> {alias.target_id}"
        for alias in inventory.aliases
        if alias.target_id not in op_ids
    ]
    errors.extend(missing_aliases)

    if errors:
        sys.stderr.write("CLI registry preflight failed:\n")
        sys.stderr.write("\n".join(errors))
        sys.stderr.write("\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
