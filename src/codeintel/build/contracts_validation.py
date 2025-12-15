"""Contract validation utilities for the build system.

This module provides validation functions to ensure that OutputContract
definitions are complete and consistent with the schema provider registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.schemas import get_schema_provider
from codeintel.config.primitives import BuildPaths

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.targets import TargetGraph


def validate_contracts(graph: TargetGraph, *, repo_root: Path | None = None) -> list[str]:
    """Validate that all target contracts are complete and consistent.

    Parameters
    ----------
    graph
        Target graph to validate.
    repo_root
        Optional repository root for artifact path validation.

    Returns
    -------
    list[str]
        List of validation errors. Empty list means validation passed.

    Examples
    --------
    >>> from codeintel.build.registry import get_target_graph
    >>> graph = get_target_graph()
    >>> errors = validate_contracts(graph)
    >>> assert len(errors) == 0, "Contract validation failed"
    """
    errors: list[str] = []
    provider = get_schema_provider()

    for target in graph.all_targets:
        # Check that all contract table_keys exist in schema provider
        errors.extend(
            f"Target '{target.name}' references unknown table_key: '{table_key}'"
            for table_key in target.contract.table_keys
            if provider.get_table_schema(table_key) is None
        )

        # Check artifact path templates are renderable
        if repo_root is not None:
            paths = BuildPaths.from_repo_root(repo_root)
            for artifact in target.contract.artifacts:
                try:
                    # Test rendering with standard paths
                    artifact.path_template.format(
                        build_dir=str(paths.build_dir),
                        scip_dir=str(paths.scip_dir),
                        export_dir=str(paths.document_output_dir),
                        repo_root=str(repo_root),
                    )
                except KeyError as e:
                    errors.append(
                        f"Target '{target.name}' artifact '{artifact.name}' "
                        f"has invalid template: {e}"
                    )

        # Check for targets with plugins but no contract outputs
        if target.plugin and not target.contract.tables and not target.contract.artifacts:
            errors.append(
                f"Target '{target.name}' has plugin '{target.plugin}' "
                f"but declares no contract outputs (tables or artifacts)"
            )

    return errors


__all__ = ["validate_contracts"]
