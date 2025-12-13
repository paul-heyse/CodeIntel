"""DEPRECATED: Analytics adapters have been removed.

.. deprecated:: 4.0.0
    The analytics adapters package has been removed. Migrate to:

    - For ``DeleteScope``: Use ``codeintel.analytics.utilities.persistence.DeleteScope``
    - For row builders: Use ``codeintel.analytics.compute.row_builders``
    - For dependency types: Use ``codeintel.config.datasets.dependencies``
    - For semantic role types: Use ``codeintel.config.datasets.semantic_roles``
    - For GOID types: Use ``codeintel.analytics.compute.functions.goids``
    - For table writes: Use ``ctx.write_table()`` in Hamilton plugins

This stub module exists only to provide helpful error messages.
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Provide helpful error messages for deprecated imports.

    Parameters
    ----------
    name
        Name of the attribute being accessed.

    Raises
    ------
    ImportError
        Raised for known deprecated imports with migration guidance.
    AttributeError
        Raised for unknown attributes.
    """
    migration_map = {
        "DeleteScope": "codeintel.analytics.utilities.persistence.DeleteScope",
        "BatchAdapter": "use ctx.write_table() in Hamilton plugins",
        "AnalyticsAdapter": "use ctx.write_table() in Hamilton plugins",
        "SimpleBatchAdapter": "use ctx.write_table() in Hamilton plugins",
        "ComputeAdapter": "use ctx.write_table() in Hamilton plugins",
        "InputAdapter": "use ctx.write_table() in Hamilton plugins",
        "OutputAdapter": "use ctx.write_table() in Hamilton plugins",
        "DependencyCallRow": "codeintel.config.datasets.dependencies.DependencyCallRow",
        "DependencyAggregateRow": "codeintel.config.datasets.dependencies.DependencyAggregateRow",
        "compute_dep_id": "codeintel.config.datasets.dependencies.compute_dep_id",
        "to_decimal": "codeintel.config.datasets.dependencies.to_decimal",
        "FunctionGoid": "codeintel.analytics.compute.functions.goids.FunctionGoid",
        "FunctionGoidLoader": "codeintel.analytics.compute.functions.goids.FunctionGoidLoader",
        "GoidRow": "codeintel.analytics.compute.functions.goids.GoidRow",
        "FunctionSemanticRoleRow": "codeintel.config.datasets.semantic_roles.FunctionSemanticRoleRow",
        "ModuleSemanticRoleRow": "codeintel.config.datasets.semantic_roles.ModuleSemanticRoleRow",
        "SchemaValidationMixin": "use ctx.write_validated_table() in Hamilton plugins",
        "SchemaAwareBatchAdapter": "use ctx.write_validated_table() in Hamilton plugins",
    }

    if name in migration_map:
        suggestion = migration_map[name]
        message = (
            f"'{name}' has been removed from codeintel.analytics.adapters. "
            f"Use {suggestion} instead."
        )
        raise ImportError(message)

    message = f"module 'codeintel.analytics.adapters' has no attribute '{name}'"
    raise AttributeError(message)


__all__: list[str] = []
