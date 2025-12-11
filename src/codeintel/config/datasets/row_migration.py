"""Row model migration utilities for transitioning to schema-generated models.

This module provides utilities for migrating from manually-defined TypedDict
row models to Pandera schema-generated models. It includes compatibility
checking and validation to ensure a safe migration path.

Architecture Reference: Section 5.3.1 - Migrate row models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import ModuleType

from codeintel.config.datasets.schema_registry import SCHEMA_REGISTRY

_rows_module: ModuleType | None

try:
    from codeintel.config.datasets import rows as _rows_module
except ImportError:
    _rows_module = None

__all__ = [
    "MigrationStatus",
    "RowModelMigrationResult",
    "get_row_model",
    "validate_all_row_models",
    "validate_row_model_compatibility",
]

log = logging.getLogger(__name__)


@dataclass
class MigrationStatus:
    """Status of a single row model migration.

    Parameters
    ----------
    table_key
        Fully qualified table name.
    has_manual_model
        Whether a manual TypedDict exists.
    has_schema_model
        Whether a schema-generated model is available.
    compatible
        Whether the models are compatible.
    differences
        List of differences between models.
    """

    table_key: str
    has_manual_model: bool
    has_schema_model: bool
    compatible: bool
    differences: list[str]


@dataclass
class RowModelMigrationResult:
    """Result of row model migration validation.

    Parameters
    ----------
    total_datasets
        Total number of datasets checked.
    compatible_count
        Number of datasets with compatible models.
    incompatible_count
        Number of datasets with incompatible models.
    missing_schema_count
        Number of datasets missing schema-generated models.
    statuses
        Detailed status for each dataset.
    """

    total_datasets: int
    compatible_count: int
    incompatible_count: int
    missing_schema_count: int
    statuses: list[MigrationStatus]

    def is_ready_for_migration(self) -> bool:
        """Check if all datasets are ready for migration.

        Returns
        -------
        bool
            True if all datasets have compatible models.
        """
        return self.incompatible_count == 0 and self.missing_schema_count == 0


def get_row_model(table_key: str) -> type:
    """Get row model for a dataset, preferring schema-generated over manual.

    This function provides a migration path from manual TypedDict definitions
    to schema-generated models. Currently returns the manual model if available,
    falling back to schema-generated when manual is not found.

    Parameters
    ----------
    table_key
        Fully qualified table name (e.g., "analytics.function_metrics").

    Returns
    -------
    type
        The TypedDict row model type.

    Raises
    ------
    KeyError
        If no row model is available for the table.

    Notes
    -----
    Schema-generated models can be activated as primary source once
    validate_row_model_compatibility() confirms all datasets compatible.
    See architecture Section 5.3.1 - Migrate row models for activation steps.
    """
    # First, try to get schema-generated model
    schema = SCHEMA_REGISTRY.get(table_key)
    if schema is not None:
        try:
            return schema.get_row_model()
        except (KeyError, AttributeError) as exc:
            log.debug("Could not generate row model for %s: %s", table_key, exc)

    # Fall back to manual model lookup
    manual_model = _get_manual_row_model(table_key)
    if manual_model is not None:
        return manual_model

    msg = f"No row model available for {table_key}"
    raise KeyError(msg)


def _get_manual_row_model(table_key: str) -> type | None:
    """Look up manually-defined row model by table key.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    type | None
        The manual TypedDict if found, None otherwise.
    """
    # Map of table_key to manual model import paths
    manual_models: dict[str, str] = {
        "analytics.function_metrics": "FunctionMetricsRow",
        "analytics.function_types": "FunctionTypesRow",
        "analytics.coverage_lines": "CoverageLineRow",
        "analytics.typedness": "TypednessRow",
        "analytics.static_diagnostics": "StaticDiagnosticRow",
        "analytics.function_validation": "FunctionValidationRow",
        "analytics.graph_validation": "GraphValidationRow",
        "analytics.hotspots": "HotspotRow",
        "core.goids": "GoidRow",
        "core.goid_crosswalk": "GoidCrosswalkRow",
        "core.docstrings": "DocstringRow",
    }

    model_name = manual_models.get(table_key)
    if model_name is None:
        return None

    if _rows_module is None:
        return None

    return getattr(_rows_module, model_name, None)


def validate_row_model_compatibility(table_key: str) -> MigrationStatus:
    """Compare generated vs manual models for migration validation.

    This function checks if a schema-generated row model is compatible
    with the existing manual TypedDict definition. Compatibility means
    both models have the same fields with compatible types.

    Parameters
    ----------
    table_key
        Fully qualified table name.

    Returns
    -------
    MigrationStatus
        Detailed compatibility status.

    Examples
    --------
    >>> status = validate_row_model_compatibility("analytics.function_metrics")
    >>> status.compatible  # doctest: +SKIP
    True
    """
    differences: list[str] = []

    # Check for manual model
    manual_model = _get_manual_row_model(table_key)
    has_manual = manual_model is not None

    # Check for schema-generated model
    schema = SCHEMA_REGISTRY.get(table_key)
    has_schema = schema is not None

    if schema is None:
        return MigrationStatus(
            table_key=table_key,
            has_manual_model=has_manual,
            has_schema_model=False,
            compatible=False,
            differences=["No schema registered for this dataset"],
        )

    if manual_model is None:
        # No manual model means we can use schema-generated directly
        return MigrationStatus(
            table_key=table_key,
            has_manual_model=False,
            has_schema_model=True,
            compatible=True,
            differences=[],
        )

    # At this point, both schema and manual_model are not None
    # Compare the two models
    try:
        generated_model = schema.get_row_model()
        differences = _compare_typed_dicts(manual_model, generated_model)
    except (KeyError, AttributeError) as exc:
        differences = [f"Failed to generate model: {exc}"]

    return MigrationStatus(
        table_key=table_key,
        has_manual_model=has_manual,
        has_schema_model=has_schema,
        compatible=len(differences) == 0,
        differences=differences,
    )


def _compare_typed_dicts(manual: type, generated: type) -> list[str]:
    """Compare two TypedDict types for field compatibility.

    Parameters
    ----------
    manual
        The manually-defined TypedDict.
    generated
        The schema-generated TypedDict.

    Returns
    -------
    list[str]
        List of differences found.
    """
    differences: list[str] = []

    # Get annotations from both
    manual_annotations = getattr(manual, "__annotations__", {})
    generated_annotations = getattr(generated, "__annotations__", {})

    manual_fields = set(manual_annotations.keys())
    generated_fields = set(generated_annotations.keys())

    # Check for missing fields
    missing_in_generated = manual_fields - generated_fields
    differences.extend(
        f"Field '{field}' in manual but not in generated" for field in missing_in_generated
    )

    extra_in_generated = generated_fields - manual_fields
    differences.extend(
        f"Field '{field}' in generated but not in manual" for field in extra_in_generated
    )

    # Check for type mismatches in common fields
    common_fields = manual_fields & generated_fields
    for field in common_fields:
        manual_type = manual_annotations[field]
        generated_type = generated_annotations[field]
        if not _types_compatible(manual_type, generated_type):
            differences.append(
                f"Type mismatch for '{field}': manual={manual_type}, generated={generated_type}"
            )

    return differences


def _types_compatible(manual_type: object, generated_type: object) -> bool:
    """Check if two types are compatible for migration purposes.

    Parameters
    ----------
    manual_type
        Type from manual definition.
    generated_type
        Type from generated definition.

    Returns
    -------
    bool
        True if types are compatible.

    Notes
    -----
    More sophisticated type comparison could handle Union types, Optional,
    and type aliases. See architecture Section 5.3.1 - Migrate row models.
    """
    # Simple string comparison for now
    return str(manual_type) == str(generated_type)


def validate_all_row_models() -> RowModelMigrationResult:
    """Validate row model compatibility for all datasets.

    Returns
    -------
    RowModelMigrationResult
        Aggregated migration status for all datasets.
    """
    statuses: list[MigrationStatus] = []
    compatible_count = 0
    incompatible_count = 0
    missing_schema_count = 0

    for table_key in SCHEMA_REGISTRY.all():
        status = validate_row_model_compatibility(table_key)
        statuses.append(status)

        if not status.has_schema_model:
            missing_schema_count += 1
        elif status.compatible:
            compatible_count += 1
        else:
            incompatible_count += 1

    return RowModelMigrationResult(
        total_datasets=len(statuses),
        compatible_count=compatible_count,
        incompatible_count=incompatible_count,
        missing_schema_count=missing_schema_count,
        statuses=statuses,
    )
