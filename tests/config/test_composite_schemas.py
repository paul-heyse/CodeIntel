"""Test CompositeSchema validation against actual profile schemas.

This module validates that the COMPOSITE_SCHEMAS registry correctly describes
how profile tables are composed from their source tables.
"""

from __future__ import annotations

import pytest

from codeintel.config.datasets import (
    get_composite_schemas,
    get_dataset_contracts,
    get_table_schemas,
)


def _require(*, condition: bool, message: str) -> None:
    """Fail test if condition is not met (avoid S101)."""
    if not condition:
        pytest.fail(message)


# ---------------------------------------------------------------------------
# CompositeSchema Validation Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile_key",
    list(get_composite_schemas().keys()),
    ids=lambda k: k.split(".")[-1],
)
def test_composite_schema_sources_exist(profile_key: str) -> None:
    """Verify all source tables referenced by CompositeSchema exist."""
    composite = get_composite_schemas()[profile_key]
    table_schemas = get_table_schemas()

    for source_key in composite.composed_of:
        _require(
            condition=source_key in table_schemas,
            message=(
                f"CompositeSchema for {profile_key} references non-existent "
                f"source table: {source_key}"
            ),
        )


@pytest.mark.parametrize(
    "profile_key",
    list(get_composite_schemas().keys()),
    ids=lambda k: k.split(".")[-1],
)
def test_composite_schema_profile_exists(profile_key: str) -> None:
    """Verify the profile table referenced by CompositeSchema exists."""
    _require(
        condition=profile_key in get_table_schemas(),
        message=f"CompositeSchema defined for non-existent profile: {profile_key}",
    )


@pytest.mark.parametrize(
    "profile_key",
    list(get_composite_schemas().keys()),
    ids=lambda k: k.split(".")[-1],
)
def test_composite_schema_validation_passes(profile_key: str) -> None:
    """Verify CompositeSchema validation passes for each profile.

    All source columns should either:
    - Be present in the profile schema
    - Be explicitly listed in excluded_columns (with a comment explaining why)
    - Be remapped via column_mappings
    """
    table_schemas = get_table_schemas()
    composite = get_composite_schemas()[profile_key]
    profile_schema = table_schemas[profile_key]

    errors = composite.validate_against_profile(profile_schema, table_schemas)

    _require(
        condition=not errors,
        message=(
            f"CompositeSchema validation failed for {profile_key}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        ),
    )


# ---------------------------------------------------------------------------
# Source Column Tracking Tests
# ---------------------------------------------------------------------------


def test_get_source_for_column_shared() -> None:
    """Test that shared columns return the first source."""
    composite = get_composite_schemas()["analytics.function_profile"]
    table_schemas = get_table_schemas()

    # function_goid_h128 is in FUNCTION_ENTITY_COLS (shared)
    source = composite.get_source_for_column("function_goid_h128", table_schemas)

    _require(
        condition=source == "analytics.function_metrics",
        message=f"Expected source 'analytics.function_metrics', got {source!r}",
    )


def test_get_source_for_column_unique() -> None:
    """Test that unique columns return their correct source."""
    composite = get_composite_schemas()["analytics.function_profile"]
    table_schemas = get_table_schemas()

    # is_pure is from function_effects
    source = composite.get_source_for_column("is_pure", table_schemas)

    _require(
        condition=source == "analytics.function_effects",
        message=f"Expected source 'analytics.function_effects', got {source!r}",
    )


def test_get_source_for_column_additional() -> None:
    """Test that additional columns return None (profile-specific, no source)."""
    composite = get_composite_schemas()["analytics.function_profile"]
    table_schemas = get_table_schemas()

    # risk_component_coverage is an additional column (profile-specific)
    source = composite.get_source_for_column("risk_component_coverage", table_schemas)

    # None means profile-specific with no source table
    _require(
        condition=source is None,
        message=f"Expected source None for additional column, got {source!r}",
    )


def test_get_source_for_column_mapped() -> None:
    """Test that mapped columns are resolved correctly."""
    composite = get_composite_schemas()["analytics.function_profile"]
    table_schemas = get_table_schemas()

    # keyword_params is the profile name, keyword_only_params is the source name
    source = composite.get_source_for_column("keyword_params", table_schemas)

    # Should find keyword_only_params in function_metrics
    _require(
        condition=source == "analytics.function_metrics",
        message=f"Expected source 'analytics.function_metrics', got {source!r}",
    )


# ---------------------------------------------------------------------------
# Source Column Name Generation Tests
# ---------------------------------------------------------------------------


def test_source_column_names_includes_shared() -> None:
    """Test that shared fragment columns are included."""
    composite = get_composite_schemas()["analytics.function_profile"]
    table_schemas = get_table_schemas()

    col_names = composite.source_column_names(table_schemas)

    # FUNCTION_ENTITY_COLS includes these
    _require(
        condition="function_goid_h128" in col_names,
        message="Expected 'function_goid_h128' in source columns",
    )
    _require(
        condition="urn" in col_names,
        message="Expected 'urn' in source columns",
    )
    _require(
        condition="repo" in col_names,
        message="Expected 'repo' in source columns",
    )


def test_source_column_names_excludes_excluded() -> None:
    """Test that excluded columns are not included."""
    composite = get_composite_schemas()["analytics.function_profile"]
    table_schemas = get_table_schemas()

    col_names = composite.source_column_names(table_schemas)

    # These are in excluded_columns
    _require(
        condition="effects_json" not in col_names,
        message="Expected 'effects_json' to be excluded from source columns",
    )
    _require(
        condition="preconditions_json" not in col_names,
        message="Expected 'preconditions_json' to be excluded from source columns",
    )


def test_source_column_names_applies_mappings() -> None:
    """Test that column mappings are applied."""
    composite = get_composite_schemas()["analytics.function_profile"]
    table_schemas = get_table_schemas()

    col_names = composite.source_column_names(table_schemas)

    # keyword_only_params should be mapped to keyword_params
    _require(
        condition="keyword_params" in col_names,
        message="Expected mapped column 'keyword_params' in source columns",
    )
    _require(
        condition="keyword_only_params" not in col_names,
        message="Expected original 'keyword_only_params' to be mapped out",
    )


# ---------------------------------------------------------------------------
# CompositeSchema Completeness Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile_key",
    list(get_composite_schemas().keys()),
    ids=lambda k: k.split(".")[-1],
)
def test_composite_has_sources(profile_key: str) -> None:
    """Verify each CompositeSchema has at least one source table."""
    composite = get_composite_schemas()[profile_key]

    _require(
        condition=len(composite.composed_of) > 0,
        message=f"CompositeSchema for {profile_key} has no source tables",
    )


@pytest.mark.parametrize(
    "profile_key",
    list(get_composite_schemas().keys()),
    ids=lambda k: k.split(".")[-1],
)
def test_composite_has_shared_fragments(profile_key: str) -> None:
    """Verify each CompositeSchema has at least one shared fragment."""
    composite = get_composite_schemas()[profile_key]

    _require(
        condition=len(composite.shared_fragments) > 0,
        message=f"CompositeSchema for {profile_key} has no shared fragments",
    )


def test_all_profiles_have_composite_schemas() -> None:
    """Verify all profile tables have corresponding CompositeSchemas."""
    composite_schemas = get_composite_schemas()
    profile_keys = [
        "analytics.function_profile",
        "analytics.file_profile",
        "analytics.module_profile",
        "analytics.test_profile",
    ]

    for profile_key in profile_keys:
        _require(
            condition=profile_key in composite_schemas,
            message=f"Missing CompositeSchema for {profile_key}",
        )


# ---------------------------------------------------------------------------
# DatasetContract Integration Tests
# ---------------------------------------------------------------------------


def test_profile_contracts_have_composition() -> None:
    """Verify profile DatasetContracts have composition field set."""
    dataset_contracts = get_dataset_contracts()
    profile_names = [
        "function_profile",
        "file_profile",
        "module_profile",
        "test_profile",
    ]

    for name in profile_names:
        contract = dataset_contracts.get(name)
        _require(
            condition=contract is not None,
            message=f"Missing DatasetContract for {name}",
        )
        # Use 'contract is not None' check above, now safe to access
        if contract is not None:
            _require(
                condition=contract.composition is not None,
                message=f"DatasetContract for {name} should have composition metadata",
            )


def test_non_profile_contracts_no_composition() -> None:
    """Verify non-profile DatasetContracts don't have composition."""
    dataset_contracts = get_dataset_contracts()
    non_profile_names = [
        "function_metrics",
        "function_types",
        "test_catalog",
        "ast_nodes",
    ]

    for name in non_profile_names:
        contract = dataset_contracts.get(name)
        _require(
            condition=contract is not None,
            message=f"Missing DatasetContract for {name}",
        )
        # Use 'contract is not None' check above, now safe to access
        if contract is not None:
            _require(
                condition=contract.composition is None,
                message=f"DatasetContract for {name} should not have composition metadata",
            )


def test_composition_matches_composite_schemas() -> None:
    """Verify DatasetContract.composition matches COMPOSITE_SCHEMAS."""
    dataset_contracts = get_dataset_contracts()
    for table_key, composite in get_composite_schemas().items():
        _, name = table_key.split(".", maxsplit=1)
        contract = dataset_contracts.get(name)

        _require(
            condition=contract is not None,
            message=f"Missing DatasetContract for {name}",
        )
        # Use 'contract is not None' check above, now safe to access
        if contract is not None:
            _require(
                condition=contract.composition is composite,
                message=(
                    f"DatasetContract.composition should be the same object as "
                    f"COMPOSITE_SCHEMAS[{table_key!r}]"
                ),
            )
