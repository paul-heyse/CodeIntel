"""PR51: Tests for writer_guard deprecation migration.

This module verifies:
1. Deprecation warnings are emitted for deprecated functions
2. Functions still work for backward compatibility
3. File is in allowlist for architecture guardrails
4. write_rows_via_policy_backend does NOT emit warnings (preferred path)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.profiles.writer_guard import (
    PolicyWriterConfig,
    WriterContext,
    create_profile_writer,
    write_rows_via_policy_backend,
    write_rows_with_registry_guard,
)
from tests.build.hamilton.test_pr50_architecture_guardrails import (
    ALLOWLIST_IBIS_WRITE_FILES,
)

if TYPE_CHECKING:
    from tests._helpers import TestContext


# =============================================================================
# Deprecation warning tests
# =============================================================================


def test_write_rows_with_registry_guard_emits_deprecation_warning(
    test_ctx: TestContext,
) -> None:
    """Verify write_rows_with_registry_guard emits DeprecationWarning."""
    context = WriterContext(
        table_key="analytics.test_catalog",
        columns=["repo", "commit"],
        serialize_row=lambda r: (r["repo"], r["commit"]),
        repo=test_ctx.repo,
        commit=test_ctx.commit,
        ensure_schema_fn=lambda _gw, _table: None,
    )

    with pytest.warns(DeprecationWarning, match="write_rows_with_registry_guard is deprecated"):
        # Call with empty rows to avoid actual DB operations
        write_rows_with_registry_guard(
            test_ctx.gateway,
            rows=[],
            context=context,
            delete_on_empty=False,
        )


def test_create_profile_writer_emits_deprecation_warning() -> None:
    """Verify create_profile_writer emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="create_profile_writer is deprecated"):
        create_profile_writer(
            "analytics.test_catalog",
            ["repo", "commit"],
            lambda r: (r["repo"], r["commit"]),
        )


# =============================================================================
# Backward compatibility tests
# =============================================================================


def test_write_rows_with_registry_guard_still_works(test_ctx: TestContext) -> None:
    """Verify write_rows_with_registry_guard still works for backward compat."""
    context = WriterContext(
        table_key="analytics.test_catalog",
        columns=["repo", "commit"],
        serialize_row=lambda r: (r["repo"], r["commit"]),
        repo=test_ctx.repo,
        commit=test_ctx.commit,
        ensure_schema_fn=lambda _gw, _table: None,
    )

    with pytest.warns(DeprecationWarning, match="write_rows_with_registry_guard"):
        result = write_rows_with_registry_guard(
            test_ctx.gateway,
            rows=[],
            context=context,
            delete_on_empty=False,
        )

    assert result == 0  # noqa: S101


def test_create_profile_writer_produces_callable() -> None:
    """Verify create_profile_writer still produces a callable writer."""
    with pytest.warns(DeprecationWarning, match="create_profile_writer"):
        writer = create_profile_writer(
            "analytics.test_catalog",
            ["repo", "commit"],
            lambda r: (r["repo"], r["commit"]),
        )

    assert callable(writer)  # noqa: S101


# =============================================================================
# Preferred path tests (no deprecation warnings)
# =============================================================================


def test_write_rows_via_policy_backend_no_deprecation_warning(
    test_ctx: TestContext,
) -> None:
    """Verify write_rows_via_policy_backend does NOT emit DeprecationWarning."""
    import warnings  # noqa: PLC0415

    config = PolicyWriterConfig(
        table_key="analytics.test_catalog",
        columns=["repo", "commit"],
        serialize_row=lambda r: (r["repo"], r["commit"]),
        repo=test_ctx.repo,
        commit=test_ctx.commit,
    )

    # Capture warnings to verify none are emitted for this function
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = write_rows_via_policy_backend(
            test_ctx.gateway,
            rows=[],
            config=config,
        )

    # Filter for DeprecationWarning mentioning this function
    deprecation_warnings = [
        w
        for w in captured
        if issubclass(w.category, DeprecationWarning)
        and "write_rows_via_policy_backend" in str(w.message)
    ]
    assert len(deprecation_warnings) == 0  # noqa: S101
    assert result == 0  # noqa: S101


# =============================================================================
# Architecture guardrail tests
# =============================================================================


def test_writer_guard_in_allowlist() -> None:
    """Verify writer_guard.py is in allowlist for backward compatibility."""
    assert "src/codeintel/analytics/profiles/writer_guard.py" in ALLOWLIST_IBIS_WRITE_FILES  # noqa: S101


# =============================================================================
# Module exports tests
# =============================================================================


def test_module_exports_expected_symbols() -> None:
    """Verify writer_guard module exports expected symbols."""
    from codeintel.analytics.profiles import writer_guard  # noqa: PLC0415

    expected_symbols = {
        "WriterContext",
        "PolicyWriterConfig",
        "SerializeRow",
        "write_rows_with_registry_guard",
        "write_rows_via_policy_backend",
        "create_profile_writer",
    }

    # Check that all expected symbols are present
    for symbol in expected_symbols:
        assert hasattr(writer_guard, symbol), f"Missing expected symbol: {symbol}"  # noqa: S101
