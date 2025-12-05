"""Shared fixtures for analytics adapter tests.

This module provides adapter-specific constants. For general test fixtures like
TestContext, test_ctx, fresh_gateway, etc., use the fixtures from the main conftest.py.

Adapter tests now use the standard `fresh_gateway` fixture from the main conftest.py.
"""

from __future__ import annotations

from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO

# =============================================================================
# Constants (aliases for backward compatibility)
# =============================================================================

ADAPTER_TEST_REPO = DEFAULT_REPO
ADAPTER_TEST_COMMIT = DEFAULT_COMMIT

__all__ = [
    "ADAPTER_TEST_COMMIT",
    "ADAPTER_TEST_REPO",
]
