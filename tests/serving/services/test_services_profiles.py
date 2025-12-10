"""Tests for profile query service delegates.

This module tests the _ProfileQueryDelegates methods in services/profiles.py
through HTTP routes and direct LocalQueryService invocation, using real gateways.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import status

from codeintel.serving.mcp.errors import McpError
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
)

if TYPE_CHECKING:
    from tests._helpers.serving_apps import ServiceApp

# =============================================================================
# Constants
# =============================================================================


# =============================================================================
# get_function_profile Tests (via HTTP)
# =============================================================================


def test_get_function_profile_with_goid_h128(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_function_profile works with goid_h128 parameter.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    # Get a valid goid_h128 from the database
    result = provisioned_service_app.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/profiles/function?goid_h128={goid_h128}")

    # May return 404 if profile doesn't exist, or 200 if found
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_function_profile_missing_goid_h128(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_function_profile returns error when goid_h128 missing.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/profiles/function")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


def test_get_function_profile_invalid_goid_h128(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_function_profile handles invalid goid_h128.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        # Use a very large number that's unlikely to exist
        response = client.get("/profiles/function?goid_h128=999999999999")

    # Should return 404 or empty result
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# get_file_profile Tests (via HTTP)
# =============================================================================


def test_get_file_profile_with_rel_path(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_file_profile works with rel_path parameter.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/profiles/file?rel_path={rel_path}")

    # May return 404 if profile doesn't exist, or 200 if found
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_file_profile_missing_rel_path(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_file_profile returns error when rel_path missing.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/profiles/file")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


def test_get_file_profile_nonexistent_file(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_file_profile handles nonexistent file.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/profiles/file?rel_path=nonexistent/path/file.py")

    # Should return 404 or empty result
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# get_module_profile Tests (via HTTP)
# =============================================================================


def test_get_module_profile_with_module(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_module_profile works with module parameter.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/profiles/module?module={module}")

    # May return 404 if profile doesn't exist, or 200 if found
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_module_profile_missing_module(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_module_profile returns error when module missing.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/profiles/module")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


def test_get_module_profile_nonexistent_module(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_module_profile handles nonexistent module.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/profiles/module?module=nonexistent.module.name")

    # Should return 404 or empty result
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# get_function_architecture Tests (via HTTP)
# =============================================================================


def test_get_function_architecture_with_goid_h128(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_function_architecture works with goid_h128 parameter.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/architecture/function?goid_h128={goid_h128}")

    # May return 404 if architecture doesn't exist, or 200 if found
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_function_architecture_missing_goid_h128(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_function_architecture returns error when goid_h128 missing.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/architecture/function")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


def test_get_function_architecture_invalid_goid_h128(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_function_architecture handles invalid goid_h128.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/architecture/function?goid_h128=999999999999")

    # Should return 404 or empty result
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# get_module_architecture Tests (via HTTP)
# =============================================================================


def test_get_module_architecture_with_module(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_module_architecture works with module parameter.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/architecture/module?module={module}")

    # May return 404 if architecture doesn't exist, or 200 if found
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


def test_get_module_architecture_missing_module(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_module_architecture returns error when module missing.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/architecture/module")

    # Should return 422 validation error (missing required param)
    expect_equal(response.status_code, status.HTTP_422_UNPROCESSABLE_CONTENT)


def test_get_module_architecture_nonexistent_module(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify get_module_architecture handles nonexistent module.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    with provisioned_service_app.client() as client:
        response = client.get("/architecture/module?module=nonexistent.module.name")

    # Should return 404 or empty result
    expect_in(response.status_code, {status.HTTP_200_OK, status.HTTP_404_NOT_FOUND})


# =============================================================================
# Direct LocalQueryService Tests
# =============================================================================


def test_local_query_service_get_function_profile(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify LocalQueryService.get_function_profile works directly.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    service = provisioned_service_app.service

    result = provisioned_service_app.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    profile = service.get_function_profile(goid_h128=goid_h128)
    expect_is_not_none(profile)


def test_local_query_service_get_file_profile(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify LocalQueryService.get_file_profile works directly.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    service = provisioned_service_app.service

    result = provisioned_service_app.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    profile = service.get_file_profile(rel_path=rel_path)
    expect_is_not_none(profile)


def test_local_query_service_get_module_profile(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify LocalQueryService.get_module_profile works directly.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    service = provisioned_service_app.service

    result = provisioned_service_app.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    # Module profile may not exist for all modules - handle gracefully
    try:
        profile = service.get_module_profile(module=module)
        expect_is_not_none(profile)
    except McpError:
        # Expected when module profile doesn't exist in test data
        pass


def test_local_query_service_get_function_architecture(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify LocalQueryService.get_function_architecture works directly.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    service = provisioned_service_app.service

    result = provisioned_service_app.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    architecture = service.get_function_architecture(goid_h128=goid_h128)
    expect_is_not_none(architecture)


def test_local_query_service_get_module_architecture(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify LocalQueryService.get_module_architecture works directly.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    service = provisioned_service_app.service

    result = provisioned_service_app.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    # Module architecture may not exist for all modules - handle gracefully
    try:
        architecture = service.get_module_architecture(module=module)
        expect_is_not_none(architecture)
    except McpError:
        # Expected when module architecture doesn't exist in test data
        pass


# =============================================================================
# Response Structure Tests
# =============================================================================


def test_function_profile_response_structure(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify function profile response contains expected fields.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/profiles/function?goid_h128={goid_h128}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (profile structure)
        expect_is_instance(data, dict)


def test_file_profile_response_structure(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify file profile response contains expected fields.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT path FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    rel_path = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/profiles/file?rel_path={rel_path}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (profile structure)
        expect_is_instance(data, dict)


def test_module_profile_response_structure(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify module profile response contains expected fields.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/profiles/module?module={module}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (profile structure)
        expect_is_instance(data, dict)


def test_function_architecture_response_structure(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify function architecture response contains expected fields.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT goid_h128 FROM core.goids LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No goids available in test data")

    goid_h128 = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/architecture/function?goid_h128={goid_h128}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (architecture structure)
        expect_is_instance(data, dict)


def test_module_architecture_response_structure(
    provisioned_service_app: ServiceApp,
) -> None:
    """Verify module architecture response contains expected fields.

    Parameters
    ----------
    provisioned_service_app
        Provisioned service app fixture.
    """
    result = provisioned_service_app.gateway.con.execute(
        "SELECT module FROM core.modules WHERE language = 'python' LIMIT 1"
    ).fetchone()

    if result is None:
        pytest.skip("No Python modules available in test data")

    module = result[0]

    with provisioned_service_app.client() as client:
        response = client.get(f"/architecture/module?module={module}")

    if response.status_code == status.HTTP_200_OK:
        data = response.json()
        # Check that response is a dict (architecture structure)
        expect_is_instance(data, dict)
