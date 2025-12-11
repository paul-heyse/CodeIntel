"""Tests for subsystem handlers following the unified handler pattern."""

from __future__ import annotations

import pytest

from codeintel.cli.handlers.subsystem import (
    SubsystemCoverageResult,
    SubsystemListResult,
    SubsystemMembershipResult,
    SubsystemProfilesResult,
    SubsystemShowResult,
    subsystem_coverage_handler,
    subsystem_list_handler,
    subsystem_module_memberships_handler,
    subsystem_profiles_handler,
    subsystem_show_handler,
)
from codeintel.cli.services.params import ParamError
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.serving_contexts import ProvisionedServiceContext
from tests.cli.handlers.conftest import HandlerContextBuilder

HTTP_NOT_FOUND = 404


def _first_subsystem_id(service_ctx: ProvisionedServiceContext) -> str:
    subsystems = service_ctx.backend.list_subsystems(limit=1).subsystems
    if not subsystems:
        pytest.skip("Subsystem data not available in test context")
    return subsystems[0].subsystem_id


def test_subsystem_list_handler_returns_ok_with_subsystems(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result with subsystem list."""
    ctx = handler_context_builder(architecture_service_context, "subsystem.list", {"limit": 5})
    result = subsystem_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemListResult)
    if result.data is not None:
        expect_true(len(result.data.subsystems) > 0)
        expect_is_not_none(result.data.subsystems[0].get("subsystem_id"))


def test_subsystem_list_handler_passes_filter_params(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler passes role, query, and limit params to backend."""
    ctx = handler_context_builder(
        architecture_service_context,
        "subsystem.list",
        {"role": "model", "query": "search", "limit": 1},
    )

    result = subsystem_list_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    if result.data is not None:
        expect_equal(result.data.meta.get("requested_limit"), 1)


def test_subsystem_show_handler_returns_ok_when_found(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result when subsystem is found."""
    subsystem_id = _first_subsystem_id(architecture_service_context)
    ctx = handler_context_builder(
        architecture_service_context,
        "subsystem.show",
        {"subsystem_id": subsystem_id},
    )

    result = subsystem_show_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemShowResult)
    if result.data is not None:
        expect_equal(result.data.subsystem["subsystem_id"], subsystem_id)
        expect_true(len(result.data.modules) > 0)


def test_subsystem_show_handler_returns_fail_when_not_found(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns failure result when subsystem not found."""
    ctx = handler_context_builder(
        architecture_service_context,
        "subsystem.show",
        {"subsystem_id": "nonexistent"},
    )

    result = subsystem_show_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_NOT_FOUND)


def test_subsystem_show_handler_raises_when_id_missing(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler raises ParamError when subsystem_id is missing."""
    ctx = handler_context_builder(architecture_service_context, "subsystem.show", {})

    with pytest.raises(ParamError, match="Required parameter 'subsystem_id' not provided"):
        subsystem_show_handler(ctx)


def test_subsystem_profiles_handler_returns_ok(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result with profile list."""
    ctx = handler_context_builder(
        architecture_service_context,
        "subsystem.profiles",
        {"limit": 2},
    )

    result = subsystem_profiles_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemProfilesResult)
    if result.data is not None:
        expect_true(len(result.data.profiles) >= 0)


def test_subsystem_coverage_handler_returns_ok(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result with coverage list."""
    ctx = handler_context_builder(
        architecture_service_context,
        "subsystem.coverage",
        {"limit": 2},
    )

    result = subsystem_coverage_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemCoverageResult)
    if result.data is not None:
        expect_true(len(result.data.coverage) >= 0)


def test_subsystem_module_memberships_handler_returns_ok(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result with module memberships."""
    subsystem_id = _first_subsystem_id(architecture_service_context)
    modules = architecture_service_context.backend.get_subsystem_modules(
        subsystem_id=subsystem_id
    ).modules
    module_name = modules[0].module if modules else "pkg.mod1"
    ctx = handler_context_builder(
        architecture_service_context, "subsystem.memberships", {"module": module_name}
    )

    result = subsystem_module_memberships_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemMembershipResult)
    if result.data is not None:
        expect_true(len(result.data.memberships) >= 0)
