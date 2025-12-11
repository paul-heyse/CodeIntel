"""Tests for subsystem handlers following the unified handler pattern."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from codeintel.serving.mcp.models import (
    ModuleWithSubsystemRow,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemCoverageRow,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemProfileRow,
    SubsystemSummaryResponse,
    SubsystemSummaryRow,
)
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.serving_contexts import ProvisionedServiceContext
from tests.cli.handlers.conftest import CommandContextFactory, HandlerContextBuilder

HTTP_NOT_FOUND = 404
TEST_REPO = "test/repo"
TEST_COMMIT = "abc123"


def test_subsystem_list_handler_returns_ok_with_subsystems(
    architecture_service_context: ProvisionedServiceContext,
    command_context_factory: CommandContextFactory,
) -> None:
    """Handler returns success result with subsystem list."""
    mock_response = SubsystemSummaryResponse(
        subsystems=[
            SubsystemSummaryRow(
                repo=TEST_REPO,
                commit=TEST_COMMIT,
                subsystem_id="core",
                name="Core",
                description="Core components",
                module_count=10,
            ),
        ],
        meta=ResponseMeta(),
    )

    ctx_params: dict[str, object] = {"repo": TEST_REPO, "commit": TEST_COMMIT}
    backend = architecture_service_context.backend
    backend.list_subsystems = MagicMock(return_value=mock_response)  # type: ignore[attr-defined]

    with command_context_factory(ctx_params) as ctx:
        result = subsystem_list_handler(ctx)

    backend.list_subsystems.assert_called_once_with(limit=0, role=None, q=None)
    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemListResult)
    if result.data is not None:
        expect_equal(len(result.data.subsystems), 1)
        expect_equal(result.data.subsystems[0]["subsystem_id"], "core")


def test_subsystem_list_handler_passes_filter_params(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler passes role, query, and limit params to backend."""
    mock_response = SubsystemSummaryResponse(
        subsystems=[],
        meta=ResponseMeta(),
    )

    with (
        patch(
            "codeintel.cli.handlers.subsystem.build_backend_resource",
            return_value=SimpleNamespace(backend=architecture_service_context.backend),
        ),
        patch.object(
            architecture_service_context.backend,
            "list_subsystems",
            return_value=mock_response,
        ) as list_subsystems,
    ):
        ctx = handler_context_builder(
            architecture_service_context,
            "subsystem.test",
            {"role": "model", "query": "search", "limit": 10},
        )

        subsystem_list_handler(ctx)

    list_subsystems.assert_called_once_with(limit=10, role="model", q="search")


def test_subsystem_show_handler_returns_ok_when_found(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result when subsystem is found."""
    mock_response = SubsystemModulesResponse(
        found=True,
        subsystem=SubsystemSummaryRow(
            repo=TEST_REPO,
            commit=TEST_COMMIT,
            subsystem_id="core",
            name="Core",
            description="Core components",
            module_count=2,
        ),
        modules=[
            ModuleWithSubsystemRow(
                repo=TEST_REPO,
                commit=TEST_COMMIT,
                module="pkg.mod1",
                subsystem_id="core",
                subsystem_name="Core",
            ),
            ModuleWithSubsystemRow(
                repo=TEST_REPO,
                commit=TEST_COMMIT,
                module="pkg.mod2",
                subsystem_id="core",
                subsystem_name="Core",
            ),
        ],
        meta=ResponseMeta(),
    )

    with (
        patch(
            "codeintel.cli.handlers.subsystem.build_backend_resource",
            return_value=SimpleNamespace(backend=architecture_service_context.backend),
        ),
        patch.object(
            architecture_service_context.backend,
            "get_subsystem_modules",
            return_value=mock_response,
        ),
    ):
        ctx = handler_context_builder(
            architecture_service_context,
            "subsystem.test",
            {"subsystem_id": "core"},
        )

        result = subsystem_show_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemShowResult)
    if result.data is not None:
        expect_equal(result.data.subsystem["subsystem_id"], "core")
        expect_equal(len(result.data.modules), 2)


def test_subsystem_show_handler_returns_fail_when_not_found(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns failure result when subsystem not found."""
    mock_response = SubsystemModulesResponse(
        found=False,
        subsystem=None,
        modules=[],
        meta=ResponseMeta(),
    )

    with (
        patch(
            "codeintel.cli.handlers.subsystem.build_backend_resource",
            return_value=SimpleNamespace(backend=architecture_service_context.backend),
        ),
        patch.object(
            architecture_service_context.backend,
            "get_subsystem_modules",
            return_value=mock_response,
        ),
    ):
        ctx = handler_context_builder(
            architecture_service_context,
            "subsystem.test",
            {"subsystem_id": "nonexistent"},
        )

        result = subsystem_show_handler(ctx)

    expect_true(not result.success)
    expect_is_not_none(result.error)
    if result.error is not None:
        expect_equal(result.error.status, HTTP_NOT_FOUND)
        expect_is_not_none(result.error.detail)
        if result.error.detail is not None:
            expect_true("nonexistent" in result.error.detail)


def test_subsystem_show_handler_raises_when_id_missing(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler raises ParamError when subsystem_id is missing."""
    ctx = handler_context_builder(architecture_service_context, "subsystem.test", {})

    with pytest.raises(ParamError, match="Required parameter 'subsystem_id' not provided"):
        subsystem_show_handler(ctx)


def test_subsystem_profiles_handler_returns_ok(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result with profile list."""
    mock_response = SubsystemProfileResponse(
        profiles=[
            SubsystemProfileRow(
                repo=TEST_REPO,
                commit=TEST_COMMIT,
                subsystem_id="core",
                name="Core",
                module_count=10,
            ),
        ],
        meta=ResponseMeta(),
    )

    with (
        patch(
            "codeintel.cli.handlers.subsystem.build_backend_resource",
            return_value=SimpleNamespace(backend=architecture_service_context.backend),
        ),
        patch.object(
            architecture_service_context.backend.service,
            "list_subsystem_profiles",
            return_value=mock_response,
        ),
    ):
        ctx = handler_context_builder(
            architecture_service_context,
            "subsystem.test",
            {"limit": 5},
        )

        result = subsystem_profiles_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemProfilesResult)
    if result.data is not None:
        expect_equal(len(result.data.profiles), 1)


def test_subsystem_coverage_handler_returns_ok(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result with coverage data."""
    mock_response = SubsystemCoverageResponse(
        coverage=[
            SubsystemCoverageRow(
                repo=TEST_REPO,
                commit=TEST_COMMIT,
                subsystem_id="core",
                name="Core",
            ),
        ],
        meta=ResponseMeta(),
    )

    with (
        patch(
            "codeintel.cli.handlers.subsystem.build_backend_resource",
            return_value=SimpleNamespace(backend=architecture_service_context.backend),
        ),
        patch.object(
            architecture_service_context.backend.service,
            "list_subsystem_coverage",
            return_value=mock_response,
        ),
    ):
        ctx = handler_context_builder(
            architecture_service_context,
            "subsystem.test",
            {},
        )

        result = subsystem_coverage_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemCoverageResult)
    if result.data is not None:
        expect_equal(len(result.data.coverage), 1)


def test_subsystem_memberships_handler_returns_ok(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler returns success result with membership data."""
    mock_response = MagicMock()
    mock_response.found = True
    mock_response.memberships = [
        MagicMock(
            model_dump=lambda: {
                "module": "pkg.mod",
                "subsystem_id": "core",
                "subsystem_name": "Core",
            }
        ),
    ]
    mock_response.meta = MagicMock(model_dump=lambda: {"total_count": 1})

    with (
        patch(
            "codeintel.cli.handlers.subsystem.build_backend_resource",
            return_value=SimpleNamespace(backend=architecture_service_context.backend),
        ),
        patch.object(
            architecture_service_context.backend,
            "get_module_subsystems",
            return_value=mock_response,
        ),
    ):
        ctx = handler_context_builder(
            architecture_service_context,
            "subsystem.test",
            {"module": "pkg.mod"},
        )

        result = subsystem_module_memberships_handler(ctx)

    expect_true(result.success)
    expect_is_not_none(result.data)
    expect_is_instance(result.data, SubsystemMembershipResult)
    if result.data is not None:
        expect_true(result.data.found)
        expect_equal(len(result.data.memberships), 1)


def test_subsystem_memberships_handler_raises_when_module_missing(
    architecture_service_context: ProvisionedServiceContext,
    handler_context_builder: HandlerContextBuilder,
) -> None:
    """Handler raises ParamError when module is missing."""
    ctx = handler_context_builder(architecture_service_context, "subsystem.test", {})

    with pytest.raises(ParamError, match="Required parameter 'module' not provided"):
        subsystem_module_memberships_handler(ctx)


def test_subsystem_list_result_to_dict() -> None:
    """SubsystemListResult.to_dict returns expected structure."""
    result = SubsystemListResult(
        subsystems=[{"subsystem_id": "core", "name": "Core"}],
        meta={"total_count": 1},
    )
    data = result.to_dict()

    expect_equal(data["subsystems"], [{"subsystem_id": "core", "name": "Core"}])
    expect_equal(data["meta"], {"total_count": 1})


def test_subsystem_show_result_to_dict() -> None:
    """SubsystemShowResult.to_dict returns expected structure."""
    result = SubsystemShowResult(
        subsystem={"subsystem_id": "core", "name": "Core"},
        modules=[{"module": "pkg.mod"}],
        meta={"total_count": 1},
    )
    data = result.to_dict()

    expect_equal(data["subsystem"], {"subsystem_id": "core", "name": "Core"})
    expect_equal(data["modules"], [{"module": "pkg.mod"}])
    expect_equal(data["meta"], {"total_count": 1})


def test_subsystem_profiles_result_to_dict() -> None:
    """SubsystemProfilesResult.to_dict returns expected structure."""
    result = SubsystemProfilesResult(
        profiles=[{"subsystem_id": "core", "module_count": 10}],
        meta={"total_count": 1},
    )
    data = result.to_dict()

    expect_equal(data["profiles"], [{"subsystem_id": "core", "module_count": 10}])
    expect_equal(data["meta"], {"total_count": 1})


def test_subsystem_coverage_result_to_dict() -> None:
    """SubsystemCoverageResult.to_dict returns expected structure."""
    result = SubsystemCoverageResult(
        coverage=[{"subsystem_id": "core", "coverage_pct": 80.0}],
        meta={"total_count": 1},
    )
    data = result.to_dict()

    expect_equal(data["coverage"], [{"subsystem_id": "core", "coverage_pct": 80.0}])
    expect_equal(data["meta"], {"total_count": 1})


def test_subsystem_membership_result_to_dict() -> None:
    """SubsystemMembershipResult.to_dict returns expected structure."""
    result = SubsystemMembershipResult(
        found=True,
        memberships=[{"module": "pkg.mod", "subsystem_id": "core"}],
        meta={"total_count": 1},
    )
    data = result.to_dict()

    expect_true(data["found"])
    expect_equal(data["memberships"], [{"module": "pkg.mod", "subsystem_id": "core"}])
    expect_equal(data["meta"], {"total_count": 1})
