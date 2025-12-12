"""Tests for subsystem service delegates.

This module tests subsystem query delegates via LocalQueryService.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import status

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import BackendLimits
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.http_payloads import (
    RequestRecorder,
    make_subsystem_http_responses,
)
from tests._helpers.serving_harnesses import HttpSubsystemHarness, SubsystemDelegateHarness

if TYPE_CHECKING:
    from tests._helpers.analytics_samples import AnalyticsSamples
    from tests._helpers.serving_contexts import ProvisionedServiceContext


@pytest.mark.parametrize("query", ["", "limit=5"])
def test_list_subsystems_returns_data(
    architecture_service_ctx: ProvisionedServiceContext,
    query: str,
) -> None:
    """Verify subsystem listing returns results with optional limit."""
    path = "/architecture/subsystems"
    if query:
        path = f"{path}?{query}"

    with architecture_service_ctx.client() as client:
        response = client.get(path)

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_get_subsystem_modules(
    architecture_service_ctx: ProvisionedServiceContext,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify subsystem modules endpoint functions.

    Parameters
    ----------
    architecture_service_ctx
        Service app wired to architecture data.
    architecture_samples
        Sample analytics identifiers for architecture subsystem.
    """
    with architecture_service_ctx.client() as client:
        response = client.get(
            f"/architecture/subsystem?subsystem_id={architecture_samples.subsystem_id}"
        )

    expect_true(
        response.status_code
        in {
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
        }
    )


def test_get_module_subsystems(
    architecture_service_ctx: ProvisionedServiceContext,
    architecture_samples: AnalyticsSamples,
) -> None:
    """Verify module subsystems endpoint functions.

    Parameters
    ----------
    architecture_service_ctx
        Service app wired to architecture data.
    architecture_samples
        Sample analytics identifiers for architecture module.
    """
    with architecture_service_ctx.client() as client:
        response = client.get(
            f"/architecture/module-subsystems?module={architecture_samples.module}"
        )

    expect_true(
        response.status_code
        in {
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
        }
    )


def test_subsystem_coverage_endpoint(
    architecture_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify subsystem coverage endpoint returns data.

    Parameters
    ----------
    architecture_service_ctx
        Service app wired to architecture data.
    """
    with architecture_service_ctx.client() as client:
        response = client.get("/architecture/subsystem-coverage")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_subsystem_profiles_endpoint(
    architecture_service_ctx: ProvisionedServiceContext,
) -> None:
    """Verify subsystem profiles endpoint returns data.

    Parameters
    ----------
    architecture_service_ctx
        Service app wired to architecture data.
    """
    with architecture_service_ctx.client() as client:
        response = client.get("/architecture/subsystem-profiles")

    expect_equal(response.status_code, status.HTTP_200_OK)


def test_subsystem_delegate_normalization_variants() -> None:
    """Ensure subsystem delegates normalize dict and response payloads to domain."""
    payloads: dict[str, object] = {
        "list_subsystems": {"subsystems": [], "meta": ResponseMeta().model_dump()},
        "get_module_subsystems": ModuleSubsystemResponse(
            found=True, memberships=[], meta=ResponseMeta()
        ),
        "get_file_hints": FileHintsResponse(
            found=True,
            hints=[],
            meta=ResponseMeta(),
        ),
        "get_subsystem_modules": SubsystemModulesResponse(
            found=True,
            modules=[],
            meta=ResponseMeta(),
        ),
        "search_subsystems": SubsystemSearchResponse(
            subsystems=[],
            meta=ResponseMeta(),
        ),
        "list_subsystem_profiles": SubsystemProfileResponse(profiles=[], meta=ResponseMeta()),
        "list_subsystem_coverage": SubsystemCoverageResponse(coverage=[], meta=ResponseMeta()),
    }
    delegates = SubsystemDelegateHarness(payloads)

    subsystems = delegates.list_subsystems(limit=1)
    module_subs = delegates.get_module_subsystems(module="m")
    hints = delegates.get_file_hints(rel_path="a.py")
    modules = delegates.get_subsystem_modules(subsystem_id="s")
    search = delegates.search_subsystems(q="s")
    summary = delegates.summarize_subsystem(subsystem_id="s")
    profiles = delegates.list_subsystem_profiles(limit=1)
    coverage = delegates.list_subsystem_coverage(limit=1)

    expect_true(isinstance(subsystems, dm.SubsystemSummaryResult))
    expect_true(isinstance(module_subs, dm.ModuleSubsystemResult))
    expect_true(isinstance(hints, dm.FileHintsResult))
    expect_true(isinstance(modules, dm.SubsystemModulesResult))
    expect_true(isinstance(search, dm.SubsystemSearchResult))
    expect_true(isinstance(summary, dm.SubsystemModulesResult))
    expect_true(isinstance(profiles, dm.SubsystemProfileResult))
    expect_true(isinstance(coverage, dm.SubsystemCoverageResult))
    expect_true(("list_subsystem_profiles", "docs.v_subsystem_profile") in delegates.called)
    expect_true(("list_subsystem_coverage", "docs.v_subsystem_coverage") in delegates.called)


def test_http_subsystem_mixin_clamp_and_problem_fallback() -> None:
    """Cover clamp short-circuit and ProblemError fallback in HTTP mixin."""
    responses: dict[str, object] = {
        "/architecture/subsystems": SubsystemSummaryResponse(
            subsystems=[],
            meta=ResponseMeta(),
        ),
        "/architecture/module-subsystems": ModuleSubsystemResponse(
            found=True, memberships=[], meta=ResponseMeta()
        ),
        "/ide/hints": FileHintsResponse(found=True, hints=[], meta=ResponseMeta()),
        "/architecture/subsystem": SubsystemModulesResponse(
            found=True,
            modules=[],
            meta=ResponseMeta(),
        ),
        "/architecture/subsystem-profiles": SubsystemProfileResponse(
            profiles=[],
            meta=ResponseMeta(),
        ),
        "/architecture/subsystem-coverage": SubsystemCoverageResponse(
            coverage=[],
            meta=ResponseMeta(),
        ),
    }
    requester = RequestRecorder(responses, error_paths={"/architecture/subsystem"})
    http_subs = HttpSubsystemHarness(
        limits=BackendLimits(default_limit=1, max_rows_per_call=1),
        requester=requester,
    )

    empty_summary = http_subs.list_subsystems(limit=-1)
    empty_profiles = http_subs.list_subsystem_profiles(limit=-5)
    empty_coverage = http_subs.list_subsystem_coverage(limit=-2)
    modules = http_subs.get_subsystem_modules(subsystem_id="missing")
    summary = http_subs.summarize_subsystem(subsystem_id="missing")

    expect_true(empty_summary.subsystems == [])
    expect_true(empty_profiles.profiles == [])
    expect_true(empty_coverage.coverage == [])
    expect_true(modules.found is False)
    expect_true(summary.found is False)


def test_http_subsystem_mixin_normalization_paths() -> None:
    """Validate HTTP mixin normalizes module subsystems and hints payloads."""
    responses = make_subsystem_http_responses()
    requester = RequestRecorder(responses)
    http_subs = HttpSubsystemHarness(
        limits=BackendLimits(default_limit=5, max_rows_per_call=10),
        requester=requester,
    )

    subsystems = http_subs.list_subsystems()
    module_subs = http_subs.get_module_subsystems(module="m")
    hints = http_subs.get_file_hints(rel_path="a.py")
    modules = http_subs.get_subsystem_modules(subsystem_id="s")
    profiles = http_subs.list_subsystem_profiles(limit=2)
    coverage = http_subs.list_subsystem_coverage(limit=2)

    expect_true(isinstance(subsystems, dm.SubsystemSummaryResult))
    expect_true(isinstance(module_subs, dm.ModuleSubsystemResult))
    expect_true(isinstance(hints, dm.FileHintsResult))
    expect_true(isinstance(modules, dm.SubsystemModulesResult))
    expect_true(isinstance(profiles, dm.SubsystemProfileResult))
    expect_true(isinstance(coverage, dm.SubsystemCoverageResult))
