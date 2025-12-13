"""Subsystem, hints, and coverage delegates for query services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    FileHintsResponse,
    Message,
    ModuleSubsystemResponse,
    ResponseMeta,
    SubsystemCoverageResponse,
    SubsystemModulesResponse,
    SubsystemProfileResponse,
    SubsystemSearchResponse,
    SubsystemSummaryResponse,
)
from codeintel.serving.services.conversion import to_domain_result
from codeintel.serving.services.errors import ProblemError
from codeintel.serving.services.transport import _HttpTransportMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.backend.query_api import DuckDBQueryApi


class _SubsystemQueryDelegates:
    """Local delegates for subsystem-related queries."""

    query: DuckDBQueryApi
    _call: Callable[..., Any]

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSummaryResult:
        raw = self._call(
            "list_subsystems",
            lambda: self.query.subsystems.list_subsystems(limit=limit, role=role, q=q),
        )
        return to_domain_result(raw, dm.SubsystemSummaryResult, SubsystemSummaryResponse)

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        raw = self._call(
            "get_module_subsystems",
            lambda: self.query.subsystems.get_module_subsystems(module=module),
        )
        return to_domain_result(raw, dm.ModuleSubsystemResult, ModuleSubsystemResponse)

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        raw = self._call(
            "get_file_hints", lambda: self.query.modules.get_file_hints(rel_path=rel_path)
        )
        return to_domain_result(raw, dm.FileHintsResult, FileHintsResponse)

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        raw = self._call(
            "get_subsystem_modules",
            lambda: self.query.subsystems.get_subsystem_modules(
                subsystem_id=subsystem_id, module_limit=module_limit
            ),
        )
        return to_domain_result(raw, dm.SubsystemModulesResult, SubsystemModulesResponse)

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSearchResult:
        raw = self._call(
            "search_subsystems",
            lambda: self.query.subsystems.search_subsystems(limit=limit, role=role, q=q),
        )
        return to_domain_result(raw, dm.SubsystemSearchResult, SubsystemSearchResponse)

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        raw = self._call(
            "summarize_subsystem",
            lambda: self.query.subsystems.summarize_subsystem(
                subsystem_id=subsystem_id, module_limit=module_limit
            ),
        )
        return to_domain_result(raw, dm.SubsystemModulesResult, SubsystemModulesResponse)

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        raw = self._call(
            "list_subsystem_profiles",
            lambda: self.query.subsystems.list_subsystem_profiles(limit=limit),
            dataset="docs.v_subsystem_profile",
        )
        return to_domain_result(raw, dm.SubsystemProfileResult, SubsystemProfileResponse)

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        raw = self._call(
            "list_subsystem_coverage",
            lambda: self.query.subsystems.list_subsystem_coverage(limit=limit),
            dataset="docs.v_subsystem_coverage",
        )
        return to_domain_result(raw, dm.SubsystemCoverageResult, SubsystemCoverageResponse)


class _HttpSubsystemQueryMixin(_HttpTransportMixin):
    """HTTP-based subsystem query APIs.

    Architecture Note
    -----------------
    Implements HTTP transport path for subsystem queries. Uses ``_http_query()``
    for methods with limit clamping and the standard pattern for methods without.

    See ``codeintel.serving.domain_models`` for the full architecture contract.
    """

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSummaryResult:
        return self._http_query(
            "list_subsystems",
            "/architecture/subsystems",
            {"role": role, "q": q},
            SubsystemSummaryResponse,
            dm.SubsystemSummaryResult,
            empty_data=SubsystemSummaryResponse(subsystems=[]),
            limit=limit,
        )

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        def _run() -> ModuleSubsystemResponse:
            payload = self.request_json("/architecture/module-subsystems", {"module": module})
            if isinstance(payload, dm.ModuleSubsystemResult):
                return ModuleSubsystemResponse.from_domain(payload)
            if isinstance(payload, ModuleSubsystemResponse):
                return payload
            return ModuleSubsystemResponse.model_validate(payload)

        pydantic_resp: ModuleSubsystemResponse = self._http_call("get_module_subsystems", _run)
        return pydantic_resp.to_domain()

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        def _run() -> FileHintsResponse:
            payload = self.request_json("/ide/hints", {"rel_path": rel_path})
            if isinstance(payload, dm.FileHintsResult):
                return FileHintsResponse.from_domain(payload)
            if isinstance(payload, FileHintsResponse):
                return payload
            return FileHintsResponse.model_validate(payload)

        pydantic_resp: FileHintsResponse = self._http_call("get_file_hints", _run)
        return pydantic_resp.to_domain()

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        # Special handling for ProblemError - can't use _http_query
        def _run() -> SubsystemModulesResponse:
            payload: dict[str, object] = {"subsystem_id": subsystem_id}
            if module_limit is not None:
                payload["module_limit"] = module_limit
            try:
                response = self.request_json("/architecture/subsystem", payload)
            except ProblemError:
                return SubsystemModulesResponse(
                    found=False,
                    modules=[],
                    meta=ResponseMeta(
                        messages=[
                            Message(
                                code="not_found",
                                severity="warning",
                                detail="Subsystem not found",
                            )
                        ]
                    ),
                )
            if isinstance(response, dm.SubsystemModulesResult):
                return SubsystemModulesResponse.from_domain(response)
            if isinstance(response, SubsystemModulesResponse):
                return response
            return SubsystemModulesResponse.model_validate(response)

        pydantic_resp: SubsystemModulesResponse = self._http_call("get_subsystem_modules", _run)
        return pydantic_resp.to_domain()

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSearchResult:
        return self._http_query(
            "search_subsystems",
            "/architecture/subsystems",
            {"role": role, "q": q},
            SubsystemSearchResponse,
            dm.SubsystemSearchResult,
            empty_data=SubsystemSearchResponse(subsystems=[]),
            limit=limit,
        )

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return self.get_subsystem_modules(subsystem_id=subsystem_id, module_limit=module_limit)

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        return self._http_query(
            "list_subsystem_profiles",
            "/architecture/subsystem-profiles",
            {},
            SubsystemProfileResponse,
            dm.SubsystemProfileResult,
            empty_data=SubsystemProfileResponse(profiles=[]),
            limit=limit,
            dataset="docs.v_subsystem_profile",
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        return self._http_query(
            "list_subsystem_coverage",
            "/architecture/subsystem-coverage",
            {},
            SubsystemCoverageResponse,
            dm.SubsystemCoverageResult,
            empty_data=SubsystemCoverageResponse(coverage=[]),
            limit=limit,
            dataset="docs.v_subsystem_coverage",
        )


__all__ = ["_HttpSubsystemQueryMixin", "_SubsystemQueryDelegates"]
