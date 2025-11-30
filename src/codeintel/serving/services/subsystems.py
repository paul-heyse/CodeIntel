"""Subsystem, hints, and coverage delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving import domain_models as dm
from codeintel.serving.backend import DuckDBQueryService, clamp_limit_value
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
from codeintel.serving.services.http_transport import _HttpTransportMixin


class _SubsystemQueryDelegates:
    """Local delegates for subsystem-related queries."""

    query: DuckDBQueryService
    _call: Callable[..., Any]

    def list_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSummaryResult:
        pydantic_resp: SubsystemSummaryResponse = self._call(
            "list_subsystems",
            lambda: self.query.list_subsystems(limit=limit, role=role, q=q),
        )
        return pydantic_resp.to_domain()

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        pydantic_resp: ModuleSubsystemResponse = self._call(
            "get_module_subsystems",
            lambda: self.query.get_module_subsystems(module=module),
        )
        return pydantic_resp.to_domain()

    def get_file_hints(self, *, rel_path: str) -> dm.FileHintsResult:
        pydantic_resp: FileHintsResponse = self._call(
            "get_file_hints", lambda: self.query.get_file_hints(rel_path=rel_path)
        )
        return pydantic_resp.to_domain()

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        pydantic_resp: SubsystemModulesResponse = self._call(
            "get_subsystem_modules",
            lambda: self.query.get_subsystem_modules(
                subsystem_id=subsystem_id, module_limit=module_limit
            ),
        )
        return pydantic_resp.to_domain()

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> dm.SubsystemSearchResult:
        pydantic_resp: SubsystemSearchResponse = self._call(
            "search_subsystems",
            lambda: self.query.search_subsystems(limit=limit, role=role, q=q),
        )
        return pydantic_resp.to_domain()

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> dm.SubsystemModulesResult:
        pydantic_resp: SubsystemModulesResponse = self._call(
            "summarize_subsystem",
            lambda: self.query.summarize_subsystem(
                subsystem_id=subsystem_id, module_limit=module_limit
            ),
        )
        return pydantic_resp.to_domain()

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        pydantic_resp: SubsystemProfileResponse = self._call(
            "list_subsystem_profiles",
            lambda: self.query.list_subsystem_profiles(limit=limit),
            dataset="docs.v_subsystem_profile",
        )
        return pydantic_resp.to_domain()

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        pydantic_resp: SubsystemCoverageResponse = self._call(
            "list_subsystem_coverage",
            lambda: self.query.list_subsystem_coverage(limit=limit),
            dataset="docs.v_subsystem_coverage",
        )
        return pydantic_resp.to_domain()


class _HttpSubsystemQueryMixin(_HttpTransportMixin):
    """HTTP-based subsystem query APIs."""

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSummaryResult:
        def _run() -> SubsystemSummaryResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return SubsystemSummaryResponse(subsystems=[], meta=ResponseMeta())
            payload = self.request_json(
                "/architecture/subsystems",
                {"limit": clamp.applied, "role": role, "q": q},
            )
            if isinstance(payload, dm.SubsystemSummaryResult):
                return SubsystemSummaryResponse.from_domain(payload)
            if isinstance(payload, SubsystemSummaryResponse):
                return payload
            return SubsystemSummaryResponse.model_validate(payload)

        pydantic_resp: SubsystemSummaryResponse = self._http_call("list_subsystems", _run)
        return pydantic_resp.to_domain()

    def get_module_subsystems(self, *, module: str) -> dm.ModuleSubsystemResult:
        def _run() -> ModuleSubsystemResponse:
            payload = self.request_json("/architecture/module-subsystems", {"module": module})
            if isinstance(payload, dm.ModuleSubsystemResult):
                return ModuleSubsystemResponse.from_domain(payload)
            if isinstance(payload, ModuleSubsystemResponse):
                return payload
            return ModuleSubsystemResponse.model_validate(payload)

        pydantic_resp: ModuleSubsystemResponse = self._http_call(
            "get_module_subsystems", _run
        )
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
        def _run() -> SubsystemModulesResponse:
            payload: dict[str, object] = {"subsystem_id": subsystem_id}
            if module_limit is not None:
                payload["module_limit"] = module_limit
            response = self.request_json("/architecture/subsystem", payload)
            if isinstance(response, dm.SubsystemModulesResult):
                return SubsystemModulesResponse.from_domain(response)
            if isinstance(response, SubsystemModulesResponse):
                return response
            return SubsystemModulesResponse.model_validate(response)

        pydantic_resp: SubsystemModulesResponse = self._http_call(
            "get_subsystem_modules", _run
        )
        return pydantic_resp.to_domain()

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> dm.SubsystemSearchResult:
        def _run() -> SubsystemSearchResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return SubsystemSearchResponse(subsystems=[], meta=ResponseMeta())
            payload = self.request_json(
                "/architecture/subsystems",
                {"limit": clamp.applied, "role": role, "q": q},
            )
            if isinstance(payload, dm.SubsystemSearchResult):
                return SubsystemSearchResponse.from_domain(payload)
            if isinstance(payload, SubsystemSearchResponse):
                return payload
            return SubsystemSearchResponse.model_validate(payload)

        pydantic_resp: SubsystemSearchResponse = self._http_call("search_subsystems", _run)
        return pydantic_resp.to_domain()

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> dm.SubsystemModulesResult:
        return self.get_subsystem_modules(subsystem_id=subsystem_id, module_limit=module_limit)

    def list_subsystem_profiles(self, *, limit: int | None = None) -> dm.SubsystemProfileResult:
        def _run() -> SubsystemProfileResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit, default=applied_limit, max_limit=self.limits.max_rows_per_call
            )
            if clamp.has_error:
                return SubsystemProfileResponse(profiles=[], meta=ResponseMeta())
            payload = self.request_json(
                "/architecture/subsystem-profiles",
                {"limit": clamp.applied},
            )
            if isinstance(payload, dm.SubsystemProfileResult):
                return SubsystemProfileResponse.from_domain(payload)
            if isinstance(payload, SubsystemProfileResponse):
                return payload
            return SubsystemProfileResponse.model_validate(payload)

        pydantic_resp: SubsystemProfileResponse = self._http_call(
            "list_subsystem_profiles",
            _run,
            dataset="docs.v_subsystem_profile",
        )
        return pydantic_resp.to_domain()

    def list_subsystem_coverage(self, *, limit: int | None = None) -> dm.SubsystemCoverageResult:
        def _run() -> SubsystemCoverageResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit, default=applied_limit, max_limit=self.limits.max_rows_per_call
            )
            if clamp.has_error:
                return SubsystemCoverageResponse(coverage=[], meta=ResponseMeta())
            payload = self.request_json(
                "/architecture/subsystem-coverage",
                {"limit": clamp.applied},
            )
            if isinstance(payload, dm.SubsystemCoverageResult):
                return SubsystemCoverageResponse.from_domain(payload)
            if isinstance(payload, SubsystemCoverageResponse):
                return payload
            return SubsystemCoverageResponse.model_validate(payload)

        pydantic_resp: SubsystemCoverageResponse = self._http_call(
            "list_subsystem_coverage",
            _run,
            dataset="docs.v_subsystem_coverage",
        )
        return pydantic_resp.to_domain()


__all__ = ["_HttpSubsystemQueryMixin", "_SubsystemQueryDelegates"]
