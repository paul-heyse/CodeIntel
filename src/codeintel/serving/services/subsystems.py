"""Subsystem, hints, and coverage delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    ) -> SubsystemSummaryResponse:
        return self._call(
            "list_subsystems",
            lambda: self.query.list_subsystems(limit=limit, role=role, q=q),
        )

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        return self._call(
            "get_module_subsystems",
            lambda: self.query.get_module_subsystems(module=module),
        )

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        return self._call("get_file_hints", lambda: self.query.get_file_hints(rel_path=rel_path))

    def get_subsystem_modules(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        return self._call(
            "get_subsystem_modules",
            lambda: self.query.get_subsystem_modules(
                subsystem_id=subsystem_id, module_limit=module_limit
            ),
        )

    def search_subsystems(
        self, *, limit: int | None = None, role: str | None = None, q: str | None = None
    ) -> SubsystemSearchResponse:
        return self._call(
            "search_subsystems",
            lambda: self.query.search_subsystems(limit=limit, role=role, q=q),
        )

    def summarize_subsystem(
        self, *, subsystem_id: str, module_limit: int | None = None
    ) -> SubsystemModulesResponse:
        return self._call(
            "summarize_subsystem",
            lambda: self.query.summarize_subsystem(
                subsystem_id=subsystem_id, module_limit=module_limit
            ),
        )

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        return self._call(
            "list_subsystem_profiles",
            lambda: self.query.list_subsystem_profiles(limit=limit),
            dataset="docs.v_subsystem_profile",
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        return self._call(
            "list_subsystem_coverage",
            lambda: self.query.list_subsystem_coverage(limit=limit),
            dataset="docs.v_subsystem_coverage",
        )


class _HttpSubsystemQueryMixin(_HttpTransportMixin):
    """HTTP-based subsystem query APIs."""

    def list_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSummaryResponse:
        def _run() -> SubsystemSummaryResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return SubsystemSummaryResponse(subsystems=[], meta=ResponseMeta())
            return SubsystemSummaryResponse.model_validate(
                self.request_json(
                    "/architecture/subsystems",
                    {"limit": clamp.applied, "role": role, "q": q},
                )
            )

        return self._http_call("list_subsystems", _run)

    def get_module_subsystems(self, *, module: str) -> ModuleSubsystemResponse:
        def _run() -> ModuleSubsystemResponse:
            return ModuleSubsystemResponse.model_validate(
                self.request_json("/architecture/module-subsystems", {"module": module})
            )

        return self._http_call("get_module_subsystems", _run)

    def get_file_hints(self, *, rel_path: str) -> FileHintsResponse:
        def _run() -> FileHintsResponse:
            return FileHintsResponse.model_validate(
                self.request_json("/ide/hints", {"rel_path": rel_path})
            )

        return self._http_call("get_file_hints", _run)

    def get_subsystem_modules(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        def _run() -> SubsystemModulesResponse:
            payload: dict[str, object] = {"subsystem_id": subsystem_id}
            if module_limit is not None:
                payload["module_limit"] = module_limit
            return SubsystemModulesResponse.model_validate(
                self.request_json("/architecture/subsystem", payload)
            )

        return self._http_call("get_subsystem_modules", _run)

    def search_subsystems(
        self,
        *,
        limit: int | None = None,
        role: str | None = None,
        q: str | None = None,
    ) -> SubsystemSearchResponse:
        def _run() -> SubsystemSearchResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit,
                default=applied_limit,
                max_limit=self.limits.max_rows_per_call,
            )
            if clamp.has_error:
                return SubsystemSearchResponse(subsystems=[], meta=ResponseMeta())
            return SubsystemSearchResponse.model_validate(
                self.request_json(
                    "/architecture/subsystems",
                    {"limit": clamp.applied, "role": role, "q": q},
                )
            )

        return self._http_call("search_subsystems", _run)

    def summarize_subsystem(
        self,
        *,
        subsystem_id: str,
        module_limit: int | None = None,
    ) -> SubsystemModulesResponse:
        return self.get_subsystem_modules(subsystem_id=subsystem_id, module_limit=module_limit)

    def list_subsystem_profiles(self, *, limit: int | None = None) -> SubsystemProfileResponse:
        def _run() -> SubsystemProfileResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit, default=applied_limit, max_limit=self.limits.max_rows_per_call
            )
            if clamp.has_error:
                return SubsystemProfileResponse(profiles=[], meta=ResponseMeta())
            return SubsystemProfileResponse.model_validate(
                self.request_json(
                    "/architecture/subsystem-profiles",
                    {"limit": clamp.applied},
                )
            )

        return self._http_call(
            "list_subsystem_profiles",
            _run,
            dataset="docs.v_subsystem_profile",
        )

    def list_subsystem_coverage(self, *, limit: int | None = None) -> SubsystemCoverageResponse:
        def _run() -> SubsystemCoverageResponse:
            applied_limit = self.limits.default_limit if limit is None else limit
            clamp = clamp_limit_value(
                applied_limit, default=applied_limit, max_limit=self.limits.max_rows_per_call
            )
            if clamp.has_error:
                return SubsystemCoverageResponse(coverage=[], meta=ResponseMeta())
            return SubsystemCoverageResponse.model_validate(
                self.request_json(
                    "/architecture/subsystem-coverage",
                    {"limit": clamp.applied},
                )
            )

        return self._http_call(
            "list_subsystem_coverage",
            _run,
            dataset="docs.v_subsystem_coverage",
        )


__all__ = ["_HttpSubsystemQueryMixin", "_SubsystemQueryDelegates"]
