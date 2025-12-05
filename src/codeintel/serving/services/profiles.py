"""Profile and architecture delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving import domain_models as dm
from codeintel.serving.backend.query_api import DuckDBQueryApi
from codeintel.serving.mcp.models import (
    FileProfileResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
)
from codeintel.serving.services.http_transport import _HttpTransportMixin


class _ProfileQueryDelegates:
    """Local profile-query delegates calling DuckDBQueryService."""

    query: DuckDBQueryApi
    _call: Callable[..., Any]

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        raw_resp = self._call(
            "get_function_profile",
            lambda: self.query.functions.get_function_profile(goid_h128=goid_h128),
        )
        if isinstance(raw_resp, dm.FunctionProfileResult):
            return raw_resp
        if isinstance(raw_resp, FunctionProfileResponse):
            return raw_resp.to_domain()
        return FunctionProfileResponse.model_validate(raw_resp).to_domain()

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        raw_resp = self._call(
            "get_file_profile", lambda: self.query.modules.get_file_profile(rel_path=rel_path)
        )
        if isinstance(raw_resp, dm.FileProfileResult):
            return raw_resp
        if isinstance(raw_resp, FileProfileResponse):
            return raw_resp.to_domain()
        return FileProfileResponse.model_validate(raw_resp).to_domain()

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        raw_resp = self._call(
            "get_module_profile", lambda: self.query.modules.get_module_profile(module=module)
        )
        if isinstance(raw_resp, dm.ModuleProfileResult):
            return raw_resp
        if isinstance(raw_resp, ModuleProfileResponse):
            return raw_resp.to_domain()
        return ModuleProfileResponse.model_validate(raw_resp).to_domain()

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        raw_resp = self._call(
            "get_function_architecture",
            lambda: self.query.functions.get_function_architecture(goid_h128=goid_h128),
        )
        if isinstance(raw_resp, dm.FunctionArchitectureResult):
            return raw_resp
        if isinstance(raw_resp, FunctionArchitectureResponse):
            return raw_resp.to_domain()
        return FunctionArchitectureResponse.model_validate(raw_resp).to_domain()

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        raw_resp = self._call(
            "get_module_architecture",
            lambda: self.query.modules.get_module_architecture(module=module),
        )
        if isinstance(raw_resp, dm.ModuleArchitectureResult):
            return raw_resp
        if isinstance(raw_resp, ModuleArchitectureResponse):
            return raw_resp.to_domain()
        return ModuleArchitectureResponse.model_validate(raw_resp).to_domain()


class _HttpProfileQueryMixin(_HttpTransportMixin):
    """HTTP-based profile query mixin.

    Architecture Note
    -----------------
    Implements HTTP transport path for profile queries. Performs bidirectional
    domain/response conversion: receives HTTP responses, normalizes to Pydantic
    models, and converts to domain models via ``to_domain()`` to satisfy the
    service layer contract.

    See ``codeintel.serving.domain_models`` for the full architecture contract.
    """

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        def _run() -> FunctionProfileResponse:
            payload = self.request_json("/profiles/function", {"goid_h128": goid_h128})
            if isinstance(payload, dm.FunctionProfileResult):
                return FunctionProfileResponse.from_domain(payload)
            if isinstance(payload, FunctionProfileResponse):
                return payload
            return FunctionProfileResponse.model_validate(payload)

        pydantic_resp: FunctionProfileResponse = self._http_call("get_function_profile", _run)
        return pydantic_resp.to_domain()

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        def _run() -> FileProfileResponse:
            payload = self.request_json("/profiles/file", {"rel_path": rel_path})
            if isinstance(payload, dm.FileProfileResult):
                return FileProfileResponse.from_domain(payload)
            if isinstance(payload, FileProfileResponse):
                return payload
            return FileProfileResponse.model_validate(payload)

        pydantic_resp: FileProfileResponse = self._http_call("get_file_profile", _run)
        return pydantic_resp.to_domain()

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        def _run() -> ModuleProfileResponse:
            payload = self.request_json("/profiles/module", {"module": module})
            if isinstance(payload, dm.ModuleProfileResult):
                return ModuleProfileResponse.from_domain(payload)
            if isinstance(payload, ModuleProfileResponse):
                return payload
            return ModuleProfileResponse.model_validate(payload)

        pydantic_resp: ModuleProfileResponse = self._http_call("get_module_profile", _run)
        return pydantic_resp.to_domain()

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        def _run() -> FunctionArchitectureResponse:
            payload = self.request_json("/architecture/function", {"goid_h128": goid_h128})
            if isinstance(payload, dm.FunctionArchitectureResult):
                return FunctionArchitectureResponse.from_domain(payload)
            if isinstance(payload, FunctionArchitectureResponse):
                return payload
            return FunctionArchitectureResponse.model_validate(payload)

        pydantic_resp: FunctionArchitectureResponse = self._http_call(
            "get_function_architecture", _run
        )
        return pydantic_resp.to_domain()

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        def _run() -> ModuleArchitectureResponse:
            payload = self.request_json("/architecture/module", {"module": module})
            if isinstance(payload, dm.ModuleArchitectureResult):
                return ModuleArchitectureResponse.from_domain(payload)
            if isinstance(payload, ModuleArchitectureResponse):
                return payload
            return ModuleArchitectureResponse.model_validate(payload)

        pydantic_resp: ModuleArchitectureResponse = self._http_call("get_module_architecture", _run)
        return pydantic_resp.to_domain()


__all__ = ["_HttpProfileQueryMixin", "_ProfileQueryDelegates"]
