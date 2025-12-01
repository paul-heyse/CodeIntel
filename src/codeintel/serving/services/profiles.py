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
        pydantic_resp: FunctionProfileResponse = self._call(
            "get_function_profile",
            lambda: self.query.functions.get_function_profile(goid_h128=goid_h128),
        )
        return pydantic_resp.to_domain()

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        pydantic_resp: FileProfileResponse = self._call(
            "get_file_profile", lambda: self.query.modules.get_file_profile(rel_path=rel_path)
        )
        return pydantic_resp.to_domain()

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        pydantic_resp: ModuleProfileResponse = self._call(
            "get_module_profile", lambda: self.query.modules.get_module_profile(module=module)
        )
        return pydantic_resp.to_domain()

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        pydantic_resp: FunctionArchitectureResponse = self._call(
            "get_function_architecture",
            lambda: self.query.functions.get_function_architecture(goid_h128=goid_h128),
        )
        return pydantic_resp.to_domain()

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        pydantic_resp: ModuleArchitectureResponse = self._call(
            "get_module_architecture",
            lambda: self.query.modules.get_module_architecture(module=module),
        )
        return pydantic_resp.to_domain()


class _HttpProfileQueryMixin(_HttpTransportMixin):
    """HTTP-based profile query mixin."""

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
