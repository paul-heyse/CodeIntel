"""Profile and architecture delegates for query services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codeintel.serving.backend import DuckDBQueryService
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

    query: DuckDBQueryService
    _call: Callable[..., Any]

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        return self._call(
            "get_function_profile",
            lambda: self.query.get_function_profile(goid_h128=goid_h128),
        )

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        return self._call(
            "get_file_profile", lambda: self.query.get_file_profile(rel_path=rel_path)
        )

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        return self._call(
            "get_module_profile", lambda: self.query.get_module_profile(module=module)
        )

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        return self._call(
            "get_function_architecture",
            lambda: self.query.get_function_architecture(goid_h128=goid_h128),
        )

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        return self._call(
            "get_module_architecture",
            lambda: self.query.get_module_architecture(module=module),
        )


class _HttpProfileQueryMixin(_HttpTransportMixin):
    """HTTP-based profile query mixin."""

    def get_function_profile(self, *, goid_h128: int) -> FunctionProfileResponse:
        def _run() -> FunctionProfileResponse:
            return FunctionProfileResponse.model_validate(
                self.request_json("/profiles/function", {"goid_h128": goid_h128})
            )

        return self._http_call("get_function_profile", _run)

    def get_file_profile(self, *, rel_path: str) -> FileProfileResponse:
        def _run() -> FileProfileResponse:
            return FileProfileResponse.model_validate(
                self.request_json("/profiles/file", {"rel_path": rel_path})
            )

        return self._http_call("get_file_profile", _run)

    def get_module_profile(self, *, module: str) -> ModuleProfileResponse:
        def _run() -> ModuleProfileResponse:
            return ModuleProfileResponse.model_validate(
                self.request_json("/profiles/module", {"module": module})
            )

        return self._http_call("get_module_profile", _run)

    def get_function_architecture(self, *, goid_h128: int) -> FunctionArchitectureResponse:
        def _run() -> FunctionArchitectureResponse:
            return FunctionArchitectureResponse.model_validate(
                self.request_json("/architecture/function", {"goid_h128": goid_h128})
            )

        return self._http_call("get_function_architecture", _run)

    def get_module_architecture(self, *, module: str) -> ModuleArchitectureResponse:
        def _run() -> ModuleArchitectureResponse:
            return ModuleArchitectureResponse.model_validate(
                self.request_json("/architecture/module", {"module": module})
            )

        return self._http_call("get_module_architecture", _run)


__all__ = ["_HttpProfileQueryMixin", "_ProfileQueryDelegates"]
