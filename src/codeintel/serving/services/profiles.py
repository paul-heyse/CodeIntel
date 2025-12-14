"""Profile and architecture delegates for query services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    FileProfileResponse,
    FunctionArchitectureResponse,
    FunctionProfileResponse,
    ModuleArchitectureResponse,
    ModuleProfileResponse,
)
from codeintel.serving.services.conversion import to_domain_result
from codeintel.serving.services.transport import HttpQuerySpec, _HttpTransportMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.serving.backend.query_api import DuckDBQueryApi


class _ProfileQueryDelegates:
    """Local profile-query delegates calling DuckDBQueryService."""

    query: DuckDBQueryApi
    _call: Callable[..., Any]

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        raw = self._call(
            "get_function_profile",
            lambda: self.query.functions.get_function_profile(goid_h128=goid_h128),
        )
        return to_domain_result(raw, dm.FunctionProfileResult, FunctionProfileResponse)

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        raw = self._call(
            "get_file_profile", lambda: self.query.modules.get_file_profile(rel_path=rel_path)
        )
        return to_domain_result(raw, dm.FileProfileResult, FileProfileResponse)

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        raw = self._call(
            "get_module_profile", lambda: self.query.modules.get_module_profile(module=module)
        )
        return to_domain_result(raw, dm.ModuleProfileResult, ModuleProfileResponse)

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        raw = self._call(
            "get_function_architecture",
            lambda: self.query.functions.get_function_architecture(goid_h128=goid_h128),
        )
        return to_domain_result(raw, dm.FunctionArchitectureResult, FunctionArchitectureResponse)

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        raw = self._call(
            "get_module_architecture",
            lambda: self.query.modules.get_module_architecture(module=module),
        )
        return to_domain_result(raw, dm.ModuleArchitectureResult, ModuleArchitectureResponse)


class _HttpProfileQueryMixin(_HttpTransportMixin):
    """HTTP-based profile query mixin.

    Architecture Note
    -----------------
    Implements HTTP transport path for profile queries. Uses ``_http_query()``
    for consistent domain/response conversion.

    See ``codeintel.serving.domain_models`` for the full architecture contract.
    """

    def get_function_profile(self, *, goid_h128: int) -> dm.FunctionProfileResult:
        return self._http_query(
            HttpQuerySpec(
                name="get_function_profile",
                path="/profiles/function",
                params={"goid_h128": goid_h128},
                response_type=FunctionProfileResponse,
                domain_type=dm.FunctionProfileResult,
            )
        )

    def get_file_profile(self, *, rel_path: str) -> dm.FileProfileResult:
        return self._http_query(
            HttpQuerySpec(
                name="get_file_profile",
                path="/profiles/file",
                params={"rel_path": rel_path},
                response_type=FileProfileResponse,
                domain_type=dm.FileProfileResult,
            )
        )

    def get_module_profile(self, *, module: str) -> dm.ModuleProfileResult:
        return self._http_query(
            HttpQuerySpec(
                name="get_module_profile",
                path="/profiles/module",
                params={"module": module},
                response_type=ModuleProfileResponse,
                domain_type=dm.ModuleProfileResult,
            )
        )

    def get_function_architecture(self, *, goid_h128: int) -> dm.FunctionArchitectureResult:
        return self._http_query(
            HttpQuerySpec(
                name="get_function_architecture",
                path="/architecture/function",
                params={"goid_h128": goid_h128},
                response_type=FunctionArchitectureResponse,
                domain_type=dm.FunctionArchitectureResult,
            )
        )

    def get_module_architecture(self, *, module: str) -> dm.ModuleArchitectureResult:
        return self._http_query(
            HttpQuerySpec(
                name="get_module_architecture",
                path="/architecture/module",
                params={"module": module},
                response_type=ModuleArchitectureResponse,
                domain_type=dm.ModuleArchitectureResult,
            )
        )


__all__ = ["_HttpProfileQueryMixin", "_ProfileQueryDelegates"]
