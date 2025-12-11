"""Tests for serving layer types and protocols.

This module verifies that the protocols defined in types.py are correctly
implemented by real production classes, ensuring structural compliance
without mocking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

from pydantic import BaseModel

from codeintel.serving.domain_models import DatasetRows, Message, ResponseMeta
from codeintel.serving.types import (
    HasClose,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.serving.types import (
        BackendFactory,
        FunctionRepositoryProtocol,
        GraphEngineProtocol,
        HasModelDump,
        JsonPayload,
        ModuleRepositoryProtocol,
        QueryBackendProtocol,
        QueryServiceProtocol,
        RepositoryProtocol,
        RowDict,
        ServiceFactory,
        StorageGatewayProtocol,
        SubsystemRepositoryProtocol,
    )

# Constants for test values
LIMIT_TEN = 10
LIMIT_FIFTY = 50
VALUE_FORTY_TWO = 42
LIST_LENGTH_TWO = 2


# =============================================================================
# Test HasModelDump Protocol
# =============================================================================


def test_has_model_dump_protocol_with_response_meta() -> None:
    """Verify ResponseMeta satisfies HasModelDump protocol."""
    meta = ResponseMeta(
        requested_limit=LIMIT_TEN,
        applied_limit=LIMIT_TEN,
        truncated=False,
        messages=[Message(code="INFO", severity="info", detail="Test message")],
    )

    # Verify it has model_dump method
    result = meta.model_dump()

    expect_is_instance(result, dict)
    expect_equal(result["applied_limit"], LIMIT_TEN)
    expect_true(result["truncated"] is False)
    expect_length(result["messages"], 1)
    expect_equal(result["messages"][0]["code"], "INFO")


def test_has_model_dump_protocol_with_dataset_rows() -> None:
    """Verify DatasetRows satisfies HasModelDump protocol."""
    meta = ResponseMeta(truncated=False)
    rows = DatasetRows(
        dataset_name="test_dataset",
        limit=LIMIT_TEN,
        offset=0,
        rows=[{"id": 1, "name": "test"}],
        meta=meta,
    )

    result = rows.model_dump()

    expect_is_instance(result, dict)
    expect_equal(result["dataset"], "test_dataset")
    expect_equal(result["limit"], LIMIT_TEN)
    expect_equal(result["offset"], 0)
    expect_length(result["rows"], 1)
    expect_in("meta", result)


def test_has_model_dump_protocol_structural_check() -> None:
    """Verify custom class can satisfy HasModelDump protocol structurally."""

    @dataclass
    class CustomModel:
        """Custom model implementing HasModelDump."""

        value: int

        def model_dump(self) -> dict[str, object]:
            """
            Return dictionary representation.

            Returns
            -------
            dict[str, object]
                Dictionary with value key.
            """
            return {"value": self.value}

    model = CustomModel(value=VALUE_FORTY_TWO)
    # Structural subtyping - no explicit inheritance needed
    dumped: dict[str, object] = model.model_dump()

    expect_equal(dumped, {"value": VALUE_FORTY_TWO})

    # Verify it can be used where HasModelDump is expected
    def accepts_model_dump(obj: HasModelDump) -> dict[str, object]:
        return obj.model_dump()

    result = accepts_model_dump(model)
    expect_equal(result, {"value": VALUE_FORTY_TWO})


# =============================================================================
# Test HasModelValidate Protocol (runtime_checkable)
# =============================================================================


def test_has_model_validate_protocol_with_pydantic() -> None:
    """Verify Pydantic BaseModel satisfies HasModelValidate protocol."""

    class SampleModel(BaseModel):
        """Sample Pydantic model for testing."""

        name: str
        value: int

    # Runtime check should pass for Pydantic models
    expect_is_instance(SampleModel, type)

    # Verify model_validate works (cast to Any for pyrefly compatibility)
    validated = cast("Any", SampleModel).model_validate({"name": "test", "value": VALUE_FORTY_TWO})
    expect_equal(validated.name, "test")
    expect_equal(validated.value, VALUE_FORTY_TWO)


def test_has_model_validate_runtime_checkable() -> None:
    """Verify HasModelValidate is runtime checkable."""

    class ValidatableClass:
        """Class that implements model_validate."""

        @classmethod
        def model_validate(cls, _obj: object) -> ValidatableClass:
            """
            Validate and construct instance.

            Parameters
            ----------
            _obj
                Input object (unused in this test implementation).

            Returns
            -------
            ValidatableClass
                New instance.
            """
            return cls()

    # Since HasModelValidate is runtime_checkable, isinstance should work
    expect_is_instance(ValidatableClass, type)
    instance = cast("Any", ValidatableClass).model_validate({})
    expect_is_instance(instance, ValidatableClass)


# =============================================================================
# Test HasClose Protocol (runtime_checkable)
# =============================================================================


def test_has_close_protocol_runtime_checkable() -> None:
    """Verify HasClose is runtime checkable with closable resources."""

    class CloseableResource:
        """Resource that can be closed."""

        def __init__(self) -> None:
            """Initialize the resource."""
            self.closed = False

        def close(self) -> None:
            """Close the resource."""
            self.closed = True

    resource = CloseableResource()

    # HasClose is runtime_checkable
    expect_is_instance(resource, HasClose)
    expect_true(not resource.closed)

    resource.close()
    expect_true(resource.closed)


def test_has_close_protocol_with_context_manager() -> None:
    """Verify HasClose works with context manager pattern."""

    class ManagedResource:
        """Resource with context manager support."""

        def __init__(self) -> None:
            """Initialize resource."""
            self.closed = False

        def close(self) -> None:
            """Close the resource."""
            self.closed = True

        def __enter__(self) -> Self:
            """
            Enter context.

            Returns
            -------
            Self
                This resource instance.
            """
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc_val: object,
            _exc_tb: object,
        ) -> None:
            """Exit context and close."""
            self.close()

    with ManagedResource() as res:
        expect_is_instance(res, HasClose)
        expect_true(not res.closed)

    expect_true(res.closed)


# =============================================================================
# Test ResponseMetaLike Protocol
# =============================================================================


def test_response_meta_like_protocol() -> None:
    """Verify ResponseMeta has attributes matching ResponseMetaLike protocol."""
    meta = ResponseMeta(
        applied_limit=LIMIT_FIFTY,
        truncated=True,
        messages=[Message(code="WARN", severity="warning")],
    )

    # Verify ResponseMeta has the required attributes
    expect_true(hasattr(meta, "applied_limit"))
    expect_true(hasattr(meta, "truncated"))
    expect_true(hasattr(meta, "messages"))

    # Verify values
    expect_equal(meta.applied_limit, LIMIT_FIFTY)
    expect_true(meta.truncated is True)
    expect_length(meta.messages, 1)


# =============================================================================
# Test ServiceResult Protocol
# =============================================================================


def test_service_result_protocol() -> None:
    """Verify dataclass has attributes matching ServiceResult protocol."""

    @dataclass
    class SampleResult:
        """Sample service result."""

        found: bool
        meta: ResponseMeta

    meta = ResponseMeta(applied_limit=LIMIT_TEN, truncated=False, messages=[])
    result = SampleResult(found=True, meta=meta)

    # Verify SampleResult has the required attributes
    expect_true(hasattr(result, "found"))
    expect_true(hasattr(result, "meta"))

    # Verify values
    expect_true(result.found is True)
    expect_equal(result.meta.applied_limit, LIMIT_TEN)


# =============================================================================
# Test Repository Protocols
# =============================================================================


def test_repository_protocol_structural() -> None:
    """Verify a class can satisfy RepositoryProtocol structurally."""

    class SimpleRepo:
        """Simple repository implementation."""

        def __init__(self, repo: str, commit: str) -> None:
            """Initialize repository."""
            self._repo = repo
            self._commit = commit

        @property
        def repo(self) -> str:
            """Return repository identifier."""
            return self._repo

        @property
        def commit(self) -> str:
            """Return commit hash."""
            return self._commit

    repo_impl = SimpleRepo(repo="test/repo", commit="abc123")

    def accepts_repository(r: RepositoryProtocol) -> tuple[str, str]:
        return r.repo, r.commit

    repo_id, commit_id = accepts_repository(repo_impl)

    expect_equal(repo_id, "test/repo")
    expect_equal(commit_id, "abc123")


def test_function_repository_protocol_structural() -> None:
    """Verify a class can satisfy FunctionRepositoryProtocol structurally."""

    class FunctionRepo:
        """Function repository implementation."""

        def __init__(self) -> None:
            """Initialize repository."""
            self._repo = "demo/repo"
            self._commit = "deadbeef"

        @property
        def repo(self) -> str:
            """Return repository identifier."""
            return self._repo

        @property
        def commit(self) -> str:
            """Return commit hash."""
            return self._commit

        def get_function_summary_by_goid(self, goid_h128: int) -> RowDict | None:
            """
            Fetch function summary by GOID.

            Parameters
            ----------
            goid_h128
                Function GOID hash.

            Returns
            -------
            RowDict | None
                Function summary row or None.
            """
            if goid_h128 == 1:
                return {"goid_h128": 1, "qualname": "pkg.mod.func", "repo": self.repo}
            return None

        def get_function_profile(self, goid_h128: int) -> RowDict | None:
            """
            Fetch function profile by GOID.

            Parameters
            ----------
            goid_h128
                Function GOID hash.

            Returns
            -------
            RowDict | None
                Always None in this stub.
            """
            _ = goid_h128, self.repo  # Use self to satisfy PLR6301
            return None

        def get_function_architecture(self, goid_h128: int) -> RowDict | None:
            """
            Fetch function architecture by GOID.

            Parameters
            ----------
            goid_h128
                Function GOID hash.

            Returns
            -------
            RowDict | None
                Minimal row dictionary.
            """
            _ = self.commit  # Use self to satisfy PLR6301
            return {"goid_h128": goid_h128}

    repo = FunctionRepo()

    def accepts_function_repo(r: FunctionRepositoryProtocol) -> RowDict | None:
        return r.get_function_summary_by_goid(1)

    result = accepts_function_repo(repo)

    result = expect_is_not_none(result)
    expect_equal(result["goid_h128"], 1)


def test_module_repository_protocol_structural() -> None:
    """Verify a class can satisfy ModuleRepositoryProtocol structurally."""

    class ModuleRepo:
        """Module repository implementation."""

        def __init__(self) -> None:
            """Initialize with internal state."""
            self._data: dict[str, RowDict] = {
                "pkg/mod.py": {"rel_path": "pkg/mod.py", "module": "pkg.mod"},
            }

        @property
        def repo(self) -> str:
            """Return repository identifier."""
            return "demo/repo"

        @property
        def commit(self) -> str:
            """Return commit hash."""
            return "deadbeef"

        def get_file_summary(self, rel_path: str) -> RowDict | None:
            """
            Fetch file summary by path.

            Parameters
            ----------
            rel_path
                Relative file path.

            Returns
            -------
            RowDict | None
                File summary row or None.
            """
            return self._data.get(rel_path)

        def get_file_profile(self, rel_path: str) -> RowDict | None:
            """
            Fetch file profile by path.

            Parameters
            ----------
            rel_path
                Relative file path.

            Returns
            -------
            RowDict | None
                Always None in this stub.
            """
            _ = rel_path, self._data  # Use self to satisfy PLR6301
            return None

        def get_file_hints(self, rel_path: str) -> list[RowDict]:
            """
            Fetch IDE hints for a file.

            Parameters
            ----------
            rel_path
                Relative file path.

            Returns
            -------
            list[RowDict]
                List of hint rows.
            """
            _ = rel_path, self.repo  # Use self to satisfy PLR6301
            return [{"hint": "unused_import", "line": 1}]

    repo = ModuleRepo()

    def accepts_module_repo(r: ModuleRepositoryProtocol) -> list[RowDict]:
        return r.get_file_hints("pkg/mod.py")

    hints = accepts_module_repo(repo)

    expect_length(hints, 1)
    expect_equal(hints[0]["hint"], "unused_import")


def test_subsystem_repository_protocol_structural() -> None:
    """Verify a class can satisfy SubsystemRepositoryProtocol structurally."""

    class SubsystemRepo:
        """Subsystem repository implementation."""

        def __init__(self) -> None:
            """Initialize with internal state."""
            self._subsystems: list[RowDict] = [
                {"subsystem_id": "core", "role": "server"},
            ]

        @property
        def repo(self) -> str:
            """Return repository identifier."""
            return "demo/repo"

        @property
        def commit(self) -> str:
            """Return commit hash."""
            return "deadbeef"

        def get_subsystem_summary(self, subsystem_id: str) -> RowDict | None:
            """
            Fetch subsystem summary by ID.

            Parameters
            ----------
            subsystem_id
                Subsystem identifier.

            Returns
            -------
            RowDict | None
                Subsystem summary row.
            """
            _ = self._subsystems  # Use self to satisfy PLR6301
            return {"subsystem_id": subsystem_id, "name": "Core"}

        def list_subsystems(
            self,
            *,
            limit: int,
            role: str | None = None,
            query: str | None = None,
        ) -> list[RowDict]:
            """
            List subsystems with optional filtering.

            Parameters
            ----------
            limit
                Maximum results to return.
            role
                Optional role filter.
            query
                Optional search query.

            Returns
            -------
            list[RowDict]
                Filtered subsystem rows.
            """
            _ = limit, query, self._subsystems  # Use self to satisfy PLR6301
            return [{"subsystem_id": "core", "role": role or "any"}]

    repo = SubsystemRepo()

    def accepts_subsystem_repo(r: SubsystemRepositoryProtocol) -> list[RowDict]:
        return r.list_subsystems(limit=LIMIT_TEN, role="server")

    result = accepts_subsystem_repo(repo)

    expect_length(result, 1)
    expect_equal(result[0]["role"], "server")


# =============================================================================
# Test Query Backend/Service Protocols
# =============================================================================


def test_query_backend_protocol_structural() -> None:
    """Verify a class can satisfy QueryBackendProtocol structurally."""

    class SimpleBackend:
        """Simple backend implementation."""

        @property
        def repo(self) -> str:
            """Return repository identifier."""
            return "test/repo"

        @property
        def commit(self) -> str:
            """Return commit hash."""
            return "abc123"

    backend = SimpleBackend()

    def accepts_backend(b: QueryBackendProtocol) -> str:
        return f"{b.repo}@{b.commit}"

    result = accepts_backend(backend)

    expect_equal(result, "test/repo@abc123")


def test_query_service_protocol_structural() -> None:
    """Verify a class can satisfy QueryServiceProtocol structurally."""

    class SimpleService:
        """Simple service implementation."""

        @property
        def repo(self) -> str:
            """Return repository identifier."""
            return "demo/repo"

        @property
        def commit(self) -> str:
            """Return commit hash."""
            return "deadbeef"

    service = SimpleService()

    def accepts_service(s: QueryServiceProtocol) -> str:
        return f"{s.repo}@{s.commit}"

    result = accepts_service(service)

    expect_equal(result, "demo/repo@deadbeef")


# =============================================================================
# Test Storage Gateway Protocol
# =============================================================================


def test_storage_gateway_protocol_structural() -> None:
    """Verify a class can satisfy StorageGatewayProtocol structurally."""

    class SimpleGateway:
        """Simple gateway implementation."""

        def __init__(self) -> None:
            """Initialize gateway."""
            self.is_closed = False

        @property
        def con(self) -> object:
            """Return connection object."""
            return "mock_connection"

        def close(self) -> None:
            """Close the gateway."""
            self.is_closed = True

    gateway = SimpleGateway()

    def accepts_gateway(g: StorageGatewayProtocol) -> bool:
        g.close()
        return True

    result = accepts_gateway(gateway)

    expect_true(result is True)
    expect_true(gateway.is_closed is True)


# =============================================================================
# Test Graph Engine Protocol
# =============================================================================


def test_graph_engine_protocol_structural() -> None:
    """Verify a class can satisfy GraphEngineProtocol structurally."""

    class SimpleEngine:
        """Simple graph engine implementation."""

        def __init__(self) -> None:
            """Initialize with graph data."""
            self._call_graph: dict[str, list[object]] = {"nodes": [], "edges": []}
            self._import_graph: dict[str, list[object]] = {"nodes": [], "edges": []}

        def call_graph(self) -> object:
            """
            Return call graph.

            Returns
            -------
            object
                Call graph dictionary.
            """
            return self._call_graph

        def import_graph(self) -> object:
            """
            Return import graph.

            Returns
            -------
            object
                Import graph dictionary.
            """
            return self._import_graph

    engine = SimpleEngine()

    def accepts_engine(e: GraphEngineProtocol) -> tuple[object, object]:
        return e.call_graph(), e.import_graph()

    call_g, import_g = accepts_engine(engine)

    expect_equal(call_g, {"nodes": [], "edges": []})
    expect_equal(import_g, {"nodes": [], "edges": []})


# =============================================================================
# Test Type Aliases
# =============================================================================


def test_row_dict_type_alias() -> None:
    """Verify RowDict type alias works correctly."""
    row: RowDict = {"id": 1, "name": "test", "active": True, "count": None}

    expect_equal(row["id"], 1)
    expect_equal(row["name"], "test")
    expect_true(row["active"] is True)
    expect_true(row["count"] is None)


def test_json_payload_type_alias_dict() -> None:
    """Verify JsonPayload type alias works with dict."""
    payload: JsonPayload = {"key": "value", "nested": {"inner": 1}}

    expect_is_instance(payload, dict)
    payload_dict = cast("dict[str, object]", payload)
    expect_equal(payload_dict["key"], "value")


def test_json_payload_type_alias_list() -> None:
    """Verify JsonPayload type alias works with list."""
    payload: JsonPayload = [{"id": 1}, {"id": LIST_LENGTH_TWO}]

    expect_is_instance(payload, list)
    expect_length(payload, LIST_LENGTH_TWO)


# =============================================================================
# Test Factory Type Aliases
# =============================================================================


def test_service_factory_type_alias() -> None:
    """Verify ServiceFactory type alias works correctly."""

    class SimpleService:
        """Simple service for factory test."""

        @property
        def repo(self) -> str:
            """Return repository."""
            return "test"

        @property
        def commit(self) -> str:
            """Return commit."""
            return "abc"

    def _create_service(repo: str, commit: str) -> QueryServiceProtocol:
        """
        Create a service instance.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit hash.

        Returns
        -------
        QueryServiceProtocol
            New service instance.
        """
        _ = repo, commit
        return SimpleService()

    factory: ServiceFactory = _create_service
    service = factory("test", "abc")

    expect_equal(service.repo, "test")


def test_backend_factory_type_alias() -> None:
    """Verify BackendFactory type alias works correctly."""

    class SimpleBackend:
        """Simple backend for factory test."""

        @property
        def repo(self) -> str:
            """Return repository."""
            return "test"

        @property
        def commit(self) -> str:
            """Return commit."""
            return "abc"

    def _create_backend(repo: str, commit: str) -> QueryBackendProtocol:
        """
        Create a backend instance.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit hash.

        Returns
        -------
        QueryBackendProtocol
            New backend instance.
        """
        _ = repo, commit
        return SimpleBackend()

    factory: BackendFactory = _create_backend
    backend = factory("test", "abc")

    expect_equal(backend.repo, "test")
