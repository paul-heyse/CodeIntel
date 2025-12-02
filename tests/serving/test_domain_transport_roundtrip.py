"""Roundtrip conversions between domain and MCP transport models."""

from __future__ import annotations

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import (
    CallGraphNeighborsResponse,
    FileProfileResponse,
    FunctionProfileResponse,
    FunctionSummaryResponse,
    GraphNeighborhoodResponse,
    HighRiskFunctionsResponse,
    ImportBoundaryResponse,
    ModuleProfileResponse,
    TestsForFunctionResponse,
)

REQUESTED_LIMIT = 10
APPLIED_LIMIT = 5
OUTGOING_CALLEE = 2
INCOMING_CALLER = 3


def _expect(*, condition: bool, message: str) -> None:
    """Raise an assertion error when condition is false.

    Raises
    ------
    AssertionError
        When the condition evaluates to False.
    """
    if not condition:
        raise AssertionError(message)


def _meta() -> dm.ResponseMeta:
    """Build standard metadata used across roundtrip tests.

    Returns
    -------
    dm.ResponseMeta
        Metadata with applied/requested limits and a sample message.
    """
    return dm.ResponseMeta(
        requested_limit=REQUESTED_LIMIT,
        applied_limit=APPLIED_LIMIT,
        truncated=True,
        messages=[dm.Message(code="test", severity="info", detail="hi")],
    )


def test_function_summary_roundtrip() -> None:
    """Ensure function summaries survive domain ↔ transport conversion."""
    meta = _meta()
    domain = dm.FunctionSummaryResult(
        found=True,
        summary={"urn": "urn:codeintel:test", "goid_h128": 123},
        meta=meta,
    )

    transport = FunctionSummaryResponse.from_domain(domain)
    back = transport.to_domain()

    _expect(condition=back == domain, message="FunctionSummaryResult should remain stable")
    _expect(
        condition=back.meta.truncated is True,
        message="Metadata should retain truncation flag",
    )
    _expect(
        condition=back.meta.messages[0].code == "test",
        message="Message code should survive roundtrip",
    )


def test_high_risk_functions_roundtrip() -> None:
    """Ensure high-risk listings preserve truncation and values."""
    domain = dm.HighRiskFunctionsResult(
        functions=[{"goid_h128": 1, "qualname": "f", "rel_path": "a.py", "risk_score": 0.9}],
        truncated=True,
        meta=_meta(),
    )

    transport = HighRiskFunctionsResponse.from_domain(domain)
    back = transport.to_domain()

    _expect(
        condition=back.functions[0]["qualname"] == "f",
        message="Function qualifier should remain",
    )
    _expect(condition=back.truncated is True, message="Truncation marker should remain")
    _expect(
        condition=back.meta.applied_limit == APPLIED_LIMIT,
        message="Applied limit should survive",
    )


def test_tests_for_function_roundtrip() -> None:
    """Ensure tests-for-function payloads preserve content."""
    domain = dm.TestsForFunctionResult(
        tests=[{"urn": "urn:codeintel:test", "rel_path": "tests/test_example.py"}],
        meta=_meta(),
    )

    transport = TestsForFunctionResponse.from_domain(domain)
    back = transport.to_domain()

    _expect(
        condition=back.tests[0]["rel_path"] == "tests/test_example.py",
        message="Test path should remain stable",
    )
    _expect(
        condition=back.meta.requested_limit == REQUESTED_LIMIT,
        message="Requested limit should remain attached",
    )


def test_callgraph_neighbors_roundtrip() -> None:
    """Ensure call graph neighbor payloads preserve edges."""
    domain = dm.CallGraphNeighbors(
        outgoing=[{"caller_goid_h128": 1, "callee_goid_h128": 2}],
        incoming=[{"caller_goid_h128": INCOMING_CALLER, "callee_goid_h128": 1}],
        meta=_meta(),
    )

    transport = CallGraphNeighborsResponse.from_domain(domain)
    back = transport.to_domain()

    _expect(
        condition=back.outgoing[0]["callee_goid_h128"] == OUTGOING_CALLEE,
        message="Outgoing callee should remain",
    )
    _expect(
        condition=back.incoming[0]["caller_goid_h128"] == INCOMING_CALLER,
        message="Incoming caller should remain",
    )


def test_graph_neighborhood_roundtrip() -> None:
    """Ensure graph neighborhoods retain nodes, edges, and metadata."""
    meta = dm.ResponseMeta(truncated=True)
    domain = dm.GraphNeighborhood(
        nodes=[{"function_goid_h128": 1}],
        edges=[{"caller_goid_h128": 1, "callee_goid_h128": 2}],
        meta=meta,
    )

    transport = GraphNeighborhoodResponse.from_domain(domain)
    back = transport.to_domain()

    _expect(condition=back.nodes == domain.nodes, message="Nodes should remain unchanged")
    _expect(condition=back.meta.truncated is True, message="Truncation flag should persist")


def test_import_boundary_roundtrip() -> None:
    """Ensure import boundary payloads retain nodes and edges."""
    domain = dm.ImportBoundary(
        nodes=[{"id": "a"}, {"id": "b"}],
        edges=[{"source": "a", "target": "b", "weight": 1.0}],
        meta=_meta(),
    )

    transport = ImportBoundaryResponse.from_domain(domain)
    back = transport.to_domain()

    _expect(
        condition=back.nodes[0]["id"] == "a",
        message="Node identifiers should survive roundtrip",
    )
    _expect(
        condition=back.edges[0]["target"] == "b",
        message="Edge targets should survive roundtrip",
    )


def test_profile_roundtrip_with_empty_profiles() -> None:
    """Ensure empty profile payloads retain metadata and flags."""
    meta = _meta()
    function_profile = dm.FunctionProfileResult(found=False, profile=None, meta=meta)
    file_profile = dm.FileProfileResult(found=False, profile=None, meta=meta)
    module_profile = dm.ModuleProfileResult(found=False, profile=None, meta=meta)

    _expect(
        condition=FunctionProfileResponse.from_domain(function_profile).to_domain()
        == function_profile,
        message="Function profiles should roundtrip",
    )
    _expect(
        condition=FileProfileResponse.from_domain(file_profile).to_domain() == file_profile,
        message="File profiles should roundtrip",
    )
    _expect(
        condition=ModuleProfileResponse.from_domain(module_profile).to_domain() == module_profile,
        message="Module profiles should roundtrip",
    )
