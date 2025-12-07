"""Validation feature flag behavior tests."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from codeintel.analytics.runtime import GraphRuntimeOptions
from codeintel.config.primitives import GraphFeatureFlags
from codeintel.graphs.validation import apply_severity_overrides, resolve_validation_options
from codeintel.storage.gateway import StorageGateway
from tests._helpers.factories import make_snapshot
from tests._helpers.graphs import GraphStubEngine


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def _runtime_options(
    tmp_path: Path,
    gateway: StorageGateway,
    *,
    strict: bool,
) -> GraphRuntimeOptions:
    snapshot = make_snapshot(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    return GraphRuntimeOptions(
        snapshot=snapshot,
        features=GraphFeatureFlags(validation_strict=strict),
        engine=GraphStubEngine(
            gateway=gateway,
            snapshot=snapshot,
            call_graph_obj=nx.DiGraph(),
            import_graph_obj=nx.DiGraph(),
            symbol_module_graph_obj=nx.Graph(),
            symbol_function_graph_obj=nx.Graph(),
            config_bipartite_obj=nx.Graph(),
            test_function_bipartite_obj=nx.Graph(),
        ),
    )


def test_validation_strict_escalates_findings(
    tmp_path: Path, fresh_gateway: StorageGateway
) -> None:
    """Strict validation should convert warnings to errors and hard-fail semantics."""
    runtime_opts = _runtime_options(tmp_path, fresh_gateway, strict=True)
    opts = resolve_validation_options(runtime=runtime_opts, options=None)
    findings: list[dict[str, object]] = [
        {"check_name": "missing_goids", "severity": "warning", "detail": "stub"},
    ]
    normalized = apply_severity_overrides(findings, opts.severity_overrides)
    severities = {finding["severity"] for finding in normalized}
    _expect(
        condition=severities == {"error"},
        detail="Strict validation should escalate severities",
    )
    _expect(condition=opts.hard_fail, detail="Strict validation should enable hard_fail")


def test_validation_non_strict_allows_warnings(
    tmp_path: Path, fresh_gateway: StorageGateway
) -> None:
    """Non-strict validation should leave warning findings unchanged."""
    runtime_opts = _runtime_options(tmp_path, fresh_gateway, strict=False)
    opts = resolve_validation_options(runtime=runtime_opts, options=None)
    findings: list[dict[str, object]] = [
        {"check_name": "missing_goids", "severity": "warning", "detail": "stub"},
    ]
    normalized = apply_severity_overrides(findings, opts.severity_overrides)
    severities = {finding["severity"] for finding in normalized}
    _expect(
        condition=severities == {"warning"},
        detail="Non-strict validation should keep warnings intact",
    )
    _expect(condition=not opts.hard_fail, detail="Non-strict validation should avoid hard_fail")
