"""Validation feature flag behavior tests."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.config.primitives import GraphFeatureFlags, SnapshotRef
from codeintel.graphs.engine import GraphEngine
from codeintel.graphs.validation import apply_severity_overrides, resolve_validation_options
from codeintel.storage.gateway import StorageGateway


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


class _DummyEngine(GraphEngine):
    """Minimal stub satisfying GraphEngine protocol for strict validation tests."""

    def __init__(self, gateway: StorageGateway, snapshot: SnapshotRef) -> None:
        self.gateway: StorageGateway = gateway
        self._snapshot: SnapshotRef = snapshot
        self._empty_graph = nx.DiGraph()
        self._empty_undirected = nx.Graph()

    @property
    def use_gpu(self) -> bool:
        return False

    def call_graph(self) -> nx.DiGraph:
        return self._empty_graph

    def load_call_graph(self) -> nx.DiGraph:
        return self._empty_graph

    def import_graph(self) -> nx.DiGraph:
        return self._empty_graph

    def load_import_graph(self) -> nx.DiGraph:
        return self._empty_graph

    def symbol_module_graph(self) -> nx.Graph:
        return self._empty_undirected

    def load_symbol_module_graph(self) -> nx.Graph:
        return self._empty_undirected

    def symbol_function_graph(self) -> nx.Graph:
        return self._empty_undirected

    def load_symbol_function_graph(self) -> nx.Graph:
        return self._empty_undirected

    def config_module_bipartite(self) -> nx.Graph:
        return self._empty_undirected

    def load_config_module_bipartite(self) -> nx.Graph:
        return self._empty_undirected

    def test_function_bipartite(self) -> nx.Graph:
        return self._empty_undirected

    def load_test_function_bipartite(self) -> nx.Graph:
        return self._empty_undirected

    @property
    def snapshot(self) -> SnapshotRef:
        return self._snapshot


def _runtime_options(
    tmp_path: Path,
    gateway: StorageGateway,
    *,
    strict: bool,
) -> GraphRuntimeOptions:
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=tmp_path)
    return GraphRuntimeOptions(
        snapshot=snapshot,
        features=GraphFeatureFlags(validation_strict=strict),
        engine=_DummyEngine(gateway, snapshot),
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
