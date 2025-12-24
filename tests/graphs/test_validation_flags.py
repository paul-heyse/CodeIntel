"""Validation feature flag behavior tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.config.primitives import GraphFeatureFlags
from codeintel.graphs.validation import apply_severity_overrides, resolve_validation_options
from tests._helpers.assertions import expect_true
from tests._helpers.factories import make_graph_runtime_options
from tests._helpers.fakes.graph_runtime import runtime_with_graphs
from tests._helpers.fixtures.graphs import empty_digraph, empty_graph

if TYPE_CHECKING:
    from codeintel.graphs.runtime import GraphRuntimeOptions
    from tests._helpers.fakes.graph_contexts import GraphTestEnv


def _runtime_options(
    env: GraphTestEnv,
    *,
    strict: bool,
) -> GraphRuntimeOptions:
    options, _ = runtime_with_graphs(
        env.gateway,
        env.snapshot,
        call_graph=empty_digraph(),
        import_graph=empty_digraph(),
        symbol_module_graph=empty_graph(),
        symbol_function_graph=empty_graph(),
        config_graph=empty_graph(),
        test_function_graph=empty_graph(),
    )
    return make_graph_runtime_options(
        snapshot=env.snapshot,
        features=GraphFeatureFlags(validation_strict=strict),
        engine=options.engine,
    )


def test_validation_strict_escalates_findings(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Strict validation should convert warnings to errors and hard-fail semantics."""
    runtime_opts = _runtime_options(graph_executor_env, strict=True)
    opts = resolve_validation_options(runtime=runtime_opts, options=None)
    findings: list[dict[str, object]] = [
        {"check_name": "missing_goids", "severity": "warning", "detail": "stub"},
    ]
    normalized = apply_severity_overrides(findings, opts.severity_overrides)
    severities = {finding["severity"] for finding in normalized}
    expect_true(severities == {"error"}, message="Strict validation should escalate severities")
    expect_true(opts.hard_fail, message="Strict validation should enable hard_fail")


def test_validation_non_strict_allows_warnings(
    graph_executor_env: GraphTestEnv,
) -> None:
    """Non-strict validation should leave warning findings unchanged."""
    runtime_opts = _runtime_options(graph_executor_env, strict=False)
    opts = resolve_validation_options(runtime=runtime_opts, options=None)
    findings: list[dict[str, object]] = [
        {"check_name": "missing_goids", "severity": "warning", "detail": "stub"},
    ]
    normalized = apply_severity_overrides(findings, opts.severity_overrides)
    severities = {finding["severity"] for finding in normalized}
    expect_true(
        severities == {"warning"}, message="Non-strict validation should keep warnings intact"
    )
    expect_true(not opts.hard_fail, message="Non-strict validation should avoid hard_fail")
