"""Unit tests for semantic role classification signals."""

from __future__ import annotations

import pytest

from codeintel.analytics.semantic_roles import FunctionContext, classify_function_role


class _FunctionContextBuilder:
    def __init__(self) -> None:
        self.goid = 0
        self.rel_path = "pkg/api.py"
        self.qualname = "pkg.api.fn"
        self.decorators: list[str] = []
        self.effects: dict[str, object] = {}
        self.contracts: dict[str, object] = {}
        self.module_tags: list[str] = []
        self.module_name = "pkg.api"
        self.graph: dict[str, object] = {}
        self.loc = 10

    def with_decorators(self, *decorators: str) -> _FunctionContextBuilder:
        self.decorators = list(decorators)
        return self

    def with_module_tags(self, *tags: str) -> _FunctionContextBuilder:
        self.module_tags = list(tags)
        return self

    def with_effects(self, effects: dict[str, object]) -> _FunctionContextBuilder:
        self.effects = dict(effects)
        return self

    def with_contracts(self, contracts: dict[str, object]) -> _FunctionContextBuilder:
        self.contracts = dict(contracts)
        return self

    def with_location(self, loc: int) -> _FunctionContextBuilder:
        self.loc = loc
        return self

    def with_graph(self, graph: dict[str, object]) -> _FunctionContextBuilder:
        self.graph = dict(graph)
        return self

    def build(self) -> FunctionContext:
        return FunctionContext(
            goid=self.goid,
            rel_path=self.rel_path,
            qualname=self.qualname,
            decorators=self.decorators,
            effects=self.effects,
            contracts=self.contracts,
            module_tags=self.module_tags,
            module_name=self.module_name,
            graph=self.graph,
            loc=self.loc,
        )


def _make_context(**overrides: object) -> FunctionContext:
    builder = _FunctionContextBuilder()
    _apply_iterable_override(builder, overrides, "decorators", builder.with_decorators)
    _apply_iterable_override(builder, overrides, "module_tags", builder.with_module_tags)
    _apply_mapping_override(builder, overrides, "effects", builder.with_effects)
    _apply_mapping_override(builder, overrides, "contracts", builder.with_contracts)
    _apply_mapping_override(builder, overrides, "graph", builder.with_graph)
    if "loc" in overrides:
        builder.with_location(int(overrides.pop("loc")))
    builder.rel_path = str(overrides.pop("rel_path", builder.rel_path))
    builder.qualname = str(overrides.pop("qualname", builder.qualname))
    builder.module_name = str(overrides.pop("module_name", builder.module_name))
    builder.goid = int(overrides.pop("goid", builder.goid))
    return builder.build()


def _apply_iterable_override(
    overrides: dict[str, object],
    key: str,
    applier: callable[..., object],
) -> None:
    if key not in overrides:
        return
    items = overrides.pop(key)
    if isinstance(items, (list, tuple)):
        applier(*tuple(str(item) for item in items))


def _apply_mapping_override(
    overrides: dict[str, object],
    key: str,
    applier: callable[[dict[str, object]], object],
) -> None:
    if key not in overrides:
        return
    mapping = overrides.pop(key)
    if isinstance(mapping, dict):
        applier(mapping)


def test_fastapi_role_detected() -> None:
    """FastAPI decorators should yield api_handler role."""
    ctx = _make_context(decorators=["router.get('/hello')"])
    role, confidence, framework, sources = classify_function_role(ctx)

    _expect(role == "api_handler", "role not api_handler for fastapi")
    _expect(framework == "fastapi", f"framework unexpected: {framework}")
    _expect(confidence > 0.0, "confidence should be positive")
    _expect("decorator:router.get('/hello')" in _signals(sources), "missing fastapi signal")


def test_flask_role_detected() -> None:
    """Flask route decorators should yield api_handler role."""
    ctx = _make_context(decorators=["app.route('/hi')"])
    role, _, _, sources = classify_function_role(ctx)

    _expect(role == "api_handler", "role not api_handler for flask")
    _expect(
        any(sig.startswith("decorator:app.route") for sig in _signals(sources)),
        "missing flask decorator signal",
    )


def test_typer_cli_detected() -> None:
    """Typer commands should be classified as cli_command."""
    ctx = _make_context(
        rel_path="cli/app.py",
        module_name="cli.app",
        decorators=["typer.command()"],
    )
    role, _, framework, sources = classify_function_role(ctx)

    _expect(role == "cli_command", "role not cli_command for typer")
    _expect(framework == "typer", f"framework unexpected: {framework}")
    _expect(
        any("decorator:typer.command" in sig for sig in _signals(sources)),
        "missing typer decorator signal",
    )


def test_pytest_fixture_vs_test_role() -> None:
    """Distinguish pytest fixtures from plain test functions."""
    fixture_ctx = _make_context(
        rel_path="tests/util.py",
        module_name="tests.util",
        qualname="tests.util.fixture_helper",
        decorators=["pytest.fixture"],
    )
    role, _, _, sources = classify_function_role(fixture_ctx)
    _expect(role == "test_helper", "fixture should be test_helper")
    _expect("decorator:pytest.fixture" in _signals(sources), "missing fixture signal")

    test_ctx = _make_context(
        rel_path="tests/test_app.py",
        module_name="tests.test_app",
        qualname="tests.test_app.test_hello",
    )
    role_test, _, _, sources_test = classify_function_role(test_ctx)
    _expect(role_test == "test", "test function should be test role")
    _expect("path:tests" in _signals(sources_test), "missing tests path signal")


def test_service_tag_and_graph_signal() -> None:
    """Service tags and fan-in/out metrics should tilt toward service role."""
    ctx = _make_context(
        module_tags=["service"],
        module_name="pkg.service",
        graph={"call_fan_in": 10, "call_fan_out": 8},
    )
    role, _, _, sources = classify_function_role(ctx)

    _expect(role == "service", "service role not detected")
    _expect(
        {"tag:service", "graph:fan_in", "graph:fan_out"}.issubset(set(_signals(sources))),
        "missing service signals",
    )


def _expect(condition: object, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _signals(source_payload: dict[str, object]) -> list[str]:
    signals = source_payload.get("signals", [])
    if isinstance(signals, list):
        return [str(sig) for sig in signals]
    return []
