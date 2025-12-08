"""Unit tests for semantic role classification signals."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from codeintel.analytics.semantic_roles import FunctionContext, classify_function_role
from tests._helpers.assertions.expectation_assertions import expect_true


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
        self.graph: dict[str, int] = {}
        self.loc = 10

    def with_decorators(self, *decorators: str) -> _FunctionContextBuilder:
        self.decorators = list(decorators)
        return self

    def with_module_tags(self, *tags: str) -> _FunctionContextBuilder:
        self.module_tags = list(tags)
        return self

    def with_effects(self, effects: Mapping[str, object]) -> _FunctionContextBuilder:
        self.effects = dict(effects)
        return self

    def with_contracts(self, contracts: Mapping[str, object]) -> _FunctionContextBuilder:
        self.contracts = dict(contracts)
        return self

    def with_location(self, loc: int) -> _FunctionContextBuilder:
        self.loc = loc
        return self

    def with_graph(self, graph: Mapping[str, object]) -> _FunctionContextBuilder:
        self.graph = {str(k): int(v) for k, v in graph.items() if isinstance(v, (int, float, str))}
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
    _apply_iterable_override(overrides, "decorators", builder.with_decorators)
    _apply_iterable_override(overrides, "module_tags", builder.with_module_tags)
    _apply_mapping_override(overrides, "effects", builder.with_effects)
    _apply_mapping_override(overrides, "contracts", builder.with_contracts)
    _apply_mapping_override(overrides, "graph", builder.with_graph)
    if "loc" in overrides:
        loc_value = overrides.pop("loc")
        if isinstance(loc_value, (int, str)):
            builder.with_location(int(loc_value))
    builder.rel_path = str(overrides.pop("rel_path", builder.rel_path))
    builder.qualname = str(overrides.pop("qualname", builder.qualname))
    builder.module_name = str(overrides.pop("module_name", builder.module_name))
    goid_value = overrides.pop("goid", builder.goid)
    if isinstance(goid_value, (int, str)):
        builder.goid = int(goid_value)
    return builder.build()


def _apply_iterable_override(
    overrides: dict[str, object],
    key: str,
    applier: Callable[..., object],
) -> None:
    if key not in overrides:
        return
    items = overrides.pop(key)
    if isinstance(items, (list, tuple)):
        applier(*tuple(str(item) for item in items))


def _apply_mapping_override(
    overrides: dict[str, object],
    key: str,
    applier: Callable[[Mapping[str, object]], object],
) -> None:
    if key not in overrides:
        return
    mapping = overrides.pop(key)
    if isinstance(mapping, dict):
        if key == "graph":
            normalized: dict[str, object] = {
                str(k): int(v) for k, v in mapping.items() if isinstance(v, (int, float, str))
            }
            applier(normalized)
            return
        applier({str(k): v for k, v in mapping.items()})


def test_fastapi_role_detected() -> None:
    """FastAPI decorators should yield api_handler role."""
    ctx = _make_context(decorators=["router.get('/hello')"])
    role, confidence, framework, sources = classify_function_role(ctx)

    expect_true(role == "api_handler", message="role not api_handler for fastapi")
    expect_true(framework == "fastapi", message=f"framework unexpected: {framework}")
    expect_true(confidence > 0.0, message="confidence should be positive")
    expect_true(
        "decorator:router.get('/hello')" in _signals(sources),
        message="missing fastapi signal",
    )


def test_flask_role_detected() -> None:
    """Flask route decorators should yield api_handler role."""
    ctx = _make_context(decorators=["app.route('/hi')"])
    role, _, _, sources = classify_function_role(ctx)

    expect_true(role == "api_handler", message="role not api_handler for flask")
    expect_true(
        any(sig.startswith("decorator:app.route") for sig in _signals(sources)),
        message="missing flask decorator signal",
    )


def test_typer_cli_detected() -> None:
    """Typer commands should be classified as cli_command."""
    ctx = _make_context(
        rel_path="cli/app.py",
        module_name="cli.app",
        decorators=["typer.command()"],
    )
    role, _, framework, sources = classify_function_role(ctx)

    expect_true(role == "cli_command", message="role not cli_command for typer")
    expect_true(framework == "typer", message=f"framework unexpected: {framework}")
    expect_true(
        any("decorator:typer.command" in sig for sig in _signals(sources)),
        message="missing typer decorator signal",
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
    expect_true(role == "test_helper", message="fixture should be test_helper")
    expect_true(
        "decorator:pytest.fixture" in _signals(sources),
        message="missing fixture signal",
    )

    test_ctx = _make_context(
        rel_path="tests/test_app.py",
        module_name="tests.test_app",
        qualname="tests.test_app.test_hello",
    )
    role_test, _, _, sources_test = classify_function_role(test_ctx)
    expect_true(role_test == "test", message="test function should be test role")
    expect_true("path:tests" in _signals(sources_test), message="missing tests path signal")


def test_service_tag_and_graph_signal() -> None:
    """Service tags and fan-in/out metrics should tilt toward service role."""
    ctx = _make_context(
        module_tags=["service"],
        module_name="pkg.service",
        graph={"call_fan_in": 10, "call_fan_out": 8},
    )
    role, _, _, sources = classify_function_role(ctx)

    expect_true(role == "service", message="service role not detected")
    expect_true(
        {"tag:service", "graph:fan_in", "graph:fan_out"}.issubset(set(_signals(sources))),
        message="missing service signals",
    )


def _signals(source_payload: dict[str, object]) -> list[str]:
    signals = source_payload.get("signals", [])
    if isinstance(signals, list):
        return [str(sig) for sig in signals]
    return []
