"""Unit tests for semantic role classification signals."""

from __future__ import annotations

import pytest

from codeintel.analytics.semantic_roles import FunctionContext, classify_function_role


def _make_context(**overrides: object) -> FunctionContext:
    defaults: dict[str, object] = {
        "rel_path": "pkg/api.py",
        "qualname": "pkg.api.fn",
        "decorators": [],
        "effects": {},
        "contracts": {},
        "module_tags": [],
        "module_name": "pkg.api",
        "graph": {},
        "loc": 10,
    }
    defaults.update(overrides)
    return FunctionContext(**defaults)  # type: ignore[arg-type]


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
