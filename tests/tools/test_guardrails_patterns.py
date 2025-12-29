"""Tests for guardrail pattern detection."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Protocol

    class GuardrailsApi(Protocol):
        """Protocol for guardrails helpers used by tests."""

        BASE_DIRS: tuple[str, ...]

        def find_violations(self, repo_root: Path) -> list[str]:
            """Return guardrail violations for a given repo root."""
            ...


guardrails = cast("GuardrailsApi", importlib.import_module("tools.guardrails"))


def _write_fixture(tmp_path: Path, rel_path: str, content: str) -> Path:
    path = tmp_path / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _collect_violations(
    tmp_path: Path,
    *,
    rel_path: str,
    content: str,
    base_dirs: tuple[str, ...],
) -> list[str]:
    original_dirs = guardrails.BASE_DIRS
    try:
        guardrails.BASE_DIRS = base_dirs
        _write_fixture(tmp_path, rel_path, content)
        return guardrails.find_violations(tmp_path)
    finally:
        guardrails.BASE_DIRS = original_dirs


def test_streaming_guardrail_flags_fetchall(tmp_path: Path) -> None:
    """Detect fetchall usage in build modules."""
    violations = _collect_violations(
        tmp_path,
        rel_path="src/codeintel/build/sample.py",
        content="relation.fetchall()\\n",
        base_dirs=("src",),
    )
    assert any("streaming_fetchall" in violation for violation in violations)


def test_streaming_guardrail_allows_tests(tmp_path: Path) -> None:
    """Ignore streaming guardrails inside tests/."""
    violations = _collect_violations(
        tmp_path,
        rel_path="tests/test_sample.py",
        content="relation.fetchall()\\n",
        base_dirs=("tests",),
    )
    assert not violations


def test_guardrail_flags_direct_schema_modifier_import(tmp_path: Path) -> None:
    """Flag direct schema modifier imports outside allowed modules."""
    violations = _collect_violations(
        tmp_path,
        rel_path="src/codeintel/serving/sample.py",
        content="from hamilton.function_modifiers import schema\\n",
        base_dirs=("src",),
    )
    assert any("direct_hamilton_schema_modifier" in violation for violation in violations)


def test_guardrail_flags_static_targets_registry(tmp_path: Path) -> None:
    """Flag static target registry declarations."""
    violations = _collect_violations(
        tmp_path,
        rel_path="src/codeintel/sample.py",
        content="TARGETS = []\\n",
        base_dirs=("src",),
    )
    assert any("module_discovery_static_targets" in violation for violation in violations)


def test_guardrail_flags_codeintel_targets_import(tmp_path: Path) -> None:
    """Flag direct codeintel_targets imports."""
    violations = _collect_violations(
        tmp_path,
        rel_path="src/codeintel/sample.py",
        content="import codeintel_targets.demo\\n",
        base_dirs=("src",),
    )
    assert any("module_discovery_codeintel_targets_import" in violation for violation in violations)
