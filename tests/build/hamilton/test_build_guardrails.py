"""Guardrail tests for the Hamilton build module."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_no_forbidden_any_cast_in_native_targets() -> None:
    """Ensure native build targets do not use forbidden `cast("Any", ...)` patterns."""
    repo_root = Path(__file__).resolve().parents[3]
    native_root = repo_root / "src" / "codeintel" / "build" / "hamilton" / "native"
    if not native_root.is_dir():
        message = f"Expected native build dir to exist: {native_root}"
        pytest.fail(message)

    forbidden = ('cast("Any",', "cast('Any',", "cast(Any,")
    offenders: list[str] = []
    for path in native_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(path.relative_to(repo_root).as_posix())

    if offenders:
        message = f"Forbidden cast(Any) usage in: {offenders}"
        pytest.fail(message)


def test_no_compute_result_any_in_build() -> None:
    """Ensure the legacy `ComputeResult = Any` alias does not reappear in build code."""
    repo_root = Path(__file__).resolve().parents[3]
    build_root = repo_root / "src" / "codeintel" / "build"
    if not build_root.is_dir():
        message = f"Expected build dir to exist: {build_root}"
        pytest.fail(message)

    offenders: list[str] = []
    for path in build_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "ComputeResult" not in text:
            continue
        if "ComputeResult = Any" in text or "type ComputeResult = Any" in text:
            offenders.append(path.relative_to(repo_root).as_posix())

    if offenders:
        message = f"Forbidden ComputeResult = Any usage in: {offenders}"
        pytest.fail(message)
