"""Tests for workspace module discovery."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.runtime.compose import compose_runtime
from tests._helpers.harnesses.hamilton_build import BuildEnvSpec, build_test_env

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


def _write_target_module(path: Path, target_name: str) -> None:
    """Write a minimal target module fixture."""
    path.write_text(
        "\n".join(
            [
                "from codeintel.sdk import target_anchor",
                "",
                f'@target_anchor(domain="analytics", target="{target_name}")',
                f"def t__{target_name}() -> int:",
                "    return 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_add_module_adds_target(test_ctx: TestContext) -> None:
    """Include newly added workspace targets in runtime catalog."""
    source_root = test_ctx.repo_root / "src"
    workspace_root = source_root / "codeintel_targets"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "__init__.py").write_text("", encoding="utf-8")

    module_path = workspace_root / "demo_target.py"
    _write_target_module(module_path, "demo_target")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    env = build_test_env(
        BuildEnvSpec(
            gateway=test_ctx.gateway,
            snapshot=test_ctx.snapshot,
            paths=test_ctx.build_paths,
        )
    )

    runtime = compose_runtime(env=env).bundle
    assert "demo_target" in runtime.catalog.targets
