"""Tests for plugin pack fingerprint invalidation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.runtime.module_resolver import resolve_module_set
from codeintel.runtime.plugins.config import PluginConfig
from tests._helpers.harnesses.hamilton_build import BuildEnvSpec, build_test_env

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


def _write_workspace_module(path: Path, value: int) -> None:
    """Write a simple module file with the provided value."""
    path.write_text(f"VALUE = {value}\n", encoding="utf-8")


def test_pack_fingerprint_invalidation(test_ctx: TestContext) -> None:
    """Update a workspace module and ensure the fingerprint changes."""
    source_root = test_ctx.repo_root / "src"
    workspace_root = source_root / "codeintel_targets"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "__init__.py").write_text("", encoding="utf-8")

    module_path = workspace_root / "fingerprint_target.py"
    _write_workspace_module(module_path, 1)

    env = build_test_env(
        BuildEnvSpec(
            gateway=test_ctx.gateway,
            snapshot=test_ctx.snapshot,
            paths=test_ctx.build_paths,
        )
    )

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    resolved_first = resolve_module_set(
        env=env,
        plugin_config=PluginConfig(allow_workspace_modules=True),
        hamilton_config={},
        include_planning=False,
        codeintel_version="unknown",
    )

    _write_workspace_module(module_path, 2)
    resolved_second = resolve_module_set(
        env=env,
        plugin_config=PluginConfig(allow_workspace_modules=True),
        hamilton_config={},
        include_planning=False,
        codeintel_version="unknown",
    )

    assert resolved_first.fingerprint != resolved_second.fingerprint
