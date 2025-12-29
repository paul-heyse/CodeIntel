"""Tests for entrypoint-based plugin pack loading."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.runtime.module_resolver import resolve_module_set
from codeintel.runtime.plugins import loader as plugin_loader
from codeintel.runtime.plugins.config import PluginConfig
from tests._helpers.harnesses.hamilton_build import BuildEnvSpec, build_test_env

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.runtime.module_resolver import ModuleProvenance
    from tests._helpers.context import TestContext


@dataclass(frozen=True)
class FakeEntryPoint:
    """Minimal entry point stub for plugin discovery tests."""

    name: str
    value: str
    group: str = plugin_loader.TARGET_PACK_ENTRYPOINT_GROUP
    dist: object | None = None

    def load(self) -> object:
        """Load the entry point target pack.

        Returns
        -------
        object
            Loaded entry point factory.
        """
        module_path, attr = self.value.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, attr)


def _write_pack(root: Path, name: str, *, target_module: str) -> None:
    pkg_dir = root / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "targets.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (pkg_dir / "plugin.py").write_text(
        "\n".join(
            [
                "from codeintel.runtime.plugins.spec import TargetPack, TargetPackModule",
                "",
                "def codeintel_target_pack() -> TargetPack:",
                "    return TargetPack(",
                f'        name="{name}",',
                '        version="0.1.0",',
                "        modules=(",
                f'            TargetPackModule(import_path="{target_module}"),',
                "        ),",
                '        requires_codeintel=">=0",',
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _lookup_provenance(
    provenance: Mapping[str, ModuleProvenance],
    module_name: str,
) -> ModuleProvenance:
    entry = provenance.get(module_name)
    if entry is None:
        message = f"Expected provenance for {module_name}"
        raise AssertionError(message)
    return entry


def test_entrypoint_pack_loading_records_provenance(
    test_ctx: TestContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load entrypoint packs and verify provenance metadata."""
    pack_root = test_ctx.repo_root / "plugin_packs"
    _write_pack(pack_root, "alpha_pack", target_module="alpha_pack.targets")
    _write_pack(pack_root, "beta_pack", target_module="beta_pack.targets")

    monkeypatch.syspath_prepend(str(pack_root))
    entry_points = [
        FakeEntryPoint(name="beta", value="beta_pack.plugin:codeintel_target_pack"),
        FakeEntryPoint(name="alpha", value="alpha_pack.plugin:codeintel_target_pack"),
    ]
    monkeypatch.setattr(
        plugin_loader.importlib_metadata,
        "entry_points",
        lambda: entry_points,
    )

    env = build_test_env(
        BuildEnvSpec(
            gateway=test_ctx.gateway,
            snapshot=test_ctx.snapshot,
            paths=test_ctx.build_paths,
        )
    )
    resolved = resolve_module_set(
        env=env,
        plugin_config=PluginConfig(),
        hamilton_config={},
        include_planning=False,
        codeintel_version="unknown",
    )

    pack_names = [pack.name for pack in resolved.packs]
    assert pack_names == ["alpha_pack", "beta_pack"]

    alpha_prov = _lookup_provenance(resolved.provenance, "alpha_pack.targets")
    assert alpha_prov.origin == "plugin"
    assert alpha_prov.plugin_name == "alpha_pack"

    beta_prov = _lookup_provenance(resolved.provenance, "beta_pack.targets")
    assert beta_prov.origin == "plugin"
    assert beta_prov.plugin_name == "beta_pack"
