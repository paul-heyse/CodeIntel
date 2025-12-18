"""Tooling metadata helpers shared across transports.

Single source of truth for:
- runtime package version reporting for key dependencies
- snapshot vs runtime mismatch warnings
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version
from typing import Final

_DEFAULT_TOOL_NAMES: Final[tuple[str, ...]] = (
    "codeintel",
    "duckdb",
    "ibis-framework",
    "sqlglot",
    "pyarrow",
)


def runtime_versions(*, tools: tuple[str, ...] = _DEFAULT_TOOL_NAMES) -> dict[str, str]:
    """Return runtime versions for the configured tool set."""
    versions: dict[str, str] = {}
    for tool in tools:
        try:
            versions[tool] = get_package_version(tool)
        except PackageNotFoundError:
            versions[tool] = "not-installed"
    return versions


def tooling_mismatch_warnings(
    environment: Mapping[str, object],
    *,
    runtime: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Compute snapshot vs runtime version mismatch warnings."""
    tools_obj = environment.get("tools")
    snapshot_tools = tools_obj if isinstance(tools_obj, Mapping) else {}
    runtime_versions_map = dict(runtime) if runtime is not None else runtime_versions()

    warnings: list[str] = []
    for key, runtime_version in runtime_versions_map.items():
        snapshot_version_obj = snapshot_tools.get(key)
        if snapshot_version_obj is None:
            continue
        snapshot_version = str(snapshot_version_obj)
        if snapshot_version != runtime_version:
            warnings.append(
                f"tool-version-mismatch: {key} snapshot={snapshot_version} runtime={runtime_version}"
            )
    return tuple(warnings)


__all__ = ["runtime_versions", "tooling_mismatch_warnings"]
