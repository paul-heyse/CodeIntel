"""Helpers for extracting common analytics sample identifiers from gateways."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.build.analytics.compute.dependencies.classification import (
    DependencyModePattern,
    LibraryPattern,
)
from codeintel.build.analytics.compute.dependencies.detection import DependencyCall

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class AnalyticsSamples:
    """Representative analytics identifiers for tests."""

    goid_h128: int
    urn: str
    rel_path: str
    qualname: str
    subsystem_id: str
    module: str


def load_analytics_samples(gateway: StorageGateway) -> AnalyticsSamples:
    """Retrieve representative analytics identifiers from a seeded gateway.

    Parameters
    ----------
    gateway
        Gateway seeded with analytics data.

    Returns
    -------
    AnalyticsSamples
        Sample goids, URNs, subsystem IDs, modules, and paths.
    """

    def _pick(query: str, label: str) -> str | int:
        row = gateway.con.execute(query).fetchone()
        if row is None:
            pytest.skip(f"No {label} available in seeded analytics data")
        return row[0]

    goid_h128 = _pick("SELECT goid_h128 FROM core.goids LIMIT 1", "goid")
    urn = _pick("SELECT urn FROM core.goids WHERE urn IS NOT NULL LIMIT 1", "URN")
    rel_path, qualname = gateway.con.execute(
        "SELECT rel_path, qualname FROM analytics.function_metrics LIMIT 1"
    ).fetchone() or (None, None)
    if rel_path is None or qualname is None:
        pytest.skip("No function metrics available in seeded analytics data")
    subsystem_id = _pick(
        "SELECT DISTINCT subsystem_id FROM analytics.subsystems LIMIT 1",
        "subsystem_id",
    )
    module = _pick("SELECT module FROM analytics.subsystem_modules LIMIT 1", "module")

    return AnalyticsSamples(
        goid_h128=int(goid_h128),
        urn=str(urn),
        rel_path=str(rel_path),
        qualname=str(qualname),
        subsystem_id=str(subsystem_id),
        module=str(module),
    )


def architecture_seed_selector(gateway: StorageGateway) -> AnalyticsSamples:
    """Return architecture-friendly analytics samples with graceful skips.

    Parameters
    ----------
    gateway
        Gateway seeded with architecture analytics data.

    Returns
    -------
    AnalyticsSamples
        Sample identifiers suitable for subsystem/profile/function tests.
    """
    return load_analytics_samples(gateway)


def dependency_library_patterns() -> dict[str, LibraryPattern]:
    """Return standard library patterns for dependency tests.

    Returns
    -------
    dict[str, LibraryPattern]
        Patterns keyed by library name.
    """
    return {
        "requests": LibraryPattern(
            library="requests",
            service_name="HTTP Client",
            category="http",
            matchers=[
                DependencyModePattern(modes=["read"], method="get"),
                DependencyModePattern(modes=["write"], method_prefix="post"),
                DependencyModePattern(modes=["write"], method="put"),
                DependencyModePattern(modes=["delete"], method="delete"),
            ],
            severity="medium",
            criticality=2.0,
        ),
        "sqlalchemy": LibraryPattern(
            library="sqlalchemy",
            service_name="Database ORM",
            category="database",
            matchers=[
                DependencyModePattern(modes=["query"], method="execute"),
                DependencyModePattern(modes=["write"], method_prefix="insert"),
                DependencyModePattern(modes=["write"], method_prefix="update"),
                DependencyModePattern(modes=["delete"], method_prefix="delete"),
            ],
            severity="high",
            criticality=3.0,
        ),
    }


def dependency_patterns_yaml(patterns: dict[str, LibraryPattern] | None = None) -> str:
    """Generate dependency_patterns.yml content from patterns.

    Parameters
    ----------
    patterns
        Optional map of library patterns to serialize. Defaults to standard patterns.

    Returns
    -------
    str
        YAML string representing dependency patterns.
    """
    libs = patterns or dependency_library_patterns()
    lines: list[str] = ["libs:"]
    for lib, pattern in libs.items():
        lines.extend(
            [
                f"  {lib}:",
                f"    severity: {pattern.severity or 'medium'}",
                f"    criticality: {pattern.criticality or 1.0}",
                "    patterns:",
            ]
        )
        for matcher in pattern.matchers:
            mode_list = ", ".join(f'"{mode}"' for mode in matcher.modes)
            lines.append(f"      - mode: [{mode_list}]")
            if matcher.method:
                lines.append(f'        method: "{matcher.method}"')
            if matcher.method_prefix:
                lines.append(f'        method_prefix: "{matcher.method_prefix}"')
            if matcher.match:
                lines.append(f'        match: "{matcher.match}"')
    return "\n".join(lines)


def dependency_alias_sources() -> dict[str, str]:
    """Return simple sources for alias-map construction tests.

    Returns
    -------
    dict[str, str]
        Mapping of file names to Python source content.
    """
    return {
        "a.py": "import requests as rq",
        "b.py": "from sqlalchemy import create_engine",
    }


def dependency_calls_sample(
    factories: Iterable[Callable[[str], DependencyCall]] | None = None,
) -> list[DependencyCall]:
    """Return representative dependency calls grouped by library.

    Parameters
    ----------
    factories
        Optional iterable of call factories keyed by library name.

    Returns
    -------
    list[DependencyCall]
        Dependency calls for requests and sqlalchemy.
    """
    calls: list[DependencyCall] = [
        DependencyCall(
            library="requests",
            target="get",
            modes=["read"],
            severity=None,
            criticality=None,
        ),
        DependencyCall(
            library="requests",
            target="post",
            modes=["write"],
            severity=None,
            criticality=None,
        ),
        DependencyCall(
            library="sqlalchemy",
            target="execute",
            modes=["query"],
            severity=None,
            criticality=None,
        ),
    ]
    if factories is not None:
        calls.extend(factory("requests") for factory in factories)
    return calls


__all__ = [
    "AnalyticsSamples",
    "architecture_seed_selector",
    "dependency_alias_sources",
    "dependency_calls_sample",
    "dependency_library_patterns",
    "dependency_patterns_yaml",
    "load_analytics_samples",
]
