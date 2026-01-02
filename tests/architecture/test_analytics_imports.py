"""Guardrails for importing analytics modules from production code."""

from __future__ import annotations

from pathlib import Path

import pytest

FORBIDDEN_IMPORTS = (
    "function_contracts",
    "function_effects",
    "functions.config",
    "functions.metrics",
    "graph_metrics",
    "graph_metrics_ext",
    "module_graph_metrics_ext",
    "graph_stats",
    "config_graph_metrics",
    "config_data_flow",
    "subsystem_graph_metrics",
    "subsystem_agreement",
    "symbol_graph_metrics",
)


def test_external_imports_use_domain_apis() -> None:
    """Ensure production modules rely on domain APIs instead of deep imports."""
    root = Path("src/codeintel")
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if "analytics" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "codeintel.build.analytics." not in line or "import" not in line:
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for suffix in FORBIDDEN_IMPORTS:
                needle = f"codeintel.build.analytics.{suffix}"
                if needle in stripped:
                    violations.append(f"{path}:{suffix}")
    if violations:
        pytest.fail(f"Use domain APIs for analytics imports: {violations}")
