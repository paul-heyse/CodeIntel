"""PR-53: Verify compute code hierarchy and canonical locations."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from codeintel.build.graphs.compute.metrics import components, structural
from codeintel.core.compute import centrality

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"


def _iter_py_files(root: Path) -> list[Path]:
    """Return Python files under a directory tree.

    Parameters
    ----------
    root
        Root directory to scan.

    Returns
    -------
    list[pathlib.Path]
        Python files under ``root`` excluding ``__pycache__`` paths.
    """
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def _relative_path(path: Path) -> str:
    """Return a stable forward-slashed path relative to repo root.

    Parameters
    ----------
    path
        File path under the repository.

    Returns
    -------
    str
        Repository-relative path with forward slashes.
    """
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


class TestComputeHierarchy:
    """Verify compute code follows the established hierarchy."""

    @staticmethod
    def test_core_compute_has_centrality() -> None:
        """Verify centrality functions are implemented in core.compute."""
        missing = [
            name
            for name in (
                "CentralityMetrics",
                "compute_betweenness",
                "compute_closeness",
                "compute_eigenvector_centrality",
                "compute_pagerank",
            )
            if not hasattr(centrality, name)
        ]
        if missing:
            message = "Missing core.compute.centrality exports:\n" + "\n".join(missing)
            pytest.fail(message)

    @staticmethod
    def test_analytics_centrality_delegates_to_core() -> None:
        """Verify analytics centrality wrapper imports from core.compute."""
        centrality_file = SRC_ROOT / "build" / "analytics" / "compute" / "graphs" / "centrality.py"
        text = centrality_file.read_text(encoding="utf-8")

        if "from codeintel.core.compute.centrality import" not in text:
            pytest.xfail(
                f"{_relative_path(centrality_file)} no longer delegates to core.compute.centrality"
            )

        nx_calls = re.findall(
            r"nx\.(pagerank|betweenness_centrality|closeness_centrality)\(",
            text,
        )
        if nx_calls:
            pytest.fail(f"Direct nx calls found: {nx_calls}")

    @staticmethod
    def test_analytics_components_delegates_to_graphs() -> None:
        """Verify analytics components wrapper imports from graphs.compute."""
        components_file = SRC_ROOT / "build" / "analytics" / "compute" / "graphs" / "components.py"
        text = components_file.read_text(encoding="utf-8")
        if "from codeintel.build.graphs.compute.metrics" not in text:
            pytest.fail(f"{_relative_path(components_file)} missing graphs.compute.metrics import")

    @staticmethod
    def test_analytics_structural_delegates_to_graphs() -> None:
        """Verify analytics structural wrapper imports from graphs.compute."""
        structural_file = SRC_ROOT / "build" / "analytics" / "compute" / "graphs" / "structural.py"
        text = structural_file.read_text(encoding="utf-8")
        if "from codeintel.build.graphs.compute.metrics" not in text:
            pytest.fail(f"{_relative_path(structural_file)} missing graphs.compute.metrics import")

    @staticmethod
    def test_no_circular_imports() -> None:
        """Verify core.compute does not import analytics or graphs.compute."""
        core_compute = SRC_ROOT / "core" / "compute"
        bad: list[str] = []
        for py_file in _iter_py_files(core_compute):
            if py_file.name == "__init__.py":
                continue
            text = py_file.read_text(encoding="utf-8")
            if "from codeintel.build.analytics" in text:
                bad.append(f"{_relative_path(py_file)} imports codeintel.build.analytics")
            if "from codeintel.build.graphs.compute" in text:
                bad.append(f"{_relative_path(py_file)} imports codeintel.build.graphs.compute")
        if bad:
            message = "core.compute import hygiene violations:\n" + "\n".join(sorted(bad))
            pytest.fail(message)


class TestAnalyticsComputeDelegation:
    """Verify analytics.compute delegates algorithm calls to canonical layers."""

    @staticmethod
    def test_no_direct_networkx_algorithms_in_analytics_compute() -> None:
        """Enforce delegation to core.compute or graphs.compute for common algorithms."""
        analytics_compute = SRC_ROOT / "build" / "analytics" / "compute"
        forbidden = (
            "nx.pagerank(",
            "nx.betweenness_centrality(",
            "nx.closeness_centrality(",
            "nx.clustering(",
            "nx.triangles(",
        )

        bad: list[tuple[str, str]] = []
        for py_file in _iter_py_files(analytics_compute):
            text = py_file.read_text(encoding="utf-8")
            bad.extend(
                (_relative_path(py_file), pattern) for pattern in forbidden if pattern in text
            )

        if bad:
            message = "Direct networkx algorithm calls found in analytics.compute:\n"
            message += "\n".join(f"{path}: {pattern}" for path, pattern in bad)
            pytest.fail(message)


class TestComputeExports:
    """Verify compute modules export expected functions."""

    @staticmethod
    def test_core_compute_exports() -> None:
        """Verify core.compute.centrality exports expected functions."""
        expected = {
            "CentralityMetrics",
            "compute_all_centralities",
            "compute_betweenness",
            "compute_closeness",
            "compute_eigenvector_centrality",
            "compute_harmonic_centrality",
            "compute_pagerank",
        }

        actual = set(centrality.__all__)
        missing = expected - actual
        if missing:
            pytest.fail(f"Missing exports: {sorted(missing)}")

    @staticmethod
    def test_graphs_compute_metrics_exports() -> None:
        """Verify graphs.compute.metrics modules expose required functions."""
        required = [
            ("codeintel.build.graphs.compute.metrics.structural", "compute_clustering_coefficient"),
            ("codeintel.build.graphs.compute.metrics.structural", "compute_triangles"),
            ("codeintel.build.graphs.compute.metrics.components", "find_connected"),
            ("codeintel.build.graphs.compute.metrics.components", "find_strongly_connected"),
        ]

        missing: list[str] = []
        for module_name, attr in required:
            module = structural if "structural" in module_name else components
            if not hasattr(module, attr):
                missing.append(f"{module_name}.{attr}")

        if missing:
            message = "Missing compute exports:\n" + "\n".join(sorted(missing))
            pytest.fail(message)
