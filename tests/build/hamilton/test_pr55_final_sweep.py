"""PR-55: Final sweep verification tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from codeintel.build import hamilton
from codeintel.build.hamilton.native import analytics
from tests.build.hamilton.snapshots._manifest import load_snapshot_manifest


def _find_repo_root(start: Path) -> Path:
    """Find repository root by locating pyproject.toml.

    Parameters
    ----------
    start
        Starting path for upward search.

    Returns
    -------
    Path
        Repository root path.

    Raises
    ------
    RuntimeError
        If pyproject.toml cannot be found in any parent directory.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    message = f"Unable to locate pyproject.toml from start={start}"
    raise RuntimeError(message)


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
SRC_ROOT = REPO_ROOT / "src" / "codeintel"
SNAPSHOTS_DIR = REPO_ROOT / "tests" / "build" / "hamilton" / "snapshots"
MANIFEST_PATH = REPO_ROOT / "config" / "hamilton" / "cli_snapshots" / "manifest.yaml"

PR_TAG_RE = re.compile(r"^pr\d{2,}$")
PHASE_TAG_RE = re.compile(r"^phase\d+$")

COMMAND_TAGS = {
    "assets",
    "compile",
    "diff",
    "explain",
    "graph",
    "history",
    "lineage",
    "migrate",
    "plan",
    "promote",
    "resolve",
    "run",
    "schema",
    "spec",
    "status",
    "validate",
}
DOMAIN_TAGS = {"analytics", "export", "graphs"}
FORMAT_TAGS = {"dot", "json", "mermaid", "text"}
SCOPE_TAGS = {"integration", "tiny"}
MODE_TAGS = {"generated", "native"}

SNAPSHOT_SUFFIXES = {".dot", ".json", ".mmd", ".mermaid", ".txt"}


class TestSnapshotManifestTaxonomy:
    """Verify CLI snapshot manifest has valid tag taxonomy and no orphan files."""

    @staticmethod
    def test_manifest_tags_are_valid() -> None:
        """Verify all case tags conform to the snapshot taxonomy."""
        manifest = load_snapshot_manifest(MANIFEST_PATH)

        invalid: list[tuple[str, str]] = []
        duplicated: list[str] = []

        for case in manifest.cases:
            if len(case.tags) != len(set(case.tags)):
                duplicated.append(case.name)

            for tag in case.tags:
                if tag == "phase0":
                    invalid.append((case.name, tag))
                    continue

                if PR_TAG_RE.fullmatch(tag) or PHASE_TAG_RE.fullmatch(tag):
                    continue

                if (
                    tag in COMMAND_TAGS
                    or tag in DOMAIN_TAGS
                    or tag in FORMAT_TAGS
                    or tag in SCOPE_TAGS
                    or tag in MODE_TAGS
                ):
                    continue

                invalid.append((case.name, tag))

        if duplicated:
            message = "Cases contain duplicated tags:\n" + "\n".join(sorted(duplicated))
            pytest.fail(message)

        if invalid:
            message = "Invalid tags found:\n" + "\n".join(f"{name}: {tag}" for name, tag in invalid)
            pytest.fail(message)

    @staticmethod
    def test_all_snapshots_referenced() -> None:
        """Verify all snapshot files are referenced in the manifest (and vice versa)."""
        manifest = load_snapshot_manifest(MANIFEST_PATH)

        referenced = {case.snapshot for case in manifest.cases if case.snapshot}
        actual = {
            p.name for p in SNAPSHOTS_DIR.iterdir() if p.is_file() and p.suffix in SNAPSHOT_SUFFIXES
        }

        missing = referenced - actual
        orphans = actual - referenced

        if missing:
            message = "Manifest references missing snapshots:\n" + "\n".join(sorted(missing))
            pytest.fail(message)

        if orphans:
            message = "Orphan snapshot files:\n" + "\n".join(sorted(orphans))
            pytest.fail(message)


class TestPublicApiClean:
    """Verify public APIs do not expose deprecated surfaces."""

    @staticmethod
    def test_analytics_no_deprecated_exports() -> None:
        """Verify analytics package doesn't export deprecated functions."""
        deprecated = [
            "build_entrypoints",
            "build_external_dependencies",
            "compute_cfg_metrics",
            "compute_data_models",
            "compute_dfg_metrics",
            "compute_test_graph_metrics",
        ]

        exported = [name for name in deprecated if hasattr(analytics, name)]
        if exported:
            message = "Deprecated functions still exported:\n" + "\n".join(exported)
            pytest.fail(message)

    @staticmethod
    def test_build_hamilton_no_legacy_exports() -> None:
        """Verify build.hamilton no longer exports legacy compatibility helpers."""
        forbidden = ("LEGACY_PHASE0", "build_driver_compat", "list_available_nodes_compat")
        leaked = [name for name in forbidden if hasattr(hamilton, name)]
        if leaked:
            message = "Legacy build.hamilton exports still present:\n" + "\n".join(leaked)
            pytest.fail(message)


class TestNoDeadImports:
    """Verify no dead imports remain for deleted module paths."""

    @staticmethod
    def test_no_deleted_module_imports() -> None:
        """Verify no imports of deleted modules remain in src."""
        bad_substrings = (
            "codeintel.build.analytics.runtime",
            "codeintel.build.plugin_registry",
            "codeintel.config.datasets.schema_registry",
        )

        bad: dict[str, list[str]] = {s: [] for s in bad_substrings}
        for py_file in SRC_ROOT.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue

            text = py_file.read_text(encoding="utf-8")
            for substring in bad_substrings:
                if substring in text:
                    bad[substring].append(str(py_file.relative_to(REPO_ROOT)))

        failures = {k: v for k, v in bad.items() if v}
        if failures:
            message = "Deleted module imports found:\n" + "\n".join(
                f"{substring}:\n  " + "\n  ".join(sorted(paths))
                for substring, paths in sorted(failures.items())
            )
            pytest.fail(message)
