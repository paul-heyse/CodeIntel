"""PR-54: Schema registry ownership moved to build.hamilton.contracts.schemas."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src" / "codeintel"


def _iter_py_files() -> list[Path]:
    """Return all Python source files under ``src/codeintel``.

    Returns
    -------
    list[pathlib.Path]
        Python files under ``src/codeintel`` excluding ``__pycache__`` paths.
    """
    return [path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts]


def test_pr54_schema_registry_importable_from_build() -> None:
    """Verify SCHEMA_REGISTRY is importable from build-owned schemas package."""
    schema = SCHEMA_REGISTRY.get("analytics.function_metrics")
    if schema is None:
        pytest.skip("analytics.function_metrics not registered in this environment")


def test_pr54_no_schema_registry_in_config_datasets() -> None:
    """Verify schema registry implementation files are removed from config/datasets."""
    config_datasets = SRC_ROOT / "config" / "datasets"
    removed = (
        "constraints.py",
        "dependency_inference.py",
        "export.py",
        "introspection.py",
        "lineage.py",
        "operation_contracts_dataset.py",
        "pandera_schemas.py",
        "plugin_constraints.py",
        "row_binding_factory.py",
        "row_migration.py",
        "schema.py",
        "schema_builder.py",
        "schema_registry.py",
        "validation.py",
    )
    missing: list[Path] = []
    for rel_path in removed:
        path = config_datasets / rel_path
        if path.exists():
            missing.append(path)

    if missing:
        message = "Schema registry modules should not exist under config/datasets:\n"
        message += "\n".join(str(path) for path in missing)
        pytest.fail(message)


def test_pr54_no_imports_from_config_schema_registry() -> None:
    """Verify no source file imports schema registry from config.datasets."""
    bad: list[str] = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        if "codeintel.config.datasets.schema_registry" in text:
            bad.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))

    if bad:
        message = "Imports from codeintel.config.datasets.schema_registry remain:\n"
        message += "\n".join(sorted(bad))
        pytest.fail(message)
