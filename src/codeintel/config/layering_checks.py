"""Report layering violations by scanning codeintel imports."""

from __future__ import annotations

import ast
import pathlib
import sys
from typing import Final

ROOT = pathlib.Path(__file__).resolve().parents[1]

LAYER_FOR_PREFIX: Final = {
    "codeintel.config.builder": "app",
    "codeintel.config.models": "app",
    "codeintel.config.serving_models": "app",
    "codeintel.config.primitives": "core",
    "codeintel.config.schemas": "core",
    "codeintel.config": "core",
    "codeintel.storage": "core",
    "codeintel.ingestion": "domain",
    "codeintel.analytics": "domain",
    "codeintel.graphs": "domain",
    "codeintel.pipeline": "app",
    "codeintel.serving": "app",
    "codeintel.cli": "app",
}

ALLOWED: Final = {
    "core": {"core"},
    "domain": {"core", "domain"},
    "app": {"core", "domain", "app"},
}


def classify_module(module: str) -> str | None:
    """
    Return the configured layer for a module path when known.

    Parameters
    ----------
    module : str
        Fully qualified module name.

    Returns
    -------
    str | None
        Layer label when mapped, otherwise None.
    """
    best: tuple[str, str] | None = None
    for prefix, layer in LAYER_FOR_PREFIX.items():
        is_match = module == prefix or module.startswith(prefix + ".")
        is_longer = best is None or len(prefix) > len(best[0])
        if is_match and is_longer:
            best = (prefix, layer)
    return best[1] if best is not None else None


def find_imports(path: pathlib.Path) -> set[str]:
    """
    Collect codeintel.* imports from a Python file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the Python source file.

    Returns
    -------
    set[str]
        Imported module names under the codeintel namespace.
    """
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("codeintel."):
                    modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            if module_name is not None and module_name.startswith("codeintel."):
                modules.add(module_name)
    return modules


def main() -> int:
    """
    Return non-zero when layering violations are detected.

    Returns
    -------
    int
        Zero when imports respect layering rules, otherwise non-zero.
    """
    errors: list[str] = []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT)
        module = "codeintel." + ".".join(rel.with_suffix("").parts)
        src_layer = classify_module(module)
        if src_layer is None:
            continue
        for imported in find_imports(py):
            dst_layer = classify_module(imported)
            if dst_layer is None:
                continue
            if dst_layer not in ALLOWED[src_layer]:
                errors.append(f"{module} ({src_layer}) -> {imported} ({dst_layer}) not allowed")
    if errors:
        for message in sorted(errors):
            sys.stdout.write(f"{message}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
