"""Helpers for loading generated SCIP protobuf bindings."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_generated_module(path: Path) -> ModuleType:
    """Load the generated scip_pb2 module from a file path.

    Returns
    -------
    ModuleType
        Imported protobuf module.

    Raises
    ------
    FileNotFoundError
        If the module file does not exist.
    ImportError
        If the module cannot be imported.
    """
    if not path.is_file():
        msg = f"SCIP protobuf module not found: {path}"
        raise FileNotFoundError(msg)
    spec = importlib.util.spec_from_file_location("scip_pb2", path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load SCIP protobuf module: {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = ["load_generated_module"]
