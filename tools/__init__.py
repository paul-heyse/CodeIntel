"""Utility scripts for CodeIntel tooling."""

import importlib
from types import ModuleType

guardrails: ModuleType = importlib.import_module("tools.guardrails")

__all__ = ["guardrails"]
