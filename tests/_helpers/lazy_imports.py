"""Lazy import helpers for test-only modules."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from types import ModuleType


@cache
def _import_module(name: str) -> ModuleType:
    return import_module(name)


def get_hamilton_build_module() -> ModuleType:
    return _import_module("tests._helpers.harnesses.hamilton_build")


def get_graph_harness_module() -> ModuleType:
    return _import_module("tests._helpers.harnesses.graph_harness")


def get_analytics_harness_module() -> ModuleType:
    return _import_module("tests._helpers.harnesses.analytics_harness")


def get_serving_harness_module() -> ModuleType:
    return _import_module("tests._helpers.harnesses.serving_harness")


def get_orchestration_graph_module() -> ModuleType:
    return _import_module("tests._helpers.orchestration.graph_orchestration")


def get_orchestration_provisioning_module() -> ModuleType:
    return _import_module("tests._helpers.orchestration.provisioning")


def get_env_module() -> ModuleType:
    return _import_module("tests._helpers.env")


def get_scenarios_module() -> ModuleType:
    return _import_module("tests._helpers.scenarios")


def get_helpers_module() -> ModuleType:
    return _import_module("tests._helpers")


def get_compose_runtime_module() -> ModuleType:
    return _import_module("codeintel.runtime.compose")
