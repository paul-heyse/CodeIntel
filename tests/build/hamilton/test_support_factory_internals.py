"""Unit tests for build Hamilton support-module internals."""

from __future__ import annotations

import inspect
from types import ModuleType

import pytest

from codeintel.build.hamilton.nodes.mappings import SupportNodeMappings
from codeintel.build.hamilton.nodes.module_attach import attach_node
from codeintel.build.hamilton.nodes.signature_tools import set_signature


def test_set_signature_attaches_signature_and_annotations() -> None:
    """set_signature attaches a synthetic signature used by Hamilton."""
    def fn(**_kwargs: object) -> object:
        return None

    signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                "x",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=int,
            ),
        ],
        return_annotation=str,
    )
    set_signature(fn, signature)
    if inspect.signature(fn) != signature:
        pytest.fail("Expected inspect.signature(fn) to match the provided signature")
    if fn.__annotations__.get("x") is not int:
        pytest.fail("Expected parameter annotation for 'x' to be int")
    if fn.__annotations__.get("return") is not str:
        pytest.fail("Expected return annotation to be str")


def test_attach_node_sets_callable_metadata() -> None:
    """attach_node updates callable metadata and attaches it to the module."""
    module = ModuleType("tests.build.hamilton.support_factory_internals")

    def fn() -> None:
        return None

    attach_node(module, node_name="some_node", fn=fn)
    if fn.__name__ != "some_node":
        pytest.fail("Expected attach_node to set __name__")
    if fn.__module__ != module.__name__:
        pytest.fail("Expected attach_node to set __module__")
    if module.some_node is not fn:
        pytest.fail("Expected attach_node to attach callable to module attribute")


def test_support_node_mappings_attach_to_module() -> None:
    """SupportNodeMappings attaches mapping dicts to the generated module."""
    module = ModuleType("tests.build.hamilton.support_factory_internals.mappings")
    mappings = SupportNodeMappings(
        target_to_node={"t": "t__t"},
        dataset_to_node={"core.repo_map": "d__core__repo_map"},
        query_to_node={"core.repo_map": "q__core__repo_map"},
        dataframe_to_node={"core.repo_map": "df__core__repo_map"},
        artifact_to_node={"export.jsonl": "a__export__jsonl"},
    )
    mappings.attach_to(module)

    if module.TARGET_TO_NODE.get("t") != "t__t":
        pytest.fail("Expected TARGET_TO_NODE mapping to be attached")
    if module.DATASET_TO_NODE.get("core.repo_map") != "d__core__repo_map":
        pytest.fail("Expected DATASET_TO_NODE mapping to be attached")
    if module.QUERY_TO_NODE.get("core.repo_map") != "q__core__repo_map":
        pytest.fail("Expected QUERY_TO_NODE mapping to be attached")
    if module.DATAFRAME_TO_NODE.get("core.repo_map") != "df__core__repo_map":
        pytest.fail("Expected DATAFRAME_TO_NODE mapping to be attached")
    if module.ARTIFACT_TO_NODE.get("export.jsonl") != "a__export__jsonl":
        pytest.fail("Expected ARTIFACT_TO_NODE mapping to be attached")
