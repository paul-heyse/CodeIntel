"""Regression tests for entrypoint evidence payloads."""

from __future__ import annotations

import textwrap

import pytest

from codeintel.analytics.entrypoint_detectors import DetectorSettings, detect_entrypoints


def test_detect_entrypoints_emits_snippet_evidence() -> None:
    """detect_entrypoints should emit decorator evidence with snippets."""
    source = textwrap.dedent(
        """\
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/ping")
        def ping() -> str:
            return "ok"
        """
    )
    candidates = detect_entrypoints(
        source,
        rel_path="api.py",
        module="api",
        settings=DetectorSettings(),
    )
    if not candidates:
        pytest.fail("Expected a FastAPI entrypoint candidate")
    evidence = candidates[0].evidence
    if not evidence:
        pytest.fail("Entrypoint candidate did not include evidence")
    sample = evidence[0]
    expected_line = 5
    if sample["path"] != "api.py":
        pytest.fail("Incorrect evidence path")
    if sample["lineno"] != expected_line:
        pytest.fail("Decorator line number did not match source")
    if sample["snippet"] != '@app.get("/ping")':
        pytest.fail("Snippet did not capture the decorator source")
    if "app.get" not in str(sample["details"]):
        pytest.fail("Decorator details were not captured")
