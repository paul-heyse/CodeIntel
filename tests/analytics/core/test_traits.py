"""Tests for trait mixins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import pytest

from codeintel.analytics.core.traits import ScopeAwareMixin
from codeintel.analytics.runtime.manifest import AnalyticsScope


class _ScopeAwarePlugin(ScopeAwareMixin):
    """Concrete helper to test scope filtering."""

    def __init__(self, supported: tuple[Literal["paths", "modules", "time_window"], ...]) -> None:
        self._supported_scopes = supported


@dataclass(frozen=True)
class ScopeCase:
    """Test case for scope filtering."""

    supported: tuple[Literal["paths", "modules", "time_window"], ...]
    scope: AnalyticsScope
    should_run: bool


@pytest.mark.parametrize(
    "case",
    [
        ScopeCase(
            (),
            AnalyticsScope(),
            should_run=True,
        ),
        ScopeCase(
            ("paths",),
            AnalyticsScope(paths=("src/",)),
            should_run=True,
        ),
        ScopeCase(
            ("paths",),
            AnalyticsScope(modules=("mod",)),
            should_run=False,
        ),
        ScopeCase(
            ("modules", "paths"),
            AnalyticsScope(paths=("src/",), modules=("mod",)),
            should_run=True,
        ),
        ScopeCase(
            ("modules",),
            AnalyticsScope(
                time_window=(
                    datetime.now(tz=UTC),
                    datetime.now(tz=UTC),
                )
            ),
            should_run=False,
        ),
    ],
)
def test_filter_by_scope_respects_supported_scopes(case: ScopeCase) -> None:
    """Scope filtering should only allow supported scope kinds."""
    plugin = _ScopeAwarePlugin(case.supported)

    assert plugin.filter_by_scope(case.scope) is case.should_run
