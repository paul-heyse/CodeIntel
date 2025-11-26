"""Configuration helpers for function analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.analytics.context import AnalyticsContext
from codeintel.config import FunctionAnalyticsStepConfig

if TYPE_CHECKING:
    from codeintel.analytics.parsing.models import ParsedModule, SourceSpan
    from codeintel.analytics.parsing.validation import FunctionValidationReporter


@dataclass(frozen=True)
class ProcessContext:
    """Shared context for building analytics rows."""

    cfg: FunctionAnalyticsStepConfig
    now: datetime


@dataclass(frozen=True)
class FunctionAnalyticsOptions:
    """Optional hooks and cached context for function analytics."""

    validation_reporter: FunctionValidationReporter | None = None
    context: AnalyticsContext | None = None


@dataclass
class ProcessState:
    """Mutable state shared across per-file processing."""

    cfg: FunctionAnalyticsStepConfig
    cache: dict[str, ParsedModule | None]
    span_index: dict[int, SourceSpan]
    reporter: FunctionValidationReporter
    ctx: ProcessContext
