"""Configuration helpers for function analytics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.function_parsing import (
    FunctionParserRegistry,
    ParsedFile,
    ParsedFileLoader,
)
from codeintel.config.models import FunctionAnalyticsConfig

if TYPE_CHECKING:
    from codeintel.analytics.functions.validation import ValidationReporter


@dataclass(frozen=True)
class ProcessContext:
    """Shared context for building analytics rows."""

    cfg: FunctionAnalyticsConfig
    now: datetime


@dataclass(frozen=True)
class FunctionAnalyticsOptions:
    """Optional hooks and cached context for function analytics."""

    parser: ParsedFileLoader | None = None
    parser_registry: FunctionParserRegistry | None = None
    validation_reporter: ValidationReporter | None = None
    context: AnalyticsContext | None = None


@dataclass
class ProcessState:
    """Mutable state shared across per-file processing."""

    cfg: FunctionAnalyticsConfig
    cache: dict[str, ParsedFile | None]
    parser: ParsedFileLoader
    ctx: ProcessContext
