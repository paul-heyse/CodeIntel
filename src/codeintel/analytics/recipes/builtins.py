"""Built-in analytics recipes.

This module defines the standard set of analytics recipes that are
available out of the box. These recipes can be used directly or
extended via the recipe DSL.
"""

from __future__ import annotations

from codeintel.analytics.recipes.model import AnalyticsRecipe

# =============================================================================
# Core Plugin Lists
# =============================================================================

# Graph-related plugins
GRAPH_PLUGINS: tuple[str, ...] = (
    "core_graph_metrics",
    "graph_metrics_functions_ext",
    "graph_metrics_modules_ext",
    "cfg_metrics",
    "dfg_metrics",
    "symbol_graph_metrics_modules",
    "symbol_graph_metrics_functions",
    "config_graph_metrics",
    "subsystem_graph_metrics",
    "subsystem_agreement",
    "graph_stats",
)

# Function analysis plugins
FUNCTION_PLUGINS: tuple[str, ...] = (
    "functions.metrics",
    "functions.ast_features",
    "functions.effects",
    "functions.contracts",
)

# Test and coverage plugins
TEST_PLUGINS: tuple[str, ...] = (
    "coverage.functions",
    "coverage.test_edges",
    "tests.profile",
    "tests.behavioral_coverage",
    "test_graph_metrics",
)

# Risk and hotspot plugins
RISK_PLUGINS: tuple[str, ...] = (
    "hotspots.build",
    "risk_factors.build",
)

# Subsystem and architecture plugins
ARCHITECTURE_PLUGINS: tuple[str, ...] = (
    "subsystems.build",
    "semantic.roles",
    "data_models.build",
    "data_models.usage",
)

# Entrypoint and API plugins
API_PLUGINS: tuple[str, ...] = ("entrypoints.build",)

# History plugins
HISTORY_PLUGINS: tuple[str, ...] = (
    "history.functions",
    "history.timeseries",
)

# Profile plugins
PROFILE_PLUGINS: tuple[str, ...] = (
    "profiles.functions",
    "profiles.modules",
    "profiles.files",
)

# =============================================================================
# Built-in Recipes
# =============================================================================

QUICK_AUDIT = AnalyticsRecipe(
    name="quick_audit",
    description="Fast codebase health check (metrics + types + hotspots).",
    plugins=(
        "functions.metrics",
        "functions.ast_features",
        "hotspots.build",
        "core_graph_metrics",
        "graph_stats",
    ),
    tags=("fast", "audit", "health"),
    max_duration_ms=120_000,  # 2 minutes
)

FULL_ANALYSIS = AnalyticsRecipe(
    name="full_analysis",
    description="Complete analytics suite for comprehensive codebase analysis.",
    plugins=(
        # Functions
        *FUNCTION_PLUGINS,
        # Graphs
        *GRAPH_PLUGINS,
        # Risk
        *RISK_PLUGINS,
        # Architecture
        *ARCHITECTURE_PLUGINS,
        # Coverage
        *TEST_PLUGINS,
        # Profiles
        *PROFILE_PLUGINS,
        # APIs
        *API_PLUGINS,
    ),
    tags=("complete", "comprehensive"),
    fail_fast=False,
)

COVERAGE_FOCUS = AnalyticsRecipe(
    name="coverage_focus",
    description="Coverage-centric analysis for test quality assessment.",
    plugins=(
        "functions.metrics",
        "coverage.functions",
        "coverage.test_edges",
        "tests.profile",
        "tests.behavioral_coverage",
        "test_graph_metrics",
    ),
    tags=("coverage", "testing", "quality"),
)

TEST_ANALYSIS = AnalyticsRecipe(
    name="test_analysis",
    description="Deep test suite analysis including behavioral classification.",
    plugins=(
        *TEST_PLUGINS,
        "functions.metrics",
        "core_graph_metrics",
    ),
    tags=("testing", "quality"),
)

GRAPH_METRICS = AnalyticsRecipe(
    name="graph_metrics",
    description="Complete graph-based metrics for architecture analysis.",
    plugins=GRAPH_PLUGINS,
    tags=("graphs", "architecture", "metrics"),
)

RISK_ANALYSIS = AnalyticsRecipe(
    name="risk_analysis",
    description="Risk-focused analysis for identifying code hotspots.",
    plugins=(
        "functions.metrics",
        "functions.ast_features",
        *RISK_PLUGINS,
        "core_graph_metrics",
        "graph_metrics_functions_ext",
        "subsystems.build",
    ),
    tags=("risk", "hotspots", "maintenance"),
)

ARCHITECTURE_ANALYSIS = AnalyticsRecipe(
    name="architecture_analysis",
    description="Architecture-focused analysis for understanding system structure.",
    plugins=(
        "functions.metrics",
        *GRAPH_PLUGINS,
        *ARCHITECTURE_PLUGINS,
        *API_PLUGINS,
    ),
    tags=("architecture", "structure", "design"),
)

HISTORY_ANALYSIS = AnalyticsRecipe(
    name="history_analysis",
    description="Historical analysis for understanding code evolution.",
    plugins=(
        "functions.metrics",
        *HISTORY_PLUGINS,
        "hotspots.build",
    ),
    tags=("history", "evolution", "trends"),
)

API_ANALYSIS = AnalyticsRecipe(
    name="api_analysis",
    description="API and entrypoint analysis for service documentation.",
    plugins=(
        "functions.metrics",
        "functions.ast_features",
        *API_PLUGINS,
        "data_models.build",
        "data_models.usage",
        "semantic.roles",
    ),
    tags=("api", "entrypoints", "documentation"),
)

__all__ = [
    "API_ANALYSIS",
    "API_PLUGINS",
    "ARCHITECTURE_ANALYSIS",
    "ARCHITECTURE_PLUGINS",
    "COVERAGE_FOCUS",
    "FULL_ANALYSIS",
    "FUNCTION_PLUGINS",
    "GRAPH_METRICS",
    "GRAPH_PLUGINS",
    "HISTORY_ANALYSIS",
    "HISTORY_PLUGINS",
    "PROFILE_PLUGINS",
    "QUICK_AUDIT",
    "RISK_ANALYSIS",
    "RISK_PLUGINS",
    "TEST_ANALYSIS",
    "TEST_PLUGINS",
]
