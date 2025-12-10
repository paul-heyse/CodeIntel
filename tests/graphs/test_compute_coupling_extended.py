"""Extended tests for coupling computation module.

This module provides additional test coverage for the coupling module
from `codeintel.graphs.compute.metrics.coupling`, including:

- Coupling metrics computation
- Instability metric calculation
- Abstractness computation
- Distance from main sequence
"""

from __future__ import annotations

from typing import Final

import networkx as nx

from codeintel.graphs.compute.metrics.coupling import (
    CouplingMetrics,
    compute_abstractness,
    compute_coupling,
    compute_distance_from_main_sequence,
)
from tests._helpers.assertions import assert_cannot_setattr, expect_equal, expect_true
from tests._helpers.fakes.networkx_graphs import (
    bidirectional_deps_graph,
    god_module_graph,
    hub_dependencies_graph,
    independent_modules_graph,
    linear_dependency_graph,
)

INSTABILITY_TOLERANCE: Final = 0.01
HUB_DEPENDENT_COUNT: Final = 4
HALF_INSTABILITY: Final = 0.5
BALANCED_AFFERENT: Final = 3
BALANCED_EFFERENT: Final = 3
HIGH_AFFERENT: Final = 5
MODERATE_EFFERENT: Final = 3
MODERATE_INSTABILITY: Final = 0.375
HALF_RATIO: Final = 0.5


def test_compute_coupling_independent() -> None:
    """Compute coupling for independent modules."""
    g = independent_modules_graph()

    coupling = compute_coupling(g)

    # No edges for any module
    expect_equal(coupling["module_a"].afferent, 0)
    expect_equal(coupling["module_a"].efferent, 0)
    expect_equal(coupling["module_a"].instability, 0.0)


def test_compute_coupling_linear() -> None:
    """Compute coupling for linear dependencies."""
    g = linear_dependency_graph()

    coupling = compute_coupling(g)

    # A has one efferent, no afferent
    expect_equal(coupling["module_a"].afferent, 0)
    expect_equal(coupling["module_a"].efferent, 1)
    expect_equal(coupling["module_a"].instability, 1.0)

    # B has one of each
    expect_equal(coupling["module_b"].afferent, 1)
    expect_equal(coupling["module_b"].efferent, 1)
    expect_true(abs(coupling["module_b"].instability - 0.5) < INSTABILITY_TOLERANCE)

    # C has one afferent, no efferent
    expect_equal(coupling["module_c"].afferent, 1)
    expect_equal(coupling["module_c"].efferent, 0)
    expect_equal(coupling["module_c"].instability, 0.0)


def test_compute_coupling_hub() -> None:
    """Compute coupling for hub module."""
    g = hub_dependencies_graph()

    coupling = compute_coupling(g)

    # Core has high afferent (4 dependents), no efferent
    expect_equal(coupling["core"].afferent, HUB_DEPENDENT_COUNT)
    expect_equal(coupling["core"].efferent, 0)
    expect_equal(coupling["core"].instability, 0.0)

    # Modules have one efferent each, no afferent
    expect_equal(coupling["module_a"].afferent, 0)
    expect_equal(coupling["module_a"].efferent, 1)
    expect_equal(coupling["module_a"].instability, 1.0)


def test_compute_coupling_god() -> None:
    """Compute coupling for god module."""
    g = god_module_graph()

    coupling = compute_coupling(g)

    # God module has high efferent (4 dependencies)
    expect_equal(coupling["god"].afferent, 0)
    expect_equal(coupling["god"].efferent, HUB_DEPENDENT_COUNT)
    expect_equal(coupling["god"].instability, 1.0)

    # Leaf modules have one afferent, no efferent
    expect_equal(coupling["module_a"].afferent, 1)
    expect_equal(coupling["module_a"].efferent, 0)
    expect_equal(coupling["module_a"].instability, 0.0)


def test_compute_coupling_bidirectional() -> None:
    """Compute coupling for bidirectional dependencies."""
    g = bidirectional_deps_graph()

    coupling = compute_coupling(g)

    # Both modules have 1 afferent and 1 efferent
    expect_equal(coupling["module_a"].afferent, 1)
    expect_equal(coupling["module_a"].efferent, 1)
    expect_true(abs(coupling["module_a"].instability - 0.5) < INSTABILITY_TOLERANCE)

    expect_equal(coupling["module_b"].afferent, 1)
    expect_equal(coupling["module_b"].efferent, 1)
    expect_true(abs(coupling["module_b"].instability - HALF_INSTABILITY) < INSTABILITY_TOLERANCE)


def test_compute_coupling_empty_graph() -> None:
    """Compute coupling for empty graph."""
    g = nx.DiGraph()

    coupling = compute_coupling(g)

    expect_equal(coupling, {})


def test_compute_abstractness_no_abstracts() -> None:
    """Compute abstractness with no abstract classes."""
    abstractness = compute_abstractness("module", abstract_count=0, total_count=10)

    expect_equal(abstractness, 0.0)


def test_compute_abstractness_all_abstract() -> None:
    """Compute abstractness with all abstract classes."""
    abstractness = compute_abstractness("module", abstract_count=5, total_count=5)

    expect_equal(abstractness, 1.0)


def test_compute_abstractness_partial() -> None:
    """Compute abstractness with some abstract classes."""
    abstractness = compute_abstractness("module", abstract_count=3, total_count=6)

    expect_equal(abstractness, HALF_RATIO)


def test_compute_abstractness_empty() -> None:
    """Compute abstractness with no classes."""
    abstractness = compute_abstractness("module", abstract_count=0, total_count=0)

    expect_equal(abstractness, 0.0)


def test_distance_from_main_ideal_stable() -> None:
    """Compute distance for ideal stable module (A=1, I=0)."""
    coupling = CouplingMetrics(afferent=HIGH_AFFERENT, efferent=0, instability=0.0)

    distance = compute_distance_from_main_sequence(coupling, abstractness=1.0)

    expect_equal(distance, 0.0)


def test_distance_from_main_ideal_unstable() -> None:
    """Compute distance for ideal unstable module (A=0, I=1)."""
    coupling = CouplingMetrics(afferent=0, efferent=HIGH_AFFERENT, instability=1.0)

    distance = compute_distance_from_main_sequence(coupling, abstractness=0.0)

    expect_equal(distance, 0.0)


def test_distance_from_main_zone_of_pain() -> None:
    """Compute distance for module in zone of pain (A=0, I=0)."""
    coupling = CouplingMetrics(afferent=HIGH_AFFERENT, efferent=0, instability=0.0)

    distance = compute_distance_from_main_sequence(coupling, abstractness=0.0)

    expect_equal(distance, 1.0)


def test_distance_from_main_zone_of_uselessness() -> None:
    """Compute distance for module in zone of uselessness (A=1, I=1)."""
    coupling = CouplingMetrics(afferent=0, efferent=HIGH_AFFERENT, instability=1.0)

    distance = compute_distance_from_main_sequence(coupling, abstractness=1.0)

    expect_equal(distance, 1.0)


def test_distance_from_main_balanced() -> None:
    """Compute distance for balanced module."""
    coupling = CouplingMetrics(
        afferent=BALANCED_AFFERENT, efferent=BALANCED_EFFERENT, instability=HALF_INSTABILITY
    )

    distance = compute_distance_from_main_sequence(coupling, abstractness=0.5)

    expect_equal(distance, 0.0)


# Tests: CouplingMetrics dataclass


def test_coupling_metrics_attributes() -> None:
    """CouplingMetrics has all expected attributes."""
    m = CouplingMetrics(
        afferent=HIGH_AFFERENT,
        efferent=MODERATE_EFFERENT,
        instability=MODERATE_INSTABILITY,
    )

    expect_equal(m.afferent, HIGH_AFFERENT)
    expect_equal(m.efferent, MODERATE_EFFERENT)
    expect_equal(m.instability, MODERATE_INSTABILITY)


def test_coupling_metrics_equality() -> None:
    """CouplingMetrics supports equality comparison."""
    m1 = CouplingMetrics(
        afferent=1,
        efferent=2,
        instability=HALF_INSTABILITY,
    )
    m2 = CouplingMetrics(
        afferent=1,
        efferent=2,
        instability=HALF_INSTABILITY,
    )

    expect_equal(m1, m2)


def test_coupling_metrics_frozen() -> None:
    """CouplingMetrics is frozen (immutable)."""
    m = CouplingMetrics(afferent=1, efferent=2, instability=0.5)

    assert_cannot_setattr(m, "afferent", 10)
