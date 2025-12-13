"""Subsystem metrics plugins.

Compute graph metrics and agreement for subsystems.
"""

from __future__ import annotations

from codeintel.build.plugins.analytics.subsystem_metrics.agreement import (
    SubsystemAgreementPlugin,
)
from codeintel.build.plugins.analytics.subsystem_metrics.graph_metrics import (
    SubsystemGraphMetricsPlugin,
)

__all__ = ["SubsystemAgreementPlugin", "SubsystemGraphMetricsPlugin"]
