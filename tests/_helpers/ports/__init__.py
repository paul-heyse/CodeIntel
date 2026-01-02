"""Port protocols for hexagonal test architecture.

This package defines protocols (interfaces) for external systems that tests
interact with. By coding to these protocols, tests remain decoupled from
specific implementations while still using real production technologies.

The ports follow the hexagonal architecture pattern:
- Tests depend on port protocols (abstractions)
- Real implementations (adapters) satisfy these protocols
- Per the Testing Charter, adapters use real technology (DuckDB, filesystem)

Available Ports
---------------
GatewayPort
    Protocol for storage gateway operations (database access).
RepoPort
    Protocol for filesystem repository operations.
ToolingPort
    Protocol for external tool runners (type checkers, linters).
"""

from __future__ import annotations

from tests._helpers.ports.gateway import GatewayPort
from tests._helpers.ports.repo import RepoPort
from tests._helpers.ports.tooling import ToolingPort

__all__ = [
    "GatewayPort",
    "RepoPort",
    "ToolingPort",
]
