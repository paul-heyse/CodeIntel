"""Dataset operations command group.

Note: Dataset ops commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from cyclopts import App, Parameter

from codeintel.cli.commands._common import SHARED_FLAGS_METADATA, SharedFlags
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.ops import (
    dataset_constraints_handler,
    dataset_describe_handler,
    dataset_flow_handler,
    dataset_info_handler,
    dataset_list_handler,
    dataset_verify_handler,
)

dataset_app = App(
    name="dataset",
    help="Dataset inspection commands.",
)

# Config for dataset commands - no runtime needed for listing/describing
_DATASET_NO_RUNTIME_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)
# Config for dataset verify - requires runtime
_DATASET_RUNTIME_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)


@cli_command("dataset.list", handler=dataset_list_handler, config=_DATASET_NO_RUNTIME_CONFIG)
@dataset_app.command(name="list")
@dataclass
class DatasetListCommand:
    """List datasets from the registry."""

    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command(
    "dataset.describe", handler=dataset_describe_handler, config=_DATASET_NO_RUNTIME_CONFIG
)
@dataset_app.command(name="describe")
@dataclass
class DatasetDescribeCommand:
    """Show contract details for a dataset."""

    table_key: Annotated[
        str,
        Parameter(
            name=None,
            help="Dataset table key (e.g., 'core.goids').",
        ),
    ]
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("dataset.verify", handler=dataset_verify_handler, config=_DATASET_RUNTIME_CONFIG)
@dataset_app.command(name="verify")
@dataclass
class DatasetVerifyCommand:
    """Verify dataset contracts against actual data."""

    table_key: Annotated[
        str | None,
        Parameter(
            name=None,
            help="Dataset table key to verify (verifies all if not specified).",
        ),
    ] = None
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("dataset.info", handler=dataset_info_handler, config=_DATASET_NO_RUNTIME_CONFIG)
@dataset_app.command(name="info")
@dataclass
class DatasetInfoCommand:
    """Show comprehensive schema information for a dataset.

    Display detailed schema information including:
    - Column names and types from the Pandera schema
    - Dataset metadata (owner, SLA, tags, etc.)
    - JSON Schema representation

    Requires the dataset to have a registered Pandera schema.
    """

    table_key: Annotated[
        str,
        Parameter(
            name=None,
            help="Dataset table key (e.g., 'analytics.function_metrics').",
        ),
    ]
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command("dataset.flow", handler=dataset_flow_handler, config=_DATASET_NO_RUNTIME_CONFIG)
@dataset_app.command(name="flow")
@dataclass
class DatasetFlowCommand:
    """Show producer/consumer graph for a dataset.

    Display which plugins produce and consume the specified dataset.
    Useful for understanding data dependencies and lineage.
    """

    table_key: Annotated[
        str,
        Parameter(
            name=None,
            help="Dataset table key (e.g., 'analytics.function_metrics').",
        ),
    ]
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


@cli_command(
    "dataset.constraints", handler=dataset_constraints_handler, config=_DATASET_NO_RUNTIME_CONFIG
)
@dataset_app.command(name="constraints")
@dataclass
class DatasetConstraintsCommand:
    """Show constraint summary for a dataset.

    Display all constraints extracted from the Pandera schema including:
    - Column type constraints
    - Nullability constraints
    - Range constraints (e.g., non-negative values)
    - Table-level cross-column constraints

    Useful for understanding data validation rules and schema structure.
    """

    table_key: Annotated[
        str,
        Parameter(
            name=None,
            help="Dataset table key (e.g., 'analytics.function_metrics').",
        ),
    ]
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)


__all__ = [
    "DatasetConstraintsCommand",
    "DatasetDescribeCommand",
    "DatasetFlowCommand",
    "DatasetInfoCommand",
    "DatasetListCommand",
    "DatasetVerifyCommand",
    "dataset_app",
]
