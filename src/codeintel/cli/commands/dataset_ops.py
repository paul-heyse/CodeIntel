"""Dataset operations command group.

Note: Dataset ops commands require runtime/gateway access via handler pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from cyclopts import App

from codeintel.build.schemas.dataset_service import DocsFilterMode, ReadOnlyFilterMode
from codeintel.cli.commands.decorators import CommandConfig, cli_command
from codeintel.cli.handlers.ops import (
    dataset_constraints_handler,
    dataset_describe_handler,
    dataset_flow_handler,
    dataset_info_handler,
    dataset_list_handler,
    dataset_verify_handler,
)
from codeintel.cli.options.registry import (
    DATASET_TABLE_KEY,
    DATASETS_DOCS_VIEW,
    DATASETS_MAX_DESCRIPTION,
    DATASETS_READ_ONLY,
)
from codeintel.cli.options.shared_flags import SharedFlagsProtocol, shared_flags_field
from codeintel.cli.options.types import CommandPath, option_param

dataset_app = App(
    name="dataset",
    help="Dataset inspection commands.",
)


_DATASET_NO_RUNTIME_CONFIG = CommandConfig(require_runtime=False, require_gateway=False)

_DATASET_RUNTIME_CONFIG = CommandConfig(require_runtime=True, require_gateway=True)

DATASET_LIST_PATH: CommandPath = ("dataset", "list")
DATASET_DESCRIBE_PATH: CommandPath = ("dataset", "describe")
DATASET_VERIFY_PATH: CommandPath = ("dataset", "verify")
DATASET_INFO_PATH: CommandPath = ("dataset", "info")
DATASET_FLOW_PATH: CommandPath = ("dataset", "flow")
DATASET_CONSTRAINTS_PATH: CommandPath = ("dataset", "constraints")

_DATASET_LIST_FLAGS_FIELD = shared_flags_field(DATASET_LIST_PATH)
_DATASET_DESCRIBE_FLAGS_FIELD = shared_flags_field(DATASET_DESCRIBE_PATH)
_DATASET_VERIFY_FLAGS_FIELD = shared_flags_field(DATASET_VERIFY_PATH)
_DATASET_INFO_FLAGS_FIELD = shared_flags_field(DATASET_INFO_PATH)
_DATASET_FLOW_FLAGS_FIELD = shared_flags_field(DATASET_FLOW_PATH)
_DATASET_CONSTRAINTS_FLAGS_FIELD = shared_flags_field(DATASET_CONSTRAINTS_PATH)


@cli_command("dataset.list", handler=dataset_list_handler, config=_DATASET_NO_RUNTIME_CONFIG)
@dataset_app.command(name="list")
@dataclass
class DatasetListCommand:
    """List datasets from the registry with optional filters."""

    docs_view: Annotated[
        DocsFilterMode,
        option_param(DATASETS_DOCS_VIEW, command_path=DATASET_LIST_PATH),
    ] = "include"
    read_only: Annotated[
        ReadOnlyFilterMode,
        option_param(DATASETS_READ_ONLY, command_path=DATASET_LIST_PATH),
    ] = "include"
    max_description: Annotated[
        int,
        option_param(DATASETS_MAX_DESCRIPTION, command_path=DATASET_LIST_PATH),
    ] = 80
    flags: SharedFlagsProtocol = _DATASET_LIST_FLAGS_FIELD


@cli_command(
    "dataset.describe", handler=dataset_describe_handler, config=_DATASET_NO_RUNTIME_CONFIG
)
@dataset_app.command(name="describe")
@dataclass
class DatasetDescribeCommand:
    """Show contract details for a dataset."""

    table_key: Annotated[
        str,
        option_param(DATASET_TABLE_KEY, command_path=DATASET_DESCRIBE_PATH),
    ]
    flags: SharedFlagsProtocol = _DATASET_DESCRIBE_FLAGS_FIELD


@cli_command("dataset.verify", handler=dataset_verify_handler, config=_DATASET_RUNTIME_CONFIG)
@dataset_app.command(name="verify")
@dataclass
class DatasetVerifyCommand:
    """Verify dataset contracts against actual data."""

    table_key: Annotated[
        str | None,
        option_param(DATASET_TABLE_KEY, command_path=DATASET_VERIFY_PATH),
    ] = None
    flags: SharedFlagsProtocol = _DATASET_VERIFY_FLAGS_FIELD


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
        option_param(DATASET_TABLE_KEY, command_path=DATASET_INFO_PATH),
    ]
    flags: SharedFlagsProtocol = _DATASET_INFO_FLAGS_FIELD


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
        option_param(DATASET_TABLE_KEY, command_path=DATASET_FLOW_PATH),
    ]
    flags: SharedFlagsProtocol = _DATASET_FLOW_FLAGS_FIELD


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
        option_param(DATASET_TABLE_KEY, command_path=DATASET_CONSTRAINTS_PATH),
    ]
    flags: SharedFlagsProtocol = _DATASET_CONSTRAINTS_FLAGS_FIELD


__all__ = [
    "DatasetConstraintsCommand",
    "DatasetDescribeCommand",
    "DatasetFlowCommand",
    "DatasetInfoCommand",
    "DatasetListCommand",
    "DatasetVerifyCommand",
    "dataset_app",
]
