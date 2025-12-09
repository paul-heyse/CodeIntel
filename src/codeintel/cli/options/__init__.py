"""Unified CLI option bundles for Cyclopts commands.

This package provides consolidated option dataclasses for Cyclopts commands,
replacing scattered RuntimeCLI, OutputFormatCLI, and other option classes.

Examples
--------
>>> from codeintel.cli.options import CommonOptions
>>> options = CommonOptions(verbose=2, output_format=OutputFormat.JSON)  # doctest: +SKIP
>>> params = options.to_params()  # doctest: +SKIP
"""

from __future__ import annotations

from codeintel.cli.options.common import CommonOptions

__all__ = [
    "CommonOptions",
]
