"""Lightweight operation context stubs for type checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OpConfig:
    """Placeholder for operation configuration."""

    settings: dict[str, Any] | None = None


@dataclass
class OpContext:
    """Minimal execution context used by operations."""

    config: OpConfig | None = None


class OpContextBuilder:
    """Builder for OpContext used in tests and type checking."""

    def __init__(self, config: OpConfig | None = None) -> None:
        self._config = config

    def with_config(self, config: OpConfig) -> OpContextBuilder:
        """Attach configuration and return self.

        Returns
        -------
        OpContextBuilder
            Builder with configuration applied.
        """
        self._config = config
        return self

    def build(self) -> OpContext:
        """Build an OpContext instance.

        Returns
        -------
        OpContext
            Constructed operation context.
        """
        return OpContext(config=self._config)


__all__ = ["OpConfig", "OpContext", "OpContextBuilder"]
