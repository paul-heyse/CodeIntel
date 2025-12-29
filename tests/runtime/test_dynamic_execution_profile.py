"""Dynamic execution configuration tests for compose helpers."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.runtime import compose as runtime_compose


@dataclass
class DummyBuilder:
    """Builder stub used to validate dynamic execution wiring."""

    enabled: bool = False
    local_executor: object | None = None
    remote_executor: object | None = None
    enabled_calls: int = 0
    local_calls: int = 0
    remote_calls: int = 0

    def enable_dynamic_execution(self, *, allow_experimental_mode: bool = False) -> DummyBuilder:
        """Enable dynamic execution for the stub builder.

        Returns
        -------
        DummyBuilder
            Updated builder instance.
        """
        self.enabled_calls += 1
        self.enabled = allow_experimental_mode
        return self

    def with_local_executor(self, local_executor: object) -> DummyBuilder:
        """Attach a local executor to the stub builder.

        Returns
        -------
        DummyBuilder
            Updated builder instance.
        """
        self.local_calls += 1
        self.local_executor = local_executor
        return self

    def with_remote_executor(self, remote_executor: object) -> DummyBuilder:
        """Attach a remote executor to the stub builder.

        Returns
        -------
        DummyBuilder
            Updated builder instance.
        """
        self.remote_calls += 1
        self.remote_executor = remote_executor
        return self


def test_dynamic_execution_gated_off() -> None:
    """Skip dynamic execution wiring when disabled."""
    builder = DummyBuilder()
    config = runtime_compose.DynamicExecutionConfig(
        enabled=False,
        local_executor="local",
        remote_executor="remote",
    )
    result = runtime_compose.apply_dynamic_execution(builder=builder, config=config)
    assert result is builder
    assert builder.enabled is False
    assert builder.local_executor is None
    assert builder.remote_executor is None
    assert builder.enabled_calls == 0


def test_dynamic_execution_wires_executors() -> None:
    """Wire local and remote executors when enabled."""
    builder = DummyBuilder()
    config = runtime_compose.DynamicExecutionConfig(
        enabled=True,
        local_executor="local",
        remote_executor="remote",
    )
    result = runtime_compose.apply_dynamic_execution(builder=builder, config=config)
    assert result is builder
    assert builder.enabled is True
    assert builder.local_executor == "local"
    assert builder.remote_executor == "remote"
    assert builder.enabled_calls == 1
    assert builder.local_calls == 1
    assert builder.remote_calls == 1
