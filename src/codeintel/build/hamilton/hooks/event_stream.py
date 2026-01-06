"""Lifecycle event stream hook for Hamilton execution."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from hamilton.lifecycle import base as lifecycle_base

if TYPE_CHECKING:
    from hamilton.graph import FunctionGraph
    from hamilton.node import Node

log = logging.getLogger(__name__)


class LifecycleEventStreamHook(
    lifecycle_base.BasePostGraphConstruct,
    lifecycle_base.BasePreGraphExecute,
    lifecycle_base.BasePostGraphExecute,
    lifecycle_base.BasePreNodeExecute,
    lifecycle_base.BasePostNodeExecute,
    lifecycle_base.BasePreTaskSubmission,
    lifecycle_base.BasePreTaskExecute,
    lifecycle_base.BasePostTaskExecute,
    lifecycle_base.BasePostTaskReturn,
    lifecycle_base.BasePostTaskExpand,
    lifecycle_base.BasePostTaskGroup,
):
    """Emit a JSONL event stream for Hamilton lifecycle events."""

    def __init__(self, *, run_id: str, output_path: Path) -> None:
        self._run_id = run_id
        self._output_path = output_path
        self._lock = threading.Lock()

    def post_graph_construct(
        self,
        *,
        graph: FunctionGraph,
        modules: list[ModuleType],
        config: dict[str, object],
    ) -> None:
        """Emit event after graph construction."""
        self._emit(
            "graph_construct",
            module_count=len(modules),
            config_keys=sorted(str(key) for key in config),
            node_count=len(getattr(graph, "nodes", {})),
        )

    def pre_graph_execute(
        self,
        *,
        run_id: str,
        graph: FunctionGraph,
        final_vars: list[str],
        inputs: dict[str, object],
        overrides: dict[str, object],
    ) -> None:
        """Emit event before graph execution."""
        self._emit(
            "graph_execute_start",
            run_id=run_id,
            final_vars=list(final_vars),
            input_keys=sorted(inputs.keys()),
            override_keys=sorted(overrides.keys()),
            node_count=len(getattr(graph, "nodes", {})),
        )

    def post_graph_execute(
        self,
        *,
        run_id: str,
        graph: FunctionGraph,
        success: bool,
        error: Exception | None,
        results: dict[str, object] | None,
    ) -> None:
        """Emit event after graph execution."""
        _ = graph
        self._emit(
            "graph_execute_end",
            run_id=run_id,
            success=success,
            error=str(error) if error else None,
            result_keys=sorted(results.keys()) if isinstance(results, dict) else None,
        )

    def pre_node_execute(
        self,
        *,
        run_id: str,
        node_: Node,
        kwargs: dict[str, object],
        task_id: str | None = None,
    ) -> None:
        """Emit event before node execution."""
        self._emit(
            "node_execute_start",
            run_id=run_id,
            node_name=node_.name,
            task_id=task_id,
            input_keys=sorted(kwargs.keys()),
        )

    def post_node_execute(
        self,
        *,
        run_id: str,
        node_: Node,
        success: bool,
        task_id: str | None = None,
        **context: object,
    ) -> None:
        """Emit event after node execution."""
        error_obj = context.get("error")
        error = error_obj if isinstance(error_obj, Exception) else None
        self._emit(
            "node_execute_end",
            run_id=run_id,
            node_name=node_.name,
            task_id=task_id,
            success=success,
            error=str(error) if error else None,
        )

    def pre_task_submission(
        self,
        *,
        run_id: str,
        task_id: str,
        nodes: list[Node],
        spawning_task_id: str | None,
        **_context: object,
    ) -> None:
        """Emit event before task submission."""
        self._emit(
            "task_submit",
            run_id=run_id,
            task_id=task_id,
            spawning_task_id=spawning_task_id,
            node_names=[node.name for node in nodes],
        )

    def pre_task_execute(
        self,
        *,
        run_id: str,
        task_id: str,
        nodes: list[Node],
        spawning_task_id: str | None,
        **_context: object,
    ) -> None:
        """Emit event before task execution."""
        self._emit(
            "task_execute_start",
            run_id=run_id,
            task_id=task_id,
            spawning_task_id=spawning_task_id,
            node_names=[node.name for node in nodes],
        )

    def post_task_execute(
        self,
        *,
        run_id: str,
        task_id: str,
        nodes: list[Node],
        success: bool,
        spawning_task_id: str | None,
        **context: object,
    ) -> None:
        """Emit event after task execution."""
        error_obj = context.get("error")
        error = error_obj if isinstance(error_obj, Exception) else None
        self._emit(
            "task_execute_end",
            run_id=run_id,
            task_id=task_id,
            spawning_task_id=spawning_task_id,
            node_names=[node.name for node in nodes],
            success=success,
            error=str(error) if error else None,
        )

    def post_task_return(
        self,
        *,
        run_id: str,
        task_id: str,
        nodes: list[Node],
        success: bool,
        spawning_task_id: str | None,
        **context: object,
    ) -> None:
        """Emit event after task return."""
        error_obj = context.get("error")
        error = error_obj if isinstance(error_obj, Exception) else None
        self._emit(
            "task_return",
            run_id=run_id,
            task_id=task_id,
            spawning_task_id=spawning_task_id,
            node_names=[node.name for node in nodes],
            success=success,
            error=str(error) if error else None,
        )

    def post_task_expand(
        self,
        *,
        run_id: str,
        task_id: str,
        parameters: dict[str, object],
    ) -> None:
        """Emit event after task expansion."""
        self._emit(
            "task_expand",
            run_id=run_id,
            task_id=task_id,
            parameter_keys=sorted(parameters.keys()),
        )

    def post_task_group(
        self,
        *,
        run_id: str,
        task_ids: list[str],
    ) -> None:
        """Emit event after task grouping."""
        self._emit(
            "task_group",
            run_id=run_id,
            task_ids=list(task_ids),
        )

    def _emit(self, event: str, **fields: object) -> None:
        payload: dict[str, object] = {
            "event": event,
            "run_id": self._run_id,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        for key, value in fields.items():
            if value is None:
                continue
            payload[key] = value
        self._write(payload)

    def _write(self, payload: dict[str, object]) -> None:
        line = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        try:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self._output_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{line}\n")
        except OSError as exc:
            log.warning("build.telemetry.event_stream_write_failed error=%s", exc)


__all__ = ["LifecycleEventStreamHook"]
