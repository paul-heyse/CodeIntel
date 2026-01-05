"""Immutable catalog derived from the Hamilton FunctionGraph."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from hamilton.caching import fingerprinting

from codeintel.build.hamilton.tag_spec import TagSpec
from codeintel.build.hamilton.tagging import required_dataset_node_tags, required_table_output_tags
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from codeintel.build.parameters import TargetParameters
    from codeintel.build.resources import TargetExecution, TargetResources
    from codeintel.build.targets import TargetModule


OutputKind = Literal["table", "artifact"]
OutputRole = Literal["contract", "internal"]
PreflightIssueKind = Literal[
    "layering_violation",
    "missing_contract",
    "missing_data_node",
    "missing_saver",
    "missing_tags",
]


@dataclass(frozen=True, slots=True)
class NodeDescriptor:
    """Hamilton node descriptor with parsed tags and dependencies."""

    name: str
    deps: tuple[str, ...]
    tags: Mapping[str, object]
    tag_spec: TagSpec | None = None
    module: str | None = None
    module_path: Path | None = None


@dataclass(frozen=True, slots=True)
class OutputDescriptor:
    """Single output declared by a data saver node."""

    kind: OutputKind
    key: str
    role: OutputRole
    producer_target: str
    saver_node: str
    sink: str
    artifact_path_template: str | None = None
    tags: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class DagPreflightIssue:
    """Issue discovered during DAG preflight validation."""

    kind: PreflightIssueKind
    node_name: str | None
    table_key: str | None
    message: str
    missing_tags: tuple[str, ...] = ()
    module: str | None = None
    module_path: Path | None = None

    def to_log_entry(self) -> dict[str, object]:
        """Return a structured log entry for this issue.

        Returns
        -------
        dict[str, object]
            Structured log entry payload.
        """
        payload: dict[str, object] = {
            "kind": self.kind,
            "message": self.message,
        }
        if self.node_name:
            payload["node_name"] = self.node_name
        if self.table_key:
            payload["table_key"] = self.table_key
        if self.missing_tags:
            payload["missing_tags"] = list(self.missing_tags)
        if self.module:
            payload["module"] = self.module
        if self.module_path:
            payload["module_path"] = str(self.module_path)
        return payload


@dataclass(frozen=True, slots=True)
class DagPreflightReport:
    """Summary of DAG preflight validation."""

    issues: tuple[DagPreflightIssue, ...]
    duration_ms: float

    @property
    def ok(self) -> bool:
        """Return True when no issues were found."""
        return not self.issues

    def log_entries(self) -> tuple[dict[str, object], ...]:
        """Return structured log entries for issues.

        Returns
        -------
        tuple[dict[str, object], ...]
            Structured log entry payloads.
        """
        return tuple(issue.to_log_entry() for issue in self.issues)

    def summary(self) -> str:
        """Return a concise summary string for error reporting.

        Returns
        -------
        str
            Human-friendly summary string.
        """
        if not self.issues:
            return "Preflight ok"
        parts = [issue.message for issue in self.issues]
        return "Preflight failed: " + "; ".join(parts)


@dataclass(frozen=True, slots=True)
class TableRead:
    """Table read derived from loader/dataset nodes."""

    table_key: str
    producer_target: str | None
    loader_node: str
    loader_type: str


@dataclass(frozen=True, slots=True)
class TableWrite:
    """Table write derived from data saver nodes."""

    table_key: str
    sink: str
    saver_node: str


@dataclass(frozen=True, slots=True)
class ArtifactWrite:
    """Artifact write derived from data saver nodes."""

    artifact_name: str
    sink: str
    saver_node: str


@dataclass(frozen=True, slots=True)
class IOSurface:
    """Read/write surface for a target."""

    target: str
    reads: tuple[TableRead, ...]
    table_writes: tuple[TableWrite, ...]
    artifact_writes: tuple[ArtifactWrite, ...]


@dataclass(frozen=True, slots=True)
class TargetDescriptor:
    """Immutable target descriptor derived from Hamilton tags."""

    name: str
    module: TargetModule
    anchor_node: str
    dependencies: tuple[str, ...]
    resources: TargetResources
    execution: TargetExecution
    parameters: TargetParameters
    description: str
    spec_version: str
    table_keys: tuple[str, ...] = ()
    artifact_names: tuple[str, ...] = ()

    @property
    def estimated_duration_ms(self) -> int:
        """Return estimated execution duration in milliseconds."""
        return self.execution.estimated_duration_ms()

    @property
    def domain(self) -> TargetModule:
        """Return the target domain for this descriptor."""
        return self.module


@dataclass(frozen=True, slots=True)
class DagCatalog:
    """Compiled view over the Hamilton FunctionGraph."""

    nodes: Mapping[str, NodeDescriptor]
    targets: Mapping[str, TargetDescriptor]
    target_nodes: Mapping[str, str]
    node_to_target: Mapping[str, str]
    target_dependencies: Mapping[str, tuple[str, ...]]
    target_dependents: Mapping[str, tuple[str, ...]]
    table_outputs: Mapping[str, OutputDescriptor]
    artifact_outputs: Mapping[str, OutputDescriptor]
    table_outputs_by_target: Mapping[str, tuple[OutputDescriptor, ...]]
    artifact_outputs_by_target: Mapping[str, tuple[OutputDescriptor, ...]]
    io_surfaces: Mapping[str, IOSurface]

    def __contains__(self, target_name: str) -> bool:
        """Return True if a target is present in the catalog.

        Parameters
        ----------
        target_name
            Target name to check.

        Returns
        -------
        bool
            True when the catalog contains the target.
        """
        return target_name in self.targets

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over target names.

        Returns
        -------
        Iterator[str]
            Iterator of target names in the catalog.
        """
        return iter(self.targets)

    def __len__(self) -> int:
        """Return the number of targets in the catalog.

        Returns
        -------
        int
            Count of targets.
        """
        return len(self.targets)

    @property
    def all_targets(self) -> tuple[TargetDescriptor, ...]:
        """Return all targets in deterministic order.

        Returns
        -------
        tuple[TargetDescriptor, ...]
            Targets sorted by name.
        """
        return tuple(self.targets[name] for name in sorted(self.targets))

    def target_node(self, target_name: str) -> str:
        """Return the anchor node name for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        str
            Hamilton anchor node name.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        node = self.target_nodes.get(target_name)
        if node is None:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return node

    def dependencies_of(self, target_name: str) -> tuple[str, ...]:
        """Return direct dependencies for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        tuple[str, ...]
            Direct dependencies for the target.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        if target_name not in self.targets:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return self.target_dependencies.get(target_name, ())

    def get(self, target_name: str) -> TargetDescriptor:
        """Return a target descriptor by name.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        TargetDescriptor
            Target metadata for the requested name.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        if target_name not in self.targets:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return self.targets[target_name]

    def get_target(self, target_name: str) -> TargetDescriptor | None:
        """Return a target descriptor by name, or None if missing.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        TargetDescriptor | None
            Target descriptor if present, otherwise None.
        """
        return self.targets.get(target_name)

    def dependents_of(self, target_name: str) -> tuple[str, ...]:
        """Return direct dependents for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        tuple[str, ...]
            Direct dependents for the target.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        if target_name not in self.targets:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return self.target_dependents.get(target_name, ())

    def closure(self, targets: Sequence[str]) -> tuple[str, ...]:
        """Return dependency closure in topological order.

        Parameters
        ----------
        targets
            Target names to compute the closure for.

        Returns
        -------
        tuple[str, ...]
            Dependency closure ordered by dependencies-first.

        Raises
        ------
        ValueError
            If a cycle is detected in the target dependencies.
        """
        all_names: set[str] = set()
        for name in targets:
            all_names.add(name)
            all_names.update(self._transitive_deps(name))

        in_degree: dict[str, int] = dict.fromkeys(all_names, 0)
        for name in all_names:
            for dep in self.target_dependencies.get(name, ()):
                if dep in all_names:
                    in_degree[name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            queue.sort()
            current = queue.pop(0)
            result.append(current)
            for dependent in self.target_dependents.get(current, ()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) != len(all_names):
            remaining = sorted(all_names - set(result))
            msg = f"Cycle detected in target dependencies: {remaining}"
            raise ValueError(msg)

        return tuple(result)

    def validate(self) -> tuple[str, ...]:
        """Validate catalog integrity.

        Returns
        -------
        tuple[str, ...]
            Validation error messages (empty when valid).
        """
        errors = [
            f"Target '{target_name}' depends on unknown target '{dep}'"
            for target_name, deps in self.target_dependencies.items()
            for dep in deps
            if dep not in self.targets
        ]
        if not errors:
            try:
                self.closure(tuple(self.targets))
            except ValueError as exc:
                errors.append(str(exc))
        return tuple(errors)

    def preflight_report(self, *, repo_root: Path | None = None) -> DagPreflightReport:
        """Run preflight validation for tag contracts and layering rules.

        Parameters
        ----------
        repo_root
            Optional repository root for resolving module import checks.

        Returns
        -------
        DagPreflightReport
            Report of any preflight issues.
        """
        start = time.perf_counter()
        issues: list[DagPreflightIssue] = []
        issues.extend(_preflight_output_tag_issues(self))
        issues.extend(_preflight_dataset_node_issues(self))
        if repo_root is not None:
            issues.extend(_preflight_layering_issues(self, repo_root=repo_root))
        duration_ms = (time.perf_counter() - start) * 1000
        return DagPreflightReport(issues=tuple(issues), duration_ms=duration_ms)

    def producer_of_table(self, table_key: str) -> str | None:
        """Return producer target for a table key.

        Parameters
        ----------
        table_key
            Fully-qualified table key.

        Returns
        -------
        str | None
            Producer target name when known, otherwise None.
        """
        output = self.table_outputs.get(table_key)
        if output is None:
            return None
        return output.producer_target

    def producer_of_artifact(self, artifact_name: str) -> str | None:
        """Return producer target for an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name to resolve.

        Returns
        -------
        str | None
            Producer target name when known, otherwise None.
        """
        output = self.artifact_outputs.get(artifact_name)
        if output is None:
            return None
        return output.producer_target

    def outputs_for_target(self, target_name: str) -> tuple[OutputDescriptor, ...]:
        """Return contract outputs for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        tuple[OutputDescriptor, ...]
            Contract outputs for the target.
        """
        tables = self.table_outputs_by_target.get(target_name, ())
        artifacts = self.artifact_outputs_by_target.get(target_name, ())
        return (*tables, *artifacts)

    def find_nodes(
        self,
        tag_key: str,
        tag_value: object | None = None,
    ) -> tuple[NodeDescriptor, ...]:
        """Find nodes by tag key/value.

        Parameters
        ----------
        tag_key
            Tag key to search for.
        tag_value
            Optional tag value to match exactly.

        Returns
        -------
        tuple[NodeDescriptor, ...]
            Nodes matching the tag criteria.
        """
        matches: list[NodeDescriptor] = []
        for node in self.nodes.values():
            if tag_key not in node.tags:
                continue
            if tag_value is None or node.tags.get(tag_key) == tag_value:
                matches.append(node)
        return tuple(matches)

    def targets_for_module(self, module: TargetModule) -> tuple[TargetDescriptor, ...]:
        """Return targets for a specific module.

        Parameters
        ----------
        module
            Target module name to filter by.

        Returns
        -------
        tuple[TargetDescriptor, ...]
            Targets in the requested module.
        """
        return tuple(t for t in self.all_targets if t.module == module)

    def all_table_keys(self) -> frozenset[str]:
        """Return all contract table keys.

        Returns
        -------
        frozenset[str]
            Set of all contract table keys.
        """
        return frozenset(self.table_outputs)

    def all_artifact_names(self) -> frozenset[str]:
        """Return all contract artifact names.

        Returns
        -------
        frozenset[str]
            Set of all contract artifact names.
        """
        return frozenset(self.artifact_outputs)

    def _transitive_deps(self, target_name: str) -> frozenset[str]:
        result: set[str] = set()
        stack = list(self.target_dependencies.get(target_name, ()))
        while stack:
            dep = stack.pop()
            if dep in result:
                continue
            result.add(dep)
            stack.extend(self.target_dependencies.get(dep, ()))
        return frozenset(result)


def freeze_mapping[T](mapping: Mapping[str, T]) -> Mapping[str, T]:
    """Return an immutable view of a mapping.

    Returns
    -------
    Mapping[str, T]
        Frozen mapping view.
    """
    return cast("Mapping[str, T]", MappingProxyType(dict(mapping)))


def _tag_value(tags: Mapping[str, object], key: str) -> str | None:
    value = tags.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _missing_required_tags(
    tags: Mapping[str, object],
    required: Iterable[str],
) -> tuple[str, ...]:
    missing = [key for key in required if _tag_value(tags, key) is None]
    return tuple(sorted(missing))


def _preflight_output_tag_issues(catalog: DagCatalog) -> list[DagPreflightIssue]:
    issues: list[DagPreflightIssue] = []
    required_tags = required_table_output_tags()
    for table_key, output in catalog.table_outputs.items():
        missing = _missing_required_tags(output.tags, required_tags)
        if missing:
            issues.append(
                DagPreflightIssue(
                    kind="missing_tags",
                    node_name=output.saver_node,
                    table_key=table_key,
                    message=(
                        f"Saver node {output.saver_node} missing required tags: "
                        f"{', '.join(missing)}"
                    ),
                    missing_tags=missing,
                )
            )
        data_node = _tag_value(output.tags, "ci.data_node")
        if data_node is None:
            issues.append(
                DagPreflightIssue(
                    kind="missing_data_node",
                    node_name=output.saver_node,
                    table_key=table_key,
                    message=f"Saver node {output.saver_node} missing ci.data_node tag",
                )
            )
            continue
        node = catalog.nodes.get(data_node)
        if node is None:
            issues.append(
                DagPreflightIssue(
                    kind="missing_data_node",
                    node_name=output.saver_node,
                    table_key=table_key,
                    message=f"Saver node {output.saver_node} references missing data node {data_node}",
                )
            )
            continue
        node_type = node.tags.get(ht.TAG_NODE_TYPE)
        if node_type != ht.NODE_TYPE_DATASET:
            issues.append(
                DagPreflightIssue(
                    kind="missing_data_node",
                    node_name=output.saver_node,
                    table_key=table_key,
                    message=(
                        f"Saver node {output.saver_node} references non-dataset node {data_node}"
                    ),
                )
            )
        node_table_key = _tag_value(node.tags, ht.TAG_TABLE_KEY)
        if node_table_key and node_table_key != table_key:
            issues.append(
                DagPreflightIssue(
                    kind="missing_data_node",
                    node_name=output.saver_node,
                    table_key=table_key,
                    message=(
                        f"Saver node {output.saver_node} data node {data_node} "
                        f"uses table_key {node_table_key}"
                    ),
                )
            )
    return issues


def _preflight_dataset_node_issues(catalog: DagCatalog) -> list[DagPreflightIssue]:
    issues: list[DagPreflightIssue] = []
    required_tags = required_dataset_node_tags()
    for node in catalog.nodes.values():
        if node.tags.get(ht.TAG_NODE_TYPE) != ht.NODE_TYPE_DATASET:
            continue
        table_key = _tag_value(node.tags, ht.TAG_TABLE_KEY)
        missing = _missing_required_tags(node.tags, required_tags)
        if missing:
            issues.append(
                DagPreflightIssue(
                    kind="missing_tags",
                    node_name=node.name,
                    table_key=table_key,
                    message=f"Dataset node {node.name} missing required tags: {', '.join(missing)}",
                    missing_tags=missing,
                )
            )
        if table_key is None:
            continue
        output = catalog.table_outputs.get(table_key)
        if output is None:
            issues.append(
                DagPreflightIssue(
                    kind="missing_contract",
                    node_name=node.name,
                    table_key=table_key,
                    message=f"Dataset node {node.name} missing contract output for {table_key}",
                )
            )
            continue
        data_node = _tag_value(output.tags, "ci.data_node")
        if data_node is None or data_node != node.name:
            issues.append(
                DagPreflightIssue(
                    kind="missing_saver",
                    node_name=node.name,
                    table_key=table_key,
                    message=(
                        f"Dataset node {node.name} not wired to saver for {table_key}; "
                        f"ci.data_node={data_node}"
                    ),
                )
            )
    return issues


def _preflight_layering_issues(
    catalog: DagCatalog,
    *,
    repo_root: Path,
) -> list[DagPreflightIssue]:
    issues: list[DagPreflightIssue] = []
    module_paths: dict[Path, list[NodeDescriptor]] = {}
    for node in catalog.nodes.values():
        module_path = node.module_path
        if module_path is None:
            continue
        module_paths.setdefault(module_path, []).append(node)
    for path, nodes in module_paths.items():
        if not _is_build_module_path(path, repo_root=repo_root):
            continue
        forbidden = _find_forbidden_imports(path)
        if not forbidden:
            continue
        node_names = sorted({node.name for node in nodes})
        module_name = nodes[0].module if nodes else None
        message = f"Module imports forbidden namespaces: {', '.join(sorted(forbidden))}"
        issues.append(
            DagPreflightIssue(
                kind="layering_violation",
                node_name=node_names[0] if node_names else None,
                table_key=None,
                message=message,
                module=module_name,
                module_path=path,
            )
        )
    return issues


def _is_build_module_path(path: Path, *, repo_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return relative.parts[:3] == ("src", "codeintel", "build")


def _find_forbidden_imports(path: Path) -> set[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    scanner = _ForbiddenImportScanner()
    scanner.visit(tree)
    return scanner.found


def _is_type_checking_guard(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING"
    )


class _ForbiddenImportScanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found: set[str] = set()
        self._type_checking_depth = 0

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            self._type_checking_depth += 1
            self.generic_visit(node)
            self._type_checking_depth -= 1
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self._type_checking_depth:
            return
        for alias in node.names:
            if _is_forbidden_import(alias.name):
                self.found.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._type_checking_depth:
            return
        if node.level:
            return
        if node.module and _is_forbidden_import(node.module):
            self.found.add(node.module)


def _is_forbidden_import(module_name: str) -> bool:
    return module_name.startswith(("codeintel.storage", "codeintel.serving"))


def register_dag_catalog_hashing() -> None:
    """Register deterministic hashing for DAG catalogs."""

    @fingerprinting.hash_value.register(DagCatalog)
    def _hash_dag_catalog(
        value: DagCatalog,
        *args: object,
        **kwargs: object,
    ) -> str:
        _ = (value, args, kwargs)
        return fingerprinting.hash_value("codeintel.dag_catalog")


register_dag_catalog_hashing()


__all__ = [
    "ArtifactWrite",
    "DagCatalog",
    "DagPreflightIssue",
    "DagPreflightReport",
    "IOSurface",
    "NodeDescriptor",
    "OutputDescriptor",
    "OutputKind",
    "OutputRole",
    "TableRead",
    "TableWrite",
    "TargetDescriptor",
    "freeze_mapping",
]
