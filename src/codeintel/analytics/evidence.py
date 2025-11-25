"""Evidence dataclasses and helpers for analytics outputs."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from codeintel.analytics.ast_utils import snippet_from_lines

MAX_EVIDENCE_SAMPLES = 5


@dataclass(frozen=True)
class EvidenceSample:
    """Single evidence sample anchored to source."""

    path: str
    lineno: int | None
    end_lineno: int | None
    snippet: str
    details: dict[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dedup_key(self) -> tuple[str, int | None, int | None, str]:
        """
        Canonical key used for de-duplicating evidence samples.

        Returns
        -------
        tuple[str, int | None, int | None, str]
            Stable key composed of path, line bounds, and snippet.
        """
        return (self.path, self.lineno, self.end_lineno, self.snippet)

    @classmethod
    def from_ast(
        cls,
        *,
        path: str,
        lines: Sequence[str] | Iterable[str],
        node: ast.AST,
        details: dict[str, object] | None = None,
        tags: Iterable[str] | None = None,
    ) -> EvidenceSample:
        """
        Build a sample from an AST node with snippet capture.

        Parameters
        ----------
        path
            Repository-relative file path for the sample.
        lines
            Source lines for snippet extraction.
        node
            AST node carrying line metadata.
        details
            Optional structured details to attach.
        tags
            Optional tags for downstream consumers.

        Returns
        -------
        EvidenceSample
            Concrete sample populated from AST metadata.
        """
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", lineno)
        snippet = snippet_from_lines(lines, lineno, end_lineno)
        return cls(
            path=path,
            lineno=lineno,
            end_lineno=end_lineno,
            snippet=snippet,
            details=details or {},
            tags=tuple(tags or ()),
        )

    def to_dict(self) -> dict[str, object]:
        """
        Convert the sample to a JSON-serializable mapping.

        Returns
        -------
        dict[str, object]
            Mapping with path, line bounds, snippet, and optional details payload.
        """
        return {
            "path": self.path,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "snippet": self.snippet,
            "details": self.details or None,
            "tags": list(self.tags) or None,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    """Grouped evidence samples with optional tags."""

    samples: tuple[EvidenceSample, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the bundle to a JSON-serializable mapping.

        Returns
        -------
        dict[str, object]
            Mapping with bundled samples and optional tags.
        """
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "tags": list(self.tags) or None,
        }


@dataclass
class EvidenceCollector:
    """Collect evidence samples with max-sample enforcement."""

    max_samples: int = MAX_EVIDENCE_SAMPLES
    samples: list[EvidenceSample] = field(default_factory=list)
    _seen: set[tuple[str, int | None, int | None, str]] = field(
        init=False, default_factory=set, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize de-duplication set and enforce max sample cap."""
        for sample in self.samples[: self.max_samples]:
            self._seen.add(sample.dedup_key)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[: self.max_samples]

    def add(self, sample: EvidenceSample) -> None:
        """Add a sample if capacity and not already present."""
        if len(self.samples) >= self.max_samples:
            return
        key = sample.dedup_key
        if key in self._seen:
            return
        self._seen.add(key)
        self.samples.append(sample)

    def add_sample(
        self,
        *,
        path: str,
        line_span: tuple[int | None, int | None],
        snippet: str,
        details: dict[str, object] | None = None,
        tags: Iterable[str] | None = None,
    ) -> None:
        """
        Add a sample without constructing the dataclass manually.

        Parameters
        ----------
        path
            Repository-relative file path for the sample.
        line_span
            Tuple of starting and ending line numbers for the snippet.
        snippet
            Source snippet associated with the evidence.
        details
            Optional structured metadata to attach.
        tags
            Optional tags for downstream consumers.
        """
        lineno, end_lineno = line_span
        self.add(
            EvidenceSample(
                path=path,
                lineno=lineno,
                end_lineno=end_lineno,
                snippet=snippet,
                details=details or {},
                tags=tuple(tags or ()),
            )
        )

    def add_from_ast(
        self,
        *,
        path: str,
        lines: Sequence[str] | Iterable[str],
        node: ast.AST,
        details: dict[str, object] | None = None,
        tags: Iterable[str] | None = None,
    ) -> None:
        """Add a sample derived from an AST node and captured snippet."""
        self.add(
            EvidenceSample.from_ast(
                path=path,
                lines=lines,
                node=node,
                details=details,
                tags=tags,
            )
        )

    def extend(self, samples: Iterable[EvidenceSample]) -> None:
        """Add multiple samples with de-duplication."""
        for sample in samples:
            self.add(sample)

    def to_dicts(self) -> list[dict[str, object]]:
        """
        Return collected samples as JSON-serializable dicts.

        Returns
        -------
        list[dict[str, object]]
            Ordered collection of samples capped at `max_samples`.
        """
        return [sample.to_dict() for sample in self.samples]


EVIDENCE_SAMPLE_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EvidenceSample",
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "lineno": {"type": ["integer", "null"]},
        "end_lineno": {"type": ["integer", "null"]},
        "snippet": {"type": "string"},
        "details": {"type": ["object", "null"]},
        "tags": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
    },
    "required": ["path", "lineno", "end_lineno", "snippet"],
    "additionalProperties": False,
}

EVIDENCE_COLLECTION_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EvidenceCollection",
    "type": "array",
    "items": EVIDENCE_SAMPLE_SCHEMA,
}


def validate_evidence_samples(samples: Iterable[dict[str, object]]) -> None:
    """
    Lightweight validation for evidence payloads.

    Raises
    ------
    TypeError
        When a sample is not a mapping or has incorrectly typed fields.
    ValueError
        When a sample is missing required fields or has invalid types.
    """
    for sample in samples:
        if not isinstance(sample, dict):
            message = "Evidence sample must be a mapping"
            raise TypeError(message)
        for key in ("path", "lineno", "end_lineno", "snippet"):
            if key not in sample:
                message = f"Evidence sample missing required field: {key}"
                raise ValueError(message)
        if not isinstance(sample["path"], str):
            message = "Evidence sample 'path' must be a string"
            raise TypeError(message)
        snippet = sample["snippet"]
        if not isinstance(snippet, str):
            message = "Evidence sample 'snippet' must be a string"
            raise TypeError(message)
