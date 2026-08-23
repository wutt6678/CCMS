"""Base adapter protocol for source dataset normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from causal_mllm.data.schemas import CanonicalSourceExample


@dataclass
class NormalizationRejection:
    """Record of a rejected source row during normalization.

    No row should ever disappear from the dataset without an
    explicit rejection record.
    """
    source_id: str
    stage: str  # "normalization" | "media" | "schema"
    error_type: str  # e.g. "MediaLoadError", "ValueError", "KeyError"
    reason: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "stage": self.stage,
            "error_type": self.error_type,
            "reason": self.reason,
        }


# Valid on_error modes
OnErrorMode = Literal["raise", "record", "skip"]


class DatasetAdapter(ABC):
    """Abstract base class for source dataset adapters.

    Every adapter must implement load() and normalize() to convert
    raw source examples into the canonical schema.
    """

    @abstractmethod
    def load(self, split: str | None = None) -> Iterator[dict]:
        """Load raw examples from the source dataset.

        Args:
            split: Dataset split to load (e.g., 'train', 'test').
                   If None, load the default/preferred split.

        Yields:
            Raw dictionaries from the source dataset.
        """
        ...

    @abstractmethod
    def normalize(self, raw_example: dict) -> CanonicalSourceExample:
        """Normalize a raw source example into the canonical schema.

        Args:
            raw_example: A single raw record from the source dataset.

        Returns:
            A CanonicalSourceExample with all fields populated.

        Raises:
            ValueError: If the example is malformed or missing required fields.
        """
        ...

    @abstractmethod
    def inspect_schema(self, n: int = 20) -> dict[str, Any]:
        """Inspect the source dataset schema without normalization.

        Examines up to n rows to identify field names, types, and structure.

        Args:
            n: Number of rows to inspect.

        Returns:
            A schema report dictionary with field descriptions, types,
            and example values.
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier for this source (e.g., 'mtmcs')."""
        ...

    def load_and_normalize(
        self,
        split: str | None = None,
        *,
        max_examples: int | None = None,
        on_error: OnErrorMode = "raise",
        rejections: list[NormalizationRejection] | None = None,
    ) -> list[CanonicalSourceExample]:
        """Load and normalize examples.

        Args:
            split: Dataset split.
            max_examples: Maximum number of examples to return.
            on_error: Error handling mode:
                "raise" (default): re-raise immediately. Fail-closed.
                    Use for dataset construction.
                "record": append a NormalizationRejection to ``rejections``
                    and continue. Every rejected row is accounted for.
                    Use for candidate selection with rejection manifests.
                "skip": log a warning and continue. Only for one-off
                    inspection. Not suitable for research data.
            rejections: Output list for rejected rows (only used when
                on_error="record"). Must be provided by the caller.

        Returns:
            List of normalized CanonicalSourceExample instances.

        Raises:
            ValueError: If on_error="record" but rejections list not provided.
            Exception: If on_error="raise" and a row fails normalization.
        """
        if on_error == "record" and rejections is None:
            raise ValueError(
                "on_error='record' requires a rejections list to be provided"
            )

        results: list[CanonicalSourceExample] = []
        for raw in self.load(split):
            try:
                example = self.normalize(raw)
                results.append(example)
            except Exception as e:
                source_id = str(raw.get("source_id", raw.get("id", "unknown")))
                error_type = type(e).__name__

                if on_error == "raise":
                    raise

                if on_error == "record":
                    rejections.append(NormalizationRejection(
                        source_id=source_id,
                        stage="normalization",
                        error_type=error_type,
                        reason=str(e),
                    ))
                else:  # skip
                    from causal_mllm.data.logging import get_logger
                    get_logger("causal_mllm.adapters").warning(
                        "Skipping malformed example from %s (id=%s): [%s] %s",
                        self.source_name, source_id, error_type, e,
                    )

            if max_examples and len(results) >= max_examples:
                break

        return results
