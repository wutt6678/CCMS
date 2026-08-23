"""Base adapter protocol for source dataset normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from causal_mllm.data.schemas import CanonicalSourceExample


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
        self, split: str | None = None, max_examples: int | None = None
    ) -> list[CanonicalSourceExample]:
        """Convenience: load and normalize examples.

        Args:
            split: Dataset split.
            max_examples: Maximum number of examples to return.

        Returns:
            List of normalized CanonicalSourceExample instances.
        """
        results: list[CanonicalSourceExample] = []
        for raw in self.load(split):
            try:
                example = self.normalize(raw)
                results.append(example)
            except (ValueError, KeyError) as e:
                # Log and skip malformed examples
                from causal_mllm.data.logging import get_logger
                get_logger("causal_mllm.adapters").warning(
                    "Skipping malformed example from %s: %s", self.source_name, e
                )
            if max_examples and len(results) >= max_examples:
                break
        return results
