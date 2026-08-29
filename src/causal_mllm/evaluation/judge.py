"""Response judge protocol and backends (Iteration 9).

The judge evaluates a MODEL RESPONSE, not a trajectory.  The variant
name is deliberately NOT passed — the judge sees only the conversation
context (system prompt, history messages, terminal query) and the
response text.  This keeps the judgment variant-blind.

Backends:
  * CallableResponseJudge — wraps a callable (for LLM judges later)
  * HumanLabelJudge — loads pre-existing human labels from JSON
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from causal_mllm.evaluation.errors import EvaluationError
from causal_mllm.evaluation.schema import validate_judgment


@runtime_checkable
class ResponseJudge(Protocol):
    """Protocol: judge a single model response.

    The variant name is NOT passed — the judge sees only the
    conversation context and the model response.
    """

    def judge(
        self,
        system_prompt: str,
        history_messages: list[dict],
        terminal_query: str,
        response: str,
    ) -> dict:
        """Return a judgment dict matching JUDGE_FIELDS."""
        ...

    def provenance(self) -> dict:
        """Return judge provenance metadata."""
        ...


class CallableResponseJudge:
    """Wraps any callable (LLM/VLM) as a response judge.

    The callable receives (system_prompt, history_messages,
    terminal_query, response) and returns a dict with at least the
    JUDGE_FIELDS.  Model identity is mandatory provenance.
    """

    def __init__(
        self,
        fn: Callable[[str, list[dict], str, str], dict],
        *,
        model_name: str,
        model_revision: str | None = None,
        prompt_version: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ):
        if not model_name:
            raise ValueError(
                "CallableResponseJudge requires model_name for "
                "provenance")
        self._fn = fn
        self._provenance = {
            "backend": "callable",
            "model": model_name,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "seed": seed,
        }

    def judge(
        self,
        system_prompt: str,
        history_messages: list[dict],
        terminal_query: str,
        response: str,
    ) -> dict:
        raw = self._fn(system_prompt, history_messages,
                        terminal_query, response)
        return validate_judgment(raw)

    def provenance(self) -> dict:
        return dict(self._provenance)


class HumanLabelJudge:
    """Loads human labels from a JSON file.

    Supports two file schemas:

    **Wrapped** (produced by ``save_human_labels``)::

        {
          "labels": {
            "<family_id>": {
              "<variant>": { "refusal_type": ..., ... }
            }
          },
          "provenance": {
            "labels_sha256": "...",
            "rubric_version": "...",
            "annotator_id": "...",
            "adjudicated": false
          }
        }

    **Raw** (legacy flat format)::

        {
          "<family_id>": {
            "<variant>": { "refusal_type": ..., ... }
          }
        }

    The wrapped format is verified: label SHA256 must match, and
    completeness (120 labels), rubric, annotator, and adjudication
    metadata are exposed via ``provenance()``.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"Human labels file not found: {self._path}")
        with self._path.open(encoding="utf-8") as f:
            raw: dict = json.load(f)

        self._provenance_meta: dict = {}
        self._wrapped = False

        if "labels" in raw:
            # Wrapped schema from save_human_labels
            self._wrapped = True
            labels_data = raw["labels"]
            prov = raw.get("provenance", {})

            # Verify labels SHA256
            labels_json = json.dumps(
                labels_data, sort_keys=True, ensure_ascii=False)
            actual_sha = hashlib.sha256(
                labels_json.encode("utf-8")).hexdigest()
            expected_sha = prov.get("labels_sha256", "")
            if expected_sha and actual_sha != expected_sha:
                raise EvaluationError(
                    f"human labels SHA256 mismatch in {self._path}: "
                    f"expected {expected_sha}, got {actual_sha}")

            self._provenance_meta = {
                "labels_sha256": actual_sha,
                "label_file_sha256": prov.get("label_file_sha256"),
                "rubric_version": prov.get("rubric_version"),
                "annotator_id": prov.get("annotator_id"),
                "adjudicated": prov.get("adjudicated", False),
                "n_families": prov.get("n_families", len(labels_data)),
                "n_labels": prov.get("n_labels"),
            }
        else:
            # Raw / legacy format
            labels_data = raw

        # Build a flat lookup: (family_id, variant) -> label
        self._lookup: dict[tuple[str, str], dict] = {}
        for family_id, variants in labels_data.items():
            for variant, label in variants.items():
                self._lookup[(family_id, variant)] = label

    def judge_for(
        self,
        family_id: str,
        variant: str,
    ) -> dict:
        """Look up the human label for a (family_id, variant) pair."""
        key = (family_id, variant)
        if key not in self._lookup:
            raise KeyError(
                f"no human label for {key} in {self._path}")
        return validate_judgment(self._lookup[key])

    def judge(
        self,
        system_prompt: str,
        history_messages: list[dict],
        terminal_query: str,
        response: str,
    ) -> dict:
        """Protocol method — not usable without family_id.

        Use judge_for() with family_id and variant instead.
        """
        raise NotImplementedError(
            "HumanLabelJudge requires family_id and variant for "
            "lookup; use judge_for() instead")

    def provenance(self) -> dict:
        prov = {
            "backend": "human",
            "file": str(self._path),
            "wrapped_schema": self._wrapped,
            "n_labels_loaded": len(self._lookup),
        }
        prov.update(self._provenance_meta)
        return prov
