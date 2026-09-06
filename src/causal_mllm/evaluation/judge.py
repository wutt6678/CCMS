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

        # Require wrapped schema from save_human_labels
        if "labels" not in raw:
            raise EvaluationError(
                f"human labels file {self._path} must use wrapped schema "
                f"with 'labels' and 'provenance' keys — "
                f"use save_human_labels() to create the file")

        if "provenance" not in raw:
            raise EvaluationError(
                f"human labels file {self._path} is missing 'provenance' — "
                f"wrapped schema requires provenance metadata")

        self._wrapped = True
        labels_data = raw["labels"]
        prov = raw["provenance"]

        # Require nonempty labels_sha256
        labels_json = json.dumps(
            labels_data, sort_keys=True, ensure_ascii=False)
        actual_sha = hashlib.sha256(
            labels_json.encode("utf-8")).hexdigest()
        expected_sha = prov.get("labels_sha256", "")
        if not expected_sha:
            raise EvaluationError(
                f"human labels file {self._path} has empty labels_sha256")
        if actual_sha != expected_sha:
            raise EvaluationError(
                f"human labels SHA256 mismatch in {self._path}: "
                f"expected {expected_sha}, got {actual_sha}")

        # Require explicit annotator_id and rubric_version
        if not prov.get("annotator_id"):
            raise EvaluationError(
                f"human labels file {self._path} has empty annotator_id")
        if not prov.get("rubric_version"):
            raise EvaluationError(
                f"human labels file {self._path} has empty rubric_version")

        self._provenance_meta = {
            "labels_sha256": actual_sha,
            "label_file_sha256": prov.get("label_file_sha256"),
            "rubric_version": prov.get("rubric_version"),
            "annotator_id": prov.get("annotator_id"),
            "adjudicated": prov.get("adjudicated", False),
            "n_families": prov.get("n_families", len(labels_data)),
            "n_labels": prov.get("n_labels"),
        }

        # Build a flat lookup: (family_id, variant) -> label
        self._lookup: dict[tuple[str, str], dict] = {}
        for family_id, variants in labels_data.items():
            for variant, label in variants.items():
                self._lookup[(family_id, variant)] = label

        # Require exact 120-key coverage
        n_labels = len(self._lookup)
        if n_labels != 120:
            raise EvaluationError(
                f"human labels file {self._path} has {n_labels} labels, "
                f"expected exactly 120 (20 families × 6 variants)")

    def verify_response_shas(
        self,
        expected_shas: dict[tuple[str, str], str],
    ) -> None:
        """Verify each label's response_sha256 against replay hashes.

        Args:
            expected_shas: Dict mapping (family_id, variant) to the
                SHA256 of the actual replay response text.

        Raises:
            EvaluationError: On any mismatch or missing label.
        """
        errors: list[str] = []
        for key, expected in sorted(expected_shas.items()):
            label = self._lookup.get(key)
            if label is None:
                errors.append(f"missing label for {key}")
                continue
            label_sha = label.get("response_sha256", "")
            # Require nonempty 64-character hex hash
            if not label_sha or len(label_sha) != 64:
                errors.append(
                    f"{key}: response_sha256 must be a nonempty "
                    f"64-character hex hash, got '{label_sha}'")
                continue
            if label_sha != expected:
                errors.append(
                    f"{key}: response_sha256 mismatch — "
                    f"label has {label_sha}, replay has {expected}")
        if errors:
            raise EvaluationError(
                f"response SHA256 verification failed "
                f"in {self._path}:\n  " + "\n  ".join(errors))

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


class LLMEnsembleLabelJudge:
    """Loads LLM-ensemble adjudicated labels from a JSON file.

    This is the LLM analogue of ``HumanLabelJudge``. It requires the
    wrapped schema produced by ``save_llm_ensemble_labels`` with
    ``backend == "llm_ensemble"`` and reports the ensemble backend in
    ``provenance()`` so the final report correctly identifies the label
    source as an LLM ensemble rather than a human annotator.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"LLM ensemble labels file not found: {self._path}")
        with self._path.open(encoding="utf-8") as f:
            raw: dict = json.load(f)

        if "labels" not in raw or "provenance" not in raw:
            raise EvaluationError(
                f"LLM ensemble labels file {self._path} must use wrapped "
                f"schema with 'labels' and 'provenance' keys — use "
                f"save_llm_ensemble_labels() to create the file")

        labels_data = raw["labels"]
        prov = raw["provenance"]

        # Require the llm_ensemble backend marker
        if prov.get("backend") != "llm_ensemble":
            raise EvaluationError(
                f"LLM ensemble labels file {self._path} has backend "
                f"{prov.get('backend')!r}, expected 'llm_ensemble'")

        # Verify label SHA256
        labels_json = json.dumps(
            labels_data, sort_keys=True, ensure_ascii=False)
        actual_sha = hashlib.sha256(
            labels_json.encode("utf-8")).hexdigest()
        expected_sha = prov.get("labels_sha256", "")
        if not expected_sha:
            raise EvaluationError(
                f"LLM ensemble labels file {self._path} has empty "
                f"labels_sha256")
        if actual_sha != expected_sha:
            raise EvaluationError(
                f"LLM ensemble labels SHA256 mismatch in {self._path}: "
                f"expected {expected_sha}, got {actual_sha}")

        self._provenance_meta = {
            "labels_sha256": actual_sha,
            "rubric_version": prov.get("rubric_version"),
            "rubric_sha256": prov.get("rubric_sha256"),
            "annotator_id": prov.get("annotator_id"),
            "adjudicated": prov.get("adjudicated", True),
            "n_families": prov.get("n_families", len(labels_data)),
            "n_labels": prov.get("n_labels"),
            "ensemble": prov.get("ensemble", {}),
        }

        # Cells the ensemble could not label, declared BY THE ARTIFACT rather
        # than by the caller: a file that is short has to say so itself, so
        # that no caller can excuse a truncated write by passing a list of
        # cells it happens to be missing. Empty for every sealed artifact.
        excluded: set[tuple[str, str]] = set()
        for cell in prov.get("excluded_cells") or []:
            if (not isinstance(cell, (tuple, list)) or len(cell) != 2
                    or not all(isinstance(p, str) and p for p in cell)):
                raise EvaluationError(
                    f"LLM ensemble labels file {self._path} declares a "
                    f"malformed excluded cell {cell!r}; expected "
                    f"[family_id, variant]")
            excluded.add((cell[0], cell[1]))

        # Build a flat lookup: (family_id, variant) -> label
        self._lookup: dict[tuple[str, str], dict] = {}
        for family_id, variants in labels_data.items():
            for variant, label in variants.items():
                self._lookup[(family_id, variant)] = label

        if excluded:
            labelled_anyway = sorted(excluded & set(self._lookup))
            if labelled_anyway:
                raise EvaluationError(
                    f"LLM ensemble labels file {self._path} declares "
                    f"{len(labelled_anyway)} cell(s) excluded that it also "
                    f"carries a label for: "
                    f"{[f'{fid}/{v}' for fid, v in labelled_anyway[:5]]}")
            self._provenance_meta["excluded_cells"] = sorted(
                [list(cell) for cell in excluded])
            self._provenance_meta["n_excluded_cells"] = len(excluded)
        self._excluded_cells = frozenset(excluded)

        # Require full factorial coverage: exactly six variant labels per
        # family, less any cell the artifact itself declares excluded,
        # consistent with the declared family count (works for any panel size
        # — Scale-B 20x6, Scale-C 100x6, ...).
        per_family_excluded: dict[str, int] = {}
        for family_id, _ in excluded:
            per_family_excluded[family_id] = (
                per_family_excluded.get(family_id, 0) + 1)
        n_labels = len(self._lookup)
        n_families = len(labels_data)
        bad = [fid for fid, vs in labels_data.items()
               if len(vs) != 6 - per_family_excluded.get(fid, 0)]
        # Only exclusions whose family is still present reduce the expected
        # total: a family that lost all six is absent from the file entirely.
        expected = 6 * n_families - sum(
            n for fid, n in per_family_excluded.items() if fid in labels_data)
        if bad or n_labels != expected:
            raise EvaluationError(
                f"LLM ensemble labels file {self._path} does not have "
                f"exactly six variant labels per family"
                + (f" less its {len(excluded)} declared exclusion(s)"
                   if excluded else "")
                + f" ({n_labels} labels over {n_families} families, expected "
                f"{expected}; families with the wrong count: {bad[:5]})")
        declared = prov.get("n_families")
        if declared is not None and int(declared) != n_families:
            raise EvaluationError(
                f"LLM ensemble labels file {self._path}: provenance "
                f"declares {declared} families but the file carries "
                f"{n_families}")

    def verify_response_shas(
        self,
        expected_shas: dict[tuple[str, str], str],
    ) -> None:
        """Verify each label's response_sha256 against replay hashes."""
        errors: list[str] = []
        for key, expected in sorted(expected_shas.items()):
            label = self._lookup.get(key)
            if label is None:
                errors.append(f"missing label for {key}")
                continue
            label_sha = label.get("response_sha256", "")
            if not label_sha or len(label_sha) != 64:
                errors.append(
                    f"{key}: response_sha256 must be a nonempty "
                    f"64-character hex hash, got '{label_sha}'")
                continue
            if label_sha != expected:
                errors.append(
                    f"{key}: response_sha256 mismatch — "
                    f"label has {label_sha}, replay has {expected}")
        if errors:
            raise EvaluationError(
                f"response SHA256 verification failed "
                f"in {self._path}:\n  " + "\n  ".join(errors))

    @property
    def declared_excluded_cells(self) -> frozenset:
        """The cells THIS labels file declares absent, empty when complete.

        Always available on an ensemble judge, and authoritative: the exclusion
        is a fact about the labels, recorded in the artifact that has it, so
        the evaluation stage derives its panel restriction from here rather
        than from whatever a caller passed in. Two independent sources for one
        decision is how a report comes to describe cells the labels never
        mentioned -- and because a family that loses ANY variant is dropped
        whole, two different lists naming the same family produce identical
        estimates while disagreeing about the evidence.

        Deliberately a property and not a provenance key: the sealed Iteration
        9 and 10 reports carry no ``excluded_cells`` in ``judge_provenance``,
        and adding an empty one to keep this discoverable would change what a
        re-run of those iterations writes.
        """
        return self._excluded_cells

    def judge_for(self, family_id: str, variant: str) -> dict:
        """Look up the LLM label for a (family_id, variant) pair."""
        key = (family_id, variant)
        if key not in self._lookup:
            raise KeyError(
                f"no LLM label for {key} in {self._path}")
        return validate_judgment(self._lookup[key])

    def judge(self, system_prompt, history_messages,
              terminal_query, response) -> dict:
        raise NotImplementedError(
            "LLMEnsembleLabelJudge requires family_id and variant for "
            "lookup; use judge_for() instead")

    def provenance(self) -> dict:
        prov = {
            "backend": "llm_ensemble",
            "file": str(self._path),
            "n_labels_loaded": len(self._lookup),
        }
        prov.update(self._provenance_meta)
        return prov
