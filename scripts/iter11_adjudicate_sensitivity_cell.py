#!/usr/bin/env python3
"""Adjudicate the one cell the differential exclusion cost, into a NEW artifact.

``CMST_456921/text_only`` is the cell judge A's provider refused in the
``ministral3_3b`` arm alone and served in the other three. Because the union of
refused cells is dropped from EVERY arm, the cell left all four ensembles, and
with it the whole ``CMST_456921`` family: the frozen estimator needs all six
variants of a family to produce any of its five estimands. That is why the
confirmatory analysis runs on 98 families rather than 100.

Three of the four arms could have kept it. Judge A and judge B both labelled the
cell in ``qwen35_2b``, ``qwen35_4b`` and ``phi4_mm``; it was the UNION, not
either primary, that removed it. So for those three arms the 99th family is
recoverable, and this script recovers it:

  * ``qwen35_4b`` — A and B agree exactly, so the frozen routing rule
    (``adjudicate_pairwise_with_model``: "items with full agreement keep the
    agreed label") determines the ensemble label offline. NO call is made.
  * ``qwen35_2b`` and ``phi4_mm`` — A and B differ on the score alone, so the
    frozen rule routes the cell to the distinct-model adjudicator. TWO kimi-k3
    calls, one per arm.
  * ``ministral3_3b`` — judge A has no label at all, and
    ``compute_pairwise_agreement`` requires full mutual coverage. No ensemble
    label exists or can be produced, so this arm gets a worst-case bound in
    ``iter11_differential_censoring_bound.py`` instead of a point estimate.

THE FROZEN LABELS ARE NOT TOUCHED. ``llm_labels_adjudicated.json`` is sealed
597-cell evidence per arm, and regenerating the four sets would re-adjudicate
the 1,009 disagreements they record between them (``ministral3_3b`` 299,
``qwen35_2b`` 269, ``qwen35_4b`` 243, ``phi4_mm`` 198) against a gateway whose
moderation verdict has already been shown to move over days. Those four counts
are READ from each set's own ``provenance.ensemble.n_disagreements`` and summed,
not transcribed here: quoting one arm's number for all four -- 243 is
``qwen35_4b``'s -- understates the cost of a regeneration by 766 calls, which is
exactly the kind of error a number in prose makes when nothing recomputes it.
This writes a separate artifact, records the SHA-256 of each arm's frozen label
file before and after, and a test requires the two to be equal. What is produced
here is a SENSITIVITY input, not a replacement label set: the confirmatory
analysis stays on the 98-family common panel.

A RE-FILE SPENDS NO CALLS. The two kimi-k3 calls were made once, when this
artifact was first filed. Re-filing it -- which a code commit forces, because
the provenance block names the commit -- reuses them, bound by the request they
answered: the hash of the request the frozen rule would send NOW is recomputed
offline from the committed blinded item and the two committed primary
judgments, through the production ``LLMAdjudicator`` and ``MultimodalLLMJudge``
with the transport stubbed and nothing sent, and a filed label is reused only
where that hash equals the one its own call recorded. Same request, filed
response. Where the hash differs the call is made; ``--fresh-calls`` makes it
unconditionally. Without this, every re-file would be a second piece of
evidence wearing the first one's name -- the label is free text at temperature
0, which is not the same thing as deterministic -- and a gateway that had begun
refusing this cell would leave the sensitivity unregenerable at the committed
state.

The calls reuse the pipeline's own adjudicator configuration object rather than
a copy of it, so there is one definition of "the frozen adjudicator" and this
script cannot silently drift from it. ``--verify`` performs no calls and needs
no credentials: it checks the artifact's structure, the recorded adjudicator
identity, that the frozen label files still hash to what was recorded, that the
four sealed sets still record the disagreement counts filed here, and that every
filed label still answers the request the frozen rule would send. That last one
needs no key because ``request_hash`` binds the prompt, the image hashes, the
model id, the temperature and the seed -- not the credentials -- so where the
gitignored credentials file is absent, as it is in CI, the hash is computed from
the identity constants pinned below and the substitution is recorded in the
artifact rather than hidden. A test cross-checks those constants against
``ADJUDICATOR_CONFIG`` wherever credentials exist.

Usage:
    python3 scripts/iter11_adjudicate_sensitivity_cell.py --write
    python3 scripts/iter11_adjudicate_sensitivity_cell.py --write --fresh-calls
    python3 scripts/iter11_adjudicate_sensitivity_cell.py --verify
    python3 scripts/iter11_adjudicate_sensitivity_cell.py --write-call-receipt

``--write-call-receipt`` commits the preserved evidence of the calls this
artifact's labels came from, once. It exists because a re-file used to cite the
previous version of the artifact itself as the source of a reused call, and that
version was overwritten by the re-file and never committed: the citation named a
sha256 and a commit that resolve to nothing, and ``--verify`` asked only whether
the sha256 was a string. The receipt is written once and refused thereafter, is
committed before anything cites it, and ``--verify`` resolves it out of the git
object store, hashes what it resolves, and compares that hash with the cited one.

Exit codes: 0 verified / written; 1 a disagreement with the frozen rule, a
missing primary label where one is required, a filed label that does not answer
the request the frozen rule would send, a frozen label file that moved, a reuse
citation that does not resolve to the bytes it names or that no commit holds, or
an attempt to overwrite the receipt; 2 no artifact to verify against, or --write
needing a call it has no credentials for (nothing is written, so a failed call
cannot overwrite a filed one); 3 the adjudicator refused a cell, so the label
could not be produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from causal_mllm.evaluation.adjudication import (  # noqa: E402
    judgments_disagree,
)
from causal_mllm.evaluation.censoring import (  # noqa: E402
    DIFFERENTIAL_CELL,
    DIFFERENTIAL_TARGET,
)
from causal_mllm.seeds import (  # noqa: E402
    code_tree_status,
    get_git_commit,
    sha256_bytes,
)

JUDGE_ROOT = REPO_ROOT / "outputs" / "iteration_11" / "judge"
OUT_PATH = REPO_ROOT / "outputs" / "iteration_11" / "analysis" \
    / "differential_censoring" / "labels_adjudicated.sensitivity_99f.json"

#: The immutable receipt for the two adjudicator calls this artifact's labels
#: come from. It exists because a re-file used to cite the PREVIOUS VERSION OF
#: THE ARTIFACT ITSELF as the source of a reused call -- sha256 2d00760c..., filed
#: at commit fe455929.... That commit was amended before it was pushed and the
#: bytes it cited were never committed at all, so the pointer resolved to
#: nothing: the blob is absent from the object store, the commit is reachable
#: from no ref, and its tree never held the path. A citation a reviewer cannot
#: resolve is not provenance. The receipt is committed beside the artifact, is
#: written once and refused thereafter, and ``--verify`` resolves it out of the
#: object store and hashes what it resolves.
RECEIPT_PATH = REPO_ROOT / "outputs" / "iteration_11" / "analysis" \
    / "differential_censoring" / "call_receipts.sensitivity_99f.json"

ARMS = ("ministral3_3b", "phi4_mm", "qwen35_2b", "qwen35_4b")
FAMILY, VARIANT = DIFFERENTIAL_CELL.split("/")
ITEM_ID = "item-0365"

#: The frozen adjudicator's identity, pinned here so ``--verify`` can check the
#: artifact without importing the pipeline -- which raises at import time when
#: no API key is present, and a verification that needs credentials is a
#: verification CI cannot run. A test cross-checks these against the pipeline's
#: own ``ADJUDICATOR_CONFIG`` so they cannot drift.
ADJUDICATOR_MODEL_ID = "kimi-k3"
ADJUDICATOR_PROVIDER = "aliyun"
ADJUDICATOR_MODEL_SEED = 99
ADJUDICATOR_TEMPERATURE = 0.0
ADJUDICATOR_JUDGE_ID = "ADJ"

#: ``LLMAdjudicator``'s presentation-order seed, which is NOT the model seed:
#: the pipeline constructs it as ``LLMAdjudicator(judge, seed=0)`` while the
#: judge's own config carries 99. Two different seeds with two different jobs,
#: and conflating them would reproduce the call with the primaries in the wrong
#: order.
ORDER_SEED = 0

FROZEN_LABELS = "llm_labels_adjudicated.json"
FROZEN_LABEL_COUNT = 597

#: Stand-ins for the two config fields a request hash does NOT bind. They exist
#: so the hash of the request the frozen rule would send can be recomputed where
#: the gitignored credentials file does not -- CI, and any checkout that has not
#: been given a key. ``request_hash`` is built from the prompt hash, the image
#: hashes, the model id, the temperature and the seed (see
#: ``MultimodalLLMJudge.judge``), so neither value can reach it, and the
#: artifact records which source the identity came from rather than leaving the
#: substitution to be inferred from a hash that happens to match.
HASHING_ONLY_BASE_URL = "https://credentials-absent.invalid/v1"
HASHING_ONLY_API_KEY = "not-a-key: request_hash binds neither this nor base_url"

#: The judgment the stubbed transport returns. It is DISCARDED -- only the
#: provenance the production code computes alongside it is kept -- and it is
#: deliberately a label no judge would give, so a bug that filed it instead of
#: the reused one would be visible in the artifact rather than plausible in it.
STUB_JUDGMENT = {
    "refusal_type": "none", "unsafe_compliance_score": 0.5,
    "compliance_level": 2, "safe_redirection": False, "confidence": 0.5,
    "rationale": "STUBBED TRANSPORT: nothing was sent and this judgment must "
                 "never be filed -- only the request hash computed beside it "
                 "is used",
}

#: This stage's own output, excluded from the tree dirtiness it RECORDS.
OWN_OUTPUT_PREFIXES = (
    "outputs/iteration_11/analysis/differential_censoring/",)


def _rel(path: Path | str) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def fatal(message: str, code: int) -> NoReturn:
    """Exit with the code the docstring says this failure means.

    ``raise SystemExit("text")`` exits 1 and prints the text, and 1 is the code
    this script documents for a DIFFERENT failure -- a disagreement with the
    frozen rule. A missing credential reported as a disagreement, or a missing
    input reported as a moved label, sends whoever reads the exit code to the
    wrong half of the artifact, so the message goes to stderr and the code goes
    to ``SystemExit`` on its own.
    """
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> tuple[int, bytes]:
    """``(returncode, stdout or stderr)`` for one git plumbing call."""
    import subprocess
    try:
        proc = subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 1, f"{type(exc).__name__}: {exc}".encode("utf-8")
    return proc.returncode, (proc.stdout if proc.returncode == 0
                             else proc.stderr)


def resolve_evidence(path: Path) -> dict:
    """Resolve a cited artifact out of the object store, not off this disk.

    Hashing the working-tree copy of a file proves only that this machine has a
    file, which is exactly the claim that failed: the reused calls cited a
    sha256 nothing could resolve. Resolving ``HEAD:<path>`` and reading that blob
    proves the bytes are reachable from the commit, so any clone can perform the
    same check and get the same answer. An untracked file falls back to the
    working tree and SAYS SO, because "resolved" and "resolved from the commit"
    are different statements and the difference is the whole point.
    """
    rel = _rel(path)
    out: dict = {
        "path": rel,
        "exists_on_disk": path.exists(),
        "working_tree_sha256": sha256_file(path),
        "resolved_from": None,
        "bytes_sha256": None,
        "git_blob_sha1": None,
        "head_commit": None,
        "why_not": None,
    }
    code, head = _git("rev-parse", "HEAD")
    if code == 0:
        out["head_commit"] = head.decode("utf-8", "replace").strip()
    code, blob = _git("rev-parse", f"HEAD:{rel}")
    if code == 0:
        blob_sha = blob.decode("utf-8", "replace").strip()
        code, content = _git("cat-file", "blob", blob_sha)
        if code == 0:
            out["resolved_from"] = "the blob HEAD holds at this path"
            out["bytes_sha256"] = sha256_bytes(content)
            out["git_blob_sha1"] = blob_sha
            return out
        out["why_not"] = (f"HEAD names blob {blob_sha} for {rel} but it cannot "
                          f"be read: "
                          f"{content.decode('utf-8', 'replace').strip()[:200]}")
    else:
        out["why_not"] = (
            f"no commit reachable from HEAD holds {rel}: "
            f"{blob.decode('utf-8', 'replace').strip()[:200]}")
    if path.exists():
        out["resolved_from"] = ("the working tree, which no commit reachable "
                                "from HEAD holds")
        out["bytes_sha256"] = out["working_tree_sha256"]
    return out


def arm_paths(arm: str) -> dict[str, Path]:
    root = JUDGE_ROOT / arm
    return {
        "blinded_items": root / "blinded_items.json",
        "judge_A": root / "llm_labels_judge_A.json",
        "judge_B": root / "llm_labels_judge_B.json",
        "frozen_labels": root / FROZEN_LABELS,
    }


def blinded_item(arm: str) -> dict | None:
    """The item this arm's judges were actually shown, read not reconstructed."""
    path = arm_paths(arm)["blinded_items"]
    if not path.exists():
        return None
    for item in load_json(path):
        if item["family_id"] == FAMILY and item["variant"] == VARIANT:
            return item
    return None


def primary_record(arm: str, judge_id: str) -> dict | None:
    """One primary's own label for the cell, or None if it has none."""
    path = arm_paths(arm)[f"judge_{judge_id}"]
    if not path.exists():
        return None
    for record in load_json(path):
        if record["family_id"] == FAMILY and record["variant"] == VARIANT:
            return record
    return None


def presentation_order() -> list[int]:
    """The order the two primaries are shown in, for a single-item call.

    Computed from a FRESH ``Random(ORDER_SEED)`` because that is the state the
    adjudicator is in when it shuffles for the first and only item. Recorded so
    the call can be reproduced: the adjudicator anonymizes the primaries as
    Judge X and Judge Y in this order, and position bias is the reason the
    order is randomized at all.
    """
    order = [0, 1]
    random.Random(ORDER_SEED).shuffle(order)
    return order


_PIPELINE_CACHE: dict[str, object] = {}
_CONFIG_CACHE: dict[str, object] = {}


def _pipeline():
    """The judge pipeline module, for its adjudicator configuration.

    Loaded lazily and by path: importing it at module level would make
    ``--verify`` need an API key, and this script's own constants would then be
    the only thing CI could check. Cached, because this script now asks for the
    frozen identity several times per run -- once per arm's request hash, once
    for the rubric, once to decide whether a call could be sent at all -- and
    executing the pipeline module for each of those would be slow for no gain.
    """
    if "module" in _PIPELINE_CACHE:
        return _PIPELINE_CACHE["module"]
    spec = importlib.util.spec_from_file_location(
        "run_llm_judge_pipeline_for_sensitivity",
        REPO_ROOT / "scripts" / "run_llm_judge_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _PIPELINE_CACHE["module"] = mod
    return mod


def _identity_of(config) -> dict:
    return {"model_id": config.model_id, "provider": config.provider,
            "seed": config.seed, "temperature": config.temperature,
            "judge_id": ADJUDICATOR_JUDGE_ID, "order_seed": ORDER_SEED}


def _require_frozen_identity(config, source: str) -> dict:
    """Refuse an adjudicator that is not the frozen one, naming where it came
    from. One comparison used by both the live path and the hashing path, so
    "the frozen adjudicator" has one definition in this script too."""
    identity = _identity_of(config)
    for field, expected in (("model_id", ADJUDICATOR_MODEL_ID),
                            ("provider", ADJUDICATOR_PROVIDER),
                            ("seed", ADJUDICATOR_MODEL_SEED),
                            ("temperature", ADJUDICATOR_TEMPERATURE)):
        if identity[field] != expected:
            fatal(
                f"{source} gives adjudicator {field} {identity[field]!r}, not "
                f"the frozen {expected!r}: this script would not be "
                f"reproducing the frozen adjudicator", 1)
    return identity


def adjudicator_config() -> tuple[object, str, bool]:
    """The frozen adjudicator's config, where it came from, and may it be sent.

    The pipeline's own ``ADJUDICATOR_CONFIG`` whenever the gitignored
    credentials file exists, because that object is the definition of "the
    frozen adjudicator" and a copy of it is a second definition that can drift.
    Where it does not exist -- CI, or any checkout without a key -- the identity
    constants pinned above stand in, and the substitution is returned as a
    string so the artifact can record it rather than let a matching hash imply
    an identity nobody checked.

    ``sendable`` is False for the stand-in. Hashing a request needs the model
    id, the temperature and the seed; sending one needs a key, and a key-shaped
    placeholder must never reach the transport.
    """
    if "config" in _CONFIG_CACHE:
        return (_CONFIG_CACHE["config"], _CONFIG_CACHE["source"],
                bool(_CONFIG_CACHE["sendable"]))
    why_no_credentials = None
    config = None
    try:
        config = _pipeline().ADJUDICATOR_CONFIG
    except EnvironmentError as exc:
        why_no_credentials = f"{type(exc).__name__}: {exc}"
    if config is not None:
        source = ("the pipeline's own ADJUDICATOR_CONFIG, loaded by path so "
                  "there is one definition of the frozen adjudicator")
        _require_frozen_identity(config, "the pipeline's ADJUDICATOR_CONFIG")
        _CONFIG_CACHE.update(config=config, source=source, sendable=True)
        return config, source, True

    from causal_mllm.evaluation.llm_judge import LLMJudgeConfig
    config = LLMJudgeConfig(
        model_id=ADJUDICATOR_MODEL_ID, provider=ADJUDICATOR_PROVIDER,
        base_url=HASHING_ONLY_BASE_URL, api_key=HASHING_ONLY_API_KEY,
        temperature=ADJUDICATOR_TEMPERATURE, seed=ADJUDICATOR_MODEL_SEED)
    _require_frozen_identity(config, "this script's pinned constants")
    source = (
        "the identity constants pinned in this script, because the pipeline "
        "cannot be imported without credentials"
        + (f" ({why_no_credentials})" if why_no_credentials else "")
        + ". api_key and base_url are stand-ins that enter no hash, and a "
          "test cross-checks these constants against ADJUDICATOR_CONFIG "
          "wherever credentials exist. Good for hashing a request, NOT for "
          "sending one")
    _CONFIG_CACHE.update(config=config, source=source, sendable=False)
    return config, source, False


def rubric_identity() -> dict:
    """The rubric the adjudicator sends, read from the judge that loads it.

    ``MultimodalLLMJudge`` defaults to ``annotation_rubric_v1_1.md`` and hashes
    it, and the frozen protocol names that hash rather than the version string.
    Constructing the judge needs no credentials, so this is available to
    ``--verify`` and to CI as well: a rubric edited after the calls were made
    changes this hash and the filed artifact stops verifying, which is the point
    of filing a hash instead of "1.1".
    """
    from causal_mllm.evaluation.llm_judge import MultimodalLLMJudge

    config, source, _sendable = adjudicator_config()
    judge = MultimodalLLMJudge(config, judge_id=ADJUDICATOR_JUDGE_ID)
    return {"rubric_version": judge.rubric_version,
            "rubric_sha256": judge.rubric_sha256,
            "rubric_path": _rel(judge.rubric_path),
            "source": source}


def _stub_transport(judge) -> None:
    """Replace the only network call in the judge with one that sends nothing.

    Set on the instance rather than on the class, so no other judge in the
    process is affected and nothing has to be restored afterwards.
    """
    judge._call_api = lambda eval_prompt, image_contents: (
        dict(STUB_JUDGMENT), "STUBBED_TRANSPORT_NOT_SENT", "stop", 0,
        "stubbed-no-provider-response-id", "stubbed-no-provider-model", "")


def expected_request(arm: str) -> dict:
    """The request the frozen rule would send for this arm's cell, hashed.

    Computed by running the production call path with the transport stubbed:
    ``LLMAdjudicator.adjudicate_item`` builds the instruction and the augmented
    terminal query, ``MultimodalLLMJudge.judge`` builds the prompt and folds it
    into ``request_hash``, and ``_call_api`` returns a judgment this script
    throws away. What comes back is therefore the hash the real call would
    carry, from the real code, having sent nothing -- which is why reuse is a
    check rather than a trust, and why it needs no credentials.
    """
    from causal_mllm.evaluation.adjudication import LLMAdjudicator
    from causal_mllm.evaluation.llm_judge import MultimodalLLMJudge

    item = blinded_item(arm)
    rec_a = primary_record(arm, "A")
    rec_b = primary_record(arm, "B")
    absent = [name for name, obj in (("the blinded item", item),
                                     ("judge A's label", rec_a),
                                     ("judge B's label", rec_b))
              if obj is None]
    if absent:
        return {"available": False, "request_hash": None,
                "prompt_sha256": None, "image_hashes": None,
                "config_source": None,
                "why_not": "no request exists to hash: "
                           + ", ".join(absent)
                           + " is not in the committed evidence for this arm"}

    config, source, _sendable = adjudicator_config()
    judge = MultimodalLLMJudge(config, judge_id=ADJUDICATOR_JUDGE_ID)
    _stub_transport(judge)
    adjudicator = LLMAdjudicator(judge, seed=ORDER_SEED)
    try:
        _discarded, provenance = adjudicator.adjudicate_item(
            system_prompt=item["system_prompt"],
            history_messages=item["conversation_history"],
            terminal_query=item["terminal_query"],
            response=item["response"],
            primary_judgments=[rec_a["judgment"], rec_b["judgment"]],
        )
    except Exception as exc:  # noqa: BLE001 - a payload this checkout cannot
        return {"available": False, "request_hash": None,   # build is evidence
                "prompt_sha256": None, "image_hashes": None,
                "config_source": source,
                "why_not": f"the request could not be built offline: "
                           f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "request_hash": provenance.request_hash,
        "prompt_sha256": provenance.prompt_sha256,
        "image_hashes": provenance.image_hashes,
        "rubric_sha256": provenance.rubric_sha256,
        "rubric_version": provenance.rubric_version,
        "config_source": source,
        "how": "LLMAdjudicator.adjudicate_item and MultimodalLLMJudge.judge "
               "run for real over the committed blinded item and the two "
               "committed primary judgments, with _call_api -- the only "
               "network call in the path -- replaced by a stub whose judgment "
               "is discarded. Nothing was sent; the hash is the one the real "
               "call would have carried",
    }


def sealed_label_sets() -> dict:
    """What the four sealed label sets record about themselves, read not quoted.

    Each ``llm_labels_adjudicated.json`` carries its own
    ``provenance.ensemble.n_disagreements``: the number of adjudicator calls a
    regeneration of that set would spend. Summing them is what makes the claim
    about that cost a measurement, and refusing to file when a set does not
    record one is what keeps it from becoming a transcription of whatever this
    script said last time.
    """
    per_arm = {}
    for arm in ARMS:
        path = arm_paths(arm)["frozen_labels"]
        if not path.exists():
            fatal(f"no sealed {FROZEN_LABELS} for {arm} at {_rel(path)}, so "
                  f"the cost of regenerating it cannot be read", 2)
        doc = load_json(path)
        prov = doc.get("provenance") or {}
        ensemble = prov.get("ensemble") or {}
        n_dis = ensemble.get("n_disagreements")
        n_labels = prov.get("n_labels")
        if not isinstance(n_dis, int) or not isinstance(n_labels, int):
            fatal(f"{arm}'s sealed {FROZEN_LABELS} records "
                  f"n_disagreements={n_dis!r} and n_labels={n_labels!r}, not "
                  f"two integers. This script quotes those numbers rather than "
                  f"counting disagreements a second time, so a set that does "
                  f"not record them is a set it cannot describe", 1)
        if n_labels != FROZEN_LABEL_COUNT:
            fatal(f"{arm}'s sealed {FROZEN_LABELS} holds {n_labels} labels, "
                  f"not the {FROZEN_LABEL_COUNT} this script pins: the sealed "
                  f"set is not the one the sensitivity was designed beside", 1)
        n_in_file = sum(len(by_variant)
                        for by_variant in (doc.get("labels") or {}).values())
        if n_in_file != n_labels:
            fatal(f"{arm}'s sealed {FROZEN_LABELS} says {n_labels} labels and "
                  f"holds {n_in_file}: its own provenance does not describe "
                  f"it, so nothing quoted from that provenance is evidence", 1)
        per_arm[arm] = {
            "path": _rel(path),
            "sha256": sha256_file(path),
            "n_labels": n_labels,
            "n_labels_present": n_in_file,
            "n_disagreements": n_dis,
            "adjudicator_model": ensemble.get("adjudicator_model"),
            "adjudication_method": ensemble.get("adjudication_method"),
        }
    total = sum(row["n_disagreements"] for row in per_arm.values())
    return {
        "per_arm": per_arm,
        "n_disagreements_total": total,
        "n_labels_each": FROZEN_LABEL_COUNT,
        "read_from": "each set's own provenance.ensemble.n_disagreements, "
                     "summed over the four arms",
        "per_arm_counts": {arm: per_arm[arm]["n_disagreements"]
                           for arm in ARMS},
    }


def reusable_call(arm: str, state: dict, filed_entry: dict | None,
                  expected: dict,
                  source_path: Path | None = None
                  ) -> tuple[dict | None, str | None]:
    """A filed call this re-file may reuse, or the reason it may not.

    Reuse is bound by everything that determines the request -- the response
    hash, both primaries' judgments byte for byte, the differing fields the
    frozen rule routed on, the presentation order, the adjudicator identity --
    and then by the request hash itself, recomputed above. A filed label that
    survives all of it is the answer to the question the frozen rule asks now.
    One that does not is evidence about a different question, and the call is
    made instead.
    """
    if not filed_entry:
        return None, (f"{_rel(source_path or RECEIPT_PATH)} preserves no call "
                      f"for this arm, so there is nothing to reuse")
    if filed_entry.get("call_failed"):
        return None, ("the filed call failed"
                      f" ({filed_entry.get('call_failed')}), so there is no "
                      "label to reuse")
    checks = (
        ("ensemble_label_status", filed_entry.get("ensemble_label_status"),
         state["ensemble_label_status"]),
        ("differing_fields", filed_entry.get("differing_fields"),
         state["differing_fields"]),
        ("response_sha256", filed_entry.get("response_sha256"),
         state["response_sha256"]),
        ("item_id", filed_entry.get("item_id"), ITEM_ID),
        ("primary_A_judgment", filed_entry.get("primary_A_judgment"),
         (state["primary_A"] or {}).get("judgment")),
        ("primary_B_judgment", filed_entry.get("primary_B_judgment"),
         (state["primary_B"] or {}).get("judgment")),
        ("presentation_order", filed_entry.get("presentation_order"),
         presentation_order()),
    )
    for field, was, is_now in checks:
        if was != is_now:
            return None, (f"the filed {field} is not what the committed "
                          f"evidence gives now, so the filed label describes "
                          f"a different call")
    if filed_entry.get("ensemble_label") is None:
        return None, "the artifact files no ensemble_label, so there is no " \
                     "label to reuse"
    provenance = filed_entry.get("call_provenance") or {}
    if not expected["available"]:
        return None, ("the request the frozen rule would send cannot be "
                      f"recomputed here: {expected.get('why_not')}")
    if provenance.get("request_hash") != expected["request_hash"]:
        return None, ("the filed call answered request "
                      f"{str(provenance.get('request_hash'))[:16]} but the "
                      f"frozen rule now sends {expected['request_hash'][:16]}: "
                      "the label is not an answer to this question")
    if filed_entry.get("adjudicated_by") != ADJUDICATOR_MODEL_ID:
        return None, ("the filed label says it was adjudicated by "
                      f"{filed_entry.get('adjudicated_by')!r}, not "
                      f"{ADJUDICATOR_MODEL_ID!r}")
    return {"judgment": filed_entry["ensemble_label"],
            "adjudicator": filed_entry.get("adjudicator"),
            "call_provenance": provenance}, None


def adjudicate(arm: str) -> tuple[dict, dict]:
    """One live adjudicator call for this arm's cell, using the frozen config."""
    from causal_mllm.evaluation.adjudication import LLMAdjudicator
    from causal_mllm.evaluation.llm_judge import MultimodalLLMJudge

    config, source, sendable = adjudicator_config()
    if not sendable:
        fatal(f"{arm}'s cell needs a live adjudicator call and this checkout "
              f"has no credentials to make one ({source}). Nothing is written: "
              f"a re-file that could not call must not overwrite the artifact "
              f"that did", 2)
    identity = _require_frozen_identity(
        config, "the pipeline's ADJUDICATOR_CONFIG")

    item = blinded_item(arm)
    rec_a = primary_record(arm, "A")
    rec_b = primary_record(arm, "B")
    absent = [name for name, obj in (("the blinded item", item),
                                     ("judge A's label", rec_a),
                                     ("judge B's label", rec_b))
              if obj is None]
    if absent:
        fatal(f"cannot adjudicate {arm}'s cell without " + ", ".join(absent)
              + "; that is missing evidence, not an adjudicator refusal", 1)
    adjudicator = LLMAdjudicator(
        MultimodalLLMJudge(config, judge_id=ADJUDICATOR_JUDGE_ID),
        seed=ORDER_SEED)
    judgment, provenance = adjudicator.adjudicate_item(
        system_prompt=item["system_prompt"],
        history_messages=item["conversation_history"],
        terminal_query=item["terminal_query"],
        response=item["response"],
        primary_judgments=[rec_a["judgment"], rec_b["judgment"]],
    )
    from dataclasses import asdict
    return judgment, {"adjudicator": identity,
                      "call_provenance": asdict(provenance)}


def derive_from_agreement(arm: str) -> dict:
    """The ensemble label the frozen routing rule gives when A and B agree.

    No call: ``adjudicate_pairwise_with_model`` keeps the agreed label and
    marks it ``primary_agreement``, so reproducing that here is reproducing the
    rule, not approximating it.
    """
    record = primary_record(arm, "A")
    return {"judgment": dict(record["judgment"]),
            "adjudicated_by": "primary_agreement"}


def arm_state(arm: str) -> dict:
    """What the frozen rules say about this arm's cell, before any call."""
    item = blinded_item(arm)
    rec_a = primary_record(arm, "A")
    rec_b = primary_record(arm, "B")
    paths = arm_paths(arm)
    state = {
        "item_id": (item or {}).get("item_id"),
        "response_sha256": (item or {}).get("response_sha256"),
        "response_chars": len((item or {}).get("response") or ""),
        "primary_A": rec_a,
        "primary_B": rec_b,
        "frozen_labels_sha256": sha256_file(paths["frozen_labels"]),
    }
    if rec_a is None or rec_b is None:
        missing = sorted(name for name, rec in (("A", rec_a), ("B", rec_b))
                         if rec is None)
        state["ensemble_label_status"] = "not_derivable"
        state["why_not_derivable"] = (
            f"judge {' and '.join(missing)} has no label for this cell in this "
            f"arm: its provider refused the request, and "
            f"compute_pairwise_agreement requires full mutual coverage, so a "
            f"cell one primary could not judge must not become a label from "
            f"the other primary alone. No ensemble label exists for "
            f"{DIFFERENTIAL_CELL} in {arm} and none can be produced from the "
            f"committed evidence -- which is why this arm is bounded rather "
            f"than recomputed")
        state["differing_fields"] = None
        return state
    differing = judgments_disagree([rec_a["judgment"], rec_b["judgment"]])
    state["differing_fields"] = differing
    state["ensemble_label_status"] = (
        "derived_from_primary_agreement" if not differing
        else "requires_adjudication")
    return state


def provenance() -> dict:
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    return {
        "produced_by": "scripts/iter11_adjudicate_sensitivity_cell.py",
        "kind": "iteration_11_sensitivity_labels_99f_v1",
        "code_commit": get_git_commit(),
        "git_dirty": tree["dirty"],
        "code_dirty_paths": tree["code_dirty_paths"],
        "untracked_code_paths": tree["untracked_paths"],
        "excluded_own_outputs": tree["excluded_own_outputs"],
        "excluded_cache_paths": tree["excluded_cache_paths"],
    }


def filed_artifact(path: Path | None = None) -> tuple[dict | None, str | None]:
    """The artifact already on disk, and its hash: the pool reuse draws from."""
    path = OUT_PATH if path is None else path
    if not path.exists():
        return None, None
    return load_json(path), sha256_file(path)


#: What a reuse is bound by, and therefore what the receipt has to preserve for
#: each arm. These are exactly the fields :func:`reusable_call` compares, plus
#: the identity of the call itself, so a re-file drawing on the receipt is bound
#: by the same evidence a re-file drawing on the artifact was.
RECEIPT_CALL_FIELDS = (
    "item_id", "response_sha256", "ensemble_label_status", "differing_fields",
    "primary_A_judgment", "primary_B_judgment", "primary_A_model_id",
    "primary_B_model_id", "presentation_order", "ensemble_label",
    "adjudicated_by", "adjudicator", "call_provenance",
    "request_the_frozen_rule_would_send",
)

#: The citation this receipt replaces. Kept as constants rather than read back
#: out of the artifact, because the artifact no longer carries them: filing what
#: could not be resolved is the only record that it was ever claimed.
SUPERSEDED_CITATION_SHA256 = (
    "2d00760c5035d9049d571a6517b2ffd7505d557ac630bf805c46a69d9bed268c")
SUPERSEDED_CITATION_COMMIT = "fe455929575382e5e57e2346b271d095085d39ee"
SUPERSEDED_CITATION_GENERATED_AT = "2026-09-08T11:04:10+00:00"


def committed_versions_of(rel_path: str) -> list[dict]:
    """Every blob any reachable commit holds at ``rel_path``, hashed.

    This is how "the bytes a citation names were never committed" is MEASURED
    rather than asserted: a sha256 of file content is not a git object id, so
    asking git for it proves nothing, while enumerating the committed versions of
    the path and hashing each one answers the question that was actually asked.
    """
    code, out = _git("log", "--all", "--format=%H", "--", rel_path)
    commits = out.decode("utf-8", "replace").split() if code == 0 else []
    versions = []
    for commit in commits:
        code, blob = _git("rev-parse", f"{commit}:{rel_path}")
        if code != 0:
            continue
        blob_sha = blob.decode("utf-8", "replace").strip()
        code, content = _git("cat-file", "blob", blob_sha)
        if code != 0:
            continue
        versions.append({"commit": commit, "git_blob_sha1": blob_sha,
                         "sha256": sha256_bytes(content)})
    return versions


def superseded_citation() -> dict:
    """What the unusable citation pointed at, and the measurements that say so."""
    rel = _rel(OUT_PATH)
    versions = committed_versions_of(rel)
    matching = [v for v in versions
                if v["sha256"] == SUPERSEDED_CITATION_SHA256]
    code, _ = _git("cat-file", "-e", f"{SUPERSEDED_CITATION_COMMIT}^{{commit}}")
    exists_here = code == 0
    code, _ = _git("merge-base", "--is-ancestor", SUPERSEDED_CITATION_COMMIT,
                   "HEAD")
    reachable = code == 0
    held_the_path = False
    if exists_here:
        code, _ = _git("rev-parse", f"{SUPERSEDED_CITATION_COMMIT}:{rel}")
        held_the_path = code == 0
    return {
        "the_citation_that_could_not_be_resolved": {
            "path": rel,
            "sha256": SUPERSEDED_CITATION_SHA256,
            "code_commit": SUPERSEDED_CITATION_COMMIT,
            "generated_at": SUPERSEDED_CITATION_GENERATED_AT,
            "where_it_appeared": "per_arm.<arm>.call_reused_from, in the "
                                 "re-file of this artifact that cited the "
                                 "previous version of the artifact itself",
        },
        "measured": {
            "n_committed_versions_of_that_path": len(versions),
            "committed_versions": versions,
            "n_of_them_hashing_to_the_cited_sha256": len(matching),
            "the_cited_commit_is_in_this_object_store": exists_here,
            "the_cited_commit_is_reachable_from_head": reachable,
            "the_cited_commit_holds_that_path_in_its_tree": held_the_path,
        },
        "what_it_shows": (
            "no committed version of the artifact hashes to the cited sha256, "
            "and the commit named beside it is reachable from no ref and holds "
            "no such path in its tree, so the two calls' source evidence was "
            "resolvable nowhere: not by a reviewer, not by a clone, and not by "
            "--verify, which required only that the sha256 be a string. The "
            "bytes themselves were overwritten by the re-file that cited them "
            "and are not recoverable; what IS recoverable is the calls' "
            "preserved provenance, which the re-file carried forward, and that "
            "is what this receipt commits"),
        "what_this_receipt_does_about_it": (
            "it holds the preserved evidence of each call in a file of its own, "
            "committed beside the artifact, written once and refused "
            "thereafter, and --verify resolves it out of the object store and "
            "hashes what it resolves rather than trusting a string"),
    }


def call_receipt(source: Path | None = None) -> dict:
    """The preserved evidence of every call this artifact's labels rest on.

    Extracted from the artifact that is committed and reachable, not from the
    version that was neither. A receipt is a historical record, so it is verified
    by resolution and hash rather than re-derived: re-deriving it would mean
    re-asking a gateway that has already been shown to move.
    """
    source = OUT_PATH if source is None else Path(source)
    if not source.exists():
        fatal(f"there is no artifact at {_rel(source)} to preserve call "
              f"evidence from", 2)
    artifact = load_json(source)
    per_arm = artifact.get("per_arm") or {}
    calls = {}
    for arm in ARMS:
        entry = per_arm.get(arm) or {}
        made_a_call = entry.get("call_provenance") is not None
        calls[arm] = {
            "a_call_was_made": made_a_call,
            "ensemble_label_status": entry.get("ensemble_label_status"),
            "why_no_call_was_made": None if made_a_call else (
                "the frozen routing rule resolves this cell without a call"
                if entry.get("ensemble_label_status")
                == "derived_from_primary_agreement"
                else "no ensemble label exists or can be produced for this "
                     "cell in this arm, so there was never a call to make"),
            "preserved": {field: entry.get(field)
                          for field in RECEIPT_CALL_FIELDS},
        }
    resolution = resolve_evidence(source)
    tree = code_tree_status(exclude_prefixes=OWN_OUTPUT_PREFIXES)
    return {
        "question": "what were the adjudicator calls behind these labels, and "
                    "can a reviewer resolve them",
        "produced_by": "scripts/iter11_adjudicate_sensitivity_cell.py "
                       "--write-call-receipt",
        "kind": "iteration_11_adjudicator_call_receipt_v1",
        "immutable": True,
        "code_commit": get_git_commit(),
        "git_dirty": tree["dirty"],
        "code_dirty_paths": tree["code_dirty_paths"],
        "untracked_code_paths": tree["untracked_paths"],
        "excluded_own_outputs": tree["excluded_own_outputs"],
        "excluded_cache_paths": tree["excluded_cache_paths"],
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "cell": DIFFERENTIAL_CELL, "family": FAMILY, "variant": VARIANT,
        "item_id": ITEM_ID,
        "n_arms": len(ARMS),
        "n_calls_preserved": sum(1 for c in calls.values()
                                 if c["a_call_was_made"]),
        "adjudicator_identity": artifact.get("adjudicator_identity"),
        "source_of_the_preserved_evidence": {
            "path": resolution["path"],
            "sha256": resolution["bytes_sha256"],
            "resolved_from": resolution["resolved_from"],
            "git_blob_sha1": resolution["git_blob_sha1"],
            "head_commit": resolution["head_commit"],
            "the_artifact_files_its_own_code_commit_as":
                artifact.get("code_commit"),
            "why": "the calls' provenance survives in this artifact because a "
                   "re-file rewrites the provenance block and not the "
                   "evidence, so the artifact is a faithful source -- but it is "
                   "rewritten on every re-file, and a receipt has to stop "
                   "moving",
        },
        "supersedes": superseded_citation(),
        "per_arm": calls,
        "how_it_is_verified": (
            "--verify resolves this file out of the object store, hashes what it "
            "resolves, requires that hash to equal the one the artifact cites, "
            "requires every reused arm's label and call provenance to be the "
            "ones this receipt holds, and recomputes each request hash offline "
            "through the production call path with the transport stubbed"),
        "what_this_does_not_do": (
            "it does not make the two calls again, and it does not recover the "
            "bytes the superseded citation named: those were overwritten and "
            "were never committed. It commits what survived"),
    }


def write_call_receipt(source: Path | None = None) -> int:
    """Write the receipt once. A second attempt is refused, not reconciled."""
    if RECEIPT_PATH.exists():
        existing = sha256_file(RECEIPT_PATH)
        print(f"REFUSING to overwrite {_rel(RECEIPT_PATH)} (sha256 {existing}). "
              f"A receipt is a record of what happened, written once: rewriting "
              f"it would make the sha256 the artifact cites mean something "
              f"else, which is the failure this receipt exists to prevent. If "
              f"the evidence it preserves has genuinely changed, that is a "
              f"different receipt and belongs beside this one.",
              file=sys.stderr)
        return 1
    receipt = call_receipt(source)
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {_rel(RECEIPT_PATH)}")
    print(f"  calls preserved  {receipt['n_calls_preserved']} of "
          f"{receipt['n_arms']} arms")
    print(f"  source           {receipt['source_of_the_preserved_evidence']['path']}")
    print(f"  resolved from    "
          f"{receipt['source_of_the_preserved_evidence']['resolved_from']}")
    print(f"  sha256           {sha256_file(RECEIPT_PATH)}")
    return 0


def reuse_source(explicit: Path | None = None) -> tuple[dict, str | None, dict]:
    """``({arm: preserved call evidence}, sha256, document)`` to reuse from.

    The RECEIPT, by default and by preference. A re-file used to draw on the
    previous version of the artifact itself, which is rewritten by the very
    re-file doing the drawing, so the sha256 it cited named bytes that stopped
    existing at the moment of citation. The receipt is written once, refused
    thereafter, and committed beside the artifact, so what a re-file cites is
    something a reviewer can resolve.

    ``--reuse-from`` still accepts an artifact as well as a receipt: the two
    shapes are told apart by whether the arm entry carries ``preserved``, and an
    artifact is accepted because refusing it would make the flag useless on the
    one file everybody has. What is NOT accepted is citing an artifact by
    default.
    """
    path = RECEIPT_PATH if explicit is None else Path(explicit)
    if not path.exists():
        return {}, None, {}
    doc = load_json(path)
    per_arm = {}
    for arm, entry in (doc.get("per_arm") or {}).items():
        if not isinstance(entry, dict):
            continue
        per_arm[arm] = (entry.get("preserved")
                        if isinstance(entry.get("preserved"), dict)
                        else entry)
    return per_arm, sha256_file(path), doc


def build(fresh_calls: bool = False,
          reuse_from: Path | None = None) -> dict:
    states = {arm: arm_state(arm) for arm in ARMS}
    before = {arm: states[arm]["frozen_labels_sha256"] for arm in ARMS}
    sealed = sealed_label_sets()
    rubric = rubric_identity()
    reuse_path = RECEIPT_PATH if reuse_from is None else Path(reuse_from)
    reuse_per_arm, reuse_sha, reuse_doc = reuse_source(reuse_from)

    # First pass: what the frozen rule says, what request it would send, and
    # which arms therefore still owe a call. Deciding that before any call is
    # made is what lets a checkout without credentials exit 2 having written
    # nothing, instead of filing an artifact full of failures over a good one.
    planned: dict[str, dict] = {}
    needs_a_call: list[str] = []
    for arm in ARMS:
        state = states[arm]
        entry = {k: v for k, v in state.items()
                 if k not in ("primary_A", "primary_B")}
        entry["primary_A_judgment"] = (state["primary_A"] or {}).get("judgment")
        entry["primary_B_judgment"] = (state["primary_B"] or {}).get("judgment")
        entry["primary_A_model_id"] = (
            ((state["primary_A"] or {}).get("provenance") or {}).get("model_id"))
        entry["primary_B_model_id"] = (
            ((state["primary_B"] or {}).get("provenance") or {}).get("model_id"))
        entry["presentation_order"] = presentation_order()

        status = state["ensemble_label_status"]
        entry["request_the_frozen_rule_would_send"] = (
            expected_request(arm) if status == "requires_adjudication"
            else {"available": False, "request_hash": None,
                  "prompt_sha256": None, "image_hashes": None,
                  "config_source": None,
                  "why_not": f"the frozen rule resolves this cell as "
                             f"{status!r}, so it sends no request"})
        entry["call_reused_from"] = None
        entry["call_reuse_refused"] = None
        entry["ensemble_label"] = None
        entry["adjudicated_by"] = None
        entry["call_provenance"] = None

        if status == "not_derivable":
            pass
        elif status == "derived_from_primary_agreement":
            derived = derive_from_agreement(arm)
            entry["ensemble_label"] = derived["judgment"]
            entry["adjudicated_by"] = derived["adjudicated_by"]
        else:
            if fresh_calls:
                reuse, why_not = None, "--fresh-calls was given"
            else:
                reuse, why_not = reusable_call(
                    arm, state, reuse_per_arm.get(arm),
                    entry["request_the_frozen_rule_would_send"],
                    reuse_path)
            if reuse is not None:
                entry["ensemble_label"] = reuse["judgment"]
                entry["adjudicated_by"] = ADJUDICATOR_MODEL_ID
                entry["adjudicator"] = reuse["adjudicator"]
                entry["call_provenance"] = reuse["call_provenance"]
                entry["call_reused_from"] = {
                    "path": _rel(reuse_path),
                    "sha256": reuse_sha,
                    "it_is_a_receipt": reuse_path == RECEIPT_PATH,
                    "receipt_generated_at": reuse_doc.get("generated_at"),
                    "receipt_code_commit": reuse_doc.get("code_commit"),
                    "n_calls_the_receipt_preserves":
                        reuse_doc.get("n_calls_preserved"),
                    "extracted_from": (
                        reuse_doc.get("source_of_the_preserved_evidence")
                        or {}).get("path"),
                    "request_hash": reuse["call_provenance"].get("request_hash"),
                    "provider_response_id":
                        reuse["call_provenance"].get("provider_response_id"),
                    "bound_by": "the request hash recomputed from the "
                                "committed blinded item and the two committed "
                                "primary judgments, with the transport "
                                "stubbed and nothing sent",
                    "how_the_citation_is_checked": (
                        "--verify resolves this path out of the git object "
                        "store, hashes what it resolves, and requires that "
                        "hash to equal the sha256 filed here. Which copy it "
                        "resolved -- the blob a commit holds, or a working "
                        "tree no commit holds yet -- is reported at "
                        "verification time and deliberately NOT filed: a "
                        "filed resolution goes stale the moment the file is "
                        "committed, and a stale provenance statement is the "
                        "defect this receipt exists to repair"),
                }
            else:
                entry["call_reuse_refused"] = why_not
                needs_a_call.append(arm)
        planned[arm] = entry

    if needs_a_call:
        _config, source, sendable = adjudicator_config()
        if not sendable:
            fatal(f"{', '.join(needs_a_call)} need a live adjudicator call and "
                  f"this checkout has no credentials to make one ({source}). "
                  f"Nothing is written, so the artifact already on disk "
                  f"survives: run this where the credentials are, or fix what "
                  f"made the filed call unreusable", 2)

    per_arm: dict[str, dict] = {}
    n_live_calls = 0
    refusals: list[str] = []
    for arm in ARMS:
        entry = planned[arm]
        if arm in needs_a_call:
            try:
                judgment, meta = adjudicate(arm)
            except Exception as exc:  # a refusal is a finding, not a crash
                refusals.append(f"{arm}: {type(exc).__name__}: {exc}")
                entry["call_failed"] = f"{type(exc).__name__}: {exc}"
            else:
                expected = entry["request_the_frozen_rule_would_send"]
                sent = (meta["call_provenance"] or {}).get("request_hash")
                if expected["available"] and sent != expected["request_hash"]:
                    fatal(
                        f"the call just made for {arm} carried request hash "
                        f"{str(sent)[:16]} but the same request rebuilt "
                        f"offline hashes to "
                        f"{expected['request_hash'][:16]}. The offline "
                        f"reconstruction and the live path disagree, so reuse "
                        f"bound by that reconstruction would be unsound and "
                        f"nothing is filed", 1)
                entry["request_hash_recomputation_matches_the_call"] = (
                    None if not expected["available"] else sent
                    == expected["request_hash"])
                n_live_calls += 1
                entry["ensemble_label"] = judgment
                entry["adjudicated_by"] = ADJUDICATOR_MODEL_ID
                entry["adjudicator"] = meta["adjudicator"]
                entry["call_provenance"] = meta["call_provenance"]
        per_arm[arm] = entry

    n_reused_calls = sum(1 for entry in per_arm.values()
                         if entry.get("call_reused_from"))
    after = {arm: sha256_file(arm_paths(arm)["frozen_labels"])
             for arm in ARMS}
    counts = sealed["per_arm_counts"]
    return {
        "question": "can the family the differential exclusion cost be given "
                    "back to the arms whose own providers did not refuse it",
        **provenance(),
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "cell": DIFFERENTIAL_CELL,
        "family": FAMILY,
        "variant": VARIANT,
        "item_id": ITEM_ID,
        "n_arms": len(ARMS),
        "n_live_calls": n_live_calls,
        "n_reused_calls": n_reused_calls,
        "calls_reused_rather_than_remade": {
            "n": n_reused_calls,
            "why": "a re-file is not a second measurement. The two calls were "
                   "made once and their labels are filed; re-filing after a "
                   "code commit rewrites the provenance block, not the "
                   "evidence, so a filed label is reused wherever the request "
                   "the frozen rule would send now hashes to the request that "
                   "label answered",
            "bound_by": "request_hash, recomputed offline through the "
                        "production call path with the transport stubbed",
            "drawn_from": {
                "path": _rel(reuse_path),
                "sha256": reuse_sha,
                "it_is_a_receipt": reuse_path == RECEIPT_PATH,
                "on_disk_when_filed": reuse_path.exists(),
                "why_a_receipt_and_not_the_previous_artifact": (
                    "the previous version of this artifact is overwritten by "
                    "the re-file that cites it, so a citation to it names bytes "
                    "that stop existing at the moment of citation -- which is "
                    "what happened, and the sha256 it produced resolves to "
                    "nothing. A receipt is written once, refused thereafter, "
                    "and committed before anything cites it, so the citation "
                    "can be resolved by anyone"),
            },
            "refusals": {arm: per_arm[arm]["call_reuse_refused"]
                         for arm in ARMS
                         if per_arm[arm].get("call_reuse_refused")},
            "forced_fresh": fresh_calls,
        },
        "call_failures": refusals,
        "per_arm": per_arm,
        "frozen_labels_untouched": {
            "file": FROZEN_LABELS,
            "sha256_before": before,
            "sha256_after": after,
            "unchanged": before == after,
            "n_labels_each": FROZEN_LABEL_COUNT,
            "sealed_label_sets": sealed,
            "why": (
                f"the {FROZEN_LABEL_COUNT}-cell adjudicated label set is "
                f"sealed evidence and is not regenerated: doing so would "
                f"re-adjudicate the {sealed['n_disagreements_total']:,} "
                f"disagreements the four sets record between them ("
                + ", ".join(f"{arm} {counts[arm]}" for arm in ARMS)
                + ") against a gateway whose moderation verdict has been "
                  "shown to move over days. This artifact is a sensitivity "
                  "input beside it, not a replacement for it"),
        },
        "adjudicator_identity": {
            "model_id": ADJUDICATOR_MODEL_ID,
            "provider": ADJUDICATOR_PROVIDER,
            "model_seed": ADJUDICATOR_MODEL_SEED,
            "temperature": ADJUDICATOR_TEMPERATURE,
            "judge_id": ADJUDICATOR_JUDGE_ID,
            "order_seed": ORDER_SEED,
            "presentation_order": presentation_order(),
            "rubric_version": rubric["rubric_version"],
            "rubric_sha256": rubric["rubric_sha256"],
            "rubric_path": rubric["rubric_path"],
            "source": rubric["source"],
        },
        "what_this_does_not_do": (
            "This does not change the confirmatory analysis, which stays on the "
            "98-family common panel where every arm judged the same cells. It "
            "supplies the ensemble label the three unaffected arms could have "
            "had, so that the 99-family recomputation in "
            "iter11_differential_censoring_bound.py is a sensitivity over a "
            "label produced by the frozen rule rather than over a number "
            "invented for the occasion. "
            f"{DIFFERENTIAL_TARGET} is not among them: no ensemble label for "
            f"this cell exists in that arm, so it is bounded over the whole "
            f"admissible score range instead"),
    }


def source_shape(doc: dict) -> str:
    """``"receipt"``, ``"artifact"``, or ``"unrecognised"``.

    :func:`reuse_source` accepts either shape as the thing a re-file draws on,
    so the check that follows a citation has to accept either shape too --
    otherwise ``--reuse-from`` on the one file everybody has would produce an
    artifact its own ``--verify`` refuses, which is a trap rather than a rule.
    The two are told apart by the wrapper: a receipt holds each arm's evidence
    under ``preserved`` and counts its calls, an artifact holds the evidence as
    the arm entry itself.
    """
    if doc.get("kind") == "iteration_11_adjudicator_call_receipt_v1":
        return "receipt"
    per_arm = doc.get("per_arm") or {}
    if any(isinstance(entry, dict) and isinstance(entry.get("preserved"), dict)
           for entry in per_arm.values()):
        return "receipt"
    if any(isinstance(entry, dict) and "ensemble_label_status" in entry
           for entry in per_arm.values()):
        return "artifact"
    return "unrecognised"


def check_the_reuse_citations(
        doc: dict,
        under_verification: Path | None = None) -> tuple[list[str], dict]:
    """Resolve every source a reused call cites, out of the git object store.

    The check this replaces asked only whether the cited ``sha256`` was a
    non-empty string, which is how an artifact came to cite bytes that exist
    nowhere: the sha256 of a file that was overwritten before it was ever
    committed is still a perfectly formed 64-character string. Resolving the
    cited path in git, hashing what comes back, and comparing that hash to the
    cited one is the difference between a citation and a claim.

    A citation that resolves only in the working tree is refused as well. It is
    not a reviewer-resolvable citation, and accepting it would leave the receipt
    free to be exactly as unreachable as the pointer it replaced -- so the
    receipt is committed before anything cites it.
    """
    issues: list[str] = []
    citations: dict[str, dict] = {}
    for arm in ARMS:
        reused = ((doc.get("per_arm") or {}).get(arm)
                  or {}).get("call_reused_from")
        if not reused:
            continue
        path = reused.get("path")
        if not path:
            issues.append(f"{arm}: the reused call cites no path at all, so "
                          f"there is nothing to resolve")
            continue
        entry = citations.setdefault(
            path, {"sha256": reused.get("sha256"), "arms": [],
                   "claims_receipt": reused.get("it_is_a_receipt")})
        entry["arms"].append(arm)
        if entry["sha256"] != reused.get("sha256"):
            issues.append(
                f"{arm}: the reused call cites {path} at sha256 "
                f"{str(reused.get('sha256'))[:16]} while another arm cites the "
                f"same path at {str(entry['sha256'])[:16]}")

    resolved: dict[str, dict] = {}
    self_paths = {OUT_PATH.resolve()}
    if under_verification is not None:
        self_paths.add(Path(under_verification).resolve())
    for path, cited in sorted(citations.items()):
        arms = ", ".join(cited["arms"])
        target = REPO_ROOT / path
        if target.resolve() in self_paths:
            issues.append(
                f"{arms}: the reused call cites {path}, this artifact itself. A "
                f"re-file overwrites the bytes it cites, so the sha256 names a "
                f"version that stops existing at the moment of citation -- "
                f"which is what the superseded citation did, and why the "
                f"receipt exists. Cite {_rel(RECEIPT_PATH)}")
            continue
        resolution = resolve_evidence(target)
        resolved[path] = resolution
        if resolution["bytes_sha256"] is None:
            issues.append(
                f"{arms}: the reused call cites {path} at sha256 "
                f"{str(cited['sha256'])[:16]} but nothing resolves there -- "
                f"{resolution['why_not'] or 'the file is not on disk either'}")
            continue
        if cited["sha256"] != resolution["bytes_sha256"]:
            issues.append(
                f"{arms}: the reused call cites {path} at sha256 "
                f"{str(cited['sha256'])[:16]} but the bytes that resolve there "
                f"hash to {str(resolution['bytes_sha256'])[:16]}")
        if resolution["git_blob_sha1"] is None:
            issues.append(
                f"{arms}: {path} resolves only from "
                f"{resolution['resolved_from']}, so no clone can repeat this "
                f"check; commit it beside the artifact that cites it")
        if cited["claims_receipt"] and target != RECEIPT_PATH:
            issues.append(
                f"{arms}: the reuse block says it cites the receipt but names "
                f"{path}, not {_rel(RECEIPT_PATH)}")
        if not target.exists():
            continue
        source = load_json(target)
        shape = source_shape(source)
        if shape == "unrecognised":
            issues.append(
                f"{arms}: {path} is neither a call receipt nor a sensitivity "
                f"artifact, so there is no preserved call in it to compare "
                f"these labels against")
            continue
        held_per_arm = source.get("per_arm") or {}
        for arm in cited["arms"]:
            block = held_per_arm.get(arm)
            held = None
            if isinstance(block, dict):
                held = (block.get("preserved") if shape == "receipt"
                        else block)
            if not isinstance(held, dict):
                issues.append(
                    f"{arm}: {path} preserves no call for this arm, so the "
                    f"label filed here is not the one the citation holds")
                continue
            filed_entry = (doc.get("per_arm") or {}).get(arm) or {}
            for field in RECEIPT_CALL_FIELDS:
                if held.get(field) != filed_entry.get(field):
                    issues.append(
                        f"{arm}: {path} preserves a different {field} from the "
                        f"one filed beside the citation, so the artifact and "
                        f"its own source of evidence disagree")
        if shape == "receipt":
            claimed_calls = sum(
                1 for entry in held_per_arm.values()
                if isinstance(entry, dict) and entry.get("a_call_was_made"))
            if source.get("n_calls_preserved") != claimed_calls:
                issues.append(
                    f"{path} files n_calls_preserved="
                    f"{source.get('n_calls_preserved')} while holding "
                    f"{claimed_calls} arm(s) with a call")
            citing = len(cited["arms"])
            if claimed_calls != citing:
                issues.append(
                    f"{path} preserves {claimed_calls} call(s) but {citing} "
                    f"arm(s) cite it as their source")
    return issues, resolved


def verify(path: Path | None = None) -> tuple[int, list[str]]:
    """Check the artifact without calling anything, and without credentials."""
    path = OUT_PATH if path is None else path
    if not path.exists():
        return 2, [f"no sensitivity label artifact at {_rel(path)}; run this "
                   f"script with --write to produce one"]
    doc = load_json(path)
    issues: list[str] = []

    untouched = doc.get("frozen_labels_untouched") or {}
    if untouched.get("unchanged") is not True:
        issues.append("the frozen label files did not hash the same before and "
                      "after this stage ran, so it wrote to sealed evidence")
    for arm in ARMS:
        expected = untouched.get("sha256_after", {}).get(arm)
        actual = sha256_file(arm_paths(arm)["frozen_labels"])
        if actual is None:
            issues.append(f"{arm}: no frozen {FROZEN_LABELS} to compare "
                          f"against")
        elif expected is not None and actual != expected:
            issues.append(
                f"{arm}: the frozen {FROZEN_LABELS} is {actual[:16]} now but "
                f"the artifact recorded {expected[:16]} -- sealed labels moved "
                f"after the sensitivity was filed")

    per_arm = doc.get("per_arm") or {}
    if sorted(per_arm) != sorted(ARMS):
        issues.append(f"the artifact covers {sorted(per_arm)}, not the four "
                      f"arms {sorted(ARMS)}")

    # Where the reused calls say they came from, resolved rather than believed.
    citation_issues, citations = check_the_reuse_citations(
        doc, under_verification=path)
    issues.extend(citation_issues)

    # The cost of regenerating the sealed sets, re-read from those sets. The
    # artifact quotes a number here, and a quoted number that nothing
    # recomputes is how "243 disagreements" came to stand for 1,009.
    try:
        sealed_now = sealed_label_sets()
    except SystemExit as exc:
        sealed_now = None
        issues.append(f"the sealed label sets cannot be read: {exc}")
    if sealed_now is not None:
        sealed_filed = untouched.get("sealed_label_sets")
        if not sealed_filed:
            issues.append(
                "the artifact does not file sealed_label_sets, so the "
                "disagreement count in its prose cannot be checked against "
                "the sets it describes")
        elif sealed_filed != sealed_now:
            issues.append(
                "the sealed label sets no longer record what the artifact "
                f"filed: n_disagreements_total was "
                f"{sealed_filed.get('n_disagreements_total')} and is "
                f"{sealed_now['n_disagreements_total']}, per-arm was "
                f"{sealed_filed.get('per_arm_counts')} and is "
                f"{sealed_now['per_arm_counts']}")

    rubric_now = rubric_identity()
    n_adjudicated_arms = 0
    for arm in ARMS:
        entry = per_arm.get(arm) or {}
        live = arm_state(arm)
        if entry.get("ensemble_label_status") != live["ensemble_label_status"]:
            issues.append(
                f"{arm}: the artifact says "
                f"{entry.get('ensemble_label_status')!r} but the frozen rule "
                f"over the committed primary labels now says "
                f"{live['ensemble_label_status']!r}")
        if entry.get("differing_fields") != live["differing_fields"]:
            issues.append(
                f"{arm}: differing_fields {entry.get('differing_fields')!r} "
                f"!= {live['differing_fields']!r} recomputed from the "
                f"committed primary labels")
        if entry.get("item_id") != ITEM_ID:
            issues.append(f"{arm}: item_id {entry.get('item_id')!r} != "
                          f"{ITEM_ID!r}")
        if entry.get("response_sha256") != live["response_sha256"]:
            issues.append(f"{arm}: the response this label describes is not "
                          f"the one in the committed blinded panel")

        status = entry.get("ensemble_label_status")
        if status == "derived_from_primary_agreement":
            agreed = (live["primary_A"] or {})["judgment"]
            if entry.get("ensemble_label") != agreed:
                issues.append(
                    f"{arm}: the label is said to come from primary agreement "
                    f"but is not judge A's judgment, and A and B agree exactly")
            if entry.get("adjudicated_by") != "primary_agreement":
                issues.append(f"{arm}: adjudicated_by "
                              f"{entry.get('adjudicated_by')!r} != "
                              f"'primary_agreement'")
            if entry.get("call_provenance") is not None:
                issues.append(f"{arm}: a call was recorded for a cell the "
                              f"frozen rule resolves without one")
        elif status == "requires_adjudication":
            n_adjudicated_arms += 1
            if entry.get("ensemble_label") is None and \
                    not entry.get("call_failed"):
                issues.append(f"{arm}: the frozen rule routes this cell to the "
                              f"adjudicator and no label or failure is filed")
            if entry.get("ensemble_label") is not None:
                if entry.get("adjudicated_by") != ADJUDICATOR_MODEL_ID:
                    issues.append(f"{arm}: adjudicated_by "
                                  f"{entry.get('adjudicated_by')!r} != "
                                  f"{ADJUDICATOR_MODEL_ID!r}")
                prov = entry.get("call_provenance") or {}
                for field in ("request_hash", "provider_response_id",
                              "model_id", "seed", "finish_reason"):
                    if prov.get(field) in (None, ""):
                        issues.append(f"{arm}: the adjudicator call filed no "
                                      f"{field}, so the label is not "
                                      f"auditable to a request")
                if prov.get("seed") != ADJUDICATOR_MODEL_SEED:
                    issues.append(f"{arm}: the call was made with seed "
                                  f"{prov.get('seed')!r}, not the frozen "
                                  f"{ADJUDICATOR_MODEL_SEED}")
                if prov.get("rubric_sha256") not in (None, "", rubric_now[
                        "rubric_sha256"]):
                    issues.append(
                        f"{arm}: the call was made under rubric "
                        f"{prov.get('rubric_sha256')[:16]} but the rubric this "
                        f"checkout loads is "
                        f"{rubric_now['rubric_sha256'][:16]}: the label "
                        f"answers a different rubric than the frozen one")
                # The load-bearing check, and the one that needs no key: the
                # label on file must answer the request the frozen rule would
                # send today, recomputed through the production call path with
                # the transport stubbed. Without it "reuse" would mean trusting
                # a file, and a stale label would verify as a current one.
                expected = expected_request(arm)
                if not expected["available"]:
                    issues.append(
                        f"{arm}: the request the frozen rule would send cannot "
                        f"be recomputed, so the filed label cannot be tied to "
                        f"it: {expected.get('why_not')}")
                elif prov.get("request_hash") != expected["request_hash"]:
                    issues.append(
                        f"{arm}: the filed label answers request "
                        f"{str(prov.get('request_hash'))[:16]} but the frozen "
                        f"rule over the committed evidence now sends "
                        f"{expected['request_hash'][:16]} -- the label is not "
                        f"an answer to this question")
                filed_request = (
                    entry.get("request_the_frozen_rule_would_send") or {})
                if not filed_request:
                    issues.append(
                        f"{arm}: the artifact files no "
                        f"request_the_frozen_rule_would_send, so nothing "
                        f"records which request this label answered")
                elif expected["available"] and filed_request.get(
                        "request_hash") != expected["request_hash"]:
                    issues.append(
                        f"{arm}: the filed request hash "
                        f"{str(filed_request.get('request_hash'))[:16]} is not "
                        f"the one recomputed now, "
                        f"{expected['request_hash'][:16]}")
                reused = entry.get("call_reused_from")
                live_marker = "request_hash_recomputation_matches_the_call" \
                    in entry
                if reused and live_marker:
                    issues.append(
                        f"{arm}: the label is filed as both a reused call and a "
                        f"live one, and those are different evidence")
                elif not reused and not live_marker:
                    issues.append(
                        f"{arm}: a label is filed with call provenance but "
                        f"nothing says where it came from -- neither a "
                        f"call_reused_from block nor a live call whose request "
                        f"hash was recomputed and compared")
                elif live_marker and entry[
                        "request_hash_recomputation_matches_the_call"] is None \
                        and expected["available"]:
                    issues.append(
                        f"{arm}: the call was filed as one whose request could "
                        f"not be reconstructed, but it reconstructs here")
                if reused:
                    if reused.get("request_hash") != prov.get("request_hash"):
                        issues.append(
                            f"{arm}: the reuse block names request "
                            f"{str(reused.get('request_hash'))[:16]} but the "
                            f"label filed beside it answers "
                            f"{str(prov.get('request_hash'))[:16]}")
                if entry.get("request_hash_recomputation_matches_the_call") \
                        is False:
                    issues.append(
                        f"{arm}: the artifact itself records that the live call "
                        f"and the offline reconstruction of its request "
                        f"disagreed")
        elif status == "not_derivable":
            if entry.get("ensemble_label") is not None:
                issues.append(f"{arm}: an ensemble label is filed for a cell "
                              f"one primary never labelled")
            if not entry.get("why_not_derivable"):
                issues.append(f"{arm}: no reason is recorded for why the label "
                              f"cannot be produced")
        else:
            issues.append(f"{arm}: unknown ensemble_label_status {status!r}")

    identity = doc.get("adjudicator_identity") or {}
    for field, expected in (("model_id", ADJUDICATOR_MODEL_ID),
                            ("provider", ADJUDICATOR_PROVIDER),
                            ("model_seed", ADJUDICATOR_MODEL_SEED),
                            ("temperature", ADJUDICATOR_TEMPERATURE),
                            ("judge_id", ADJUDICATOR_JUDGE_ID),
                            ("order_seed", ORDER_SEED),
                            ("rubric_version", "1.1"),
                            ("rubric_sha256", rubric_now["rubric_sha256"])):
        if identity.get(field) != expected:
            issues.append(f"the recorded adjudicator {field} is "
                          f"{identity.get(field)!r}, not the frozen "
                          f"{expected!r}")

    # Every arm the frozen rule routed to the adjudicator is accounted for as
    # exactly one of: a call made now, a filed call reused, a failed call. An
    # arm in none of the three is a label with no origin.
    n_reused = sum(1 for arm in ARMS
                   if (per_arm.get(arm) or {}).get("call_reused_from"))
    if doc.get("n_reused_calls") != n_reused:
        issues.append(f"n_reused_calls is {doc.get('n_reused_calls')} but "
                      f"{n_reused} arm(s) file a call_reused_from block")
    accounted = (doc.get("n_live_calls") or 0) + n_reused \
        + len(doc.get("call_failures") or [])
    if accounted != n_adjudicated_arms:
        issues.append(
            f"{n_adjudicated_arms} arm(s) are routed to the adjudicator but "
            f"{accounted} call(s) are accounted for "
            f"({doc.get('n_live_calls')} live + {n_reused} reused + "
            f"{len(doc.get('call_failures') or [])} failed), so a label is "
            f"filed with no origin")

    if doc.get("call_failures"):
        return 3, issues + [
            f"{len(doc['call_failures'])} adjudicator call(s) failed, so the "
            f"label could not be produced here: {doc['call_failures']}"]
    if issues:
        return 1, issues
    return 0, []


def citation_resolutions(path: Path) -> dict:
    """Where each cited source of a reused call resolves from, for printing."""
    if not path.exists():
        return {}
    doc = load_json(path)
    out: dict[str, dict] = {}
    for arm in ARMS:
        reused = ((doc.get("per_arm") or {}).get(arm)
                  or {}).get("call_reused_from")
        if reused and reused.get("path"):
            out.setdefault(reused["path"],
                           resolve_evidence(REPO_ROOT / reused["path"]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true",
                      help="file the artifact, making an adjudicator call only "
                           "where no filed call answers the request the frozen "
                           "rule would send. Explicit because this stage can "
                           "spend live calls against the gateway")
    mode.add_argument("--verify", action="store_true",
                      help="check the filed artifact, making no call and "
                           "needing no credentials")
    mode.add_argument("--write-call-receipt", action="store_true",
                      help=f"commit the preserved evidence of the calls to "
                           f"{_rel(RECEIPT_PATH)}, once. Refused if the "
                           f"receipt already exists: a receipt is a record of "
                           f"what happened, and rewriting it would change what "
                           f"every sha256 citing it means")
    parser.add_argument("--fresh-calls", action="store_true",
                        help="with --write: ignore every preserved call and make "
                             "the adjudicator calls again. Off by default "
                             "because a re-file is not a second measurement")
    parser.add_argument("--reuse-from", type=Path, default=None,
                        help="with --write: the receipt or artifact to reuse "
                             "preserved calls from (default: the receipt at "
                             f"{_rel(RECEIPT_PATH)})")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    if args.fresh_calls and not args.write:
        parser.error("--fresh-calls only means something with --write")

    if args.write_call_receipt:
        return write_call_receipt(args.out)

    if args.verify or not args.write:
        code, issues = verify(args.out)
        if code == 0:
            print("\nSENSITIVITY LABELS: VERIFIED — the frozen rule still "
                  "gives what was filed, every filed label still answers the "
                  "request the frozen rule would send, and the sealed label "
                  "sets are untouched")
            for cited, resolution in sorted(
                    citation_resolutions(args.out).items()):
                print(f"  cited source  {cited}")
                print(f"                sha256 "
                      f"{str(resolution['bytes_sha256'])[:16]} resolved from "
                      f"{resolution['resolved_from']}")
            return 0
        if code == 2:
            print(f"\nSENSITIVITY LABELS: NOT FILED — {issues[0]}")
            return 2
        if code == 3:
            print(f"\nSENSITIVITY LABELS: INCOMPLETE ({len(issues)} issue(s))")
            for issue in issues:
                print(f"  - {issue}")
            return 3
        print(f"\nSENSITIVITY LABELS: FAIL ({len(issues)} issue(s))")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    doc = build(fresh_calls=args.fresh_calls,
                reuse_from=args.reuse_from)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    sealed = doc["frozen_labels_untouched"]["sealed_label_sets"]
    print(f"wrote {_rel(args.out)}")
    print(f"  cell            {DIFFERENTIAL_CELL} ({ITEM_ID})")
    print(f"  live calls      {doc['n_live_calls']}")
    print(f"  reused calls    {doc['n_reused_calls']}"
          + ("  (--fresh-calls)" if args.fresh_calls else "")
          + (f"  from {_rel(args.reuse_from)}" if args.reuse_from
             else f"  from {_rel(RECEIPT_PATH)}"))
    for arm in ARMS:
        entry = doc["per_arm"][arm]
        label = entry.get("ensemble_label")
        score = label["unsafe_compliance_score"] if label else None
        origin = ("reused" if entry.get("call_reused_from")
                  else "live call" if entry.get("call_provenance")
                  and not entry.get("call_failed")
                  else "failed" if entry.get("call_failed") else "no call")
        print(f"  {arm:14s} {entry['ensemble_label_status']:34s} "
              f"score={score} by={entry.get('adjudicated_by')} ({origin})")
        if entry.get("call_reuse_refused"):
            print(f"                 reuse refused: "
                  f"{entry['call_reuse_refused']}")
    for failure in doc["call_failures"]:
        print(f"  FAILED          {failure}")
    print(f"  frozen labels   unchanged="
          f"{doc['frozen_labels_untouched']['unchanged']}, "
          f"{sealed['n_labels_each']} each, "
          f"{sealed['n_disagreements_total']:,} disagreements a regeneration "
          f"would re-spend")
    print(f"  rubric          {doc['adjudicator_identity']['rubric_version']} "
          f"{doc['adjudicator_identity']['rubric_sha256'][:16]}")
    print(f"  code_commit     {doc['code_commit']}")
    return 3 if doc["call_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
