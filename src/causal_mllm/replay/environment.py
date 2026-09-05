"""Iteration 11 environment certification.

Shared by the technical preflight (11.2-11.4), the 11.5 eligibility run and
anything else that mints Iteration 11 evidence, so that "is this
environment certifiable?" has ONE answer rather than one per script.

The frozen protocol names a ``reference_env`` and a set of
``reference_versions``. Both are checked here against the interpreter that
is actually running, because an artifact that records versions nobody
measured is indistinguishable from one that records the truth.
"""

from __future__ import annotations

import os

from causal_mllm.replay.registry import dependency_lock_snapshot


def certify_environment(protocol: dict) -> tuple[dict, list[str]]:
    """Certify the environment this process is running in.

    Three separate questions, with deliberately different severities:

    1. Does the environment hold a third-party editable install? **Fatal.**
       ``pip freeze`` identifies an editable dependency by the sibling
       repository's COMMITTED HEAD and is blind to that repository's
       uncommitted working-tree changes, so no hash of freeze output can
       prove which dependency source would execute — and a local-path
       editable install names no revision at all. Evidence minted here
       could not be reproduced, so every stage that mints Iteration 11
       evidence refuses before loading a checkpoint.
    2. Do the runtime versions match the frozen ``reference_versions``?
       **Fatal** on mismatch — these are what determine model behaviour, so
       a different transformers or torch is a different measurement
       instrument.
    3. Is the conda environment NAME the frozen ``reference_env``?
       **Recorded as an explicit deviation, NOT fatal.** The name labels
       where the frozen versions were observed; a dedicated clone carrying
       byte-identical versions preserves every scientific property while
       removing the editable install question 1 forbids. The frozen
       protocol file is never edited to accommodate this — the deviation is
       declared in the artifact instead.

    Returns:
        ``(values, problems)``: the recorded environment identity, and the
        list of violations (empty means certifiable).

    Note:
        ``dependency_lock_snapshot`` raises rather than returning a partial
        snapshot when ``pip freeze`` fails, so a failed probe is never
        recorded as a clean environment.
    """
    snapshot = dependency_lock_snapshot()
    # EVERY editable form counts, not just the ones naming a revision: a
    # local-path or file:// editable install records no revision at all and
    # is therefore the most mutable case, yet it used to be invisible here.
    offenders = dict(snapshot.get("editable_installs") or {})
    problems: list[str] = []
    if offenders:
        detail = "; ".join(
            f"{name} ({info.get('kind')}: {info.get('target')})"
            for name, info in sorted(offenders.items()))
        problems.append(
            f"environment holds third-party editable install(s): {detail}. An "
            f"editable dependency's source can change without anything in a "
            f"`pip freeze` hash moving — a VCS install hides uncommitted "
            f"sibling changes behind its committed HEAD, and a local-path "
            f"install names no revision at all — so this environment cannot "
            f"be certified reproducible. Use a dedicated Iteration 11 "
            f"environment with no third-party editable installs.")

    frozen_lock = protocol.get("dependency_lock") or {}
    frozen_versions = dict(frozen_lock.get("reference_versions") or {})
    observed: dict = {}
    try:
        import torch
        import transformers
        observed = {"transformers": transformers.__version__,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda}
    except ImportError as exc:
        # Reported as its own problem rather than left to surface as three
        # version mismatches: "could not read the versions" and "read
        # different versions" need different fixes.
        problems.append(f"cannot read runtime versions: {exc}")
    mismatched = {
        key: {"frozen": frozen_versions.get(key), "observed": observed.get(key)}
        for key in sorted(frozen_versions)
        if observed.get(key) != frozen_versions.get(key)
    }
    if mismatched:
        detail = "; ".join(
            f"{k}: frozen={v['frozen']!r} observed={v['observed']!r}"
            for k, v in mismatched.items())
        problems.append(
            f"runtime versions do not match the frozen reference_versions — "
            f"{detail}")

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    frozen_env = frozen_lock.get("reference_env")
    values = {
        "python_version": snapshot.get("python_version"),
        "executable": snapshot.get("executable"),
        "conda_env": conda_env,
        "n_packages": snapshot.get("n_packages"),
        "pip_freeze_sha256": snapshot.get("pip_freeze_sha256"),
        "pyproject_sha256": snapshot.get("pyproject_sha256"),
        "excluded_self_distributions":
            snapshot.get("excluded_self_distributions"),
        "third_party_editable_installs": offenders,
        "third_party_editable_vcs": {
            name: info.get("revision")
            for name, info in offenders.items() if info.get("revision")},
        "observed_versions": observed,
        "frozen_reference_versions": frozen_versions,
        "frozen_reference_env": frozen_env,
        "reference_env_matches_frozen": conda_env == frozen_env,
    }
    if conda_env != frozen_env:
        values["reference_env_deviation"] = {
            "claim": f"the frozen protocol names reference_env={frozen_env!r}",
            "observation": f"this process runs in conda env {conda_env!r}",
            "rationale": (
                "a dedicated clone of that environment with the third-party "
                "editable install removed; every frozen reference_version "
                "matches exactly, so only the environment NAME differs. The "
                "clone is required because the shared environment carries an "
                "editable sibling install whose source can change without its "
                "recorded revision moving, which is not certifiable. The "
                "frozen protocol file is immutable and was not edited."),
            "frozen_protocol_modified": False,
        }
    return values, problems
