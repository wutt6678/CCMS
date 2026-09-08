"""Re-deriving a committed artifact on a machine that did not make it.

Every ``--verify`` gate in this repository re-derives an artifact and compares
it with the committed one. Until now that comparison was ``fresh != committed``
over the parsed JSON, which is exact equality on every leaf including every
float. Exact equality is the right demand inside one interpreter, and it is not
a demand this repository can make of any other one.

MEASURED, NOT ARGUED. The frozen bootstrap in
:mod:`causal_mllm.evaluation.bootstrap` is pure Python -- no NumPy anywhere in
it -- and it averages each resample with the builtin ``sum``. CPython 3.12
changed ``sum`` to use Neumaier summation for floats, so the same seed, the same
resample indices and the same inputs give a more accurate and therefore
different last bit. Running the 11.8 derivation under /usr/bin/python3.12.3
against artifacts written under the certified 3.10.20:

* ``cross_model_analysis.json`` -- 166 differing leaves, ALL of them numeric,
  ZERO non-numeric. 155 are continuous quantities (means, intervals, standard
  deviations, shifts) with a worst absolute difference of 2.61e-15 and a worst
  relative difference of 1.22e-12. The other 11 are p-values, which move
  DISCRETELY because a bootstrap p-value is a count of resamples on one side of
  the null: every observed movement is a multiple of 2/5000, the largest being
  0.0044, eleven resamples. Every verdict, sign, count and string is identical,
  including H2's ``confirmed`` at an adjusted p of 0.0272.
* ``reference_restriction.json`` -- 34 differing leaves, all numeric, worst
  absolute 5.55e-16, worst relative 6.37e-15.
* the sealed Iteration 10 reference mean ``0.11506760000000027`` becomes
  ``0.11506759999999999`` -- a difference of -2.78e-16 -- and its published
  ``CI_upper`` ``0.17999999999999997`` becomes ``0.18``. Its ``CI_lower`` is
  exactly ``0.0495`` under both.

So the exact-equality verifier did not fail because the analysis was wrong. It
failed because it demanded bit equality from a quantity whose last bit is a
property of the interpreter that summed it.

THE RULE THIS MODULE IMPLEMENTS.

1. Non-numeric leaves are compared EXACTLY, always, in every environment. That
   includes strings, booleans, and integers -- and an integer that arrives as a
   float is a type change, which is non-numeric drift, not rounding. Every
   verdict, sign, hypothesis name, family id and count in the artifact is
   therefore still reproduced bit for bit, which is what makes the numeric
   slack safe: a p-value moving inside its tolerance can never carry a verdict
   with it, because the verdict is a string and strings are exact.
2. Continuous floats are compared with :data:`FLOAT_TOLERANCE`, the same 1e-12
   ``iter11_reference_restriction.py`` already documents for its reproduction
   self-check and already records inside the artifact it writes. Reusing it
   matters: that script was already telling the truth about floating-point
   portability in one comparison and demanding bit equality in the next.
3. p-values get their own tolerance, in RESAMPLE STEPS rather than in absolute
   units, because they are counts and can only move in steps of
   ``2 / n_bootstrap``. An absolute 1e-12 on a p-value is not a tight standard,
   it is an impossible one.
4. The tolerance is licensed ONLY by a demonstrated deviation from the recorded
   dependency lock. Inside the certified environment the comparison is exact,
   because there the last bit is reproducible and a difference is a defect.
   Absence of a lock licenses nothing: not being able to tell which environment
   you are in is not evidence that it deviates.

A verification that needed the tolerance returns a distinct code rather than
the same 0 an exact reproduction returns, and names the deviation that licensed
it. "Reproduced" and "reproduced within tolerance under a different
interpreter" are different statements and the exit code keeps them apart.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from causal_mllm.replay.registry import (
    load_dependency_lock,
    verify_active_dependency_lock,
)

#: Absolute agreement required of a continuous float. Twelve orders of
#: magnitude above the worst difference measured across two interpreters
#: (2.61e-15) and nine orders below anything that would change a reported
#: digit: the artifacts print means to six decimals and intervals to four.
FLOAT_TOLERANCE = 1e-12

#: How many resamples may change side of the null before a p-value is treated
#: as a disagreement rather than as summation noise. The worst movement measured
#: across CPython 3.10.20 and 3.12.3 was 11 of 5000; 16 is the next power of
#: two, chosen so the bound is not fitted to the single observation it was
#: derived from. It is also smaller than the distance that matters: the nearest
#: raw p-value to its Holm critical value is 0.0136 against 0.025, a gap of
#: 0.0114, and the nearest adjusted p to alpha=0.05 is 0.0272, a gap of 0.0228.
#: At 5000 resamples this bound is 16 * 2/5000 = 0.0064, which cannot reach
#: either -- and rule 1 above means a verdict could not move with it anyway.
P_VALUE_TOLERANCE_STEPS = 16

#: The frozen protocol's resample count. A p-value's granularity is
#: ``2 / n_bootstrap``, so the tolerance has to be expressed in steps and
#: scaled rather than written down as a number.
DEFAULT_N_BOOTSTRAP = 5000

#: Field names holding a bootstrap p-value. Matched on the leaf name AND on its
#: parent, because the same quantity appears both as a field
#: (``verdicts.H2.raw_p``) and as a value in a map keyed by hypothesis
#: (``holm_bonferroni.raw_p.H2``). A rule that reads only the leaf classifies
#: the second as a continuous quantity and holds a COUNT of resamples to a
#: floating-point standard, which is how a verifier comes to fail on 0.0008 of
#: p-value while accepting 2.6e-15 of mean.
P_VALUE_FIELD_NAMES = frozenset({
    "bootstrap_p_two_sided",
    "raw_p",
    "adjusted_p",
    "confirmatory_p",
    "sensitivity_p",
    "p_value",
    "two_sided_p",
})


def p_value_tolerance(n_bootstrap: int = DEFAULT_N_BOOTSTRAP) -> float:
    """How far a p-value may move and still be the same measurement."""
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    return P_VALUE_TOLERANCE_STEPS * 2.0 / n_bootstrap


def is_p_value_path(path: str) -> bool:
    """Whether a JSON path from :func:`differing_leaves` holds a p-value."""
    parts = [part for part in
             path.replace("[", ".").replace("]", "").split(".") if part]
    return any(part in P_VALUE_FIELD_NAMES for part in parts[-2:])


def differing_leaves(filed: Any, fresh: Any,
                     path: str = "$") -> Iterator[tuple[str, Any, Any]]:
    """Every leaf where two parsed artifacts disagree, by JSON path.

    A type change is a difference even when ``==`` says otherwise: ``100`` and
    ``100.0`` are equal in Python and are not the same JSON value, and a count
    that started arriving as a float is a change in what the field is.
    """
    if isinstance(filed, dict) and isinstance(fresh, dict):
        for key in sorted(set(filed) | set(fresh)):
            yield from differing_leaves(filed.get(key), fresh.get(key),
                                        f"{path}.{key}")
        return
    if isinstance(filed, list) and isinstance(fresh, list):
        if len(filed) != len(fresh):
            yield path, f"{len(filed)} element(s)", f"{len(fresh)} element(s)"
            return
        for index, (a, b) in enumerate(zip(filed, fresh)):
            yield from differing_leaves(a, b, f"{path}[{index}]")
        return
    if type(filed) is not type(fresh):
        yield path, filed, fresh
        return
    if filed != fresh:
        yield path, filed, fresh


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare(filed: dict, fresh: dict, *, tolerate_numerics: bool,
            n_bootstrap: int = DEFAULT_N_BOOTSTRAP) -> dict:
    """Compare a re-derivation against the committed artifact.

    ``tolerate_numerics`` is decided by the caller from the environment, not
    here: this function reports what moved and whether the movement is inside
    the documented tolerance, and never licenses the tolerance itself.
    """
    numeric: list[dict] = []
    nonnumeric: list[dict] = []
    p_tolerance = p_value_tolerance(n_bootstrap)
    for path, a, b in differing_leaves(filed, fresh):
        if not (_is_number(a) and _is_number(b)) \
                or type(a) is not type(b):
            nonnumeric.append({"path": path, "filed": a, "fresh": b})
            continue
        if isinstance(a, int):
            # A count. Exact or wrong: there is no summation order in an
            # integer, so tolerance here would only hide a changed family set.
            within = False
            kind = "count"
        elif is_p_value_path(path):
            within = abs(a - b) <= p_tolerance
            kind = "p_value"
        else:
            within = abs(a - b) <= FLOAT_TOLERANCE
            kind = "continuous"
        numeric.append({
            "path": path, "filed": a, "fresh": b, "kind": kind,
            "absolute_difference": abs(a - b),
            "relative_difference": (abs(a - b) / abs(a)) if a else None,
            "within_tolerance": within,
        })
    continuous = [d for d in numeric if d["kind"] == "continuous"]
    p_values = [d for d in numeric if d["kind"] == "p_value"]
    counts = [d for d in numeric if d["kind"] == "count"]
    outside = [d for d in numeric if not d["within_tolerance"]]
    equal = not nonnumeric and not outside and not (
        numeric and not tolerate_numerics)
    return {
        "equal": equal,
        "exactly_equal": not numeric and not nonnumeric,
        "tolerance_applied": bool(numeric) and tolerate_numerics,
        "n_differing_leaves": len(numeric) + len(nonnumeric),
        "n_numeric_differences": len(numeric),
        "n_nonnumeric_differences": len(nonnumeric),
        "nonnumeric_differences": nonnumeric[:20],
        "numeric_differences_outside_tolerance": outside[:20],
        "worst_continuous_absolute_difference": max(
            (d["absolute_difference"] for d in continuous), default=0.0),
        "worst_continuous_relative_difference": max(
            (d["relative_difference"] for d in continuous
             if d["relative_difference"] is not None), default=0.0),
        "worst_p_value_absolute_difference": max(
            (d["absolute_difference"] for d in p_values), default=0.0),
        "worst_p_value_movement_in_resample_steps": int(round(
            max((d["absolute_difference"] for d in p_values), default=0.0)
            / (2.0 / n_bootstrap))) if p_values else 0,
        "n_count_differences": len(counts),
        "float_tolerance": FLOAT_TOLERANCE,
        "p_value_tolerance": p_tolerance,
        "p_value_tolerance_in_resample_steps": P_VALUE_TOLERANCE_STEPS,
        "n_bootstrap": n_bootstrap,
        "differing_top_level_keys": sorted({
            path.split(".", 2)[1].split("[")[0]
            for path in
            [d["path"] for d in numeric] + [d["path"] for d in nonnumeric]
            if path.startswith("$.")}),
    }


def environment_deviation(lock_path: str | Path | None = None) -> dict:
    """Whether THIS interpreter is the one the lock certifies.

    The tolerance in this module is licensed by a measured deviation and by
    nothing else, so the measurement is the gate: no lock recorded means no
    deviation can be demonstrated, which means no tolerance.

    Measured in two stages, because the cheap one is the one that actually
    decides the last bit and the expensive one is not always available. The
    interpreter version needs nothing but ``sys.version``; the package set needs
    ``pip freeze``, and a minimal interpreter -- the kind a reviewer reaches for
    to check portability, and the kind that produced the measurement this module
    documents -- often has no pip at all. Failing closed there would deny the
    tolerance to exactly the machine that needs it, so a differing
    ``python_version`` is itself a demonstrated deviation and is reported with
    ``package_set_comparable: false`` rather than being treated as an inability
    to say anything.
    """
    try:
        recorded = load_dependency_lock(lock_path)
    except Exception as exc:
        recorded = None
        unreadable = f"{type(exc).__name__}: {exc}"
    else:
        unreadable = None
    active_python = sys.version.split()[0]
    if recorded is None:
        return {
            "certified": False,
            "deviates": False,
            "package_set_comparable": False,
            "reason": (f"the recorded dependency lock could not be read "
                       f"({unreadable}), so no deviation from it can be "
                       f"demonstrated and the numeric tolerance is not "
                       f"licensed; the comparison is exact")
            if unreadable else
            "no dependency lock is recorded, so no deviation from it can be "
            "demonstrated and the numeric tolerance is not licensed; the "
            "comparison is exact",
            "differences": {},
            "active_python_version": active_python,
            "recorded_python_version": None,
        }
    differences: dict = {}
    if recorded.get("python_version") != active_python:
        differences["python_version"] = {
            "locked": recorded.get("python_version"),
            "active": active_python}
    try:
        active = verify_active_dependency_lock(lock_path, strict=False)
    except Exception as exc:
        return {
            "certified": False,
            "deviates": bool(differences),
            "package_set_comparable": False,
            "reason": ("the active interpreter is "
                       f"{active_python} and the lock records "
                       f"{recorded.get('python_version')}, which is a "
                       "demonstrated deviation; the installed package set "
                       "could not be compared at all "
                       f"({type(exc).__name__}: {exc})")
            if differences else
            f"the active interpreter matches the recorded "
            f"{active_python}, but the installed package set could not be "
            f"measured ({type(exc).__name__}: {exc}), so the environment is "
            f"not certified and no deviation from it is demonstrated either; "
            f"the comparison stays exact",
            "differences": differences,
            "active_python_version": active_python,
            "recorded_python_version": recorded.get("python_version"),
        }
    if active.get("reason") == "no_dependency_lock_recorded":
        return {
            "certified": False,
            "deviates": False,
            "package_set_comparable": False,
            "reason": "no dependency lock is recorded, so no deviation from it "
                      "can be demonstrated and the numeric tolerance is not "
                      "licensed; the comparison is exact",
            "differences": {},
            "checked_fields": active.get("checked_fields") or [],
            "active_python_version": active_python,
            "recorded_python_version": None,
        }
    differences.update(active.get("differences") or {})
    offenders = dict(active.get("third_party_editable_installs") or {})
    if offenders:
        differences["third_party_editable_installs"] = sorted(offenders)
    deviates = bool(differences)
    return {
        "certified": not deviates,
        "deviates": deviates,
        "package_set_comparable": True,
        "reason": ("the active environment matches the recorded dependency "
                   "lock on every identity field" if not deviates else
                   "the active environment deviates from the recorded "
                   "dependency lock"),
        "differences": differences,
        "informational_differences":
            active.get("informational_differences") or {},
        "checked_fields": active.get("checked_fields") or [],
        "active_python_version": active_python,
        "recorded_python_version": (active.get("locked_identity") or {}).get(
            "python_version"),
        "active_executable": active.get("active_executable"),
        "recorded_executable": active.get("locked_executable"),
        "dependency_lock_sha256": active.get("dependency_lock_sha256"),
    }


#: What a verification concluded. Kept as constants because three scripts print
#: them and a reader comparing two gates should see the same words.
EXACT = "reproduced_exactly"
WITHIN_TOLERANCE = "reproduced_within_the_documented_numeric_tolerance"
DIFFERS = "differs"


def verdict(comparison: dict, deviation: dict) -> tuple[int, str, list[str]]:
    """``(exit_code, verdict, issues)`` for one re-derivation.

    0 -- reproduced exactly, in the certified environment.
    3 -- reproduced within the documented numeric tolerance, under a deviation
        from the lock that is named in the report. Not 0, because "the numbers
        agree to 1e-12 under a different interpreter" is a weaker statement
        than "the numbers are the same numbers".
    1 -- differs: a non-numeric leaf moved, a numeric leaf moved outside its
        tolerance, or anything at all moved inside the certified environment
        where the last bit is reproducible and a difference is a defect.
    """
    issues: list[str] = []
    if comparison["n_nonnumeric_differences"]:
        for difference in comparison["nonnumeric_differences"][:6]:
            issues.append(
                f"{difference['path']}: a NON-NUMERIC field differs -- filed "
                f"{difference['filed']!r}, re-derived "
                f"{difference['fresh']!r}. No tolerance applies to a verdict, "
                f"a sign, a name or a count")
    if comparison["numeric_differences_outside_tolerance"]:
        for difference in comparison[
                "numeric_differences_outside_tolerance"][:6]:
            issues.append(
                f"{difference['path']}: a {difference['kind']} differs by "
                f"{difference['absolute_difference']:.6g} (filed "
                f"{difference['filed']!r}, re-derived "
                f"{difference['fresh']!r}), outside the documented tolerance")
    if not issues and comparison["n_numeric_differences"] \
            and not deviation["deviates"]:
        issues.append(
            f"{comparison['n_numeric_differences']} numeric field(s) differ "
            f"but the active environment matches the recorded dependency lock "
            f"on every identity field, so the last bit IS reproducible here "
            f"and the difference is a defect rather than summation noise; the "
            f"numeric tolerance is licensed only by a demonstrated deviation")
    if issues:
        return 1, DIFFERS, issues
    if comparison["n_numeric_differences"]:
        return 3, WITHIN_TOLERANCE, []
    if not deviation["certified"]:
        # Exact agreement reached without a lock to compare against is still
        # worth saying out loud: the numbers reproduced, the environment could
        # not be certified.
        return 0, EXACT, [deviation["reason"]]
    return 0, EXACT, []
