"""What the moderation exclusion is blind to, and what it is not.

Two claims used to be filed together under one key, ``outcome_independent``:

    "which cells a provider refuses is a function of the request bytes, so the
     surviving family set was fixed before any label existed"

The second half is true. The first half is not, and filing them as one sentence
made the true half carry the false one: "a function of the request bytes" reads
as "not a function of the model's behaviour", and the request a provider
moderates CARRIES the evaluated response, so it is both.

The measurement that settles it is in
``outputs/iteration_11/diagnostics/judge_moderation/cell_probe_exclusion_union_reprobe.json``:
of the three cells the four arms lost, ``CMST_456921/text_only`` was refused in
the ``ministral3_3b`` arm ALONE and served in the other three, while both
``CMST_795308`` cells were refused in all four. A refusal that tracks the arm
tracks what that arm's model said. That is differential censoring, and it is
correlated with the outcome by construction.

So the exclusion is LABEL-BLIND and RESPONSE-DEPENDENT, and this module exists
to keep those two apart everywhere they are filed. One wording, imported by
both producers, because two copies of a correction drift apart the same way two
copies of the original claim did.

The 21 committed Iteration 11 artifacts that still carry the superseded key are
NOT rewritten: they are sealed judge and evaluation evidence, and regenerating
them means re-running a pipeline whose adjudication pass re-calls the
adjudicator. They are enumerated, quoted and corrected beside themselves by
``scripts/iter11_correct_exclusion_metadata.py``, and a test scans ``outputs/``
live so a NEW artifact carrying the old key fails.
"""

from __future__ import annotations

#: The key this module replaces. Kept as a constant so the correction artifact,
#: the producers' tests and the live scan of ``outputs/`` all name the same
#: string rather than three transcriptions of it.
SUPERSEDED_KEY = "outcome_independent"

#: The cell that makes the exclusion response-dependent, and the arm whose
#: model's reply triggered the refusal.
DIFFERENTIAL_CELL = "CMST_456921/text_only"
DIFFERENTIAL_TARGET = "ministral3_3b"

#: The cells refused in EVERY arm, which are consistent with a request-bytes
#: cause and are still not independent of the response they travel with.
UNIFORM_CELLS = ("CMST_795308/cross_modal", "CMST_795308/shuffle")

#: Where the differential/uniform split was measured.
EVIDENCE_ARTIFACT = ("outputs/iteration_11/diagnostics/judge_moderation/"
                     "cell_probe_exclusion_union_reprobe.json")

#: Where the superseded key is enumerated and corrected.
CORRECTION_ARTIFACT = ("outputs/iteration_11/diagnostics/"
                       "exclusion_metadata_correction.json")


def label_blind(scope: str) -> str:
    """The half of the old claim that survives."""
    return (
        f"{scope} is settled before any judge has scored anything, so no "
        f"label -- no score, no compliance level, no refusal type -- "
        f"influenced which cells survived, and no arm's family set was chosen "
        f"after seeing how it came out. That half of the claim formerly filed "
        f"as '{SUPERSEDED_KEY}' is true and is kept")


def response_dependent(scope: str) -> str:
    """The half the old claim got wrong, with the measurement behind it."""
    return (
        f"{scope} is NOT independent of the outcome, and the key that said so "
        f"was wrong. The request a provider moderates carries the evaluated "
        f"response, so which cells are excluded is a function of what the model "
        f"said as well as of the request's other bytes. Measured, not argued: "
        f"{DIFFERENTIAL_CELL} was refused in the {DIFFERENTIAL_TARGET} arm "
        f"alone and served in the other three, while {UNIFORM_CELLS[0]} and "
        f"{UNIFORM_CELLS[1]} were refused in all four -- a refusal that tracks "
        f"the arm tracks that arm's reply, which is differential censoring and "
        f"is correlated with the outcome by construction "
        f"({EVIDENCE_ARTIFACT}). The uniformly refused pair is consistent with "
        f"a request-bytes cause, at roughly 1.55 MB of image payload that the "
        f"provider's input moderation rejects, but consistency is not "
        f"independence and those cells travel in the same request as the "
        f"response. This exclusion is therefore LABEL-BLIND and "
        f"RESPONSE-DEPENDENT; the superseded '{SUPERSEDED_KEY}' claimed the "
        f"stronger and false thing, and the artifacts still carrying it are "
        f"enumerated and corrected in {CORRECTION_ARTIFACT} rather than "
        f"rewritten, because they are sealed judge and evaluation evidence")


def exclusion_metadata(scope: str) -> dict:
    """The two claims, filed as two fields so neither carries the other.

    ``scope`` names the exclusion set being described -- the union a provider
    refused, or the cells no judge could label -- so each artifact states the
    rule it actually applied rather than a generic one.
    """
    return {
        "label_blind": label_blind(scope),
        "response_dependent": response_dependent(scope),
    }
