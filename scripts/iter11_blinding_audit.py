#!/usr/bin/env python3
"""Iteration 11.7 — the model-identity blinding audit, and its verify gate.

WHY THIS IS A COMMITTED SCRIPT
``outputs/iteration_11/diagnostics/judge_blinding/identity_leak_audit.json``
is evidence for a frozen protocol clause, and nothing in the repository could
regenerate it: it was produced by an ad-hoc file outside the tree, so the
claim "target-model identity is blinded in judge prompts" rested on a script
no reviewer could run. Every other Iteration 11 evidence artifact has a
committed producer and a read-only verify mode
(:mod:`scripts.iter11_write_selection`, :mod:`scripts.iter11_replay_checks`,
:mod:`scripts.iter11_judge_vision_ablation`). This brings the blinding audit
into line.

THE CLAUSE UNDER TEST
    "output_blinding": "target-model identity is blinded in judge prompts;
                        variant/family blinded as in Iteration 10"

Iteration 10 had ONE target, so the first half was vacuous — there was nothing
to distinguish. Iteration 11 has four, which makes it testable, and there are
four channels a de-blinding could travel down. Three are leaks; the fourth is
structural:

1. PAYLOAD — the blinded item, or the prompt actually rendered from it,
   carries a ``model_key``, ``model_id``, adapter or marketed size label.
   Scanned for EVERY arm's vocabulary in EVERY arm's payload, not just its
   own: a judge that could separate the arms by a stray mention of a
   DIFFERENT model would be just as unblinded.
2. CONTEXT — the system prompt or template differs between targets, so a
   judge could separate the arms with no name appearing anywhere.
3. RESPONSE — the model names itself inside its own answer ("As Qwen, I
   cannot…"). No payload hygiene blinds this, and one target was enough that
   Iteration 10 could never have exposed it.
4. ITEM-ID ALIGNMENT — whether the four sessions' blinded ids denote the same
   cell. Not a leak, but the property that makes pooling the four sessions
   into one silently catastrophic: every id would collide four ways and merge
   distinct models' responses under one label.

WHAT THIS ADDS OVER THE FIRST RUN
* Targets and run directories come from the committed
  ``configs/evaluation/scale_profiles.json`` ``iteration_11_*`` profiles
  rather than being hardcoded, so a profile edit moves the audit with it. The
  first run predates those profiles and had to hardcode them.
* The re-derivation of ``prepare_blinded_items`` is no longer an unchecked
  copy. Where the judge pipeline has already written its own
  ``blinded_items.json``, this compares the two cell by cell and records the
  result. A mirror that has never been checked against what it mirrors is an
  assumption, not evidence — and channels 1-4 are all computed from the
  mirror.
* Channel 4 refuses to draw a pooling verdict from journals of different
  lengths. ``item_id`` is an index into a per-target seed-42 shuffle, so
  partial journals shuffle differently and the comparison measures nothing;
  reporting "not aligned" for a design that is perfectly aligned at
  completion is the wrong answer in the unsafe direction.

``tests/unit/test_iter11_judge_profiles.py`` pins channels 1-3 over a sample
as a regression guard. This is the full-population audit.

Usage:
    python3 scripts/iter11_blinding_audit.py             # derive and write
    python3 scripts/iter11_blinding_audit.py --verify    # read-only gate
    python3 scripts/iter11_blinding_audit.py --json      # print, write nothing
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal_mllm.data.io import read_jsonl  # noqa: E402
from causal_mllm.data.schemas import CausalFamily  # noqa: E402
from causal_mllm.evaluation.human_template import (  # noqa: E402
    _build_anonymization_map,
    _extract_conversation_context,
)
from causal_mllm.evaluation.llm_judge import (  # noqa: E402
    LLMJudgeConfig,
    MultimodalLLMJudge,
)
from causal_mllm.replay.config import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from causal_mllm.seeds import sha256_bytes, sha256_text  # noqa: E402

PROFILES_FILE = REPO_ROOT / "configs" / "evaluation" / "scale_profiles.json"
PROTOCOL_FILE = REPO_ROOT / "outputs" / "iteration_11" / "protocol" \
    / "iteration_11_protocol.json"
OUT_FILE = REPO_ROOT / "outputs" / "iteration_11" / "diagnostics" \
    / "judge_blinding" / "identity_leak_audit.json"

#: The frozen panel every arm replays. Read from the legacy ``scale_c``
#: profile rather than restated, so the audit follows the configuration it is
#: auditing instead of a copy of it that could drift.
PANEL_KEY = "scale_c"

#: The confirmatory panel length. A journal shorter than this is incomplete,
#: which is what makes channel 4 inconclusive rather than merely negative.
FULL_PANEL_CELLS = 600

#: Identity vocabulary per arm, deliberately over-inclusive: a false positive
#: costs a manual look, a false negative silently de-blinds the confirmatory
#: judging. Case-insensitive, word-boundary anchored where a term is short
#: enough to occur inside unrelated words.
IDENTITY_TERMS = {
    "qwen35_2b": [r"\bqwen35_2b\b", r"\bQwen/Qwen3\.5-2B\b", r"\bqwen3\.5\b",
                  r"\bqwen\b", r"\b2B\b", r"\bqwen35\b"],
    "qwen35_4b": [r"\bqwen35_4b\b", r"\bQwen/Qwen3\.5-4B\b", r"\bqwen3\.5\b",
                  r"\bqwen\b", r"\b4B\b", r"\bqwen35\b"],
    "ministral3_3b": [r"\bministral3_3b\b", r"\bministral\b", r"\bmistral\b",
                      r"\b3B\b", r"\bministral3\b"],
    "phi4_mm": [r"\bphi4_mm\b", r"\bphi-4\b", r"\bphi4\b", r"\bmicrosoft\b",
                r"\bphi\b"],
    # The frozen Iteration 10 upper-Qwen reference is an arm of the 11.8
    # cross-model analysis too, so its name must not appear in a sibling's
    # payload any more than a sibling's may. Its own journal is sealed
    # Iteration 10 evidence and is not re-audited here.
    "qwen35_9b": [r"\bqwen35_9b\b", r"\bQwen/Qwen3\.5-9B\b", r"\bqwen3\.5\b",
                  r"\bqwen\b", r"\b9B\b", r"\bqwen35\b"],
}

#: The judges' OWN model ids. These must not appear either — a prompt that
#: names the judge is a different defect, but still one. Scanning for them is
#: also what makes the placeholder identity passed to ``MultimodalLLMJudge``
#: below safe: if a model id could reach the prompt, this finds it.
JUDGE_TERMS = [r"\bqwen3\.8-max\b", r"\bglm-5\.2\b", r"\bkimi-k3\b"]

#: Keys of a blinded item the judge is shown. ``family_id`` and ``variant``
#: are kept by the pipeline for provenance and must never be rendered.
SHOWN_KEYS = ("item_id", "system_prompt", "conversation_history",
              "terminal_query", "response", "response_sha256")

#: How many items to render a full prompt for. Rendering reads and base64s
#: every referenced image, so this is the expensive channel; the payload scan
#: that covers all 600 items is cheap and stays exhaustive.
PROMPT_SAMPLE = 20


def _profiles() -> dict:
    return json.loads(PROFILES_FILE.read_text(encoding="utf-8"))


def _iteration_11_profiles() -> dict:
    """``{model_key: profile}`` for the four confirmatory arms."""
    profiles = _profiles()
    prefix = "iteration_11_"
    return {k[len(prefix):]: v for k, v in profiles.items()
            if k.startswith(prefix)}


def _load_families(panel: str | Path) -> dict:
    families = {}
    for rec in read_jsonl(Path(panel)):
        fam = CausalFamily.from_dict(rec)
        families[fam.family_id] = fam
    return families


def _protocol_identity(model_key: str) -> dict:
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    for entry in protocol["model_matrix"]:
        if entry["model_key"] == model_key:
            return entry
    raise KeyError(f"{model_key} is not in the frozen protocol model_matrix")


def derive_blinded_items(records: list[dict], families: dict,
                         seed: int = 42) -> list[dict]:
    """Mirror :func:`run_llm_judge_pipeline.prepare_blinded_items` exactly.

    Duplicated rather than imported because that module raises at import time
    without ``LLM_JUDGE_API_KEY`` and binds its journal to ONE scale profile
    at module scope, so it cannot be pointed at four targets in one process.
    The duplication is not left unverified: :func:`crosscheck_mirror` compares
    it against the pipeline's own committed ``blinded_items.json``.
    """
    _build_anonymization_map(seed)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    items = []
    for idx, rec in enumerate(shuffled):
        variant = rec.get("variant")
        family_id = rec.get("family_id")
        family = families.get(family_id)
        if family is None:
            raise ValueError(f"family {family_id} not found")
        _, history_msgs, terminal_q = _extract_conversation_context(
            family, variant)
        items.append({
            "item_id": f"item-{idx:04d}",
            "family_id": family_id,
            "variant": variant,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "conversation_history": history_msgs,
            "terminal_query": terminal_q,
            "response": rec.get("response", ""),
            "response_sha256": sha256_text(rec.get("response", "")),
        })
    return items


def crosscheck_mirror(items: list[dict], pipeline_items_path: Path) -> dict:
    """Does the mirror agree with what the judge pipeline actually wrote?

    Channels 1-4 are all computed from :func:`derive_blinded_items`. If that
    copy ever drifts from ``prepare_blinded_items`` the audit silently stops
    describing the real payloads, so wherever the pipeline has already written
    its own ``blinded_items.json`` the two are compared field by field.
    """
    if not pipeline_items_path.exists():
        return {"present": False,
                "reason": "the judge pipeline has not written blinded_items."
                          "json for this target yet"}
    written = json.loads(pipeline_items_path.read_text(encoding="utf-8"))
    mismatches = []
    for mine, theirs in zip(items, written):
        for key in ("item_id", "family_id", "variant", "response_sha256"):
            if mine[key] != theirs.get(key):
                mismatches.append({"item_id": mine["item_id"], "field": key})
    same_context = all(
        mine["conversation_history"] == theirs.get("conversation_history")
        and mine["terminal_query"] == theirs.get("terminal_query")
        for mine, theirs in zip(items, written))
    return {
        "present": True,
        "path": str(pipeline_items_path.relative_to(REPO_ROOT)),
        "n_derived": len(items),
        "n_written": len(written),
        "n_field_mismatches": len(mismatches),
        "context_identical": same_context,
        "identical": (len(items) == len(written) and not mismatches
                      and same_context),
        "mismatches": mismatches[:5],
    }


def _scan(text: str, terms: list[str]) -> list[dict]:
    """``[{term, count, context}]`` for every term found in ``text``."""
    hits = []
    for term in terms:
        matches = list(re.finditer(term, text, re.IGNORECASE))
        if not matches:
            continue
        m = matches[0]
        lo, hi = max(0, m.start() - 60), min(len(text), m.end() + 60)
        hits.append({"term": term, "count": len(matches),
                     "context": text[lo:hi].replace("\n", " ")})
    return hits


def _collapse(hits: list[dict]) -> list[dict]:
    """Aggregate per-term, keeping one example, so the artifact stays small."""
    agg: dict = {}
    for h in hits:
        entry = agg.setdefault(h["term"], {"count": 0, "example": h["context"]})
        entry["count"] += h["count"]
    return [{"term": t, **v} for t, v in sorted(agg.items())]


def _judge() -> MultimodalLLMJudge:
    """Judge A's frozen rubric, with no reachable endpoint.

    Only ``_build_prompt`` is ever called, which is pure text assembly plus
    image encoding — it makes no request. The ``model_id`` cannot reach the
    prompt; :data:`JUDGE_TERMS` is scanned for precisely so that this stays
    true rather than being assumed.
    """
    return MultimodalLLMJudge(
        LLMJudgeConfig(model_id="qwen3.8-max", provider="aliyun",
                       base_url="https://invalid.example/v1", api_key="unused",
                       temperature=0.0, seed=42),
        judge_id="A")


def audit_target(key: str, profile: dict, families: dict,
                 judge: MultimodalLLMJudge, arms: list[str]) -> tuple[dict, dict]:
    """Audit one arm. Returns ``(report_entry, item_id -> cell map)``."""
    run_dir = REPO_ROOT / profile["replay_run"]
    journal = run_dir / "replay_outputs.jsonl"
    entry: dict = {
        "run_dir": profile["replay_run"],
        "complete": (run_dir / "replay_report.json").exists(),
    }
    if not journal.exists():
        entry["skipped"] = "no journal yet"
        return entry, {}

    records = list(read_jsonl(journal))
    entry["n_records"] = len(records)
    items = derive_blinded_items(records, families)
    id_map = {it["item_id"]: (it["family_id"], it["variant"]) for it in items}

    entry["mirror_crosscheck"] = crosscheck_mirror(
        items, REPO_ROOT / profile["output_dir"] / "blinded_items.json")

    # --- channel 1: payload -------------------------------------------------
    payload_own, payload_other, prompt_own, prompt_judge = [], [], [], []
    for it in items:
        shown = json.dumps({k: it[k] for k in SHOWN_KEYS}, ensure_ascii=False)
        payload_own += _scan(shown, IDENTITY_TERMS[key])
        for other in arms:
            if other != key:
                payload_other += [dict(h, arm=other)
                                  for h in _scan(shown, IDENTITY_TERMS[other])]

    rendered, render_errors = 0, []
    for it in items[:PROMPT_SAMPLE]:
        try:
            prompt, _imgs, _hashes = judge._build_prompt(
                it["system_prompt"], it["conversation_history"],
                it["terminal_query"], it["response"])
        except Exception as exc:            # noqa: BLE001 - recorded, not fatal
            render_errors.append({"item_id": it["item_id"],
                                  "error": f"{type(exc).__name__}: {exc}"})
            continue
        rendered += 1
        prompt_own += _scan(prompt, IDENTITY_TERMS[key])
        prompt_judge += _scan(prompt, JUDGE_TERMS)

    entry["payload_identity_hits"] = _collapse(payload_own)
    entry["payload_other_arm_identity_hits"] = _collapse(
        [{**h, "term": f"{h['arm']}:{h['term']}"} for h in payload_other])
    entry["prompt_identity_hits"] = _collapse(prompt_own)
    entry["prompt_judge_identity_hits"] = _collapse(prompt_judge)
    entry["prompts_rendered"] = rendered
    entry["prompt_render_errors"] = render_errors

    # The Iteration 10 half of the clause: variant/family blinded.
    violations = []
    for it in items[:PROMPT_SAMPLE]:
        try:
            prompt, _, _ = judge._build_prompt(
                it["system_prompt"], it["conversation_history"],
                it["terminal_query"], it["response"])
        except Exception:                   # noqa: BLE001 - already recorded
            continue
        for forbidden in (it["family_id"], it["variant"]):
            if forbidden and forbidden in prompt:
                violations.append({"item_id": it["item_id"],
                                   "leaked": forbidden})
    entry["iteration10_blinding_violations"] = violations

    # --- channel 2: context -------------------------------------------------
    # If the shared context differed per target a judge could separate the
    # arms with no name appearing anywhere, so the hash is recorded per arm
    # and the caller asserts every arm produced the same one. The journal's
    # own recorded hash is kept alongside: the blinded item's system prompt is
    # what the judge is SHOWN, the journal's is what the replay USED, and the
    # two agreeing is what makes channel 2 a statement about the run rather
    # than about this script's reconstruction of it.
    entry["system_prompt_sha256"] = sorted(
        {sha256_text(it["system_prompt"]) for it in items})
    shas = {r.get("system_prompt_sha256") for r in records}
    entry["journal_system_prompt_sha256"] = sorted(s for s in shas if s)
    entry["prompt_template_revision"] = sorted(
        {r.get("prompt_template_revision") for r in records
         if r.get("prompt_template_revision")})

    # --- channel 3: response self-identification ----------------------------
    selfid = []
    for it in items:
        hits = _scan(it["response"], IDENTITY_TERMS[key])
        if hits:
            selfid.append({"item_id": it["item_id"], "variant": it["variant"],
                           "hits": [{"term": h["term"], "context": h["context"]}
                                    for h in hits]})
    by_variant: dict = {}
    for s in selfid:
        by_variant[s["variant"]] = by_variant.get(s["variant"], 0) + 1
    entry["response_self_identification"] = {
        "n_items": len(selfid),
        "rate": round(len(selfid) / len(items), 4) if items else None,
        "by_variant": by_variant,
        "examples": selfid[:8],
    }

    # --- declared identity, so a reader can see what was searched for -------
    declared = _protocol_identity(key)
    entry["declared_identity"] = {
        "model_id": declared["model_id"],
        "adapter": declared["adapter"],
        "marketed_label": (declared.get("size_metadata")
                           or {}).get("marketed_label"),
    }
    entry["identity_terms"] = IDENTITY_TERMS[key]
    return entry, id_map


def audit() -> dict:
    """Derive the whole report. Deterministic: no clock, no ordering hazard."""
    profiles = _iteration_11_profiles()
    all_profiles = _profiles()
    arms = sorted(IDENTITY_TERMS)
    panel = REPO_ROOT / all_profiles[PANEL_KEY]["validated_families"]
    families = _load_families(panel)
    judge = _judge()
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))

    # The panel the audit reads must be the panel the protocol froze. Checked
    # rather than assumed: a regenerated panel would silently re-audit a
    # different set of cells than the ones the judges were shown.
    panel_sha = sha256_bytes(panel.read_bytes())
    frozen_panel_sha = protocol["frozen_inputs"][
        "panel_validated_families_sha256"]
    if panel_sha != frozen_panel_sha:
        raise SystemExit(
            f"FATAL: {panel} hashes to {panel_sha} but the frozen protocol "
            f"records {frozen_panel_sha}; this audit would describe a panel "
            "that is not the one the judges were shown")

    report: dict = {
        "protocol_clause": protocol["frozen_inputs"]["judging"][
            "output_blinding"],
        "rubric_sha256": judge.rubric_sha256,
        "rubric_version": judge.rubric_version,
        "default_system_prompt_sha256": sha256_text(DEFAULT_SYSTEM_PROMPT),
        "deterministic": True,
        "derived_from": {
            "panel": all_profiles[PANEL_KEY]["validated_families"],
            "panel_validated_families_sha256": panel_sha,
            "n_panel_families": len(families),
            "protocol": str(PROTOCOL_FILE.relative_to(REPO_ROOT)),
            "protocol_sha256": sha256_text(
                PROTOCOL_FILE.read_text(encoding="utf-8")),
            "scale_profiles": str(PROFILES_FILE.relative_to(REPO_ROOT)),
            "producer": "scripts/iter11_blinding_audit.py",
        },
        "prompt_sample_per_target": PROMPT_SAMPLE,
        "targets": {},
        "cross_target": {},
    }

    id_maps = {}
    for key in sorted(profiles):
        entry, id_map = audit_target(key, profiles[key], families, judge, arms)
        report["targets"][key] = entry
        if id_map:
            id_maps[key] = id_map

    # Channel 2 verdict: one shared context across every audited arm.
    contexts = {tuple(e["system_prompt_sha256"])
                for e in report["targets"].values()
                if "system_prompt_sha256" in e}
    report["cross_target"]["shared_context"] = {
        "n_distinct_system_prompts": len(contexts),
        "uniform": len(contexts) <= 1,
        "sha256": sorted(next(iter(contexts))) if len(contexts) == 1 else [],
    }

    # --- channel 4: cross-target item_id alignment --------------------------
    lengths = {k: e.get("n_records") for k, e in report["targets"].items()}
    incomplete = sorted(k for k, n in lengths.items()
                        if n != FULL_PANEL_CELLS)
    n_complete = len(lengths) - len(incomplete)
    if incomplete or len(id_maps) < len(profiles):
        report["cross_target"]["item_id_alignment"] = {
            "status": "inconclusive",
            "reason": (
                f"{n_complete}/{len(profiles)} journals "
                f"hold all {FULL_PANEL_CELLS} cells; item_id is an index into "
                "a per-target seed-42 shuffle, so journals of different "
                "lengths shuffle differently and the comparison measures "
                "nothing. Reporting 'not aligned' here would condemn a design "
                "that is perfectly aligned at completion."),
            "journal_lengths": lengths,
            "incomplete": incomplete,
            "must_rerun_after": "every target has a replay_report.json",
        }
    else:
        keys = sorted(id_maps)
        base = keys[0]
        aligned = {}
        for other in keys[1:]:
            shared = set(id_maps[base]) & set(id_maps[other])
            same = [i for i in shared
                    if id_maps[base][i] == id_maps[other][i]]
            aligned[f"{base}~{other}"] = {
                "n_shared_ids": len(shared),
                "n_identical_cell": len(same),
                "fully_aligned": len(same) == len(shared),
            }
        report["cross_target"]["item_id_alignment"] = {
            "status": "measured",
            "journal_lengths": lengths,
            "pairs": aligned,
            "pooling_verdict": (
                "item IDs are aligned across targets, so the sessions MUST "
                "stay separate: pooling them would collide each item_id "
                f"{len(keys)} ways and silently merge distinct models' "
                "responses under one label."
                if any(v["fully_aligned"] for v in aligned.values())
                else "item IDs are not aligned; pooling would not collide."),
        }

    return report


def _verdict(report: dict) -> tuple[bool, list[str]]:
    """Is the frozen clause satisfied? Returns ``(ok, reasons)``."""
    problems = []
    for key, e in report["targets"].items():
        if "skipped" in e:
            problems.append(f"{key}: skipped ({e['skipped']})")
            continue
        for field in ("payload_identity_hits",
                      "payload_other_arm_identity_hits",
                      "prompt_identity_hits", "prompt_judge_identity_hits",
                      "iteration10_blinding_violations",
                      "prompt_render_errors"):
            if e.get(field):
                problems.append(f"{key}: {len(e[field])} {field}")
        mirror = e.get("mirror_crosscheck", {})
        if mirror.get("present") and not mirror.get("identical"):
            problems.append(f"{key}: mirror disagrees with the pipeline's own "
                            f"blinded_items.json ({mirror})")
        selfid = e["response_self_identification"]
        if selfid["n_items"]:
            # Channel 3 is a property of the MODEL, not of the blinding code:
            # no payload hygiene can stop a model naming itself. Recorded as a
            # finding to report, not as a blinding failure.
            problems.append(f"{key}: {selfid['n_items']} responses "
                            f"self-identify (channel 3, model behaviour)")
    shared = report["cross_target"].get("shared_context", {})
    if not shared.get("uniform", False) and shared:
        problems.append(f"channel 2: {shared['n_distinct_system_prompts']} "
                        "distinct system prompts across arms")
    alignment = report["cross_target"].get("item_id_alignment", {})
    if alignment.get("status") == "inconclusive":
        problems.append(f"channel 4: inconclusive ({alignment.get('reason')})")
    # Only the leak channels and channel 2 gate the verdict; channel 3 and an
    # inconclusive channel 4 are reported but are not blinding failures.
    blocking = [p for p in problems
                if "channel 3" not in p and "channel 4" not in p
                and "skipped" not in p]
    return not blocking, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-derive and compare to the committed artifact; "
                         "write nothing")
    ap.add_argument("--json", action="store_true",
                    help="print the derived report to stdout; write nothing")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    report = audit()
    ok, problems = _verdict(report)

    for key in sorted(report["targets"]):
        e = report["targets"][key]
        if "skipped" in e:
            print(f"{key:15s} SKIPPED ({e['skipped']})")
            continue
        sid = e["response_self_identification"]
        mirror = e.get("mirror_crosscheck", {})
        print(f"{key:15s} n={e['n_records']:<4d} complete={e['complete']}")
        print(f"{'':15s}   payload own-arm hits  : "
              f"{len(e['payload_identity_hits'])}")
        print(f"{'':15s}   payload other-arm hits: "
              f"{len(e['payload_other_arm_identity_hits'])}")
        print(f"{'':15s}   prompt  identity hits : "
              f"{len(e['prompt_identity_hits'])} "
              f"({e['prompts_rendered']} prompts rendered)")
        print(f"{'':15s}   prompt  judge-id hits : "
              f"{len(e['prompt_judge_identity_hits'])}")
        print(f"{'':15s}   iter10 blinding viols : "
              f"{len(e['iteration10_blinding_violations'])}")
        print(f"{'':15s}   mirror vs pipeline    : "
              f"{'identical' if mirror.get('identical') else mirror}")
        print(f"{'':15s}   response self-id      : {sid['n_items']}/"
              f"{e['n_records']} ({sid['rate']})")
    print()
    print("channel 2:", json.dumps(
        report["cross_target"].get("shared_context", {}), indent=2))
    print("channel 4:", json.dumps(
        report["cross_target"].get("item_id_alignment", {}), indent=2)[:1200])
    print()
    for p in problems:
        print(f"  FINDING {p}")
    print(f"\nBLINDING {'PASS' if ok else 'FAIL'}")

    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.json:
        print(text)
        return 0 if ok else 1

    out = Path(args.out)
    if args.verify:
        if not out.exists():
            print(f"VERIFY FAIL: no committed artifact at {out}")
            return 1
        committed = out.read_text(encoding="utf-8")
        if json.loads(committed) != report:
            print(f"VERIFY FAIL: {out} does not match a fresh derivation")
            print("  (journals have grown, or the derivation changed; "
                  "re-run without --verify and commit)")
            return 1
        print(f"VERIFY PASS: {out} matches a fresh derivation")
        return 0 if ok else 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
