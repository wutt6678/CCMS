#!/usr/bin/env python3
"""Iteration 11.8: one panel for all four arms, derived before and gated after.

11.8 compares four models WITH EACH OTHER over the frozen 100-family panel. A
comparison is only a comparison if every arm is measured on the same cells, and
the arms do not guarantee that by themselves: ``build_judge_coverage`` unions
the refusals of judges A and B for ONE target session, so a cell that aliyun's
input moderation refused while judging qwen35_2b but served while judging
phi4_mm twenty minutes later is dropped from the first arm and kept in the
second. Both arms then report a Delta_TV, the two numbers are put side by side,
and nothing in either artifact says they describe different families.

The moderation verdict is a property of the request bytes and of the provider's
policy at the moment of the call. It is not stable over time, and the four arms
reach a given cell minutes apart, so agreement between them is a coincidence
rather than a property of the design. This script makes it a property.

TWO MODES, because the union has to exist before it can be applied:

  (no flag)  DERIVE. Read the eight completed primary outputs -- four targets
             times judges A and B -- and write the union of the cells any of
             them lost to ``<group>/common_panel.json``. Phase 2 reads that
             artifact and drops the union from every arm, so all four analyses
             land on one panel by construction. Refuses to derive while any arm
             is incomplete: a union over three arms and a half is not the union.

  --verify   GATE. Load the four ACTUAL ``judge_coverage.json`` artifacts plus
             the cells each analysis really evaluated, and require identical
             excluded-cell sets and identical surviving-family sets. Exits
             non-zero when they differ and names the global union the analyses
             have to be regenerated on.

Membership is derived the same way ``collect_provider_refusals`` derives it: a
cell is unjudged iff the arm's completed output -- the one its fingerprint
sidecar binds -- carries no judgment for it. A refusal sidecar is evidence
about WHY, never about WHICH, and is only believed under a matching
fingerprint. This script is read-only over the judge artifacts; the only file
it writes is the common panel itself.

Usage:
    python scripts/iter11_common_panel.py             # derive, before phase 2
    python scripts/iter11_common_panel.py --verify    # gate, after phase 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Which profiles are the arms of the Iteration 11 cross-model comparison, and
#: where they say the group lives. Read from tracked config rather than from a
#: directory listing: a listing would silently shrink the group if one target
#: had not been judged yet, and a union over three arms is not the union.
SCALE_PROFILES = REPO_ROOT / "configs" / "evaluation" / "scale_profiles.json"
PROFILE_PREFIX = "iteration_11_"
GROUP_KEY = "cross_arm_group"

#: Written by DERIVE, read by phase 2 and by --verify.
COMMON_PANEL_ARTIFACT = "common_panel.json"

#: Written by phase 2, per target: what that arm's ensemble judged.
COVERAGE_ARTIFACT = "judge_coverage.json"

#: Written by the evaluation stage inside ``evaluation_results``. The jsonl is
#: the per-cell record of the panel the analysis ACTUALLY used, which is the
#: surviving-family set asked for; the report carries the restriction it
#: applied and why.
EVALUATION_DIR = "evaluation_results"
EVALUATION_REPORT = "evaluation_report.json"
EVALUATION_CELLS = "evaluation_outputs.jsonl"

PRIMARIES = ("A", "B")


def _rel(path: Path) -> str:
    """Repo-relative when it is in the repo, absolute when it is a fixture."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _as_list(obj, key: str) -> list:
    """Judge artifacts are a bare list; checkpoints wrap one under a key."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return obj.get(key, [])
    return []


def group_arms(profiles_path: Path = SCALE_PROFILES) -> tuple[Path, dict]:
    """``(group_dir, {target: judge_dir})`` for the Iteration 11 arms.

    Every arm has to name the same group. Two arms declaring different groups
    would each derive a union over a different set of siblings and both would
    look complete.
    """
    profiles = _load_json(profiles_path)
    groups: dict[str, Path] = {}
    arms: dict[str, Path] = {}
    for name, profile in sorted(profiles.items()):
        if not name.startswith(PROFILE_PREFIX):
            continue
        group = profile.get(GROUP_KEY)
        if not group:
            raise SystemExit(
                f"{name} is an Iteration 11 arm but declares no "
                f"{GROUP_KEY!r}, so which other arms it is compared with is "
                f"unstated and no common panel can be derived")
        groups.setdefault(group, Path(group))
        if len(groups) > 1:
            raise SystemExit(
                f"Iteration 11 arms declare {len(groups)} different "
                f"{GROUP_KEY} values ({sorted(groups)}); one cross-model "
                f"comparison has one group")
        arms[Path(profile["output_dir"]).name] = REPO_ROOT / profile[
            "output_dir"]
    if len(arms) < 2:
        raise SystemExit(
            f"found {len(arms)} Iteration 11 arm(s) in {_rel(profiles_path)}; "
            f"a cross-arm common panel needs at least two")
    return next(iter(groups.values())), dict(sorted(arms.items()))


def _load_blinded(judge_dir: Path) -> list[dict] | None:
    path = judge_dir / "blinded_items.json"
    if not path.exists():
        return None
    return _as_list(_load_json(path), "items")


def arm_state(judge_dir: Path, primary: str,
              items_by_id: dict[str, dict]) -> dict:
    """What one primary of one target judged, and which cells it left unjudged.

    Read-only. Membership comes from absence in the completed output, never
    from the refusal sidecar -- the sidecar is written in sequence after the
    output, so a stop between the two leaves a completed arm beside a refusal
    file from an earlier configuration, and ``response_sha256`` is a property
    of the frozen panel rather than of the run, so nothing about a stale
    sidecar disagrees with the current one.
    """
    output = judge_dir / f"llm_labels_judge_{primary}.json"
    fingerprint_sidecar = output.with_name(output.name + ".fingerprint")
    refusals = output.with_suffix(".refusals.json")
    state: dict = {
        "primary": primary,
        "model_id": None,
        "complete": False,
        "output_file": _rel(output),
    }
    if not output.exists() or not fingerprint_sidecar.exists():
        checkpoint = judge_dir / f"llm_labels_judge_{primary}.checkpoint.json"
        held = len(_as_list(_load_json(checkpoint), "judgments")) \
            if checkpoint.exists() else 0
        state["reason"] = (
            "no completed output bound by a fingerprint sidecar; the "
            f"checkpoint holds {held} judgment(s)")
        state["n_checkpointed"] = held
        return state

    fingerprint = fingerprint_sidecar.read_text(encoding="utf-8").strip()
    judgments = _as_list(_load_json(output), "judgments")
    judged = {j["item_id"] for j in judgments if isinstance(j, dict)}
    unknown = sorted(judged - set(items_by_id))
    unjudged = sorted(set(items_by_id) - judged)
    state.update({
        "complete": True,
        "fingerprint": fingerprint,
        "output_sha256": _file_sha256(output),
        "n_judged": len(judgments),
        "n_unjudged": len(unjudged),
        "unjudged_item_ids": unjudged,
        "unjudged_cells": sorted(
            f"{items_by_id[i]['family_id']}/{items_by_id[i]['variant']}"
            for i in unjudged),
        "judgments_outside_the_panel": unknown,
    })
    prov = judgments[0].get("provenance", {}) if judgments else {}
    state["model_id"] = prov.get("model_id")

    # The sidecar explains WHY. It is reported, and believed, only under the
    # fingerprint the completed output was produced under.
    sidecar: dict = {"exists": refusals.exists()}
    if refusals.exists():
        try:
            data = _load_json(refusals)
        except (json.JSONDecodeError, OSError) as exc:
            data = None
            sidecar["unreadable"] = str(exc)
        if isinstance(data, dict):
            records = _as_list(data, "refusals")
            sidecar.update({
                "fingerprint": data.get("fingerprint"),
                "fingerprint_matches": data.get("fingerprint") == fingerprint,
                "n_records": len(records),
                "cells": sorted(
                    f"{r.get('family_id')}/{r.get('variant')}"
                    for r in records if isinstance(r, dict)),
            })
    state["refusal_sidecar"] = sidecar
    return state


def target_state(judge_dir: Path) -> dict:
    """One arm of the comparison: both primaries, and the cells it lost."""
    rec: dict = {"judge_dir": _rel(judge_dir), "status": "incomplete_arms",
                 "arms": {}, "excluded_cells": [], "n_excluded": 0}
    items = _load_blinded(judge_dir)
    if items is None:
        rec["status"] = "no_blinded_items"
        rec["reason"] = "no blinded_items.json, so there is no panel to lose"
        return rec
    items_by_id = {it["item_id"]: it for it in items}
    rec["n_panel_items"] = len(items)
    rec["n_families"] = len({it["family_id"] for it in items})
    # The alignment 11.7 assumed and never re-checked at 600 cells: item_id is
    # derived from a seed-42 shuffle of THAT target's journal, so the four arms
    # should map every id to the same cell. The union is taken over CELLS, not
    # ids, so a divergence would not corrupt it -- but it would mean the arms
    # are not comparable in the way the blinding audit claimed, and that is
    # worth stating from the artifacts rather than from memory.
    rec["item_cell_map_sha256"] = hashlib.sha256(
        "".join(f"{i}\t{items_by_id[i]['family_id']}/"
                f"{items_by_id[i]['variant']}\n"
                for i in sorted(items_by_id)).encode("utf-8")).hexdigest()

    unjudged: dict[str, list[str]] = {}
    for primary in PRIMARIES:
        arm = arm_state(judge_dir, primary, items_by_id)
        rec["arms"][primary] = arm
        if arm["complete"]:
            unjudged[primary] = arm["unjudged_item_ids"]
    if len(unjudged) != len(PRIMARIES):
        return rec

    excluded_ids = sorted(set().union(*unjudged.values()))
    cells = sorted({f"{items_by_id[i]['family_id']}/{items_by_id[i]['variant']}"
                    for i in excluded_ids})
    rec.update({
        "status": "complete",
        "excluded_item_ids": excluded_ids,
        "excluded_cells": cells,
        "n_excluded": len(excluded_ids),
        "n_judged": len(items) - len(excluded_ids),
        # Within one target the rule is already the union over both primaries:
        # a cell one primary could not judge must not become a label from the
        # other primary alone.
        "exclusion_rule": (
            "the union over judges A and B of the cells whose judgment is "
            "absent from that judge's completed output"),
    })
    return rec


def derive(group_dir: Path, arms: dict[str, Path]) -> dict:
    """The common panel: what every arm lost, and what all of them must lose."""
    per_target = {name: target_state(judge_dir)
                  for name, judge_dir in arms.items()}
    incomplete = sorted(name for name, rec in per_target.items()
                        if rec["status"] != "complete")
    artifact: dict = {
        "kind": "iteration_11_common_panel_v1",
        "status": "pending_arms" if incomplete else "derived",
        "group": _rel(group_dir),
        "n_targets": len(arms),
        "targets": sorted(arms),
        "incomplete_targets": incomplete,
        "per_target": per_target,
        "union_excluded_cells": None,
        "rule": (
            "a cell is excluded from EVERY arm iff any arm's provider refused "
            "it, so all four analyses are computed over one identical panel"),
        "why": (
            "11.8 compares the four models with each other. aliyun's input "
            "moderation verdict depends on the request bytes and on the "
            "provider's policy at the moment of the call, and the four arms "
            "reach a given cell minutes apart, so without a union one arm can "
            "drop a family the others kept and the two Delta_TV values placed "
            "side by side would describe different families"),
    }
    if incomplete:
        artifact["pending_reason"] = (
            f"{len(incomplete)} of {len(arms)} target(s) have an incomplete "
            f"primary: {', '.join(incomplete)}. A union over a partial set of "
            f"arms would be smaller than the truth and phase 2 would apply it")
        return artifact

    panel_sizes = {rec["n_panel_items"] for rec in per_target.values()}
    family_counts = {rec["n_families"] for rec in per_target.values()}
    if len(panel_sizes) != 1 or len(family_counts) != 1:
        raise SystemExit(
            f"the arms do not hold one panel: n_panel_items={sorted(panel_sizes)}"
            f", n_families={sorted(family_counts)}")
    n_families = family_counts.pop()

    union = sorted(set().union(
        *[set(rec["excluded_cells"]) for rec in per_target.values()]))
    distinct = {frozenset(rec["excluded_cells"]) for rec in per_target.values()}
    maps = {name: rec["item_cell_map_sha256"]
            for name, rec in per_target.items()}
    reference = sorted(maps)[0]
    artifact.update({
        "n_panel_items": panel_sizes.pop(),
        "n_families_in_panel": n_families,
        "union_excluded_cells": union,
        "n_union_excluded_cells": len(union),
        "identical_across_targets": len(distinct) == 1,
        "cells_added_by_the_union": {
            name: sorted(set(union) - set(rec["excluded_cells"]))
            for name, rec in sorted(per_target.items())},
        "families_dropped": sorted({cell.split("/", 1)[0] for cell in union}),
        "n_families_common": n_families - len(
            {cell.split("/", 1)[0] for cell in union}),
        "item_id_alignment": {
            "reference_target": reference,
            "identical_across_targets": len(set(maps.values())) == 1,
            "per_target": {
                name: {"item_cell_map_sha256": digest,
                       "matches_reference": digest == maps[reference]}
                for name, digest in sorted(maps.items())},
        },
    })
    if not artifact["item_id_alignment"]["identical_across_targets"]:
        artifact["item_id_alignment"]["note"] = (
            "the arms map item_id to different cells, so the union is taken "
            "over (family_id, variant) rather than over item_id; the blinding "
            "audit's cross-target alignment channel does not hold at this "
            "panel size and has to be re-read")
    return artifact


def verify(group_dir: Path, arms: dict[str, Path],
           artifact_path: Path) -> list[str]:
    """The gate over the four ACTUAL artifacts phase 2 wrote."""
    issues: list[str] = []
    coverage: dict[str, set] = {}
    panels: dict[str, dict] = {}

    for name, judge_dir in arms.items():
        cov_path = judge_dir / COVERAGE_ARTIFACT
        if not cov_path.exists():
            issues.append(
                f"{name}: phase 2 has not finalized it -- no "
                f"{_rel(cov_path)}, so which cells that arm excluded is "
                f"unstated")
            continue
        cov = _load_json(cov_path)
        cells = {f"{c.get('family_id')}/{c.get('variant')}"
                 for c in cov.get("excluded_cells", [])}
        coverage[name] = cells
        if cov.get("cross_arm") is None:
            issues.append(
                f"{name}: {_rel(cov_path)} records no cross_arm block, so it "
                f"was finalized on this target's own refusals alone and never "
                f"saw the other three arms")

        cells_path = judge_dir / EVALUATION_DIR / EVALUATION_CELLS
        if not cells_path.exists():
            issues.append(f"{name}: no {_rel(cells_path)}, so the panel that "
                          f"analysis actually used cannot be read")
            continue
        seen = set()
        with cells_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    seen.add((rec["family_id"], rec["variant"]))
        panels[name] = {
            "n_cells": len(seen),
            "cells": seen,
            "families": {fam for fam, _ in seen},
        }

        # Every family the analysis does not carry has to be accounted for by
        # the restriction it says it applied. A family that quietly vanished
        # -- a truncated labels file, an estimand that raised and was caught
        # -- would otherwise leave the surviving-set comparison comparing two
        # sets that are each internally consistent and jointly wrong.
        report_path = judge_dir / EVALUATION_DIR / EVALUATION_REPORT
        items = _load_blinded(judge_dir) or []
        in_panel = {it["family_id"] for it in items}
        lost = in_panel - panels[name]["families"]
        declared: set = set()
        if report_path.exists():
            declared = set(_load_json(report_path).get(
                "panel_restriction", {}).get(
                    "families_dropped_incomplete", []))
        elif lost:
            issues.append(
                f"{name}: the analysis is missing {sorted(lost)} but wrote no "
                f"{_rel(report_path)}, so nothing states why")
        if declared != lost:
            issues.append(
                f"{name}: the report says it dropped {sorted(declared)} but "
                f"the analysis is missing {sorted(lost)}")

    if len(coverage) > 1:
        distinct = {frozenset(cells) for cells in coverage.values()}
        if len(distinct) != 1:
            union = sorted(set().union(*coverage.values()))
            per_arm = "; ".join(
                f"{n} excluded {sorted(c)}" for n, c in sorted(coverage.items()))
            issues.append(
                "the arms do not exclude the same cells (" + per_arm + "). "
                f"The common panel is the union {union}: every per-model "
                "analysis has to be regenerated on it, because two Delta_TV "
                "values computed over different families are not a "
                "cross-model comparison")

    if len(panels) > 1:
        families = {name: rec["families"] for name, rec in panels.items()}
        if len({frozenset(f) for f in families.values()}) != 1:
            union_lost = sorted(set().union(*families.values())
                                - set.intersection(*families.values()))
            issues.append(
                "the arms analysed different family sets: "
                + "; ".join(f"{n} analysed {len(f)}"
                            for n, f in sorted(families.items()))
                + f". Families not common to all four: {union_lost}")

    if not artifact_path.exists():
        issues.append(
            f"no derived common panel at {_rel(artifact_path)}; run this "
            f"script without --verify before phase 2 so the union exists")
    else:
        artifact = _load_json(artifact_path)
        if artifact.get("status") != "derived":
            issues.append(
                f"{_rel(artifact_path)} has status "
                f"{artifact.get('status')!r}, not 'derived'")
        else:
            union = set(artifact.get("union_excluded_cells", []))
            for name, cells in sorted(coverage.items()):
                if cells != union:
                    issues.append(
                        f"{name}: {COVERAGE_ARTIFACT} excludes "
                        f"{sorted(cells)} but the derived common panel is "
                        f"{sorted(union)}")
            lost = {cell.split("/", 1)[0] for cell in union}
            for name, rec in sorted(panels.items()):
                if rec["families"] & lost:
                    issues.append(
                        f"{name}: the analysis still carries "
                        f"{sorted(rec['families'] & lost)}, which the common "
                        f"panel drops")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true",
                        help="gate the four judge_coverage.json artifacts "
                             "instead of deriving the union")
    parser.add_argument("--profiles", type=Path, default=SCALE_PROFILES,
                        help="scale profile config naming the arms")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the common panel (default: "
                             "<cross_arm_group>/common_panel.json)")
    args = parser.parse_args(argv)

    group_dir, arms = group_arms(args.profiles)
    artifact_path = args.out or group_dir / COMMON_PANEL_ARTIFACT
    print(f"cross-arm group: {_rel(group_dir)}")
    print(f"arms:            {', '.join(sorted(arms))}")

    if args.verify:
        issues = verify(group_dir, arms, artifact_path)
        for name, judge_dir in sorted(arms.items()):
            cov = judge_dir / COVERAGE_ARTIFACT
            if cov.exists():
                data = _load_json(cov)
                cells = sorted(f"{c['family_id']}/{c['variant']}"
                               for c in data.get("excluded_cells", []))
                print(f"  {name:14s} judged {data.get('n_judged')}"
                      f"/{data.get('n_panel_items')}  excluded {cells}")
            else:
                print(f"  {name:14s} not finalized")
        if issues:
            print(f"\nCOMMON-PANEL GATE: FAIL ({len(issues)} issue(s))")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print("\nCOMMON-PANEL GATE: PASS — all four arms excluded the same "
              "cells and analysed the same families")
        return 0

    artifact = derive(group_dir, arms)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {_rel(artifact_path)}")
    for name, rec in sorted(artifact["per_target"].items()):
        if rec["status"] == "complete":
            print(f"  {name:14s} judged {rec['n_judged']}"
                  f"/{rec['n_panel_items']}  excluded {rec['excluded_cells']}")
        else:
            print(f"  {name:14s} {rec['status']}: "
                  f"{rec.get('reason') or 'a primary is incomplete'}")
    if artifact["status"] != "derived":
        print(f"\nCOMMON PANEL: {artifact['status']} — "
              f"{artifact['pending_reason']}")
        return 1
    print(f"\nunion excluded cells: {artifact['union_excluded_cells']}")
    print(f"identical across targets: "
          f"{artifact['identical_across_targets']}")
    print(f"families analysed by every arm: "
          f"{artifact['n_families_common']}"
          f"/{artifact['n_families_in_panel']}")
    print(f"item_id -> cell alignment identical: "
          f"{artifact['item_id_alignment']['identical_across_targets']}")
    added = {n: c for n, c in artifact["cells_added_by_the_union"].items() if c}
    if added:
        print("cells the union adds to each arm: "
              + "; ".join(f"{n} +{c}" for n, c in sorted(added.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
