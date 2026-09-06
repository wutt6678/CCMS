"""Iteration 11.8: the four arms are compared over ONE panel, or not at all.

``build_judge_coverage`` unions the refusals of judges A and B for one target
session. That is the right rule inside a session and silent about the one that
matters for 11.8: the four TARGETS are placed side by side, and each derives
its own exclusion from its own two primaries. Aliyun's input-moderation verdict
is a function of the request bytes and of the provider's policy at the moment
of the call, and the four arms reach a given cell minutes apart -- so one arm
can lose a family the other three kept. Both still report a Delta_TV, both
artifacts look complete, and nothing in either says the two numbers describe
different families.

These tests pin the two halves of the fix: the producer that derives the union
from the eight completed primary outputs BEFORE phase 2, so the arms agree by
construction, and the gate that reads the four actual ``judge_coverage.json``
artifacts afterwards and refuses to pass arms that do not exclude the same
cells and analyse the same families.

CI-safe: no network, and no dependency on the real judge tree -- the fixtures
build a two-arm group under ``tmp_path`` so a divergence can be manufactured
rather than waited for.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from causal_mllm.evaluation.errors import EvaluationError

ROOT = Path(__file__).resolve().parents[2]

VARIANTS = ("neutral", "text_only", "vision_only", "cross_modal", "shuffle",
            "history_reset")

#: The cell judge A actually loses on the frozen panel, and one it does not.
REFUSED = ("CMST_000001", "cross_modal")
OTHER = ("CMST_000002", "shuffle")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_pipeline():
    """The pipeline raises at import time without an API key."""
    os.environ.setdefault("LLM_JUDGE_API_KEY", "test-key-never-used")
    return _load_script("run_llm_judge_pipeline")


common_panel = _load_script("iter11_common_panel")
pipeline = _load_pipeline()


def _cell(item: dict) -> str:
    return f"{item['family_id']}/{item['variant']}"


def _panel(n_families: int = 3) -> list[dict]:
    items, i = [], 0
    for f in range(n_families):
        for variant in VARIANTS:
            items.append({
                "item_id": f"item-{i:04d}", "family_id": f"CMST_{f:06d}",
                "variant": variant, "response": f"r{i}",
                "response_sha256": f"{i:064d}"})
            i += 1
    return items


@pytest.fixture
def group(tmp_path):
    """A two-arm cross-model group, plus the profiles file that declares it."""
    root = tmp_path / "judge"
    items = _panel()
    targets = {}
    for name in ("t1", "t2"):
        judge_dir = root / name
        judge_dir.mkdir(parents=True)
        (judge_dir / "blinded_items.json").write_text(
            json.dumps(items), encoding="utf-8")
        targets[name] = judge_dir
    profiles = tmp_path / "scale_profiles.json"
    profiles.write_text(json.dumps({
        # A single-arm profile, which must be left alone.
        "scale_c": {"output_dir": "outputs/scale_c/llm_judge_artifacts"},
        "iteration_11_t1": {"output_dir": str(root / "t1"),
                            "cross_arm_group": str(root)},
        "iteration_11_t2": {"output_dir": str(root / "t2"),
                            "cross_arm_group": str(root)},
    }), encoding="utf-8")
    return {"root": root, "items": items, "targets": targets,
            "profiles": profiles, "artifact": root / "common_panel.json"}


def _arm(group, target: str, primary: str, refused=(),
         fingerprint: str | None = None, sidecar=None) -> Path:
    """A completed primary. ``refused`` is ``(family_id, variant)`` pairs.

    Membership is what the completed output does NOT carry, exactly as in the
    real tree; the sidecar is optional and is only ever evidence about why.
    """
    judge_dir = group["targets"][target]
    lost = set(refused)
    judged = [it for it in group["items"]
              if (it["family_id"], it["variant"]) not in lost]
    out = judge_dir / f"llm_labels_judge_{primary}.json"
    out.write_text(json.dumps([
        {"item_id": it["item_id"], "family_id": it["family_id"],
         "variant": it["variant"], "response_sha256": it["response_sha256"],
         "judgment": {}} for it in judged]), encoding="utf-8")
    out.with_name(out.name + ".fingerprint").write_text(
        fingerprint or f"fp-{target}-{primary}", encoding="utf-8")
    if sidecar is not None:
        out.with_suffix(".refusals.json").write_text(
            json.dumps(sidecar), encoding="utf-8")
    return out


def _checkpoint(group, target: str, primary: str, held: int) -> Path:
    """An arm still running: resume state, no completed output, no sidecar."""
    path = group["targets"][target] / \
        f"llm_labels_judge_{primary}.checkpoint.json"
    path.write_text(json.dumps({
        "fingerprint": f"fp-{target}-{primary}",
        "judgments": [{"item_id": it["item_id"], "judgment": {}}
                      for it in group["items"][:held]]}), encoding="utf-8")
    return path


def _complete(group, target: str, refused=()) -> None:
    for primary in common_panel.PRIMARIES:
        _arm(group, target, primary, refused=refused)


def _derive(group, write: bool = True) -> dict:
    """Derive the union over the group's arms, and write it where main() does.

    ``derive`` is pure; the file is what phase 2 and ``--verify`` read, so a
    test that wants the gate to see a union has to produce the artifact too.
    """
    artifact = common_panel.derive(
        group["root"], {name: judge_dir
                        for name, judge_dir in sorted(
                            group["targets"].items())})
    if write:
        group["artifact"].write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


def _coverage(group, target: str, cells, cross_arm: bool = True) -> Path:
    """The artifact phase 2 would have written for one target."""
    items = {_cell(it): it for it in group["items"]}
    judge_dir = group["targets"][target]
    detail = [{"item_id": items[c]["item_id"],
               "family_id": items[c]["family_id"],
               "variant": items[c]["variant"],
               "response_sha256": items[c]["response_sha256"]}
              for c in sorted(cells)]
    coverage = {"n_panel_items": len(group["items"]),
                "n_excluded": len(detail),
                "n_judged": len(group["items"]) - len(detail),
                "excluded_cells": detail}
    if cross_arm:
        coverage["cross_arm"] = {"artifact": "common_panel.json"}
    path = judge_dir / common_panel.COVERAGE_ARTIFACT
    path.write_text(json.dumps(coverage), encoding="utf-8")
    return path


def _analysed(group, target: str, dropped=(), declared=None) -> None:
    """The per-cell record of the panel one analysis actually used.

    ``dropped`` is ``family_id/variant`` strings, the form every artifact in
    this chain uses. A family that loses ANY variant is dropped WHOLE -- the
    bootstrap resamples families and evaluates all five estimands on the same
    resample, so a family with five of six cells cannot contribute any of them.
    The fixture has to do what ``restrict_panel_to_labels`` does or it would
    describe a panel the estimator cannot produce.

    ``declared`` overrides what the report SAYS it dropped, so the two can be
    made to disagree the way a family that quietly vanished would.
    """
    out = group["targets"][target] / common_panel.EVALUATION_DIR
    out.mkdir(parents=True, exist_ok=True)
    lost_families = {cell.split("/", 1)[0] for cell in dropped}
    kept = [it for it in group["items"]
            if it["family_id"] not in lost_families]
    families = sorted({it["family_id"] for it in kept})
    (out / common_panel.EVALUATION_CELLS).write_text(
        "".join(json.dumps({"family_id": it["family_id"],
                            "variant": it["variant"]}) + "\n" for it in kept),
        encoding="utf-8")
    (out / common_panel.EVALUATION_REPORT).write_text(json.dumps({
        "estimands": {"n_families": len(families)},
        "panel_restriction": {
            "n_families_analysed": len(families),
            "excluded_cells": sorted(dropped),
            "families_dropped_incomplete": sorted(
                lost_families if declared is None else declared)},
    }), encoding="utf-8")


class TestTheGroupIsReadFromConfig:
    def test_the_arms_are_the_iteration_11_profiles(self, group):
        root, arms = common_panel.group_arms(group["profiles"])
        assert root == group["root"]
        assert sorted(arms) == ["t1", "t2"]

    def test_a_single_arm_profile_is_not_part_of_a_group(self, group):
        # scale_c has no cross_arm_group and must not be swept in: it is one
        # arm, its exclusion is its own, and its sealed artifacts are not to
        # be re-derived against a comparison it is not part of.
        _, arms = common_panel.group_arms(group["profiles"])
        assert "llm_judge_artifacts" not in arms

    def test_an_arm_that_does_not_name_its_group_is_an_error(self, group):
        profiles = json.loads(group["profiles"].read_text(encoding="utf-8"))
        del profiles["iteration_11_t2"]["cross_arm_group"]
        group["profiles"].write_text(json.dumps(profiles), encoding="utf-8")
        with pytest.raises(SystemExit, match="declares no"):
            common_panel.group_arms(group["profiles"])

    def test_two_arms_naming_different_groups_is_an_error(self, group):
        profiles = json.loads(group["profiles"].read_text(encoding="utf-8"))
        profiles["iteration_11_t2"]["cross_arm_group"] = str(
            group["root"] / "elsewhere")
        group["profiles"].write_text(json.dumps(profiles), encoding="utf-8")
        with pytest.raises(SystemExit, match="different"):
            common_panel.group_arms(group["profiles"])

    def test_one_arm_alone_cannot_have_a_common_panel(self, tmp_path):
        profiles = tmp_path / "profiles.json"
        profiles.write_text(json.dumps({
            "iteration_11_only": {
                "output_dir": str(tmp_path / "judge" / "only"),
                "cross_arm_group": str(tmp_path / "judge")}}),
            encoding="utf-8")
        with pytest.raises(SystemExit, match="at least two"):
            common_panel.group_arms(profiles)


class TestMembershipIsDerivedNotInherited:
    def test_a_complete_arm_that_refused_nothing_excludes_nothing(self, group):
        _complete(group, "t1")
        rec = common_panel.target_state(group["targets"]["t1"])
        assert rec["status"] == "complete"
        assert rec["excluded_cells"] == []
        assert rec["n_judged"] == rec["n_panel_items"] == 18

    def test_a_cell_with_no_judgment_is_excluded_without_a_sidecar(
            self, group):
        _arm(group, "t1", "A", refused=[REFUSED])
        _arm(group, "t1", "B")
        rec = common_panel.target_state(group["targets"]["t1"])
        assert rec["excluded_cells"] == ["CMST_000001/cross_modal"]
        assert rec["arms"]["A"]["refusal_sidecar"] == {"exists": False}

    def test_the_arms_of_one_target_are_unioned(self, group):
        # Judge A lost one cell, judge B another. Neither arm can label the
        # other's gap, so the target as a whole excludes both.
        _arm(group, "t1", "A", refused=[REFUSED])
        _arm(group, "t1", "B", refused=[OTHER])
        rec = common_panel.target_state(group["targets"]["t1"])
        assert rec["excluded_cells"] == ["CMST_000001/cross_modal",
                                         "CMST_000002/shuffle"]

    def test_a_stale_sidecar_naming_a_judged_cell_is_not_honoured(
            self, group):
        # The read-only half of the P1 defect. The sidecar claims a cell this
        # arm DID judge, under a different fingerprint; believing it would
        # exclude a valid cell. Membership comes from the output, so the claim
        # is recorded and ignored.
        items = {_cell(it): it for it in group["items"]}
        judged = "CMST_000000/neutral"
        _arm(group, "t1", "A", refused=[REFUSED], fingerprint="current",
             sidecar={"fingerprint": "stale-run",
                      "refusals": [{"item_id": items[judged]["item_id"],
                                    "family_id": items[judged]["family_id"],
                                    "variant": items[judged]["variant"],
                                    "response_sha256":
                                        items[judged]["response_sha256"]}]})
        _arm(group, "t1", "B")
        rec = common_panel.target_state(group["targets"]["t1"])
        assert rec["excluded_cells"] == ["CMST_000001/cross_modal"]
        sidecar = rec["arms"]["A"]["refusal_sidecar"]
        assert sidecar["fingerprint_matches"] is False
        assert sidecar["cells"] == [judged], \
            "reported, so the disagreement is visible rather than absorbed"

    def test_a_sidecar_under_the_right_fingerprint_says_so(self, group):
        items = {_cell(it): it for it in group["items"]}
        _arm(group, "t1", "A", refused=[REFUSED], fingerprint="fp-t1-A",
             sidecar={"fingerprint": "fp-t1-A",
                      "refusals": [{"item_id": items["CMST_000001/cross_modal"]
                                    ["item_id"],
                                    "family_id": REFUSED[0],
                                    "variant": REFUSED[1]}]})
        _arm(group, "t1", "B")
        rec = common_panel.target_state(group["targets"]["t1"])
        assert rec["arms"]["A"]["refusal_sidecar"]["fingerprint_matches"]

    def test_an_incomplete_arm_is_not_a_small_exclusion(self, group):
        # The dangerous reading of a half-judged arm is that the cells it has
        # not reached were refused. It is an incomplete arm, and the union
        # cannot be derived until it finishes.
        _checkpoint(group, "t1", "A", 9)
        _arm(group, "t1", "B")
        rec = common_panel.target_state(group["targets"]["t1"])
        assert rec["status"] == "incomplete_arms"
        assert rec["excluded_cells"] == []
        assert "checkpoint holds 9" in rec["arms"]["A"]["reason"]

    def test_a_judgment_outside_the_panel_is_reported(self, group):
        out = _arm(group, "t1", "A")
        records = json.loads(out.read_text(encoding="utf-8"))
        records.append({"item_id": "item-9999", "judgment": {}})
        out.write_text(json.dumps(records), encoding="utf-8")
        _arm(group, "t1", "B")
        rec = common_panel.target_state(group["targets"]["t1"])
        assert rec["arms"]["A"]["judgments_outside_the_panel"] == ["item-9999"]


class TestTheUnion:
    def test_arms_that_lost_the_same_cell_agree(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        artifact = _derive(group)
        assert artifact["status"] == "derived"
        assert artifact["identical_across_targets"] is True
        assert artifact["union_excluded_cells"] == ["CMST_000001/cross_modal"]
        assert artifact["cells_added_by_the_union"] == {"t1": [], "t2": []}
        assert artifact["families_dropped"] == ["CMST_000001"]
        assert artifact["n_families_common"] == 2
        assert artifact["n_families_in_panel"] == 3

    def test_arms_that_lost_different_cells_are_unioned(self, group):
        # THE CASE build_judge_coverage cannot see: t1's provider refused a
        # cell t2's served twenty minutes later. Left alone, both finalize,
        # both report a Delta_TV over their own families, and 11.8 places the
        # two numbers side by side.
        _complete(group, "t1", refused=[REFUSED])
        _complete(group, "t2", refused=[OTHER])
        artifact = _derive(group)
        assert artifact["identical_across_targets"] is False
        assert artifact["union_excluded_cells"] == [
            "CMST_000001/cross_modal", "CMST_000002/shuffle"]
        assert artifact["cells_added_by_the_union"] == {
            "t1": ["CMST_000002/shuffle"],
            "t2": ["CMST_000001/cross_modal"]}
        assert artifact["families_dropped"] == ["CMST_000001", "CMST_000002"]
        assert artifact["n_families_common"] == 1

    def test_an_incomplete_arm_blocks_the_union(self, group):
        _complete(group, "t1", refused=[REFUSED])
        _arm(group, "t2", "B")
        artifact = _derive(group)
        assert artifact["status"] == "pending_arms"
        assert artifact["union_excluded_cells"] is None
        assert artifact["incomplete_targets"] == ["t2"]
        assert "A union over a partial set of arms" in \
            artifact["pending_reason"]

    def test_derive_exits_non_zero_while_an_arm_is_incomplete(self, group,
                                                              capsys):
        _complete(group, "t1", refused=[REFUSED])
        _arm(group, "t2", "B")
        assert common_panel.main(
            ["--profiles", str(group["profiles"]),
             "--out", str(group["artifact"])]) == 1
        assert json.loads(
            group["artifact"].read_text(encoding="utf-8"))[
                "status"] == "pending_arms"
        assert "pending_arms" in capsys.readouterr().out

    def test_the_artifact_is_reproducible(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        assert _derive(group) == _derive(group), \
            "a --verify that cannot reproduce the artifact verifies nothing"

    def test_the_item_id_alignment_is_read_from_the_artifacts(self, group):
        # Assumed since 11.7 and never re-checked at 600 cells: item_id comes
        # from a seed-42 shuffle of that target's own journal. The union is
        # taken over cells so a divergence would not corrupt it, but the claim
        # is worth stating from the panels rather than from memory.
        for target in ("t1", "t2"):
            _complete(group, target)
        assert _derive(group)["item_id_alignment"][
            "identical_across_targets"] is True

    def test_a_divergent_item_id_map_is_reported_not_absorbed(self, group):
        for target in ("t1", "t2"):
            _complete(group, target)
        items = json.loads((group["targets"]["t2"] / "blinded_items.json")
                           .read_text(encoding="utf-8"))
        # item-0000 and item-0006 are the neutral cell of two DIFFERENT
        # families: swapping them keeps the panel the same size and the same
        # families, and changes only which id names which cell.
        items[0]["family_id"], items[6]["family_id"] = \
            items[6]["family_id"], items[0]["family_id"]
        (group["targets"]["t2"] / "blinded_items.json").write_text(
            json.dumps(items), encoding="utf-8")
        alignment = _derive(group)["item_id_alignment"]
        assert alignment["identical_across_targets"] is False
        assert alignment["per_target"]["t2"]["matches_reference"] is False
        assert "taken over (family_id, variant)" in alignment["note"]

    def test_arms_holding_different_panels_are_not_comparable(self, group):
        # A union is only meaningful over arms that hold the same panel. Two
        # targets whose journals differ in size are not a cross-model
        # comparison of the frozen panel and must not be averaged.
        _complete(group, "t1")
        bigger = _panel(4)
        (group["targets"]["t2"] / "blinded_items.json").write_text(
            json.dumps(bigger), encoding="utf-8")
        held = group["items"]
        group["items"] = bigger
        try:
            _complete(group, "t2")
        finally:
            group["items"] = held
        with pytest.raises(SystemExit, match="do not hold one panel"):
            _derive(group)


class TestTheGateOverTheActualArtifacts:
    def test_four_arms_on_one_panel_pass(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        for target in ("t1", "t2"):
            _coverage(group, target, ["CMST_000001/cross_modal"])
            _analysed(group, target, dropped=["CMST_000001/cross_modal"])
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert issues == []

    def test_arms_that_excluded_different_cells_fail(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        _coverage(group, "t1", ["CMST_000001/cross_modal"])
        _coverage(group, "t2", ["CMST_000002/shuffle"])
        for target in ("t1", "t2"):
            _analysed(group, target, dropped=["CMST_000001/cross_modal"])
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert any("do not exclude the same cells" in i for i in issues)
        assert any("regenerated" in i for i in issues)
        assert any("CMST_000001/cross_modal" in i and "CMST_000002/shuffle"
                   in i for i in issues), "the union has to be named"

    def test_arms_that_analysed_different_families_fail(self, group):
        # Same exclusion, different surviving panel: the coverage artifacts
        # agree, so only the per-cell record of what each analysis actually
        # used catches it.
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        for target in ("t1", "t2"):
            _coverage(group, target, ["CMST_000001/cross_modal"])
        _analysed(group, "t1", dropped=["CMST_000001/cross_modal"])
        _analysed(group, "t2", dropped=["CMST_000001/cross_modal",
                                        "CMST_000002/shuffle"])
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert any("analysed different family sets" in i for i in issues)

    def test_a_coverage_artifact_with_no_cross_arm_block_fails(self, group):
        # Finalized on its own refusals, before the producer existed. The
        # numbers may well be right; the artifact does not say they are, and
        # that is the failure.
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        for target in ("t1", "t2"):
            _coverage(group, target, ["CMST_000001/cross_modal"],
                      cross_arm=False)
            _analysed(group, target, dropped=["CMST_000001/cross_modal"])
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert sum("records no cross_arm block" in i for i in issues) == 2

    def test_a_coverage_artifact_disagreeing_with_the_union_fails(self, group):
        _complete(group, "t1", refused=[REFUSED])
        _complete(group, "t2", refused=[OTHER])
        _derive(group)
        union = ["CMST_000001/cross_modal", "CMST_000002/shuffle"]
        for target in ("t1", "t2"):
            _coverage(group, target, union)
            _analysed(group, target, dropped=union)
        # One arm is then re-finalized on a narrower panel.
        _coverage(group, "t2", ["CMST_000002/shuffle"])
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert any("the derived common panel is" in i for i in issues)

    def test_an_analysis_still_carrying_a_dropped_family_fails(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        for target in ("t1", "t2"):
            _coverage(group, target, ["CMST_000001/cross_modal"])
        _analysed(group, "t1", dropped=["CMST_000001/cross_modal"])
        _analysed(group, "t2")
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert any("still carries" in i for i in issues)

    def test_a_target_phase_2_has_not_finalized_is_an_issue(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert sum("has not finalized" in i for i in issues) == 2

    def test_no_derived_union_is_an_issue_not_a_pass(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
            _coverage(group, target, ["CMST_000001/cross_modal"])
            _analysed(group, target, dropped=["CMST_000001/cross_modal"])
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert any("no derived common panel" in i for i in issues)

    def test_a_family_the_analysis_dropped_without_saying_so_fails(
            self, group):
        # The surviving-set comparison compares sets. A family that vanished
        # for a reason the report does not state leaves both sets internally
        # consistent, so the report's own restriction is checked against what
        # the analysis actually carries.
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        for target in ("t1", "t2"):
            _coverage(group, target, ["CMST_000001/cross_modal"])
            _analysed(group, target, dropped=["CMST_000001/cross_modal",
                                              "CMST_000002/shuffle"],
                      declared=["CMST_000001"])
        issues = common_panel.verify(
            group["root"], {"t1": group["targets"]["t1"],
                            "t2": group["targets"]["t2"]}, group["artifact"])
        assert sum("the report says it dropped" in i for i in issues) == 2

    def test_verify_exits_non_zero_on_any_issue(self, group, capsys):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        assert common_panel.main(
            ["--verify", "--profiles", str(group["profiles"]),
             "--out", str(group["artifact"])]) == 1
        assert "COMMON-PANEL GATE: FAIL" in capsys.readouterr().out

    def test_verify_exits_zero_when_the_arms_agree(self, group, capsys):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        for target in ("t1", "t2"):
            _coverage(group, target, ["CMST_000001/cross_modal"])
            _analysed(group, target, dropped=["CMST_000001/cross_modal"])
        assert common_panel.main(
            ["--verify", "--profiles", str(group["profiles"]),
             "--out", str(group["artifact"])]) == 0
        assert "COMMON-PANEL GATE: PASS" in capsys.readouterr().out


class TestTheProducerAndThePipelineDeriveTheSameThing:
    """Two implementations of one rule, pinned against each other.

    The producer is read-only and runs before phase 2; ``collect_provider_refusals``
    runs inside phase 2 and repairs the sidecars as it derives. If the two ever
    disagree about WHICH cells an arm lost, the producer's union would not be
    the union phase 2 applies, and the disagreement would be invisible: the
    gate compares coverage artifacts with the producer's own output.
    """

    def test_membership_agrees_on_a_half_written_arm(self, group, tmp_path):
        _arm(group, "t1", "A", refused=[REFUSED])
        _arm(group, "t1", "B")
        derived = common_panel.target_state(group["targets"]["t1"])
        by_judge, _ = pipeline.collect_provider_refusals(
            group["targets"]["t1"], group["items"])
        applied = sorted({
            f"{r['family_id']}/{r['variant']}"
            for records in by_judge.values() for r in records})
        assert applied == derived["excluded_cells"] == [
            "CMST_000001/cross_modal"]

    def test_membership_agrees_when_a_stale_sidecar_disagrees(self, group):
        items = {_cell(it): it for it in group["items"]}
        stale = {"fingerprint": "stale-run",
                 "refusals": [{"item_id": items["CMST_000000/neutral"]
                               ["item_id"],
                               "family_id": "CMST_000000",
                               "variant": "neutral",
                               "response_sha256":
                                   items["CMST_000000/neutral"]
                                   ["response_sha256"]}]}
        _arm(group, "t1", "A", refused=[REFUSED], sidecar=stale)
        _arm(group, "t1", "B", sidecar=stale)
        derived = common_panel.target_state(group["targets"]["t1"])
        by_judge, stale_found = pipeline.collect_provider_refusals(
            group["targets"]["t1"], group["items"])
        applied = sorted({
            f"{r['family_id']}/{r['variant']}"
            for records in by_judge.values() for r in records})
        assert applied == derived["excluded_cells"]
        assert stale_found, "both report the sidecar rather than believing it"


class TestThePipelineAppliesTheUnion:
    def _own(self, group, target: str, refused=()) -> tuple[dict, set]:
        """``(by_judge, own_cells)`` the way main() computes them."""
        for primary in common_panel.PRIMARIES:
            _arm(group, target, primary, refused=refused)
        by_judge, _ = pipeline.collect_provider_refusals(
            group["targets"][target], group["items"])
        return by_judge, set(refused)

    def test_a_profile_with_no_group_is_untouched(self, group):
        assert pipeline.load_cross_arm_panel(
            None, group["targets"]["t1"], group["items"], set()) is None

    def test_a_declared_group_with_no_artifact_is_an_error(self, group):
        _, own = self._own(group, "t1", refused=[REFUSED])
        with pytest.raises(EvaluationError,
                           match="does not exist"):
            pipeline.load_cross_arm_panel(
                str(group["root"]), group["targets"]["t1"], group["items"],
                own)

    def test_a_pending_artifact_is_an_error(self, group):
        _complete(group, "t1", refused=[REFUSED])
        _arm(group, "t2", "B")
        _derive(group)
        _, own = self._own(group, "t1", refused=[REFUSED])
        with pytest.raises(EvaluationError, match="pending_arms"):
            pipeline.load_cross_arm_panel(
                str(group["root"]), group["targets"]["t1"], group["items"],
                own)

    def test_an_artifact_derived_over_other_arms_is_an_error(self, group):
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        _, own = self._own(group, "t1", refused=[REFUSED])
        with pytest.raises(EvaluationError, match="different set of arms"):
            pipeline.load_cross_arm_panel(
                str(group["root"]),
                group["root"] / "t3", group["items"], own)

    def test_a_stale_union_is_an_error_not_a_quiet_restriction(self, group):
        # The arm lost a cell after the union was derived. Applying the old
        # union would analyze a panel the other three are not on.
        for target in ("t1", "t2"):
            _complete(group, target, refused=[REFUSED])
        _derive(group)
        _, own = self._own(group, "t1", refused=[REFUSED, OTHER])
        with pytest.raises(EvaluationError, match="re-run"):
            pipeline.load_cross_arm_panel(
                str(group["root"]), group["targets"]["t1"], group["items"],
                own)

    def test_the_union_extends_the_exclusion_to_another_arms_cell(self,
                                                                  group):
        _complete(group, "t1", refused=[REFUSED])
        _complete(group, "t2", refused=[OTHER])
        _derive(group)
        by_judge, own = self._own(group, "t1", refused=[REFUSED])
        cross = pipeline.load_cross_arm_panel(
            str(group["root"]), group["targets"]["t1"], group["items"], own)
        coverage, excluded = pipeline.build_judge_coverage(
            by_judge, [], group["items"], ("a-model", "b-model"),
            cross_arm=cross)
        assert coverage["n_excluded"] == 2
        assert coverage["n_judged"] == 16
        by_origin = {f"{c['family_id']}/{c['variant']}":
                     c["exclusion_origin"] for c in coverage["excluded_cells"]}
        assert by_origin == {"CMST_000001/cross_modal": "this_target",
                             "CMST_000002/shuffle": "another_arm"}
        other = next(c for c in coverage["excluded_cells"]
                     if c["exclusion_origin"] == "another_arm")
        assert other["refused_by"] == []
        assert other["refused_in_targets"] == ["t2"]
        assert coverage["cross_arm"]["n_cells_added_by_the_union"] == 1
        assert coverage["cross_arm"]["identical_across_targets"] is False
        assert len(excluded) == 2

    def test_a_single_arm_coverage_artifact_is_unchanged(self, group):
        # scale_b and scale_c are sealed. Same input, no group: the artifact
        # must carry exactly the keys it carried before, in the same order.
        by_judge, _ = self._own(group, "t1", refused=[REFUSED])
        coverage, _ = pipeline.build_judge_coverage(
            by_judge, [], group["items"], ("a-model", "b-model"))
        assert list(coverage) == [
            "n_panel_items", "n_excluded", "n_judged", "excluded_item_ids",
            "excluded_cells", "exclusion_rule", "outcome_independent",
            "per_judge", "stale_refusals_ignored"]
        assert list(coverage["excluded_cells"][0]) == [
            "item_id", "family_id", "variant", "response_sha256",
            "refused_by", "reason"]
        assert coverage["exclusion_rule"].startswith(
            "the union of cells any primary's provider refused")

    def test_a_union_naming_a_cell_outside_the_panel_is_an_error(self,
                                                                 group):
        by_judge, own = self._own(group, "t1", refused=[REFUSED])
        cross = {"cells": {("CMST_000001", "cross_modal"),
                           ("CMST_999999", "neutral")},
                 "artifact": group["artifact"], "artifact_data": {}}
        with pytest.raises(EvaluationError, match="not a cell of this"):
            pipeline.build_judge_coverage(
                by_judge, [], group["items"], ("a-model", "b-model"),
                cross_arm=cross)
