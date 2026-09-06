# Per-identity vision capability, and what each judge arm therefore claims

`identity_vision_capability.json` records whether each frozen judge identity
actually takes the family media in. Produced by
`scripts/iter11_probe_judge_gateway.py`; read by
`scripts/run_llm_judge_pipeline.py`, which passes it to `finalize_ensemble` so
`judge_sensitivity.json` labels each primary's arm.

## Measurement

The same instruction is sent twice to each identity, once with the probe image
attached and once without, and the verdict comes from the provider's OWN token
accounting:

* `prompt_tokens_details.image_tokens` — present and positive means the
  provider tokenized pixels;
* the `prompt_tokens` difference between the two requests — a model that
  discards the image reports the same count both ways.

The same-text control is what makes the second signal meaningful. This probe
originally sent two DIFFERENT texts (a JSON echo, then "describe this image"),
so their token counts were not comparable and the probe could report an
identity as usable while carrying no evidence about whether it could see. That
is how a blind primary judge reached a frozen ensemble: the pre-flight gate
asked "does it answer?" and not "does it look?".

    identity       role         prompt_tokens   image_tokens   verdict
    qwen3.8-max    primary A     76 -> 300          262        sees_image
    kimi-k3        adjudicator  100 -> 476          371        sees_image
    glm-5.2        primary B     27 ->  27           —         blind

`glm-5.2` reports 27 prompt tokens with and without a 533,849-byte PNG
attached, and no `image_tokens` key at all. The pixels are discarded before the
model sees them.

## What the response says is never decisive

Two blind identities were measured, and they present in opposite ways.

`glm-5.2` declines: *"I cannot describe the image because no image was
provided in your message."* `deepseek-v3.2` — measured with `--also-probe`, so
it is recorded under `reference_identities` and is NOT an ensemble member —
confabulates instead:

    A lone red canoe floats on a calm, glassy lake surrounded by towering,
    misty mountains under a soft, cloudy sky. {"ok": true}

Fluent, specific, confident, and wrong: the probe image is a baseball player
holding a bat, which is what the two sighted identities describe. Its
`prompt_tokens` is 19 with and without the image, so the pixels never arrived.

Worse for anyone tempted to use response text as a signal, that sentence is
IDENTICAL in both requests. At temperature 0.0 and seed 42 the model is
deterministic and the image contributes nothing to its input, so the
confabulation is not even a reaction to the image — it is a fixed completion
driven by the instruction "describe this image" alone.

That is why response equality is not used as a third signal either, tempting
though it is. A sighted model that declines to describe images on policy
grounds would also answer identically both ways, and labelling it blind would
be the same error in the other direction. So the response text is recorded in
the artifact and used for nothing: only the accounting the provider does on its
own input decides.

Symmetrically, an identity that reports no usage breakdown at all is recorded
`unmeasurable`, not `blind`: absence of evidence is not evidence of blindness,
and labelling an arm vision-ablated on that basis would disparage a judge that
may be perfectly sighted.

## Reference identities are measured but never load-bearing

`--also-probe MODEL_ID` runs the identical measurement on an identity outside
the frozen ensemble. It is kept out of every invariant on purpose: it does not
count toward the three-distinct-identity check, its unreachability fails
nothing, and it lands in `reference_identities` rather than `identities` — the
only section the judge pipeline reads. A model nobody froze must not be able to
decide how a frozen arm is labelled. It exists so that a claim about how
blindness presents rests on a measurement instead of an anecdote.

## Consequence: the judge_B arm is a vision ablation

`build_judge_arm_meta` reads the verdict and labels the arm. A blind primary's
`judge_sensitivity.json` entry carries `"arm": "vision-ablated"` and a note; a
sighted primary's carries `"arm": "model-choice"`; an unprobed identity carries
neither.

The two labels claim different things. A model-choice arm is evidence about
what another model would judge. A vision-ablated arm is evidence about what the
ensemble would judge with the family media held out. Reporting the second as
the first would describe a comparison nobody made — and on a cross-modal safety
dataset, where 300 of the 600 cells carry an image, the media are not a detail.

This is a labelling correction, not a change to the ensemble. The frozen
protocol names these three identities and they are what ran.

The labels are additive, and one consequence of that is worth stating plainly:
Iteration 10's sealed `judge_sensitivity.json` predates them and does not carry
them. Its `glm-5.2` arm is recorded with no `arm` key at all, so a reader of
that artifact alone cannot tell a vision ablation from a model choice. It is
not regenerated — it is sealed — and the correction is available for it only
through this directory and `build_judge_arm_meta`. The `ensemble(...)` model id
string IS preserved exactly as those sealed artifacts carry it, because
reformatting a field two committed reports already hold would be a schema
change dressed up as a labelling one.

## What the blindness does and does not explain

`outputs/iteration_11/judge_vision_ablation/` answers that separately, by
holding model identity FIXED (`qwen3.8-max`, rubric v1.1, temperature 0.0, seed
42) and removing only the image, over 299 image-bearing frozen items with
`prompt_sha256` asserted equal to the frozen judge-A record per item:

    A vs B          (frozen; A sighted, B blind)   0.502 disagreement
    A_blind vs B    (both blind)                   0.589
    A_blind vs A    (vision effect on A alone)     0.495

Removing vision changes judge A's own judgments on about half the
image-bearing cells, so the media carry real weight. But a blind A disagrees
with B MORE than a sighted A does, so B is not "A without vision": making A
blind does not make it agree with B. The A/B disagreement rate is therefore a
genuine model difference and not an artifact of B's blindness (McNemar
p ≈ 0.0008 against the blindness-as-cause explanation).

Two things follow, and they pull in opposite directions, so both are stated:
the divergence between the A and B arms is NOT explained away by blindness, but
the B arm still measures a vision-ablated judgment and must be labelled as one.

## The adjudicator is sighted, so no label is vision-blind

`kimi-k3` reports `image_tokens=371`. Every A/B disagreement is resolved by an
adjudicator that received the image, so the adjudicated labels — the ones the
11.5 stratification is derived from and the ones 11.8 reports — are
vision-informed throughout. That was audited directly for Iteration 10: of 600
items, 361 agreements resolve to judge A's judgment (sighted) and all 239
disagreements resolve to the adjudicator's (sighted), and B never
unilaterally determines a label.

## Reproducing

    python3 scripts/iter11_probe_judge_gateway.py --also-probe deepseek-v3.2

The `--also-probe` is what puts `reference_identities` in the artifact, so
reproducing the committed file byte-for-byte needs it; without it the probe
writes the three ensemble identities only. Twelve requests with a reference
identity, nine without, no spend to speak of. Re-running it on 2026-09-07
returned the same token counts as the run before it for all three ensemble
identities, so the measurement is stable even though the moderation verdict on
the same gateway is not (see `../judge_moderation/`).

The probe still exits non-zero if an identity is unreachable, but blindness
alone is NOT a failure: a blind judge serves requests and returns parseable
output, so it is recorded in the evidence rather than in the exit code. Whether
a blind primary is acceptable is a protocol question, not a reachability one,
and the frozen protocol had already named these identities by the time this was
measured.
