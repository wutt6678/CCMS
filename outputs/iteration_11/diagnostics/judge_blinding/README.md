# Model-identity blinding audit

The frozen clause is

    "output_blinding": "target-model identity is blinded in judge prompts;
                        variant/family blinded as in Iteration 10"

Iteration 10 had one target, so the first half was vacuous. Iteration 11 has
four, which makes it a testable claim with three channels a leak could travel
down, plus a fourth that is structural rather than a leak:

1. **payload** — the blinded item or the rendered judge prompt carries the
   `model_key`, `model_id`, adapter or marketed size label. Scanned for EVERY
   arm's vocabulary in EVERY arm's payload, not just its own: a judge that
   could separate the arms by a stray mention of a *different* model would be
   just as unblinded.
2. **context** — the system prompt or conversation context differs between
   targets, so a judge could separate the arms with no name appearing.
   It does not differ: all four targets record
   `system_prompt_sha256 = e51b41e6a8226440…` and `prompt_template_revision
   = v1`.
3. **response** — the model names itself inside its own answer ("As Qwen, I
   cannot…"). No payload hygiene blinds this channel, and one target was
   enough that Iteration 10 could never have exposed it.
4. **item-id alignment** — whether the four sessions' blinded ids denote the
   same cell. Not a leak, but the property that makes pooling the four
   sessions into one silently catastrophic.

## Result

All four journals complete at 600 cells, so this is the full population:
**2400 blinded items**, and channels 1-3 found **zero** leaks. No target
identity in any payload, none in the rendered prompt built by the frozen judge
A configuration, no judge identity (`qwen3.8-max`, `glm-5.2`, `kimi-k3`) in any
prompt, no Iteration-10 blinding violation (`family_id` or `variant` reaching
the prompt), and no response self-identification in any of the 2400 answers.
The rubric in use hashes to the frozen `ce6c2005…` v1.1.

The payload scan is exhaustive over all 2400 items; the rendered-prompt scan
covers 20 items per arm (80 prompts), because rendering reads and base64s every
referenced image and the payload it renders is a deterministic function of the
item the exhaustive scan already cleared.

The identity vocabulary was deliberately over-inclusive — `\bqwen\b`, `\b2B\b`,
`\bmistral\b`, `\bmicrosoft\b`, `\bphi\b` and the full `model_id` strings among
others — because a false positive costs a look while a false negative silently
de-blinds the confirmatory judging.

### Channel 4, now measured

The first run recorded this as `inconclusive` and had to: the journals were at
438, 60, 600 and 83 records, and since `item_id` is an index into a per-target
seed-42 shuffle, journals of different lengths are not comparable. Drawing a
pooling verdict from partial journals would have reported "not aligned" for a
design that is perfectly aligned at completion — the wrong answer in the unsafe
direction.

With all four at 600 it is `measured`, and every pair is fully aligned:

    ministral3_3b ~ phi4_mm      600 shared ids, 600 identical cells
    ministral3_3b ~ qwen35_2b    600 shared ids, 600 identical cells
    ministral3_3b ~ qwen35_4b    600 shared ids, 600 identical cells

So `item-0164` denotes `CMST_795308`/`cross_modal` in all four arms and every
other id likewise. The sessions must therefore stay separate: pooling them
would collide each `item_id` four ways and merge distinct models' responses
under one label. That is also what made the provider-refusal diagnosis in
`../judge_moderation/` transferable across arms — a refusal at index 164 was
the same cell everywhere.

### The mirror is checked, not assumed

Channels 1-4 are all computed from a re-derivation of the judge pipeline's
`prepare_blinded_items`, which cannot be imported (that module raises at import
time without `LLM_JUDGE_API_KEY`). An unchecked copy would be an assumption
carrying the whole audit, so wherever the pipeline has already written its own
`blinded_items.json` this compares the two field by field —
`item_id`, `family_id`, `variant`, `response_sha256`, conversation history and
terminal query. `mirror_crosscheck.identical` is `true` for all four arms over
all 2400 items.

## Reproducing

    python3 scripts/iter11_blinding_audit.py             # derive and write
    python3 scripts/iter11_blinding_audit.py --verify    # read-only gate

The panel hash is checked against the frozen protocol's
`panel_validated_families_sha256` rather than restated, so a regenerated panel
fails loudly instead of silently re-auditing different cells. `--verify`
re-derives and compares to the committed artifact; it currently reports
`VERIFY PASS`.

`tests/unit/test_iter11_judge_profiles.py` pins channels 1-3 as a regression
test over a sample, so this artifact is the full-population audit and the test
is the guard that stays green.
