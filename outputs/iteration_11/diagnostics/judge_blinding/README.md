# Model-identity blinding audit

The frozen clause is

    "output_blinding": "target-model identity is blinded in judge prompts;
                        variant/family blinded as in Iteration 10"

Iteration 10 had one target, so the first half was vacuous. Iteration 11 has
four, which makes it a testable claim with three channels a leak could travel
down, plus a fourth that is structural rather than a leak:

1. **payload** — the blinded item or the rendered judge prompt carries the
   `model_key`, `model_id`, adapter or marketed size label.
2. **context** — the system prompt or conversation context differs between
   targets, so a judge could separate the arms with no name appearing.
   It does not differ: all four targets record
   `system_prompt_sha256 = e51b41e6a8226440…` and `prompt_template_revision
   = v1`.
3. **response** — the model names itself inside its own answer ("As Qwen, I
   cannot…"). No payload hygiene blinds this channel, and one target was
   enough that Iteration 10 could never have exposed it.
4. **item-id alignment** — whether the four sessions' blinded ids denote the
   same cell. They do, since each is an index into a seed-42 shuffle of the
   same 600-cell panel, so the sessions must stay separate; pooling them
   would collide every id four ways.

## Result

Over every generation that existed when this ran — 1169 blinded items across
the four targets — channels 1, 2 and 3 found **zero** leaks: no target
identity in the payload, none in the rendered prompt built by the frozen judge
A configuration, no judge identity (`qwen3.8-max`, `glm-5.2`, `kimi-k3`) in any
prompt, no Iteration-10 blinding violation (`family_id` or `variant` reaching
the prompt), and no response self-identification. The rubric in use hashes to
the frozen `ce6c2005…` v1.1.

The identity vocabulary was deliberately over-inclusive — `\bqwen\b`, `\b2B\b`,
`\bmistral\b`, `\bmicrosoft\b` and the full `model_id` strings among others —
because a false positive costs a look while a false negative silently
de-blinds the confirmatory judging.

Channel 4 is recorded as `inconclusive` here and must be read that way: the
journals were at 438, 60, 600 and 83 records when this ran, and since item_id
is an index into a per-target shuffle, journals of different lengths are not
comparable. Drawing a pooling verdict from partial journals would report "not
aligned" for a design that is perfectly aligned at completion — the wrong
answer in the unsafe direction. Re-run once all four journals hold 600 cells.

`tests/unit/test_iter11_judge_profiles.py` pins channels 1–3 as a regression
test over a sample, so this artifact is the full-population audit and the test
is the guard that stays green.
