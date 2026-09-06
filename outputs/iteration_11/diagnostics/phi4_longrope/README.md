# Phi-4 LongRoPE 4096-token cap: defect, diagnosis and repair

Machine-readable evidence for shim 9 in
`src/causal_mllm/replay/adapters/phi4_multimodal.py`. Everything here was
produced outside the repository while three Iteration 11.6 confirmatory
runs were still generating, because `provenance.code_dirty_paths` is
computed when each report is *written* — editing `src/` mid-run would have
failed runs that were otherwise perfect.

## What was observed

The 11.6 phi4_mm run recorded one failed cell out of 197:

    family CMST_779995 (mtmcs:type_b:000176), variant cross_modal
    error  {'category': 'generation', 'type': 'TypeError',
            'message': "'NoneType' object is not subscriptable"}

`_replay_family` stores only `classify_error(exc)` — category, type,
message — so the frame that raised is not in the evidence and had to be
recovered separately.

The journal itself pointed at the mechanism before the traceback did. Of
143 phi4 cells that had token counts at the time:

    max total (input+output) = 4086
    512-token histogram      = {..., 3072: 4, 3584: 5}   and nothing above 4096
    max input                = 4068
    max output               = 1536 (one cell hit the cap)

Inputs reach 4068 and outputs reach 1536, so the absence of any total
above 4096 is a cliff, not a short-response artifact. On the same panel
qwen35_2b reaches 16753 total tokens. `Phi4MMConfig` declares
`original_max_position_embeddings = 4096`.

## Mechanism

`modeling_phi4mm.py:2184`, inside the vendored
`Phi4MMForCausalLM.prepare_inputs_for_generation`:

    if (past_key_values
            and self.config.rope_scaling
            and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1):
        past_length = cache_position[0]        # <-- TypeError

That guard is the vendor's LongRoPE switch: the first time a sequence
passes 4096 it discards the KV cache so the whole prefix is recomputed
under `long_factor` instead of `short_factor`. transformers 5.x stopped
passing `cache_position` into a remote-code model's
`prepare_inputs_for_generation` — it builds one only inside the base
method's own return value (`generation/utils.py:595-605`, the block that
emits the "seems to expect `cache_position`" warning this checkpoint
triggered at load). The parameter therefore arrives `None`.

## Reproduction

`boundary_bracket_cpu.json` and `boundary_bracket_cuda1.json` force the
length directly, on the failing cell, with EOS suppressed so generation
runs to a chosen budget instead of stopping at 74 tokens. Both devices
give the identical bracket:

    total 4088  ok      total 4128  TypeError
    total 4095  ok      total 4328  TypeError
    total 4097  ok      total 5464  TypeError

4097 passes only because generation stops before the next
`prepare_inputs_for_generation` call; the guard fires on the first decode
step whose sequence length *is* 4097. The CPU and GPU brackets agreeing
exactly is what makes this a Python-level API incompatibility rather than
a device or memory fault.

## Repair

Shim 9 expresses the vendor's intent in 5.x terms instead of its 4.x
spelling. Two intermediate attempts are recorded here because both were
wrong in instructive ways:

* Supplying `cache_position` alone removes the `TypeError` and then hits
  `ValueError: max() arg is an empty sequence` — the vendor's guard nulls
  the cache, the vendored forward's legacy-BC branch builds a config-less
  `DynamicCache()` with zero layers, and `get_max_length()` is
  `max(layer.get_max_length() for layer in self.layers)`.
* Also tolerating that reaches `AttributeError: 'DynamicCache' object has
  no attribute 'to_legacy_cache'` at `modeling_phi4mm.py:1781`, from the
  same branch. Restoring `to_legacy_cache` would have been the wrong fix:
  by then the base has already sliced `input_ids` to the single new token
  (`next_sequence_length=1`, which `_sample` pins whenever `use_cache` is
  set, independently of whether a cache survives), so the forward would
  have run one token against an empty cache. The exception would have
  stopped being loud and started being wrong — a continuation that
  silently lost its whole 3928-token prefix.

The shim therefore keeps a `Cache` object, substitutes a fresh empty one
of the same class built from the model config, and asks for the whole
sequence. It supplies `cache_position` on every step past the switch too,
not just the switching one, because the vendor's *outer* condition stays
true for every later position and reads `cache_position[0]` before the
inner test that decides nothing has to be done.

## Verification

`shim_verification.json` — one model load, two passes over the same cell:

    inert_below_boundary              True   totals 4088/4095/4097 byte-identical
    before_failed_above               True   4128/4328/5464 all raise without it
    after_ok_above                    True   4128/4328/5464 all complete with it
    deterministic                     True   a repeated 400-token run reproduces
    reached_full_cap                  True   out=1536 at total 5464
    full_prefix_recompute_observed    True   one forward, input_len=4097,
                                             cache_layers=32, cache_len=0
    recompute_at_boundary_and_once    True

The sharpest of these is not in the checks list. The first 890 characters
— the first 169 generated tokens — of the 4128-token generation are
byte-identical to the 169-token generation, and divergence begins exactly
at the switch. That is the vendor's own documented behaviour ("it will
cause downside of slower at this single token position"), and it is what
shows the shim changed nothing before the boundary rather than merely
suppressing an exception after it.

The forward trace is what separates a repair from a cover-up: had the base
still been slicing to one token, the generation would have completed and
produced fluent, prefix-less garbage with no exception at all.

`prefill_over_boundary.json` covers the other way in. A prompt padded to a
6328-token prefill — past the switch before any decoding — generates 64
tokens with exactly one long-input/empty-cache forward, at step 0, i.e. an
ordinary prefill and no spurious mid-generation recompute. The unshimmed
code fails this case too, so any panel cell whose prompt alone exceeds
4096 would have failed the replay.

## Consequence for the analysis

The repair removes a censoring that was informative, not random: a cell
crashed exactly when phi4 was about to keep talking past 4096 tokens, so
the responses it lost were the long ones. That biases phi4's truncation
rate downward — the single cell in the partial run that did hit the 1536
cap could only do so because its input was small (1996 tokens) — and it
biases its length distribution against arms with no such ceiling. The
partial journal is quarantined outside the repository and phi4_mm is
replayed from scratch under the fix.

Crossing the switch is still a Phi-4-specific behaviour: the vendor
swaps rope factors and recomputes the prefix, which the Qwen and Ministral
arms do not do at that position. `_check_rope_headroom` in
`scripts/iter11_model_preflight.py` keeps the preflight smoke prompt below
the switch for exactly that comparability reason, and that gate is
deliberately left as it is. It cannot be extended to the confirmatory
panel — prompt lengths are fixed by the frozen panel and output lengths
are the model's own — so cells that cross have to be identified after the
fact. They are: the switch point is `rope_switch_position: 4096` in
`outputs/iteration_11/preflight/phi4_mm/preflight.json`, and a cell crosses
when `input_token_count + output_token_count` exceeds it, both of which are
already per-record fields in the journal. Any phi4 effect should be
re-estimated with those cells held out before it is reported as comparable.

One limitation is worth stating rather than leaving to be discovered. The
adapter docstring says every shim "is recorded per run in
`runtime_metadata()["phi4_shims"]`", and it is — but `run_replay_stage`
takes only `hardware` and `deterministic_algorithms` out of that dict, so
neither `phi4_shims` nor `rope_switch_position` reaches a replay report or
an eligibility report. Only the 11.2 preflight artifact persists them.
Which shims were active during a confirmatory run is therefore auditable
through `provenance.code_commit` plus the clean-tree assertion rather than
from the journal itself. That is sufficient — it is precisely what the
code-scoped assertion guarantees — but it is indirect, and adding the shim
list to run-level provenance is a reasonable follow-up. It is deliberately
not done here: it would change the provenance schema underneath three
already-committed reports that do not carry it.
