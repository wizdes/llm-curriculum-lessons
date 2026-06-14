# L24 — Capstone: The Ablation Harness

You trained one GPT on TinyStories. Now the real question of any modeling
project: **which choices actually mattered?** Did 8 attention heads beat 2? Was
the learning rate the bottleneck, or the model size? Guessing is cheap and
usually wrong. The honest way to answer is an **ablation**.

## What an ablation is

To *ablate* something is to remove or change it and watch what breaks. An
**ablation** in machine learning is one disciplined experiment: you **change
exactly ONE thing, hold everything else fixed, retrain, and compare the
results.** If only one knob moved, then any difference in the outcome can be
blamed on that knob — nothing else changed to muddy the picture.

Change two things at once and you learn nothing: if the model got better, was it
the extra heads or the higher learning rate? You can't say. The discipline of
"one thing at a time" is the entire reason an ablation is trustworthy.

A few words we'll use throughout:

- A **variant** is one version of the model you train — the baseline plus a
  single override (e.g. "the baseline, but with `n_head=2`").
- A **sweep** is a set of variants you run together to compare. The heads sweep,
  below, is three variants: `n_head` set to 2, 4, and 8.
- The **baseline** is the variant that changes nothing — your reference point.
  Every other variant is described relative to it.
- An **ablation harness** is the small program that runs a sweep *fairly* and
  collects the results into a table you can rank. This lesson builds that
  harness.

## Warm-up: the config diff

A **config** here is the training configuration — the bag of settings that
defines a run: number of attention heads, embedding size, learning rate, batch
size, and so on. We carry it as a Python dict (and, on disk, as a YAML file).

A variant's config is built by the **merge-order rule**: start from the base
config and lay the variant's `override` dict on top, so any key the override
names wins and every untouched key keeps its base value. In Python that is
`{**base, **override}` — splat the base first, then the override, so the override
is applied *last* and therefore takes precedence. This is exactly why "one knob
moved": the override is the only thing layered on, so everything it does not
mention is held fixed at the baseline.

Before running anything, recall a small tool you'll lean on: the **config diff**.
Given a base config and a variant config, it returns **only** the keys whose
values differ:

```python
config_diff({"n_head": 8, "lr": 3e-4}, {"n_head": 2, "lr": 3e-4})
# -> {"n_head": 2}
```

Read that aloud: "the base has `n_head=8` and `lr=3e-4`; the variant has
`n_head=2` and the same `lr`; so the only difference is `n_head`, and its new
value is 2." The diff returns `{"n_head": 2}` — the single changed key, with the
variant's value.

Three behaviors define it exactly, and the grader checks all three:

1. **Only changed keys appear.** Keys that match are dropped. `lr` was equal in
   both, so it's absent from the diff.
2. **Identical configs diff to `{}`** (the empty dict). The baseline changes
   nothing relative to itself, so its diff is empty — that's how you spot the
   baseline row at a glance.
3. **Nested dicts are compared recursively.** If a value is itself a dict, the
   diff descends into it and reports only the changed *sub*-keys:

```python
config_diff({"opt": {"lr": 3e-4, "wd": 0.1}},
            {"opt": {"lr": 1e-3, "wd": 0.1}})
# -> {"opt": {"lr": 1e-3}}
```

Inside `opt`, only `lr` changed (`wd` matched), so the diff is `{"opt": {"lr":
1e-3}}` — not the whole `opt` block.

The procedure that produces all three behaviors is one walk over the keys:

1. Take the **union** of both configs' keys (so a key present in only one side is
   still considered).
2. For each key, fetch `b = base[k]` and `v = variant[k]` (a missing side reads
   as `None`).
3. If **both** values are dicts, **recurse** — diff them — and keep the result
   only if that sub-diff is non-empty (rule 3, the nested case).
4. Otherwise keep the key when `b != v`, storing the *variant's* value (rules 1
   and 2: equal values are dropped, so identical configs leave the diff `{}`).

The diff is what makes an ablation row legible. Instead of dumping a full config
next to every run, you label each run by *what it changed*: not "here are 30
settings" but "this run is the baseline **except** `n_head=2`."

## The rule that makes a sweep fair: reproducibility

Here is the trap an ablation harness exists to avoid. An ablation compares
variants. But neural-network training uses randomness — the initial weights are
drawn at random, and the training batches are sampled. If you *rerun the very
same variant* and get different numbers, then your comparison between *different*
variants is meaningless: you can't tell whether `heads_4` beat `heads_2` because
of the architecture or just because the dice rolled differently that day.

The cure is the **seed**. A seed is a single integer that fixes the starting
point of the pseudo-random number generator (RNG). Feed the RNG the same seed and
it produces the exact same "random" stream every time — the same initial weights,
the same batch order. **Reproducibility** means: same seed in, same numbers out.

So every variant must seed **deterministically from `spec.seed`** (the seed
declared in the sweep). Concretely, inside `run_variant` you:

```python
torch.manual_seed(spec.seed)        # before building the model
model = GPT(cfg.to_gpt_config())    # weights initialized from THIS seed
...
train_loop(model, get_batch, optimizer, steps, seed=spec.seed)  # batches too
```

`torch.manual_seed(spec.seed)` must come *before* `GPT(...)`, because the model's
weights are randomized at construction; seed first and that randomization is
pinned. Then `train_loop` is also passed `seed=spec.seed`, so the batch stream
replays identically. With both pinned, two runs of the same variant on one
machine are **bit-for-bit identical** — and the grader proves it: it runs one
variant twice at `seed=42` and asserts the two runs' `final_train_loss` and
`final_val_loss` match exactly.

The classic way to break this rule is to seed from the wall clock instead of
`spec.seed` — the exact mistake the grader hunts by name. It's spelled out in
**The named trap**, under "How the grader checks you" below.

(One honest caveat: this bit-for-bit guarantee is the same-machine,
fixed-CPU-seed case. Reproducing the exact same floats across different machines
or hardware backends is a harder problem — a different fight, not this one.)

## A fair token budget

To compare variants fairly, they must each do the *same amount of work*. The
obvious dial is **step count** — the number of training updates. But a step isn't
a fixed amount of work: a variant with a bigger batch processes more text per
step. So "same number of steps" can quietly mean "one variant saw twice as much
text," and you'd be comparing apples to oranges.

The fair unit is the **token budget**: the total number of tokens (the model's
unit of text) every variant is allowed to train on. A **token** is one chunk of
text — a word-piece — and one batch contains `batch_size * block_size` tokens
(`batch_size` sequences, each `block_size` tokens long). The harness derives the
step count *from* the budget:

```python
steps = token_budget // (batch_size * block_size)
```

Read aloud: "steps equals the token budget, integer-divided by the number of
tokens in one batch." So a variant with a larger batch runs *fewer* steps to stay
under the same budget. The integer division floors to whole batches, so each
variant trains on the largest whole-batch multiple that fits under the budget;
the `tokens` field records that actual count (`steps * tokens_per_step`), which is
what you compare. Fixing the budget — not the steps — is what keeps the
comparison fair. (The harness also clamps to at least 1 step via `max(1, ...)`,
so a tiny budget still runs one batch, even if that nudges just over the budget.)

## The results table

Each call to `run_variant` returns one **result row**: a dict with these eight
keys (the schema the grader checks for completeness):

| key                | what it holds |
|--------------------|---------------|
| `config_name`      | the variant's name, e.g. `"heads_2"` |
| `seed`             | the seed it ran at (`spec.seed`) |
| `tokens`           | tokens actually trained on (`steps * tokens_per_step`) |
| `final_train_loss` | the last training loss — error on the final training batch |
| `final_val_loss`   | the **eval loss** — one extra no-grad call to `get_batch` (the ranking key) |
| `steps`            | how many update steps ran |
| `wall_clock_s`     | how long the run took, in seconds |
| `config_diff`      | the diff vs the base config — `{}` for the baseline |

A word on the two losses. **Training loss** (`final_train_loss`) is the error on
the final batch the model just updated on. The **eval loss** (`final_val_loss`)
is computed by `_estimate_val_loss`: one extra forward pass with gradients turned
off, via one more call to the same injected `get_batch`. It's a separate
measurement, but it draws from the same source as training — not a truly held-out
split. (The name keeps the conventional "val" label and the convention of ranking
by it; a production setup would point this eval call at a dedicated validation
set.) We rank by `final_val_loss` because it's the eval-side number, the one you'd
trust over training loss when picking a winner.

The rows go to a **dedicated results JSON** — a plain file holding a list of
these dicts — and **NOT** the lesson progress store. Ablation rows are experiment
artifacts (run data you might throw away and regenerate), not lesson-completion
state. `write_results` dumps the list to that file; `read_results` reads it back
**sorted by `final_val_loss` ascending**, so the lowest validation loss — the
best variant — is first. A missing file reads as an empty list `[]`.

Why sort by `final_val_loss` and not training loss? Training loss can drop just
because a variant fit its last batch well, so it flatters whichever variant
happened to land an easy batch. The eval loss is measured on a separate no-grad
batch, so it's the more trustworthy number for picking a winner. Sorting
ascending puts that winner at the top of the table where you'll look first.

## A worked sweep, end to end

Here's the heads sweep (`sweeps/heads_sweep.yaml`), the runnable example. A
**sweep YAML** is just the spec written down:

```yaml
base_config: configs/tinystories-4m.yaml   # the baseline config on disk
token_budget: 5000000                       # 5M tokens, same for every variant
seed: 42                                     # the one seed every variant uses
variants:
  - name: heads_2
    override: { n_head: 2 }
  - name: heads_4
    override: { n_head: 4 }
  - name: heads_8        # baseline: n_head unchanged from the base config
    override: { n_head: 8 }
```

Read aloud: "start from the `tinystories-4m` config; give every variant a 5
million token budget and seed 42; run three variants that set `n_head` to 2, 4,
and 8." The base config already has `n_head=8`, so `heads_8` is the baseline —
its override changes nothing, which is exactly why its `config_diff` will be `{}`.

One constraint worth understanding: every `n_head` value must **divide
`n_embd`** (the embedding size, 160 in this config). Attention splits the
embedding into `n_head` heads, each of size `n_embd / n_head`; a head count that
doesn't divide evenly isn't a slower model, it's a hard error. 2, 4, and 8 all
divide 160, so all three are legal.

Running the sweep is four steps — predict, dry-run, train, read:

1. **Predict the order first.** Write down where you think `heads_2`,
   `heads_4`, and `heads_8` will land before you see a single number. (The
   prediction prompts live in `experiments.md`.) A prediction you got wrong
   teaches more than ten you got right.
2. **Dry-run** to confirm the sweep parses and the overrides are what you meant
   — it loads the spec, prints each variant's resolved config diff, and exits
   *without training*:

   ```bash
   uv run python lessons/L27-ablation-harness/ablate.py --sweep sweeps/heads_sweep.yaml --dry-run
   ```

   You'd see each variant's diff: `heads_2 -> {"n_head": 2}`, `heads_4 ->
   {"n_head": 4}`, `heads_8 -> {}` (the baseline).
3. **Run for real** — this trains all three variants under the shared seed and
   budget and writes the results table to the dedicated results JSON:

   ```bash
   uv run python lessons/L27-ablation-harness/ablate.py --sweep sweeps/heads_sweep.yaml
   ```
4. **Read the table.** `read_results` ranks it best-first (lowest
   `final_val_loss` on top), so you compare your prediction against the winner.

The ranked table might look like this (illustrative numbers, lowest
`final_val_loss` on top):

| config_name | final_val_loss | final_train_loss | steps | config_diff |
|-------------|----------------|------------------|-------|-------------|
| heads_8     | 1.80           | 1.50             | …     | `{}`        |
| heads_4     | 2.30           | 2.00             | …     | `{"n_head": 4}` |
| heads_2     | 2.90           | 2.50             | …     | `{"n_head": 2}` |

Reading the table: `heads_8` won, with the lowest validation loss; the `{}` diff
flags it as the baseline. Each row labels itself by *what it changed*, every run
used seed 42 and the same 5M-token budget, and the table is sorted so the winner
is on top. That's a fair, legible ablation — the thing the harness exists to
produce. (These numbers are illustrative; whether more heads *really* mean lower
validation loss, or whether splitting the embedding into too many tiny heads
stops helping, is the open question the heads experiment makes you predict and
then check.)

## The grand finale: turn off the causal mask (thought experiment)

There's a second sweep, `sweeps/mask_off.yaml`, and it's deliberately **not
runnable** — it's a documented thought experiment. A decoder-only GPT works
because of its **causal mask**: when predicting token *t*, the model may attend
only to tokens at positions `<= t`. Token *t* can never peek at its own future.

The frozen `references.gpt` applies that mask **unconditionally** — there is no
flag to disable it. So the `causal_mask: false` override in `mask_off.yaml` has
nothing to bind to, and the harness won't execute it. It exists only to show the
override *syntax* and to anchor the prediction.

Reason it through: suppose token *t* could attend to token *t+1* — the very token
it's trying to predict. The answer would be sitting right there in the input.
Training loss would collapse toward zero, because the model just copies the
visible answer instead of learning to predict it. This is **information
leakage** — the objective is no longer measuring prediction at all, it's
measuring the model's ability to read off an answer it can already see.

Notice the catch for *this* harness: its eval loss runs the same forward pass on
the same kind of batch, so an unmasked model would leak at eval time too — both
losses would look artificially low together. The honest diagnosis needs a
non-leaking evaluation (re-mask the model when you measure held-out loss); only
then does the gap appear — training loss near zero, true held-out loss high or
worse. That gap is the signature you'd hunt for.

That's the whole point of an ablation harness. An ablation that "improves"
training loss by removing the causal mask has measured cheating, not learning.
The causal mask isn't a performance knob — it's the thing that makes next-token
prediction *mean* anything. Catching a number that looks better for the wrong
reason is exactly the job you built this harness to do.

See `experiments.md` for the prediction prompts for both the heads sweep and the
causal-mask thought experiment — predict first, then check.

## Exercise

Implement in `my_work/L27-ablation-harness/ablation_harness.py` (run
`uv run grade start L27-ablation-harness` to get the scaffold). For imports, lean on
`references.*` (plus numpy/torch and ordinary stdlib like `json`, `time`,
`dataclasses`, `pathlib`, and `yaml` for the YAML parse) — the one hard rule is
**no cross-lesson imports**, so nothing reaches into another lesson's `my_work`.
The scaffold's three
stubs are the functions below: `config_diff`, `load_sweep_spec`, and
`run_variant`. Your `load_sweep_spec` returns a `SweepSpec` — a small dataclass
holding a parsed sweep (`base_config` path, `token_budget`, `seed`, and
`variants`, a list of `{"name": str, "override": dict}`) — so define that
dataclass in the module too. Two more helpers, `write_results` and
`read_results`, round-trip the dedicated results JSON (write the list, read it
back sorted by `final_val_loss` ascending); the grader imports them, so they live
in this module as well.

- `config_diff(base, variant)` — the **warm-up**, the diff from the section
  above. Parameters: `base` and `variant` are two config dicts. It returns a new
  dict holding **only** the keys whose value differs, taking the *variant's*
  value; nested dicts are compared recursively (keep a sub-diff only if it is
  non-empty); two identical configs return `{}`.
- `load_sweep_spec(path)` — the YAML parser. Parameter: `path` is the path to a
  sweep YAML carrying `base_config`, `token_budget`, `seed`, and `variants:
  [{name, override}]`. It returns a `SweepSpec` with those four fields parsed
  (`token_budget` and `seed` coerced to `int`, `variants` to a list).
- `run_variant(spec, variant, get_batch)` — train one variant and return its
  result row. Parameters: `spec` is the `SweepSpec` (it supplies `base_config`,
  `token_budget`, and `seed`); `variant` is one `{"name", "override"}` dict;
  `get_batch` is an **injected** zero-argument function that returns one
  `(x, y)` batch of token ids — the grader passes a synthetic one, the CLI a
  dataset-backed one, so `run_variant` never touches a dataset itself. Inside it
  you: build a `TrainingConfig` from `spec.base_config`, apply
  `variant["override"]` with the merge-order rule (`{**base, **override}`),
  derive `steps = max(1, spec.token_budget // (batch_size * block_size))`, call
  `torch.manual_seed(spec.seed)` **before** constructing `GPT`, build the
  optimizer, then `train_loop(model, get_batch, optimizer, steps,
  seed=spec.seed)` (the `seed` argument is keyword-only). It returns a result
  dict with all eight schema keys: `config_name`, `seed`, `tokens`
  (`steps * batch_size * block_size`), `final_train_loss` (last entry of the
  loop's `loss_history`), `final_val_loss` (one extra no-grad forward pass on a
  fresh `get_batch`), `steps`, `wall_clock_s`, and `config_diff` (the diff vs the
  unmodified base).

Then grade it: the site's Run Grader button or `uv run grade L27-ablation-harness`.

## How the grader checks you

The grader is CPU-only, synthetic, and seconds-fast — no dataset and no real run.
It checks three things, in dependency order (a broken `config_diff` reports its
own message instead of masking the heavier checks downstream):

- **`config_diff`** is run on three fixed pairs: a flat case (only the changed
  key surfaces — `{"n_head": 2}`), a nested-dict case (only the changed sub-key
  surfaces — `{"opt": {"lr": 1e-3}}`), and two identical configs (must diff to
  `{}`).
- **Reproducibility** runs `run_variant` twice on a tiny on-disk config with a
  fixed synthetic `get_batch` at `seed=42`, and asserts the two runs are *bitwise
  identical*: `final_train_loss` and `final_val_loss` must match exactly across
  the reruns. It also checks `steps == 50` (the token budget,
  `4 * 16 * 50`, divided by one batch's `4 * 16` tokens) and that the returned
  row carries the full eight-key schema. Same machine, fixed CPU seed, *is*
  bitwise reproducible — this is the gate, not the harder cross-platform case.
- **The results table** round-trips three synthetic rows through `write_results`
  then `read_results`: all three come back, each carrying the full schema, sorted
  by `final_val_loss` ascending so the lowest-loss row (`heads_8`, the baseline
  with `config_diff == {}`) is first.

### The named trap: a non-deterministic seed

The graded trap is `reproducibility_seed`. A broken harness seeds from a
*non-deterministic source* — the wall clock — instead of `spec.seed`:

```python
torch.manual_seed(int(time.time() * 1e6) % (2**31))   # WRONG: time, not the spec
```

The clock reads differently every microsecond, so every run draws a fresh random
seed. Each run *looks* fine on its own. But rerun the same variant and the
numbers move — the two runs diverge — and the bitwise-reproducibility check fires
its teaching message: **each variant must seed from spec.seed** so the sweep is
reproducible. The fix is the one rule from the reproducibility section: seed from
`spec.seed` (before building the model, and passed to `train_loop`), never from
the clock or from "whatever the RNG happened to be set to."

## The hint ladder

If the grader's message is not enough, open the hints. `config_diff` and
`reproducibility` each have three levels: a conceptual nudge (what's wrong), then
the invariant with the key formula (walk the union of keys / seed from
`spec.seed` and derive steps from the budget), then pseudocode. Try level 1
before climbing — and if a rerun's numbers ever drift, that is the
reproducibility trap, not flaky hardware: go back and check where you seeded.

## Done when

`uv run grade L27-ablation-harness` shows all-green: `config_diff` reports only changed keys
(flat and nested) and `{}` for identical configs; `run_variant` is bitwise
reproducible across two `seed=42` runs, takes the budget-derived 50 steps, and
returns the full eight-key schema; and `write_results`/`read_results` round-trip
the table sorted by `final_val_loss` ascending. Then run the experiments and
record your predictions — the heads sweep and the causal-mask thought
experiment.
