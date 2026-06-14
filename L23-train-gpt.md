# L21 — Training a GPT

In L20 you assembled the whole model: a `GPT` is a stack of transformer blocks
with a token-embedding table at the bottom (a lookup table that turns each token
id into a vector) and a `lm_head` at the top (a final `Linear` layer that turns
each position's vector into one score per vocabulary word). That model can run a
forward pass — feed in a sequence of token ids, get back scores for what the next
token might be — but its weights are still random. It predicts noise. This lesson
is the step that makes it *learn*: a full training run that drives the loss down
until the model produces plausible next characters.

You will not write the inner training loop here, because you already have it. The
`references/train_loop.py` you have used since L14 owns the per-step machinery:
`zero_grad → forward → cross-entropy → backward → optimizer.step()`, the
per-step metrics log, and checkpointing with exact resume. (Each of those terms
is unpacked below, so don't worry if "cross-entropy" or "checkpoint" is fuzzy.)
Your job is to wire the pieces the loop needs — build the model, build the
optimizer *the right way*, and hand the loop a function that produces training
batches — then watch the loss fall.

We open with a 30-second warm-up: re-build the L18 single causal-attention head
from memory, cold, before any new material. (Recalling a thing right before you
build on it is what makes it stick.) Then we build the optimizer, the data
pipeline, and the run.

## The objective: next-token prediction

Before any code, here is what "training" means for this model, in plain words.

The model reads a sequence of tokens and, at every position, predicts the *next*
token. "Token" here means one character — this is a **character-level** language
model, so the vocabulary is the set of distinct characters (65 of them) and a
"token id" is just an integer index into that set. Given the characters seen so
far, the model outputs a score for each of the 65 possible next characters. Train
it well and the high scores land on the characters that actually tend to come
next.

To turn those scores into a number we can minimize, we use the **next-token
cross-entropy loss**. Cross-entropy is the standard loss for "pick the right
class out of K" problems; here the K classes are the 65 characters. It works in
two moves. First it runs the model's raw scores (called **logits** — unbounded
real numbers, one per class) through **softmax**, which exponentiates them and
divides by their sum so they become a probability distribution over the 65
characters (all non-negative, summing to 1). Then it reports
$-\log p_{\text{correct}}$: the negative natural logarithm of the probability the
model assigned to the character that *actually* came next. Read aloud:
*"how surprised was the model by the true next character?"* If the model gave the
right character probability 1, the loss is $-\log 1 = 0$ — no surprise. If it gave
it a tiny probability, $-\log(\text{tiny})$ is large — a big penalty. Averaged
over every position in every sequence, that average surprise is the single number
training pushes downward.

One number to anchor it: at the start, weights are random, so the model spreads
its probability roughly evenly over all 65 characters, giving the true one about
$1/65$. The loss is then about $-\log(1/65) = \log 65 \approx 4.17$. A trained
character model on this kind of data reaches roughly **1.5**. That drop from
~4.17 to ~1.5 is the whole story of the run.

You do not implement cross-entropy in this lesson — the frozen `train_loop` calls
`torch.nn.CrossEntropyLoss`, which fuses the softmax and the $-\log p$ step into
one numerically-stable call. What you *do* have to get right is feeding it the
correct **targets**, which is the next subtlety.

## The targets are the inputs shifted by one

The loss needs to know, at every position, what the correct next token *is*. We
never label this data by hand — the text labels itself. For any window of tokens,
the correct next token at each position is simply the token that follows it in the
original text. So the **targets are the inputs shifted one position to the left.**

Concretely, take the string `"hello"` as token ids and a window of length 4:

```text
position:   0    1    2    3
X (input):  h    e    l    l
Y (target): e    l    l    o
```

`X` is `"hell"`, `Y` is `"ello"` — `Y` is `X` slid over by one. At position 0 the
model sees `h` and must predict `e`; at position 1 it sees `h e` and must predict
`l`; and so on across all four positions at once. If a window starts at index `s`
in the token stream and has length `T`, then `X = tokens[s : s+T]` and
`Y = tokens[s+1 : s+1+T]`. That single index offset is the entire labeling scheme.

A common first wrong guess: *"do I predict the whole next window from the whole
current window?"* No — prediction is per position. Position $t$ predicts token
$t+1$ using only tokens $\le t$ (the causal mask from L18 enforces "only look
backward"), and the loss is averaged over all positions. One window of length `T`
gives `T` next-token predictions, not one.

## Batching sequences

The model does not train on one window at a time — it trains on a **batch** of
windows at once, because GPUs (and Apple's MPS) are far faster running many
sequences in parallel than looping one at a time. A batch is built by picking
`batch_size` random start positions in the corpus and slicing a length-`T` window
at each. Stacking those windows gives `X` of shape `(batch_size, T)` and `Y` the
same shape — `Y` each row shifted by one. (Shape `(batch_size, T)` reads as
"`batch_size` rows, each a sequence of `T` token ids.") The model processes all
rows together; the loss averages over every position in every row.

`make_get_batch(config, batch_size)` builds the function that produces these
batches. It returns a **closure** — a small function, `get_batch()`, that
remembers the corpus and the window length `T` and, each time it is called, draws
a fresh random batch:

1. **Tokenize the corpus once.** Map the fixed corpus text to token ids in
   `[0, vocab_size)` and keep them as a 1-D tensor (done once, outside the inner
   function — re-tokenizing every step would be wasteful).
2. **Pick random start positions.** Sample `batch_size` integers in
   `[0, len(tokens) - T - 1)` with `torch.randint`. The `- T - 1` upper bound
   leaves room to slice a full length-`T` window *and* its one-position-shifted
   target without running off the end.
3. **Slice the inputs.** For each start `s`, take `tokens[s : s+T]`; stack the
   `batch_size` windows into `X` of shape `(batch_size, T)`.
4. **Slice the shifted targets.** For each start `s`, take `tokens[s+1 : s+1+T]`;
   stack into `Y`, same shape. Return `(X, Y)`.

The frozen loop calls `get_batch()` once per step and feeds the `(X, Y)` it gets
straight into the model and the loss.

One detail matters for the resume guarantee later: step 2 draws from the
**global** torch random-number generator (`torch.randint` with no private
generator argument). "Global RNG" means the one shared random-number stream that
`torch.manual_seed` sets. The loop seeds it at entry and a checkpoint saves its
state, so the *exact sequence of batches* a run sees is part of what gets
restored on resume. A private generator's state would not be checkpointed, and a
resumed run would draw different batches and diverge.

## The training step, as numbered steps

Here is the one thing the model does, repeated `steps` times. The frozen
`train_loop` runs this; you are reading it so you know exactly what your wiring
feeds. One **step** is one parameter update from one batch:

1. **Get a batch.** Call `get_batch()` to draw `(X, Y)` — a batch of input
   windows and their shifted next-token targets.
2. **Zero the gradients.** Call `optimizer.zero_grad()`. Gradients accumulate by
   default in PyTorch, so last step's gradients must be cleared first or they'd
   add to this step's.
3. **Forward pass.** Run `logits = model(X)` — push the batch through the GPT to
   get a score for every next-token candidate at every position, shape
   `(batch_size, T, vocab_size)`.
4. **Cross-entropy against the shifted target.** Compute the next-token loss:
   flatten `logits` to `(batch_size * T, vocab_size)` and `Y` to
   `(batch_size * T,)`, then `CrossEntropyLoss` — the average surprise over every
   position, the single number to minimize. The flatten is just "treat every
   position in every row as one more classification example."
5. **Backward pass.** Call `loss.backward()`. Autograd walks the computation
   graph and fills each parameter's `.grad` with the partial derivative
   $\partial \text{loss} / \partial \theta$ — the slope of the loss with respect to that parameter.
   ("Forward/backward pass" is layered-network language: forward runs data
   through the layers, backward runs the chain rule back through them. We
   differentiate the *loss with respect to the parameters* — never "the model" —
   and this analytic gradient, computed by an exact formula, is what every step
   uses; the slow finite-difference checks from the autograd lessons are only for
   debugging.)
6. **Optimizer step.** Call `optimizer.step()`. AdamW reads every `.grad` and
   nudges each parameter a little in the direction that lowers the loss. This is
   the same gradient-descent engine from L1 and L10 — only the optimizer's update
   rule and the loss differ; the loop structure is loss-agnostic.

That is identical in spirit to the L14 `train_loop` and the L10 optimizer step.
The only model-specific parts are the model itself (the GPT) and the loss
(next-token cross-entropy); the loop around them never changes.

## AdamW and the weight-decay split

This is the part you build, and the idea the lesson turns on.

L10 trained with plain **Adam** — an optimizer that adapts the step size per
parameter using running averages of the gradient. Production GPT training uses
**AdamW**, whose one real difference is **decoupled weight decay**.

First, *weight decay*. It is L2 regularization — the same idea you met in L10 as
"add a penalty proportional to the squared size of the weights." Its job is to
keep weights from growing without bound, which is a form of regularization: large
weights let the model contort itself to memorize the training data, so gently
pulling every weight toward zero each step keeps the model smoother and
generalizing better. (This is exactly the motivation the M5 capstone write-up
asks you to articulate — *why* weight decay, not just *that* you used it: it is
regularization that keeps the weights small.) "Decoupled" is the `W` in AdamW: the
decay is applied directly to the weight, separate from Adam's gradient-based
update, rather than folded into the gradient. The distinction does not change what
*you* build, but it is why the optimizer is named AdamW.

The decision this lesson turns on is *which* parameters get decayed. The rule,
straight from GPT-2 and nanoGPT practice: **decay only the weight matrices.** Your
`make_adamw` walks `model.named_parameters()` — every `(name, parameter)` pair in
the model — and sorts each into one of two groups:

1. **The decay group** — every parameter that is a real projection weight: a 2-D
   `Linear` weight matrix. The test "is it 2-D and not an embedding" is
   `param.ndim >= 2 and "emb" not in name`, where `param.ndim` is the number of
   axes (1 for a vector, 2 for a matrix). These get `weight_decay` as passed in.
   Weight decay here is honest regularization keeping the matrices bounded.
2. **The no-decay group** — everything else: every 1-D parameter (all biases,
   every LayerNorm `weight` and `bias`) **and** the token / positional embedding
   tables (named `token_emb` / `pos_emb`). These get `weight_decay = 0.0`.

Why exclude biases, LayerNorm gains, and embeddings? Decaying a bias or a
LayerNorm scale pulls a single offset toward zero for no regularization benefit —
it just fights the model. Embeddings are a lookup table, not a linear map that
multiplies an input; shrinking their rows toward zero degrades the representation
of rare tokens for nothing. So `make_adamw` builds a list of two **param-group
dicts** — Python dicts of the form `{"params": [...], "weight_decay": ...}` — and
hands them to `torch.optim.AdamW`. The grader inspects `optimizer.param_groups`
directly: a weight matrix sitting in the zero-decay group, or a
bias/LayerNorm/embedding sitting in the decayed group, fails the check.

Note the `betas=(0.9, 0.95)` default. `betas` are Adam's two smoothing factors
for its running gradient averages; the GPT-2 recipe uses `0.95` for the
second-moment beta, not Adam's usual `0.999`. Leave the default as given.

## The training entrypoint: train

`train(config, steps, ...)` is the thin wrapper that ties everything together. It
does only four things, in order:

1. `torch.manual_seed(seed)` — seed the global RNG so a fixed-seed run is
   reproducible.
2. `model = GPT(config)` — build the model from L20.
3. `optimizer = make_adamw(model, lr=lr, weight_decay=weight_decay)` — build the
   optimizer with the decay split.
4. `get_batch = make_get_batch(config, batch_size)`, then `return train_loop(...)`
   — hand the model, the batch function, the optimizer, and the run controls
   (`steps`, `seed`, `metrics_path`, `checkpoint_dir`, `checkpoint_every`,
   `resume_from`) to the frozen loop and return its result.

The model is built *before* `train_loop` reseeds at entry, so a fixed-seed run is
reproducible. `train` itself owns no training logic — the loop owns the
cross-entropy, the metrics sink, and exact resume.

## Warmup and the cosine learning-rate schedule

A flat learning rate works, but the standard GPT recipe *shapes* the learning rate
over the run. (The learning rate, from L1, is the step-size multiplier: how big a
step the optimizer takes along the gradient.) `cosine_lr_with_warmup(step, ...)`
is a **pure function** — given a step number it returns the learning rate for that
step, holding no state — that computes a two-phase schedule:

1. **Linear warmup** for the first `warmup_steps` steps: ramp the rate from ~0 up
   to `max_lr`, as `max_lr * (step + 1) / warmup_steps`. Starting at full rate on
   a freshly-initialized model takes huge, destabilizing steps; warmup lets the
   optimizer's running averages settle before the steps get large.
2. **Cosine decay** for the rest: smoothly anneal from `max_lr` down to a small
   `min_lr` following half a cosine curve. With $p$ ("progress") running from 0
   at the end of warmup to 1 at `max_steps`, the rate is

$$\text{min\_lr} + \tfrac{1}{2}\,\bigl(1 + \cos(\pi\, p)\bigr)\,(\text{max\_lr} - \text{min\_lr}).$$

   Read aloud: *"start at `max_lr`, glide down a cosine to `min_lr`."* Late in
   training, small steps refine the weights rather than thrash them. Steps past
   `max_steps` clamp to `min_lr`.

The frozen `train_loop` has **no scheduler hook**, so this schedule lives in prose
and the experiments rather than threaded through the loop. If you owned the loop,
you would call `cosine_lr_with_warmup` each step and set `param_group["lr"]` on
every optimizer group before `optimizer.step()`. You still build and test the
function here so the idea is concrete.

## Gradient clipping

One more stabilizer from the standard recipe, also taught as prose because the
frozen loop has no hook for it: **global gradient-norm clipping.** After
`backward()` and before `optimizer.step()`, you measure the **global L2 norm** of
all gradients — `‖g‖`, read "the norm of g," the square root of the sum of every
gradient entry squared, a single number measuring overall gradient size — and if
it exceeds a cap (typically `1.0`), you rescale *all* gradients down so the norm
equals the cap. One call does it:
`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. It bounds the
occasional exploding-gradient spike — a single bad batch producing a giant
gradient — that would otherwise blow up a long run. It is not in the frozen loop
and is not graded; reach for it when you see loss spikes on your real run.

## Memory discipline: log floats, not Tensors

`loss_history` holds Python `float`s — the loop appends `loss.item()`, which pulls
the single number out of the loss tensor, never the live tensor itself. Appending
the raw `loss` tensor would keep a reference to its autograd graph (all the
intermediate activations needed for `backward`), pinning every step's memory until
the run runs out. The frozen loop already does `loss.item()`, so the
`loss_history` it hands back is already plain floats — the grader confirms
`loss_history[0]` is a `float`. This lesson's `record_loss(history, loss)` helper
is a one-line restatement of that same discipline (`history.append(loss.item())`,
never the live tensor) that you write yourself so the rule is in your hands, not
only the loop's.

## Checkpointing and exact resume

The loop checkpoints every `checkpoint_every` steps into `checkpoint_dir` using
the frozen `checkpoint_save` — a **checkpoint** is a saved snapshot of the run on
disk, here a "Schema-v1" dict with all nine keys: `schema_version`, `step`,
`model_state` (the weights), `opt_state` (the optimizer's running averages),
`loss_history`, `config`, `tokens_seen`, `rng_cpu`, and `rng_mps` (the saved
random-number-generator states for CPU and MPS). Pass `resume_from=<a checkpoint
path>` and the loop restores model, optimizer, loss history, step count, **and the
RNG state** — so a run that checkpoints at step N, reloads, and continues is
bit-for-bit identical to an uninterrupted run of the same seed. Restoring the RNG
is what makes the *batch stream* identical on resume; that is why `get_batch` had
to draw from the global RNG. The grader proves this match to `1e-6`.

## Watching the loss curve live

When you pass `metrics_path`, the loop appends one `{step, loss, timestamp}` JSONL
line per step (JSONL = one JSON object per line). The lesson page renders that
file as a **live loss curve** — train with `metrics_path` pointing at your metrics
cache (default `~/.llm-curriculum-cache/metrics/train-gpt.jsonl`) and watch your
run's loss fall in real time on this page while it trains.

## Running on Apple Silicon (MPS)

MPS is Apple's GPU backend for PyTorch (Metal Performance Shaders). The grader
runs the **SMOKE** config on CPU — tiny, deterministic, seconds-fast — so none of
this affects grading. The real `BABY_GPT` run is where these notes matter:

- **`compile=False` on MPS.** `torch.compile` (PyTorch's graph optimizer) is
  unreliable on the MPS backend; leave it off and run eager (un-compiled).
- **Set `PYTORCH_ENABLE_MPS_FALLBACK=1`** so any operation without an MPS kernel
  falls back to CPU instead of erroring out mid-run.
- **Keep all model and data tensors `float32`** on MPS — `float64` (double
  precision) is not supported there.
- **Gradient checking runs on CPU in `float64`** (as in the earlier autograd
  lessons); never trust a finite-difference check on MPS/`float32`.
- **The exact-resume test runs CPU + tiny SMOKE** — deterministic resume is
  verified on CPU, not on MPS.
- **MLX caveat:** Apple's MLX is a different array framework with its own
  autograd, sometimes faster on Apple Silicon — but it is *not* PyTorch, and none
  of this curriculum's frozen modules run on it. Stick with PyTorch + MPS here.

## Self-check: the real BABY_GPT run (NOT a grader step)

This is a self-check you run on your own machine, **not** part of the grader.
Train `BABY_GPT` (`n_layer=6, n_head=6, n_embd=384, block_size=256,
vocab_size=65`) on your character-level corpus with `metrics_path` set. On an
Apple-Silicon laptop expect roughly **10–15 minutes**. A healthy run reaches a
character-level validation loss around **~1.5** — coherent-looking character
n-grams, not yet good prose. Watch the live loss curve on this page as it trains;
if it spikes, that is where warmup and gradient clipping earn their keep.

## Exercise

Implement in `my_work/L23-train-gpt/train_gpt.py` (run `uv run grade start L23-train-gpt` to get
the scaffold). Two frozen canonicals are imported for you —
`from references.gpt import GPT, GPTConfig` and the frozen `train_loop` — and the
`SMOKE` / `BABY_GPT` configs are given. You write:

- `SelfAttentionHead(nn.Module)` — the **warm-up**, the L18 single causal head
  recalled cold. `__init__(self, n_embd, head_size, max_T=256)` makes bias-free
  `nn.Linear(n_embd, head_size)` query/key/value projections (`n_embd` is the
  input embedding width, `head_size` the output width per head, both ints) and
  registers a lower-triangular `tril` buffer of shape `(max_T, max_T)`.
  `forward(self, x)` takes `x` of shape `(B, T, n_embd)` (batch `B`, sequence
  length `T`, width `n_embd`) and returns `(B, T, head_size)`: project to q/k/v,
  compute `scores = q @ k.transpose(-2, -1) / sqrt(head_size)` (shape `(B, T, T)`),
  mask the strict-upper triangle to `-inf` *before* softmax, softmax over the last
  axis, then `weights @ v`. The `1/sqrt(head_size)` scale is not optional.
- `make_adamw(model, *, lr, weight_decay, betas=(0.9, 0.95))` — build an `AdamW`
  with the two-group decay split. `model` is the GPT; `lr` is the learning rate
  (float); `weight_decay` is the decay strength applied to the decay group (float);
  `betas` is the `(beta1, beta2)` pair (keep the default). Returns a
  `torch.optim.AdamW`. Put every `ndim >= 2` non-embedding parameter in the decay
  group with `weight_decay`, and every 1-D parameter and embedding in a
  `weight_decay=0.0` group.
- `cosine_lr_with_warmup(step, *, warmup_steps, max_steps, max_lr, min_lr)` — the
  pure schedule. `step` is the current step (int); `warmup_steps` how many warmup
  steps; `max_steps` the total horizon; `max_lr` / `min_lr` the rate ceiling and
  floor (floats). Returns the learning rate (float) for that step: linear ramp to
  `max_lr` over warmup, then cosine decay to `min_lr`, clamped to `min_lr` past
  `max_steps`.
- `record_loss(history, loss)` — append `loss.item()` (a Python float) to
  `history` (a list) and return `history`. Never append the live `loss` tensor.
- `make_get_batch(config, batch_size)` — return a `get_batch()` closure. `config`
  is the `GPTConfig` (its `block_size` is the window length `T`, `vocab_size` sizes
  the token range); `batch_size` is the number of windows per batch (int). Each
  `get_batch()` call returns `(X, Y)`, both shape `(batch_size, T)`, with `Y` equal
  to `X` shifted one token, drawing start positions from the **global** torch RNG.
- `train(config, steps, *, seed=0, lr=3e-4, weight_decay=0.1, metrics_path=None,
  checkpoint_dir=None, checkpoint_every=500, resume_from=None)` — wire it all up.
  `config` is the `GPTConfig`; `steps` the number of training steps (int); `seed`
  the RNG seed (int); `lr` / `weight_decay` are passed to `make_adamw`;
  `metrics_path` (path or `None`) is where to log per-step metrics;
  `checkpoint_dir` / `checkpoint_every` control checkpointing; `resume_from` (path
  or `None`) resumes from a saved checkpoint. Seed, build `GPT(config)`, build the
  optimizer via `make_adamw`, build `get_batch` via `make_get_batch`, and return
  `train_loop(...)` with all the run controls passed straight through.

Then grade it: the site's Run Grader button or `uv run grade L23-train-gpt`.

## How the grader checks you

All checks run on CPU, deterministic and seconds-fast, on the tiny SMOKE config
(`n_layer=2, n_head=2, n_embd=32, block_size=32, vocab_size=65`) — no dataset
reads.

- **`SelfAttentionHead`** is checked against scaled-dot-product causal attention
  recomputed by hand on a fixed `(2, 5, 8)` input using the head's own weights:
  same shape, max absolute difference under `1e-5`. This warm-up is an independent
  root — a wrong head does not block the training tests.
- **`train(SMOKE, steps=100)`** must log exactly 100 losses (one `loss.item()` per
  step) and end at least `0.1` below where it started — a mechanics sanity check
  that the loss is actually falling, not a quality bar.
- **Exact resume:** a run checkpointed at step 50 and resumed for 50 more must
  reproduce the uninterrupted 100-step run's step-100 loss to `1e-6`. This passes
  only if the RNG state is restored from the checkpoint (so the batch stream
  matches).
- **Checkpoint schema:** the saved `.pt` must carry all nine Schema-v1 keys with
  `schema_version == 1` — written by the frozen `checkpoint_save`, so don't
  hand-roll the save.
- **AdamW split:** `make_adamw` must put every weight matrix in a decayed group and
  every bias / LayerNorm param / embedding in a `weight_decay == 0.0` group.
- **Memory discipline:** `loss_history[0]` must be a Python `float`, not a tensor.

### The named trap: the embedding in the decay group

The grader hunts one specific `make_adamw` mistake by name. If you split the
parameters on `param.ndim >= 2` *alone* — "decay everything 2-D" — the embedding
tables (`token_emb` and `pos_emb`) are 2-D, so they land in the **decayed** group,
where they must not be. The AdamW-split check recomputes the correct grouping and
catches it, with the message *"bias/layernorm/embedding parameters must have
weight_decay=0.0"* — telling you to also exclude by name (`"emb" not in name`),
not just by `ndim`. The grader also names a second trap on the warm-up head:
dropping the `1/sqrt(head_size)` scale changes the softmax, so the head disagrees
with the hand-computed reference and the check reports it.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the formula with shapes, then pseudocode. Try
level 1 before climbing — the `make_adamw` ladder, for instance, goes from "AdamW
decays only the weight matrices" to "split on name + ndim, embeddings are 2-D but
go in the no-decay group" to the actual loop.

## Done when

`uv run grade L23-train-gpt` shows all-green: the warm-up `SelfAttentionHead` matches
hand-computed causal attention to `1e-5`; `train(SMOKE, 100)` logs 100 losses and
drives the loss down by at least `0.1`; the checkpoint-then-`resume_from` run
reproduces the uninterrupted step-100 loss to `1e-6`; the saved checkpoint carries
all nine Schema-v1 keys with `schema_version == 1`; `make_adamw` splits the decay
groups correctly; and `loss_history[0]` is a Python `float`. Then run the
experiments and record your predictions.
