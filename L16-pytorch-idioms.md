# L14 — PyTorch Idioms

Through M1–M3 you built everything by hand. Weights were bare NumPy arrays. The
forward pass — the calculation that turns an input into a prediction — was
spelled out operation by operation. Gradients were threaded through the chain
rule yourself. (A **gradient** is the vector of slopes that says, for each knob,
which way to nudge it to lower the loss; the **loss** is the single number that
scores how wrong the prediction is.) And the descent step — the move that nudges
the knobs downhill — you wrote as `x <- x - lr * grad`: take the current
parameters `x`, and subtract the learning rate `lr` times the gradient `grad`.

That hand work was the point. You now know what the framework does under the
hood. This lesson ports that knowledge to *idiomatic* PyTorch — the conventional,
expected way to write it. "Idiom" here means the standard PyTorch phrasing for a
job, the way a native speaker would say it: parameters live inside an
`nn.Module`, autograd supplies the backward pass, and an `optimizer` object hides
the update. By the end you will have rebuilt the exact same training process from
L8/L13 — start the parameters somewhere, repeatedly score and nudge them — but in
four lines of PyTorch instead of forty of NumPy.

The destination is one function, the **canonical training loop**. "Canonical"
means *the one official version everyone shares*: six later lessons (and all of
M5, the final capstone milestone) import this exact function as the single place
training happens. Get it right once here and you never rewrite it.

A note on what carries over from L13, the lesson right before this one. L13 was
the NumPy↔PyTorch bridge: it introduced the `nn.Module` base class, `nn.Linear`
layers, autograd's automatic backward pass, and the one-line `zero_grad`. This
lesson assumes none of that stuck — every idiom is defined from zero below. What
L13 gives you is the *trust* that PyTorch's gradients match the ones you computed
by hand; here we put those gradients to work inside a real loop.

## The warm-up: the descent engine, recalled cold

Before the optimizer hides the update inside a method call, write it one more
time by hand, so the thing being hidden is fresh in your fingers. This is the L1
gradient-descent engine, in pure NumPy.

The setup, in plain words first. You have a function `f` you want to *minimize* —
drive to its lowest value. You have a starting point `x0` — a vector of numbers,
your initial guess. You have `grad_f`, a function that, handed any point, returns
the gradient of `f` there: the direction of steepest *increase*. To go *down*,
step the opposite way. `lr` is the **learning rate**, the size of each step
(written `lr` for "learning rate"; the symbol $\eta$, read "eta", is the same
thing). `steps` is how many times to repeat.

The procedure, as explicit numbered steps:

1. **Copy the start.** Set `x` to a fresh copy of `x0` in float64 (double
   precision, so rounding error never muddies the trace). Copying matters because
   the caller may reuse `x0` afterward — mutating it would be a surprise side
   effect.
2. **Evaluate the gradient.** At the current `x`, call `g = grad_f(x)`. This `g`
   is a vector with one slope per parameter; it points uphill.
3. **Step downhill.** Update `x <- x - lr * g`. Read aloud: "the new x is the old
   x minus the learning rate times the gradient." Subtracting moves *against* the
   uphill direction — i.e. downhill.
4. **Repeat.** Do steps 2–3 a total of `steps` times.
5. **Return** the final `x`.

```python
def gradient_descent(f, grad_f, x0, lr, steps):
    x = np.asarray(x0, dtype=np.float64).copy()      # (d,) fresh copy, float64
    for _ in range(steps):
        g = np.asarray(grad_f(x), dtype=np.float64)  # (d,) slope at current x
        x = x - lr * g                               # (d,) step downhill
    return x                                          # (d,)
```

The shape comment `(d,)` reads "a 1-D array of length d" — `d` is the number of
parameters. Notice `f` itself is never called inside the loop: the *update* only
needs the gradient. `f` is in the signature so the function honestly states "I
minimize this f," and so a caller can log the value if they want.

One misconception to head off now, because it returns in every later lesson: this
loop **iterates** toward the minimum; it does not jump there. A gradient is a
slope at one point — it tells you which way is downhill *from here*, not where the
bottom is. You only learn the bottom's location by repeatedly stepping toward it.
(For a few special problems a closed-form shortcut to the exact minimum exists,
but it does not generalize, which is why every model in this course trains by
iterating.)

## Tensors: NumPy arrays that remember

A torch `Tensor` is the NumPy array you already know, plus two extra abilities. It
can live on a **device** — CPU, or a GPU-style accelerator (Apple's `MPS`,
NVIDIA's `CUDA`) — and it can **record the operations performed on it** so
autograd can replay them backward. ("Autograd" is PyTorch's automatic
differentiation engine: it watches the forward computation and, on request,
computes every gradient for you.)

Broadcasting — the rule for combining arrays of different shapes by stretching
size-1 dimensions — works identically to NumPy: trailing dimensions line up, a
dimension of size 1 stretches to match. So an expression like
`logits - logits.max(dim=1, keepdim=True).values` reads the same in torch as in
NumPy. (`logits` are the model's raw output scores, before they are turned into
probabilities; `dim=1` means "along the columns"; `keepdim=True` preserves the
reduced dimension as size 1 so it broadcasts back cleanly.)

The new discipline is the boundary between a Tensor and a plain number. A scalar
loss Tensor — a Tensor holding one value — is **not** a Python float. It still
carries the live computation graph that produced it, the recorded chain of
operations autograd needs for `backward()`. The moment you only want the *number*
(to print it, store it, or compare it), pull it out with `loss.item()`, which
returns a plain Python `float` and drops the graph. Keep the Tensor itself only as
long as you still need to call `loss.backward()`. Forgetting this is the named
trap of this lesson — more on it below.

## nn.Module: a container that tracks your parameters

An **`nn.Module`** is PyTorch's base class for any model. You subclass it (write
`class YourModel(nn.Module)`), and in exchange it tracks your **parameters** for
you. A *parameter* is a tensor the module owns and will train — every weight and
bias. You never list them by hand; the module discovers them.

Here is the L9 MLP — a multi-layer perceptron, the plain fully-connected
network where you carried `W1/b1, W2/b2, ...` by hand — rebuilt as an
`nn.Module`. Each hand-rolled weight-matrix-plus-bias becomes one **`nn.Linear`**
layer. `nn.Linear(a, b)` is a ready-made layer that holds a weight matrix and a
bias vector and computes $xW^\top + b$ — exactly the matmul-plus-bias you wrote by
hand — mapping an input of width `a` to an output of width `b`.

```python
class MnistMLP(nn.Module):
    def __init__(self, in_dim=64, hidden=32, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))   # (N, hidden)
        x = torch.relu(self.fc2(x))   # (N, hidden)
        return self.fc3(x)            # (N, n_classes)
```

Reading it line by line. `__init__` is the constructor — it builds the layers
once. `super().__init__()` runs `nn.Module`'s own constructor *first*; skip it and
the tracking machinery is never set up, so the module finds zero parameters.
`fc1, fc2, fc3` ("fully-connected" layers) are the three `nn.Linear`s. `forward`
is the method that runs the network: given input `x`, it returns the output. It
reads top to bottom exactly like the math — `relu(fc1(x))`, then `relu(fc2(x))`,
then `fc3`. `torch.relu` is the rectified-linear activation, $\max(0, \cdot)$ — it
zeroes out negatives and leaves positives unchanged, which is what lets a stack of
linear layers learn non-straight-line patterns.

The shape comments trace one batch. Read `(N, hidden)` as "N rows by `hidden`
columns" — `N` is the number of examples in the batch, one row each. Input
`x` is `(N, in_dim)`; `fc1` maps it to `(N, hidden)`; `fc2` keeps it
`(N, hidden)`; `fc3` produces `(N, n_classes)` — one row of class scores per
example.

Three things happen *for free*, and they are the whole reason to use `nn.Module`:

1. **Registration.** Assigning an `nn.Linear` to `self.fc1` **registers** it — the
   module records its weight and bias as parameters. So `model.parameters()`
   returns all six tensors (three weights, three biases) and `model.state_dict()`
   can serialize them to disk. This only works because you assigned the layer to
   `self`; a layer stored in a plain local variable is invisible to the module.
2. **Readable forward.** `forward` is just the math, top to bottom. No bookkeeping.
3. **Automatic backward.** Because every operation in `forward` was recorded,
   calling `loss.backward()` later fills each parameter's `.grad` field with no
   hand-written backward pass. This is the chain rule, run for you.

Why `in_dim=64`? Real MNIST digit images are 28×28 = 784 pixels. We use a
synthetic 64-feature input instead so the grader needs no MNIST download and runs
in seconds — the *idioms* are identical at any width.

A point worth pinning down: when you eventually call `loss.backward()`, you are
differentiating the **loss** with respect to the **parameters** — not "the model
with respect to anything." The thing being scored is the loss; the knobs being
adjusted are the parameters; the gradient connects the two.

## The optimizer and the canonical update

An **`optimizer`** is an object that owns a list of parameters and knows how to
update them from their gradients. You pick the update rule by picking the
optimizer class:

- `torch.optim.SGD(params, lr=...)` — Stochastic Gradient Descent. Its update is
  exactly the warm-up's `x <- x - lr * grad`, applied to every parameter at once.
- `torch.optim.Adam(params, lr=...)` — a fancier rule that adapts the step size
  per parameter from the recent gradient history. Same job, smarter steps; M5
  covers why.

You hand it `model.parameters()` so it knows which tensors to update. The
**canonical training loop** — the four-line core every training run shares — is:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
optimizer.zero_grad()   # 1. clear .grad — torch ACCUMULATES into it otherwise
loss.backward()         # 2. fill every parameter's .grad (chain rule)
optimizer.step()        # 3. x <- x - lr * grad, for every parameter at once
```

These three calls are the entire update, and the **order is load-bearing**. As
explicit steps, with the reason each exists:

1. **`optimizer.zero_grad()` — clear the gradients.** PyTorch *adds* new gradients
   into each parameter's `.grad` field rather than overwriting it. If you skip
   this line, step 2's fresh gradient lands on top of last step's leftover, so the
   gradients pile up and the step is wrong. Clearing first guarantees `.grad`
   holds *only* this step's gradient.
2. **`loss.backward()` — compute the gradients.** This walks the recorded
   computation graph backward (the chain rule) and writes each parameter's
   gradient into its `.grad`. Nothing has moved yet — this only fills `.grad`.
3. **`optimizer.step()` — apply the update.** Now that every `.grad` holds the
   correct gradient, the optimizer nudges each parameter downhill:
   `x <- x - lr * grad` for SGD. This is the warm-up's update, run for all
   parameters simultaneously.

The full loop wraps these three in a `for` over your batches, plus a forward pass
and a loss in between (we assemble it as one function below).

Two clarifications a beginner often needs here. First, the loss and the optimizer
are *separate* concerns: the loss is task-specific (cross-entropy for
classification, MSE for regression), but the optimizer is loss-agnostic — SGD does
not know or care which loss produced the gradients, it just steps downhill on
whatever `.grad` holds. Swap the loss and the same optimizer keeps working.
Second, in a multi-layer network "forward pass" and "backward pass" name real
directions through the graph — forward computes the output, backward propagates
gradients in reverse. (In the one-step warm-up there is no graph and so no
"direction"; the words only earn their meaning once layers stack up, as they do
here.)

## Porting the L9 MLP, mechanically

Once the anatomy is clear, porting a hand-rolled model is a checklist. Run it on
any M1–M3 model:

1. Each hand-rolled weight matrix (with its bias) → one `nn.Linear(in, out)`,
   assigned to `self`.
2. Each manual matmul-plus-bias in the forward pass → a layer call `self.fcK(x)`.
3. The entire hand-written backward pass → *deleted*; `loss.backward()` replaces it.
4. The hand-written descent loop → `zero_grad()` / `backward()` / `step()`.

What you keep from the by-hand era is the **understanding**. You can still say
exactly what each line computes, which is why, when autograd does something
surprising, you can debug it instead of treating the framework as a black box.

## The canonical training loop, as one function

Everything converges on one function, `train_loop`. This is the frozen interface —
the exact signature six later lessons and all of M5 import. "Frozen" means the
name, the arguments, and the return shape do not change; downstream code depends
on them.

```python
def train_loop(model, get_batch, optimizer, steps, *, seed=0,
               metrics_path=None, checkpoint_dir=None, checkpoint_every=500,
               resume_from=None) -> dict:
    # returns {"loss_history": list[float], "steps_run": int}
```

The `*` in the signature marks everything after it as **keyword-only** — callers
must name those arguments (`seed=7`, not a bare `7`), which keeps a long call
readable and prevents positional mix-ups.

Four contracts make this loop trustworthy. Each is a property the grader checks.

**1. Determinism under a fixed seed.** A **seed** is the starting number for a
random-number generator; fix it and the "random" sequence is identical every run.
At entry, `train_loop` seeds *both* generators it might consult —
`torch.manual_seed(seed)` (torch's own RNG) and `random.seed(seed)` (Python's
`random` module) — so the same seed produces a byte-identical `loss_history` on
CPU. (Seed before *constructing* the model too, so its randomly-initialized
weights are reproducible; the loop only governs the randomness *inside* the loop.)

**2. Log `loss.item()`, never the live Tensor.** `loss_history` must hold Python
floats. As covered above, a loss Tensor carries its whole autograd graph; append
the live `loss` and every step's graph is kept alive forever, so memory grows
without bound until the process stalls. This is a real, recurring bug — append
`loss.item()`, the plain number.

**3. Metrics as JSONL.** When `metrics_path` is set, append one JSON line —
`{"step", "loss", "timestamp"}` — per step. JSONL ("JSON Lines") is one JSON
object per line, an append-only log you can `tail` while training is still
running.

**4. Checkpoint and exact resume.** A **checkpoint** is a saved snapshot of
training, written so an interrupted run can pick up exactly where it stopped.
Every `checkpoint_every` steps the loop writes a **Schema-v1** checkpoint — a dict
with a fixed set of keys (a "schema" is the agreed-on shape of the saved data;
"v1" is version 1, frozen so M5 can read it). The keys are: `schema_version`
(== 1), `step`, `model_state` (the weights), `opt_state` (the optimizer's
internal state), `loss_history`, `config`, `tokens_seen`, `rng_cpu`, and
`rng_mps`.

The two RNG keys are what make resume *exact* whenever a step's randomness flows
through the global generator. `rng_cpu` is `torch.get_rng_state()` — a snapshot of
torch's random-number-generator state at save time; `rng_mps` is the same for the
MPS accelerator (or `None` when no MPS device exists). If `get_batch` draws a
*fresh* random minibatch each step (the realistic case), the loss after step N
depends on the RNG *state* at step N, so restoring that state on resume is what
makes the remaining steps replay identically: reload a checkpoint, run the
remaining steps, and `loss_history` matches an uninterrupted run to float32
precision. Skip the RNG restore there and the post-resume losses drift, because
each batch is now drawn from a different random state. (If `get_batch` returns the
same fixed batch every step — as the grader's does, to stay fast — restoring model
and optimizer state alone already reproduces the losses; the RNG keys are in the
schema for the general, fresh-batch case M5 runs, so they are frozen regardless.)

Numbered, the procedure inside the loop is:

1. **Seed** both RNGs from `seed`.
2. **If resuming**, load the checkpoint: restore model weights, optimizer state,
   `loss_history`, the step counter, `tokens_seen`, and the RNG state.
3. **For each step**: call `get_batch()` to get one minibatch `(X, Y)`; run the
   three-line update — `zero_grad()`, forward to `logits`, cross-entropy `loss`,
   `backward()`, `step()`; append `loss.item()` to `loss_history`; write the JSONL
   metrics line if a path was given; save a checkpoint if this step is a multiple
   of `checkpoint_every`.
4. **Return** `{"loss_history": ..., "steps_run": steps}`.

## Exercise

Fill in `my_work/pytorch-idioms/pytorch_idioms.py`. Run
`uv run grade start L16-pytorch-idioms` to get the scaffold. The module is fully
self-contained — `torch`, `numpy`, `random`, `json`, `time` only, no cross-lesson
or `my_work` imports — because the lesson is about *writing* these idioms, not
importing them.

You will implement five pieces:

- `gradient_descent(f, grad_f, x0, lr, steps)` — the NumPy warm-up, the L1 engine
  recalled cold. Parameters: `f` is the function to minimize (it takes a parameter
  vector and returns one number — `f` is short for "function"; it is *not* used by
  the update, only carried in the signature); `grad_f` is its gradient function
  (hand it a point, it returns the gradient there, a vector with one slope per
  parameter); `x0` is the starting parameter vector ("x-zero", where the walk
  begins); `lr` is the learning rate (the step size); `steps` is how many updates
  to take. Copy `x0` to float64 first, apply `x <- x - lr * grad_f(x)` `steps`
  times, and **return** the final vector, shape `(d,)` (a 1-D array of length `d`,
  the number of parameters). Do not mutate the caller's `x0`.

- `MnistMLP(nn.Module)` — the L9 MLP as a module. Its `__init__(self, in_dim=64,
  hidden=32, n_classes=10)` must call `super().__init__()` then define three
  `nn.Linear` layers — `self.fc1` (`in_dim` → `hidden`), `self.fc2` (`hidden` →
  `hidden`), `self.fc3` (`hidden` → `n_classes`). Its `forward(self, x)` takes the
  input batch `x` (shape `(N, in_dim)`, `N` examples) and returns logits of shape
  `(N, n_classes)` via `relu(fc1)` → `relu(fc2)` → `fc3`.

- `checkpoint_save(path, *, step, model, optimizer, loss_history, config,
  tokens_seen)` — write a Schema-v1 checkpoint dict to `path` with `torch.save`.
  Parameters: `path` is where to write the `.pt` file; `step` is the current step
  count (stored as an int); `model` and `optimizer` supply `.state_dict()`s;
  `loss_history` is the list of losses (store as plain floats, never Tensors);
  `config` is a dict of run settings; `tokens_seen` is the running count of
  examples seen (an int). Also capture `torch.get_rng_state()` as `rng_cpu`, and
  `torch.mps.get_rng_state()` as `rng_mps` (or `None` when no MPS device exists).
  All nine schema keys are required.

- `checkpoint_load(path)` — load a Schema-v1 checkpoint dict written by
  `checkpoint_save`. Parameter: `path` is the `.pt` file to read. Returns the dict
  (use `torch.load(path, weights_only=False)`).

- `train_loop(model, get_batch, optimizer, steps, *, seed=0, metrics_path=None,
  checkpoint_dir=None, checkpoint_every=500, resume_from=None)` — the frozen loop.
  Parameters: `model` is an `nn.Module` to train; `get_batch` is a zero-argument
  callable returning one minibatch `(X, Y)` of tensors; `optimizer` is its
  optimizer; `steps` is how many updates this call should run; `seed` (keyword-only)
  seeds both RNGs at entry; `metrics_path`, if set, is an append-only JSONL sink
  (one `{"step","loss","timestamp"}` line per step, `None` = no-op);
  `checkpoint_dir`, if set, is where to save a checkpoint every `checkpoint_every`
  steps; `resume_from`, if set, is a checkpoint path whose model/optimizer/RNG/step
  state is restored before continuing. **Returns** the dict
  `{"loss_history": list[float], "steps_run": int}`. Each step does
  `get_batch` → `zero_grad` → forward → cross-entropy loss → `backward` → `step`,
  then appends `loss.item()` (a float — **never** the Tensor) to `loss_history`.

Then grade it: the site's Run Grader button or `uv run grade L16-pytorch-idioms`.

## How the grader checks you

The grader runs on a tiny synthetic classification batch (16 examples, 64
features, 10 classes), all on CPU and deterministic, so it finishes in seconds —
no real MNIST.

- **The `gradient_descent` warm-up** is an independent root check. It descends the
  ill-conditioned quadratic $f(x, y) = x^2 + 100y^2$ from $(3, 3)$ with `lr=0.005`
  for 200 steps, and compares your result against the same recurrence run
  independently, to a relative tolerance of `1e-4`. A wrong warm-up does not block
  the rest.

- **The interface contract** runs 50 steps and checks three things: `steps_run`
  equals 50, `loss_history` has exactly 50 entries (one loss per step), every
  entry is finite, and the loss *decreased* (final < first). A loss that fails to
  drop on a fixed batch points straight at a broken `zero_grad`/`backward`/`step`
  ordering. Most other checks ride on this one passing first.

- **Determinism.** It builds the model under the same seed, then runs 30 steps
  twice, and requires the two `loss_history` lists to be *exactly* equal —
  `rtol=0, atol=0`, no tolerance at all. With the grader's fixed batch, this
  exactness rests on the reproducible weight initialization (same seed → same init)
  plus the deterministic CPU math; the entry-time seeding is what guarantees it
  once you move to a fresh-batch loop.

- **Exact resume.** It runs 60 steps uninterrupted, then runs 50 steps saving a
  checkpoint at step 50, then resumes from that checkpoint twice: resume + 0 steps
  must reproduce the first 50 losses, and resume + 10 steps must match the full
  60-step run, both to float32 tolerance (`rtol=1e-4, atol=1e-5`). With the
  grader's fixed batch this is carried by restoring model and optimizer state; the
  RNG keys you save are what extend the same guarantee to the fresh-batch runs M5
  does, where each step's minibatch draw consumes the generator.

- **Checkpoint schema.** It loads the step-50 checkpoint and confirms all nine
  Schema-v1 keys are present and `schema_version` is the int `1` — the contract M5
  consumes.

### The named trap: logging the live Tensor

This grader hunts one specific mistake by name. If your `train_loop` appends the
raw `loss` Tensor to `loss_history` instead of `loss.item()`, the grader's check
sees that `loss_history[0]` is a Tensor, not a Python `float`, and fails with a
message naming `loss.item()` as the fix. The buggy version is identical to a
correct solution in every other respect — it just stores `loss` where it should
store `loss.item()`. The reason this is worth a named trap: in a real run that one
character leaks memory, because each retained Tensor pins its whole autograd graph
and the graphs pile up step after step until training stalls. The grader catches
it on step 0 so you fix it here, cheaply, instead of debugging a slow leak later.
This check runs first and independently, so a wrong-type `loss_history` fires its
own clear message rather than crashing a later float-consuming check.

## The hint ladder

If the grader's message is not enough, open the hints. Each piece has three
levels, mildest first: a conceptual nudge (the symptom and what it points to),
then the formula or invariant with shapes (e.g. that a scalar loss Tensor holds
the whole autograd graph, so you must `.item()` it before storing), then
pseudocode you can read almost line for line. Try level 1 before climbing — the
goal is for the hint to jog the idea, not hand you the answer.

## Done when

`uv run grade L16-pytorch-idioms` shows all-green: the `gradient_descent` warm-up matches the
independent recurrence on $x^2 + 100y^2$, `train_loop` reports `steps_run=50` with
50 finite, decreasing losses, two same-seed runs are *exactly* equal, a resumed
run matches an uninterrupted one to float32 (model, optimizer, and — for the
fresh-batch runs M5 does — RNG state all riding along in the checkpoint), and the
saved checkpoint carries all nine Schema-v1 keys with `schema_version == 1`. Then
run the experiments and record your predictions.
