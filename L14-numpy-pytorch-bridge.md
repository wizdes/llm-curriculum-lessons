# L13 — The NumPy<->PyTorch Bridge

For the last twelve lessons you built everything by hand in **NumPy** — the
Python library for arrays of numbers. In L8 you wrote a tiny **autograd** engine
(short for "automatic differentiation": code that computes derivatives for you).
In L12 you used the same idea to build a small **CNN** — a convolutional neural
network, the kind of model that slides small filters over an image — and you
wrote both halves of it by hand: the **forward pass** (run the input through the
layers to get an answer) and the **backward pass** (work the chain rule of
calculus backward through those same layers to get the gradient of the loss with
respect to every weight).

This lesson is the moment you stop hand-writing the backward pass. From here on
the course uses **PyTorch**, the library nearly all modern deep-learning research
runs on. PyTorch builds the *same* CNN you built in L12, and it hands you the
backward pass for free: its own autograd engine records the forward operations
and replays them in reverse — exactly the trick your L8 engine used, just
industrial-strength.

But this lesson is not "PyTorch is easier, so switch." It is easier, and you will
switch. The real job here is to **prove the two libraries compute the same
thing** — gradient for gradient, to nearly the last decimal place — and to nail
the three facts you must get right whenever you carry weights or gradients across
the boundary between them. If you can show your hand-built NumPy CNN and
PyTorch's CNN produce identical gradients, you have proven you understand what
PyTorch is doing under the hood. After that, trusting it is earned, not blind.

The network is the same tiny configuration from L12: 3 input channels
(`in=3`, e.g. the red/green/blue planes of a color image), two convolutional
layers producing 2 then 3 channels (`conv_channels=[2, 3]`), `n_classes=4`
output categories, and an `8x8` input image. We keep it small on purpose: small
enough that a full forward+backward runs in a blink, so the gradient check is
instant.

```text
conv(3x3, pad 1) -> ReLU -> maxpool(2,2)
conv(3x3, pad 1) -> ReLU -> maxpool(2,2)
flatten -> Linear -> logits
```

Read that pipeline left to right: a `3x3` convolution (a filter that looks at
3-by-3 patches) with `pad 1` (one ring of zeros added around the border so the
output keeps the same height and width); then **ReLU** (the activation
`max(0, x)`, which zeros out negatives); then **maxpool(2,2)** (take the largest
value in each non-overlapping 2-by-2 block, halving height and width). That block
runs twice. Then **flatten** (reshape the small grid of numbers into one long
vector), a **Linear** layer (multiply by a weight matrix and add a bias), and out
come the **logits** — the raw, pre-softmax scores, one per class.

## What a tensor is, and what carries over from your autograd engine

A **tensor** is PyTorch's array. For everyday purposes it *is* a NumPy array —
the same grid of numbers with a shape like `(8, 3, 8, 8)` — with two extra powers
bolted on:

1. **It can track its own gradient.** Set `tensor.requires_grad = True` (read it
   as "requires gradient: yes, please remember how to differentiate through me")
   and PyTorch starts recording every operation you do to that tensor into a
   graph. Later, calling `.backward()` on a final scalar (the loss) walks that
   graph in reverse and fills in `tensor.grad` — the gradient of the loss with
   respect to that tensor. This is your L8 autograd engine, productionized. In L8
   you built the graph nodes and the reverse walk yourself; PyTorch does both for
   you. The mental model is unchanged.
2. **It can live on a different device.** A **device** is just where the numbers
   physically sit and compute: `"cpu"` (the main processor) or a GPU. On Apple
   Silicon the GPU backend is called **MPS** (Metal Performance Shaders). A NumPy
   array only ever lives on the CPU; a tensor can be moved to the GPU with
   `tensor.to("mps")` to run the math far faster on large data.

A tensor also has a **dtype** — its numeric type, i.e. how many bits each number
uses. The two that matter here are **float32** (single precision, about 7
decimal digits of accuracy) and **float64** (double precision, about 16 digits).
NumPy defaults to float64; PyTorch defaults to float32, because deep learning
trades a little precision for a lot of speed and memory. You will see in a moment
why that default forces a careful choice in this lesson.

Here is the same operation in both libraries, side by side, so the parallel is
concrete:

```python
# NumPy: a plain array, no gradient tracking.
import numpy as np
a = np.array([2.0, 3.0])
b = (a ** 2).sum()          # b = 13.0, and that is the end of it.

# PyTorch: a tensor that records its history, then differentiates.
import torch
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = (a ** 2).sum()          # b = tensor(13., grad_fn=...) — it remembers
b.backward()                # walk the graph backward
print(a.grad)               # tensor([4., 6.])  == d b / d a == 2 * a
```

`a.grad` is `[4., 6.]` because the derivative of `a**2` summed is `2*a`, and
that is exactly what your L8 engine computed by hand. The `grad_fn=...` PyTorch
prints on `b` is the recorded graph node — the breadcrumb autograd left so it
knows how to reverse the squaring. That recorded graph is the whole point of
**Misconception #6** from earlier in the course: the autograd graph is real, it
is built during the forward pass, and `.backward()` is the reverse walk over it.
PyTorch is not doing anything magical or different from what you built — it is
**automating exactly the L8 engine**.

One thing to keep straight: `.backward()` differentiates the **loss with respect
to the parameters**, never "the model with respect to anything." There is no such
thing as the gradient of a whole network; there is the gradient of a single
scalar (the loss) with respect to each weight. That scalar is what you call
`.backward()` on.

## Crossing the bridge: converting between NumPy and tensors

To compare the two libraries you have to move numbers across the boundary. There
are two conversions, and one sharp edge.

- **NumPy array -> tensor:** `torch.from_numpy(arr)`, or the more general
  `torch.as_tensor(arr, dtype=...)` which also lets you pick the dtype.
- **tensor -> NumPy array:** `tensor.numpy()`. (If the tensor has gradient
  tracking on, call `tensor.detach().numpy()` first — `.detach()` returns a new tensor cut
  loose from the autograd graph (it shares the original's memory — not a copy),
  since a NumPy array cannot carry gradient tracking.)

The sharp edge is **shared memory**. `torch.from_numpy` and `.numpy()` do not, by
default, copy the numbers — they wrap the *same* block of memory in a new
object. So the NumPy array and the tensor are two windows onto one buffer, and an
**in-place** change through either window (an operation that overwrites the
existing memory rather than allocating new memory, like `arr += 1` or
`tensor.add_(1)` — the trailing underscore in torch marks in-place) mutates what
the *other* sees:

```python
import numpy as np, torch
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)   # shares arr's memory — NOT a copy
arr += 100                   # in-place change to the numpy array
print(t)                     # tensor([101., 102., 103.])  <- t changed too!
```

This bites when you "convert, then keep editing the original" and are surprised
the converted copy moved with it. If you want an independent copy, make one
explicitly: `torch.as_tensor(arr).clone()` or `arr.copy()`. In this lesson the
weight-copying uses `tensor.copy_(...)` into pre-allocated parameter tensors,
which sidesteps the issue, but the gotcha is worth carrying forward — it is the
kind of bug that produces "impossible" results with no error message.

### Converting a weight across the bridge, step by step

When you copy one parameter from the NumPy model into the PyTorch model, the
procedure is:

1. Read the NumPy array out of the source model (e.g. `p["W1"]`).
2. Wrap it as a tensor in the *target's* dtype: `torch.as_tensor(p["W1"], dtype=dtype)`.
3. Reshape/transpose it into the layout PyTorch expects (for conv weights and
   biases the layout already matches — see below — so no transpose; for the
   linear weight, transpose it).
4. Copy it into the existing parameter in place under `torch.no_grad()`:
   `torch_model.conv1.weight.copy_(...)`. (`torch.no_grad()` tells autograd
   "don't record this — it is setup, not part of the forward pass we want to
   differentiate.")

Step 3 is where the one real subtlety lives. That is the next section.

## The weight-layout transpose

The whole bridge stands or falls on one detail: **the linear weight is stored
transposed between the two libraries.** ("Transposed" = with its two axes
swapped, so an `(m, n)` matrix becomes `(n, m)`; in code, `W.T`.)

Conv weights are *not* transposed. They share the exact layout
`(C_out, C_in, kH, kW)` — that is, (number of output channels, number of input
channels, kernel height, kernel width) — in both NumPy and torch, so they copy
straight across. Biases are 1-D `(C_out,)` in both. Only the linear layer
differs, and here is why:

- Your NumPy model stores the linear weight `Wfc` with shape `(D_in, D_out)` —
  (number of inputs, number of outputs) — so that the forward step is
  `flat @ Wfc`, a `(N, D_in)` batch times a `(D_in, D_out)` weight giving
  `(N, D_out)` logits. (`@` is matrix multiply; `N` is the batch size, the number
  of images processed at once.)
- PyTorch's `nn.Linear` stores its `.weight` with shape `(D_out, D_in)` — the
  axes flipped — because its forward step is `x @ weight.T`. PyTorch transposes
  the weight internally, so it stores the already-flipped version.

Both produce the same logits; they just keep the matrix in opposite orientations.
So copying the linear weight across requires a transpose: `torch.fc.weight = Wfc.T`
(read: "set torch's linear weight to NumPy's `Wfc` with its two axes swapped").
And reading the linear *gradient* back the other way needs the same transpose in
reverse: `numpy_dWfc = torch.fc.weight.grad.T`. (`.grad` is the gradient PyTorch
filled in; `.T` flips it back to NumPy's `(D_in, D_out)` layout so the two are
comparable.)

Get this one wrong and the failure is sneaky: every *other* parameter still
matches perfectly, while the linear weight's gradient is completely off. The
equivalence test below names that exact parameter and reminds you about the
transpose, because this is the single most common mistake when porting a layer
between frameworks — and the whole lesson is built to make you feel it.

## Two different experiments: float64 on CPU for correctness, float32 on MPS for speed

Recall the dtype fact from earlier: float32 carries about 7 decimal digits,
float64 about 16. That gap forces a clean split, and conflating the two sides is
the trap people fall into.

A gradient agreement check compares two independent computations of the same
quantity and asks "are these the same number?" At float32, the small rounding
errors that build up across a multi-stage network can reach the `1e-4` level
(`1e-4` reads as "ten to the minus four", i.e. `0.0001`) — large enough that a
*correct* implementation can disagree with PyTorch by that much purely from
noise. You would not be able to tell a real bug from rounding. So the
**correctness** comparison runs in **float64**, where two correct computations
agree to about `1e-13` (a ten-thousand-billionth) and any real bug stands out by
many orders of magnitude.

Now the catch that forces the split. Apple's **MPS** backend — the GPU on Apple
Silicon — **does not support float64**. It is a float32-only device. That is
fine, because correctness and speed are two genuinely different questions and
must never be measured in the same run:

- **Correctness** -> CPU, float64. This is what `compare_grads` and the grader
  do. The function asserts float64 and CPU at the top and refuses anything else,
  with an error message that explains why.
- **Speed** -> MPS, float32. A separate script, `experiments/mps_timing.py`,
  times one forward+backward on CPU versus MPS in float32. It never touches the
  gradient check.

So if you ever catch yourself thinking "let me just run the gradient check on the
GPU to make it faster," stop: MPS physically cannot hold the float64 that check
needs to mean anything. Speed lives in its own experiment, and the experiments
file walks you through both.

## The equivalence test

`compare_grads(numpy_model, torch_model, X, y)` is the headline function — the one
that proves the two libraries agree. Here `X` is the batch of input images
(shape `(N, 3, 8, 8)`) and `y` is the array of correct class labels (one integer
in `0..3` per image). It runs this procedure:

1. Load **identical** weights into both models (conv weights and biases copied
   directly, the linear weight transposed).
2. Run one forward + backward on the **same batch**, on CPU in float64.
3. For each parameter, compute the **max relative error** between your NumPy
   gradient (from your hand-written `backward`) and torch's gradient (from
   autograd), with the linear gradient transposed back to NumPy's layout before
   comparing.

The "relative error" of two numbers is how far apart they are, scaled by how big
they are — so a disagreement of `0.001` counts as huge between two numbers near
`0.001` but tiny between two numbers near `1000`. The per-parameter score is the
worst such error across all that parameter's entries:

$$\text{rel\_err}(a, b) = \max_i \frac{|a_i - b_i|}{\max(|a_i| + |b_i|,\ \varepsilon)}$$

Read aloud: "for each entry $i$, divide its absolute difference $|a_i - b_i|$ by
the larger of that entry's summed magnitude $|a_i| + |b_i|$ and a tiny floor `ε`,
then take the **largest** of those per-entry ratios." Scoring each entry's gap
*relative to its own size* (rather than one big difference over one big total) is
what makes a `0.001` disagreement count as huge near `0.001` but negligible near
`1000`. That floor `ε` (`eps`, here `1e-12`) only exists to avoid dividing by zero
when both numbers are essentially zero; it never affects a real comparison.

Every parameter should come back under `1e-4` — in practice around `1e-13`,
because two correct float64 computations of the same function agree to nearly the
limit of the number format. A parameter that lands *above* the `1e-4` threshold
is a real disagreement: your NumPy backward and torch's autograd are computing
different gradients for that layer, which means the chain rule (or, far more
likely, the weight transpose) broke there.

## Why a torch training step must clear gradients first

One last bridge fact, and it bites everyone exactly once. In your NumPy code,
`backward` *returns* a fresh gradient dictionary every time — last step's
gradients are gone. PyTorch does the opposite: `loss.backward()` **adds** its
result into each parameter's `.grad`. It accumulates.

That accumulation is a deliberate feature (it lets you sum gradients across
several micro-batches before taking one step), but it means a normal training
step has to **clear** the gradients first, or last step's leftovers pollute this
one. The order is the whole lesson here:

```python
def torch_train_step(model, optimizer, X, y):
    optimizer.zero_grad()          # 1. clear every .grad to zero FIRST
    loss = F.cross_entropy(model(X), y)  # 2. forward: logits -> loss
    loss.backward()                # 3. backward: write fresh gradients into .grad
    optimizer.step()               # 4. update each weight from its .grad
    return float(loss.item())
```

As numbered steps, one correct training step is:

1. `optimizer.zero_grad()` — reset every parameter's `.grad` to zero, so this
   step starts from a clean slate.
2. Forward pass — run the batch through the model and compute the loss
   (`F.cross_entropy` is PyTorch's softmax-cross-entropy, the same loss your
   NumPy `softmax_cross_entropy` computes).
3. `loss.backward()` — fill each parameter's `.grad` with this step's gradient.
4. `optimizer.step()` — nudge each weight downhill using its `.grad`.

(`optimizer` here is the object that owns the update rule — e.g.
`torch.optim.SGD`, plain gradient descent — and holds references to the model's
parameters.)

Omit `zero_grad()` and the second step's `.grad` is step-1's gradient *plus*
step-2's. On identical data that is roughly **2x** the correct gradient, and the
error compounds every step after. The grader catches this exact bug by running
your step twice on identical data and checking that the per-step gradient norm
stays stable; a ~2x blow-up means the gradients are piling up and `zero_grad` is
missing. The fix is one line — but it is the line everyone forgets first, which
is why it is this lesson's named trap.

## Exercise

Fill in `my_work/L14-numpy-pytorch-bridge/numpy_pytorch_bridge.py`. Run
`uv run grade start L14-numpy-pytorch-bridge` to get the scaffold. The module is fully
self-contained: you inline your own small NumPy CNN *and* the matching torch
`nn.Module` — no cross-lesson imports, just `numpy` and `torch`.

You will implement six pieces:

- `NumpyCNN` — the L12 architecture by hand in NumPy. Needs `forward(X)`
  (`X` is the input batch, shape `(N, 3, 8, 8)`; returns logits of shape
  `(N, 4)` and caches intermediate values for the backward pass), `backward(dlogits)`
  (`dlogits` is the gradient of the loss with respect to the logits, shape
  `(N, 4)`; it backprops through the chain and *sets* `self.grads`), and
  `params()` (returns the dict of trainable arrays keyed `W1, b1, W2, b2, Wfc, bfc`).
  The linear weight `Wfc` has layout `(D_in, D_out)` — the side that needs a
  transpose.
- `TorchCNN(nn.Module)` — the same architecture in PyTorch:
  `nn.Conv2d` -> ReLU -> maxpool, twice, then `nn.Linear`. Its `forward(X)` takes
  the input batch `X` (shape `(N, 3, 8, 8)`) and returns logits of shape
  `(N, 4)`. Autograd supplies its backward pass — you write no backward here.
- `load_weights_into_torch(numpy_model, torch_model)` — copy identical weights
  from the NumPy model into the torch model: conv weights and biases **directly**,
  the linear weight **transposed** (`Wfc.T`). Returns nothing; it mutates
  `torch_model` in place. Copies happen under `torch.no_grad()`.
- `compare_grads(numpy_model, torch_model, X, y)` — load identical weights, run
  one forward+backward on **CPU in float64**, and return a dict
  `{param_name -> max relative error}` comparing the NumPy gradient to torch's,
  with the linear gradient transposed back to `(D_in, D_out)`. `X` is the float64
  input batch, `y` the integer label array. It must reject non-float64 / non-CPU
  inputs.
- `torch_train_step(model, optimizer, X, y)` — one correct training step:
  `zero_grad()` **before** `backward()`, then `step()`. `model` is a `TorchCNN`,
  `optimizer` its optimizer, `X` the input batch tensor, `y` the label tensor.
  Returns the scalar loss as a Python `float`.
- `conv2d_backward(dout, X, W, stride, padding)` — the L11 warm-up the grader
  checks on its own. `dout` is the upstream gradient flowing into the conv output
  (same shape as that output), `X` the conv input `(N, C_in, H, W)`, `W` the
  filters `(C_out, C_in, kH, kW)`, `stride` and `padding` the conv settings. It
  returns `(dX, dW, db)` — the gradients with respect to the input, the filters,
  and the bias.

Then grade it: the site's Run Grader button or `uv run grade L14-numpy-pytorch-bridge`.

## How the grader checks you

The grader runs four checks, all on a tiny net (batch 8, CPU, float64 — so it is
fast):

- **`conv2d_backward` warm-up** (an independent root check). It builds a small
  random conv, computes `(dX, dW, db)` with your function, and confirms `dX` and
  `dW` have the right shapes. Then it verifies a few entries of `dW` against a
  **finite-difference** estimate (nudge a weight by a tiny `h`, see how the
  scalar loss changes, divide) to within a relative tolerance of `1e-4`. This is
  the L11 conv backward, recalled cold — it stands on its own and does not block
  the rest.
- **gradient equivalence** (the headline). It calls `compare_grads` on a fixed
  batch and requires the max relative error to be **below `1e-4` for every one of
  the six parameters** (`W1, b1, W2, b2, Wfc, bfc`). Before running it asserts the
  torch model is float64 and the batch is float64 — the correctness regime. If any
  parameter exceeds the bar, the failure message names the worst parameter, prints
  every per-parameter error, and reminds you that the usual cause is the
  linear-weight transpose.
- **interface contract.** It confirms `TorchCNN.forward` on a batch of 8 returns
  logits of shape `(8, 4)`. Every other check rides on this passing first.
- **`zero_grad` (the named trap).** Covered next.

### The named trap: missing zero_grad

This grader hunts one specific mistake by name: a `torch_train_step` that omits
`optimizer.zero_grad()` before `loss.backward()`. The buggy version is identical
to a correct solution in every other respect — it just leaves out the one clearing
line. To catch it, the grader runs your `torch_train_step` **twice on the same
data** with a learning rate of zero (so the weights never move and a correct step
must produce the *identical* gradient both times). It then compares the gradient
norm of step 2 to step 1.

If you forgot `zero_grad`, torch *adds* step 2's gradient on top of step 1's
leftover, so step 2's norm is roughly **2x** step 1's. The grader sees a ratio
above `1.5x` and fails with the message **"gradients are accumulating"** — telling
you in plain words that the step is missing its `optimizer.zero_grad()`, not
sending you hunting for a phantom bug elsewhere. Add the one line and the ratio
drops back to `1.0`.

## The hint ladder

If the grader's message is not enough, open the hints. Each piece has three
levels, mildest first: a conceptual nudge (the symptom and what it points to),
then the formula or invariant with shapes (e.g. numpy `Wfc` is `(D_in, D_out)`,
torch `.weight` is `(D_out, D_in)` — copy needs `.T`), then pseudocode you can
read almost line for line. Try level 1 before climbing — the goal is for the hint
to jog the idea, not hand you the answer.

## Done when

`uv run grade L14-numpy-pytorch-bridge` shows all-green: the `conv2d_backward` warm-up
matches its finite-difference check, `TorchCNN.forward` returns shape `(8, 4)`,
`compare_grads` reports every one of the six parameters under `1e-4` (you will see
errors around `1e-13`), and `torch_train_step` keeps the per-step gradient norm
stable across two identical steps (no ~2x blow-up). Then run the experiments and
record your predictions.
