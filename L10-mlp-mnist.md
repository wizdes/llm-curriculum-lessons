# L9 — MLP on MNIST

This is the first time you train a real multi-layer network end to end. In L7 you
ran **backprop** — the chain rule that pushes the loss's gradient back through a
network — by hand on one tiny 2-2-1 network (two inputs, two hidden units, one
output), with a single example and one number at a time. In L8 you built an
**autograd** engine that did the same chain rule automatically, still one scalar
at a time. Both worked, and both would take forever on a real dataset, where one
layer has tens of thousands of weights and you push a whole **batch** (many
examples stacked together) through at once.

This lesson makes the leap to **vectorized** backprop. A network built from
matrix-multiplying layers like this is a **multi-layer perceptron**, or **MLP**:
"perceptron" is the historical name for one artificial neuron, and "multi-layer"
means we stack layers of them. Every layer is now a matrix operation, and the
chain rule is the *same* chain rule — just written with the matrix-multiply
operator `@` instead of Python loops.

The math has not changed from L7. What changes is the bookkeeping, and three
specific pieces of bookkeeping trip up almost everyone the first time: the shape
of the weight gradient (the `dW` transpose), the axis you sum the bias gradient
over, and keeping the **softmax** from overflowing. The lesson is built around
those three.

A note on terms you will meet below, each defined in full where it first appears:
a **hidden layer** is any layer between the input and the final output; **softmax**
turns a vector of raw scores into a probability distribution; **cross-entropy** is
the loss we use when the answer is one of several classes; **one-hot** is how we
write "the answer is class 3" as a vector. The task is **MNIST**: classify a
28×28 grayscale image of a handwritten digit (flattened to a vector of 784 pixel
values) as one of the 10 digits 0–9.

## From scalars to matrices: the same chain rule

### Step zero: vectorizing the L7 network

Before touching MNIST, prove to yourself that the matrix form produces the *same
numbers* as L7. That is what the `vectorize_xor` step does — it is the
cliff-softener between "one scalar at a time" and "a whole layer at once."

We take L7's exact 2-2-1 network — `tanh` hidden units, a **linear** output unit
(no squashing on the output), and **half-squared-error** loss
`0.5 * (out - y)**2` — at its exact reference point `(x1=0, x2=1, y=1)`, and write
the whole thing with matrices. ("Linear output" means the output is just a weighted
sum, with no `tanh` applied; we keep it linear because L7 did, so the numbers must
match.) Here is the forward pass, with the shape of every array in a comment.
Read each shape as "(rows, columns)"; a trailing comma like `(2,)` means a flat
vector of length 2:

```python
Z1 = X @ W1 + b1     # X (1,2), W1 (2,2), b1 (2,) -> Z1 (1,2)
H  = tanh(Z1)        # (1,2)
out = H @ W2 + b2    # W2 (2,1), b2 (1,) -> out (1,1)  — LINEAR output
loss = 0.5 * (out - y)**2
```

Read aloud: *the hidden pre-activation `Z1` is the input row times the
first weight matrix plus the first bias; squash it with `tanh` to get the hidden
activations `H`; the output is `H` times the second weight matrix plus the second
bias; the loss is half the squared miss.* `@` is matrix multiply; the single `X`
row holds the two inputs `(x1, x2)`.

The backward pass is the matrix chain rule. Each `d`-prefixed name means "the
gradient of the loss with respect to this array" — for example `dout` is how much
the loss changes per unit change in `out`:

```python
dout = out - y                # (1,1)  — gradient of the loss w.r.t. out
dW2 = H.T @ dout              # (2,1)  — shape of W2
db2 = dout.sum(axis=0)        # (1,)   — summed over the batch
dH  = dout @ W2.T             # (1,2)
dZ1 = dH * (1 - H**2)         # (1,2)  — tanh'(z) = 1 - tanh(z)**2
dW1 = X.T @ dZ1               # (2,2)  — shape of W1
db1 = dZ1.sum(axis=0)         # (2,)
```

The full procedure, as numbered steps:

1. **Output-error.** Set `dout = out - y`. (This is the gradient of
   half-squared-error through the linear output — the `1/2` cancels the `2` from
   the square.)
2. **Second-layer weights and bias.** `dW2 = H.T @ dout` (takes the shape of
   `W2`); `db2 = dout.sum(axis=0)` (summed over the batch).
3. **Backprop into the hidden activations.** `dH = dout @ W2.T`.
4. **Through the `tanh`.** `dZ1 = dH * (1 - H**2)`, because the derivative of
   `tanh(z)` is `1 - tanh(z)**2` and `H = tanh(Z1)`.
5. **First-layer weights and bias.** `dW1 = X.T @ dZ1` (shape of `W1`);
   `db1 = dZ1.sum(axis=0)`.

Pack L7's nine scalar weights into the matrices column by column — **column `j`
holds the weights feeding unit `j`** — and you recover L7's hand-derived
`dw1..dw6, db1..db3` to within `1e-7` (one part in ten million). Same gradients,
new shape. Once you have *felt* that the matrices give back numbers you already
trust, the layers below are just this same pattern at scale.

One misconception to head off here, because it is the one people get backwards:
backprop differentiates the **loss with respect to the parameters** (the weights
and biases), not "the network with respect to anything." Every `d`-name above is a
gradient *of the loss*. The whole reason we compute them is to nudge each
parameter in the direction that lowers the loss.

### A Linear layer and its gradients

A **Linear layer** (also called fully connected, or dense) computes
`Z = X @ W + b`. Its weight matrix `W` has shape `(D_in, D_out)` — `D_in` inputs
feeding `D_out` outputs — and its bias `b` has shape `(D_out,)`. For a batch `X`
of shape `(N, D_in)` (`N` examples, each a row of `D_in` numbers), the output `Z`
is `(N, D_out)`.

The backward pass receives the **upstream gradient** `dZ` (the gradient of the
loss with respect to this layer's output, same shape as `Z`, handed to us by the
layer above) and produces three things:

- `dX`, the gradient to pass *downstream* to the layer below, shape `(N, D_in)`;
- `dW`, the gradient for this layer's weights, shape `(D_in, D_out)`;
- `db`, the gradient for this layer's bias, shape `(D_out,)`.

The three formulas are `dX = dZ @ W.T`, `dW = X.T @ dZ`, and
`db = dZ.sum(axis=0)`. The next section is about why the last two are exactly the
spots beginners get wrong.

## The three classic bugs

### Transpose confusion

The single most common vectorized-backprop bug is getting the `dW` transpose
backwards — putting the `.T` on the wrong array.

The rule is `dW = X.T @ dZ`. Here `X` is `(N, D_in)`, so `X.T` (its transpose,
rows and columns swapped) is `(D_in, N)`; `dZ` is `(N, D_out)`; the product is
`(D_in, N) @ (N, D_out) = (D_in, D_out)` — exactly the shape of `W`. That last
fact is the check that saves you: **`dW` always takes the shape of `W`.** A
gradient says "how much to nudge each weight," so there is one gradient number per
weight, and they must line up element for element.

Write `dZ.T @ X` instead and you get `(D_out, N) @ (N, D_in) = (D_out, D_in)` —
the transpose landed on the wrong array, the shape is `W` flipped, and the
grader's flagship message says, in these words, to *revisit which side the
transpose lands on in the chain rule*. When in doubt, do not memorize the operand
order; derive it from the shape that has to come out.

### Bias grad axis

The bias `b` is **shared across every example in the batch** — the very same `b`
is added to every row of `Z`. When one parameter feeds many outputs, the chain
rule says its gradient is the **sum** of the gradients from all of them. So the
bias gradient is the upstream gradient summed over the batch:
`db = dZ.sum(axis=0)`, which collapses the `N` rows and gives shape `(D_out,)`.

Read `dZ.sum(axis=0)` aloud as *add up `dZ` down the rows (across the `N`
examples), leaving one number per output.* The trap is to return `dZ` unchanged
(or to sum over the wrong axis), which leaves the full `(N, D_out)` batch shape.
The tell is simple: **if `db` comes out two-dimensional, you forgot to sum over
`axis=0`.**

### Softmax stability

To turn the network's final scores into a prediction we need two new pieces, so
define them first.

The final Linear layer outputs **logits**: a length-10 vector of raw, unbounded
scores, one per digit class, with no constraint to be positive or to sum to
anything. **Softmax** turns those logits into a **probability distribution** — ten
non-negative numbers that sum to 1, which we can read as "the model's probability
that the image is each digit." For a logit vector `z`, softmax is
`exp(z_i) / sum_j exp(z_j)`: exponentiate every score (so all are positive), then
divide by the total (so they sum to 1).

We score the prediction with **cross-entropy**, the standard loss for picking one
of several classes. The true answer is written **one-hot**: the label "this image
is a 3" becomes the vector `[0,0,0,1,0,0,0,0,0,0]` — a 1 in the true class's slot
and 0 everywhere else. Cross-entropy is `-sum_i onehot_i * log(prob_i)`, which
because the one-hot vector is zero everywhere except the true class collapses to
just `-log(prob of the true class)`. Read aloud: *the loss is the negative log of
the probability the model assigned to the correct digit* — high probability on the
right answer means a small loss; near-zero probability means a large one. (This is
the multi-class cousin of the binary cross-entropy from L4. Like every loss in
this course, it is **task-specific** — but the optimizer that consumes its
gradient is the same plain gradient descent from L1; only the loss's gradient
formula changes.)

Now the numerical problem. Softmax exponentiates the logits, and `exp` of a large
number overflows to `inf` — a single logit of 1000 gives `exp(1000) = inf`, the
normalization becomes `inf / inf = NaN` ("not a number"), and the loss prints
`nan`. The fix costs one line and changes the answer not at all: **subtract the
per-row maximum logit before `exp`.**

```python
shifted = logits - logits.max(axis=1, keepdims=True)   # overflow guard
exp = np.exp(shifted)
probs = exp / exp.sum(axis=1, keepdims=True)
```

(`axis=1` is "across the columns within each row," i.e. across the 10 classes for
one image; `keepdims=True` keeps the result a column so it subtracts row-wise by
broadcasting.)

Why this is mathematically free: subtracting the same constant `m` from every
logit in a row multiplies every `exp` by `exp(-m)`, and that common factor appears
in both the numerator and the denominator of the softmax, so it **cancels**
exactly. The probabilities are identical — but now the largest shifted logit is
`0`, so its `exp` is `exp(0) = 1`, and nothing can overflow. This is the same
shift trick that made L4's binary cross-entropy stable.

Here is the cancellation on a concrete row. Take logits `[1, 2, 1000]`:

- **Naively:** `exp([1, 2, 1000]) = [2.7, 7.4, inf]`, so dividing by the (infinite)
  sum gives `probs = [0, 0, nan]` (`inf / inf` is `nan`) — the loss is `NaN` and
  training dies on the first step.
- **Shifted:** subtract the row max `1000` to get `[-999, -998, 0]`; then
  `exp = [~0, ~0, 1]`, the sum is `~1`, and `probs ≈ [0, 0, 1]` — finite, correct,
  and the same distribution the math intends.

For the backward pass, the gradient of cross-entropy through softmax simplifies to
a famously clean form: `dlogits = (softmax - onehot) / N`. Read aloud: *the
gradient on each logit is the predicted probability minus the one-hot target,
averaged over the batch* — for the true class softmax is too low (the term is
negative, push it up); for the wrong classes softmax is too high (positive, push
it down). The `/ N` averages over the `N` examples so the step size does not grow
with the batch.

## The reference implementation

`worked/mlp_mnist.py` is the full reference. It contains `sigmoid` (the L4 warm-up,
recalled cold), `vectorize_xor` (the step-zero check above), the `Linear` and
`ReLU` layers, the stable `softmax_cross_entropy`, and `train_epoch`, which runs
one forward/backward/SGD pass. (`ReLU` is the activation `max(0, Z)`: it zeroes
out negatives and passes positives through unchanged; its backward pass lets a
gradient through only where the forward input was positive.) Its `__main__` block
builds a 784→128→10 MLP and trains a few epochs on MNIST as a self-check. It
imports `load_mnist` lazily, so merely importing the module never touches the
dataset.

How one training epoch fits together, as numbered steps (this is `train_epoch`):

1. **Forward.** Feed the batch `X` through the layers in order — each Linear's
   `forward`, each ReLU's `forward` — to reach the final logits.
2. **Loss.** Call `softmax_cross_entropy(logits, y)` to get the scalar loss and
   the starting gradient `dlogits`.
3. **Backward.** Pass `dlogits` back through the layers in *reverse* order,
   calling each layer's `backward`, which sets each Linear's `self.dW` and
   `self.db` along the way.
4. **Update.** For each Linear layer, take one SGD step: `W -= lr * dW` and
   `b -= lr * db`.
5. **Return** the scalar loss for this epoch.

## Exercise

Fill in `my_work/L10-mlp-mnist/mlp_mnist.py` (run `uv run grade start L10-mlp-mnist` to get the
scaffold). numpy only. Do the warm-up and the step-zero check first, then build
the layers, deriving each gradient from the shapes that must come out.

- `sigmoid(z)` — the **warm-up**, recalled cold from L4. Parameter: `z` is a numpy
  array (any shape) of pre-activations. Returns `1 / (1 + exp(-z))`, the logistic
  squashing function, same shape as `z`; clip `z` to a safe range first so a
  large-magnitude value cannot overflow `exp`. Graded on its own; not used by the
  functions below.
- `vectorize_xor(init=INIT)` — the **step-zero** check. Parameter: `init` is a dict
  of the nine L7 init weights `w1..w6, b1..b3` (defaulted to the shared `INIT`).
  Pack them into `X (1,2)`, `W1 (2,2)`, `b1 (2,)`, `W2 (2,1)`, `b2 (1,)` per the
  column-`j`-is-unit-`j` mapping, run forward + backward **once** at the reference
  point, and return a dict of the nine gradients keyed `dw1..dw6, db1..db3` (each a
  Python float). They must reproduce L7's numbers.
- `Linear(d_in, d_out, rng=None)` — the fully connected layer. Constructor params:
  `d_in` and `d_out` are the input and output widths (ints); `rng` is an optional
  numpy random generator for the weight init. Implement two methods.
  `forward(X)`: `X` is `(N, D_in)`; return `Z = X @ W + b`, shape `(N, D_out)`, and
  cache `X` for backward. `backward(dZ)`: `dZ` is the upstream gradient `(N,
  D_out)`; set `self.dW = X.T @ dZ` (shape `(D_in, D_out)`) and
  `self.db = dZ.sum(axis=0)` (shape `(D_out,)`), and return `dX = dZ @ W.T` (shape
  `(N, D_in)`).
- `ReLU()` — the activation. `forward(Z)`: return `max(0, Z)`, same shape as `Z`,
  and cache the positivity mask `Z > 0`. `backward(dA)`: `dA` is the upstream
  gradient; return `dA` masked so it passes through only where the forward input
  was positive (`dA * mask`).
- `softmax_cross_entropy(logits, labels)` — the loss. Params: `logits` is `(N, C)`
  (`C` classes); `labels` is integer class indices, shape `(N,)`. Subtract the
  per-row max before `exp` (the stability guard), compute the mean negative
  log-probability of the true class, and return `(loss, dlogits)` where `loss` is a
  scalar and `dlogits = (softmax - onehot) / N`, shape `(N, C)`.
- `train_epoch(X, y, layers, lr)` — one SGD pass. Params: `X` is the batch
  `(N, D_in)`; `y` is integer labels `(N,)`; `layers` is the list
  `[Linear, ReLU, ..., Linear]` ending in the logits Linear; `lr` is the learning
  rate. Run the five steps above and return the scalar mean loss.

Then grade it: the site's Run Grader button or `uv run grade L10-mlp-mnist`.

## How the grader checks you

- The **warm-up** `sigmoid` is an independent unit: checked at `sigmoid(0) = 0.5`
  and `sigmoid(2) = 1/(1+e^-2)` to a relative tolerance of `1e-7`. A wrong warm-up
  does not block the rest.
- The **headline** `vectorize_xor` is run and each of the nine returned gradients
  is compared against the L7 reference value (`dw1..dw6, db1..db3`) to within
  `1e-7`. A mismatch on any one points at the row/column mapping or at a wrong
  `dW1`/`dW2` formula, and the message names the offending key.
- `Linear.forward` is checked against `X @ W + b` on a fixed `(2,3)` float64 batch.
  `Linear.backward`'s `dW` and `db` are checked against a **numerical gradient**
  (the finite-difference estimator from L1) on a tiny fixed batch, to a relative
  tolerance around `1e-5`. `ReLU.backward` is checked against `dA * (Z > 0)`.
- `train_epoch` is run twice on a fixed seeded batch (20 examples, a
  6→5→3 network) and the loss on the second call **must be strictly lower** than on
  the first — one SGD step should reduce the loss. A non-decreasing loss points at
  a gradient sign error, a missing bias sum, or a transposed `dW`.

### The named trap: the three vectorized-backprop bugs

This grader is built to catch three mistakes by name — the three this lesson is
organized around — and each fires its own teaching message:

- **The `dW` transpose.** Writing `dW = dZ.T @ X` gives shape `(D_out, D_in)`
  instead of `(D_in, D_out)`. The shape check fails with the flagship message to
  *revisit which side the transpose lands on in the chain rule* and quotes the
  rule `dW = X.T @ dZ`.
- **The bias batch-sum.** Returning `self.db = dZ` (no sum) leaves `db` with the
  `(N, D_out)` batch shape. The grader catches the wrong shape and tells you to
  *sum dZ over the batch: db = dZ.sum(axis=0)*.
- **Softmax overflow.** Skipping the max-subtraction makes the loss non-finite on
  a logit of 1000. The grader checks the loss is finite and, when it is not, tells
  you to *Subtract max(logits)* per row before `exp`.

## The hint ladder

If the grader's message is not enough, open the hints. There is a three-level
ladder for each of the three traps — `dw_transpose`, `bias_axis`, and
`softmax_stability` — climbing from a conceptual nudge, to the formula with its
shapes, to pseudocode. Try level 1 before climbing.

## Done when

`uv run grade L10-mlp-mnist` shows all-green: `sigmoid` matches the known values; the
headline `vectorize_xor` reproduces all nine L7 gradients to `1e-7`;
`Linear.forward` matches `X @ W + b`; `Linear.backward`'s `dW` and `db` match the
numerical gradient with the right shapes; `ReLU.backward` masks correctly; and
`train_epoch` strictly reduces the loss on the fixed seeded batch. Then run the
experiments and record your predictions.
