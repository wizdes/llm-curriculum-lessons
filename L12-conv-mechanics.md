# L11 — Convolution Mechanics

In L9 you built an MLP — a *dense* network, where every layer was a single
matrix multiply (a **matmul**: rows-times-columns, the operation you wired up in
L9) and each output number depended on *every* input number. That works for a
flat vector, but an image is mostly **local**: the pixels that tell you "this is
an edge" sit right next to each other, and the same edge can appear in any
corner of the picture. A **convolution** is the layer built for exactly that. It
makes two bets that a dense layer does not:

- **Local connectivity** — each output looks at only a small square *window* of
  the input, not the whole thing.
- **Weight sharing** — the *same* small set of weights is reused at every window
  position as it slides across the image.

Those two bets are what make vision networks both small (few weights) and
*translation-aware* (an edge is recognized wherever it appears). This lesson
builds the mechanics from scratch on tiny fixed tensors — no dataset, no
training. You will write the forward pass two ways, the backward pass, and
pooling, and check them all against the gradient-checking idea from L7.

The calculus is the same **chain rule** you used in L7 and L9 (to get the
gradient through a chain of steps, multiply the local gradients of each step).
What is new is the *bookkeeping* of windows and shared weights — and two pieces
of that bookkeeping trip up almost everyone: summing the filter's gradient over
the batch, and the output-size arithmetic. The grader is built around those two
traps, so this lesson names them up front and comes back to each.

## The shapes and notation, defined once

Every tensor in this lesson follows one fixed layout. A **tensor** is just an
n-dimensional array of numbers; its **shape** is the tuple of its dimension
sizes. Here are the four we use, with every letter spelled out:

```text
X       (N, C_in,  H,    W   )   the input batch
weights (C_out, C_in, kH, kW )   the filters
bias    (C_out,)                 one bias number per output channel
output  (N, C_out, H_out, W_out) the result
```

Read those letters as:

- `N` — the **batch size**, how many images we process at once.
- `C_in` — the number of **input channels**. A channel is one "layer" of the
  image at every pixel: a color photo has `C_in = 3` (red, green, blue); a
  grayscale image has `C_in = 1`. Stacked deeper in a network, channels are just
  parallel feature maps (defined below).
- `H`, `W` — the input's **height** and **width** in pixels. (`W` the dimension
  is unrelated to `weights`; the code calls the width variable `Wd` to keep them
  apart.)
- `C_out` — the number of **output channels**, which equals the number of
  filters: each filter produces one output channel.
- `kH`, `kW` — the **kernel** height and width: how big each sliding window is,
  e.g. `3 × 3`.
- `H_out`, `W_out` — the output's height and width, set by the formula below.

The single formula that ties input size to output size, one axis at a time:

$$H_{\text{out}} = \left\lfloor \frac{H + 2\,\text{padding} - kH}{\text{stride}} \right\rfloor + 1$$

Read aloud: "the output height is the input height plus twice the padding, minus
the kernel height, all divided by the stride and rounded *down*, then plus one."
The $\lfloor \cdot \rfloor$ bars mean *floor* — round down to the nearest whole
number, which is what Python's `//` integer division does. The width uses the
exact same formula with `kW` and `W`. We define `stride` and `padding` precisely
in their own section; for now they are knobs that change the output size, and
this formula is the one source of truth for that size everywhere in the lesson.

## A filter that slides

A **filter** (also called a **kernel** — the two words mean the same thing here)
is a small block of weights, shape `(C_in, kH, kW)`. **Convolution** is the act
of taking that filter, laying it over one `(kH, kW)` window of the input,
multiplying the overlapping numbers together element by element, adding up all
those products into a single number, and adding a bias. That single number is
one cell of the output. Then you slide the window over by `stride` pixels and do
it again, sweeping across and down until you have covered the whole image. The
grid of numbers one filter produces is called a **feature map** — it lights up
wherever that filter's pattern appears in the input.

Two more terms you will meet constantly:

- **Receptive field** — the patch of the input that a given output cell "can
  see." With a `3 × 3` filter, each output cell's receptive field is the `3 × 3`
  window it was computed from. Stack convolutions and the receptive field grows:
  a cell deep in the network indirectly sees a large region of the original
  image.
- **Channels**, again, in the filter: a filter spans *all* `C_in` input
  channels at once. A `3 × 3` filter on an RGB image is really `3 × 3 × 3`
  numbers — three channels deep — and the elementwise multiply-and-sum runs over
  all of them.

Because the *same* filter visits every window position, one filter has only
`C_in * kH * kW` weights no matter how large the image is. A `3 × 3` filter on a
3-channel image is 27 weights whether the image is `28 × 28` or `1000 × 1000`.
That is **weight sharing**, and it is why convolutions stay small.

### One output cell by hand

Take the smallest concrete case: a single grayscale image (`C_in = 1`), one
filter (`C_out = 1`), a `3 × 3` kernel, stride 1, no padding. The input is a
`4 × 4` grid and the filter is `3 × 3`:

```text
input X (4×4)            filter W (3×3)
  1  2  3  0               1  0  -1
  0  1  2  3               1  0  -1
  4  5  6  1               1  0  -1
  2  0  1  3
```

To compute the *top-left* output cell, lay the `3 × 3` filter over the top-left
`3 × 3` window of the input — the bold region:

```text
[ 1  2  3] 0
[ 0  1  2] 3
[ 4  5  6] 1
  2  0  1  3
```

Multiply each input number by the filter number sitting on top of it, then sum:

$$(1{\cdot}1) + (2{\cdot}0) + (3{\cdot}{-1}) + (0{\cdot}1) + (1{\cdot}0) + (2{\cdot}{-1}) + (4{\cdot}1) + (5{\cdot}0) + (6{\cdot}{-1})$$

Working left to right: $1 + 0 - 3 + 0 + 0 - 2 + 4 + 0 - 6 = -6$. So the top-left
output cell is $-6$ (we chose bias $0$ here). Now slide the window one column
right and repeat for the next cell, and so on. With a `4 × 4` input, a `3 × 3`
filter, stride 1, and no padding, the formula gives
$H_{\text{out}} = (4 + 0 - 3)/1 + 1 = 2$, so the output is `2 × 2`. That filter
— positive on the left column, negative on the right — is a vertical-edge
detector: it returns a large number wherever the left side of the window is
brighter than the right.

Spelled out as a procedure, the whole forward pass is:

1. Compute `H_out` and `W_out` from the formula above.
2. Zero-pad the input by `padding` on all four spatial sides (skip if
   `padding = 0`).
3. For each image `n`, each output channel `oc`, and each output position
   `(i, j)`:
   a. Find the window's top-left corner: `hs = i * stride`, `ws = j * stride`.
   b. Slice the `(C_in, kH, kW)` window out of the padded input.
   c. Multiply that window elementwise by filter `W[oc]`, sum *every* product
      into one number, and add `b[oc]`.
   d. Store it at `out[n, oc, i, j]`.

This is `conv2d_forward` in the exercise. Four nested loops over
`(n, oc, i, j)`. It is slow — millions of tiny Python iterations on a real image
— but it *is* the definition of a convolution, so we keep it as the reference
that the faster version must match exactly.

## im2col: trading memory for one matmul

### im2col tradeoff

The nested loops are clear but slow, because Python runs each multiply-add as a
separate interpreted step. Real frameworks avoid this with a trick called
**im2col** (short for "image to columns"). The idea: instead of looping, *unroll*
every window into a flat column, stack all those columns into one big matrix, and
let a single matrix multiply do all the multiply-adds at once. A matmul is the
one operation that numerical libraries (**BLAS**, the low-level math library
NumPy calls) make extremely fast, so one big matmul beats millions of tiny loop
steps.

Concretely, im2col does this:

1. Compute `H_out` and `W_out` (same formula as always) and zero-pad the input.
2. For each output position `(i, j)`, slice out its `(C_in, kH, kW)` window and
   flatten it into a single column of length `C_in * kH * kW`.
3. Stack those columns side by side into a matrix `cols` of shape
   `(N, C_in*kH*kW, H_out*W_out)` — one column per output position.
4. Flatten each filter the same way: reshape `W` from `(C_out, C_in, kH, kW)`
   into a matrix `W_row` of shape `(C_out, C_in*kH*kW)`.
5. Multiply: `W_row @ cols` gives `(N, C_out, H_out*W_out)`. Each row of
   `W_row` (one filter) dotted with each column of `cols` (one window) is exactly
   one output cell — the same multiply-and-sum as before, now done in bulk.
6. Add the bias and reshape the result back to `(N, C_out, H_out, W_out)`.

That is `conv2d_forward_im2col` in the exercise. The reference solution writes
step 5 as a single `np.einsum("oc,ncp->nop", W_row, cols)` (Einstein summation —
a compact way to spell a batched matmul), but a plain `@` matmul per image is the
same computation.

Here is the tradeoff, made concrete. Every input pixel falls inside several
overlapping windows — with a `3 × 3` filter at stride 1, an interior pixel sits
in 9 different windows. im2col copies that pixel once into *each* of those
windows' columns. So `cols` holds roughly `kH * kW` copies of the input: for a
`3 × 3` filter, about 9× the input's memory. You pay that memory to buy speed —
**speed for memory** is the tradeoff, and it is the bargain every deep-learning
framework makes.

Because im2col is just a rearrangement, its output must be *numerically equal* to
the naive loops (equal up to tiny floating-point rounding, since the additions
happen in a different order). If it ever disagrees, the first thing to recheck is
the output size — because `cols` is sized by `H_out * W_out`, and a wrong size
builds the wrong number of columns. That is exactly the first named trap, next.

The single most common im2col bug is the **output size** itself. Per axis it is
$(H + 2\,\text{padding} - kH)\,/\!/\,\text{stride} + 1$ (with `//` the
round-down division). Drop the `+1`, or forget the `2 * padding` term, and
`cols` ends up with the wrong number of columns — the matmul still runs and
returns a wrong-shaped output, no error raised. The grader asserts the spatial
output equals that exact formula and names this mistake when it fires.

## Backward: where the gradient flows

### conv backward

Training a conv layer means computing **gradients** — for each weight, how much
the loss would change if you nudged that weight a hair (the partial derivative of
the loss with respect to that weight). `conv2d_backward(dout, X, W, stride,
padding)` takes `dout`, the gradient already flowing back from the layers above
(shape `(N, C_out, H_out, W_out)`, one number per output cell, the same shape the
forward pass produced), and returns three gradients:

```text
dX (N, C_in, H, W)        gradient flowing further back, to the input
dW (C_out, C_in, kH, kW)  gradient of the SHARED filters
db (C_out,)               gradient of the bias
```

The chain rule for one output cell is short. Say the upstream gradient at output
cell `(n, oc, i, j)` is the number `g = dout[n, oc, i, j]`. Because that cell was
computed as `sum(patch * W[oc]) + b[oc]`, the local derivatives are immediate,
and the chain rule (multiply the upstream gradient `g` by each local derivative)
gives, for that one cell:

```text
dW[oc]                  += g * patch       # patch = the input window that fed this cell
dX[window for (n,i,j)]  += g * W[oc]
db[oc]                  += g
```

As a procedure over the whole batch:

1. Initialize `dX`, `dW`, `db` to zeros of the right shapes.
2. Compute `db` in one shot: `db = dout.sum(axis=(0, 2, 3))` — sum every output
   cell's gradient, per channel.
3. For each `(n, oc, i, j)`, with `g = dout[n, oc, i, j]` and the window at
   `hs = i*stride, ws = j*stride`:
   a. `dW[oc] += g * X_window` (accumulate the filter gradient).
   b. `dX[window] += g * W[oc]` (scatter gradient back to the input pixels).
4. If `padding > 0`, the loop accumulated `dX` on the *padded* grid; slice off
   the border to return `dX` at the original `(H, W)` size.

The cell that trips people is **`dW`**. The same filter `W[oc]` was used at every
position of every example in the batch, so its gradient is the **sum** of *all*
those contributions — that is the flip side of weight sharing: shared in the
forward pass means summed in the backward pass. The correct `dW` is 4-D, exactly
`W.shape = (C_out, C_in, kH, kW)`.

The trap: if you accumulate a *separate* `dW` per example and forget to collapse
the batch, `dW` comes out 5-D — `(N, C_out, C_in, kH, kW)`, with a leftover
leading `N` axis. The fix is to **sum over axis 0** (the batch axis):
`dW = dW_per_sample.sum(axis=0)`, or — cleaner — accumulate straight into a
W-shaped array as in step 3a above. The grader checks `dW`'s number of dimensions
and names this exact mistake. (Notice `db` collapses the batch the same way: it
sums over the batch axis *and* both spatial axes at once,
`dout.sum(axis=(0, 2, 3))`, which is why it is one line.)

A note on what we are differentiating, so the picture stays straight: we take the
gradient of the **loss** with respect to the **parameters** (`W`, `b`) and the
input (`X`), never "the gradient of the model." `dout` is the loss's gradient
arriving from above; `conv2d_backward` just carries it one layer further back via
the chain rule. The grader confirms each gradient against a **finite-difference
check** — nudge one number, see how the loss changes, compare to the formula
— which is the same slow per-parameter sanity check from L7, used here only to
*verify* the analytic formulas, not to train.

## Pooling, stride, and padding

### stride padding

`stride` and `padding` are the two dials that set the output size, through the
one formula $(\text{in} + 2\,\text{padding} - k)\,/\!/\,\text{stride} + 1$:

- **padding** adds a border of zeros around the input before convolving. Without
  it, the filter can never center on an edge pixel (it would hang off the side),
  so every convolution shrinks the image and the borders get under-counted.
  Padding lets the filter sit centered on edge pixels. The choice
  `padding = (k - 1) // 2` — for a `3 × 3` filter that is padding 1 — makes the
  output the *same* size as the input at stride 1 (often called "same"
  padding). Padding 0 (no border) is called "valid" padding and always shrinks.
- **stride** is the step, in pixels, between one window and the next. Stride 1
  moves one pixel at a time and visits every position. Stride 2 skips every other
  position, so it roughly *halves* each spatial axis — a cheap way to shrink the
  feature map.

Worked sizes for a `4 × 4` input and a `3 × 3` filter, straight from the
formula:

| stride | padding | $H_\text{out}$ | why |
|--------|---------|----------------|-----|
| 1 | 0 | $(4+0-3)/1+1 = 2$ | valid: shrinks |
| 1 | 1 | $(4+2-3)/1+1 = 4$ | same: size preserved |
| 2 | 1 | $(4+2-3)/2+1 = 2$ | stride halves it |

Predict the output shape from the formula before you run anything — the grader's
shape-contract check exercises several `(stride, padding)` combinations against
exactly that arithmetic.

**Max pooling** is a different way to shrink a feature map: slide a window over
it and keep only the *maximum* value in each window, throwing the rest away. It
has **no weights** to learn — it is a fixed operation, useful for collapsing a
region down to its strongest response. Its backward pass is pure *routing*: the
gradient flows only to the cell that *was* the max (the **argmax** — the position
of the maximum) in each window; every other cell in the window gets exactly zero.
Intuitively, only the winning cell affected the output, so only it gets gradient.
As a procedure, for each window: find the argmax position, send that window's
upstream gradient to it, and leave the rest at zero. (Ties go to the first max,
matching `np.argmax`.) That is `maxpool2d_forward` / `maxpool2d_backward` in the
exercise.

## Exercise

Implement the six functions in `my_work/L12-conv-mechanics/conv_mechanics.py` (run
`uv run grade start L12-conv-mechanics` to get the scaffold). NumPy only — no other
libraries. Throughout, `X` is the input batch `(N, C_in, H, W)`, `W` is the
filters `(C_out, C_in, kH, kW)`, `b` is the bias `(C_out,)`, `stride` and
`padding` are non-negative integers, and `dout` is the upstream gradient.

- `relu_backward(dA, Z)` — the **warm-up**, recalled from L9. Parameters: `dA`
  is the gradient arriving from above and `Z` is the layer's pre-activation
  input (the values that were fed into the ReLU); both share the same shape.
  Returns an array of that same shape: pass `dA` through where `Z` was positive
  and zero it elsewhere (`dA * (Z > 0)`). It is graded on its own and not used by
  the other functions — re-authoring it from memory anchors the chain-rule
  intuition before the conv machinery.
- `conv2d_forward(X, W, b, stride, padding)` — the naive nested-loop
  convolution, the reference. Returns `out` of shape
  `(N, C_out, H_out, W_out)`. Follow the four-step forward procedure above; this
  is the definition every other function is checked against.
- `conv2d_forward_im2col(X, W, b, stride, padding)` — the same convolution as
  one matmul over unrolled patches. Same inputs, and it must return an output
  *numerically equal* to `conv2d_forward` (same shape `(N, C_out, H_out,
  W_out)`). Follow the six-step im2col procedure; get the output size right.
- `conv2d_backward(dout, X, W, stride, padding)` — the backward pass. Parameters:
  `dout` is the upstream gradient `(N, C_out, H_out, W_out)`; `X` and `W` are the
  same input and filters from the forward pass. Returns the triple
  `(dX, dW, db)` with shapes `(N, C_in, H, W)`, `(C_out, C_in, kH, kW)`,
  `(C_out,)`. Remember the batch sum on `dW` — it must be 4-D, never 5-D.
- `maxpool2d_forward(X, pool_size, stride)` — max pooling. Parameters: `X` is the
  input `(N, C, H, W)`; `pool_size` is the window edge length (an int); `stride`
  is the step between windows. Returns `(out, cache)` where `out` is
  `(N, C, H_out, W_out)` and `cache` is `X` itself, reused by the backward pass
  to find each window's argmax.
- `maxpool2d_backward(dout, X, pool_size, stride)` — route each upstream gradient
  to its window's argmax. Parameters: `dout` is `(N, C, H_out, W_out)`; `X` is the
  original pooling input; `pool_size` and `stride` as above. Returns `dX`
  `(N, C, H, W)` with the upstream gradient placed only at each window's argmax
  cell and zeros everywhere else.

Then grade it: the site's Run Grader button or `uv run grade L12-conv-mechanics`.

## How the grader checks you

- The **warm-up** `relu_backward` is an independent unit, graded on its own
  against a fixed `2 × 2` example; a wrong warm-up never hides a working
  convolution.
- `conv2d_forward`'s output shape is checked first (it is the root every other
  check depends on), then `conv2d_forward_im2col` is checked **equal to** the
  naive forward for both padding 0 and padding 1, to a relative tolerance of
  about $10^{-7}$ — they must produce the same numbers.
- `dX`, `dW`, and `db` are each checked against a **centered finite-difference**
  gradient (the loss is reduced to a single number by a fixed random weighting,
  then each input is nudged by a tiny `eps` and the change measured), to a
  tolerance of about $10^{-6}$. Each gradient's shape is checked too.
- The two `(stride, padding)` shape contracts — `(1, 0)`, `(1, 1)`, `(2, 1)` —
  must match the output-size formula exactly.
- `maxpool2d_forward`'s output shape is checked, then `maxpool2d_backward` is run
  on a hand-built `4 × 4` with a clear max per `2 × 2` window: the gradient must
  land *only* on those four argmax cells, with the right value at each.

### The named trap: the two that almost everyone hits

This grader hunts two specific mistakes by name, the ones flagged in the intro.

- **`conv2d_backward`'s batch axis.** If your `dW` comes back 5-D —
  `(N, C_out, C_in, kH, kW)`, because you accumulated a separate filter gradient
  per example and never collapsed the batch — the grader sees the extra leading
  axis and tells you the fix in words: filters are shared, so
  `dW = dW_per_sample.sum(axis=0)`. The correct `dW` is 4-D, matching
  `W.shape`.
- **`conv2d_forward_im2col`'s output size.** If you drop the `+1` or the
  `2 * padding` term, the spatial output comes out the wrong size; the matmul
  still runs, so there is no crash, just wrong numbers. The grader compares your
  output size to `(in + 2*padding - k)//stride + 1` and names the dropped term
  as the usual cause, so you fix the arithmetic instead of hunting a phantom bug.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the formula with shapes, then pseudocode. Try
level 1 before climbing — there are dedicated ladders for the `dW` batch-axis
sum, the im2col output size, the `dW`-vs-`dX` correlation orientation, and the
maxpool argmax routing.

## Done when

`uv run grade L12-conv-mechanics` shows all-green: the warm-up passes; `conv2d_forward` has
the right shape and `conv2d_forward_im2col` matches it numerically at padding 0
and 1; `dX`, `dW`, and `db` all match the finite-difference gradients (with `dW`
4-D, not 5-D); every `(stride, padding)` shape contract holds; and
`maxpool2d_backward` routes gradient only to each window's argmax. Then run the
experiments and record your predictions.
