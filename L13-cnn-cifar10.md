# L12 — CNN on CIFAR-10

This is the lesson where the convolution from L11 stops being a toy and becomes a
real image classifier. You will assemble the pieces you already built — the
convolution and max-pooling from L11, the Adam optimizer from L10, the softmax
cross-entropy loss from L9 — into a small **convolutional neural network** (a
CNN: a network whose main layers are convolutions, which we define properly in a
moment) and train it to sort color photos into ten categories.

Almost nothing here is new math. You already know how a single convolution
computes its output and its gradients; you already know how Adam updates a
parameter; you already know how cross-entropy turns scores into a loss. The one
genuinely new skill is **threading the chain rule cleanly through a stack of
stages**. A CNN is conv, then activation, then pool, then conv, then activation,
then pool, then a flatten, then one ordinary linear layer. Each stage's
*input-gradient* is the previous stage's *output-gradient*, and getting that
hand-off right — in the exact reverse order of the forward pass — is the whole
job of `backward`.

## The dataset: CIFAR-10

CIFAR-10 is a standard collection of 32×32 color photographs sorted into ten
classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
"Color" is the important word. Each image is not a flat grid of brightness values
like the grayscale MNIST digits from L9 — it has three **channels**, one each for
red, green, and blue. A channel is just one of those color planes: a full 32×32
grid of numbers giving the red intensity at every pixel, then another for green,
then another for blue. So one CIFAR-10 image is a stack of three 32×32 grids,
which we write as the shape `(3, 32, 32)` — read "three channels, each 32 pixels
tall and 32 pixels wide."

We process many images at once for speed, so the input to the network is a
**batch**: a stack of `N` images with shape `(N, 3, 32, 32)`. Throughout this
lesson, `N` is the number of images in the batch, and the leading axis of every
tensor is always that batch axis. CIFAR-10 is genuinely harder than MNIST —
cats and dogs in cluttered scenes are far less tidy than centered black-on-white
digits — which is exactly why it is worth pointing a CNN at.

## Why a CNN, not the MLP from L9

The MLP in L9 was a **multilayer perceptron**: it flattened the whole image into
one long vector and connected every input number to every neuron in the first
layer with its own weight. That works, but it is wasteful on images, and seeing
*why* is the motivation for everything that follows.

Take L9's approach on a CIFAR-10 image. Flatten `(3, 32, 32)` and you get a
vector of `3 * 32 * 32 = 3072` numbers. If the first dense layer has, say, 256
neurons, that one layer alone needs `3072 * 256 ≈ 786{,}000` weights — and it
gets worse than the raw count suggests. Because every weight is tied to one
specific pixel position, the network has to *relearn* what a cat's ear looks like
separately for every spot in the frame where an ear might appear. A pattern in
the top-left teaches it nothing about the same pattern in the bottom-right.

A convolution makes two bets that pay off enormously on images:

- **Local connectivity.** Each output cell looks at only a small `(kH, kW)`
  window of the input — here a 3×3 patch — not the entire image. The bet is that
  the pixels which decide "is there an edge right here?" are the nearby ones. The
  symbols `kH` and `kW` are the kernel height and kernel width: how tall and wide
  that little window is. A "kernel" (also called a filter) is the small grid of
  weights that gets multiplied against the patch.
- **Weight sharing.** The *same* filter slides across every position of the
  image. A feature the filter learns to detect in one corner is therefore
  detected everywhere, for free — one set of weights, reused at every location.

The payoff in parameter count is stark. A 3×3 convolution that produces 32 output
channels from a 3-channel input has `32 * 3 * 3 * 3 + 32 = 896` weights (the
`+ 32` is one bias per output channel), and that handful of weights scans the
*entire* image. The equivalent fully-connected layer cost hundreds of thousands.
Fewer parameters, translation-awareness built in, and far less training data
needed to generalize. That is the case for a CNN over an MLP, and the experiments
let you measure the parameter gap yourself.

## Pooling: shrinking the grid on purpose

Between convolutions, a CNN usually **pools**. Pooling is **downsampling**:
deliberately shrinking the spatial grid so later layers see a coarser, smaller
picture. We use **max pooling** with a 2×2 window and stride 2 (the *stride* is
how far the window jumps each step — 2 means non-overlapping 2×2 tiles). For each
2×2 tile, max pooling keeps only the single largest value and throws the other
three away, so each pooling layer halves the height and halves the width.

Why throw away three-quarters of the numbers on purpose? Two reasons. First, it
cuts the amount of computation the next layer must do. Second, keeping the
*maximum* response in a neighborhood says "the strongest evidence for this
feature anywhere in this little region" — which makes the network a bit
insensitive to exactly *where* a feature sat, a property you want when a cat can
be a few pixels left or right. Max pooling has no weights of its own; it is a
fixed operation. That fact matters a lot for its backward pass, which we return
to below.

## The architecture, shape by shape

The network is deliberately small and its shape is fixed. In plain words it is:
two rounds of "convolve, then ReLU, then pool," followed by a flatten and a
single linear layer that produces the class scores.

```text
conv(3x3, pad 1, stride 1) -> ReLU -> maxpool(2,2)
conv(3x3, pad 1, stride 1) -> ReLU -> maxpool(2,2)
flatten -> Linear -> logits
```

A few terms in that diagram, defined once:

- **pad 1** means we add a one-pixel border of zeros around the image before
  convolving. With a 3×3 kernel, that padding makes the convolution's output the
  same height and width as its input — the spatial size is *preserved*, so only
  the pooling layers change the grid size. Without padding, a 3×3 conv would
  shave a pixel off each edge.
- **ReLU** is the activation "rectified linear unit": `ReLU(z) = max(z, 0)`. It
  replaces every negative number with zero and leaves positives untouched, which
  is what lets the network represent something more interesting than a single
  big linear function. You met it in L9.
- **flatten** reshapes the final stack of feature grids into one long vector per
  image, so an ordinary linear layer can consume it.
- **logits** are the raw, un-normalized class scores the network outputs — one
  number per class, before softmax turns them into probabilities. (You named
  these "logits" back in L9's softmax.)

Now the part that trips people up: tracking how the tensor *shape* changes as
data flows through. Let's walk one full CIFAR-10 batch through the stack, using
the experiments' full configuration: input channels `3`, the two conv layers
producing `32` and then `64` channels, `10` classes, and a `32×32` image. Call
the batch size `N`.

1. **Input:** `(N, 3, 32, 32)` — `N` images, 3 color channels, 32×32 pixels.
2. **conv1 (3×3, pad 1, 32 out-channels):** padding keeps the spatial size, and
   we now have 32 channels instead of 3 → `(N, 32, 32, 32)`.
3. **ReLU:** elementwise, shape unchanged → `(N, 32, 32, 32)`.
4. **maxpool (2×2, stride 2):** halves height and width → `(N, 32, 16, 16)`.
5. **conv2 (3×3, pad 1, 64 out-channels):** padding keeps the spatial size, 64
   channels out → `(N, 64, 16, 16)`.
6. **ReLU:** shape unchanged → `(N, 64, 16, 16)`.
7. **maxpool (2×2, stride 2):** halves each axis again → `(N, 64, 8, 8)`.
8. **flatten:** collapse the per-image `(64, 8, 8)` block into one vector of
   `64 * 8 * 8 = 4096` numbers → `(N, 4096)`.
9. **Linear → logits:** multiply by a `(4096, 10)` weight matrix and add a
   `(10,)` bias → `(N, 10)`, one score per class for each image.

Two pools, each halving each axis, take a 32×32 image down to 8×8. That is why
the flattened Linear input is `c2 * (H/4) * (W/4)` in general — `c2` is the
second conv's channel count, and `H/4`, `W/4` are the height and width after two
halvings. For the full config that is `64 * 8 * 8 = 4096`; for the *tiny* config
the grader uses (channels `[2, 3]`, an 8×8 input), it is `3 * 2 * 2 = 12`. If you
can reproduce this shape walk for any config, the rest of the assembly follows.

## Weight sharing changes only one thing in backward

The forward pass is just calling the L11 and L9 primitives in order. The backward
pass is where assembly demands care, and there is exactly one place where a CNN's
gradient differs in kind from an MLP's: **weight sharing**.

A single filter `W[oc]` (the weights for output channel `oc`) is applied at
*every* spatial position of *every* image in the batch. The chain rule says: when
one parameter influences the loss through many separate uses, its total gradient
is the **sum** of the contributions from all those uses. So a filter's gradient
sums over the whole batch and over every output position it touched:

```text
dW[oc] = sum over (n, i, j) of   dout[n, oc, i, j] * patch(n, i, j)
db[oc] = sum over (n, i, j) of   dout[n, oc, i, j]
```

Read aloud: *the gradient of filter `oc` is, for every image `n` and every output
row `i` and column `j`, the upstream gradient at that spot times the input patch
the filter saw there, all added up.* Here `dout` is the gradient flowing back
into the conv's output, and `patch(n, i, j)` is the receptive-field patch the
filter multiplied during forward. The bias gradient `db[oc]` is the same sum
without the patch. The crucial consequence: `dW` comes out with the filter's own
shape `(C_out, C_in, kH, kW)` — **no batch axis**. If you forget the batch sum,
`dW` arrives with a stray leading axis of size `N`, and the gradient check fails.
This is exactly what `conv2d_backward` does, and what the `weight_sharing` hint
ladder addresses.

Max pooling's backward is the other stage with a twist, and it is a twist of the
opposite kind: pooling has *no weights*, so its backward is pure **routing**.
During forward, each 2×2 window kept only its single largest cell. During
backward, the upstream gradient for that window flows back to *only that one
winning cell* — the one that was the max — and every other cell in the window
gets exactly zero. The intuition: only the cell that actually contributed to the
output could have affected the loss; nudging a loser changes nothing, so its
gradient is zero.

The classic mistake is to spread the upstream gradient *uniformly* across all
four cells of the window instead of routing it to the single argmax. That is
wrong, and the grader is built to prove it: it feeds a uniform-spread maxpool
backward into the end-to-end gradient check and confirms the check *fails*,
demonstrating the check is sharp enough to catch the bug in your version too.

## Threading the chain through the whole stack

With those two special cases understood, the backward pass is mechanical: walk
the forward stages in reverse, and at each stage feed the previous stage its
output-gradient. Here is the full sequence as numbered steps. The naming
convention is `d<thing>` = "the gradient of the loss with respect to `<thing>`,"
so `dlogits` is the gradient coming in at the logits, `dflat` is the gradient at
the flattened vector, and so on.

1. **Linear layer.** From `dlogits` (shape `(N, 10)`), compute the Linear's own
   parameter gradients and pass the gradient back to the flattened vector:
   `dWfc = flat.T @ dlogits`, `dbfc = dlogits.sum(0)`, and
   `dflat = dlogits @ Wfc.T`.
2. **Un-flatten.** Reshape `dflat` back into the shape of `p2`, the second pool's
   output: `dp2 = dflat.reshape(p2.shape)`.
3. **Second pool (route).** `da2 = maxpool2d_backward(dp2, a2, 2, 2)` — send each
   gradient to the argmax cell of its window. (`a2` is the pre-pool activation
   that was cached in forward.)
4. **Second ReLU (mask).** ReLU passed only positive values, so its backward
   keeps the gradient only where the input was positive:
   `dz2 = da2 * (z2 > 0)`. The expression `(z2 > 0)` is a 0/1 mask of where the
   pre-activation was positive.
5. **Second conv.** `dp1, dW2, db2 = conv2d_backward(dz2, p1, W2, 1, 1)` — this
   returns the conv's parameter gradients *and* the gradient handed back to the
   first pool's output, `dp1`. Remember the batch sum lives inside here.
6. **First pool (route).** `da1 = maxpool2d_backward(dp1, a1, 2, 2)`.
7. **First ReLU (mask).** `dz1 = da1 * (z1 > 0)`.
8. **First conv.** `_, dW1, db1 = conv2d_backward(dz1, X, W1, 1, 1)` — the first
   conv's input is the original image `X`; we don't need the gradient w.r.t. the
   image, so we discard it.

After step 8 you hold a gradient for every parameter: `dW1, db1, dW2, db2, dWfc,
dbfc`. Store all six in `self.grads`, keyed by name. A missing key means a stage
of the chain was skipped, and the grader names exactly which parameter's gradient
is wrong, so you know which stage of the reverse walk to re-check.

## What the first-layer filters learn

A trained conv's first-layer filters are themselves small images, so you can
literally look at them. Take `W1[oc]` — the weights of one output channel, shape
`(3, 3, 3)` for a 3-channel input — normalize its values into the `[0, 1]`
display range, and show it as a tiny color image. Trained on real photos, the
first layer reliably learns **oriented edge detectors** and **color-blob
detectors**: the same low-level primitives a biological visual system starts
with. Deeper layers compose these into textures and object parts.

You will *not* see crisp edges from the grader, because the grader trains a tiny
network on a fixed batch for only a couple of steps — far too little to learn
anything legible. The full-config run in the experiments trains long enough to
get there. The conceptual takeaway is the one that matters: weight sharing means
each of those filters is *one* reusable feature detector, and it is applied at
every position of the image.

## Exercise

Fill in `my_work/L13-cnn-cifar10/cnn_cifar10.py` (run `uv run grade start L13-cnn-cifar10` to
get the scaffold). This module is fully self-contained: inline your own
primitives, with no cross-lesson imports — `numpy` only.

The conventions, inherited from L11 and used everywhere below: an input batch `X`
has shape `(N, C_in, H, W)`; a conv weight tensor `W` has shape
`(C_out, C_in, kH, kW)`; a conv bias `b` has shape `(C_out,)`. Output spatial
size on one axis is `H_out = (H + 2*padding - kH) // stride + 1`.

- `im2col(Xp, kH, kW, stride, H_out, W_out)` — the **warm-up**, recalled cold
  from L11 (the grader checks it on its own). Parameters: `Xp` is the
  already-padded input, shape `(N, C_in, H_pad, W_pad)`; `kH, kW` are the kernel
  height and width; `stride` is the step between windows; `H_out, W_out` are the
  output spatial sizes you computed. Returns the patch matrix of shape
  `(N, C_in*kH*kW, H_out*W_out)` — one column per output position, each column
  the flattened receptive-field patch. This is the trick that turns a convolution
  into a single matmul.
- `conv2d_forward(X, W, b, stride, padding)` / `conv2d_backward(dout, X, W,
  stride, padding)` — the convolution from L11. Forward returns the output of
  shape `(N, C_out, H_out, W_out)`. Backward takes `dout` (the gradient of the
  loss w.r.t. that output, same shape) and returns `(dX, dW, db)`: the gradient
  w.r.t. the input, the weights, and the bias. **`dW` must sum over the batch** —
  it has shape `(C_out, C_in, kH, kW)` with no batch axis; `db` sums over batch
  and both spatial axes to shape `(C_out,)`.
- `maxpool2d_forward(X, pool_size, stride)` / `maxpool2d_backward(dout, X,
  pool_size, stride)` — 2×2 max pooling. Forward returns `(out, cache)` where
  `out` has the down-sampled shape and `cache` is what backward needs (the input
  `X`). Backward routes each entry of `dout` to the argmax cell of its window and
  returns `dX` of the input's shape, zeros everywhere except the winners.
- `softmax_cross_entropy(logits, labels)` — the L9 loss, numerically stable.
  Parameters: `logits` of shape `(N, K)` (`K` classes); `labels` of shape `(N,)`,
  integer class indices. Subtract the per-row max before `exp` to avoid overflow,
  average the per-example loss over `N`, and return `(loss, dlogits)` where
  `dlogits = (softmax - one_hot) / N` has shape `(N, K)`.
- `adam_step(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8)` — one
  bias-corrected Adam step from L10. Parameters: `params` and its matching
  `grads` (same shape); `state` is `{"m": array, "v": array, "t": int}` starting
  at zeros and `t=0`; `lr` is the learning rate. Returns `(new_params,
  new_state)`.
- `cnn_param_count(in_channels, conv_channels, n_classes, input_hw)` — return the
  total scalar parameter count for the fixed architecture from its config.
  `conv_channels` is the list `[c1, c2]`; `input_hw` is `(H, W)`. Count conv1
  (`c1*in*9 + c1`), conv2 (`c2*c1*9 + c2`), and the Linear
  (`c2*(H//4)*(W//4)*n_classes + n_classes`), then sum.
- The `CNN` class — constructed as `CNN(in_channels, conv_channels, n_classes,
  input_hw, rng)`, where `rng` is a NumPy random generator used for weight
  initialization. Implement:
  - `forward(X)` — `X` of shape `(N, in_channels, H, W)` → logits `(N,
    n_classes)`. Chain conv → ReLU → pool → conv → ReLU → pool → flatten →
    Linear, caching every intermediate `backward` will need.
  - `backward(dlogits)` — thread the chain rule back through every stage (the
    eight numbered steps above), **set `self.grads` for every parameter** (`W1,
    b1, W2, b2, Wfc, bfc`), and return `self.grads`.
  - `params()` — return a dict mapping each parameter name to its tensor.
  - `param_count()` — return the total scalar count (must match
    `cnn_param_count`).

Then grade it: the site's Run Grader button or `uv run grade L13-cnn-cifar10`.

## How the grader checks you

The grader builds a *tiny* fixed instance of this same architecture for speed —
`in=3`, `conv_channels=[2, 3]`, `n_classes=4`, an `8×8` input, and a fixed
`(4, 3, 8, 8)` batch in float64 — then runs five checks:

- **`im2col` warm-up.** On a fixed 1-image, 1-channel 4×4 padded tensor with a
  3×3 kernel and stride 1 (a 2×2 output), it confirms the output shape is
  `(1, 9, 4)` and that column 0 is the top-left 3×3 patch flattened and column 3
  is the bottom-right patch. This is graded on its own, so a wrong warm-up never
  hides a working network.
- **Forward shape.** `forward` on the fixed batch must return logits of shape
  `(4, 4)` — `(N, n_classes)`. Every assembled check below rides on this, so a
  broken forward shape stops the rest early.
- **End-to-end gradient check.** This is the heart of the lesson. For *every*
  parameter tensor, the grader compares your analytic gradient from `backward`
  against a **centered finite-difference** gradient of a fixed scalar loss —
  nudge a parameter entry up by a tiny `eps`, nudge it down, and divide the
  change in loss by `2*eps`. Because a full-tensor difference would be slow, it
  samples a seeded random subset of up to 6 entries per tensor and requires them
  to agree to a tolerance of `1e-4`. If a tensor's gradient is wrong, the failure
  *names that tensor*, telling you which stage of the reverse walk broke.
- **Train step decreases loss.** Two Adam steps on the fixed batch must lower the
  loss. A non-decreasing loss points at a sign error in `backward` or a wrong key
  mapping into `adam_step`.
- **Param-count formula.** `param_count()` must equal `165` for the tiny config:
  conv1 `2*3*9 + 2 = 56`, conv2 `3*2*9 + 3 = 57`, Linear
  `(3*(8//4)*(8//4))*4 + 4 = 52`, summing to `165`. A wrong total usually drops a
  bias term or mis-sizes the flattened Linear input `c2 * (H/4) * (W/4)`.

This lesson has **no named trap** in `grader/traps.yaml` (the file ships an empty
trap list on purpose). Every primitive here was already trap-tested back in L10
and L11, so there is no whole-module buggy version to re-catch. The one
assembly-specific failure mode worth proving — a max-pool backward that spreads
gradient uniformly instead of routing to the argmax — is exercised *inside* the
grader as a fifth check: it patches in the wrong max-pool backward and confirms
the same finite-difference gradient check flags it on at least one weight that
flows through a pool. That is the safety proof that your gradient check is sharp
enough to catch your own mistakes.

## The hint ladder

If the grader's message is not enough, open the hints. They are grouped by
failure mode — `assembled_gradient_check` for a chain-order or hand-off bug,
`weight_sharing` for a `dW` with a stray batch axis, `maxpool_routing` for a pool
backward that spreads instead of routes — and each has three levels: a conceptual
nudge, then the formula with shapes, then pseudocode. Try level 1 before
climbing. When the gradient check names a single parameter, that name tells you
where in the reverse walk to look: a grad wrong only on `W1`/`b1` means the error
is early in the backward order (the *last* layers you touch); wrong only on `Wfc`
means the Linear step.

## Done when

`uv run grade L13-cnn-cifar10` shows all-green: the `im2col` warm-up passes on its own,
`forward` returns the right logits shape, your `backward` matches the
finite-difference gradient to `1e-4` on every parameter tensor, two Adam steps
decrease the loss, `param_count()` returns `165` for the tiny config, and the
built-in break-it check confirms a uniform-spread max-pool would be flagged. Then
run the experiments and record your predictions.
