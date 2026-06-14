# L7 — Backprop on Paper

In M2 you computed gradients for one-layer models — linear regression, the
perceptron, the SVM. A **gradient** is the vector of slopes that says, for each
knob (each weight), which way to nudge it to make the loss go down, and how
steeply. In those models the loss depended on the weights *directly*: change a
weight, and you could read off its effect on the loss in one short formula.

A neural network stacks layers. The word **layer** just means a row of small
computing units; the output of one row becomes the input of the next. So a weight
in the first layer no longer touches the loss directly — it reaches the loss only
through a *chain* of intermediate values, each of which feeds the next.
**Backpropagation** (or "backprop") is the bookkeeping that recovers the gradient
anyway. The good news: there is **nothing new** to learn. Backprop is the chain
rule from L1, applied one node at a time. (The **chain rule** is the calculus fact
that if `a` affects `b` and `b` affects `c`, then `a`'s effect on `c` is the
*product* of the two intermediate effects — `da→c = da→b × db→c`. We unpack it
fully below.)

This lesson does backprop on paper — a small network, with exact numbers — so that
when L8 builds an automatic gradient engine and L9 vectorizes it, you already know
exactly what those machines are computing.

The network is a **2-2-1 MLP**. "MLP" stands for *multi-layer perceptron*, the
plain name for a stack of these unit-rows; "2-2-1" reads as the sizes of the
rows: two inputs, then two hidden units, then one output unit. A **hidden unit**
is a unit that is neither an input nor the final answer — it sits in the middle,
so we never see its value directly. Our two hidden units each apply `tanh`
(introduced below); the one output unit is **linear**, meaning it just adds up its
inputs with no squashing function on top. The loss is **half-squared-error**:
half of the squared gap between the prediction and the target, i.e.
`0.5*(output - y)**2` (the `0.5` earns its keep in the backward pass — you will
see why). Parameters are named by a fixed contract used across L7–L9: weights
`w1..w6`, biases `b1..b3`. A **weight** scales an input; a **bias** is an added
constant that shifts a unit's value. The wiring is: `w1,w2` feed hidden unit 1,
`w3,w4` feed hidden unit 2, `w5,w6` feed the output, and `b1,b2,b3` are the three
units' biases.

## The chain rule as a path of gates

Picture one parameter and the loss connected by a path of nodes — a **node** being
any single arithmetic step (an add, a multiply, a `tanh`). Nudge the parameter by
a hair. The nudge propagates forward along the path, and at each node it gets
scaled by that node's own *local* sensitivity — how much that node's output moves
when its input moves by a hair. By the time the nudge arrives at the loss, it has
been multiplied by every local scaling on the way. The total sensitivity — the
parameter's gradient — is the **product** of all those local scalings along the
path. That product *is* the chain rule, read forward.

Two pieces of vocabulary make the backward direction sayable:

- A **local gradient** is one node's own slope: how much *its* output changes per
  unit change in *its* input, computed using only what that node knows. The `tanh`
  node's local gradient is `1 - h**2`; an add node's local gradient is `1`.
- An **upstream gradient** is the gradient that has already been accumulated from
  the loss *down to this node's output* — everything the chain rule has multiplied
  in so far, arriving from the loss side.

Backprop reads the chain rule *backward*. Start at the loss with its sensitivity
to itself, which is `1` (the loss changes one-for-one with itself). Then walk back
toward each parameter, and at every node multiply the upstream gradient by that
node's local gradient. The running product you carry backward is the gradient.

### Chain rule gates

A **gate** is just a node viewed as a little gradient-routing device: signal
flows in on the forward pass, gradient flows back out on the backward pass, and
the gate has one fixed rule for how it transforms the gradient passing through it.
Every node in this network is one of exactly three gates, and you only need to
memorize their local rules:

- **Add gate** — `z1 = (w1*x1 + w2*x2) + b1`. Its local gradient is `1`, so an add
  gate passes the upstream gradient straight through to each input, unchanged. A
  bias therefore just *copies* its branch's gradient: `db1 = dz1`. Addition
  distributes gradient; it never scales it.
- **Multiply gate** — `output = w5*h1 + ...`. The gradient to one factor is the
  upstream gradient times the **other** factor. So `dw5 = dout * h1` (the weight
  hears the activation it multiplied) and `dh1 = dout * w5` (the activation hears
  the weight). A multiply gate *swaps* its inputs onto the gradient.
- **tanh nonlinearity** — `h1 = tanh(z1)`. `tanh` is the hyperbolic-tangent
  squashing function: it bends any real number into the range `(-1, 1)`, and that
  bend is what lets the network represent curved decision boundaries. As a unary
  gate (one input, one output) it scales the gradient by its local slope. For
  `tanh`, that slope is `1 - tanh(z)**2`, which equals `1 - h**2` since `h` is
  already `tanh(z)`. So `dz1 = dh1 * (1 - h1**2)`. This factor is the one learners
  most often drop — and it is the named trap of this lesson.

The notation reads: `dw5` is shorthand for `∂loss/∂w5`, "the partial derivative of
the loss with respect to `w5`" — the slope of the loss as you wiggle only `w5` and
hold everything else fixed. The leading `d` always means "gradient of the loss
with respect to this thing." Note what we are differentiating: always the *loss*
with respect to a *parameter*, never the model itself. The gradient answers "which
way do I turn this knob to shrink the loss," which is a fact about the loss.

Memorize those three gates and backprop is just walking the graph backward,
applying one rule per node. The **locality** lesson falls right out: a parameter
gets a gradient only along paths the input actually flows through. We will see
`dw1 = dw3 = 0` below, purely because `x1 = 0` on our test point — no flow, no
gradient.

## A worked example, in exact numbers

We evaluate at one fixed point so every quantity is a concrete float you can
check by hand. The init (the starting values of the nine parameters) is seeded —
it lives in `references/xor_reference.py`, the single source of truth this lesson
is graded against — and we use seeded random values rather than round numbers so
a rounding bug can't sneak past the grader. The input is the XOR sheet point
`x1 = 0, x2 = 1, y = 1` (one row of the four-row XOR truth table this network is
eventually meant to fit). The nine parameters at this seed are:

```text
w1 =  0.01913   w2 =  0.23697   w3 = -0.06887
w4 = -0.69467   w5 =  1.26006   w6 = -0.50320
b1 =  0.92841   b2 = -1.25119   b3 =  0.07417
```

### Forward pass: compute the nodes left to right

The **forward pass** computes the network's output by evaluating every node in
order, inputs first, loss last. Each value below is labeled by its node name, and
each line plugs in the numbers from the line above.

1. `z1 = w1*x1 + w2*x2 + b1 = 0.01913*0 + 0.23697*1 + 0.92841 = 1.16538`
   (hidden-1 pre-activation — the affine combination before `tanh`).
2. `z2 = w3*x1 + w4*x2 + b2 = -0.06887*0 + (-0.69467)*1 + (-1.25119) = -1.94586`
   (hidden-2 pre-activation).
3. `h1 = tanh(z1) = tanh(1.16538) = 0.82278` (hidden-1 activation).
4. `h2 = tanh(z2) = tanh(-1.94586) = -0.96000` (hidden-2 activation).
5. `output = w5*h1 + w6*h2 + b3 = 1.26006*0.82278 + (-0.50320)*(-0.96000) + 0.07417 = 1.59399`
   (the **linear** output unit — note: no `tanh` here).
6. `loss = 0.5*(output - y)**2 = 0.5*(1.59399 - 1)**2 = 0.17641` (half-squared-error).

Notice that `x1 = 0` already simplifies things: in steps 1 and 2 the `w1*x1` and
`w3*x1` terms vanish, so `z1` and `z2` get nothing from those weights. That is a
foreshadow of the locality result below.

### Backward pass: propagate gradients right to left

The **backward pass** walks the same nodes in *reverse* — loss first, parameters
last — carrying the gradient backward and applying one gate rule per node. We
start at the loss, whose sensitivity to itself is `1`, and immediately differentiate
the loss with respect to its input `output`:

1. `dout = output - y = 1.59399 - 1 = 0.59399`. This is `∂loss/∂output`. The `0.5`
   in the half-squared-error cancels the `2` that the squared exponent brings down
   when you differentiate — `d/du [0.5*(u-y)^2] = (u-y)` — which is exactly why we
   put the `0.5` there: it makes this first step clean.
2. **Output gate** (multiply for the weights, add for the bias). Each weight's
   gradient is `dout` times the *other* factor it multiplied:
   `dw5 = dout*h1 = 0.59399*0.82278 = 0.48873` and
   `dw6 = dout*h2 = 0.59399*(-0.96000) = -0.57023`. The bias is an add gate, so it
   copies `dout` straight through: `db3 = dout = 0.59399`.
3. **Gradient into the hidden activations** (multiply gate, other factor again):
   `dh1 = dout*w5 = 0.59399*1.26006 = 0.74846` and
   `dh2 = dout*w6 = 0.59399*(-0.50320) = -0.29889`. These are upstream gradients
   for the two `tanh` gates.
4. **tanh gates** — scale each upstream gradient by the local slope `1 - h**2`:
   `dz1 = dh1*(1 - h1**2) = 0.74846*(1 - 0.82278**2) = 0.74846*0.32303 = 0.24177`
   and
   `dz2 = dh2*(1 - h2**2) = -0.29889*(1 - (-0.96000)**2) = -0.29889*0.07841 = -0.02344`.
   The `1 - h**2` factor is the `tanh` derivative — drop it and every downstream
   gradient on that branch is wrong.
5. **Input gates** (multiply for the weights, add for the biases):
   `dw1 = dz1*x1 = 0.24177*0 = 0`, `dw2 = dz1*x2 = 0.24177*1 = 0.24177`,
   `db1 = dz1 = 0.24177`; and on the other branch
   `dw3 = dz2*x1 = -0.02344*0 = 0`, `dw4 = dz2*x2 = -0.02344*1 = -0.02344`,
   `db2 = dz2 = -0.02344`.

**The locality lesson, in numbers.** `dw1 = dz1*x1` and `dw3 = dz2*x1`, and
`x1 = 0`, so `dw1 = dw3 = 0` — *exactly* zero. Those two weights multiply an input
that is off on this example, so this example tells us nothing about how to change
them. A gradient flows only along a path the input actually traverses. (Flip `x1`
on and they come alive — that is Experiment 1.)

One more habit worth naming: this whole worked pass is the chain rule on a
*graph*, and the words "forward" and "backward" are layered-network terminology —
they mean "compute outputs from inputs to loss" and "propagate gradients from loss
to inputs." For a one-layer model (like L2's linear regression) there is no chain
to walk, so there is no "direction" to speak of; the two passes only become a
thing once values feed values.

## Why paper, not just code

The analytic gradient you just computed by hand — the exact, formula-derived one —
is what drives every training step in a real network: it is fast and exact. Backprop
is simply the efficient procedure for evaluating it across a whole graph. The
*numerical* gradient (the finite-difference estimate from L1) plays a different
role: it is a slow, per-parameter sanity check you run during debugging. To check
one parameter numerically you must run the entire forward pass twice (once nudged
up, once nudged down) — so checking all nine here costs eighteen forward passes,
versus one backward pass for the analytic version. That cost is *per parameter*,
not per data point, which is why nobody trains with it. But it is the gold check:
if your hand-derived gradient and the finite-difference estimate disagree, one of
them is wrong, and that is the whole point of re-implementing `numerical_gradient`
as a warm-up below.

## Exercise

Implement in `my_work/backprop-paper/backprop_paper.py` (run
`grade start backprop-paper` to get the scaffold). All values are scalars and all
arithmetic is `float64`.

- **Spaced-retrieval warm-up, do this first, from memory (2–5 min):**
  `numerical_gradient(f, x, h=1e-5)` — re-implement L1's centered finite-difference
  gradient *cold*, no peeking. Parameters: `f` is a scalar function that takes a
  1-D parameter array `x` and returns one number; `x` is the point, shape `(d,)`,
  where you want the slope (`d` is the number of parameters); `h` is the tiny step
  size (`float64`). Perturb **one** coordinate at a time, up by `+h` and down by
  `-h`, and read the symmetric slope `(f(x+h) - f(x-h)) / (2h)`. Returns `grad`,
  shape `(d,)` — one partial derivative per input coordinate. It is graded as an
  independent unit (a broken warm-up never blocks the backprop chain), and it is
  the check that makes your analytic backprop trustworthy.
- `forward_pass(w1, w2, w3, w4, w5, w6, b1, b2, b3, x1, x2, y)` — the forward pass
  of the 2-2-1 MLP. Parameters: `w1..w6` are the six weights; `b1..b3` the three
  biases; `x1, x2` the two inputs; `y` the target. Compute each node in order —
  `z1, z2` (affine), `h1, h2` (`tanh`), `output` (LINEAR, no `tanh`), `loss`
  (half-squared-error). Returns a dict with keys `z1, z2, h1, h2, output, loss` —
  return **every** intermediate, because the backward pass reuses `h1, h2, output`.
- `backward_pass(w1, w2, w3, w4, w5, w6, b1, b2, b3, x1, x2, y)` — the analytic
  gradient of the loss with respect to every parameter, by the chain rule, gate by
  gate. Same parameters as `forward_pass`. Run the forward pass to get
  `h1, h2, output`, then walk the gates backward (`dout`, then the output
  multiply/add gates, then `dh1, dh2`, then the two `tanh` gates `dz1, dz2`, then
  the input multiply/add gates). Returns a dict with keys `dw1..dw6, db1..db3` —
  one gradient per parameter.

Then grade it: the site's Run Grader button or `grade backprop-paper`.

## How the grader checks you

- The **warm-up** `numerical_gradient` is graded on its own against
  `f(x) = sum(x**2)`, whose exact gradient is `2x` everywhere. The grader checks
  both the output shape (one slope per coordinate) and that the values match `2x`
  to about `1e-6`. A wrong warm-up never hides a working forward/backward pass.
- `forward_pass` is checked node by node against the reference, at the seeded init
  and the XOR point, to a relative tolerance of `1e-7`. Each node's message names
  the node and recalls its formula, so a wrong number points at the **first** node
  where your numbers diverge — that node, not the loss, is where your formula is
  off.
- `backward_pass` is checked parameter by parameter against the reference, also to
  `1e-7`. Each gradient's message names the parameter and recalls its chain-rule
  step (including the `1 - h**2` `tanh` factor), so a mismatch tells you exactly
  which gate to re-derive. This check depends on `forward_pass` passing first — the
  chain rule cannot be right if the intermediates it reuses are wrong.

### The named trap: dropping the tanh derivative

This grader is built to catch one specific mistake by name. The chain rule on a
hidden branch must pass the gradient *through* the `tanh` gate's local slope before
it reaches the input weights: `dz = dh * (1 - h**2)`. The common slip is to write
`dz2 = dh2` — dropping the `(1 - h2**2)` factor — which leaves `dw3, dw4, db2` all
off by exactly that factor while the `z1` branch (which kept its factor) still
looks fine. The grader recognizes this and its message says the cause is the
missing **tanh derivative**, so you fix the right gate instead of hunting a phantom
bug elsewhere.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels, mildest first: a conceptual nudge, then the formula with the gate it lives
on, then pseudocode. Try level 1 before climbing — and if your numbers desync from
the paper trace, the hints point you back to the first node that disagrees with the
reference.

## Done when

`grade backprop-paper` shows all-green: the warm-up `numerical_gradient` recovers
`2x` to about `1e-6` with the right shape; `forward_pass` matches every reference
node to `1e-7`; and `backward_pass` matches every reference gradient to `1e-7`,
including the two that are *exactly* zero (`dw1, dw3` at `x1 = 0`) and the `tanh`
branches with their `1 - h**2` factor intact. Then run the experiments and record
your predictions.
