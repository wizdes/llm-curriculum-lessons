# L8 — The Autograd Engine

In L7 you ran **backprop** by hand on one fixed network. (Backprop, short for
*backpropagation*, is the procedure that computes the gradient of the loss with
respect to every parameter by applying the chain rule from calculus.) The network
was a 2-2-1 **MLP** — a *multi-layer perceptron*, the plain feed-forward neural
net you built in L7: two inputs, two hidden units, one output. You did the forward
pass to get the loss, then walked the gates backward, multiplying in one local
derivative per gate until you had a gradient for every weight.

That was tedious precisely *because* it was mechanical — and anything mechanical
can be automated. This lesson builds the machine: a tiny **autograd engine** (an
*automatic-differentiation* engine — code that computes exact gradients for you)
in the style of Andrej Karpathy's *micrograd*. It runs the exact same chain rule
for *any* expression you write, with no hand derivation. L9 will then vectorize
it; here we keep everything scalar — one float per node — so you can see every
gear turn.

The whole engine is one class, `Value`. A `Value` wraps a single number and
remembers how that number was made, so the engine can later retrace the steps in
reverse and hand back gradients. Two ideas make it work: **fan-out accumulation**
and **reverse topological ordering**. Get those right and the rest is just the
per-gate local derivatives you already memorized in L7.

One framing to carry through. We always differentiate the **loss** with respect
to the **parameters** — never "the model with respect to" anything. The engine
computes, for each parameter, how much the final loss would change if you nudged
that parameter a hair. That number is its **gradient**, and gradient descent (L1)
uses it to step the parameter downhill.

## What a Value node is and how the graph gets built

A `Value` is a node that remembers its parents.

Every number in your computation becomes a `Value`. A `Value` carries four things:

- `data` — the float it holds (the actual number, like `2.0`).
- `grad` — its gradient, *how much the final loss changes per unit change in this
  node's `data`*. It starts at `0.0` and gets filled in during the backward pass.
- `_prev` — the set of `Value`s that fed into this one (its parents in the graph).
- `_backward` — a small function (a *closure*: a function that remembers the
  variables around where it was defined) that knows how to push gradient from this
  node back to its parents.

So an expression like `d = b * c` does more than compute `b.data * c.data`. It
builds a new node `d` whose `_prev` points back at `b` and `c`, and whose
`_backward` knows the multiply rule. Do this for every operation and your whole
computation becomes a **computational graph**: a network of `Value` nodes wired
parent-to-child, with the input numbers as the leaves and the loss as the single
node at the top.

That graph is a **DAG** — a *directed acyclic graph*. "Directed" because each edge
points one way (from a parent that fed in, to the result it helped produce);
"acyclic" because you can never follow the arrows in a loop back to where you
started (a number can't be its own ancestor). Read "DAG" aloud as "dag." The leaves
are your inputs and parameters; the root is the loss.

The clever part is that each operation also stashes its `_backward` closure. Each
closure knows how to push gradient from a result back to its inputs using *that
gate's local derivative* — the small calculus fact for that one operation. These
are the L7 gates, unchanged:

- **add** copies the upstream gradient unchanged to each input,
- **multiply** sends the upstream gradient times the *other* factor,
- **tanh** scales the upstream gradient by `1 - h**2` (where `h` is the tanh
  output), **relu** passes the gradient through only where the input was positive,
- **pow** (raising to a fixed power `n`) scales by `n * x**(n-1)`.

("Upstream gradient" means `out.grad`, the gradient already sitting on the result
node — the gradient flowing *down* toward the inputs during the backward pass.)

With the graph built and every node carrying its own `_backward`, the entire
backward pass reduces to one question: *fire these closures in the right order.*
That ordering is the rest of the lesson.

### Fan-out accumulation

The rule is `+=`, never `=` — and here is why it trips up almost everyone.

A `Value` can feed *multiple* consumers — it **fans out**. "Fan-out" means one
node's output is used by several later operations, the way one wire can split to
power several lights. In `b = a + a`, the single node `a` is *both* inputs of the
add. In the L7 network, each input feeds *both* hidden units (`x1` flows into the
sums for `z1` and `z2`), and on the way back the output node's gradient `dout`
splits and flows into `w5`, `w6`, `b3`, `h1`, and `h2` all at once.

When a node fans out, gradient flows back to it along *each* of those paths. The
**multivariable chain rule** — the version of the chain rule for a quantity that
affects the output through several routes — says the total gradient of a fan-out
node is the **sum** of the contributions along every outgoing path. Intuitively:
if nudging `a` changes the loss through three different channels, the total effect
is all three added up, not just one of them.

Here is the worked case. Take `b = a + a`, so `b = 2a` and the true gradient is
`db/da = 2`. The add gate copies `b.grad` to *each* of its two inputs — but both
inputs *are* `a`. So `a` should receive `b.grad` twice and end at `2`. The only
way to get that is to **accumulate**: each `_backward` *adds* its contribution
into the parent's `grad` rather than overwriting it.

```python
self.grad += local_derivative * out.grad   # += , not =
```

Read that aloud: "add (the local derivative times the upstream gradient) onto
this node's running gradient total." `grad` starts at `0.0`, so the first `+=` is
clean, and each later consumer adds its share on top.

Write `=` instead of `+=` and you keep only the *last* path's contribution — every
fan-out node is silently undercounted. For `b = a + a` the correct `a.grad = 2`
collapses to `a.grad = 1`: the second copy overwrites the first. This is the
**named trap** of the lesson, and the grader checks `b = a + a` directly — it
flags the message about `+=` the instant you assign instead of accumulate.

### Reverse topological ordering

A node's gradient is not final until *every* consumer has pushed into it. That
follows directly from fan-out accumulation: if `a` feeds three nodes, `a.grad` is
only correct once all three have deposited their share. So you cannot fire
`a`'s `_backward` until everything `a` feeds into has already run.

The fix is to process nodes in **reverse topological order**. A **topological
order** of a DAG is a list where each node appears only *after* all of its
parents — the nodes that feed *into* it — have been placed. So the leaves (inputs
and parameters) come first and the loss comes last. "Topological" just means
"respecting the wiring": the list never puts a node ahead of something it depends
on. Now walk that list *backward* — loss first, leaves last — and you are
guaranteed that by the time you reach any node, every one of its consumers (which
sit *after* it in the list) has already run and deposited gradient. Read "reverse
topological order" aloud as "loss first, then always finish a node's consumers
before the node itself."

A **depth-first post-order** traversal over `_prev` produces exactly this
topological order. "Depth-first" means you dive all the way down a branch before
backing up; "post-order" means you record a node only *after* you have recursed
into its parents — which is why a node always lands *after* its parents in the
list. The recipe is in the next section.

If you traverse in *forward* order instead — loss last — you read a parent's
`grad` before its children have finished depositing into it, and the gradients
come out wrong. The grader's **diamond-DAG** test (one node that fans out three
ways, explained in the exercise) is built to catch exactly this ordering bug.

## The backward() algorithm, step by step

`backward()` is called on the loss node and computes the gradient of the loss with
respect to every node feeding into it. Here is the full procedure as numbered
steps:

1. **Build the topological order.** Start a depth-first post-order walk from the
   loss. For each node not yet visited, mark it visited, recurse into each of its
   parents in `_prev`, and *then* append the node to a list `topo`. Because you
   append a node only after recursing into its parents, every node lands *after*
   its parents — so `topo` runs leaves first and the loss last, exactly a
   topological order.
2. **Seed the output gradient.** Set the loss node's `grad` to `1.0`. This says
   "the loss's sensitivity to itself is 1" — nudge the loss by one unit and it
   changes by one unit. It is the starting value that the chain rule multiplies
   everything else against.
3. **Replay the closures in reverse.** Walk `topo` *backward* (`reversed(topo)`,
   loss first). For each node, call its `_backward`. That closure reads the node's
   own `grad` (its upstream gradient, now final because every consumer already
   ran) and **accumulates** the local-derivative contribution into each parent's
   `grad` with `+=`.

When the loop finishes, every node's `grad` holds the exact gradient of the loss
with respect to that node — the same numbers you derived by hand in L7, computed
automatically.

## A walk through the Value class

The worked `Value` (in `worked/autograd_engine.py`) is about 120 lines. Read it
gate by gate; every piece maps to something above.

- `__init__` stores `data`, sets `grad = 0.0`, installs the leaf default
  `_backward = lambda: None` (a do-nothing closure — correct for a leaf, which has
  no parents to push into), and records `_prev` (the parents) and `_op` (a string
  tag naming the operation, handy for debugging).
- Each operation (`__add__`, `__mul__`, `__pow__`, `tanh`, `relu`) builds the
  result `out` with the right parents, defines a closure that `+=`-accumulates
  `(local derivative) * out.grad` into each parent's `grad`, attaches it as
  `out._backward`, and returns `out`.
- The reverse and derived ops (`__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`,
  `__neg__`, `__sub__`, `__truediv__`) let you mix `Value`s with plain Python
  numbers in either order, so an expression like `0.5 * (output - y)**2` just
  works — Python falls back to `__rmul__` when the left operand (`0.5`) is a plain
  float.
- `backward()` runs the three numbered steps above: build `topo`, seed
  `self.grad = 1.0`, fire `_backward` over `reversed(topo)`.

The headline check: rebuild the L7 2-2-1 network out of `Value`s, call
`loss.backward()`, and every parameter's `.grad` matches the L7 hand-derived
reference (in `references/xor_reference.py`) to within `1e-7`. The machine
reproduces the paper.

One caution while reading. There is no separate "forward pass module" and
"backward pass module" here the way layered-network diagrams suggest. The forward
pass *is* just running ordinary Python arithmetic on `Value`s — and as a side
effect, that arithmetic builds the graph. The backward pass is one method call.
The "forward/backward" language is about *which direction gradient flows through
the graph*, not about two big phases of code you write by hand.

## Exercise

Implement your engine in `my_work/L09-autograd-engine/autograd_engine.py`. Run
`uv run grade start L09-autograd-engine` to drop the starter scaffold into place. Do the
spaced-retrieval warm-up first, then build `Value` one operation at a time,
deriving each gate's local derivative *before* you transcribe it. Keep the engine
self-contained: Python's `math` module only — no numpy, no torch. Climb the hint
ladder if you get stuck.

You implement two things: a warm-up function and the `Value` class.

- `forward_pass(w1, w2, w3, w4, w5, w6, b1, b2, b3, x1, x2, y)` — the **warm-up**,
  the L7 2-2-1 forward pass recalled *cold* in plain arithmetic (plain floats, no
  `Value`). Parameters: `w1`–`w6` are the six weights and `b1`–`b3` the three
  biases of the 2-2-1 MLP (all floats); `x1`, `x2` are the two input features and
  `y` the target label (floats). Compute `z1 = w1*x1 + w2*x2 + b1` and
  `z2 = w3*x1 + w4*x2 + b2` (the two hidden **pre-activations**, i.e. the affine
  sums before the nonlinearity), then `h1 = tanh(z1)`, `h2 = tanh(z2)` (the hidden
  activations), then `output = w5*h1 + w6*h2 + b3` (a **linear** output unit — no
  tanh), then `loss = 0.5 * (output - y)**2` (half-squared-error). Return a dict
  with keys `z1, z2, h1, h2, output, loss` (all floats). This warm-up is graded on
  its own and not used by the engine; re-authoring the forward pass from memory
  anchors the math before the gradient machinery.
- `Value(data, _children=(), _op="")` — the autodiff node. Parameters: `data` is
  the number it wraps (any int/float, stored as a float); `_children` is the tuple
  of parent `Value`s that produced it (empty for a leaf); `_op` is a short string
  naming the operation (for debugging only). It exposes `.data`, `.grad`
  (initialized `0.0`), and `.backward()`. Implement these operations on it, each
  returning a new `Value` whose `_backward` accumulates with `+=`:
  - `__add__(self, other)` — addition. The add gate copies `out.grad` to *both*
    inputs: `self.grad += out.grad`, `other.grad += out.grad`.
  - `__mul__(self, other)` — multiplication. Each input gets `out.grad` times the
    *other* factor: `self.grad += other.data * out.grad`, and symmetrically for
    `other`.
  - `__pow__(self, other)` — `self ** other` for an int/float exponent `other`.
    Local derivative `d/dx x**n = n * x**(n-1)`, so
    `self.grad += (other * self.data**(other-1)) * out.grad`.
  - `tanh(self)` — hyperbolic-tangent activation. With `t = tanh(self.data)`, the
    local derivative is `1 - t*t`, so `self.grad += (1 - t*t) * out.grad`.
  - `relu(self)` — rectified linear unit, `max(0, x)`. The output is `self.data`
    if positive else `0.0`; gradient passes only where the input was positive:
    `self.grad += (1.0 if out.data > 0 else 0.0) * out.grad`.
  - `backward(self)` — run the three-step algorithm above: build the topo order
    with a depth-first post-order helper over `_prev`, set `self.grad = 1.0`, then
    call each node's `_backward` over `reversed(topo)`. Returns nothing; it fills
    in every node's `.grad` in place.

  The reverse/derived ops (`__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`,
  `__neg__`, `__sub__`, `__truediv__`) are already written for you in the starter —
  they delegate to the ops above so you can mix `Value`s with plain numbers.

Then grade it: the site's Run Grader button or `uv run grade L09-autograd-engine`.

## How the grader checks you

- The **warm-up** `forward_pass` is checked against the L7 reference at the
  canonical XOR point (`x1=0, x2=1, y=1`) and the seeded init: every returned node
  (`z1, z2, h1, h2, output, loss`) must match to a relative tolerance of `1e-7`.
  It is graded on its own and never blocks the `Value` checks.
- The **headline reproduction**: the grader rebuilds the L7 2-2-1 network using
  *your* `Value`, calls `loss.backward()`, and checks all nine parameter gradients
  (`dw1`–`dw6`, `db1`–`db3`) against the hand-derived L7 reference to `1e-7`. A
  mismatch on one parameter points at the gate on that parameter's path.
- **Fan-out accumulation** (named trap): for `b = a + a`, it requires
  `a.grad == 2.0` exactly. This is the `+=`-vs-`=` check.
- **Diamond topo-sort** (named trap): it builds the diamond `a -> b`, `a -> c`,
  `d = b*c`, `e = d + a` at `a = 3` (so `b = 2a = 6`, `c = a+1 = 4`, `d = 24`,
  `e = 27`) and requires `a.grad == 15.0` — the leaf `a` fans out to `b`, `c`,
  *and* `e`, and only a reverse-topological backward pass gets it right.
- **Per-op checks vs torch**: each gate's gradient (`add`, `mul`, `pow`, `tanh`,
  `relu`) is compared against PyTorch at a test point chosen away from any kink, to
  `1e-6`. Torch is used here only as an external gradient-check oracle; your engine
  itself uses no torch.
- **Tiny training sanity**: a small 2-4-1 net built from your `Value`s trains for
  100 steps on a 40-point moons dataset and must drive the loss below `0.5`,
  confirming the whole engine works end to end.

### The named trap: fan-out accumulation

This grader hunts one specific mistake by name. If any `_backward` overwrites a
parent's gradient (`self.grad = ...`) instead of accumulating it
(`self.grad += ...`), every node that fans out keeps only the last path's
contribution and is undercounted. The grader's `b = a + a` check catches this
directly — `a.grad` comes out `1.0` instead of `2.0` — and the failure message
names `+=` as the fix, so you know to switch every assignment in your `_backward`
closures to accumulation rather than hunting a phantom bug elsewhere. (A second
trap watches for the ordering mistake: a forward-order backward pass fails the
diamond test, and that message names the *topological ordering* as the cause.)

## The hint ladder

If the grader's message is not enough, open the hints. Each has three levels: a
conceptual nudge, then the formula or the invariant it lives on, then pseudocode.
There are ladders for `fan_out_accumulation`, `topo_sort_order`, and the
"engine grad doesn't match the L7 reference" path. Try level 1 before climbing.

## Done when

`uv run grade L09-autograd-engine` shows all-green: the warm-up `forward_pass` reproduces
the L7 forward nodes, your engine's gradients match the L7 hand-derived reference
to `1e-7` on all nine parameters, the fan-out check returns `a.grad == 2.0`, the
diamond returns `a.grad == 15.0`, every per-op gradient matches torch to `1e-6`,
and the tiny moons net trains below `0.5` loss. Then run the experiments and
record your predictions.
