# L3 — Perceptron

This is the simplest thing that *learns to classify* — to **classify** is to
sort each example into one of a fixed set of buckets (here, just two: $+1$ and
$-1$). One weight vector $\mathbf{w}$, one rule, and a boundary that nudges
itself toward its own mistakes. The whole algorithm fits on a postcard — and yet
it carries the seed of everything that follows: logistic regression, SVMs, and
the neurons inside deep networks are all this idea with a smoother knob.

The headline contrast is with **kNN** (k-nearest-neighbors), a classifier that
keeps every training example in memory and labels a new point by a vote of the
$k$ stored examples closest to it. kNN never builds a boundary — it re-reads the
whole dataset at prediction time. The perceptron does the opposite: **kNN
memorizes the training set; the perceptron compresses it into $\mathbf{w}$.**
Once training finishes you can throw the data away — the boundary is the model.

## The model: a sign on a linear score

Give every example a **linear score** $\mathbf{w} \cdot \mathbf{x}$. That dot
$\cdot$ is the **dot product**: read $\mathbf{w} \cdot \mathbf{x}$ aloud as
*"w-dot-x"* — multiply each weight by its matching feature and add the results
into one number, $w_1 x_1 + w_2 x_2 + \dots + w_d x_d$. One number per example,
positive or negative. Now classify by its **sign** — `sign` is the function that
returns $+1$ for any positive input and $-1$ for any negative one:

$$
\hat{y} = \operatorname{sign}(\mathbf{w} \cdot \mathbf{x})
$$

Read aloud: *the prediction $\hat{y}$ is the sign of w-dot-x.* The **decision
boundary** is the set of points where the score is exactly zero — a flat line (in
2-D) or, in any number of dimensions, a flat sheet called a **hyperplane**
("hyper-" just means "the line/plane idea, generalized to $d$ dimensions").
Everything on one side of it scores positive and is labeled $+1$; everything on
the other side scores negative and is labeled $-1$. Training is nothing but
sliding and tilting that one boundary.

### Why labels live in $\{-1, +1\}$, not $\{0, 1\}$

This is the single design choice the whole lesson turns on. With labels in
$\{-1, +1\}$, the score and the true label $y$ agree exactly when
$y \,(\mathbf{w} \cdot \mathbf{x}) > 0$ — they share a sign, so their product is
positive. That product, $y \,(\mathbf{w} \cdot \mathbf{x})$, is the **margin**
(read aloud: *"y times w-dot-x"*): positive means safely correct — and the bigger
it is, the more confidently correct, so training leaves it alone; non-positive
($\le 0$) means the point is not safely on the right side and training updates on
it. One scalar tells you both class branches at once. (At a score of exactly
zero, `predict` has to pick *something*: it breaks the tie to $+1$ so every output
is a valid signed label — but training still treats margin $0$ as not-safe and
updates.)

Why this beats the $\{0, 1\}$ encoding you might reach for first: with a label of
$0$, the product $y\,(\mathbf{w} \cdot \mathbf{x})$ is just $0$ no matter how
wrong the score is, so it can never say "this negative example landed on the
positive side — fix it." The $\{0, 1\}$ scheme has no notion of "wrong
direction." The $\{-1, +1\}$ scheme makes one signed number, the margin, carry
both *which class* and *right-or-wrong* — and the update rule below rides on
exactly that.

## Why not just run gradient descent?

If you just finished L1 and L2, a fair question is nagging: *we learned one
master tool — gradient descent on a loss — and L2 was "the same descent, only the
loss changed." Why does the perceptron get a special "learning rule"? Can't we
just differentiate this model and descend?*

The first thing to fix is a phrasing trap that bites everyone here. **You never
differentiate the model.** Gradient descent takes the slope of the *loss* with
respect to the *parameters* — "if I nudge each weight a little, how does my total
wrongness change?" (That is exactly the only derivative L1 ever took.) So the real
question is: what loss would we descend on for classification?

The honest loss for classification is **count the misclassifications** — how many
points land on the wrong side. But that count is built on `sign`, and here is the
wall: `sign` is a **step function**. It is flat at $-1$, then jumps to flat at
$+1$. A flat region has slope **zero**, and at the jump the slope is undefined.
So the misclassification-count loss has a **gradient of zero almost everywhere**
(and no gradient at all at the jump). There is no slope to follow: gradient
descent would multiply the learning rate by $0$ and **never move the weights**.
Nudging a weight a hair changes the score a hair, but unless that hair flips a
point across the boundary, the *count* doesn't change at all — zero slope. This
is the one place in the whole curriculum where the step truly blocks plain
gradient descent.

So the perceptron uses a different trick: a direct **learning rule** that edits
$\mathbf{w}$ by hand on each mistake. That is what `worked/perceptron.py`
implements — no loss, no gradient, just the rule below.

But it is *not* an exception to "everything is gradient descent on a loss." The
rule turns out to be **the same as** one step of stochastic gradient descent —
with learning rate $1$ — on a stand-in loss. That stand-in, a **surrogate**
chosen precisely because it has a usable slope where the step function did not, is
called the **perceptron criterion**:

$$
L = \max\bigl(0,\; -\,y\,(\mathbf{w} \cdot \mathbf{x})\bigr)
$$

Read aloud: *the loss is the larger of zero and negative-margin.* When the margin
$y\,(\mathbf{w} \cdot \mathbf{x})$ is positive (already correct) the inside is
negative, the $\max$ picks $0$ — no penalty, no slope, no update. When the margin
is non-positive (a mistake) the loss is $-y\,(\mathbf{w} \cdot \mathbf{x})$, whose
slope in $\mathbf{w}$ is $-y\,\mathbf{x}$; descending it means stepping by
$+y\,\mathbf{x}$ — exactly the update we are about to write. (The $\max$ has a
**kink** at margin $0$ where the slope is not unique, so this is technically a
*subgradient* — a stand-in for the slope at a corner — but the step is the same.)
The flat step got replaced by a tilted ramp that actually has a slope to walk
down. We will not descend this loss in code — the code uses the direct rule — but
knowing the two are the same is the bridge that keeps the perceptron inside the
one master tool.

Hold onto this, because it is the through-line of the next lessons: **logistic
regression** (next) swaps the hard `sign` step for a smooth S-curve so plain
gradient descent works directly again, and the **SVM** (L5) descends a close
cousin of this surrogate — the **hinge loss** — which has its own kink, handled
the same way, with a subgradient. The perceptron is the single spot where the
step itself fully blocks the slope.

## The learning rule: nudge toward the mistake

Walk through the data one example at a time — visiting and updating on **one
point per step** is the *stochastic* (or *online*) style of descent, in contrast
to L1, which summed the gradient over *all* the data before each step. The whole
update, applied to one example $(\mathbf{x}, y)$, is three steps:

1. **Compute the margin** $m = y\,(\mathbf{w} \cdot \mathbf{x})$ — score the
   example, then multiply by its true label so one signed number says both *which
   class* and *right-or-wrong* (the $\{-1, +1\}$ trick from above).
2. **If $m > 0$** (safely correct), leave $\mathbf{w}$ unchanged — no penalty, no
   move.
3. **If $m \le 0$** (misclassified, or sitting exactly on the boundary), nudge
   the boundary toward this point:
   $$
   \mathbf{w} \leftarrow \mathbf{w} + y\,\mathbf{x}
   $$

Read aloud: *if y-times-w-dot-x is at most zero, replace $\mathbf{w}$ with
$\mathbf{w}$ plus y-times-x.* (This is the $+y\,\mathbf{x}$ step the surrogate
loss above predicted.) The $y$ factor is the heart of it. For a positive example
($y = +1$) you add $\mathbf{x}$, tilting the boundary so this point scores higher
next time. For a negative example ($y = -1$) you subtract it, tilting the other
way. **The sign of $y$ IS the update's direction.** Drop the $y$ and the update
still fires on mistakes but always pushes the same way — it has lost all class
information. That is the one bug the grader hunts for at the 6-point trace (it has
a name: the `missing_y_sign` trap).

A point exactly on the boundary (margin $= 0$) counts as a mistake and triggers
an update: a zero-confidence prediction is not good enough to leave alone.

## Training: passes with a shuffle

One **epoch** is one full pass over every row, applying the update to each. To
**shuffle** is to randomize the visiting order; we re-shuffle each epoch so the
perceptron does not see the rows in the same fixed sequence every pass. We seed
the shuffle (give the random generator a fixed starting number) so the "random"
order is the same on every run — random enough to break stored-order artifacts,
fixed enough that the grader gets one reproducible answer. The reason to shuffle
at all: if you always scanned the data in its stored order, a learner that
happened to receive the rows pre-sorted could land on a different boundary than
one that did not. The shuffle makes the result a property of the *data*, not its
arrival order.

As a procedure, training is the single-example rule wrapped in two loops:

1. **Start** $\mathbf{w}$ at all zeros (one weight per feature), and seed the
   random generator from `seed` so the shuffle is reproducible.
2. **For each of `epochs` passes:** draw a fresh shuffled visiting order over all
   $n$ rows.
3. **For each row $i$ in that order:** replace $\mathbf{w}$ with
   `perceptron_step(w, X[i], y[i])` — the three-step rule above, run on that one
   example.
4. **After all passes,** return the final $\mathbf{w}$ — the fitted boundary.

## When it works — and the honest caveat

First a term: data is **linearly separable** when *some* straight line (or
hyperplane) can put every $+1$ point on one side and every $-1$ point on the
other, with no mistakes — a single line cleanly splits the two classes.

The **perceptron convergence theorem**, in plain words: *if the data is linearly
separable, the perceptron is guaranteed to find a separating boundary in a finite
number of steps.* More precisely, it makes only a finite number of mistakes —
each mistake triggers one update, so a finite mistake count means the updates stop
— and once they stop, the boundary it is sitting on separates the data. It does
not promise the *best* separating line, only *a* separating line. Wider margins
converge faster (fewer mistakes); razor-thin margins take longer.

The honest caveat: raw-pixel MNIST for *easy* digit pairs is **near-separable**,
not guaranteed-separable. In practice a tiny capped budget (5 epochs) classifies
0-vs-1 almost perfectly and 3-vs-5 well, but "near-separable" is an empirical
observation about these pixels, not a theorem. The grader's accuracy bounds are
set generously below the measured performance for exactly this reason — they do
not assume convergence.

## Where it breaks: XOR

Now the destination of the lesson. Some data is **not** linearly separable — no
single straight line can split the classes — and on it the perceptron never
settles. The canonical example is **XOR** ("exclusive or"): the label is $+1$
when the two inputs *differ* and $-1$ when they *match*. Written out as four
points $\to$ label:

$$
(0,0) \to -1, \quad (0,1) \to +1, \quad (1,0) \to +1, \quad (1,1) \to -1
$$

Picture them on a square: the two $+1$ points sit on one diagonal, the two $-1$
points on the other. No straight line puts both $+1$ points on one side and both
$-1$ points on the other — any line you draw splits one diagonal pair across it.
Because no separating line exists, the convergence theorem simply does not apply.
Run the perceptron on XOR for a thousand epochs and it **cycles**: each pass fixes
some points and breaks others, forever, because every update that helps one
diagonal hurts the other. This is not a bug in your code — it is a fundamental
limit of a single linear boundary.

That limit is the bridge to the rest of the curriculum. Two ways out:

- **Lift the features** so the data becomes separable in a richer space — the
  margin (L5) and feature-map (L6) story.
- **Stack boundaries** so the model can carve non-linear regions — the hidden
  layers of L7+.

You will build both. The perceptron's failure is what makes them necessary.

## Exercise

Implement in `my_work/perceptron/perceptron.py` (run `grade start perceptron` to
get the scaffold):

- `standardize(X)` — the **warm-up**, recalled cold from L2. Parameter: `X` is the
  feature matrix, shape `(n, d)` — `n` rows (examples), `d` columns (features). It
  returns the matrix with each *column* centered to mean 0 and scaled to std 1,
  same shape `(n, d)`, float64. It is graded on its own and is *not* called by the
  perceptron functions below; re-authoring it from memory keeps the operation
  fresh.
- `perceptron_step(w, x, y)` — one update on a single example. Parameters: `w` is
  the current weight vector, shape `(d,)`; `x` is one example, shape `(d,)`; `y`
  is its label, a scalar in $\{-1, +1\}$. Return the **new** weight vector, shape
  `(d,)`, float64 — do not mutate the caller's `w`. Fire the update
  $\mathbf{w} + y\,\mathbf{x}$ only when the margin
  $y\,(\mathbf{w} \cdot \mathbf{x}) \le 0$; otherwise return `w` unchanged.
- `train_perceptron(X, y, epochs, seed)` — the training loop. Parameters: `X` is
  the design matrix, shape `(n, d)` (the caller has already prepended the bias
  column of ones, so `d` includes it); `y` is the labels, shape `(n,)`, values in
  $\{-1, +1\}$; `epochs` is how many full passes to make; `seed` is the RNG seed
  for the per-epoch shuffle. Start `w` at zeros and build one generator
  `np.random.default_rng(seed)` *before* the loop; then each epoch draw a fresh
  `rng.permutation(n)` order from that same generator (so successive epochs get
  *different* orders, all reproducible from the one seed) and apply
  `perceptron_step` to every row in that order. Return the fitted weight vector,
  shape `(d,)`.
- `predict(X, w)` — signed predictions. Parameters: `X` is a design matrix, shape
  `(n, d)`; `w` is the weight vector, shape `(d,)`. Return one label per row, shape
  `(n,)`, taking the sign of the linear score $X\mathbf{w}$ and mapping it to the
  $\{-1, +1\}$ set — a score of exactly 0 maps to $+1$ so every output is a valid
  signed label.

Then grade it: the site's Run Grader button or `grade perceptron`.

## How the grader checks you

- The **warm-up** `standardize` is an independent unit: it is graded and recorded
  on its own and never blocks the rest. It is run on a fixed 5×2 array and each
  output column must come out mean ~0 (within $10^{-8}$) and std ~1. A wrong
  warm-up does not hide a working perceptron chain.
- `perceptron_step` is checked against a fixed **6-point trace** in 2-D starting
  from $\mathbf{w} = [0, 0]$. The sequence is built so three points trigger an
  update (margin $\le 0$) and three do not (margin $> 0$), exercising both
  branches; after each step the returned weights must match the reference exactly.
  This is the foundation check — the trainer and accuracy checks below depend on it.
- `predict` is run on four points with mixed scores (including a score of exactly
  0) and must return labels drawn only from the signed set $\{-1, +1\}$, matching
  the expected `[+1, -1, +1, +1]` — not the $\{0, 1\}$ set.
- `train_perceptron` is trained for 5 epochs at `seed=0` on two near-separable
  raw-pixel MNIST pairs: it must reach **> 0.85** test accuracy on **0-vs-1** and
  **> 0.75** on the messier **3-vs-5**. The bounds sit well below the measured
  ~0.999 and ~0.94, so they carry a wide safety margin and do *not* assume
  convergence.

### The named trap: the missing y-sign

This grader is built to catch one specific mistake by name. If you write
`perceptron_step` with `w + x` instead of `w + y * x` — dropping the $y$ factor
from the update — the update keeps firing on mistakes but always pushes
$\mathbf{w}$ the *same* way, regardless of class: a negative example gets shoved
toward the positive side instead of away from it. The update has lost its
direction. The 6-point trace catches it at the first negative-label mistake (the
third point, where the reference subtracts $\mathbf{x}$ but the buggy version
adds it), and the grader names the cause — it tells you the **update direction**
is wrong — so you fix the $y$ factor rather than hunting a phantom bug. The trap
has a name in the test suite: `missing_y_sign`.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the formula with shapes, then pseudocode. Try
level 1 before climbing — the `perceptron_step` hint, for instance, opens by
re-asking *why $y$ lives in $\{-1, +1\}$* before it ever shows the formula, so the
named trap above is steered around by design.

## Done when

`grade perceptron` shows all-green: the warm-up `standardize` leaves each column
mean ~0 / std ~1, `perceptron_step` matches the 6-point trace exactly (both the
update and the leave-alone branch), `predict` returns only $\{-1, +1\}$ labels,
and `train_perceptron` clears 0.85 on 0-vs-1 and 0.75 on 3-vs-5 at 5 epochs,
seed 0. Then run the experiments and record your predictions — they are where you
*watch* the perceptron cycle on XOR and feel why a single linear boundary is not
enough.
