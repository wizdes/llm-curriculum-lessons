# L4 — Logistic Regression

In L3 the perceptron classified by a **hard sign**. It took the linear score
$\mathbf{w} \cdot \mathbf{x}$ — the weight vector $\mathbf{w}$ dotted with the
input $\mathbf{x}$ — and reported only whether that score was positive or
negative: the point was $+1$ or $-1$, full stop, with no notion of *how sure* it
was. A point one inch past the boundary and a point a mile past it got the same
flat answer.

That hardness causes a second problem: we could not **differentiate** it. "To
differentiate" means to find the slope — how much the output changes when you
nudge a weight. Gradient descent (L1) needs that slope to know which way is
downhill. But `sign` has a vertical cliff right at the boundary and is perfectly
flat (zero slope) everywhere else, so there is no useful slope to follow. No
slope, no gradient, no descent.

Logistic regression fixes both problems at once by replacing the cliff with a
smooth S-shaped curve. The headline: **logistic regression is the perceptron
with calculus.** Swap the hard step for that smooth curve and every quantity
gets a real slope — so you can train the classifier with the *exact same*
gradient descent engine you built in L1. You already built that engine; here we
just hand it a new loss (a new "how wrong are we?" number to minimize).

## From a hard step to a smooth probability

Start with the same raw number the perceptron used: the linear score
$z = \mathbf{w} \cdot \mathbf{x}$. We call this $z$ the **logit** — it is just a
name for the unsquashed score before we turn it into a probability. (You can
read "logit" as "the raw input to the S-curve.") A logit can be any real number,
from very negative to very positive.

A **probability** is a number between $0$ and $1$ saying how likely something is:
$0$ means "certainly not", $1$ means "certain", $0.5$ means "a coin flip". We
want to turn the unbounded logit $z$ into a probability. The tool for that is the
**sigmoid** (Greek letter sigma, written $\sigma$), the smooth S-shaped curve:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Read aloud: *the sigmoid of $z$ is one divided by one-plus-$e$-to-the-minus-$z$*,
where $e \approx 2.718$ is the base of the natural exponential. Whatever real
number $z$ you feed in, the output lands strictly between $0$ and $1$ — exactly
the range a probability needs.

It is the soft version of the perceptron's `sign`. Plug in a few values to see:

| logit $z$ | $\sigma(z)$ | reading |
|-----------|-------------|---------|
| $-4$      | $0.018$     | almost certainly class 0 |
| $-1$      | $0.27$      | leaning class 0 |
| $0$       | $0.5$       | exactly on the fence |
| $+1$      | $0.73$      | leaning class 1 |
| $+4$      | $0.982$     | almost certainly class 1 |

Large positive $z$ goes toward $1$, large negative $z$ goes toward $0$, and
$z = 0$ sits at exactly $0.5$ — the **decision boundary**, the dividing line
between predicting class 0 and class 1. We read $\sigma(z)$ as
$P(y = 1 \mid \mathbf{x})$ — *the probability that the label $y$ is $1$, given
the input $\mathbf{x}$* (the bar "$\mid$" means "given"). So the model no longer
just picks a side; it reports how far past the boundary the input sits, and that
distance is its **confidence**. The perceptron threw all of that away by
reporting only a sign.

One bookkeeping choice before the loss. Here the two classes are labelled
$\{0, 1\}$ (call this the **binary convention**), not the perceptron's
$\{-1, +1\}$. Why the switch? Because $\sigma(z)$ is *the probability of class
$1$*, and the cross-entropy loss below is written in terms of "did the
probability match the label $0$ or $1$?". Labelling the classes $0$ and $1$ lets
the same formula handle both classes with no extra sign-juggling — you will see
exactly how when the loss appears.

## Cross-entropy: the natural loss

We need a **loss** — one number that says how wrong a prediction is, so gradient
descent has something to push down. Let $p = \sigma(z)$ be the model's predicted
probability of class $1$ for one example, and let $y \in \{0, 1\}$ be that
example's true label. The loss for that single example is the **binary
cross-entropy**, abbreviated **BCE** ("binary" = two classes; "cross-entropy" is
the standard name for this match-the-probability-to-the-label penalty):

$$
\mathcal{L} = -\big[\, y \log p + (1 - y) \log (1 - p) \,\big]
$$

Read it aloud: *the loss is minus the quantity ($y$ times log-$p$, plus
one-minus-$y$ times log-of-one-minus-$p$)*. Here $\log$ is the natural logarithm.
The formula looks busy, but the $y$ and $(1 - y)$ are just a switch that picks
one of two cases, because $y$ is always exactly $0$ or $1$:

- When $y = 1$: the $(1 - y)$ term vanishes and the loss is $-\log p$. You pay a
  penalty for assigning a *low* probability to the class that was actually true.
- When $y = 0$: the $y$ term vanishes and the loss is $-\log(1 - p)$, the same
  penalty measured against $1 - p$, the predicted probability of class $0$.

That switch is exactly why the $\{0, 1\}$ convention is so convenient — the label
*itself* turns the right term on and the wrong term off.

Make it concrete. Recall $\log 1 = 0$ and that $\log$ of a tiny number is a large
negative number, so $-\log$ of a tiny number is large and positive:

- **Confident and right** ($y = 1$, $p = 0.99$): loss is
  $-\log 0.99 \approx 0.01$. Almost no penalty.
- **Unsure** ($y = 1$, $p = 0.5$): loss is $-\log 0.5 \approx 0.69$. A moderate
  nudge.
- **Confident and wrong** ($y = 1$, $p = 0.01$): loss is
  $-\log 0.01 \approx 4.6$. A big penalty — and it keeps growing without bound as
  $p \to 0$.

So the loss is near zero when the model is confident and correct, and explodes
when it is confident and *wrong*. We call BCE the *natural* loss because it is
the **negative log-likelihood** of the data. "Likelihood" is the probability the
model assigns to the labels you actually observed; taking the log and flipping
the sign turns "make that probability as large as possible" into "make this loss
as small as possible" — the form gradient descent wants. Minimizing BCE *is*
maximizing the probability the model puts on the real labels.

## The beautiful result: $\partial \mathcal{L} / \partial z = p - y$

Here is the payoff that makes the whole lesson click. To run gradient descent we
need the slope of the loss with respect to the logit $z$. We write that slope as
$\partial \mathcal{L} / \partial z$ — read *the partial derivative of the loss
with respect to $z$*, meaning "how much the loss $\mathcal{L}$ changes when we
nudge $z$, holding everything else fixed." (The curly $\partial$ is just the
symbol for a derivative when more than one variable is around.)

If you carry out that derivative — differentiate the BCE, follow the chain rule
through the sigmoid — almost everything cancels and you are left with something
startlingly simple:

$$
\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y = p - y
$$

Read aloud: *the slope of the loss with respect to the logit equals the predicted
probability minus the true label.* That is it. The whole derivative is the
**probability gap** $p - y$ — how far the model's predicted probability sits from
the actual label. Predict $p = 0.9$ when the truth is $y = 1$ and the gap is
$-0.1$ (small, almost done); predict $p = 0.9$ when the truth is $y = 0$ and the
gap is $+0.9$ (large, big correction coming).

We do not need to re-derive this from scratch here — take it as the exact result
the calculus gives. To go from this single-logit slope to the gradient over a
whole batch of weights, follow the chain rule through two more steps:

1. **From logit to one weight.** Since $z = \mathbf{w} \cdot \mathbf{x}$, nudging
   weight $w_j$ changes $z$ by a factor of $x_j$ (the $j$-th feature). The chain
   rule multiplies the two slopes, so $\partial \mathcal{L}/\partial w_j$ picks up
   that factor of $x_j$: the per-weight slope is $(p - y)\,x_j$.
2. **Stack all $n$ examples.** Recall from L2 that $X$ is the **design matrix**,
   the $(n, d)$ array with one example per row and $d$ features per column, and
   that $\mathbf{p}$, $\mathbf{y}$ are the length-$n$ vectors of predicted
   probabilities and true labels. Summing the per-example contributions and
   averaging collapses to one matrix product:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}}
  = \frac{1}{n}\, X^\top (\,\mathbf{p} - \mathbf{y}\,)
$$

Read aloud: *the gradient with respect to the weights is one-over-$n$ times
$X$-transpose times the probability gap.* Here $X^\top$ ("$X$ transpose") is the
matrix $X$ flipped so its rows become columns; that flip makes the shapes line up
as $(d, n)$ times $(n,)$, giving one number per weight — a length-$d$ gradient.
The $\tfrac{1}{n}$ averages over the batch so the step size does not grow just
because you fed in more rows.

This is the *same* "design matrix times residual" shape you derived for linear
regression in L2 — $\tfrac{1}{n} X^\top (\text{something} - \mathbf{y})$. Only
the "something" changed: there the residual was a value gap, the raw prediction
minus the target $(\hat{y} - y)$; here it is a probability gap $(p - y)$. Same
engine, new loss.

## Numerical stability: compute BCE from the logits

The math above is correct on paper. But computers store numbers with limited
precision, and the obvious way to code BCE quietly breaks. This is the **named
trap** of the lesson — the grader tracks it as `naive_log_sigmoid`, the mistake
of forming probabilities first inside `bce_from_logits` — and the grader will
check that you avoid it.

The trap: form the probability $p = \sigma(z)$ first, then plug it into
$-[y \log p + (1-y)\log(1-p)]$. Watch what happens at a very confident logit.
When $z$ is large the sigmoid is **saturated** — pushed so far toward its ceiling
of $1$ (or floor of $0$) that, in finite precision, it rounds to *exactly* that
ceiling. At $z = 1000$, $\sigma(z)$ does not store as $0.9999\ldots$; it rounds
to exactly $1.0$. Then $1 - p = 0$, so $\log(1 - p) = \log(0) = -\infty$ (the log
of zero is negative infinity). And when the label is $y = 0$, that term is
multiplied by $(1 - y) = 1$ — but if instead $y = 1$, you get $0 \cdot (-\infty)$,
which the computer reports as **`NaN`** ("Not a Number", the value you get from a
meaningless arithmetic operation like $0 \times \infty$). One `NaN` poisons every
later step. The probability was never really the problem — *forming it at all*
was, because rounding it to exactly $0$ or $1$ destroyed information the log
needed.

The fix is to never form $p$. Compute the BCE straight from the logit $z$ using
the **softplus** function. Softplus is a smooth, always-positive curve defined as

$$
\operatorname{softplus}(z) = \log\!\big(1 + e^{z}\big),
$$

read *softplus of $z$ is the log of one-plus-$e$-to-the-$z$* (it is a "soft",
rounded version of the ramp $\max(0, z)$ — hence the name). With a little algebra,
the entire per-example BCE collapses to one clean expression in $z$:

$$
\mathcal{L}(z, y) = \operatorname{softplus}(z) - y\,z
  = \log\!\big(1 + e^{z}\big) - y\,z
$$

Read aloud: *the loss equals softplus-of-$z$ minus $y$-times-$z$.* No division,
no $\log$ of a probability that might have rounded to zero — just the logit $z$
itself.

One subtlety remains: $e^{z}$ inside softplus would itself **overflow** (exceed
the largest number the computer can store) for large positive $z$. The standard
library hands you a version that already guards against this: `np.logaddexp(0, z)`
computes $\log(e^{0} + e^{z}) = \log(1 + e^{z})$ but rearranges the arithmetic
internally so it never overflows for any $z$.

So the stable BCE for one batch is a three-step procedure:

1. **Compute the per-example loss straight from the logits**, never forming $p$:
   `per_example = np.logaddexp(0.0, z) - y * z`. This is
   $\operatorname{softplus}(z) - y\,z$, one value per example — shape $(n,)$.
2. **Average over the batch**: `np.mean(per_example)`. The mean (not the sum)
   keeps the loss from growing just because you fed in more rows.
3. **Return it as a plain `float`** — a single scalar measuring how wrong the
   whole batch is.

A reassuring check: at $z = 1000, y = 0$ this gives $\operatorname{softplus}(1000) - 0 = 1000$ (a big but *finite* loss, because the model was confidently wrong), and at $z = -1000, y = 1$ it gives $\operatorname{softplus}(-1000) - (-1000) \approx 1000$ as well. Finite, sensible numbers exactly where the naive version blew up to a non-finite value — `inf` for these two, and `NaN` for the closely related $z = 1000, y = 1$ case above.

Two side notes so you do not over- or under-apply the fix:

- This `logaddexp` trick is the **binary** (two-class) analogue of the
  "max-subtraction" trick used to stabilize the multi-class **softmax** (the
  many-class cousin of the sigmoid, coming in L9/L18). Same goal — keep the
  exponential in range — but a different identity. Do not reach for
  max-subtraction here; for the binary case `logaddexp` is exactly the tool.
- The `sigmoid` function still needs its *own* separate guard for any place you
  genuinely want a probability (predicting, reporting). The guard is to branch on
  the sign of $z$ so the exponential only ever sees a non-positive argument (which
  cannot overflow). But the **loss** must be computed from logits via the softplus
  form above, never from a saturated $p$.

## Training: gradient descent, unchanged

There is nothing new to build for the optimizer — this is the whole point of the
lesson paying off. We reuse L1's loop unchanged. The learning rate $\eta$ (the
Greek letter eta, from L1) is the step size — how far you move downhill each step.
Start the weights at $\mathbf{w} = \mathbf{0}$, then repeat these steps a fixed
number of times:

1. **Recompute the logits** for the current weights: $\mathbf{z} \leftarrow X \mathbf{w}$.
   (The "$\leftarrow$" arrow means "becomes": compute the right side, store it
   back into the left.)
2. **Form the probability gap** $\mathbf{p} - \mathbf{y}$, where
   $\mathbf{p} = \sigma(\mathbf{z})$ — the per-example residual from the
   $\partial \mathcal{L}/\partial z = p - y$ result above.
3. **Build the gradient** $\frac{1}{n} X^\top (\,\mathbf{p} - \mathbf{y}\,)$.
4. **Step downhill**:
   $\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{1}{n} X^\top (\,\mathbf{p} - \mathbf{y}\,)$.

The whole update in one line:

$$
\mathbf{z} \leftarrow X \mathbf{w}, \qquad
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot
  \frac{1}{n} X^\top (\,\mathbf{p} - \mathbf{y}\,)
$$

Read aloud: *set the logits to $X$ times $\mathbf{w}$; then replace the weights
with the old weights minus the learning rate $\eta$ times the BCE gradient.* Run
this a fixed number of steps and the weights settle into a good classifier.

Why start the weights at zero? Logistic regression's loss surface is **convex** —
bowl-shaped, with no false valleys to get stuck in — so on a well-behaved problem
the starting point does not change *where* gradient descent settles, and zero is
the simplest, fully reproducible choice. (The perceptron's order-dependence is
gone; gradient descent on a convex bowl is deterministic.)

This is verbatim the L1 gradient-descent loop with `bce_grad` plugged in. The
grader confirms two things: that the loss falls on every one of the first steps —
proof the gradient truly points downhill — and that your fitted classifier agrees
(on more than $95\%$ of a fixed $200$-row slice of MNIST handwritten digits) with
scikit-learn's battle-tested `LogisticRegression`, used only as an external
answer key.

## Exercise

Implement in `my_work/logistic-regression/logistic_regression.py` (run
`grade start logistic-regression` to get the scaffold). This is a blind rebuild:
you write each function from scratch, then the grader checks it. The shape
convention, curriculum-wide: $n$ rows (examples) and $d$ features, so $X$ is
$(n, d)$ while $z$ and $y$ are both length $n$. Keep every array in `float64`
(64-bit floating point) for precision.

- `perceptron_step(w, x, y)` — the **warm-up**: re-derive L3's single perceptron
  update from memory. Parameters: `w` is the current weight vector, shape `(d,)`;
  `x` is one example, shape `(d,)`; `y` is its label, a scalar in the *signed*
  $\{-1, +1\}$ convention. Compute the margin $y\,(\mathbf{w}\cdot\mathbf{x})$;
  on a mistake (margin $\le 0$) return $\mathbf{w} + y\,\mathbf{x}$, otherwise
  return `w` unchanged — a **new** array shape `(d,)`, never mutating `w`. It is
  graded on its own and the logistic functions do not call it, so a shaky warm-up
  never blocks the rest of the lesson.
- `sigmoid(z)` — the stable sigmoid. Parameter: `z` is an array of logits, any
  shape. Branch on the sign of `z` (with `np.where` or masked assignment) so the
  exponential only ever sees a non-positive argument and never overflows. Returns
  an array the same shape as `z`, with every entry a probability in $[0, 1]$ —
  mathematically open at $0$ and $1$, but in finite precision a saturated logit
  rounds to exactly $0.0$ or $1.0$ (which is what the grader probes for).
- `bce_from_logits(z, y)` — the mean binary cross-entropy over the batch.
  Parameters: `z` is the logits, shape `(n,)`; `y` is the labels in the
  $\{0, 1\}$ convention, shape `(n,)`. Compute it the stable way via the softplus
  form, `np.logaddexp(0, z) - y * z`; never form $p$. Returns one Python `float`
  scalar — the mean over the $n$ examples.
- `bce_grad(X, z, y)` — the gradient of the mean BCE with respect to the weights.
  Parameters: `X` is the design matrix, shape `(n, d)`; `z` is the logits
  $z = X\mathbf{w}$, shape `(n,)`; `y` is the labels in $\{0, 1\}$, shape `(n,)`.
  Returns the gradient $X^\top(\mathbf{p} - \mathbf{y})/n$, shape `(d,)` — one
  entry per weight — straight from the $\partial \mathcal{L}/\partial z = p - y$
  result above.
- `fit(X, y, lr, steps)` — **full-batch** gradient descent (every step uses all
  $n$ rows at once), which is exactly the L1 engine. Parameters: `X` is the
  design matrix, shape `(n, d)`, with the bias column already added by the caller;
  `y` is the labels in $\{0, 1\}$, shape `(n,)`; `lr` is the learning rate $\eta$
  (step size); `steps` is how many full-batch updates to take. Start weights at
  zero and run the numbered training loop above. Returns the fitted weight
  vector, shape `(d,)`.

Then grade it: the site's Run Grader button or `grade logistic-regression`.

## How the grader checks you

- The **warm-up** `perceptron_step` is an independent unit: it is graded and
  recorded on its own and never blocks the rest. It is run on a fixed 6-point
  sequence in 2-D starting from $\mathbf{w} = [0, 0]$ — three points trigger an
  update and three do not, exercising both branches — and your weight vector must
  match the expected trace at every step.
- `sigmoid` is probed at saturation: $\sigma(1000)$ must be exactly $1.0$ (not
  `NaN`), $\sigma(-1000)$ exactly $0.0$, and $\sigma(0)$ exactly $0.5$, with every
  output finite. This is the foundation the rest of the chain stands on, so it is
  checked first.
- `bce_from_logits` is checked for *stability* at the saturated logits
  $z = [1000, -1000]$ with labels $y = [0, 1]$: the result must be finite and
  equal $1000.0$. Each per-example loss is $\operatorname{softplus}(z) - y\,z$:
  the first is $\operatorname{softplus}(1000) - 0 = 1000$, the second is
  $\operatorname{softplus}(-1000) - (1)(-1000) \approx 0 + 1000 = 1000$, so their
  mean is $1000$. The naive form goes non-finite here — see the named trap below.
- `bce_grad` is checked by comparing your analytic gradient against a **numerical
  finite-difference gradient** on a tiny fixed batch (the slow, per-parameter
  central-difference estimate from L1, used here only as a debug oracle — never to
  train). The relative error must stay below $10^{-4}$, and the returned shape
  must be `(d,)`.
- `fit` is checked two ways. First, **loss monotonicity**: the BCE must fall on
  every one of the first $20$ steps at the reference learning rate — direct proof
  the gradient points downhill. Second, the **oracle**: your fitted classifier
  must agree with scikit-learn's `LogisticRegression` on more than $95\%$ of a
  fixed $200$-row slice of MNIST handwritten digits (observed agreement is
  $\approx 0.99$). scikit-learn appears *only* inside this oracle — an external
  answer key — never in your own module.

### The named trap: numerical instability (`naive_log_sigmoid`)

This grader is built to catch one specific mistake by name. If you compute BCE
the textbook way — form the probability $p = \sigma(z)$ first, then evaluate
$-[\,y \log p + (1-y)\log(1-p)\,]$ — it goes **non-finite** the moment $z$
saturates. At $z = 1000$ the sigmoid rounds to exactly $1.0$, so $1 - p = 0$ and
$\log(1 - p) = \log(0) = -\infty$; depending on the label you get `inf` (when the
$\log(0)$ term survives) or `NaN` (when a $0 \cdot \infty$ appears), and one
non-finite value poisons every later step. The grader's stability probe at
$z = \pm 1000$ checks that your result stays **finite** and equals $1000.0$, so it
catches either failure and names the cause: the fix is the softplus identity
`np.logaddexp(0, z) - y * z`, which never forms $p$ at all (see
**Numerical stability: compute BCE from the logits** above). The trap is tracked
as `naive_log_sigmoid`, and the grader's message points you at `softplus`.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the formula with shapes, then pseudocode. Try
level 1 before climbing — the level-2 hints carry the exact shapes (e.g. that
$X^\top(\mathbf{p}-\mathbf{y})/n$ maps $(d, n)\,@\,(n,) \to (d,)$, and that the
residual is $p - y$, not $y - p$), and level 3 is runnable pseudocode. If your
numbers desync from the worked module, the hints point you back to it.

## Done when

`grade logistic-regression` shows all-green: the warm-up trace matches, `sigmoid`
is exactly $1.0$/$0.0$/$0.5$ at $1000$/$-1000$/$0$ with no `NaN`,
`bce_from_logits` is finite and $1000.0$ at the saturated probe, `bce_grad`
matches the numerical gradient to $10^{-4}$, and `fit` decreases the loss every
step and agrees with scikit-learn on more than $95\%$ of the slice. Then run the
experiments and record your predictions.
