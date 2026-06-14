# L5 — SVM via Gradient Descent

The perceptron in L3 was happy with **any** line that separated the two classes.
Run it on the same blobs twice from different starting points and you get two
different boundaries — both classify the training data perfectly, neither is
special. The **support vector machine** (SVM for short — a classifier that
draws a separating line) asks a sharper question: of all the lines that separate
the classes, which one leaves the **widest empty corridor** between them? That
widest-corridor line sits as far as possible from the nearest point of either
class, which makes it the most forgiving of noise — nudge a point a little and it
still lands on the right side.

Here is the part that should feel surprising the first time: you find that line
with the **same gradient descent engine you built in L1**. Gradient descent is
the loop that repeatedly takes a small downhill step on a loss function until it
reaches the bottom. To turn it from "fit a line by least squares" (L2) into "find
the widest-margin separator," you change exactly one thing — the loss function it
descends. No new optimizer, no special solver. This is misconception #5 made
concrete: **the loss is task-specific, but the optimizer is loss-agnostic.** Swap
the loss, keep the engine.

The headline of this lesson: **the hinge loss plus an L2 penalty *is* the linear
soft-margin SVM.** We will define both of those terms from zero below. By the end
you will have plugged a new loss into L1's loop and watched the widest corridor
open up.

## Maximizing the margin

Start with how we label the two classes. In this lesson labels are **signed**:
each example's label $y_i$ is either $-1$ or $+1$, written $y_i \in \{-1, +1\}$
(read "$y_i$ is in the set containing minus one and plus one"). The subscript $i$
just indexes the examples — $y_3$ is the label of the third example. This is the
same signed convention from the perceptron in L3, not the $\{0, 1\}$ convention
logistic regression used in L4; we will see in a moment why $\{-1, +1\}$ is the
natural choice here.

A linear classifier scores an example by a dot product. Write $\mathbf{W}$ for
the weight vector (the bold letter means it is a vector of numbers, one weight per
feature) and $\mathbf{X}_i$ for the $i$-th example's feature vector. The score is
$\mathbf{X}_i \cdot \mathbf{W}$ — the **dot product**, meaning multiply the two
vectors entrywise and add up the results, giving a single number. The sign of
that score is the prediction: positive means "predict the $+1$ class," negative
means "predict the $-1$ class."

Now the central quantity of the whole lesson. For weights $\mathbf{W}$ the
**signed margin** of example $i$ is

$$
m_i = y_i \, (\mathbf{X}_i \cdot \mathbf{W}).
$$

Read this aloud: "the margin of example $i$ is its label times its score." The
multiplication by $y_i$ is a sign-flip trick. If the true label is $+1$ and the
score is positive (a correct call), the product is positive. If the true label is
$-1$ and the score is negative (also correct), the two negatives multiply to a
positive again. So a **positive margin means the example is classified
correctly**, regardless of which class it belongs to, and a negative margin means
it is on the wrong side. The *size* of the margin tracks how far the point sits
from the boundary on its correct side — a big positive margin is a point deep in
safe territory, a small one is a point hugging the line. (This sign-collapsing is
exactly why we use signed $\{-1, +1\}$ labels: one expression covers both classes
at once.)

This $m_i$ is the **functional margin** — a signed *score*, not yet a distance in
the picture. To turn it into the actual **geometric distance** from the point to
the boundary you divide by the weight norm: $m_i / \|\mathbf{w}\|$, where
$\mathbf{w}$ is the *feature* part of the weight vector — every weight except the
bias (the constant offset that lets the line sit away from the origin). The bias
shifts where the line sits but is not part of its direction, so it is left out of
the norm; that is exactly why the grader and the experiments compute the geometric
margin with `W[1:]` (all weights after the first). Dividing by $\|\mathbf{w}\|$
cancels the fact that scaling all the feature weights inflates the score without
moving the line. Keep both in mind: the functional margin is the number the loss
acts on, the geometric margin is the picture — the signed distance, on which side.

One sharp warning, because it is the easiest wrong mental model to form here. The
margin — functional or geometric — is a signed *distance-like* quantity. It is
**not a probability.** Logistic regression in L4 squashed its score through a
sigmoid to get a number in $[0, 1]$ you could read as "the model's confidence."
The SVM does no such squashing. The functional margin can be $0.3$ or $5.0$ or
$-2.0$; there is no "70% sure" reading of it. Do not import the probability
intuition from L4 — here a bigger number just means "further from the line, on the
correct side."

The SVM's goal, stated plainly: it wants every point to have a margin of **at
least $1$** — comfortably on the correct side, not just barely past the line — and
among all the weight vectors that achieve that, it wants the one with the
**smallest** $\|\mathbf{w}\|$, the norm of the feature weights. The notation
$\|\mathbf{w}\|$ is the **norm** of a vector (read "the norm of $\mathbf{w}$," its
length: the square root of the sum of its squared entries). Why smallest? Because
the geometric width of the corridor between the classes works out to
$2 / \|\mathbf{w}\|$. A smaller $\|\mathbf{w}\|$ in the denominator means a wider
corridor. **Small weights, wide corridor** — that is the slogan to remember. (The
L2 penalty below is written over the full $\mathbf{W}$ for simplicity; the bias
weight is small in practice and the curriculum keeps the penalty uniform.)

## The hinge loss

We have a goal in words ("every margin at least $1$, with the smallest weights").
To hand it to gradient descent we need it as a **loss function** — a single number
that is large when the weights are bad and small when they are good, so that
walking downhill on it improves the classifier. We build that loss in two pieces:
a penalty for points that fail to clear the margin, and a penalty for large
weights.

The first piece is the **hinge loss**. For one example it is

$$
\ell_i = \max\!\big(0,\; 1 - m_i\big) = \max\!\big(0,\; 1 - y_i (\mathbf{X}_i \cdot \mathbf{W})\big).
$$

Read it aloud: "the loss of example $i$ is the bigger of zero and one-minus-its-
margin." (The symbol $\ell$ is a script lowercase L, our name for a per-example
loss; $\max(a, b)$ just returns whichever of $a$ and $b$ is larger.) Walk through
the two cases:

- If the margin $m_i \ge 1$, the point has cleared its target. Then $1 - m_i$ is
  zero or negative, so $\max(0, 1 - m_i)$ is exactly $0$. The point pays
  **nothing** — and pays nothing no matter how far past the margin it goes.
- If the margin $m_i < 1$ (the point is on the wrong side, or correct but too
  close to the line), then $1 - m_i$ is positive, and the loss grows linearly as
  the point gets worse.

That first bullet is the defining behavior of the hinge, and it is the sharp
contrast with L4's cross-entropy loss. Cross-entropy **never** stops pushing: even
a confidently-correct point keeps contributing a small gradient forever, always
nudging the boundary to be a hair more confident. The hinge **saturates** —
"saturates" meaning its loss flattens out to a constant ($0$ here) and stops
changing once the input is good enough. Cross-entropy does not saturate. This one
difference is why the SVM boundary ends up determined by only a handful of points.
The points that have **cleared** their margin ($m_i > 1$) contribute zero loss and
zero gradient — they are irrelevant to where the line lands. The points that
matter are the ones **on or inside** the margin ($m_i \le 1$): those sitting
exactly on it, plus any soft-margin violators that fell short. Those are the
**support vectors** — the examples that "support" the line, holding it in place.
(In the soft-margin SVM, violators count too, not only the points exactly on the
margin. One implementation detail to note now and use later: our subgradient uses
the **strict** convention, so a point sitting *exactly* on the margin, $m_i = 1$,
contributes zero gradient in the code even though it is a support vector in
principle. That is a valid subgradient choice — more on it below. This
support-vector fact is from CS229; we use it but do not derive it here.)

The second piece is the weight penalty. We average the hinge over all $n$ examples
and add an **L2 penalty** on the weights:

$$
\mathcal{L}(\mathbf{W}) = \frac{1}{n} \sum_{i=1}^{n} \max\!\big(0,\; 1 - y_i (\mathbf{X}_i \cdot \mathbf{W})\big) \;+\; \lambda \, \|\mathbf{W}\|^2.
$$

Read it aloud: "the total loss is the mean hinge over all $n$ examples, plus
lambda times the squared norm of the weights." The symbol $\sum_{i=1}^{n}$ is a
**summation** (read "sum over $i$ from $1$ to $n$" — add up the expression after
it for every example), and $\frac{1}{n}\sum$ makes it a mean. The "L2 penalty"
$\|\mathbf{W}\|^2$ is just the sum of the squared weights — it is large when the
weights are large, so adding it to the loss pushes gradient descent to keep the
weights small. The Greek letter $\lambda$ (lambda) is a number you choose that
sets how hard that push is.

The two pieces pull against each other. The hinge term wants every point past its
margin, even if that means big weights. The L2 term wants $\|\mathbf{W}\|$ small
(a wide corridor), even if that means letting a few points violate their margin.
The setting of $\lambda$ decides the compromise: accept a few violations for a
wider margin, or hug the data tightly with no violations. That tunable compromise
is exactly what makes this the **soft-margin** SVM — "soft" because the margin is
allowed to be violated rather than rigidly enforced — and $\lambda$ is the knob.

### Why fix the target margin at 1 and tune only λ

You might object that pinning the target margin at $\Delta = 1$ (the Greek capital
delta, the name we give the target-margin constant) looks arbitrary — why $1$ and
not $5$? It costs us nothing, because $\Delta$ and $\lambda$ are **redundant**:
they are two knobs that do the same job. Scaling every weight in $\mathbf{W}$ by a
constant trades a larger target margin for a smaller weight penalty and vice
versa, so any $(\Delta, \lambda)$ pair you could pick is equivalent to some
$(1, \lambda')$ pair with a different $\lambda'$. We therefore pin $\Delta = 1$ and
let $\lambda$ alone set the margin-versus-violation trade-off. (If you have seen
the textbook SVM written with a constant $C$ instead, the two map as
$C = \tfrac{1}{2\lambda}$ for the *mean-hinge* objective above: large $C$, the
same as small $\lambda$, punishes violations hard and gives a narrow, data-hugging
margin; small $C$, large $\lambda$, tolerates violations for a wider corridor. A
library like scikit-learn that sums the hinge instead of averaging absorbs the
factor of $n$, so matching its $C$ to your $\lambda$ picks up an extra $n$:
$C = \tfrac{1}{2\lambda n}$ — that is the form the grader's oracle uses below.)

## The subgradient (mind the kink)

To run gradient descent we need the **gradient** of the loss — the vector of its
slopes, one per weight, telling us which way is uphill so we can step the opposite
way. (Reminder from L1: $\nabla$, the "nabla" symbol, denotes the gradient, and
$\partial$ is a partial derivative, the slope in one coordinate.) But the hinge
loss has a problem: at the margin value $m_i = 1$ it has a **kink**.

A kink is a sharp corner in a function where the slope jumps instead of changing
smoothly — picture the point of the letter V. The hinge $\max(0, 1 - m_i)$ is
**flat to the right** of $m_i = 1$ (slope $0$, the loss is pinned at zero) and
slopes **down-and-to-the-left** of it (slope $-y_i\mathbf{X}_i$, the loss rising
as the margin shrinks). At the corner itself the two slopes disagree, so the
function is **not differentiable** there — there is no single well-defined slope.
A plain gradient is undefined at a kink.

The fix is a **subgradient**: at a kink, *any* slope that sits between the
left-slope and the right-slope is a valid stand-in for the gradient, and we are
free to pick one. We adopt the **strict convention** — an example contributes to
the gradient only when it *strictly* violates its margin (margin strictly less
than $1$, i.e. $1 - m_i > 0$); exactly at the kink it contributes nothing. That
choice (slope $0$ at the corner) is one of the valid subgradients, and as we will
see it has a practical payoff for the grader. The full subgradient of the loss is

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{n} \sum_{i:\, 1 - m_i > 0} \big(-\,y_i \mathbf{X}_i\big) \;+\; 2\lambda \mathbf{W}.
$$

Read it aloud: "the slope of the loss in the weights is the mean, over only the
examples that violate their margin, of negative-label-times-features, plus two-
lambda-times-the-weights." The notation $\sum_{i:\, 1 - m_i > 0}$ means "sum only
over the examples $i$ whose margin is violated" — the others drop out entirely,
which is the gradient-side echo of the saturation we saw above. The final
$2\lambda\mathbf{W}$ is the slope of the L2 penalty $\lambda\|\mathbf{W}\|^2$ (the
derivative of a square brings down the factor of $2$).

In code this becomes a boolean mask — a true/false array marking which rows
violate their margin. The subgradient-descent update, step by step:

1. Compute the signed margins for all examples at once: `margins = y * (X @ W)`,
   shape $(n,)$ (one margin per example).
2. Build the violation mask `violated = (1 - margins) > 0`, a boolean array of
   shape $(n,)$ — `True` exactly where the example strictly violates its margin.
   The **strict** `> 0` (not `>= 0`) is what keeps the kink itself out of the sum.
3. For each violated row, form its contribution $-y_i\mathbf{X}_i$; the mask zeros
   out the non-violated rows so they add nothing.
4. Average those contributions over all $n$ examples to get the data part of the
   gradient, shape $(d,)$ (one slope per weight; $d$ is the number of features).
5. Add the L2 term $2\lambda\mathbf{W}$ to get the full subgradient.
6. Take one downhill step: $\mathbf{W} \leftarrow \mathbf{W} - \eta\,\nabla$,
   where $\eta$ (eta) is the learning rate — the same L1 update, only the gradient
   formula changed.
7. Repeat steps 1–6 for the chosen number of steps.

Why does the strict `> 0` matter beyond mathematical taste? Because of how the
grader checks your gradient. It compares your analytic gradient (the formula
above) against a **numerical gradient** — the slow finite-difference estimate from
L1, which nudges each weight by a tiny step and measures how the loss changes,
used here only as a debugging oracle, never to train. A finite-difference check
that happens to land *on* the kink (any margin within the tiny step size of $1$)
gives a meaningless wobble and would flag a false failure. The strict convention,
plus test points deliberately chosen to sit well clear of $m_i = 1$, keeps the
check honest.

## A glimpse past linear: kernels (L6 preview)

Some data is not separable by any straight line, no matter how you fit it — two
classes arranged as concentric rings, for instance, where no line can put one ring
inside and the other outside. The fix is to **lift** the data into a higher-
dimensional space — add new made-up features computed from the originals (say, the
squared distance from the center) so that in the larger space a flat plane *can*
separate the classes — then run this same SVM there. L6 does this explicitly with
**polynomial features** (the new features are products and powers of the original
ones).

Carry one fact forward. The **RBF kernel** you will meet in L6 corresponds to a
lift into an *infinite-dimensional* space — you could never write down or store
the lifted feature vector $\phi(\mathbf{x})$ (the symbol $\phi$, "phi," names the
lifting function) because it has infinitely many entries. That impossibility is
exactly *why* the **kernel trick** matters: it computes the dot products the SVM
needs without ever forming $\phi(\mathbf{x})$ at all. We only state this here; L6
develops it.

## Exercise

Implement in `my_work/L05-svm-gd/svm_gd.py` (run `uv run grade start L05-svm-gd` to get the
scaffold). This is a blind rebuild: you write each function from scratch, then the
grader checks it. The shape convention, curriculum-wide: $n$ rows (examples) and
$d$ features, so $\mathbf{X}$ is $(n, d)$, the labels $y$ are length $n$ in the
**signed** $\{-1, +1\}$ convention, and the weight vector $\mathbf{W}$ is length
$d$. Keep every array in `float64` (64-bit floating point) for precision.

- `mse_grad(X, y, w)` — the **warm-up**, the L2 mean-squared-error gradient
  recalled cold from memory. Parameters: `X` is the design matrix, shape `(n, d)`;
  `y` is the targets, shape `(n,)`; `w` is the weights, shape `(d,)`. Compute the
  residual (prediction minus target) `X @ w - y`, shape `(n,)`, then return the
  gradient $\tfrac{2}{n}\,\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$, shape
  `(d,)` — one entry per weight. It is graded on its own and the SVM functions
  never call it; re-deriving the squared-error gradient keeps it fresh against the
  hinge gradient you are about to write.
- `hinge_loss(W, X, y, lam)` — the soft-margin SVM objective: the mean hinge plus
  the L2 penalty. Parameters: `W` is the weights, shape `(d,)`; `X` is the design
  matrix, shape `(n, d)`; `y` is the labels in $\{-1, +1\}$, shape `(n,)`; `lam`
  is the L2 strength $\lambda$, a number $\ge 0$. Compute the signed margins
  `y * (X @ W)`, shape `(n,)`, the per-example hinge `max(0, 1 - margins)`, take
  the mean, and add `lam * sum(W**2)`. Return one Python `float` scalar. Use the
  single expression $\max(0, 1 - y(\mathbf{X}\cdot\mathbf{W}))$ — there is no
  per-class sum here (that is the named trap below).
- `hinge_grad(W, X, y, lam)` — the strict-`> 0` subgradient of `hinge_loss` with
  respect to `W`. Parameters are identical to `hinge_loss`. Compute the margins,
  build the boolean mask `violated = (1 - margins) > 0`, sum $-y_i\mathbf{X}_i$
  over the violated rows, divide by $n$, and add the L2 term $2\lambda\mathbf{W}$.
  Returns the gradient, shape `(d,)` — the same shape as `W`, one slope per weight.
- `fit(X, y, lr, steps, lam)` — L1's gradient descent with the hinge subgradient
  plugged in. Parameters: `X` is the design matrix, shape `(n, d)`, with the bias
  column of ones already added by the caller; `y` is the labels in $\{-1, +1\}$,
  shape `(n,)`; `lr` is the learning rate $\eta$ (step size); `steps` is how many
  full-batch updates to take; `lam` is the L2 strength $\lambda$. Start the weights
  at zero and run the numbered update loop above. Returns the fitted weight vector,
  shape `(d,)`.

Then grade it: the site's Run Grader button or `uv run grade L05-svm-gd`.

## How the grader checks you

- The **warm-up** `mse_grad` is an independent unit: it is graded and recorded on
  its own and never blocks the rest. It is run on a fixed $3\times 2$ matrix with a
  known closed-form gradient, and your result must match
  $\tfrac{2}{n}\,\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ in both shape
  `(2,)` and value. A wrong warm-up does not hide a working `hinge_loss`.
- `hinge_loss` is checked with a binary sanity value: at a near-zero $\mathbf{W}$
  the signed margins are all $\approx 0$, so every per-example hinge is
  $\max(0, 1 - 0) = 1$ and the mean must come out $\approx 1.0$ (the grader accepts
  $0.9$ to $1.1$). This is the binary, signed-label analogue of the multi-class
  "$\approx 9.0$ for 10 classes" check — and it is also where the named trap below
  is caught. `hinge_loss` is the foundation the rest of the chain stands on, so it
  is checked first.
- `hinge_grad` is checked two ways. **Shape:** the returned gradient must be shape
  `(d,)`, one entry per weight. **Value:** your analytic subgradient is compared
  against a **central finite-difference** numerical gradient on test points chosen
  to sit well clear of the kink, and the relative error must stay below $10^{-4}$.
  (The numerical gradient is the slow per-parameter debug oracle from L1, never
  used to train.) A guard fires first if any test margin lands within the
  finite-difference step of the kink — see the named trap below.
- `fit` is checked two ways. First, the **oracle**: your fitted classifier must
  agree with scikit-learn's `LinearSVC` (same hinge-plus-L2 SVM, with sklearn's
  $C$ set to $\tfrac{1}{2\lambda n}$ to match your $\lambda$) on more than $85\%$ of
  a fixed held-out 100-row slice of MNIST 3-vs-5 digits (observed agreement is
  $\approx 0.95$). It compares the **decisions** (which side each point lands on),
  not the weight vectors. scikit-learn appears *only* inside this oracle — an
  external answer key — never in your own module. Second, the **margin-width**
  check: on the seeded 2-D `blobs-outliers` data with a generous $\lambda = 0.01$,
  the minimum geometric margin (distance from the boundary to the closest point)
  must come out $> 0.5$ — proof your SVM opened a wide corridor instead of hugging
  the data.

### The named trap: a kink-blind gradient check

This grader is built to catch one mistake by name. If you compute `hinge_grad`
with a **one-sided finite difference** — $\tfrac{\text{loss}(W + h) - \text{loss}(W)}{h}$,
stepping each weight forward only — instead of the analytic strict-`> 0`
subgradient, two things go wrong. It returns a scalar rather than a per-weight
vector, and worse, a forward step can land *across* the non-differentiable kink at
margin $= 1$, where a one-sided difference wobbles and gives a garbage slope. The
grader's central-difference check disagrees sharply and names the cause as the
hinge **kink**, so you know to switch to the analytic subgradient (mask on
$1 - m_i > 0$, contribution $-y_i\mathbf{X}_i$, plus $2\lambda\mathbf{W}$) rather
than hunting a phantom bug.

(There is a second named trap, on `hinge_loss`: if you sum over a true-class index
`j=y_i` as if this were the *multi-class* hinge, you keep the true class's own term
and lay a constant $\Delta = 1$ floor on every example, so the loss never falls
toward $\approx 1.0$ even at a near-zero $\mathbf{W}$. In the common vectorized
form of this mistake the inflation is worse still — a length-$n$ score broadcast
against the labels builds an $n \times n$ pairwise matrix and sums $n$ hinge terms
per example (you can watch this hit $\approx 40$ in Experiment 3). The binary hinge
is **one expression** $\max(0, 1 - y(\mathbf{X}\cdot\mathbf{W}))$ with no per-class
sum; the grader's value check on `hinge_loss` fires the `j=y_i` message whenever
this inflation appears.)

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the formula with shapes, then runnable
pseudocode. Try level 1 before climbing — the level-2 hints carry the exact shapes
(for example, that the violation contribution stacks `(violated * -y)[:, None] * X`
to $(n, d)$ and sums over the $n$ axis to $(d,)$). If your numbers desync from the
worked module, work the subgradient by hand on a 2-example case and find the first
line where your on-paper slope diverges from your code — usually a dropped $y$
factor, a missing strict `> 0`, or the lost $2\lambda\mathbf{W}$ term.

## Done when

`uv run grade L05-svm-gd` shows all-green: the warm-up `mse_grad` matches in shape and value,
`hinge_loss` returns $\approx 1.0$ at a near-zero $\mathbf{W}$ (no $\Delta = 1$
floor), `hinge_grad` is shape `(d,)` and matches the central finite-difference
gradient to $10^{-4}$ without tripping the kink guard, and `fit` agrees with
scikit-learn's `LinearSVC` on more than $85\%$ of the held-out MNIST slice and
opens a geometric margin $> 0.5$ on the blobs. Then run the experiments and record
your predictions.
