# L6 — Feature Maps & Kernels

Back in L3 the perceptron hit a wall called **XOR** — four points whose labels no
single straight line can split — and the lesson made a promise: *lift the features
so the data becomes separable in a richer space.* This lesson keeps that promise.
A **feature map** is a function that takes each data point and adds new
coordinates computed from the ones it already has. Done right, a map turns a
problem with no straight-line answer into one that has a perfectly good straight-line
answer — just in a space with more dimensions.

The SVM you built in L5 (the "support vector machine" — the classifier that draws
the *widest* straight corridor between two classes) is the engine we will reuse,
unchanged. We only change what we feed it.

To make the problem concrete, take the **circles** dataset: an inner ring of
points labeled $+1$ surrounded by an outer ring labeled $-1$. (Throughout this
course labels are **signed**, meaning each point's class is written as the number
$+1$ or $-1$, never as $0/1$ — that is the convention the hinge loss expects.)
No straight line separates a ring from the ring around it: rotate the line
however you like and one of the two classes always straddles it. The linear SVM
is stuck, exactly the way the perceptron was stuck on XOR.

The fix is not a smarter optimizer. It is a better **representation** — a better
choice of what numbers describe each point. If we map every point into a richer
space where the answer *is* a straight line, the same L5 SVM works with no
changes. That map is the feature map, written $\phi(\mathbf{x})$ (the Greek
letter "phi", read "fie"; $\mathbf{x}$ in bold is one data point, a small vector
of its coordinates). This lesson makes $\phi$ explicit — you will write out its
columns by hand and see the circles fall apart along a clean boundary.

## Warm up on XOR: a map you can check by hand

Before the circles, here is the smallest possible example, worked end to end, so
you can see *why* a lift works on a problem with no straight-line answer. This is
the XOR problem the perceptron failed on. The four points and their labels:

$$
(0,0) \to -1, \quad (0,1) \to +1, \quad (1,0) \to +1, \quad (1,1) \to -1.
$$

Read aloud: a point gets $+1$ when its two coordinates *differ* and $-1$ when
they *match*. Plot them on a unit square and the two $+1$ points sit on one
diagonal, the two $-1$ points on the other — interleaved, so no line splits them.

Now add **one** new coordinate: the product $x_1 x_2$ of the two inputs. That is
a tiny feature map, $\phi(x_1, x_2) = (x_1,\; x_2,\; x_1 x_2)$ — keep the two raw
coordinates and append their product. Apply it to all four points:

$$
\begin{aligned}
(0,0) &\to (0,\,0,\,0), \quad &y=-1 \\
(0,1) &\to (0,\,1,\,0), \quad &y=+1 \\
(1,0) &\to (1,\,0,\,0), \quad &y=+1 \\
(1,1) &\to (1,\,1,\,1), \quad &y=-1.
\end{aligned}
$$

Look at the new third coordinate, $x_1 x_2$: it is $1$ only for the troublesome
$-1$ point $(1,1)$ and $0$ for the other three. That single new number pulls
$(1,1)$ off to one side, which is the foothold the lift gives us — even though
$x_1 x_2$ alone does not finish the job (the other $-1$ point $(0,0)$ also has
$x_1 x_2 = 0$, so we still need the raw coordinates too). Combining all three, the
single rule "predict $-1$ when $x_1 + x_2 - 2 x_1 x_2 < \tfrac12$, else $+1$"
separates them — and that rule is *linear* in the lifted coordinates
$(x_1, x_2, x_1 x_2)$ (its weights are $1$, $1$, and $-2$). Check it on $(1,1)$:
$1 + 1 - 2(1) = 0 < \tfrac12$, predict $-1$, correct. On $(0,1)$:
$0 + 1 - 0 = 1$, not $< \tfrac12$, predict $+1$, correct. A straight cut in the
3-D lifted space is a *curved* cut back in the original 2-D plane — and the curve
is exactly what XOR needed.

That is the whole idea in miniature: **a non-linear feature can turn a problem
with no linear answer into one with a linear answer, because the extra coordinate
gives a flat boundary somewhere new to sit.** The rest of the lesson does the
same trick on the circles, with a map you choose by reasoning about the data.

## Why a lift separates the circles

The two rings of the circles dataset differ in exactly one thing: their distance
from the center, the **radius**. A point's squared radius is

$$
r^2 = x_1^2 + x_2^2,
$$

read "r-squared equals x-one squared plus x-two squared" — the sum of the squares
of the two coordinates. (We use $r^2$ rather than $r$ to avoid a square root;
squaring is monotonic, so thresholding on $r^2$ is the same as thresholding on
$r$.) Notice $r^2$ is a *quadratic* of the inputs — it multiplies coordinates
together — so it does not exist as a feature in the raw $(x_1, x_2)$ plane. The
inner and outer rings have small and large $r^2$ respectively, but the linear SVM
can only add and scale $x_1$ and $x_2$; it can never form $x_1^2 + x_2^2$ on its
own. So we **lift** each point into a space that *includes* the squared terms:

$$
\phi(x_1, x_2) = \big(\, x_1,\; x_2,\; x_1^2,\; x_1 x_2,\; x_2^2,\; 1 \,\big).
$$

Read aloud: phi of a point is the two raw coordinates, then the three degree-2
combinations (the two squares $x_1^2$ and $x_2^2$, and the **cross term**
$x_1 x_2$ — the two coordinates multiplied together), then a $1$. This is the
**degree-2 polynomial feature map** — "degree-2" because the highest power of any
single input is $2$, "polynomial" because every column is a product of inputs.
The trailing $1$ is the **bias column**: a constant feature that lets the boundary
sit anywhere, not just pass through the origin (the same role the bias played in
L5). Six columns out of two inputs. A point $(3, 4)$ lifts to
$(3,\,4,\,9,\,12,\,16,\,1)$ — the raw $3$ and $4$, then $3^2=9$, $3 \cdot 4 = 12$,
$4^2 = 16$, and the bias $1$.

Why this exact list, and not just $x_1^2 + x_2^2$ on its own? Because we do not
know in advance that radius is the *only* thing that matters. By handing the SVM
all the degree-2 terms, we let it discover the right combination: it can dial up
the $x_1^2$ and $x_2^2$ columns to build the radius direction and dial the others
toward zero. We supply the raw material; the optimizer picks the recipe.

In this lifted space a **linear** boundary, written
$\mathbf{W} \cdot \phi(\mathbf{x}) = 0$, can lean on the $x_1^2 + x_2^2$
direction to threshold on radius. ($\mathbf{W}$ is the weight vector — one number
per column of $\phi$ — and the dot $\cdot$ is the **dot product**, the sum of
weight-times-feature over all columns; the boundary is where that sum is exactly
zero.) Translate that flat boundary back into the original plane and it becomes a
**circle** — exactly the shape that fences the inner ring off from the outer one.
We did not change the SVM. We changed what we feed it.

## The same SVM, in the lifted space

Lifting is a **preprocessing step**: a transformation you apply to the data once,
before training, the way you might scale or center it. The procedure is short:

1. Compute the lifted matrix $\phi(\mathbf{X})$ once — apply the feature map to
   every row of your data, turning the $n \times 2$ raw matrix into an
   $n \times 6$ matrix (for degree 2). Here $n$ is the number of data points and
   each row is one lifted point.
2. Run the *identical* L5 soft-margin SVM on $\phi(\mathbf{X})$ in place of
   $\mathbf{X}$ — same hinge-plus-L2 objective, same gradient descent.
3. To classify a new point, lift it with the same $\phi$, then apply the learned
   straight-line rule in the lifted space.

The thing the SVM minimizes (its **loss**, the single number that measures how
badly the current weights misclassify the training data) is unchanged:

$$
L(\mathbf{W}) = \frac{1}{n} \sum_i \max\!\big(0,\; 1 - y_i (\phi(\mathbf{X}_i) \cdot \mathbf{W})\big) + \lambda \, \|\mathbf{W}\|^2.
$$

Read aloud: the loss is the *average over all points* of the **hinge** —
$\max(0, 1 - \text{margin})$, which charges nothing once a point is on the correct
side by a full unit and charges linearly as it slips toward or past the boundary —
plus $\lambda$ (the Greek "lambda", a small fixed number you choose) times
$\|\mathbf{W}\|^2$, the **squared norm** of the weights (read "norm-W-squared":
the sum of the squares of every weight, a penalty that keeps the weights from
blowing up). The symbol $y_i$ is the signed label of point $i$, and
$\phi(\mathbf{X}_i) \cdot \mathbf{W}$ is that point's score. The product
$y_i \times \text{score}$ is the **margin**: positive when the point is on the
right side, negative when wrong.

Two things to keep straight, because both are easy to misread:

- **The loss is the same engine, the data is the only difference.** Gradient
  descent does not know or care that $\phi(\mathbf{X})$ came from a lift — it sees
  a design matrix and a label vector, full stop. The optimizer is loss-agnostic;
  swapping raw $\mathbf{X}$ for $\phi(\mathbf{X})$ changes the *numbers* it
  descends on, not the *procedure*.
- **The regularization constant maps back exactly as in L5**, $C = 1/(2\lambda)$,
  where $C$ is the textbook SVM constant (large $C$ = punish margin violations
  hard). Nothing here is new; we only renamed the input.

The degree is a **knob** you turn. Degree $1$ is the **identity lift**
$(x_1, x_2, 1)$ — just the raw features plus a bias, no new combinations — so
fitting it is identical to running L5 on the raw data, and the circles still fail.
That is the point: a lift only helps if the columns it adds are the ones the data
needs. Degree $2$ adds the squares and the cross term, supplying the radius
direction, and the circles separate. The interleaving **moons** dataset (two
half-moon arcs that hook into each other) needs a curvier boundary than a single
circle; a degree-$3$ lift, which *appends* the four degree-3 monomials
$x_1^3,\; x_1^2 x_2,\; x_1 x_2^2,\; x_2^3$ (every product of three inputs), gives
the linear SVM enough freedom to bend that far.

## Where explicit lifting stops: the kernel trick

Everything above relied on being able to *write down the columns* of $\phi$. A
polynomial lift has a **finite** list of columns — six for degree 2, ten for
degree 3 — so you can materialize $\phi(\mathbf{x})$ (actually build the matrix
in memory) and run an ordinary linear SVM on it. That is exactly what you do in
this lesson. But the single most-used kernel does not work that way, and it is
worth knowing why.

First, the word **kernel**. A kernel is a function $K(\mathbf{x}, \mathbf{z})$
that returns the **inner product** (another name for the dot product) of two
points *after* they have been lifted —
$K(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle$ —
but computes that number with a shortcut formula, *without* first building either
$\phi(\mathbf{x})$ or $\phi(\mathbf{z})$. The angle brackets $\langle \cdot, \cdot \rangle$
just denote that inner product.

Why would you ever want the shortcut? Consider the **RBF (Gaussian) kernel**,

$$
K(\mathbf{x}, \mathbf{z}) = \exp\!\big(-\gamma \, \|\mathbf{x} - \mathbf{z}\|^2\big),
$$

read "the exponential of minus gamma times the squared distance between the two
points" ($\gamma$, "gamma", is a width knob; $\|\mathbf{x} - \mathbf{z}\|^2$ is
the squared Euclidean distance between them). This kernel corresponds to a feature
map $\phi$ with **infinitely many columns**. There is no finite list to write
down — you literally *cannot* materialize $\phi(\mathbf{x})$. That impossibility
is exactly **why the kernel trick matters**: the trick computes the inner products
$\langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle = K(\mathbf{x}, \mathbf{z})$
through the shortcut formula above, *without ever forming* $\phi$, so an
infinite-dimensional lift becomes computable after all.

Two honest boundaries on this lesson, so you know what you are and are not
building:

- This is a stated fact (it comes from Stanford's CS229), **not** something you
  implement here. The kernel trick needs the **dual** form of the SVM — a
  different way of writing the same optimization, rewritten so the only thing it
  ever touches is inner products between points — and that derivation we
  deliberately skip. You only do explicit, finite lifts in this lesson.
- One principle carries straight over from L5: only the **support vectors** —
  here, the points on *or inside* the margin (the ones the hinge still charges:
  on the margin, between it and the boundary, or misclassified) — determine the
  boundary. Every point safely *past* its margin contributes exactly zero to the
  hinge gradient, in the lifted space the same as in the raw space, so it has no
  pull on the weights. The principle holds before and after the lift; *which*
  points end up being the support vectors can change, because the lift reshapes
  the geometry, but the rule "only the points the hinge still charges hold the
  boundary" is the same.

## Exercise

Implement in `my_work/feature-maps/feature_maps.py` (run `grade start
feature-maps` to get the scaffold):

- `hinge_loss(W, X, y, lam)` — the **warm-up**, the L5 hinge-plus-L2 loss recalled
  cold. Parameters: `W` is the weight vector, shape `(d,)` (one weight per feature
  column; `d` is the number of columns); `X` is the design matrix, shape `(n, d)`
  (`n` rows, one per data point); `y` is the label vector, shape `(n,)`, in the
  signed convention `{-1, +1}`; `lam` is the L2 regularization strength $\lambda$,
  a number `>= 0`. It returns a single scalar (a plain Python `float`): the mean
  over points of $\max(0,\, 1 - y_i (X_i \cdot W))$ plus $\lambda \sum W^2$. It is graded on its
  own and is *not* called by the functions below — re-authoring the hinge primal
  from memory keeps it fresh before you lift.
- `poly_features(X, degree=2)` — the new mechanism, the explicit polynomial lift.
  Parameters: `X` is the input matrix, shape `(n, 2)` (the 2-D synthetic data);
  `degree` is the highest polynomial degree to expand, one of `1`, `2`, or `3`. It
  returns $\phi(X)$, shape `(n, k)` float64, where `k = 3` for degree 1, `6` for
  degree 2, `10` for degree 3. The columns, in order, are the raw features, then
  each higher degree's monomials appended, then the **bias column of ones LAST**.
  Degree 2: `(x1, x2, x1^2, x1*x2, x2^2, 1)`. Hand-check your output against
  `(3, 4) -> (3, 4, 9, 12, 16, 1)`.
- `fit(X, y, lr, steps, lam)` — **provided, do not modify.** This is the L5 linear
  soft-margin SVM, inlined so the lesson is self-contained. You pass it
  $\phi(X)$ as `X` and it returns the fitted weight vector `W`, shape `(d,)`. The
  only new code you write is the lift; the SVM is the engine you already built.

Then grade it: the site's Run Grader button or `uv run grade L06-feature-maps`.

## How the grader checks you

- The **warm-up** `hinge_loss` is an independent unit: graded and recorded on its
  own, it never blocks the lift functions. The check feeds it a tiny near-zero
  weight vector, where every signed margin $y_i(X_i \cdot W)$ is about $0$, so each
  per-example hinge is $\max(0, 1 - 0) = 1$ and the mean must come out near $1.0$
  (the grader accepts the range $0.9$ to $1.1$). A wrong warm-up does not hide a
  working lift.
- `poly_features` is checked first for **shape**: a degree-2 lift of a random
  `(13, 2)` input must return `(13, 6)`. A wrong column count usually means a
  missing cross term, a missing square, or a forgotten bias column.
- Then for **value**: the hand-checked row `(3, 4)` at degree 2 must equal exactly
  `(3, 4, 9, 12, 16, 1)`, which pins both the column order (raw, then squares and
  the cross term, then bias last) and that the cross term is $x_1 x_2$, not
  $x_1 + x_2$.
- **Circles separability**: fitting the provided SVM on the degree-2 lift of the
  circles (with the lesson's pinned `lr=0.01, steps=5000, lam=0.001`) must reach
  more than `0.90` held-out accuracy. The worked solution reaches `1.00`. If the
  squared terms are missing or you fit on raw `X` instead of $\phi(X)$, this fails.
- **Degree-1 identity baseline**: `poly_features(X, degree=1)` must equal the raw
  features with a bias column appended, `(x1, x2, 1)`, *and* fitting on it must
  produce the same weights as fitting on that raw-plus-bias matrix directly. This
  confirms degree 1 adds nothing — the baseline the experiments break against.
- **Moons separability**: the degree-3 lift of the moons must clear more than
  `0.85` held-out accuracy (the worked solution reaches `0.95`), which only works
  if you actually append the degree-3 monomials.

This lesson has **no named trap** — there is no `traps.yaml` — because the shape
and value checks already pin down the common mistakes (wrong column count, wrong
order, missing cross term). The grader's own messages name whatever they catch.

## The hint ladder

If the grader's message is not enough, open the hints. `hinge_loss` and
`poly_features` each have three levels, mildest first: a conceptual nudge, then
the formula with shapes annotated, then pseudocode you can almost type. Try level
1 and the hand-check `(3, 4) -> (3, 4, 9, 12, 16, 1)` before climbing — the order
of the columns is the thing most worth getting right by reasoning, not by peeking.

## Done when

`uv run grade L06-feature-maps` shows all-green: the warm-up `hinge_loss` returns near $1.0$
at a near-zero weight vector; `poly_features` returns the right shape and the
exact hand-checked row, with degree 1 reproducing the raw-plus-bias identity lift;
the degree-2 lift separates the circles above $0.90$ held-out accuracy; and the
degree-3 lift separates the moons above $0.85$. Then run the experiments and
record your predictions.
