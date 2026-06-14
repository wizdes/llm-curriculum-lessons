# L2 — Linear Regression

In L1 you built a general optimization engine — gradient descent — and watched
it walk downhill on toy bowls. This lesson points that same engine at a real
dataset (house prices) and a real model. Nothing about the engine changes; what
is new is the model we are fitting and the loss whose gradient we feed in.

First, the one idea this lesson exists to get right, stated plainly so a common
mix-up never takes root.

## Three things, not one: model, loss, solver

People say "linear regression is gradient descent on MSE." That sentence jams
three separate things into one, and the confusion it breeds slows everything
after. Keep them apart from the start:

- The **model** is *linear regression itself*: the rule that turns an input into
  a prediction. Here that rule is "take a weighted sum of the input's features."
  The model is a *shape* of function — a straight line, or its many-feature
  cousin, a flat plane — with adjustable numbers (the weights) you can set.
- The **loss** is **MSE** — mean squared error, the score from L0 that says how
  wrong the predictions are right now. The model could be scored by other losses;
  MSE is the one we choose here. The loss is *not* part of the model. It sits
  outside, judging it.
- The **solver** is the procedure that picks the weights that make the loss
  smallest. **Gradient descent is one solver.** It is the iterative one from L1:
  feel the slope, step downhill, repeat. It is *a way to fit* linear regression,
  not the definition of it.

So linear regression is the model; MSE is the loss we score it with; gradient
descent is one solver that finds the best weights. The headline "linear
regression is GD on MSE" collapses a model, a loss, and a solver into a single
phrase — which is exactly the trap. We will see at the end that there is a
*second* solver for this same model and loss, one that needs no learning rate
and no iteration at all, and yet lands on the identical weights. That second
solver is what proves the model is not the same thing as gradient descent.

## The model: a weighted sum of features

A house in our dataset is described by a handful of numbers — its median income,
its average number of rooms, and so on. Each such number is a **feature**: one
measured property of the example. We collect a row's features into a vector
$\mathbf{x}$ (bold means "a list of numbers, not just one"). The model predicts
the house's value with a **weighted sum**: multiply each feature by its own
weight and add the results up.

$$
\hat{y} = \mathbf{x} \cdot \mathbf{w}
$$

Read aloud: *the prediction is the dot product of the feature vector and the
weight vector.* The hat on $\hat{y}$ (say "y-hat") marks it as the model's guess,
as opposed to the plain $y$, the true value. The **dot product**
$\mathbf{x} \cdot \mathbf{w}$ means "multiply the two vectors entry by entry, then
sum" — feature 1 times weight 1, plus feature 2 times weight 2, and so on. So a
weight
is the importance the model assigns to its feature: large weight, big influence
on the prediction; near-zero weight, the feature barely matters.

This is the **model**. Unlike the kNN model from L0, which kept every training
row around to compare against, linear regression memorizes nothing. Once
training finishes, the model's entire knowledge lives in the weight vector
$\mathbf{w}$ — a few numbers. The training rows can be thrown away.

## Vectorization: from a loop to a matrix

We have not one house but $n$ of them, and we want all $n$ predictions at once. A
beginner reaches for a Python loop — predict row 1, then row 2, and so on. Don't.
Stack all $n$ rows into one big matrix, the **design matrix** $X$, of shape
$(n, d)$: $n$ rows, one per house, and $d$ columns, one per feature. ("Design
matrix" is the standard name for the input arranged this way — there is nothing
designed about it; it is just the data laid out as a grid.) Then every prediction
falls out of a single matrix-vector product:

$$
\hat{\mathbf{y}} = X \mathbf{w}
$$

Read aloud: *the whole vector of predictions equals the design matrix times the
weight vector.* The shapes line up like this: $(n, d) \times (d,) \to (n,)$ — a
matrix with $d$ columns times a vector of length $d$ gives a vector of length
$n$, one prediction per row. Row $i$ of the output is exactly row $i$ of $X$
dotted with $\mathbf{w}$, which is the single-row formula from above, computed
for every row in one shot.

This is not just tidier. It is the difference between a model that trains in a
second and one that crawls: NumPy runs that one matrix product in fast compiled
code, while a Python loop pays interpreter overhead on every row. Every operation
in this course stays vectorized for the same reason.

Concrete check, small enough to verify by hand. Take two houses with two features
each,

$$
X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad
\mathbf{w} = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}.
$$

Row 0 dotted with $\mathbf{w}$ is $1\cdot 1 + 2\cdot 0.5 = 2$; row 1 is
$3\cdot 1 + 4\cdot 0.5 = 5$. So $X\mathbf{w} = (2,\ 5)$ — one number per row.
That is precisely what your `predict` function must return.

## The loss: mean squared error

To improve the weights we first need to score them. The **loss** measures how
wrong the current predictions are; smaller is better, zero is perfect. We use
**MSE** (mean squared error), the loss from L0: take each prediction's miss
(prediction minus truth), square it, and average over all $n$ houses.

$$
L(\mathbf{w}) = \frac{1}{n} \lVert X\mathbf{w} - \mathbf{y} \rVert^2
= \frac{1}{n} \sum_{i=1}^{n} (\mathbf{x}_i \cdot \mathbf{w} - y_i)^2
$$

Read aloud: *the loss is the average, over all $n$ houses, of the squared
difference between the prediction and the true value.* Two notations to unpack.
The double-bar $\lVert \cdot \rVert$ is the **vector norm** — the length of a
vector — and $\lVert \mathbf{v} \rVert^2$ is the sum of the squares of its
entries; here the vector is $X\mathbf{w} - \mathbf{y}$, the list of all $n$
misses, so its squared norm is exactly the sum of squared misses. The $\sum$
(Greek capital sigma) on the right means "add up," with $i$ indexing the houses:
$\mathbf{x}_i$ and $y_i$ are the $i$-th row's features and true value. The two
forms are the same number written compactly (left) and spelled out term by term
(right).

We **square** each miss for two reasons, both from L0: squaring makes every miss
positive, so overshooting by 2 and undershooting by 2 both count as error instead
of cancelling, and it punishes a big miss far more than a small one. The loss is
written $L(\mathbf{w})$ — a function *of the weights* — because that is the knob
we will turn: change $\mathbf{w}$, change the predictions, change the score.

Numeric check, four predictions. Suppose the true values are $(1, 2, 3, 4)$ and
the predictions are $(1, 1, 4, 6)$. The misses (prediction minus truth) are
$(0, -1, 1, 2)$; squared they are $(0, 1, 1, 4)$; the average of those four is
$6/4 = 1.5$. That is what your `mse_loss` returns — a single number.

## The gradient: differentiate the loss, not the model

To turn the weights with gradient descent we need the **gradient** of the loss:
the list of slopes saying how the loss changes as we nudge each weight. Here is
the point that trips up almost everyone, carried over from L1 and worth saying
again in this new setting: **we differentiate the loss, not the model.** The
model $X\mathbf{w}$ just makes predictions; "the slope of the model" is not a
thing training cares about. What we take the slope of is the *loss* $L$, and we
take it *with respect to the weights* $\mathbf{w}$ — because the weights are what
we get to change. The gradient answers exactly one question: *if I nudge each
weight a little, how does the loss move?*

Differentiating the loss $L(\mathbf{w}) = \frac{1}{n}\lVert X\mathbf{w} - \mathbf{y} \rVert^2$ with respect to $\mathbf{w}$ uses the same chain rule you used in L1.
Each squared miss is $(\mathbf{x}_i \cdot \mathbf{w} - y_i)^2$; its derivative
pulls a factor of $2$ down from the square and an $\mathbf{x}_i$ out of the
inside (the derivative of the miss with respect to $\mathbf{w}$). Collected over
all rows, the entry-by-entry version becomes one clean matrix expression:

$$
\nabla_{\mathbf{w}} L = \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y})
$$

Read aloud: *the gradient is two-over-$n$ times $X$-transpose times the residual
vector.* The symbol $\nabla$ is "nabla"; read $\nabla_{\mathbf{w}} L$ as "the
gradient of $L$ with respect to $\mathbf{w}$" — the list of one slope per weight.
$X^\top$ is **$X$ transposed** (rows and columns swapped), shape $(d, n)$; the
**residual** $X\mathbf{w} - \mathbf{y}$ is the vector of all $n$ misses, shape
$(n,)$. Multiplying them, $(d, n) \times (n,) \to (d,)$, gives one number per
weight — exactly the shape the gradient must be. The $\frac{2}{n}$ out front is
the $2$ from the square and the $\frac{1}{n}$ from the average; keep it, because
the gradient check in the grader compares against this *exact* analytic formula
and will flag a missing constant.

That single line is the function `mse_grad`. With it in hand, one step of the
solver is exactly L1's update rule, unchanged:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta\, \nabla_{\mathbf{w}} L
$$

Read aloud: *the new weight vector is the old one minus the learning rate $\eta$
times the gradient.* The arrow means "becomes"; $\eta$ (Greek "eta") is the
**learning rate**, the step size from L1; we subtract because the gradient points
uphill and we want to go down. This is the same engine — only the gradient we
hand it is new. (This is the general pattern of the whole course: the optimizer
is loss-agnostic. Swap in a different model and loss, and the only thing that
changes is the gradient formula you feed the same descent loop.)

## The bias term

A weighted sum $\mathbf{x} \cdot \mathbf{w}$ has a quiet limitation: when every
feature is $0$, the prediction is forced to $0$. Geometrically the fitted line
(or plane) is pinned through the origin and cannot be lifted up or down. But a
target rarely averages to zero — house values do not — so the model needs a way
to set its overall level.

The fix is the **bias** (also called the **intercept**): a constant the model can
add to every prediction, the "+ b" that lets the line cross the vertical axis
wherever it likes. The trick that keeps the math uniform is to **prepend a column
of ones** to the design matrix. That new column 0 is always $1$, so its weight
$w_0$ gets added to every prediction no matter the features — which is exactly
what a bias does. After this, the bias is just another coordinate of $\mathbf{w}$,
the same gradient formula handles it with no special case, and a design matrix
with $d$ real features becomes $(n, d+1)$.

A note on responsibility: `fit` does **not** add this column for you. The caller
builds the design matrix — standardize the features, then `np.hstack` a column of
ones in front — and hands the finished $(n, d+1)$ matrix to `fit`. The exercise
spells out exactly how.

## Why gradient descent demands feature scaling

Raw features live on wildly different scales. In the California housing data, the
block population runs into the thousands (and tens of thousands) while the median
income and average rooms sit in the low single digits. Each feature enters the
prediction through its own multiply (its weight), so a
feature on a large scale produces a large gradient component and a feature on a
small scale a tiny one. The squared loss then becomes a long, narrow valley
instead of a round bowl — steep across the large-scale direction, nearly flat
along the small-scale one.

This is the **conditioning** problem you met in L1's $x^2 + 100y^2$ bowl, now
showing up for a concrete reason. A single learning rate $\eta$ has to serve
every direction at once, and it cannot: set it small enough to stay stable on the
steep wall and it barely budges along the shallow floor (the model crawls); set
it large enough to make progress on the floor and it overshoots the steep wall
and diverges (the loss blows up to infinity). The mismatch in feature scales is
what stretches the bowl, and a stretched bowl is what no single step size can
descend cleanly. (A well-conditioned bowl is round; the closer to round, the more
forgiving the learning rate.)

The fix is **standardization**: rescale each feature, column by column, to have
mean $0$ and standard deviation $1$.

$$
X_{\text{std}}[:, j] = \frac{X[:, j] - \mu_j}{\sigma_j}
$$

Read aloud: *for feature $j$, subtract that feature's mean $\mu_j$ from every
value and divide by that feature's standard deviation $\sigma_j$.* The notation
$X[:, j]$ is NumPy for "column $j$ — every row, that one feature." The mean
$\mu_j$ (Greek "mu") is the feature's average; the standard deviation $\sigma_j$
(Greek "sigma") is its typical spread around that average. Subtracting the mean
re-centers the column on $0$; dividing by the spread makes every column's spread
$1$. Now no feature is intrinsically "bigger" than another, every direction of
the loss has comparable curvature, the valley is round, and one learning rate
descends every direction cleanly. The function `standardize` does exactly this,
and you will *feel* what skipping it costs in the first experiment.

## Fitting: gradient descent end to end

Putting the pieces together, `fit` is the L1 descent loop with `mse_grad` as its
gradient. As a procedure:

1. **Start the weights at zero**: `w = np.zeros(d)`, where $d$ is the number of
   design columns (real features plus the bias column). Zero is the neutral,
   assumption-free starting point.
2. **Compute the gradient at the current weights**: `g = mse_grad(X, y, w)`,
   the exact analytic formula $\frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y})$,
   shape $(d,)$.
3. **Step downhill**: `w = w - lr * g` — subtract a fraction `lr` of the gradient
   from every weight at once.
4. **Repeat** steps 2–3 for `steps` iterations, then **return** the final `w`.

Because the features were standardized, the bowl is round, so a stable learning
rate (say `lr=0.1`) and a couple thousand steps walk straight to the bottom — the
single best weight vector. Note that `fit` here inlines its own loop rather than
calling the warm-up `gradient_descent`; they are the same algorithm, but keeping
`fit` self-contained matters for how this lesson is graded (more in the
exercise).

## Train / test split as honesty

The loss measured on the rows you trained on always looks good — the weights were
tuned to those exact rows. The honest question is how the model does on rows it
has *never seen*, because that is the only situation that matters in practice.
**Holding out a test split** — setting aside some rows, never letting `fit` touch
them, and scoring on them at the end — is how you keep yourself honest. The gap
between training loss and test loss is the gap between "memorized these rows" and
"learned the pattern." The experiments make you feel that difference directly.

## What this hides: the second solver

Everything above fits linear regression with *one* solver, gradient descent. But
this particular model-and-loss is special: it has a **closed-form solution** — a
formula that lands on the best weights in a single calculation, no iteration and
no learning rate. It is called the **Normal Equation**:

$$
\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}
$$

Read aloud: *the best weights equal the inverse of ($X$-transpose times $X$),
times $X$-transpose times $\mathbf{y}$.* The $(\cdot)^{-1}$ is the **matrix
inverse** (the matrix analogue of dividing by a number). You compute it as a
short procedure:

1. Form $X^\top X$, a $(d, d)$ matrix.
2. Invert it: $(X^\top X)^{-1}$, still $(d, d)$.
3. Multiply by $X^\top \mathbf{y}$ to get $\mathbf{w}$, shape $(d,)$.

This is not magic, and it is not a different model. It is the algebra trick from
L1's "why iterate?" discussion, carried all the way through: set the gradient to
zero, $\frac{2}{n} X^\top(X\mathbf{w} - \mathbf{y}) = 0$, and *solve that equation
for $\mathbf{w}$* — the rearrangement is exactly the Normal Equation. Gradient
descent does not solve the equation; it crawls toward the same answer the slope
points to. Both reach the **identical** weights, because the MSE loss of a linear
model is **convex** — one single bowl, one lowest point, no other minima to land
in. (That is the deeper reason a slope formula is enough to find the bottom here:
there is only one bottom.)

So why bother with gradient descent at all if an exact one-shot formula exists?
**Cost.** Inverting the $(d, d)$ matrix in step 2 takes on the order of $d^3$
arithmetic operations — written $O(d^3)$, read "order $d$-cubed," meaning the
work grows like the cube of the number of features. Double the features and the
inverse costs roughly eight times as much; at thousands of features it becomes
ruinous, and at the scale of a neural network it is hopeless. Gradient descent
sidesteps the inverse entirely — each step is just a matrix-vector product — so it
keeps working when the closed form cannot. (And for almost every model after this
one — logistic regression, SVMs, the GPT — *no* closed form exists at all,
because their gradient-equals-zero equation cannot be solved by algebra. Gradient
descent is the only general way down.)

That contrast is the whole reason we fit linear regression with gradient descent
here on purpose: it is the gentlest place to watch the general engine work on
real data, with a closed-form answer standing by to confirm it landed in the
right spot. The engine you trust here is the one you carry forward.

## Dataset

The experiments use the **California housing** dataset: 20,640 rows, each a
California census block with 8 numeric features (median income, house age,
average rooms, and so on) and a target $y$ — the block's median house value. Load
it with `datasets.loaders.california_housing.load()`, which returns `X` of shape
`(20640, 8)` and `y` of shape `(20640,)`. The exercises work on the first 200
rows so everything runs instantly. It is real data with real, mismatched feature
scales — which is exactly why standardization earns its keep.

## Exercise

Implement six functions in `my_work/linear-regression/linear_regression.py`. Run
`grade start linear-regression` to get the scaffold, then fill in each function
(do not peek at the worked solution while you do). Keep every array **float64** —
the gradient check fails on float32 rounding noise. Do not import sklearn; it
lives only inside the grader's oracle.

- `gradient_descent(f, grad_f, x0, lr, steps)` — the **warm-up**, recalled cold
  from L1. Parameters: `f` is the scalar objective (the update itself does not
  use it); `grad_f` is its gradient function — hand it a point, it returns the
  gradient there, shape `(d,)`; `x0` is the starting parameter vector, shape
  `(d,)`; `lr` is the learning rate; `steps` is how many updates to take. Copy
  `x0` first (do not mutate the caller's array), then apply
  $x \leftarrow x - \mathtt{lr}\cdot \mathtt{grad\_f}(x)$ `steps` times. Returns
  the final parameter vector, shape `(d,)`. It is graded on its own and is *not*
  called by the functions below — re-authoring it keeps the GD engine fresh.

- `standardize(X)` — column-wise rescale to mean 0, std 1. Parameter: `X` is the
  feature matrix, shape `(n, d)`. Subtract the per-column mean (`axis=0`) and
  divide by the per-column std. Returns the standardized matrix, same shape
  `(n, d)`, float64.

- `predict(X, w)` — linear prediction. Parameters: `X` is the design matrix,
  shape `(n, d)` (it already carries the bias column if there is one); `w` is the
  weight vector, shape `(d,)`. Returns `X @ w`, the prediction per row, shape
  `(n,)`.

- `mse_loss(y, y_hat)` — mean squared error. Parameters: `y` is the true targets,
  shape `(n,)`; `y_hat` is the predictions, shape `(n,)`. Returns a single scalar
  float: the mean of the squared residuals.

- `mse_grad(X, y, w)` — the gradient of MSE w.r.t. the weights. Parameters: `X`
  is the design matrix `(n, d)`; `y` is the true targets `(n,)`; `w` is the
  weights `(d,)`. Returns $\frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y})$, shape
  `(d,)`, float64. Keep the `2/n` factor.

- `fit(X, y, lr, steps)` — fit the weights by gradient descent on MSE.
  Parameters: `X` is the design matrix `(n, d)`, **with its bias column already
  included by you** (`fit` does not add one); `y` is the targets `(n,)`; `lr` is
  the learning rate; `steps` is the number of GD steps. Start `w` at zeros, then
  loop `steps` times stepping against `mse_grad`. Inline the loop here — do not
  call the warm-up — so the module stays self-contained. Returns the fitted
  weight vector, shape `(d,)`.

Then grade it: the site's "Run Grader" button or `grade linear-regression`.

## How the grader checks you

The checks fall into two independent groups.

- The **warm-up** `gradient_descent` is graded on its own and never blocks the
  rest. It is run on the round bowl $f(x,y) = x^2 + y^2$ from the start point
  $(3, 3)$, 50 steps at `lr=0.1`; the result must land within $0.01$ of the
  origin (norm below $0.01$). A wrong warm-up does not hide working
  linear-regression functions.
- `standardize` is checked on 100 housing rows: the returned matrix keeps the
  input shape, and every column ends with mean $\approx 0$ and standard deviation
  $\approx 1$ (tolerances around $10^{-6}$). Taking the mean over the wrong axis
  (rows instead of columns) is the named cause in the failure message.
- `predict` is checked on a fixed tiny case — `X = [[1, 2], [3, 4]]`,
  `w = [1, 0.5]`, expected `[2, 5]` — to catch a transposed product (`w @ X`
  instead of `X @ w`).
- `mse_loss` is checked on the worked numeric example above (true `[1,2,3,4]`,
  predicted `[1,1,4,6]`, expected `1.5`) and must return a scalar.
- `mse_grad` is **gradient-checked**: on a fixed, well-conditioned random design,
  your analytic gradient is compared against a centered finite-difference
  estimate of `mse_loss` (the technique from L1). The relative error must be below
  $10^{-4}$, and the result must be shape `(d,)` and dtype float64. A missing
  `2/n` factor or a transposed $X^\top$ produces exactly the mismatch this catches.
- `fit` is checked against an **oracle** — `sklearn.LinearRegression()`, the
  textbook least-squares solver — on a fixed 200-row slice of California housing.
  Your `fit` runs 2000 GD steps at `lr=0.1` on the standardized, bias-augmented
  design; its predictions must agree with the oracle's to a tolerance of $0.05$.
  The check is on *predictions*, not the weight vector. Because the loss is convex
  (one minimum), enough GD steps at a stable learning rate must converge to the
  same least-squares solution the oracle computes in closed form — this is the
  "two solvers, same answer" claim, verified.

### The named trap: a float32 gradient

This grader hunts one specific mistake by name. If `mse_grad` casts its result to
**float32** (for example, wrapping the return in `np.float32(...)`), the gradient
is mathematically correct but loses precision: a float32 number carries only
about 7 significant digits, and at the finite-difference step $h = 10^{-5}$ that
rounding noise swamps the comparison and blows past the $10^{-4}$ relative-error
bar. The grader recognizes this failure and its message calls out **float64** as
the fix, so you change the dtype rather than hunting a phantom error in your
formula.

## The hint ladder

If the grader's message is not enough, open the hints. Each of the six functions
has three levels: a conceptual nudge (what the function is for and the one easy
mistake), then the formula with array shapes, then pseudocode. Try level 1 before
climbing — for `standardize` and `mse_grad` the level-1 nudge restates *why* the
scaling and the `2/n` factor matter, which is often all you need.

## Done when

`grade linear-regression` shows all-green: the warm-up reaches the origin on the
bowl; `standardize` returns mean-0/std-1 columns; `predict` returns `[2, 5]` on
the fixed case; `mse_loss` returns `1.5`; `mse_grad` matches the numerical
gradient to $10^{-4}$ in float64; and `fit` agrees with the sklearn oracle to
$0.05$ on the 200-row slice. Then run the experiments and record your predictions.
