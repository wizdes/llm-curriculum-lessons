# L0 — Hello, Grader

This lesson exists to prove the loop, not to teach hard math. By the end you will
have read a concept, rebuilt two small functions blind, been graded by a tool that
talks like a teacher, and run one experiment where you wrote your prediction down
*before* running it. Every later lesson — gradient descent through the GPT — is
this same loop with bigger ideas inside it.

## The loop you are about to run

1. **Read** this page. The concept section below explains `mse` and `mae`.
2. **Study the worked example** at `lessons/L00-hello-grader/worked/hello_grader.py`.
   It is annotated line-by-line with shape comments. Read it until it's obvious.
3. **Rebuild it blind** in `my_work/hello-grader/hello_grader.py`. Running
   `uv run grade start L00-hello-grader` copies a starter file there with signatures,
   docstrings, and TODOs — you fill in the bodies *without* looking back at the
   worked example. The blind rebuild is the whole point: the worked example
   teaches, the rebuild proves.
4. **Grade it** — from the site's "Run Grader" button, or `uv run grade L00-hello-grader`
   in the terminal. Both surfaces run the same grader.
5. **Stuck?** Use the hint ladder (below). Never paste a solution in; the hints
   escalate from nudge → formula → pseudocode, and you choose how far to climb.
6. **Experiment** — `experiments.md` has one toy experiment. You must type your
   prediction before the run step unlocks. That's deliberate: written
   predictions are what make the result stick.

## Lesson anatomy

Every lesson directory looks like this one:

- `lesson.md` — the concept, intuition-first.
- `worked/` — the annotated reference implementation. Read-only territory.
- `starter/` — the scaffold that `uv run grade start` copies into `my_work/<slug>/`.
- `grader/` — the tests that grade your `my_work/` copy. They grade mechanics
  (shapes, values, finiteness), run in seconds, and are deterministic.
- `hints.yaml` — the hint ladder, three levels per function.
- `experiments.md` — prediction-first experiments.

One rule is absolute (the **Iron Law**): `my_work/` is yours alone. The tooling
never writes code there; help arrives only through the hint ladder.

## Concept: two ways to measure "how wrong"

You have ground-truth values $y$ (the *correct* answers, the numbers you wish the
model had produced) and a model's predictions $\hat{y}$ (read "y-hat" — the hat is
the standard notation for "an estimate of"). Both are vectors of length $n$, where
$n$ is the number of samples (data points). The $i$-th entry $y_i$ is the true
value for sample $i$, and $\hat{y}_i$ is the model's guess for that same sample.
How wrong is the model overall? Two classic answers, each computed the same way:
turn the list of per-sample misses into one summary number.

**Mean squared error (MSE).** "Squared" because each miss is multiplied by
itself, "mean" because we average:

1. **Per-sample miss.** For each sample $i$, take the miss $y_i - \hat{y}_i$ (how
   far the prediction is from the truth; it can be negative).
2. **Punish it.** Square the miss: $(y_i - \hat{y}_i)^2$. Squaring throws away the
   sign (the result is always $\ge 0$) and inflates large misses.
3. **Average.** Add up all $n$ squared misses and divide by $n$:

$$
\mathrm{MSE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

(Read aloud: "one over n, times the sum from $i=1$ to $n$ of $y_i$ minus
$\hat{y}_i$, all squared." The $\sum$ symbol means "add these up over every
sample.")

**Mean absolute error (MAE).** Same three steps; only the "punish" step changes:

1. **Per-sample miss.** Same $y_i - \hat{y}_i$ as before.
2. **Punish it.** Take the absolute value $\lvert y_i - \hat{y}_i \rvert$ — the
   miss's size with its sign dropped (so $-3$ and $+3$ both count as $3$).
3. **Average.** Sum the $n$ absolute misses and divide by $n$:

$$
\mathrm{MAE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
$$

Both reduce a vector of misses to a single non-negative scalar; zero means
perfect. The difference is *how they weight big misses*. Squaring punishes
outliers quadratically: a miss of 10 contributes 100 to MSE but only 10 to MAE.
So MSE is outlier-sensitive and MAE is outlier-robust — which is exactly what
the experiment below will make you feel rather than memorize.

In NumPy each is one line, because arithmetic on arrays is elementwise:

- `y - y_hat` has shape `(n,)` — the vector of misses.
- Squaring or `np.abs` keeps shape `(n,)`.
- `.mean()` collapses `(n,)` to a scalar `()`.

That shape story — `(n,) → (n,) → ()` — is the kind of thing the grader checks
first, before it ever compares values. Get the shape right and the value bugs
become easy to see.

## Exercise

Implement in `my_work/hello-grader/hello_grader.py` (run `grade start
hello-grader` to copy the scaffold there):

- `mse(y, y_hat)` — mean squared error. Parameters: `y` is the array of true
  answers and `y_hat` the array of model predictions, both 1-D arrays (or
  array-likes) of the same length `n`. Returns a single NumPy scalar float
  (shape `()`): the average of the squared misses.
- `mae(y, y_hat)` — mean absolute error. Same parameters and same return
  contract; only the per-sample punishment differs (absolute value, not square).

Then grade it from both surfaces: the site's Run Grader button and
`uv run grade L00-hello-grader` in the terminal. Both run the same grader.

## How the grader checks you

You will meet this grader once per lesson, so it is worth knowing how it talks.
The graders in this curriculum never fail with a bare `AssertionError` (Python's
default "this check did not hold" exception, which tells you a line failed but
nothing about *why*). Every failure message instead names the variable, states
what it found versus what it expected, and points at the concept to revisit.

Concretely, here is what each function is checked against. The grader runs your
`mse` and `mae` on one fixed pair of arrays with a known answer:
`y = [1, 2, 3, 4]` and `y_hat = [1, 3, 2, 2]`, so the misses are `0, -1, 1, 2`.
For each function it runs three checks, in this dependency order:

1. **Shape** — does the function return a single scalar (shape `()`), not a
   vector? This runs first because a wrong shape makes the value comparison
   meaningless.
2. **Value** — for `mse`, does it equal `1.5` (the squared misses `0, 1, 1, 4`
   average to `1.5`)? For `mae`, does it equal `1.0` (the absolute misses
   `0, 1, 1, 2` average to `1.0`)? The value check is compared with a small
   numerical tolerance, so tiny floating-point differences do not fail you.
3. **Finiteness** — is the result an ordinary number, not `NaN` ("not a
   number", the result NumPy returns from things like dividing by zero or
   averaging an empty array) or infinity?

The shape check **gates** the value and finiteness checks for the same function:
if the shape is wrong, those two are reported as *blocked* (not run yet) rather
than failed, so you fix one thing at a time. `mse` and `mae` are graded
independently — a broken `mse` never hides a working `mae`.

For example, if your `mse` returns the *vector* of squared errors instead of
their mean, you'll see something like:

> your mse returned shape (4,), expected a scalar float () — did you forget
> `.mean()`?

Fix the shape, re-run, and the value check that was blocked behind it lights up.
That is the grader's whole job: be the tightest possible feedback loop, telling
you *where* you are wrong and *what idea* fixes it — never the solution itself.

## The hint ladder

If the grader's message isn't enough, open the hints section on this page.
Each function has three collapsed levels:

1. a conceptual nudge,
2. the formula with shape annotations,
3. pseudocode.

Level 2 only unlocks after you've opened level 1, and reveals are recorded —
not as punishment, but so your future self can see which ideas needed help.
Try the level-1 nudge before climbing.

## Done when

The full loop works end-to-end: you read this page, rebuilt `mse` and `mae`
blind, both grading surfaces show all-green, the progress page updated, and the
experiment in `experiments.md` is recorded with your written prediction.
