# L10 — Training Dynamics

In L9 you trained an MLP — a multi-layer perceptron, the small stacked-layers
network from the last lesson — on MNIST with plain SGD and a learning rate you
tuned by hand. "SGD" is **stochastic gradient descent**: gradient descent where
each step uses one small batch of examples instead of the whole dataset (more on
"stochastic" in a moment). It worked. But plain SGD on a real network is fragile
in three separate ways, and each has nothing to do with whether your gradients
are correct:

- **Bad starting weights.** If the very first weights are the wrong size, the
  numbers flowing through the layers shrink to nothing or blow up to infinity
  before a single gradient even arrives.
- **One flat learning rate.** The **learning rate** is the step size — how far
  you move at each update. A single fixed step size crawls when the loss surface
  is long and narrow.
- **Memorizing instead of learning.** With nothing holding it back, a network can
  fit the exact training examples (including their noise) instead of the pattern
  underneath, and then fail on data it has not seen.

This lesson is the toolbox that fixes all three: a smarter way to pick the
starting weights (Xavier init), smarter optimizers that adapt the step
(momentum, Adam), and a regularizer that fights memorizing (L2). These are the
dials *around* gradient descent that decide whether it converges fast, slow, or
not at all. Nothing here changes the chain rule or how a single gradient is
computed — the gradients stay exactly what L8's backprop produced.

A quick vocabulary refresher you will need throughout, all from prior lessons:

- **Gradient** — the vector of slopes of the loss with respect to the parameters;
  it points in the direction the loss increases fastest, so we step *against* it.
  Written `grads` in code below.
- **Parameters** (`params`) — the numbers the network learns: the weights (and
  biases). These are what every update below changes.
- **Loss** — one number measuring how wrong the model is right now; training
  drives it down.
- **Epoch / batch** — one *epoch* is one full pass over the training set; a
  *batch* (or *mini-batch*) is the small chunk of examples used for a single
  update. Many batches make up one epoch.

A note on "stochastic," because it trips everyone: in stochastic gradient
descent, "stochastic" means each step looks at *one mini-batch* (a random small
sample), not the whole dataset — that is what makes it cheap and fast. It does
**not** mean "a small step." The size of the step is set by the learning rate,
which is a completely separate knob.

## Momentum: remembering where you were going

Plain SGD steps in the direction of the current gradient and forgets everything
that came before it. Picture a long, narrow valley — steep side walls, a gently
sloping floor. The gradient mostly points *across* the valley toward whichever
wall is closer, only weakly *along* the floor toward the bottom. So plain SGD
zig-zags wall to wall, barely advancing along the floor where you actually want
to go.

**Momentum** fixes this by keeping a running **velocity** — an *exponential
average of past gradients* (a weighted average where recent gradients count most
and older ones fade). Instead of stepping along the raw current gradient, you
step along this accumulated velocity. The procedure for one momentum step:

1. Update the velocity: `v = momentum * velocity + grads`. The `momentum` number
   (a fraction, typically 0.9) says how much of the old velocity to keep; the new
   gradient is added on top.
2. Step the parameters along the velocity: `params = params - lr * v`, where `lr`
   is the learning rate (step size).
3. Return both the new parameters *and* the new velocity `v`, so the next step
   can keep accumulating.

Why this helps: directions the gradient agrees on step after step (along the
valley floor) **accumulate** in `v` and the step grows; directions that flip sign
step to step (across the walls) **cancel out** and the step there stays small.
The result is a heavier ball that rolls through small bumps and along the floor
instead of rattling between the walls. With `momentum = 0.9`, each step keeps 90%
of the prior velocity, so a consistent direction can build up to roughly 10x the
single-gradient step before it saturates.

## Adam: a per-parameter learning rate

Momentum gives every parameter the same learning rate. **Adam** ("adaptive moment
estimation") goes further: it gives *each parameter its own* effective step size,
scaled down for parameters whose gradients have been large and bouncy, scaled up
for ones whose gradients have been small and steady. It does this by tracking two
running averages instead of one:

- `m`, the **first moment** — a running average of the gradient itself (this is
  essentially momentum).
- `v`, the **second moment** — a running average of the gradient *squared*. A
  large `v` means that parameter's gradient has been big in magnitude.

Then it divides the step by `sqrt(v)` (the square root of the second moment), so
a parameter with a historically large gradient takes a proportionally smaller
step. Reading the symbols: `sqrt(...)` is the square root; `beta1` and `beta2`
("beta-one," "beta-two") are the decay rates for the two averages — how much old
value each keeps, typically 0.9 and 0.999; `eps` ("epsilon") is a tiny number
like `1e-8` added to the denominator only to avoid dividing by zero; `t` is the
step counter, starting at 1.

The full Adam step, as numbered operations:

1. Advance the step counter: `t = state["t"] + 1`.
2. Update the first moment: `m = beta1 * m + (1 - beta1) * grads`.
3. Update the second moment: `v = beta2 * v + (1 - beta2) * grads**2` (here
   `grads**2` is the elementwise square of the gradient).
4. **Bias-correct** the first moment: `m_hat = m / (1 - beta1**t)`.
5. **Bias-correct** the second moment: `v_hat = v / (1 - beta2**t)`.
6. Step: `params = params - lr * m_hat / (sqrt(v_hat) + eps)`.

The subtle, load-bearing piece is step 4 and step 5, the **bias correction**
(`m_hat` reads "m-hat," the corrected version of `m`). Here is the problem it
solves. Both `m` and `v` start at zero. On step 1, `m = 0.9 * 0 + 0.1 * grads`,
so `m` is only `(1 - beta1) = 0.1` — just 10% — of the true gradient; likewise
`v` is only `(1 - beta2) = 0.001` of the true squared gradient. The running
averages have not had time to fill up; both are biased toward their zero starting
point.

Bias correction undoes each bias exactly. At `t = 1` the first divisor is
`(1 - beta1**1) = (1 - 0.9) = 0.1`, so `m_hat = m / 0.1` scales `m` back up by 10x
— recovering the full-size gradient — and the second divisor
`(1 - beta2**1) = 0.001` rescales `v` the same way. As `t` grows, `beta1**t` and
`beta2**t` shrink toward zero (`0.9**10 ≈ 0.35`, `0.9**50 ≈ 0.005`), both divisors
approach 1, and the correction fades away once the averages have warmed up.

Both moments are biased, and that is exactly why you cannot eyeball the effect on
the step. The Adam step divides `m_hat` by `sqrt(v_hat)`, and `v` is corrected by
a *much* larger factor (`1/0.001`) than `m` (`1/0.1`). If you skip correction, the
under-scaled `v` in the denominator wins and the early step comes out the **wrong
size** — in the constant-gradient experiment below it is actually about 3x too
*large* on step 1, not too small. The direction of the error depends on the
gradient history; the point is that an uncorrected step 1 simply does not match
the correct trajectory — which is exactly the named trap this lesson's grader
hunts for. (You will *measure* this in Experiment 3.)

## Xavier initialization: starting at a sane scale

Before any gradient flows at all, the *initial* random weights already decide
whether the network can learn. Recall that each layer multiplies its input by a
weight matrix. If those weights are too small, the signal shrinks a little at each
layer and after several layers it has **vanished** to near-zero — there is
nothing left to learn from. If they are too large, the signal grows at each layer
and **explodes** to huge numbers. Either way training stalls before it starts.

The right scale depends on the matrix's shape. We name two sizes:

- **`fan_in`** — the number of inputs feeding into a layer (the matrix's first
  dimension): how many numbers get summed to produce one output.
- **`fan_out`** — the number of outputs the layer produces (the second
  dimension).

**Xavier initialization** (also called Glorot initialization, after Xavier
Glorot) picks the weight scale that keeps the variance — the spread, how far
numbers stray from their average — roughly constant from layer to layer. The
procedure:

1. Compute the standard deviation: `std = sqrt(2 / (fan_in + fan_out))`. ("Std,"
   the standard deviation, is the typical size of a value; it is the square root
   of the variance. Reading the formula aloud: the std is the square root of two,
   divided by the sum of the fan-in and the fan-out.)
2. Draw a `(fan_in, fan_out)` matrix of standard-normal values — random numbers
   from a bell curve with mean 0 and variance 1 — and multiply by `std`:
   `W = standard_normal((fan_in, fan_out)) * std`.

Why this exact scale? One output is a sum over `fan_in` inputs, and summing
`fan_in` random terms multiplies the variance by `fan_in`. To cancel that, you
want each weight's variance to shrink like `1 / fan`. The `+ fan_out` in the
denominator balances two things at once: the forward signal (which depends on
`fan_in`) and the backward gradient (which depends on `fan_out`). Averaging the
two fans gets both roughly right. A check you can run: the empirical variance of
a large Xavier matrix should land at `2 / (fan_in + fan_out)`.

A common wrong rule is `1 / sqrt(fan_in)`, which ignores `fan_out` entirely and
gives a different variance — the grader's variance check is calibrated to catch
exactly that substitution.

### Overfitting diagnosis

A network can be correctly built, well-initialized, and well-optimized and *still*
fail — by learning the training data too well. This is **overfitting**: the model
memorizes the specific training examples, including their random noise, instead of
the general pattern, so it scores well on data it has seen and badly on data it
has not.

To diagnose it you need two losses, not one:

- **Training loss** — the loss measured on the data the model is learning from.
- **Validation loss** — the loss measured on a *held-out* set the model never
  trains on. This is your stand-in for "data it has never seen."

The signal of overfitting is not either loss by itself — it is the **gap** between
them, the **train–val gap** (validation loss minus training loss). A concrete
worked curve makes this readable. Imagine watching both losses over 100 epochs:

| Epoch | Training loss | Validation loss | Gap (val − train) | What it means |
|------:|--------------:|----------------:|------------------:|---------------|
|     5 |          0.80 |            0.83 |              0.03 | both falling — learning real structure |
|    20 |          0.40 |            0.45 |              0.05 | still healthy, gap small |
|    40 |          0.18 |            0.30 |              0.12 | val flattening, gap opening |
|    60 |          0.08 |            0.34 |              0.26 | **val now rising** — overfitting |
|   100 |          0.02 |            0.50 |              0.48 | train near-zero, val worse — memorizing noise |

Read it left to right: early on, both losses fall together — good, the model is
learning the pattern. Around epoch 40 the validation loss bottoms out; that
turning point is the **inflection point**, the moment generalization stops
improving. Past it the training loss keeps dropping toward zero while the
validation loss *rises* — the widening gap is the unmistakable signature of
overfitting. Note what this rules out: the training loss at epoch 100 (0.02) is
the *lowest* of the whole run, yet that model is the *worst* one to ship. A lower
training loss is not always better; only the validation loss tells you whether the
model generalizes.

**How to read the two curves to decide what is wrong** — the diagnosis procedure:

1. Plot training loss and validation loss on the same axes, epoch by epoch.
2. If **both are still high and falling**, you are underfitting or under-trained —
   keep training (or grow the model / raise the learning rate).
3. If **both are low and close together** (small gap), the model is healthy —
   this is the goal.
4. If **training loss is low but validation loss is high and rising** (a wide,
   growing gap), you are overfitting — apply the fix below, get more data, or stop
   training at the inflection point.
5. Pick the model from the **epoch where validation loss was lowest**, not the
   last epoch — that is the best-generalizing checkpoint.

The standard fix for overfitting is **L2 regularization**, which adds a penalty
for large weights to the loss:

$$\text{total} = \text{loss} + \lambda \sum_w w^2$$

Reading it aloud: the total objective is the ordinary loss plus lambda times the
sum of every weight squared. Here `lam` (the Greek letter $\lambda$, "lambda") is
the **regularization strength** — a small number you choose that sets how hard the
penalty pushes; larger `lam` means a stronger pull. Because the penalty grows with
the square of each weight, it pulls weights toward zero unless the data genuinely
needs them large. The result is a smoother model that generalizes better. The
gradient of the penalty term `lam * w**2` with respect to `w` is `2 * lam * w`
(the derivative of `w**2` is `2w`, hence the factor of 2), so regularization just
adds `2 * lam * w` to each weight's existing gradient.

Finding the inflection point and watching `lam` shrink the train–val gap is an
*experiment*, not a grader assertion — the grader checks that
`l2_regularized_loss` adds the penalty correctly (the analytic `2 * lam * w`
gradient), and the experiments are where you *see* the gap open and close.

## Worked example

`worked/training_dynamics.py` is the full reference for all five functions: the
centered-difference `numerical_gradient` warm-up, `init_xavier`,
`l2_regularized_loss`, `sgd_momentum`, and `adam`. Its `__main__` block minimizes
a convex (bowl-shaped, single-minimum) quadratic with Adam as a self-check and
prints the loss falling — fully self-contained numpy, no dataset.

## Exercise

Implement in `my_work/training-dynamics/training_dynamics.py` (run
`uv run grade start L11-training-dynamics` to get the scaffold). Do the warm-up
`numerical_gradient` first — every gradient check below leans on it. All five
functions are plain numpy; no dataset, no imports beyond numpy.

- `numerical_gradient(f, x, eps=1e-6)` — the **warm-up**, recalled cold from the
  gradient-descent lesson. Parameters: `f` is a function of *no* arguments that
  reads the *current* value of `x` (you mutate `x` in place, call `f()`, then
  restore it); `x` is the point — an array of parameters — where you want the
  slope; `eps` is the tiny perturbation step ("eps" is epsilon). For each entry of
  `x`, bump it `+eps` and `-eps`, and form the **centered** difference
  `(f(x+eps) - f(x-eps)) / (2*eps)`. Returns an array shaped like `x` — one slope
  per entry.
- `init_xavier(fan_in, fan_out, rng)` — Xavier/Glorot weight init. Parameters:
  `fan_in` is the number of inputs (an int), `fan_out` the number of outputs (an
  int), and `rng` is an explicit numpy random Generator (used for
  reproducibility — call `rng.standard_normal(...)`). Returns a `(fan_in,
  fan_out)` array drawn from a zero-mean Gaussian with
  `std = sqrt(2 / (fan_in + fan_out))`.
- `l2_regularized_loss(loss, weights, lam)` — add the L2 penalty. Parameters:
  `loss` is the base scalar loss (a float); `weights` is an iterable of weight
  arrays; `lam` is the regularization strength (a float). Returns the float
  `loss + lam * sum(sum(w**2) for w in weights)` — the original loss plus lambda
  times the total of every weight squared.
- `sgd_momentum(params, grads, velocity, lr, momentum)` — one momentum step.
  Parameters: `params`, `grads`, and `velocity` are arrays of the same shape (the
  current parameters, their gradient, and the accumulated velocity); `lr` is the
  learning rate (step size); `momentum` is the fraction of old velocity to keep
  (e.g. 0.9). Compute `v = momentum * velocity + grads`, then `params - lr * v`.
  Returns the tuple `(new_params, new_velocity)` — return both so the next step
  can keep accumulating.
- `adam(params, grads, state, lr, beta1, beta2, eps)` — one Adam step. Parameters:
  `params` and `grads` are same-shape arrays; `state` is a dict
  `{"m": array, "v": array, "t": int}` holding the two running averages and the
  step count (initialized to zeros and `t = 0`); `lr` is the learning rate;
  `beta1`, `beta2` are the decay rates (0.9, 0.999); `eps` is the tiny
  divide-by-zero guard (1e-8). Apply the six numbered steps above — **including
  the bias correction**. Returns the tuple `(new_params, new_state)` where
  `new_state` is the updated `{"m", "v", "t"}` dict.

Then grade it: the site's Run Grader button or `uv run grade L11-training-dynamics`.

## How the grader checks you

- The **warm-up** `numerical_gradient` is checked first and on its own: it
  computes the gradient of `f() = sum(x**2)` at `x = [1, -2, 3]` and compares it
  against the exact analytic answer `2*x`, requiring a relative match of `1e-5`.
  Several other checks depend on this one passing, so a broken warm-up is reported
  up front rather than hidden behind later failures.
- `sgd_momentum` is run for a fixed 3-step trace against an independently written
  reference (`v = momentum*velocity + grads`, then `params -= lr*v`). Your
  parameters *and* velocity must match the reference to `1e-10` at every step — so
  forgetting to return the updated velocity, or using the old `velocity` in the
  step instead of the freshly computed `v`, is caught.
- `adam` is run the same way — a fixed 3-step trace against the canonical
  bias-corrected update, matching params, `m`, `v`, and `t` to `1e-10`. This is
  the check that carries the named trap below.
- `l2_regularized_loss` is verified by taking the *numerical* gradient of its
  penalty term and confirming it equals the analytic `2 * lam * w` — this is why
  the factor of 2 must be exactly right.
- `init_xavier` is checked statistically: it draws a large `256 × 256` matrix and
  confirms the empirical mean is ~0 (`|mean| < 0.02`) and the empirical variance
  is within 10% of `2 / (fan_in + fan_out)`. Using a different scale rule (like
  `1/sqrt(fan_in)`) lands outside that band and fails.
- An **integration check** then minimizes a self-contained convex quadratic
  `f(p) = sum((p - target)**2)` with your `adam` for 300 steps and confirms both
  the loss and the distance to the target *decrease* — proof the optimizer
  actually descends.

### The named trap: Adam without bias correction

This grader is built to catch one specific mistake by name. If you write the Adam
step using the raw moments `m` and `v` directly in the update —
`params - lr * m / (sqrt(v) + eps)` — and **skip** the bias-correction divides
(`m_hat = m / (1 - beta1**t)`, `v_hat = v / (1 - beta2**t)`), the formula *looks*
right but the early steps are badly wrong: both `m` and `v` are biased toward zero,
and because `v` sits under a square root in the denominator, the net step comes out
the wrong size on the first few steps (Experiment 3 shows it about 3x too large for
a constant gradient). The Adam trace check sees step 1 diverge from the reference
and its error message tells you to apply the **bias correction** — so you fix the
missing `m_hat`/`v_hat` rescale rather than hunting a phantom bug elsewhere.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge (the symptom and the idea behind the fix), then the
formula with shapes, then pseudocode. Try level 1 before climbing — especially for
the Adam bias-correction trap, where the level-1 nudge points you straight at the
zero-start bias in `m` and `v` that the correction undoes.

## Done when

`uv run grade L11-training-dynamics` shows all-green: the warm-up `numerical_gradient`
matches the analytic gradient to `1e-5`; `sgd_momentum` and `adam` match their
3-step reference traces to `1e-10` (with `adam` applying bias correction);
`l2_regularized_loss` has the correct `2 * lam * w` penalty gradient; `init_xavier`
hits mean ~0 and variance ~`2/(fan_in+fan_out)`; and the integration check shows
Adam driving the convex loss down. Then run the experiments and record your
predictions.
