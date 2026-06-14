# L1 — Gradient Descent

In L0 you measured how wrong a model is. This lesson is the other half: how to
make it *less* wrong. Gradient descent is the single optimization engine behind
nearly every model in this course — linear regression, logistic regression,
SVMs, and eventually the GPT. Build it once here and you reuse it everywhere; the
only thing that changes lesson to lesson is the loss whose gradient you feed it.

A few words you will see throughout, defined once so nothing below is a mystery:

- A **model** is a formula that turns an input into a prediction.
- A **parameter** is an adjustable number inside that formula — a knob you can
  turn. Training is the act of turning the knobs until the predictions are good.
- The **loss** is the single number from L0 that scores how wrong the model's
  predictions are right now. Smaller is better; zero is perfect.
- To **fit** a model is to turn its knobs until the loss is as small as you can
  get it. "Optimization" and "training" mean the same thing.

This lesson is the procedure for turning the knobs.

## Intuition: a ball rolling downhill

Picture the loss as a landscape. Every possible setting of the parameters is a
point on the ground, and the height at that point is the loss there — how wrong
the model is with those knob settings. A bad setting is a hilltop; the best
setting is the bottom of a valley. Training is just walking to the lowest point.

You cannot see the whole landscape from above — it has one dimension for every
parameter, far too many to picture. But you do not need to. Standing at your
current point, you can feel the slope under your feet: which way is downhill.

The **gradient** is the mathematical name for that slope. It points in the
direction the loss *increases* fastest. That is the opposite of what you want, so
you take a small step in the *opposite* direction — straight downhill. Then you
feel the slope at the new spot and step again. Repeat until the ground is flat
(the slope is zero) and you are at the bottom. That loop is the entire algorithm.

One thing to pin down before the formula, because it trips up almost everyone:
**we never take the derivative of the model.** The model (here it will be
$\hat{y} = w x$) just makes predictions; asking for "the slope of the model" is
not what training does. What we take the slope of is the **loss** — the single
number scoring how wrong the predictions are — and we take it **with respect to
the parameters**. So the gradient answers exactly one question: *if I nudge each
parameter a little, how does the loss change?* That is the only derivative in
play, here and in every lesson after.

In one dimension — a single parameter $w$ — "the slope of the loss" is exactly
its derivative from calculus, written $L'(w)$. (The derivative of a function is
its slope: how fast the output changes as you nudge the input.) One step of
gradient descent is:

$$
w \leftarrow w - \eta \, L'(w)
$$

Read aloud: *the new $w$ is the old $w$, minus a small fraction $\eta$ of the
slope.* The arrow $\leftarrow$ means "becomes" — overwrite $w$ with the
right-hand side. We subtract because the slope points uphill and we want to go
down. The number $\eta$ (the Greek letter eta, pronounced "ay-ta") is the
**learning rate**: how big a step to take. Too big and you leap clear over the
valley and bounce out; too small and you crawl for ages. Choosing it well is most
of the practical art, which the experiments make you feel directly.

With more than one parameter nothing really changes — the slope just becomes a
list of slopes, one per parameter. That list is the gradient, written
$\nabla L$ (the symbol $\nabla$ is called "nabla" or "del"; read $\nabla L$ as
"the gradient of $L$"). Each entry is a **partial derivative**: the slope of the
loss in one parameter's direction while the others are held still. We collect the
parameters into a vector $\mathbf{w}$ (bold means "a list of numbers, not just
one"), and the update becomes:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla L(\mathbf{w})
$$

Read aloud: *the new parameter vector is the old one, minus $\eta$ times the
gradient.* Each entry of $\nabla L$ says how the loss responds to nudging that
one parameter; stepping against the whole vector at once improves every parameter
together.

**Why step over and over, instead of jumping straight to the bottom?** This is
the natural objection — if you have a formula for the slope, why not just go to
the minimum? Because the gradient gives you the slope *wherever you happen to be
standing*; it does not tell you the *location* of the lowest point. There is one
shortcut that sometimes works: set the gradient to zero, $\nabla L = 0$, and try
to *solve that equation algebraically* for the parameters — landing on the bottom
in a single calculation. That is called a **closed-form solution**, and a few
problems have one (plain linear regression is the famous case; the next lesson
mentions it). But for almost everything else here — logistic regression, SVMs,
the GPT — the equation is far too tangled to isolate the parameters, so no such
formula exists. Gradient descent is how you reach the bottom anyway: read the
slope, take a step downhill, read it again, repeat. The iteration is not a crude
fallback; for most models it is the *only* general way down.

## By-hand derivation #1: MSE on three points

The fastest way to stop a formula from being a black box is to run it once by
hand, with real numbers, start to finish. We will compute one gradient and take
two steps with nothing but arithmetic.

**The setup — what data we have.** A dataset is a collection of examples, each a
pair $(x, y)$: an **input** $x$ and the correct **answer** $y$ we want the model
to produce for it. Ours is three pairs:

$$
\{(1, 2),\; (2, 4),\; (3, 6)\}
$$

So when the input is $1$ the right answer is $2$, when it is $2$ the answer is
$4$, and when it is $3$ the answer is $6$. Two deliberate choices here. We use
exactly **three** points because that is the fewest that still shows a clear
pattern (two points always lie on *some* line; three confirm it) and still few
enough to add up by hand. And these particular pairs sit perfectly on the line
$y = 2x$ — double the input gives the answer — so we already *know* the right
model is "multiply by 2." Knowing the answer in advance lets us check at the end
that gradient descent is heading toward it.

**The model — what we predict and the knob we turn.** Our model is:

$$
\hat{y} = w x
$$

Read aloud: *the prediction equals the knob times the input.* The hat on
$\hat{y}$ (say "y-hat") marks it as a *guess* — the number the model outputs — as
opposed to the plain $y$, the true answer. There is exactly one knob, the weight
$w$, which is why we call this a **one-parameter model**: one knob, nothing else.
We deliberately leave out a **bias** — an extra "+ b" that would let the line
start somewhere other than zero (a vertical offset). We do not need one here
because $y = 2x$ already passes through the origin, and adding a second knob would
clutter a derivation meant to isolate one moving part. Bias arrives in the next
lesson. **Fitting** this model means turning the single knob $w$ until the guesses
$\hat{y}$ line up with the true answers $y$.

**The loss — how we score the guesses.** We score with **MSE**, mean squared
error (the loss from L0):

$$
L(w) = \frac{1}{n} \sum_{i=1}^{n} (w x_i - y_i)^2
$$

Read aloud: *for each example, take the prediction minus the true answer (the
"miss"), square it, and average those squared misses over all $n$ examples.* The
$\sum$ symbol (Greek capital sigma) means "add up," and the subscript $i$ indexes
the examples: $x_i$ and $y_i$ are the $i$-th input and answer; $n$ is how many
there are ($n = 3$ for us). We **square** each miss for two reasons: it makes
every miss positive (so overshooting by 2 and undershooting by 2 both count as
error, not cancel out), and it punishes big misses far more than small ones. $L$
is always $\ge 0$, and $L = 0$ means every guess is exactly right — a perfect
fit.

**Where we start.** We have to begin somewhere, so we set the knob to
$w = 0$ — the most neutral, assumption-free starting point. At $w = 0$ the model
predicts $\hat{y} = 0 \cdot x = 0$ for every input: it guesses zero no matter
what. That is obviously wrong, which is the point — we want to watch gradient
descent fix a knob that starts useless.

**The gradient — the slope of the loss in $w$.** To know which way to turn the
knob, we need the slope of $L$ as $w$ changes, $\tfrac{dL}{dw}$. We get it by
differentiating the loss with the chain rule from calculus. Each term is
$(w x_i - y_i)^2$; its derivative with respect to $w$ is $2(w x_i - y_i)$ times
the derivative of the inside, $(w x_i - y_i)$, which is just $x_i$. Averaging over
the $n$ terms:

$$
\frac{dL}{dw} = \frac{2}{n} \sum_{i=1}^{n} (w x_i - y_i)\, x_i
$$

Read aloud: *the slope is twice the average of (miss times input) across the
examples.* This single number is "one full gradient" — one slope, computed over
**all** the data at once (not one point at a time). Computing it once is exactly
what one step of gradient descent needs.

**Plug in the numbers at the start, $w = 0$.** Every prediction is $0$, so each
miss $\hat{y}_i - y_i$ is $0 - y_i$. The three misses are
$(0 - 2,\; 0 - 4,\; 0 - 6) = (-2, -4, -6)$:

- Loss: $\frac{1}{3}\big((-2)^2 + (-4)^2 + (-6)^2\big) = \frac{1}{3}(4 + 16 + 36) = 18.6\overline{6}$
- Gradient: $\frac{2}{3}\big((-2)(1) + (-4)(2) + (-6)(3)\big) = \frac{2}{3}(-2 - 8 - 18) = \frac{2}{3}(-28) = -18.6\overline{6}$

(The bar in $18.6\overline{6}$ means the $6$ repeats forever: $18.6666\ldots$,
i.e. $\tfrac{56}{3}$.)

The gradient is **negative**. A negative slope means the loss goes *down* as $w$
goes *up* — so we should increase $w$. That is exactly right: the true $w$ is
$2$ and we started at $0$, so "increase $w$" is the correct direction.

**Take two steps** with learning rate $\eta = 0.01$. Each step is the same short
procedure, repeated:

1. **Compute the slope at the current $w$:** evaluate
   $\tfrac{dL}{dw} = \tfrac{2}{n}\sum (w x_i - y_i)\, x_i$ with the $w$ you have now.
2. **Update the knob:** $w \leftarrow w - \eta \cdot \tfrac{dL}{dw}$ — subtract a
   fraction $\eta$ of that slope from $w$.
3. **Repeat** from step 1 with the new $w$.

Running that loop twice from $w = 0$:

| step | $w$ | $L(w)$ | $dL/dw$ |
|------|-----------|------------|-------------|
| start | 0.000000 | 18.666667 | −18.666667 |
| 1 | 0.186667 | 15.344830 | −16.924444 |
| 2 | 0.355911 | 12.614132 | −15.344830 |

Step 1 in full: $w = 0 - 0.01 \times (-18.6\overline{6}) = 0 + 0.186667 = 0.186667$.
Subtracting a negative is adding, so $w$ moved up toward $2$ — the
right way. After the step the loss dropped (18.67 → 15.34) and the slope shrank in
size (−18.67 → −16.92): we are walking toward $w = 2$ and slowing as the valley
floor flattens, just as the ball-downhill picture promised. (Each row's $dL/dw$ is
the slope at *that* row's $w$, which is why the value at $w = 0.355911$ in row 2,
−15.34, is the slope that drives the next step — multiplied by $\eta = 0.01$ it
moves $w$ by $0.01 \times 15.34 \approx 0.15$.)

`worked/by_hand_check.py` runs this exact derivation in code and prints this exact
table. If your own implementation ever disagrees with these numbers, that script
is where you find the first row that diverges.

## Gradient checking: how to trust a gradient

You just derived a gradient by hand. How do you know you did not make a calculus
slip? **Gradient checking** is the answer: compute the gradient a *second*,
completely independent way, and if the two agree you can trust both. A
**gradient checker** is simply the piece of code that runs that comparison — and
every grader in this milestone is one.

The two ways are:

- The **analytic gradient**: the exact formula you get from calculus — for us,
  $\tfrac{dL}{dw} = \tfrac{2}{n}\sum (w x_i - y_i) x_i$. "Analytic" means "derived
  by symbol-pushing on paper." It is exact but easy to get subtly wrong.
- The **numerical gradient**: an approximate slope built only from *values of the
  function*, no calculus at all. Slope is "rise over run," so you nudge the input
  by a tiny amount $h$, see how much the output moved, and divide. It is slow and
  approximate but almost impossible to derive wrong — which is exactly why it
  makes a good independent check on the analytic one.

**Two gradients, two very different jobs — keep them straight, because this is
the part people get backwards.** In real training you use the **analytic**
gradient on *every single step*: it is exact, and one formula hands you the slope
for *all* the parameters at once. You almost never *train* with the numerical
gradient, because it is both approximate and slow. How slow: the centered
difference costs two full evaluations of the loss (one above, one below) **per
parameter** — to get the slopes for $1{,}000$ parameters you nudge each of the
$1{,}000$ in turn and re-measure the loss on both sides, so a million-parameter
model would need two million loss evaluations for a single step. (Note that count is per *parameter*, not per *data point*: the data
stays fixed while you wiggle one weight at a time.) So the numerical gradient
earns its keep only as a **check**: when you derive an analytic gradient by hand —
or write a brand-new loss — you compare it against the numerical estimate at a few
points; if they agree, your calculus is right, and you switch the numerical one
off. In this lesson you build the numerical estimator because it is the universal
checker and it cements what a slope *is*; from the next lesson on, each model's
*analytic* gradient is what actually drives the training.

The good way to estimate that numerical slope is the **centered difference**:
measure the function a step $h$ *above* the point and a step $h$ *below* it, and
divide the change by the total distance $2h$:

$$
\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}
$$

Read aloud: *the slope is roughly (value a bit to the right minus value a bit to
the left) divided by the distance between those two probes.* It is "centered"
because the point $x$ sits exactly in the middle of the two probes, which makes
the estimate symmetric and cancels out the leading error.

Contrast the lazy alternative, the **one-sided difference**, which only looks in
*one* direction — from the point itself to a step $h$ away:

$$
\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x)}{h}
$$

This is the classic beginner mistake, because the centered version is much more
accurate for the same step $h$. Here is the concrete difference. The step $h$ is a
small number — say $0.001$. Watch what happens to the estimate's error as you
shrink $h$ further:

- with the **one-sided** difference, halving $h$ roughly **halves** the error;
- with the **centered** difference, halving $h$ cuts the error to about a
  **quarter** — because its error shrinks with $h^2$, and $(\tfrac{1}{2})^2 = \tfrac14$.

So for the same tiny $h$, the centered estimate lands far closer to the true
slope. **Always use the centered version.** (Writing the one-sided form is exactly
the bug this lesson's grader is built to catch — see the trap below.)

The compact name for "how fast the error shrinks with $h$" is **big-O notation**:
the one-sided error is $O(h)$ and the centered error is $O(h^2)$. One caution if
you have met big-O before: in algorithms it usually measures *running time*, where
a higher power is *worse* (slower). Here it measures something different — the
*approximation error as a function of the step size $h$* — and because $h$ is a
number below $1$, raising it to a higher power makes it **smaller**. So $O(h^2)$ is
the *better*, more accurate one, not the worse one.

### The fine print on gradient checks

A few practical rules make a gradient check trustworthy (these follow CS231n's
neural-networks-3 notes — a standard reference):

- **Compare with relative error, not raw difference.** A gap of $0.1$ is tiny
  between gradients near $1000$ and huge between gradients near $0.001$. So divide
  the gap by the larger of the two magnitudes. With $f_a$ the analytic value and
  $f_n$ the numerical one:
  $$
  \text{rel err} = \frac{|f_a - f_n|}{\max(|f_a|, |f_n|)}
  $$
  Read aloud: *the size of the disagreement, as a fraction of the bigger number.*
  (When both values are exactly zero the denominator is treated as $1$ so the
  error is $0$; for a vector of gradients, take the largest entry-by-entry
  relative error.) Rule of thumb: above $10^{-2}$ your gradient is almost
  certainly wrong;
  $10^{-4}$ is usually fine; $10^{-7}$ is excellent. For the smooth functions in
  this lesson, aim for $10^{-7}$.
- **Use float64 (double precision), not float32.** A float32 number carries only
  about 7 significant digits. At $h = 10^{-5}$ the tiny difference
  $f(x+h) - f(x-h)$ can be swamped by that limited precision — and a perfectly
  correct gradient can *look* broken purely from float32 rounding noise. When a
  gradient check fails, the grader's message calls out float32 rounding as one
  likely cause.
- **Watch out for kinks.** A **kink** is a sharp corner where a function suddenly
  changes slope — like the V-shaped bottom of $f(x) = |x|$ at $x = 0$, or the
  bend in a ReLU. Right at a corner there is no single slope (the slope jumps),
  so a centered difference that straddles the corner averages the two sides into a
  misleading middle number. That is not a bug in your code; it is the function
  having a corner. Check that your test point sits *away* from any kink before you
  trust — or distrust — the result.
- **Turn off regularization when checking the data loss.** Regularization is an
  extra penalty term some losses add (you will meet it later). If it is large it
  can dominate the total and hide a real bug in the main "data" part of the
  gradient, so check the two parts separately.

Putting it together, the recipe you run to gradient-check a function is:

1. **Get the analytic gradient** $f_a$ — the slope from your calculus formula at
   the test point.
2. **Get the numerical gradient** $f_n$ — the centered difference
   $\tfrac{f(x+h) - f(x-h)}{2h}$ at the same point, in float64, perturbing one
   parameter at a time for a vector, with a small step like $h = 10^{-5}$.
3. **Compare them with relative error**
   $\tfrac{|f_a - f_n|}{\max(|f_a|, |f_n|)}$ (the largest entry-by-entry value for
   a vector), having first confirmed the point sits away from any kink.
4. **Judge:** below $\approx 10^{-7}$ here means the analytic gradient is
   trustworthy and you switch the numerical one off; above $\approx 10^{-2}$ means
   it is almost certainly wrong and you go back to the calculus.

You implement the centered-difference estimator yourself as the function
`numerical_gradient` in this lesson, so you own both halves of every gradient
check you ever run.

## What this picture hides

The ball-rolling-downhill story is clean. Real loss landscapes have texture the
smooth bowls in these experiments leave out. Naming them now means that when you
hit them later they feel like recognitions, not shocks:

- **Local minima.** A bumpy landscape can have several valleys, and not all are
  equally deep. Gradient descent only ever walks downhill from where it starts,
  so it settles into whichever valley it happens to roll into — which may not be
  the deepest one. (The bowls in this lesson have exactly one valley, so this
  never bites here.)
- **Saddle points.** A spot that slopes *down* in one direction and *up* in
  another, like a mountain pass. The gradient is zero there — flat in every
  direction at that instant — yet it is not a minimum. High-dimensional
  landscapes are full of these.
- **Conditioning.** When the landscape is far steeper in one direction than
  another (the ill-conditioned quadratic you will run in the experiments), a
  single learning rate cannot suit both: it has to be small to stay stable on the
  steep wall, but then it barely moves along the shallow direction, so the path
  zig-zags slowly down. You will *see* this happen, and it is exactly what
  motivates the smarter optimizers (momentum, Adam) in M5.

For the convex quadratics in this lesson none of this bites — each has a single
lowest point and gradient descent reliably finds it. ("Convex" means
bowl-shaped, with no extra bumps to get stuck in.) The point is to know that the
smoothness is a property of these toy problems, not of the world.

## Exercise

Implement in `my_work/gradient-descent/gradient_descent.py` (run
`uv run grade start L01-gradient-descent` to get the scaffold):

- `mse(y, y_hat)` — the **warm-up**, recalled from L0. Parameters: `y` is the
  array of true answers, `y_hat` the array of model predictions (same length); it
  returns their mean squared error, a single number. It is graded on its own and
  is *not* used by the functions below; re-authoring the loss from memory anchors
  the intuition before the gradient machinery.
- `numerical_gradient(f, x, h=1e-5)` — the centered-difference estimator.
  Parameters: `f` is the function you want the slope of (it takes a parameter
  vector and returns one number — "f" is just short for "function"); `x` is the
  point, a parameter vector, where you want the slope; `h` is the tiny step size.
  Work in float64, perturb one coordinate of `x` at a time, and return shape
  `(d,)` — one slope per parameter (`d` is the number of parameters).
- `gradient_descent(f, grad_f, x0, lr, steps)` — the optimizer loop. Parameters:
  `f` is the function to minimize; `grad_f` is its **gradient function** — hand it
  a point and it returns the gradient $\nabla f$ there (the analytic gradient from
  calculus, not the numerical estimate); `x0` is the starting parameter vector
  ("x-zero", where the walk begins); `lr` is the learning rate $\eta$ (the step
  size); `steps` is how many updates to take. Apply
  $x \leftarrow x - \eta\, \nabla f(x)$ `steps` times and return the final
  parameter vector, shape `(d,)`. Do not mutate the caller's `x0` (copy it first).

Then grade it: the site's Run Grader button or `uv run grade L01-gradient-descent`.

## How the grader checks you

- The **warm-up** `mse` is an independent unit: it is graded and recorded on its
  own and never blocks the rest. A wrong warm-up does not hide a working
  `numerical_gradient`.
- `numerical_gradient` is checked against the analytic gradient of $f(x) = x^2$
  at $x = 2$ — where the exact slope is $2x = 4$, and the point is deliberately
  away from any kink. The relative error must be below $10^{-7}$. A second check
  runs it at the kink of $f(x) = |x|$ at $x = 0$ and confirms `numerical_gradient`
  returns ~0 there (the average of the −1 and +1 slopes). That ~0 is the
  *correct* answer at a kink, not a failure — and if this check ever does fail,
  the grader's message explains the kink caveat rather than blaming your code.
- `gradient_descent` is run on the round bowl $f(x, y) = x^2 + y^2$ from
  $(3, 3)$ and must reach the minimum at the origin, and on the ill-conditioned
  $f(x, y) = x^2 + 100 y^2$ where it must still make progress but converges
  visibly slower — the learning-rate sensitivity you explore in the experiments.

### The named trap: the one-sided difference

This grader is built to catch one specific mistake by name. If you write
`numerical_gradient` with the **one-sided** difference
$\tfrac{f(x+h) - f(x)}{h}$ instead of the **centered**
$\tfrac{f(x+h) - f(x-h)}{2h}$, its error is $O(h)$ rather than $O(h^2)$ — too
large to pass the $10^{-7}$ bar on $f(x) = x^2$. The grader recognizes this exact
failure and tells you the cause is the one-sided form, so you know to switch to
the centered formula rather than hunting a phantom bug elsewhere.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the formula with shapes, then pseudocode. Try
level 1 before climbing — and if your numbers desync from the paper trace, the
hints point you back to `by_hand_check.py`.

## Done when

`uv run grade L01-gradient-descent` shows all-green: the warm-up passes, `numerical_gradient`
matches analytic gradients to $10^{-7}$ and passes the kink check (~0 at the
corner) without failing, and `gradient_descent` gets within $0.01$ of the origin
on the bowl and decreases the loss on the ill-conditioned quadratic. Then run the
experiments and record your predictions.
