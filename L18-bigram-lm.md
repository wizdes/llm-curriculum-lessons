# L16 — Bigram Language Model

This is the first model in the course that **generates text**. A **language
model** is a model that, given some text so far, assigns a probability to each
possible next token — "token" being whatever unit we chop text into; in this
lesson a token is a single character. Generate by repeatedly asking "what comes
next?" and picking from those probabilities.

The **bigram** model is the simplest language model that works. "Bigram" means
"two in a row": it predicts the next token using nothing but the *current* token.
No earlier context, no memory, no attention — just a single lookup table that
says "after token *i*, here is how likely each token is to come next." Build that
table, sample from it, and characters spill out with real letter-pair statistics
and zero idea what a word is. That gap — plausible pairs, nonsense words — is the
entire motivation for everything after this lesson.

We open with a 30-second warm-up that re-derives the L4 numerically-stable
`bce_from_logits` from scratch (`bce` = binary cross-entropy, the loss for a
yes/no prediction). Re-authoring an earlier piece cold is spaced retrieval: it
keeps the muscle warm before new material. After that, the new content begins.

## The model is one lookup table

A bigram model assigns a probability to the next token given the current one,
written `p(next | current)` and read "the probability of *next* given
*current*." With a vocabulary of `V` tokens — `V` distinct characters — that is a
`V × V` table: one **row** per current token, and along that row, a number for
every token that could come next. Row `i` is the model's answer to "if the
current token is `i`, how likely is each token to follow?"

We store this table as `nn.Embedding(V, V)`. An **embedding** here is just a
lookup table of learnable rows: `nn.Embedding(V, V)` holds `V` rows, each of
length `V`, and indexing it with token `i` returns row `i`. We do not store the
probabilities directly — we store **logits**: raw real-valued scores, one per
possible next token, that have not yet been turned into probabilities. A logit
can be any number (negative, zero, large); **softmax** is the function that
turns a row of logits into a probability distribution (exponentiate each score,
then divide by the sum so the row is non-negative and adds to 1). So row `i`
holds the logits over the next token given current token `i`; softmax turns that
row into the probabilities `p(next | current = i)`.

That is the whole model. `forward` — the method that runs the model on an input —
maps token ids of shape `(B, T)` to logits of shape `(B, T, V)`. Read the shapes
aloud: `B` is the batch size (how many sequences at once), `T` is the sequence
length (how many tokens per sequence), `V` is the vocabulary size. So for every
one of the `B × T` positions, `forward` hands back the `V` next-token logits
implied by the token sitting at that position.

One contract matters here. `forward` returns **logits only** — no loss. The
frozen canonical `train_loop` (the shared training loop from earlier lessons,
which you do not rewrite) calls `logits = model(X)` and computes the loss itself.
A model that returns its own loss does not fit that contract, so keep `forward` a
pure logits function. (The training loop flattens those logits before it scores
them; the next section explains why.)

## The next-token shift

To train a next-token predictor you need, for each position, the token that
*actually* came next. Our raw data is one long 1-D stream of token ids — the
corpus, encoded character by character. Turning that stream into (input, target)
pairs is a one-position shift:

```text
X = tokens[:-1]   every token except the last
Y = tokens[1:]    every token except the first
```

`tokens[:-1]` reads "all tokens up to but not including the last one";
`tokens[1:]` reads "all tokens from index 1 onward." Line them up and `Y[t]` is
exactly the token that **followed** `X[t]` in the original stream. That is the
supervision signal: at position `t` the model sees `X[t]` and must predict
`Y[t]`.

This shift is the single most common thing to get wrong, so be deliberate about
it. Set `Y = X` and you are asking the model to predict the token it is *already
looking at*. That task is trivial — the table just learns to copy each token to
itself — so the *training* loss may drop nicely and look healthy, while the model
has learned nothing about what comes next. The damage shows up where it counts:
its NLL on the real next-token task stays far above the counting floor. (A
correctly trained model *reaches* that floor; it never goes below it, since the
floor is the best any bigram can do.) If your trained model stays stuck above the
floor (defined two sections down) even though training "converged," check this
shift first.

## NLL as loss

What does "good" mean for a language model? It should assign **high probability
to the text that actually occurs**. So for each position `t`, look at the one
number that matters: `p(Y[t] | X[t])`, the probability the model assigned to the
token that truly came next. We want that number near 1 at every position.

A loss is something we **minimize**, but "assign high probability" is something
we want to *maximize* — so we flip it. Take the negative logarithm of that
probability and average it over every position. That average is the **negative
log-likelihood**, or **NLL**:

```text
NLL = mean over t of  -log p(Y[t] | X[t])
```

Read aloud: "for each position, take the model's probability for the true next
token, take its negative log, then average over all positions." Three pieces, in
plain terms:

- **Likelihood** is the probability the model assigns to what actually happened —
  here, `p(Y[t] | X[t])`. High likelihood = the model was confident *and right*.
- The **log** turns probabilities (numbers in `(0, 1]`) into scores. `log 1 = 0`
  (no penalty for a perfectly confident, correct prediction) and `log` of a tiny
  probability is a large negative number (a big penalty). Two reasons we use it.
  First, a sequence's total probability is a *product* of per-token
  probabilities, and `log` turns that product into a *sum* — far easier to work
  with and numerically stable (multiplying thousands of small numbers underflows
  to zero; summing their logs does not). Second, `-log` punishes confident-wrong
  predictions harshly: the closer the true token's probability gets to 0, the
  closer `-log` climbs toward `+∞`.
- The **negative** sign flips "maximize a probability near 1" into "minimize a
  loss near 0," which is the form gradient descent expects.

A worked example makes this concrete. Take a 3-character corpus over the
characters `a`, `b`, `c` — the string `abac` (vocabulary `V = 3`). Slide a
two-character window across it to read off every transition:

```text
a b a c
└─┘             a -> b
  └─┘           b -> a
    └─┘         a -> c
```

**Step 1 — count.** Tally every `(current, next)` pair into the `3 × 3` count
matrix (rows = current, columns = next):

```text
         next: a   b   c
current a   [  0   1   1  ]    a was followed once by b, once by c
        b   [  1   0   0  ]    b was followed once by a
        c   [  0   0   0  ]    c never appeared as a current token
```

**Step 2 — normalize to probabilities.** Divide each row by its own sum, turning
counts into a distribution that adds to 1 across the row:

```text
         next:  a    b    c
current a   [   0   0.5  0.5  ]    after a: b or c, equally likely
        b   [   1    0    0   ]    after b: always a
```

(Row `c` had no transitions, so it stays undefined — `c` is never a *current*
token in this corpus, so we never ask its row.) Row `a` is **uncertain** (a
50/50 split), row `b` is **certain**. That contrast is the point of the example.

**Step 3 — score with NLL.** For each of the 3 transitions, read off the
probability the table assigned to the token that actually came next, take its
negative log, and average:

```text
a -> b :  p = 0.5    -log(0.5) = 0.693
b -> a :  p = 1.0    -log(1.0) = 0.000     ← certain & correct: no penalty
a -> c :  p = 0.5    -log(0.5) = 0.693

NLL = (0.693 + 0.000 + 0.693) / 3 = 0.462
```

The certain-and-correct transition (`b -> a`) costs nothing; the two uncertain
ones each cost `log 2 ≈ 0.693`. The average, `0.462`, is this model's NLL. Lower
is better, and `0` would mean the model was perfectly confident and correct
everywhere — impossible here, because `a` genuinely branches.

This NLL is not a special new loss. It is exactly **cross-entropy** between the
model's predicted distribution over the next token and the **one-hot** target (a
vector that is 1 at the true next token and 0 everywhere else). So the training
loss the `train_loop` computes and the evaluation NLL we report are the *same
quantity*, measured in two places. This is the loss every later language model in
the course is measured by — when you read "this model gets 1.8 NLL," it means
exactly the average computed above.

One shape wrinkle makes the cross-entropy call concrete. PyTorch's
`cross_entropy` wants `(N, C)` logits and `(N,)` integer targets — `N` rows, each
a length-`C` logit vector, paired with `N` correct-class indices. But our model
emits `(B, T, V)` logits and we hold `(B, T)` targets. So flatten the leading
dims first: `(B, T, V) → (B*T, V)` and `(B, T) → (B*T,)`, collapsing "batch ×
time" into one long list of `N = B*T` positions. The `train_loop` does exactly
this flatten for the training loss; the `bigram_nll` eval helper must do its own.
Passing `(B, T, V)` straight into `cross_entropy` is a shape error, not a
slightly-off number.

## Two routes to the same table

There are two ways to fill the bigram table, and they converge on the same place:

- **Counting.** Walk the corpus, tally every `(current, next)` transition into
  the `V × V` matrix, and normalize each row to a distribution — exactly the
  worked example above. No gradient descent at all; this is the closed-form
  **maximum-likelihood** bigram model (the table that, by construction, makes the
  observed text as probable as possible).
- **Training.** Initialize `nn.Embedding(V, V)` to random logits and minimize NLL
  with the canonical `train_loop`. Gradient descent on cross-entropy slowly
  reshapes those logits until softmax of each row matches the empirical
  next-token frequencies.

These are the same objective — minimize NLL — solved two ways, so they land on
the same distribution. The trained model's NLL should therefore **reach the
counting model's NLL as a floor**: the counting model is the best any bigram can
do, so it bounds the trained NLL from below, and training should bring the
trained NLL right down to it. The grader checks exactly this: trained NLL is
within a small tolerance of the counting NLL. It is framed as a bound, not an
equality, because optimization stops a hair short of the exact minimum.

One subtlety the grader is careful about: it scores on the **training split** —
the same text the table was built from. The danger of scoring on **held-out**
text (text not used to build the table) is that this smoothing-free counting
model assigns probability zero to any bigram it never saw in training — `c`'s row
above is literally empty — and `-log(0) = +∞`, so a *single* held-out pair the
model never saw sends the whole average to `+∞`. On any varied real text a
held-out sample almost always contains some pair the training text never had, so
held-out NLL almost always blows up; scoring on the training split sidesteps this
and keeps the floor finite and the comparison meaningful. (On a tiny, highly
repetitive corpus a split might happen to contain no new pairs and stay finite —
but that is luck, not safety.) The experiments make you trigger this blow-up on
purpose.

## Generating text

With a filled table, generation is a loop. Start from a token, and repeat: read
its next-token logits, softmax to a distribution, **sample** the next token from
that distribution, then feed the sampled token back in as the new current token.
As numbered steps, the loop is:

1. Hold a current token id (start from `start_idx`).
2. Run the model on it to get that token's row of next-token logits.
3. Apply softmax to turn the row into a probability distribution.
4. **Sample** one token from that distribution — draw a token at random, each
   one weighted by its probability (`torch.multinomial`). This is *not* argmax;
   sampling is what makes the output vary instead of repeating one token forever.
5. Append the sampled id to the output; make it the new current token; go to 2
   until you have `max_new_tokens` of them.

Seed a private `torch.Generator` (PyTorch's random-number source) so a fixed seed
reproduces the exact same sequence — the grader depends on that determinism. Read
a sample aloud and you hear the bigram's limit: plausible letter pairs, nonsense
words. A model with one token of memory cannot do better, which is precisely why
the next lessons add context.

## Exercise

Implement in `my_work/bigram-lm/bigram_lm.py` (run `grade start bigram-lm` to get
the scaffold). Each function below lists its signature, every parameter, and what
it returns.

- `bce_from_logits(z, y)` — the L4 stable BCE warm-up, recalled cold.
  Parameters: `z` is the array of logits (raw real-valued scores, any shape);
  `y` is the array of 0/1 labels (same shape). Returns the mean binary
  cross-entropy as a single float, computed via the softplus form
  `softplus(z) - y*z` (= `np.logaddexp(0, z) - y*z`) so it stays **finite even at
  `z = ±1000`** — never forming `p = sigmoid(z)`, which would take `log(0)`.
- `BigramLM(nn.Module)` — the model. `__init__(self, vocab_size)` stores one
  `nn.Embedding(vocab_size, vocab_size)` (the `V × V` logit table). `forward(idx)`
  takes token ids `idx` of shape `(B, T)` and returns the table rows for them:
  `(B, T, vocab_size)` logits, **logits only** — no loss, no tuple — so it drops
  into the canonical `train_loop`.
- `make_bigram_batches(tokens)` — the next-token shift. Parameter: `tokens` is a
  1-D sequence of token ids of length `T+1`. Returns `(X, Y)`, each shape
  `(1, T)` (a leading batch dim of 1), with `X = tokens[:-1]` and
  `Y = tokens[1:]` so `Y[t]` is the token that followed `X[t]`.
- `bigram_nll(model, tokens)` — mean next-token NLL. Parameters: `model` is a
  `BigramLM`; `tokens` is the 1-D token stream to score. Builds `(X, Y)`, runs
  the model, and returns the mean NLL as a float — **owning its own**
  `(B, T, V) → (B*T, V)` and `(B, T) → (B*T,)` reshape before `cross_entropy`.
- `generate(model, start_idx, max_new_tokens, seed)` — deterministic
  autoregressive sampling. Parameters: `model` is a trained `BigramLM`;
  `start_idx` is the integer id of the first current token; `max_new_tokens` is
  how many tokens to produce; `seed` seeds a private `torch.Generator`. Returns a
  1-D `LongTensor` of length `max_new_tokens`, every id in `[0, vocab_size)`.

Then grade it: the site's Run Grader button or `grade bigram-lm`.

## How the grader checks you

The grader runs five checks over a fixed inline character-level corpus —
`"the cat sat on the mat "` repeated — all on CPU, deterministic, seconds-fast.
No dataset reads.

- **The L4 BCE warm-up** runs `bce_from_logits` on logits that include `±1000`
  and requires a finite result, then checks it equals the reference
  `mean(logaddexp(0, z) - y*z)` to within `1e-9`. A non-finite value means you
  formed `sigmoid(z)` and hit `log(0)`.
- **`make_bigram_batches`** is checked on `[5, 7, 2, 9, 4]`: it must return the
  next-token shift, `X = [5, 7, 2, 9]` and `Y = [7, 2, 9, 4]`. If `X == Y` it
  reports the missing shift by name (see below).
- **The counting floor.** It trains a `BigramLM` with the canonical `train_loop`,
  then computes both the trained NLL (`bigram_nll`) and the counting-bigram NLL
  on the training split. It requires `trained ≤ counting_floor + 0.05` — the
  trained table must descend to the floor, not stall far above it.
- **`generate`** must return a 1-D `LongTensor` of the requested length, with
  every id in `[0, vocab_size)`, and produce the *same* sequence on two calls
  with the same seed (determinism).
- **`bigram_nll`** must reshape `(B, T, V)` logits before `cross_entropy`. The
  check compares your value against the correctly-reshaped reference; an
  un-reshaped call either raises or returns a far-off number (see below). On an
  untrained near-uniform table this NLL sits near `log(vocab)`.

Each failure message names the value found, the value expected, and the concept
to revisit.

### The named trap: the missing next-token shift and the un-reshaped NLL

This grader hunts two specific mistakes by name, each on its own independent
check so its teaching message fires unmasked:

- **`bigram_offset`** — `make_bigram_batches` sets `Y = X` (often
  `y = tokens[:-1]` instead of `tokens[1:]`), so the targets are *not* shifted.
  The model is then trained to predict the token it already sees — the identity
  copy, which it can learn fine, so it is *not* learning next-token prediction and
  its NLL on the real shifted task never reaches the counting floor. The grader
  detects `X == Y` directly and tells you the targets must be shifted:
  `X = tokens[:-1]`, `Y = tokens[1:]`.
- **`cross_entropy_reshape`** — `bigram_nll` passes `(B, T, V)` logits straight
  into `cross_entropy` without flattening to `(B*T, V)` / `(B*T,)`. The grader
  recognizes this exact failure and names it — "passing (B,T,C) logits
  untransposed" — so you fix the reshape rather than hunt a phantom bug.

## The hint ladder

If a grader message is not enough, open the hints. Each has three levels, mildest
first: a conceptual nudge (the symptom and what it means), then the formula with
shapes (the invariant it lives on), then pseudocode. There are ladders for the
next-token shift, the eval-time reshape, and the logits-only `forward` contract.
Try level 1 before climbing.

## Done when

`grade bigram-lm` shows all-green: the BCE warm-up stays finite at `±1000`;
`make_bigram_batches` produces the next-token shift (`X = tokens[:-1]`,
`Y = tokens[1:]`); a trained `BigramLM` reaches the counting-bigram NLL floor on
the training split; `generate` returns a deterministic length-`max_new_tokens`
1-D tensor with ids in range; and `bigram_nll` reshapes the `(B, T, V)` logits
before cross-entropy. Then run the experiments and record your predictions.
