# L17 — Embeddings and Context (n-gram MLP)

The bigram model from L16 had exactly one token of memory. It was a `V x V`
lookup table — `V` is the **vocabulary size**, the number of distinct tokens
(here, distinct characters) — that maps the current token to a row of
**logits**, the raw, un-normalized scores that a softmax later turns into
next-token probabilities. That model could not tell that "th" is almost always
followed by "e" while a lone " t" might start anything, because it only ever saw
the *single* previous character. This lesson breaks that ceiling along two axes
that turn out to underlie every modern language model: **learned embeddings**
(each token gets a meaning) and **context** (the model reads more than one
previous token at a time). What we build here is the 2003 Bengio neural
language model, and the exact spot where its design hits a wall is the entire
motivation for attention, which comes next.

We open with a 30-second warm-up that re-derives the L5 multiclass `hinge_loss`
from scratch. This is **spaced retrieval** — pulling an old skill back from
memory at a spaced interval so it stays sharp — and it keeps the earlier muscle
warm before the new material.

## An embedding is a lookup table of learned meaning

Start with the word **embedding**, because everything else builds on it. An
embedding is a *learned dense vector* attached to each token: a short list of
real numbers (say 16 of them) that stands in for that token. "Dense" means most
entries are nonzero real numbers, as opposed to a one-hot vector that is all
zeros except a single 1. "Learned" means the network discovers these numbers
during training rather than us writing them down.

In the bigram model the lookup table held logits. Here the table holds these
embeddings instead. `nn.Embedding(V, d)` — PyTorch's embedding layer, built from
a vocabulary of `V` tokens and an embedding width `d` — is a `V x d` matrix we
will call `W`: one row per token, each row `d` numbers wide. Looking up token
`i` returns row `i`, a `d`-dimensional vector.

That lookup is not magic. It is exactly a matrix multiply by a one-hot vector. A
**one-hot vector** for token `i` is length-`V`, zero everywhere except a single
`1` at position `i`:

```text
one_hot(i) @ W = row i of W
```

Read that line aloud: "the one-hot vector for `i`, matrix-multiplied by `W`,
equals row `i` of `W`." Because the one-hot is zero everywhere except position
`i`, multiplying it by `W` picks out row `i` and adds nothing else.
`nn.Embedding` is just that selection, done efficiently without ever building the
one-hot.

Here is the payoff and the reason embeddings matter. A bigram's count table
treats every token as a separate, unrelated symbol: what it learns about "a" tells
it nothing about "e", even though both are vowels that behave alike. Embeddings
fix that. The rows start as random numbers, but training pulls tokens that appear
in similar contexts toward similar vectors — because the network does better when
it can treat similar tokens alike. Two tokens with nearby embedding vectors get
nearly the same prediction, so a pattern learned for one *generalizes* to the
other for free. Counts can't do that; embeddings can. That geometry is real and
inspectable: `nearest_neighbors` reads the embedding matrix and ranks tokens by
distance. At this tiny scale the neighbors are noisy (which is why the grader
never asserts *which* tokens come out), but it is the same mechanism that, at
scale, puts "king" near "queen".

## Context by concatenation

One token of memory is the bigram's whole limitation. Lift it the obvious way:
feed the last `k` tokens instead of just one. The phrase **n-gram** names exactly
this idea — using a fixed window of the previous tokens as context. With a
**context window** of `k` tokens (the fixed number of previous tokens the model
reads), we look at the `k` tokens of history and predict the one that comes next.
`k = 1` reproduces the bigram (one previous token). In this lesson we mostly use
`k = 3`, the three-token-context model we will call the **trigram MLP**
throughout.

How do we turn `k` tokens into one input for the network? Embed each of the `k`
context tokens to its own `d`-vector, then **concatenate** them — lay the `k`
vectors end to end into a single wide vector of length `k*d`:

```text
idx (B, k)  --embed-->  (B, k, d)  --reshape-->  (B, k*d)  --MLP-->  (B, V) logits
```

Here `B` is the **batch size** (how many context windows we process at once).
Read the shapes left to right: we start with a `(B, k)` block of token ids,
embed it to `(B, k, d)` (each id became a `d`-vector), reshape that to
`(B, k*d)` (the `k` vectors per row laid end to end), run it through the MLP, and
out come `(B, V)` logits — one next-token score row per example.

Concatenation, not averaging, is the deliberate choice. Averaging the `k`
embeddings would blur the positions together; concatenation keeps each position
in its own slot, so the MLP can learn that the token two-back and the token
one-back play different roles. On top of the concatenated context sits the **MLP**
(multi-layer perceptron) from L9: a hidden layer `nn.Linear(k*d, hidden)`
followed by a `tanh` nonlinearity, then a **separate** output head
`nn.Linear(hidden, V)` that produces the next-token logits. (`tanh` is the
squashing nonlinearity from L9; `nn.Linear(in, out)` is a learned affine map
from `in` numbers to `out` numbers.) As in L16, `forward` returns **logits
only** — the canonical `train_loop` owns the loss. Here the logits are
per-example `(B, V)` and the targets are `(B,)`, so the train_loop's universal
flatten is a no-op and it just works.

### The forward pass, step by step

For one batch `idx` of shape `(B, k)`, the model computes its logits like this:

1. **Embed.** Look up all `k` context tokens at once:
   `emb = self.embedding(idx)` gives shape `(B, k, d)` — each token id became its
   `d`-dimensional embedding vector.
2. **Concatenate.** Flatten the `k` per-token vectors of each row into one wide
   vector: `flat = emb.reshape(B, k*d)`, shape `(B, k*d)`. This is the
   concatenation step — the `k` embeddings laid end to end.
3. **Hidden layer + nonlinearity.** Push the context through the MLP's first
   layer and squash it: `h = torch.tanh(self.hidden(flat))`, shape
   `(B, hidden)`.
4. **Output head → logits.** Project to one score per vocabulary token:
   `logits = self.out(h)`, shape `(B, V)`. These raw scores are the next-token
   logits; a softmax (applied later, by the loss) would turn each row into a
   probability distribution over the `V` tokens.

## The embedding-gradient invariant (and the weight-tying trap)

Here is the subtle, important fact about using `nn.Embedding` as an **input**
lookup. A **gradient** is the vector of slopes that says, for each parameter, how
the loss would change if you nudged that parameter — it is what training uses to
update weights. When the embedding is an input lookup, gradient flows back **only
to the rows that were actually looked up**. If a batch's `k`-token contexts never
reference token 7, then row 7 of `embedding.weight` gets **exactly zero** gradient
on that step — it did not participate in the forward pass, so it cannot be blamed
for the loss. You can check this to the bit: run one forward + backward on a batch
whose contexts cover only a subset of the vocabulary, and the unused rows'
gradient is `0`.

The natural bug that destroys this invariant is **weight tying** — reusing one
weight matrix in two places. It is tempting to reuse the embedding matrix as the
output projection: drop `self.out`, project the hidden state down to `embed_dim`
(say `z = self.proj(h)`, shape `(B, embed_dim)`), then multiply by the embedding
to get logits — `logits = z @ self.embedding.weight.T`. (`A.T` is the transpose,
rows and columns swapped; `@` is matrix multiply. The projection step is what
makes the shapes line up: `z` is `(B, embed_dim)` and `embedding.weight.T` is
`(embed_dim, V)`, so the product is `(B, V)`.) But now every vocabulary row
appears in the **output** head, so every row receives gradient through the output
on *every* step — including rows that were never looked up on the input side.
Those unused input rows go nonzero, and the invariant breaks. The fix is to keep a **separate**
`self.out` head, which is what the worked solution does. (Weight tying is a real,
useful technique in production language models — but it deliberately couples input
and output, which is exactly why it is the wrong default while you are still
learning what the input lookup does on its own.) The grader builds a
subset-covering batch, runs one step, and asserts the unused rows' gradient norm
is zero.

## More context beats the bigram floor

Does the extra context actually help? Yes, measurably. The yardstick is the
**bigram counting floor** — the loss you get from the plain count-and-normalize
bigram model, the one L16 built. To compute it, count every (current, next)
transition in the corpus, normalize each row into a probability distribution, and
average the **NLL** (negative log-likelihood — `-log` of the probability the
model assigned to the token that actually came next; lower is better) over the
training split. This is "smoothing-free" (no probability is nudged off zero), and
it is the same floor L16 hit. On this lesson's corpus that floor is about
`0.5524`.

Now train the trigram MLP (`k = 3`) on that same corpus. With two extra tokens of
context, it should land **below** the floor: seeing more history lets it resolve
ambiguities the one-token bigram cannot. The grader requires the trained trigram
NLL to beat the bigram floor by at least `0.05`.

## The scaling wall — why attention exists

The n-gram MLP works, and it is genuinely better than the bigram. But look at how
it grows. Concatenation makes the first layer `k*d` wide, so the input weight
matrix has `k*d*hidden` parameters — **linear in the context length `k`**. With
the defaults (`d = 16`, `hidden = 64`), that is `1024` parameters at `k = 1`,
`2048` at `k = 2`, `3072` at `k = 3`: every extra token of context adds another
fixed `1024`-parameter slab. Want 10 tokens of context? That is ten slabs —
`10240` parameters, ten times the `k = 1` model. Want 50? Fifty slabs, fifty
times as wide.

And it is worse than just size. Because each position has its own slab of weights,
the model cannot **share** what it learns about a token across positions — "cat"
sitting in slot 1 and "cat" sitting in slot 3 are processed by completely
different parameters, so a pattern learned in one slot has to be relearned in
every other. Fixed-width concatenation is a dead end for long context.

That is the cliffhanger. The next leap — **attention** — keeps a fixed parameter
budget no matter how long the context, and lets every position consult every
other directly. Everything you just built (embeddings, context, an MLP head over
context vectors) survives into the transformer; only the way context is *mixed*
changes. Hold that thought.

## Exercise

Implement, in `my_work/ngram-mlp/ngram_mlp.py` (run `uv run grade start L19-ngram-mlp` to
get the scaffold):

- `hinge_loss(scores, y, margin=1.0)` — the L5 multiclass hinge warm-up, numpy
  only. Parameters: `scores` is an `(n, C)` array of class scores (`n` examples,
  `C` classes); `y` is a length-`n` array of true class ids; `margin` is the
  hinge margin (default `1.0`). For each example, sum
  `max(0, s_j - s_{true} + margin)` over the **wrong** classes only — exclude the
  true-class term `j == y` — and return the mean over the `n` examples, a single
  float.
- `NGramLM(nn.Module)` — the model. Its `__init__(vocab_size, embed_dim=16, k=3,
  hidden=64)` builds `self.embedding = nn.Embedding(vocab_size, embed_dim)` (the
  input lookup), a hidden layer `nn.Linear(k*embed_dim, hidden)` followed by
  `tanh`, and a **SEPARATE** `self.out = nn.Linear(hidden, vocab_size)` head.
  `forward(idx)` takes `idx` of shape `(B, k)` (a batch of `k`-token contexts)
  and returns `(B, vocab_size)` **logits only** — no loss.
- `make_ngram_batches(tokens, k)` — build the training pairs. Parameters:
  `tokens` is a 1-D token stream; `k` is the context length. Slide a `k`-token
  window over the stream: `X[i] = tokens[i:i+k]` (the context) and
  `Y[i] = tokens[i+k]` (the next token). Returns `X` of shape `(B, k)` and `Y` of
  shape `(B,)`.
- `nearest_neighbors(model, char_idx, n)` — inspection. Parameters: `model` is a
  trained `NGramLM`; `char_idx` is the token id to query; `n` is how many
  neighbors to return. Rank every *other* embedding row by Euclidean distance to
  row `char_idx` and return the `n` nearest as a list of `int`s.
- `ngram_nll(model, tokens, k)` — the mean next-token NLL of `model` on a token
  stream. Build the `(B, k)` contexts and `(B,)` targets, get `(B, vocab)`
  logits, and return their cross-entropy as a float.
- `train_ngram(tokens, vocab_size, k=3, embed_dim=16, hidden=64, steps=400,
  lr=1.0, seed=0)` — train an `NGramLM` via the canonical `train_loop` (one fixed
  full-stream batch per step) and return `(model, loss_history)`.

Then grade it: the site's Run Grader button or `uv run grade L19-ngram-mlp`.

## How the grader checks you

The grader runs five checks over a fixed inline character-level corpus
(`"the cat sat on the mat " * 24`, vocabulary size 10) — all on CPU, seconds-fast,
no dataset reads. (The model is built — and its embedding rows drawn — before the
`seed` takes effect, so the `seed` argument does not pin those initial numbers;
the floor-beat check clears its bar by a comfortable margin regardless.) Each
failure message names the value found, the value expected, and the concept to
revisit.

- **The L5 hinge warm-up** is an independent root: it is graded on its own so a
  wrong warm-up never masks a working model. It is checked on a fixed
  `scores`/`y` pair where the correct (true-class-excluded) answer is `2.5/3` and
  the wrong (true-class-included) answer is `5.5/3`; the grader recognizes the
  `5.5/3` value and tells you the true-class term leaked in.
- **The embedding-gradient invariant** is the other independent root. The grader
  builds a batch whose `k`-token contexts use only token ids `{0, 1, 2}` while the
  vocabulary has 12 tokens, runs one forward + backward, and asserts that the
  rows never looked up have **exactly zero** gradient norm.
- **The floor-beat** trains a trigram (`k = 3`, 400 steps, seed 0) and requires
  its NLL to beat the bigram counting floor (≈ `0.5524`) by at least `0.05`.
- **The shape contract** feeds an `(B, k)` batch and checks `forward` returns
  `(B, vocab_size)` logits.
- **`nearest_neighbors`** is checked for shape only: a list of `n` integer ids
  (no content assertion, since embedding geometry at this scale is noisy).

### The named trap: weight-tied output head

This grader is built to catch one mistake by name, `embedding_gradient_unused`.
If you reuse the embedding matrix as the output projection — project the hidden
state down to `embed_dim` and compute `logits = z @ self.embedding.weight.T`,
with no separate `self.out` head — then every vocabulary row sits in the output
path, so every row picks up gradient through the output on every step,
*including* input rows that were never looked up. Those unused rows go nonzero and the invariant breaks. The grader detects
exactly this failure ("Unused embedding rows received non-zero gradient") and
points you to the fix: keep a separate `self.out = nn.Linear(hidden,
vocab_size)` head so unused input rows leak no gradient.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the formula or invariant with shapes, then
pseudocode. Try level 1 before climbing — the hints cover the hinge true-class
exclusion, the weight-tying trap, and the concatenate-the-context shape walk.

## Done when

`uv run grade L19-ngram-mlp` shows all-green: the L5 hinge warm-up matches (true-class term
excluded), one backward step leaves unused embedding rows at exactly zero
gradient (the weight-tying trap), the trained trigram NLL beats the bigram
counting floor by at least `0.05`, `forward` maps `(B, k)` to `(B, vocab_size)`
logits, and `nearest_neighbors` returns `n` integer ids. Then run the experiments
and record your predictions.
