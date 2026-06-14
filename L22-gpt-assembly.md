# L20 — The Transformer Block and GPT Assembly

You have all the parts; this lesson screws them together into a working GPT.

A quick recap of where the parts came from. L16 built a **bigram language model**:
one lookup table that maps the current token straight to a row of **logits** — raw,
unnormalized scores, one per possible next token, that a softmax later turns into a
probability distribution. (A *token* is one unit of text — here a single character.
The *vocabulary* is the fixed set of distinct tokens, and `vocab_size` is how many
there are.) L17 added **embeddings** — instead of treating a token as a bare id, we
look it up in a table and get a learned vector of `n_embd` numbers (`n_embd` is the
"embedding dimension", the width of that vector) — plus a **context window**, so the
model reads several earlier tokens, not just one. L18 and L19 built **causal
self-attention**: the mechanism that lets each token gather information from the
earlier tokens in its context, first with one **head** (one attention pattern) and
then with several heads in parallel.

This lesson **assembles** those parts into a **GPT** — a decoder-only transformer
language model. Concretely: token and positional embeddings turn ids into vectors, a
stack of **transformer blocks** repeatedly refines those vectors, a final
normalization tidies them up, and a bias-free **output head** turns each refined
vector into logits over the whole vocabulary, at *every* position at once. We will
define each new term — transformer block, residual stream, LayerNorm, feed-forward
MLP, pre-norm, positional embedding, unembedding — from zero, then walk the real
tensor shapes through one block and then through the full model.

We open with a 30-second warm-up that re-builds the L16 bigram model from memory —
spaced retrieval anchors the old idea before we pile new ones on top — and then
build the GPT four pieces at a time.

## The shapes we carry, named once

Everything in this lesson moves the same kind of tensor around, so it helps to fix
the names up front. A **tensor** is just a multi-dimensional array of numbers; its
**shape** is the tuple of its axis lengths.

- `B` — the **batch** size, how many independent sequences we process at once.
- `T` — the **time** or sequence length, how many tokens are in each sequence
  (`T` for "time", since the tokens are read left to right like a clock).
- `n_embd` — the embedding width, the number of features in each token's vector.

The activation flowing through the model is shaped `(B, T, n_embd)`: for each of the
`B` sequences and each of the `T` positions, a vector of `n_embd` numbers. Read that
shape aloud as "batch by time by embedding width". The token ids that go *in* are
shaped `(B, T)` — just an integer per position, no feature axis yet. The logits that
come *out* are shaped `(B, T, vocab_size)` — a full score-per-vocabulary-word at
every position.

## LayerNorm: the per-token stabilizer

The first new piece is a small bookkeeping layer that keeps numbers from blowing up
or vanishing as they pass through a deep stack. It is called **LayerNorm** (short for
"layer normalization").

Here is the problem it solves. We are about to stack many layers on top of each
other. If each layer can scale its output up a little, those little scalings compound
across depth, and by the top of the stack the numbers are either enormous or tiny —
either way, training stalls. LayerNorm resets the scale at each step so no layer can
run away.

What it does, concretely, to one token's vector of `n_embd` numbers:

1. Compute that vector's **mean** (its average value) and **variance** (the average
   squared distance from the mean — a measure of spread).
2. Subtract the mean and divide by the standard deviation (the square root of the
   variance). After this the vector has mean ≈ 0 and **unit variance** — a standard,
   predictable spread of ≈ 1. (It is "≈" rather than exactly 1 because the divisor
   adds a tiny constant `eps` for numerical safety; the spread is essentially 1.)
3. Multiply by a learned per-feature **scale** vector and add a learned per-feature
   **shift** vector, so the layer can still represent any scale it actually needs —
   normalization is a starting point the model can undo, not a straitjacket.

Two details matter for what follows. First, LayerNorm normalizes **per token, over
the `n_embd` axis** — each token's vector is standardized on its own, independently of
the other tokens in the sequence and independently of the rest of the batch. (This is
what distinguishes it from "batch normalization", which mixes statistics across the
batch; we never use that here.) Second, the learned scale and shift give it exactly
`2 * n_embd` parameters: one weight and one bias per feature. We will need that count
later when we add up the whole model.

## The residual stream: an additive bus

The second new piece is not a layer but a *wiring pattern*, and it is the idea the
whole transformer is organized around. It is called the **residual stream**.

Picture a single running activation of shape `(B, T, n_embd)` that flows straight up
through the model from input to output. Every layer in the stack does *not* replace
this activation. Instead it **reads** the current value, computes a small correction,
and **adds that correction back**:

$$x \leftarrow x + \text{sublayer}(x)$$

Read aloud: "the new `x` is the old `x` plus whatever the sublayer computed from it."
(The arrow $\leftarrow$ means "gets reassigned to"; it is an update, not an equation
to solve.) The `+ x` part is the **residual connection** (also called a "skip
connection") — a wire that carries the input straight past the sublayer and adds it to
the output. Because the original `x` is always carried forward unchanged on this wire,
the running activation acts like a shared **bus** that every layer writes onto by
addition. That bus is the residual stream.

Two payoffs, both worth stating plainly:

- **Each sublayer only has to learn a correction**, a small delta to the running
  representation, not to rebuild the whole thing from scratch. That is a far easier
  job, which is why these models train.
- **Gradients have a clean path home.** When we later train by backpropagation,
  the gradient (the signal that says how to nudge each parameter) can flow straight
  back down the `+ x` wire to every layer without being squeezed through every
  sublayer in between. The additive bus is also a gradient highway.

Both of a block's sublayers — the attention and the MLP below — live *inside* one of
these residual adds.

> A common misread: the residual stream is not a separate stored tensor you have to
> manage. It is simply the variable `x` as it gets reassigned `x = x + sublayer(x)`
> down the forward pass. "Reading from and adding back to the stream" is exactly that
> one line of code, repeated.

## The feed-forward MLP: per-position processing

Attention, from L18–L19, moves information *between* positions — it lets a token pull
in features from earlier tokens. But on its own it does almost no nonlinear *thinking*
about the features it gathered; mechanically it is a weighted average of value vectors.
The job of crunching each position's vector through a nonlinearity belongs to the
other sublayer: a small **feed-forward network**, applied **position-wise**.

"Position-wise" means the exact same little network is applied to each token's vector
independently — position 3's vector and position 10's vector go through identical
weights, with no mixing between positions. (All the cross-position mixing already
happened in attention; this sublayer is purely per-token.) The network itself is a
two-layer **MLP** — a *multi-layer perceptron*, the plain "stack of fully-connected
layers with a nonlinearity between them" you built back in L9. Here it has a specific
shape:

1. A **Linear** layer that expands the width from `n_embd` up to `4 * n_embd`. (A
   `Linear(a, b)` layer is a learned matrix multiply from `a` inputs to `b` outputs
   plus a bias vector — `a*b + b` parameters.)
2. A **GELU** nonlinearity. GELU ("Gaussian Error Linear Unit") is a smooth,
   curved cousin of the ReLU you have seen before: like ReLU it mostly passes
   positive values and suppresses negative ones, but it does so with a smooth curve
   instead of a hard corner at zero. It is the activation GPT-2 uses, which is why we
   use it. Its only job here is to be the **nonlinearity** — the bend that lets two
   stacked Linears compute something a single Linear could not.
3. A second **Linear** that projects back down from `4 * n_embd` to `n_embd`, so the
   output fits back onto the residual stream.
4. **Dropout** — during training only, randomly zero out a fraction of the numbers to
   discourage over-reliance on any one feature. With `dropout=0.0` (our default) it
   does nothing; it is here so the architecture matches the canonical one.

In one line, with `Linear(in, out)` read as "a Linear from `in` features to `out`":

```text
FeedForward(x) = Dropout( Linear(4*n_embd, n_embd)( GELU( Linear(n_embd, 4*n_embd)(x) ) ) )
```

Both Linears carry a bias. Why the `4x` expansion specifically? It is the standard
GPT-2 ratio: widening to `4 * n_embd` in the middle gives the nonlinearity room to
compute a richer set of intermediate features before the second Linear compresses the
result back down to the stream's width. It is a deliberate, copied-from-GPT-2 choice,
not a number we derived — keeping it matches the reference so the parameter count
checks out.

## Pre-norm vs post-norm: the decision this lesson turns on

We now have two sublayers (attention and the feed-forward MLP), a residual add to wrap
each in, and a LayerNorm to keep the scale sane. The one real design choice is **where
the LayerNorm goes relative to its sublayer**. There are exactly two options, and they
behave very differently.

**Post-norm** (the original 2017 transformer paper) normalizes *after* the residual
add:

$$x \leftarrow \text{LayerNorm}\big(x + \text{sublayer}(x)\big)$$

Here the sublayer receives the **raw residual** `x` — un-normalized — and the
LayerNorm is applied to the sum at the end.

**Pre-norm** (GPT-2 and essentially everything since) normalizes *before* the
sublayer, and adds the raw residual back:

$$x \leftarrow x + \text{sublayer}\big(\text{LayerNorm}(x)\big)$$

Here the sublayer receives a **normalized input**, and the thing added back onto the
stream is the original un-normalized `x`.

Read the pre-norm line aloud: "the new `x` is the old `x` plus the sublayer applied to
the LayerNorm of `x`." The crucial difference is what travels up the residual bus. In
pre-norm, the LayerNorm sits *off to the side*, on the path into the sublayer only — so
an **un-normalized identity path** runs clean from the very bottom of the network to
the very top, never squeezed through a normalization. That clean path is what makes
deep stacks trainable in practice, and it is why every modern GPT uses pre-norm.

This curriculum uses pre-norm, matching the canonical `references/gpt.py` that later
lessons import. The grader does not take our word for it — it proves the placement with
a **probe**. It attaches a hook to the first block's attention sublayer and inspects the
exact tensor that sublayer *receives*. Under pre-norm the attention sublayer sees
`LayerNorm(x)`, so that captured tensor is per-feature normalized: each token's vector
has mean ≈ 0 and standard deviation ≈ 1. Under post-norm the attention sublayer would
see the raw residual instead, whose mean and std are all over the place — and the probe
fires. (You will reproduce this exact contrast by hand in the experiments.)

## Token and positional embeddings: the input

So far the residual stream has been a `(B, T, n_embd)` tensor of vectors, but the model
is *given* token ids of shape `(B, T)` — plain integers. Two embedding tables turn ids
into the starting vectors.

- The **token embedding** is the table from L17: an `nn.Embedding(vocab_size, n_embd)`,
  i.e. one learned `n_embd`-vector per vocabulary word. Looking up a token id returns
  its vector. This is *what* the token is.
- The **positional embedding** is new here, and it fixes a real gap. The attention
  *computation* has no built-in sense of order — it works on its inputs as an unordered
  *set* of vectors, never their positions. (The L18 causal mask does depend on position,
  but it only forbids looking forward; it does not tell a token *which* slot it sits in.)
  So unless we tell the model where each token is, it cannot tell "dog bites man" from
  "man bites dog". We restore order by adding a
  second learned table, `nn.Embedding(block_size, n_embd)`: one learned vector per
  *position slot*. `block_size` is the longest sequence the model is built to handle —
  the size of the context window — so this table has one row for each possible position
  `0, 1, ..., block_size - 1`. Position `t` always contributes the same learned vector,
  giving the model a stable handle on "this is the third token" versus "the tenth".

We combine them by plain addition, both being `n_embd`-wide vectors:

```text
x = token_emb(idx) + pos_emb(arange(T))     # (B, T, n_embd)
```

Read aloud: "the starting stream is the token's content vector plus its position's
vector." Here `arange(T)` is the list of integers `[0, 1, ..., T-1]` — the position
index for each of the `T` slots — and looking those up in the positional table, then
adding, stamps each token with both *what* it is and *where* it sits. That sum is the
initial value of the residual stream.

## The unembedding: turning vectors back into logits

After the stack and a final LayerNorm, each position holds a refined `n_embd`-vector,
but we need **logits** — one score per vocabulary word, for predicting the next token.
The layer that does this is the **output head**, also called the **unembedding** (it is
the mirror image of the input embedding: embedding maps an id *to* a vector, unembedding
maps a vector *back* to a score for every id).

It is a single `nn.Linear(n_embd, vocab_size, bias=False)` — a learned matrix that
takes each `n_embd`-vector and produces `vocab_size` scores. It is **bias-free** on
purpose, matching the reference; that detail matters for the parameter count, because a
bias would add `vocab_size` extra numbers. Applied at every position, it turns the
`(B, T, n_embd)` stream into the `(B, T, vocab_size)` logits we return.

## Assembling one block, then the whole GPT

Now we wire the pieces together. A **transformer block** is two pre-norm sublayers in
sequence, each inside its own residual add — first attention (to mix across positions),
then the feed-forward MLP (to process each position). Its forward pass, on an input `x`
of shape `(B, T, n_embd)`, is two lines:

1. **Attention sublayer.** Normalize the stream, run multi-head attention on the
   normalized copy, and add the result back onto the raw stream:
   `x = x + attn(ln1(x))`. After this, every token has mixed in information from the
   earlier tokens it attends to. Shape stays `(B, T, n_embd)`.
2. **Feed-forward sublayer.** Normalize the (updated) stream again with a *second*
   LayerNorm, run the position-wise MLP on the normalized copy, and add that back:
   `x = x + ff(ln2(x))`. After this, every token's gathered features have been crunched
   through the nonlinearity. Shape is still `(B, T, n_embd)`.

The block returns `x`, unchanged in shape — `(B, T, n_embd)` in, `(B, T, n_embd)` out.
That shape-preservation is what lets us stack blocks: the output of one is a legal input
to the next.

The full **GPT** wraps a stack of `n_layer` such blocks between the embeddings and the
head. Its `forward(idx)` takes token ids `idx` of shape `(B, T)` (with `T <= block_size`)
and runs:

1. **Embed.** `pos = arange(T)`; `x = token_emb(idx) + pos_emb(pos)`. Shape goes
   `(B, T)` → `(B, T, n_embd)`. The stream now carries content and position.
2. **Input dropout.** `x = drop(x)` — a no-op at `dropout=0.0`, present to match the
   reference.
3. **The block stack.** `for block in blocks: x = block(x)` — run all `n_layer`
   pre-norm blocks in order, each refining the same `(B, T, n_embd)` stream.
4. **Final LayerNorm.** `x = ln_f(x)` — one last normalization (the "f" is for "final")
   so the head reads a well-scaled stream.
5. **Unembed.** `logits = lm_head(x)` — the bias-free output head, mapping each
   `n_embd`-vector to `vocab_size` scores. Shape goes `(B, T, n_embd)` →
   `(B, T, vocab_size)`.

So the whole journey, as one shape walk, is:

```text
idx (B, T)
  -> token_emb + pos_emb        (B, T, n_embd)
  -> n_layer transformer blocks (B, T, n_embd)   # shape preserved at every block
  -> final LayerNorm            (B, T, n_embd)
  -> lm_head                    (B, T, vocab_size)
```

One last design point: `forward(idx)` returns **logits only** — the `(B, T, vocab_size)`
tensor and nothing else. It does **not** compute a loss. That is deliberate: the training
loop (the next lesson, L21) owns the **cross-entropy** loss — it reshapes the logits and
the shifted targets and computes the loss itself. Keeping the loss out of `forward` lets
the model stay a pure function from token ids to next-token scores, equally usable for
training and for generation. A frequent beginner assumption is that a model's `forward`
must return a loss; here it pointedly does not.

## Dataset

This lesson loads no dataset. Everything runs on tiny random integer tensors and the
closed-form parameter count, all on CPU in seconds. The only "data" is the baby config
the grader uses: `GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=32, vocab_size=65)`.

## Exercise

Implement the GPT in `my_work/gpt-assembly/gpt_assembly.py` (run
`uv run grade start L22-gpt-assembly` to get the scaffold). You will build the warm-up bigram
model, the config, and the four GPT pieces. Several pieces — `GPTConfig`,
`MultiHeadAttention`, `FeedForward` — are scaffolded for you; the lines that matter are
in `TransformerBlock.forward` and `GPT`. Each piece below lists its signature, every
parameter, and what it returns.

- `BigramLM(nn.Module)` — the **warm-up**, the L16 bigram model recalled cold and
  graded on its own. `__init__(self, vocab_size)`: `vocab_size` (int) is the number of
  distinct tokens; build a single `nn.Embedding(vocab_size, vocab_size)` named
  `token_logits`, whose row for token `i` *is* the next-token logits. `forward(self, idx)`:
  `idx` is token ids of shape `(B, T)` (long); return `token_logits(idx)`, shape
  `(B, T, vocab_size)`. `generate(self, idx, max_new_tokens, seed=0)`: `idx` is the
  starting ids `(B, T)`; `max_new_tokens` (int) is how many tokens to append; `seed`
  (int) seeds the sampler. Each step take the last position's logits `logits[:, -1, :]`,
  softmax to probabilities, `multinomial`-sample one id, and `cat` it onto `idx`; return
  the grown `(B, T + max_new_tokens)` sequence.

- `GPTConfig` — a `@dataclass` holding the hyperparameters: `n_layer` (number of
  transformer blocks), `n_head` (attention heads per block), `n_embd` (embedding width),
  `block_size` (max sequence length / positional-table size), `vocab_size` (number of
  tokens), and `dropout` (float, default `0.0`). It just bundles the knobs so they
  serialize cleanly into a checkpoint.

- `TransformerBlock(nn.Module)` — one pre-norm block. `__init__(self, config, *, store_attn_weights=False)`
  builds `ln1` and `ln2` as `nn.LayerNorm(n_embd)`, an `attn` (`MultiHeadAttention`), and
  an `ff` (`FeedForward`). `forward(self, x)`: `x` is the residual stream `(B, T, n_embd)`;
  apply the two pre-norm sublayers — `x = x + attn(ln1(x))` then `x = x + ff(ln2(x))` —
  and return `x`, shape `(B, T, n_embd)`. The LayerNorm goes **before** each sublayer,
  never after the add.

- `GPT(nn.Module)` — the assembled model. `__init__(self, config, *, store_attn_weights=False)`
  builds `token_emb` (`nn.Embedding(vocab_size, n_embd)`), `pos_emb`
  (`nn.Embedding(block_size, n_embd)`), `drop` (`nn.Dropout`), `blocks` (an
  `nn.ModuleList` of `n_layer` `TransformerBlock`s), `ln_f` (`nn.LayerNorm(n_embd)`), and
  `lm_head` (`nn.Linear(n_embd, vocab_size, bias=False)` — **bias-free**). `forward(self, idx)`:
  `idx` is token ids `(B, T)` with `T <= block_size`; embed (`token_emb(idx) + pos_emb(arange(T))`),
  dropout, run the block stack, apply `ln_f`, then `lm_head`; return **logits only**,
  shape `(B, T, vocab_size)`. No loss.

Then grade it: the site's Run Grader button or `uv run grade L22-gpt-assembly`.

## How the grader checks you

The grader runs the baby config `GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=32, vocab_size=65)`
— all CPU, deterministic, seconds-fast, no dataset reads. It has one independent root
(the bigram warm-up) and a chain of checks that hang off the parameter count.

- The **bigram warm-up** `BigramLM` is graded on its own and never blocks the rest. It
  generates 5 new tokens from a `(1, 1)` start under a fixed seed and must return shape
  `(1, 6)` (one start token plus five sampled), with every id a valid long in
  `[0, vocab_size)`. A wrong warm-up does not hide a working GPT.

- The **parameter count** is the parent check, and it is exact because the architecture
  matches `references/gpt.py` exactly. The grader recomputes the count *independently*
  from the config — LayerNorm contributes `2 * n_embd`; a biased `Linear(a, b)` adds
  `a*b + b`; the `lm_head` is bias-free — and asserts your `sum(p.numel() for p in GPT(cfg).parameters())`
  equals it. For the baby config that number is **9184**. Any missing or extra bias or
  LayerNorm shows up immediately as a nonzero diff.

- The **forward shape** check feeds `idx` of shape `(3, 8)` through the 2-block stack and
  requires logits of shape `(3, 8, 65)` — `(B, T) -> (B, T, vocab_size)`.

- The **pre-norm placement probe** hooks the first block's attention sublayer and reads
  the tensor it receives. Under pre-norm that tensor is `ln1(x)`, so its per-token mean
  is ≈ 0 (within `1e-4`) and its per-token std is ≈ 1 (within `1e-2`). A post-norm block
  feeds attention the raw residual instead, which fails this test — this is the named
  trap, below.

- The **causal isolation** check runs `idx (1, 8)`, records position 0's logits, then
  permutes the *later* tokens `idx[0, 1:]` and re-runs. Position 0's logits must be
  unchanged (to `1e-5`): position 0 attends only to itself, so nothing after it may leak
  in. This re-tests the L18 causal mask end-to-end through the full model.

- The **single-token edge case** feeds `idx (1, 1)` and requires `(1, 1, 65)` logits with
  no shape error — `arange(1) = [0]` indexes one position, attention is a trivial `1x1`
  matrix, and the stack must still run.

### The named trap: post-norm placement

This grader hunts one specific mistake by name: **post-norm placement**. If you write
the block as `x = ln1(x + attn(x))` — normalizing *after* the residual add instead of
before the sublayer — then the attention sublayer receives the **raw residual**, not
`ln1(x)`. The pre-norm placement probe captures that input, finds its per-token mean and
std are not 0 and 1, and fires with a message that names the cause: *"The LayerNorm must
come BEFORE the sublayer (pre-norm: x = x + attn(ln1(x))), not after the residual add."*
Both placements run and have the same parameter count, so only this probe distinguishes
them — when it fires, switch the LayerNorm to *before* each sublayer rather than hunting
a phantom bug elsewhere.

## The hint ladder

If the grader's message is not enough, open the hints. Each piece —
`bigram_warmup`, `param_count`, `prenorm_placement`, `forward_shape` — has three levels:
a conceptual nudge, then the formula or invariant with shapes, then pseudocode. Try
level 1 before climbing. The `param_count` ladder ends in the full closed-form sum, and
the `prenorm_placement` ladder shows the two-line pre-norm forward — reach for that one
only if the placement probe keeps firing.

## Done when

`uv run grade L22-gpt-assembly` shows all-green: the bigram warm-up generates a `(1, 6)` sequence
of valid ids, the parameter count is exactly **9184**, `forward` returns `(B, T, vocab_size)`
logits through the 2-block stack, the pre-norm probe passes (attention sees a per-feature
normalized input — your block is `x = x + attn(ln1(x))`, not `x = ln1(x + attn(x))`),
position 0 stays causally isolated when later tokens are permuted, and the single-token
`(1, 1)` forward runs. Then run the experiments and record your predictions.
