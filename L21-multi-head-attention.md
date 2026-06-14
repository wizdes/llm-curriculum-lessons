# L19 — Multi-Head Attention

In L18 you built one **self-attention head**: a layer that lets each token in a
sequence look back at the earlier tokens and pull in a weighted mix of their
information. (A **token** is one unit of text — a byte or a sub-word piece from
the L15 tokenizer. The **sequence** is the list of tokens we feed in, length `T`.
**Self-attention** means the tokens attend to *each other*, all from the same
sequence.) That single head learns **one** way to mix the sequence: one set of
query/key/value projections, so one attention pattern.

But a sentence has many relationships running at once — subject/verb agreement,
the antecedent of a pronoun ("it" → which noun?), nearby modifiers, raw position.
One head has to compromise across all of them with its single pattern.
**Multi-head attention** is the fix: run several heads in parallel, each a smaller
self-attention head working on its own slice of the embedding, then **concatenate**
their outputs (lay them end to end) and project the result back to the original
width. Each head is free to specialize — one can track agreement, another can
track the pronoun's antecedent — and together they give the layer a richer view of
the same tokens. That parallel-heads layer is the whole lesson.

We build it twice — a naive version first, then an efficient one — and prove the
two compute the same function. Before the new material, the worked file opens with
a 30-second warm-up that re-derives the L15 byte-level BPE `encode` cold: spaced
retrieval keeps an earlier skill from fading.

A reminder on two terms you will see throughout. A **projection** is just a
matrix-multiply by learned weights — `nn.Linear(in, out)` maps an `in`-wide vector
to an `out`-wide one. **Softmax** turns a row of raw numbers into a row of
positive weights that sum to 1 (a probability distribution); the attention weights
are a softmax, so each token's weights over the past form a distribution.

## Heads vs dim

Here is the arithmetic the whole lesson turns on, and the first thing to get
right. We do **not** add new width when we add heads. The model width `n_embd`
(the size of each token's embedding vector — "n-embed", the number of features per
token) is fixed. The heads **split** that fixed width between them:

$$
\texttt{head\_size} = \frac{\texttt{n\_embd}}{\texttt{n\_heads}}
$$

Read aloud: each head's working width, `head_size`, equals the model width divided
by the number of heads. (Read `n_heads` as "the number of heads".) So **adding a
head shrinks every head's slice** — the heads share one fixed budget, they do not
each get a fresh `n_embd`. Two concrete cases at `n_embd = 64`:

| `n_embd` | `n_heads` | `head_size = n_embd / n_heads` |
|---------:|----------:|-------------------------------:|
| 64       | 4         | 16                             |
| 64       | 8         | 8                              |

Going from 4 heads to 8 at the same `n_embd = 64` halves each head's width from 16
to 8. This is the single most common beginner mistake on this topic: thinking more
heads means a *bigger* layer. It does not — more heads means *more, thinner* heads
inside the **same** budget. (We require `n_embd` to divide evenly by `n_heads`, so
the split has no remainder; both classes above check this and raise a clear error
otherwise.)

So why split at all? The heavy compute is roughly the same either way — the fused
query/key/value matmul is `n_embd -> 3*n_embd` whether you have 1 head or 8, and
the per-head attention matmuls sum back to the single-head cost (eight `8`-wide
heads do the same arithmetic as one `64`-wide head, just sliced). The split adds
only minor overhead: the `T x T` scores, the mask, and the softmax now carry an
`n_heads` axis, so those scale with the head count — but they are cheap next to the
matmuls. What changes for the better is **expressiveness**: with 8 heads you get
**8 independent `T x T` attention patterns** instead of one, so different heads can
attend to different relationships in parallel. The trade-off runs the other way:
each head's subspace is only `head_size` wide, so too many heads makes each one too
thin to represent much. Typical transformers pick `head_size` around 64 and let
`n_heads` follow from `n_embd / 64`. The split is nearly free in compute and buys
parallel specialization; the real cost is shrinking each head's subspace.

(`T x T` is the shape of one head's attention pattern: with `T` tokens in the
sequence, every token gets a weight over every token, so the pattern is a `T`-by-`T`
grid of weights. Read `T x T` as "T by T".)

## Two builds: the naive wrapper and the weight split

We write the layer twice. The first is obviously correct and easy to read; the
second is what real code uses. Proving they are the same function is what makes
the efficient one trustworthy.

**`MultiHeadAttentionWrapper`** is the obviously-correct version: a Python list of
`n_heads` separate `SelfAttentionHead` modules (the single head from L18,
re-implemented here so the lesson stands alone), each built at width `head_size`.
`forward` runs every head on the same input `x` and concatenates the per-head
outputs along the last axis:

```python
def forward(self, x):
    # each head(x) is (B, T, head_size); n_heads of them concatenate to (B, T, n_embd)
    return torch.cat([head(x) for head in self.heads], dim=-1)
```

`torch.cat([...], dim=-1)` glues the tensors together along their last axis, so
`n_heads` outputs of width `head_size` become one output of width
`n_heads * head_size = n_embd`. (Here `B` is the **batch size**, how many
sequences we process at once; `T` is the sequence length. So `(B, T, n_embd)` is
"B sequences, each `T` tokens long, each token an `n_embd`-wide vector".) This
version is clear but wasteful: it fires `3 * n_heads` separate little projection
matmuls — three (Q, K, V) for each of the `n_heads` heads.

**`MultiHeadAttention`** is the same computation with a single **fused** QKV
projection and one batched matmul. "Fused" means we replace those `3 * n_heads`
little projections with one `nn.Linear(n_embd, 3*n_embd, bias=False)` that produces
every head's queries, keys, and values in a single matmul. (Every projection here —
the per-head Q/K/V, the fused QKV, and the output `proj` — is bias-free,
`bias=False`. That is what lets the equivalence proof copy weight rows around
without also having to match biases.) The two builds are **provably
equivalent**: lay the fused projection's rows out as the wrapper's per-head Q/K/V,
set the output projection to the identity, and they compute the same function.
The grader checks the two outputs match to a tolerance of `1e-5` (one part in a
hundred thousand — floating-point round-off, not a real difference). In real code
you always reach
for the fused version; the wrapper exists only to make the equivalence visible.

## The fused forward, step by step

The efficient `MultiHeadAttention.forward` is four moves on top of the single-head
attention you already have. Let `B` be the batch size, `T` the sequence length,
`n_embd` the model width, `n_heads` the number of heads, and
`head_size = n_embd // n_heads`. The input `x` is `(B, T, n_embd)`. Here is the
exact procedure, with the shape after each step:

1. **Fused projection.** Run `qkv = self.qkv(x)`, one `nn.Linear(n_embd, 3*n_embd)`
   matmul → shape `(B, T, 3*n_embd)`. Then `q, k, v = qkv.split(n_embd, dim=-1)`
   splits that last axis into three equal blocks, each `(B, T, n_embd)` — the
   combined queries, keys, and values for *all* heads.
2. **Reshape into heads.** The `n_embd`-wide axis is *already* the heads laid end
   to end. View it as `(n_heads, head_size)` and move the head axis up next to the
   batch axis:
   `q = q.view(B, T, n_heads, head_size).transpose(1, 2)` → `(B, n_heads, T, head_size)`.
   Do the same for `k` and `v`. Now the last two axes `(T, head_size)` are exactly
   what one single-head attention expects, and the leading `(B, n_heads)` axes ride
   along — so the next matmul runs every head of every sequence at once.
3. **Batched causal attention.** This is L18's attention, now over a 4-D tensor:
   - `scores = q @ k.transpose(-2, -1) / sqrt(head_size)` → `(B, n_heads, T, T)`
     (the `@` is matrix-multiply; dividing by `sqrt(head_size)` is L18's scaling).
   - Mask the **strict upper triangle** of `scores` to `-inf` — the positions
     where a token would look at a *future* token. This is the **causal** rule, and
     it happens **before** the softmax.
   - `weights = softmax(scores, dim=-1)` → `(B, n_heads, T, T)`: one `T x T`
     attention matrix per head, each row a distribution over the allowed past.
   - `out = weights @ v` → `(B, n_heads, T, head_size)`: each token's output is its
     weighted mix of the value vectors.
4. **Recombine and project.** Move the head axis back and flatten the
   `(n_heads, head_size)` axes into the single `n_embd` axis — that flatten *is* the
   concatenation, expressed as a reshape:
   `out = out.transpose(1, 2).contiguous().view(B, T, n_embd)`. Then a final
   `out = self.proj(out)` with `nn.Linear(n_embd, n_embd)` lets the heads' outputs
   mix → `(B, T, n_embd)`.

So the layer takes `(B, T, n_embd)` in and returns `(B, T, n_embd)` out — same
shape, but every token now carries information mixed from the allowed past across
`n_heads` parallel patterns.

**A worked shape trace.** Take `B = 2`, `T = 6`, `n_embd = 16`, `n_heads = 4`, so
`head_size = 16 / 4 = 4`. The input `x` is `(2, 6, 16)`. Step 1: `qkv` is
`(2, 6, 48)` and splits into `q, k, v`, each `(2, 6, 16)`. Step 2: each reshapes
to `(2, 6, 4, 4)` then transposes to `(2, 4, 6, 4)` — 2 sequences × 4 heads, each
a 6-token × 4-wide head. Step 3: `scores` and `weights` are `(2, 4, 6, 6)` — a
6×6 pattern per head; `out` is `(2, 4, 6, 4)`. Step 4: transpose+flatten back to
`(2, 6, 16)`, output projection keeps it `(2, 6, 16)`. The width you started with
is the width you end with.

**A memory-layout note (not part of the contract).** A `transpose` returns a
**non-contiguous view** — the tensor's logical axes are reordered but its bytes in
memory are not, so the flattening `.view(B, T, n_embd)` in step 4 would fail. Call
`.contiguous()` first (it makes a clean copy in the right byte order), or use
`.reshape`, which inserts that copy for you. This is a plumbing detail, not part
of what the layer computes.

## Why the wrapper and the fused version are the same function

It can feel like magic that a single fused matmul reproduces `n_heads` separate
heads exactly. It is not — it is bookkeeping. The fused `qkv.weight` is a
`(3*n_embd, n_embd)` matrix; its rows are just the per-head Q, K, V weight rows
stacked in a fixed order. Head `h`'s query weights occupy fused rows
`[h*head_size : (h+1)*head_size]`; its key weights sit `n_embd` rows further down;
its value weights `2*n_embd` further down. Copy the wrapper's per-head weights into
exactly those slots, set the output projection `proj` to the identity matrix (so
it passes the concatenated heads through untouched), and the fused version computes
the wrapper's output bit for bit (to `1e-5`). The grader does precisely this copy
and compares. Same function, one matmul instead of `3 * n_heads`.

## The attention-weight retrievability contract

Later (in M5) we will *visualize* what each head attends to — which means reading
the post-softmax attention weights back out of a trained module. To make that
possible, this lesson fixes a small contract that M5 depends on **exactly**:

```python
MultiHeadAttention(n_heads, n_embd, store_attn_weights=False)
```

- `store_attn_weights` is a **constructor** flag — you set it once when the module
  is built, *not* as an argument to `forward`. (The constructor is `__init__`, the
  code that runs when you create the object.)
- When it is `True`, **every** `forward(x)` stores `self.last_attn_weights` = the
  post-softmax attention weights, shape `[B, n_heads, T, T]`, **`.detach()`-ed**.
  (`.detach()` returns a tensor cut loose from the autograd graph — the record
  PyTorch keeps to compute gradients — so reading the weights later does not pin
  that whole graph in memory.)
- When it is `False` (the default), `self.last_attn_weights` stays `None`.

M5's visualization reads `module.last_attn_weights` **directly** — no forward
hooks (PyTorch's mechanism for snooping on a layer mid-run), no re-running the
model. Storing the detached weights on the forward pass is the entire mechanism.
The grader pins all of it: the shape `[B, n_heads, T, T]`, that each row sums to 1
(it is a softmax), that the causal upper triangle is 0 (no future leakage), that
it is detached, and that leaving the flag off leaves `last_attn_weights` as `None`.

## Exercise

Implement in `my_work/multi-head-attention/multi_head_attention.py` (run
`uv run grade start L21-multi-head-attention` to get the scaffold). The single head is
re-implemented in this file — do **not** import it from L18.

- The **warm-up** — `train_bpe`, `encode`, `decode`, the L15 byte-level BPE. These
  are recalled cold for spaced retrieval and graded as an independent unit, so a
  rusty warm-up never blocks the multi-head work. The one that matters:
  `encode(text, merges)` — `text` is the input string; `merges` is the dict of
  learned merge rules mapping a byte-pair to its new id (insertion order = rank).
  It returns a list of int token ids by encoding over the raw UTF-8 **bytes**
  (`list(text.encode("utf-8"))`, not `list(text)`) and **re-applying** the learned
  merges in rank order (earliest-learned pair first) — it does not re-count
  frequencies. `decode(ids, merges)` must round-trip it back to the original
  string.

- `SelfAttentionHead(n_embd, head_size)` — one causal head (L18), re-implemented
  here. `n_embd` is the model width (input features per token); `head_size` is
  this head's working width. `forward(x)` takes `x` of shape `(B, T, n_embd)`,
  projects to query/key/value of width `head_size`, runs scaled dot-product
  attention with the strict upper triangle masked to `-inf` **before** softmax, and
  returns `(B, T, head_size)`.

- `MultiHeadAttentionWrapper(n_heads, n_embd)` — the naive build. `n_heads` is the
  number of heads; `n_embd` the model width (must divide evenly by `n_heads`). It
  builds `n_heads` independent `SelfAttentionHead`s each of width
  `head_size = n_embd // n_heads`. `forward(x)` takes `x` `(B, T, n_embd)`, runs
  every head, and **concatenates** the `(B, T, head_size)` outputs along the last
  axis back to `(B, T, n_embd)`.

- `MultiHeadAttention(n_heads, n_embd, store_attn_weights=False)` — the efficient
  fused build. `n_heads` and `n_embd` are as above; `store_attn_weights` is the
  retrievability flag described in the contract section (a constructor argument,
  default `False`). `forward(x)` takes `x` `(B, T, n_embd)`, runs the four-step
  fused forward (project → reshape into heads → batched causal attention → recombine
  and output-project), and returns `(B, T, n_embd)`. When `store_attn_weights` is
  `True`, every `forward` also sets `self.last_attn_weights` to the detached
  post-softmax weights, shape `[B, n_heads, T, T]`; when `False`, it stays `None`.

Then grade it: the site's Run Grader button or `uv run grade L21-multi-head-attention`.

## How the grader checks you

- The **warm-up** BPE is an independent root: it is graded and recorded on its own,
  so a wrong `encode` never hides working multi-head code. It learns a few merges
  on a fixed corpus and requires `decode(encode(text, merges), merges) == text`
  (an exact round-trip), and that `encode` returns a list of `int` ids.
- **Wrapper vs. split equivalence** (the parent check the rest hang off): the
  grader builds `MultiHeadAttentionWrapper(2, 8)` and `MultiHeadAttention(2, 8)`,
  copies the wrapper's per-head Q/K/V into the fused `qkv.weight` and sets `proj`
  to the identity, runs both on the same random `(3, 5, 8)` input, and requires the
  two outputs to match to `atol=1e-5`.
- **Shape contract:** `MultiHeadAttention(n_heads=4, n_embd=16)` on `x` of shape
  `(2, 6, 16)` must return `(2, 6, 16)`.
- **Retrievability contract:** with `store_attn_weights=False`, `last_attn_weights`
  must stay `None` after a forward; with `store_attn_weights=True` it must hold a
  tensor of shape `[B, n_heads, T, T]` whose every row sums to 1, whose strict
  upper triangle is 0 (causal, no future leak), and whose `requires_grad` is
  `False` (it was detached).
- **`h=1` reduction:** with `n_heads=1`, `MultiHeadAttention` is one head plus an
  output projection. The grader copies its fused Q/K/V into a single
  `SelfAttentionHead`, sets `proj` to the identity, and requires the two to match
  to `1e-5`.

This lesson has **no named trap**. The attention design decisions that *do* have
traps — masking before softmax, the `sqrt(head_size)` scaling — were pinned by
L18's trap suite, and L19's content is structural (the head split and the
retrievability contract), not a single-line bug. The trap file is a valid empty
list, so there is nothing extra to dodge here.

## The hint ladder

If the grader's message is not enough, open the hints. Each piece has three levels:
a conceptual nudge, then the formula with shapes, then pseudocode — for the BPE
warm-up, the head concatenation and width, the reshape/transpose plumbing (including
the `.contiguous()` gotcha), and the `store_attn_weights` contract. Try level 1
before climbing.

## Done when

`uv run grade L21-multi-head-attention` shows all-green: the BPE warm-up round-trips, the
naive wrapper and the fused `MultiHeadAttention` compute the identical function to
`1e-5`, the shape contract holds (`(2, 6, 16)` in → `(2, 6, 16)` out), the
retrievability contract holds (shape `[B, n_heads, T, T]`, rows sum to 1, causal
upper triangle 0, detached, and `None` when the flag is off), and the `h=1` case
reduces to a single `SelfAttentionHead`. Then run the experiments and record your
predictions.
