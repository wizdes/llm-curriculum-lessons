# L25 — Attention Visualization

You have trained a GPT and run ablations on it. This lesson opens the box one
more way: it reads the **attention weights** the model computed back out and
turns them into something you can look at.

By the end you will be able to extract, for any prompt, the exact numbers each
attention head used to decide which earlier tokens to mix together — and you will
know what those numbers do and do not tell you.

## What an attention map is

Start from the operation you built in the single- and multi-head attention
lessons. Inside an attention layer, each position in the sequence plays two roles
at once:

- as a **query** — the position that is doing the looking, asking "which earlier
  tokens are relevant to me?"
- as a **key** — a position that can be looked *at*, offering itself as a possible
  source of information.

For one **head** (one independent copy of the attention machinery, with its own
learned projections — a layer has several heads running in parallel), the model
computes, for every query position, a number for every key position: how strongly
that query attends to that key. Those numbers are the **attention weights**. Run
them through a **softmax** — the function that turns a row of arbitrary scores
into non-negative numbers that sum to 1 — and each query's row becomes a
**post-softmax distribution**: a probability distribution over the key positions,
saying what fraction of its attention the query spends on each earlier token.

Collect every query's row and you get one matrix per head. For a sequence of $N$
tokens it is an $N \times N$ matrix — $N$ rows, $N$ columns. Read it like this:

- **row $i$** is "where token $i$ looked" — its full distribution over keys.
- **column $j$** is "who looked at token $j$" — how much attention every query
  paid to key $j$.
- **entry $(i, j)$** is the single weight: how much of query $i$'s attention went
  to key $j$. Because each row is a probability distribution, the entries in a row
  sum to 1.

This per-head $N \times N$ matrix is what we call an **attention map**. Stack the
heads within a layer, then stack the layers, and you have a complete, legible
picture of the routing the model used on this one input: which tokens each head in
each layer pulled information from, and how strongly.

## The causal mask: why half the matrix is zero

The model is a **causal** (also called **decoder-only**) transformer: it predicts
each token from the tokens *before* it, so a query at position $i$ is only allowed
to look at positions $j \le i$ — the present and the past, never the future. That
restriction is enforced by the **causal mask**, which drives the scores for future
positions to $-\infty$ *before* the softmax runs. Since $\text{softmax}$ sends
$-\infty$ to a weight of exactly $0$, every "look into the future" entry comes out
as a hard zero.

In matrix terms, the future positions for query $i$ are the columns $j > i$ —
everything strictly above the main diagonal. That region is the **strict upper
triangle** ("strict" = not counting the diagonal itself, where $j = i$). So:

- the **diagonal and lower triangle** ($j \le i$) hold the real attention weights,
  and each row there sums to 1;
- the **strict upper triangle** ($j > i$) is exactly $0$ in every valid map.

This is not a soft tendency the model happened to learn — it is a hard structural
property of the architecture. Our extractor treats any strict-upper-triangle cell
whose absolute value exceeds $10^{-6}$ (a tiny tolerance for floating-point dust,
not real weight) as a contract violation, for a reason we will come back to in the
traps.

## How we read the weights out

A subtlety: the model computes these weights on every forward pass and then
*throws them away* — they are an intermediate value, not part of the output. To
look at them, we have to ask the model to keep them around. The frozen
`references.gpt` model exposes them through exactly one mechanism, fixed in the
multi-head attention lesson:

```text
GPT(config, store_attn_weights=True)
```

- `store_attn_weights` is a **constructor** flag — you set it once, when the model
  is built, not as an argument to a forward call.
- It threads down to every block's attention module. With the flag set to `True`,
  each forward pass stashes the post-softmax weights on the module as an
  attribute named `last_attn_weights` (literally "the attention weights from the
  most recent forward pass").
- The stored value is **detached** from the autograd graph — `.detach()`-ed —
  meaning it is the same numbers with no gradient bookkeeping attached (it is not
  tracked for backpropagation), so reading it later does not pin the training
  graph in memory.
- Without the flag, `last_attn_weights` stays `None` (Python's "no value here").

The stored attribute has shape `[B, n_head, T, T]`. Read that shape aloud,
left to right:

- `B` — the **batch** size, how many sequences were run at once;
- `n_head` — the number of attention **heads** in the block;
- `T` — the **sequence length** in tokens (the time axis);
- the final `T` — the same length again, because each head's map is $T \times T$
  (one row per query, one column per key).

So `[B, n_head, T, T]` reads as: "for each sequence in the batch, for each head,
a query-by-key matrix of attention weights."

`attention_maps(model, prompt_tokens)` does the extraction. Read its procedure as
an explicit sequence:

1. Wrap `prompt_tokens` into a single-sequence batch — one row, so the batch axis
   `B` is `1`.
2. Put the model in `eval()` mode and run one forward pass under `no_grad()` (so
   nothing trains and no autograd graph is kept). This pass is what populates each
   block's `last_attn_weights`.
3. Walk the blocks in order. For each block, read its stored `last_attn_weights`.
   If that attribute is `None`, raise (the flag was never set — see below).
4. Drop the batch axis (we only ran one sequence, so `[1, n_head, T, T]` becomes
   `[n_head, T, T]`) and convert each head's matrix to a plain NumPy array.
5. Verify the causal contract on that array — every strict-upper-triangle cell
   must be zero (the check from the previous section) — and raise if it is not.
6. Append the verified `(H, N, N)` array to the per-layer list.

It returns:

```text
{"layers": [ndarray(H, N, N), ...], "tokens": [str, ...]}
```

Read aloud: `"layers"` maps to a list with one entry per transformer block; each
entry is a NumPy array of shape `(H, N, N)` — `H` heads, each an $N \times N$
query-by-key map, where `H = model.config.n_head` and `N = len(prompt_tokens)`.
`"tokens"` maps to the list of token labels, one per column/row, so you can read
the map against the actual prompt.

There is **no forward-hook fallback** — no second, sneaky way to grab the weights
by attaching a callback to the model. A fallback that always worked would make the
stored-attribute contract dead code, and the point of the contract is to have
exactly one path. If the attribute is `None` (the model was built without
`store_attn_weights=True`), `attention_maps` does not guess or fabricate; it
**raises** `AttentionContractError`, naming the missing attribute and the flag you
forgot, so the failure is loud and the fix is obvious.

## Why per-head, and why per-layer

We keep every head's map separately instead of averaging the heads into one
matrix. The reason is the whole point of multi-head attention: different heads
learn different jobs. One head might track the previous token, another might point
a pronoun back at its antecedent, another might just spread weight broadly.
Average them and those distinct patterns blur into a featureless smear — you would
destroy exactly the structure you opened the box to see. Same logic across depth:
early layers and late layers route differently, so we keep one map per layer too.

## A concrete example

Take a 5-token prompt, so $N = 5$, and look at one head's $5 \times 5$ map. A
plausible, "reasonable"-looking pattern for a trained model — weight concentrated
on recent tokens, future hard-zeroed — might be:

```text
        key0   key1   key2   key3   key4
query0  1.00   0      0      0      0
query1  0.30   0.70   0      0      0
query2  0.10   0.20   0.70   0      0
query3  0.05   0.10   0.15   0.70   0
query4  0.05   0.05   0.10   0.20   0.60
```

Read it row by row. Row 0 (token 0) can attend to nothing but itself — it is the
first token, so its only legal key is key 0, and the row is forced to `1.00`
there. Every later row puts the most weight on the diagonal and the tokens just
before it (a "local, recent-token" pattern) and tapers off toward the start. Every
row sums to $1.0$. And the entire strict upper triangle — the cells above the
diagonal — is exactly $0$, because those are future positions the causal mask
killed before the softmax. That hard-zero upper triangle is the single most
reliable thing you will see in any of these maps; it shows up trained or untrained,
because it comes from the architecture, not from learning.

## A caveat that matters

Here is the most important sentence in the lesson, and the easiest one to misuse.
Attention weights tell you this:

> which tokens the model attended to and how strongly — a window into the computation, not a causal explanation of why the model produced this output

In four words: attention ≠ explanation.

A high attention weight does *not* prove the model "used" that token to decide,
and a low weight does *not* prove it ignored it. The actual value flowing through a
head, the contributions of the other heads, and every downstream layer all shape
the final result — attention weights are only the routing of the *mix*, not the
content of what was mixed or what was done with it afterward. Read an attention
map as a **hypothesis-generator** about routing — "this head seems to look back at
the subject" — and then test that hypothesis (for example with the ablations from
the previous lesson), not as a verdict on the model's reasoning.

## The two traps

The grader is a *test-the-teacher* setup: it feeds in two deliberately broken
versions of the extractor and checks that the contract you learned would catch
each one. Understanding why each is wrong is the lesson.

**Trap 1 — reading masked cells (`mask_zeros_attn`).** This version simply skips
the upper-triangle check. On a real model that might look harmless, because a
correct causal model already zeros the future. But the check is your guarantee,
not the model's: drop it and a future-key cell — a query supposedly attending to a
token that comes *after* it — can leak straight into the map, and you would read
it as if the model peeked at the future. The intuition: the strict upper triangle
must be hard-zero, so the extractor *verifies* it and raises rather than handing
back a map you would misread. (The check fires when any strict-upper-triangle cell
exceeds $10^{-6}$ in absolute value — a tiny tolerance for floating-point dust,
not a license for real weight.)

**Trap 2 — the missing-flag error (`missing_store_flag`).** This version, when it
finds `last_attn_weights` is `None`, quietly fabricates a plausible-looking
uniform map and returns it instead of failing. That is the dangerous bug: you
forgot `store_attn_weights=True`, and instead of telling you, the tool hands back
confident-looking numbers that the model never computed. The intuition: a missing
attribute means a contract was broken, and the only safe response is to **fail
loudly** — raise `AttentionContractError` naming the missing attribute and the
flag — never to invent data. A wrong answer that looks right is worse than an error
that points at the fix.

## Exercise

Implement in `my_work/attention-viz/attention_viz.py` (run
`grade start attention-viz` to get the scaffold). You write two functions and may
import only `references.*` plus numpy/torch — no cross-lesson imports.

- `row_sum_check(attn)` — the **warm-up**, recalled cold from the softmax idea.
  Parameter: `attn` is a NumPy array of attention weights; its last axis indexes
  the key positions of one row. It returns a single boolean: `True` iff every row
  sums to `1.0`. Sum along the last axis and compare to `1.0` with `np.allclose`'s
  `atol=1e-5` (its default `rtol` then lets a sum off by `~1e-5` still pass, but a
  sum off by `1e-4` fails). This function is graded on its own and is *not* called by
  `attention_maps`; re-authoring the distribution invariant from memory anchors it
  before the extraction.
- `attention_maps(model, prompt_tokens, layer=None)` — the extractor. Parameters:
  `model` is a `references.gpt.GPT` you must build with `store_attn_weights=True`;
  `prompt_tokens` is the list of integer token ids to run (length `N`); `layer` is
  an optional integer — when given, return only that one block's map instead of
  all of them. It runs the model once and returns the dict
  `{"layers": [ndarray(H, N, N), ...], "tokens": [str, ...]}`, where `H =
  model.config.n_head` and `N = len(prompt_tokens)`. Read each block's stored
  `attn.last_attn_weights` (the single frozen mechanism — no forward hooks); raise
  `AttentionContractError` if that attribute is `None`/absent (name the missing
  attribute **and** the `store_attn_weights=True` constructor flag) or if any
  head's strict upper triangle holds a cell above `1e-6` (the message must contain
  the words "upper triangle").

Then grade it: the site's Run Grader button or `grade attention-viz`.

## How the grader checks you

- The **warm-up** `row_sum_check` is the dependency root and is graded on its own,
  so a broken warm-up reports its own message instead of being masked by the
  heavier checks downstream. It is run against five synthetic 2-row matrices: a
  valid one (rows sum to exactly 1), one off by `1e-6` (must still pass — the
  tolerance is `1e-5`, not exact equality), one off by `1e-4` (must fail), an
  all-zeros row summing to 0 (must fail), and a row summing to 1.2 (must fail).
- `attention_maps` is checked on a fixed-seed 2-layer, 2-head, 32-dim CPU model
  built with `store_attn_weights=True` and run on a 5-token prompt. The grader
  pins all of it: the result has one map per block (2 layers); each map has shape
  `(H, N, N) = (2, 5, 5)`; every row of every head sums to 1 (it reuses your
  `row_sum_check`); every head's strict upper triangle is zero; the token list has
  one label per prompt token; and **no forward hook** is registered on the model
  before or during extraction (the contract is the stored attribute, not a hook).
- A no-flag sub-check builds the same model **without** `store_attn_weights=True`
  and confirms `attention_maps` raises `AttentionContractError` rather than
  returning anything — a model that never stored its weights must fail loudly.

### The named trap: the two test-the-teacher bugs

This is a *test-the-teacher* grader: it feeds your contract two deliberately
broken extractors and checks that your code's guarantees would catch each one.
Both are spelled out in **The two traps** above. In grader terms: the
`mask_zeros_attn` trap drops the upper-triangle check, so the triangle-labeling
test fires (its message names the *unenforced causal mask*); the
`missing_store_flag` trap silently fabricates a uniform map when
`last_attn_weights` is `None`, so the no-flag sub-check fires (its message is
about *naming that missing* attribute and flag). Your job is to write the version
that *raises* in both situations.

## The hint ladder

If the grader's message is not enough, open the hints. Each function has three
levels: a conceptual nudge, then the invariant with shapes, then pseudocode —
`contract_read` walks you from "there is one access path" to the full
per-block loop, and `row_sum_check` from "rows are distributions" to the one-line
`np.allclose(attn.sum(axis=-1), 1.0, atol=1e-5)`. Try level 1 before climbing.

## Done when

`grade attention-viz` shows all-green: `row_sum_check` passes all five synthetic
cases (accepting the within-`1e-5` matrices, rejecting the off-by-`1e-4`,
all-zeros, and over-1 rows), and `attention_maps` returns 2 layers of shape
`(2, 5, 5)` with row sums of 1 and a hard-zero upper triangle, registers no
forward hook, and raises `AttentionContractError` both on the no-flag model and on
a matrix with a future-key cell. Then run the experiments and record your
predictions.
