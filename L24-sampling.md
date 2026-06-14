# L22 — Sampling from a Language Model

In L21 you trained a GPT and watched the loss fall. But a trained model does not
*say* anything on its own. Calling `model(idx)` — handing the model a sequence of
token ids and asking what comes next — returns **logits**: one raw, unnormalized
score for every possible next token. (A **token** is one unit of text the model
reads and writes — for this course, think "one character or one small chunk"; the
**vocabulary** is the fixed list of all possible tokens, and a token **id** is
that token's index into the list. A **logit** is a score before it has been turned
into a probability — higher means the model likes that token more, but the numbers
do not yet add up to 1.) Run those logits through **softmax** — the function that
exponentiates each score and divides by the total, turning a list of raw scores
into a probability distribution that sums to 1 — and you have a probability for
every token in the vocabulary.

That distribution is what the model gives you. **Sampling** is the separate
decision of how to turn it, step after step, into actual text: do you always take
the single most likely token, or roll the dice in proportion to the
probabilities, and do you reshape the distribution first? This lesson is those
decisions and the knobs that control them. The model is fixed; only the sampling
strategy changes.

We open with a 30-second warm-up — the bigram language model from L16 and its
`generate` loop, the simplest autoregressive sampler there is. Then we add four
controls that reshape the distribution before each draw: temperature, top-k,
top-p, and the greedy special case.

## Autoregressive generation: feed the model its own output

Generation is **autoregressive**: the model produces one token, you append it to
the sequence, and you feed the now-longer sequence back in to get the next token.
"Auto-regressive" literally means "regressing on itself" — each new prediction
conditions on all the tokens generated so far, including the ones the model just
wrote. The model never sees the whole sentence at once; it builds it one token at
a time, always predicting only the *next* token.

That gives us a loop. Hold a running sequence `idx` — start it as the **prompt**
(the seed text you give the model) and grow it by one token each step. The bigram
warm-up below is the bare-bones version of this loop; the `sample` function later
is the same loop with the four reshaping controls inserted before the draw.

Here is the loop as numbered steps. Each step takes the current sequence `idx`,
shaped `(B, T)` — `B` is the **batch size** (how many independent sequences you
generate in parallel; `B = 1` for a single one) and `T` is the current length —
and produces one more token:

1. **Crop the context.** A GPT can only attend to the last `block_size` tokens
   (the fixed context window it was built with), so feed it only the last
   `block_size` columns: `idx[:, -block_size:]`. (The slice `[:, -block_size:]`
   reads "all rows, last `block_size` columns.") Skip this and a prompt longer
   than `block_size` overflows the model's positional embeddings and it errors
   out.
2. **Run the model and take the last position's logits.**
   `logits = model(idx_cropped)[:, -1, :]`. The model returns logits at *every*
   position, shaped `(B, T, vocab_size)`, but we only ever want the token that
   comes *next*, which is predicted at the final position — so `[:, -1, :]` reads
   "all rows, the last time-step, all vocab scores," leaving `(B, vocab_size)`.
3. **Turn the logits into a token.** Softmax the logits into a probability
   distribution, then either take the most likely token or draw one at random in
   proportion to the probabilities (the two strategies below). The reshaping
   controls — temperature, top-k, top-p — all act here, on the logits, before the
   draw.
4. **Append and repeat.** Concatenate the new token onto `idx` and loop back to
   step 1 with the longer sequence.

### Greedy vs. sampling: two ways to pick the next token

Once step 2 hands you a distribution over the vocabulary, there are two basic ways
to choose a token from it.

- **Greedy** (also called **argmax** decoding) takes the single highest-probability
  token every time. "Argmax" reads "the argument that maximizes" — the *index* of
  the largest score, i.e. which token is on top, not the score itself. Greedy is
  **deterministic**: the same input always yields the same output, because there
  is no randomness — you just pick the winner.
- **Sampling** draws a token *at random*, but weighted by the probabilities: a
  token with probability 0.4 is drawn 40% of the time, one with probability 0.01
  about 1% of the time. This is what `torch.multinomial` does — given a
  probability distribution, it returns a random token id with exactly those odds.
  Sampling is **stochastic** (random): re-run it and you get different text.

Greedy gives you the model's single best guess and nothing else; sampling gives
you variety. The warm-up below uses plain sampling. The rest of the lesson is
about reshaping the distribution *before* you sample so the variety stays sensible.

### The warm-up: the bigram `generate` loop

The bigram model from L16 is the smallest language model there is: a single lookup
table, `nn.Embedding(vocab_size, vocab_size)`, whose row `i` holds the next-token
logits given that the current token is `i`. Its `generate` method is the loop
above with the plainest possible step 3 — softmax the last-step logits and sample,
no reshaping:

```python
def generate(self, idx, max_new_tokens, seed):
    torch.manual_seed(seed)
    out = idx
    for _ in range(max_new_tokens):
        logits = self(out)[:, -1, :]               # (B, vocab) last-step logits
        probs = F.softmax(logits, dim=-1)          # (B, vocab) distribution
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1) one draw
        out = torch.cat([out, next_id], dim=1)     # append, grow the sequence
    return out
```

`torch.manual_seed(seed)` fixes the random number generator at entry so that the
same `seed` reproduces the same sequence — handy for grading and debugging.
`max_new_tokens` is how many tokens to generate; the function returns the prompt
with those new tokens tacked on. Re-deriving this loop from memory anchors the
autoregressive idea before we complicate it.

## Temperature as logit scaling

The first reshaping knob is **temperature**, written `T`. Before the softmax,
divide every logit by `T`:

$$
\text{probs} = \text{softmax}\!\left(\frac{\text{logits}}{T}\right)
$$

Read aloud: *take each logit, divide it by the temperature, then softmax the
result into a probability distribution.*

**Why does dividing the logits reshape the distribution?** Softmax cares about the
*gaps* between logits, not their absolute size — it exponentiates each logit, and
the ratio between two probabilities depends only on the difference of their
logits. Dividing every logit by `T` scales every gap by the same factor `1/T`, and
shrinking or stretching the gaps is exactly what changes how peaked the
distribution is:

- **`T < 1` sharpens.** Dividing by a number below 1 (say 0.5) *multiplies* every
  gap by `1/T` (here, 2) — the gaps grow, so after softmax even more probability
  piles onto the top few tokens. The model becomes more confident and more
  repetitive.
- **`T > 1` flattens.** Dividing by a number above 1 (say 2) shrinks every gap, so
  the probabilities spread out toward **uniform** (every token equally likely).
  Rare tokens get a real chance — the output is more surprising and more
  error-prone.
- **`T = 1` is unchanged.** Dividing by 1 leaves the logits — and so the model's
  own distribution — exactly as they were.

A worked feel for it: suppose two tokens have logits 2 and 0, a gap of 2. At
`T = 1` the gap stays 2. At `T = 0.5` the logits become 4 and 0, a gap of 4 — the
top token's lead doubled, so it wins more often. At `T = 2` they become 1 and 0, a
gap of 1 — the lead halved, so the runner-up catches up. Same model, same logits;
only the temperature reshaped the odds.

Intuitively, temperature is a **confidence dial**: low temperature trusts the
model's top guesses, high temperature deliberately spreads the bet. You can
*measure* the effect with **entropy** — a single number that scores how spread-out
a distribution is (low entropy = concentrated on a few tokens, high entropy =
spread evenly). Lower temperature yields lower entropy, and that is exactly the
invariant the grader checks.

**`T = 0` is a special case you must handle separately.** The formula
`logits / 0` is a divide-by-zero — it produces `inf`/`NaN` and corrupts every
token from then on. So temperature 0 is implemented as an explicit **greedy
branch**: `next_id = argmax(logits)`, with no division at all. This is not a hack —
it is the exact limit. As `T -> 0` (temperature shrinking toward zero), the gaps
blow up without bound, so softmax puts essentially *all* the mass on the single
top token, which is just greedy. Writing greedy as its own branch makes that limit
exact and deterministic instead of a numerical accident. (Quizzes test this `T`
behavior directly, so make sure the three cases — sharpen, flatten, the `T -> 0`
greedy limit — are clear.)

## Greedy decoding and its repetition pathology

Greedy decoding (`temperature = 0`) always takes the single most likely next
token. It is deterministic and reproducible — the same prompt always yields the
same output — which is why it is the right choice when you want one fixed,
canonical answer.

But greedy has a notorious failure mode: **repetition loops**. Once the model
slightly favors a phrase, always taking the top token marches it straight back
into that phrase, and it gets stuck cycling — *"I think that I think that I think
that..."*. Greedy never explores the runner-up that might break the loop, because
by definition it only ever picks the winner. Sampling with some temperature, or a
top-k / top-p restriction, injects exactly the variation that escapes the cycle.
This is why interactive generation almost never uses pure greedy.

## Top-k: keep only the k best candidates

Even at `T = 1`, the softmax tail assigns a small but *nonzero* probability to
many implausible tokens. Sample long enough and you will eventually draw one,
derailing the text. **Top-k** truncates that tail before you sample.

The procedure, applied to one step's logits:

1. Find the `k` largest logits (the `k` most likely tokens).
2. Set every *other* logit to `-inf` (negative infinity). After softmax those
   become probability 0 — `exp(-inf) = 0` — so they can never be drawn.
3. Softmax the survivors and draw with `torch.multinomial`.

`k = 1` is exactly greedy (keep only the single best). Larger `k` admits more
diversity. The catch: `k` is a *fixed* count, blind to how peaked the distribution
is. When the model is very confident, `k = 50` still drags in 49 near-zero tokens;
when the model is genuinely uncertain, a small `k` may chop off good candidates it
should have kept. Top-p, next, fixes precisely that blind spot.

## Top-p (nucleus): keep the smallest set covering p of the mass

**Top-p**, also called **nucleus sampling**, truncates on *probability mass*
instead of a fixed count. The "nucleus" is the smallest group of top tokens that
together hold a probability of at least `p`.

The procedure, applied to one step's logits:

1. Softmax the logits to probabilities, and sort the tokens by probability
   **descending** (most likely first).
2. Walk down the sorted list, accumulating the probabilities, and keep tokens
   until the running total first reaches `p` — that kept prefix is the nucleus.
   Always keep at least the top-1 token (so even a single token over `p` on its
   own still gets sampled, never an empty set).
3. Set every token *outside* the nucleus to `-inf`, softmax the survivors, and
   draw.

So when the model is confident, the nucleus is just a token or two; when the model
is unsure, the nucleus widens to admit more candidates — the size *adapts to the
distribution*, which is exactly what fixed-count top-k cannot do. The kept set
always carries at least `p` of the mass — that is the construction invariant the
grader checks. `top_p = 0.9` is a common default.

A worked feel for it at `p = 0.9`: if the sorted probabilities are
`[0.6, 0.25, 0.1, 0.03, 0.02]`, walk the cumulative sum `0.6, 0.85, 0.95, ...` —
it first reaches 0.9 at the *third* token (cumulative 0.95), so the nucleus is the
top 3 tokens and the last two are dropped. If instead the top token alone had
probability 0.95, the nucleus would be just that one token.

## Composing the controls

The four controls stack in a fixed order, applied to each step's logits inside the
generation loop:

1. **temperature** — divide the logits by `T` (or take the greedy branch if
   `T = 0`, with no division);
2. **top-k** — if `top_k` is given, `-inf` all but the `k` largest logits;
3. **top-p** — if `top_p` is given, `-inf` all but the nucleus;
4. **softmax + `torch.multinomial`** — turn the surviving logits into
   probabilities and draw one token.

The signature that runs this each step is:

```python
sample(model, prompt_ids, max_new_tokens, *, temperature=1.0,
       top_k=None, top_p=None, seed=None)
```

Pass `seed` to make a run reproducible — it seeds the global random number
generator that the multinomial draw uses, so the same seed gives the same
sequence. A typical "creative but coherent" setting is
`temperature=0.8, top_k=None, top_p=0.95`: warm enough to be varied, with the
nucleus trimming off the nonsense tail.

## Exercise

Implement in `my_work/L24-sampling/sampling.py` (run `uv run grade start L24-sampling` to get the
scaffold). The only import you need is the canonical model,
`from references.gpt import GPT, GPTConfig`.

**`BigramLM`** — the warm-up, recalled cold from L16. A `nn.Module` whose
`__init__(self, vocab_size)` builds one lookup table, `nn.Embedding(vocab_size,
vocab_size)`, and whose two methods are:

- `forward(self, idx)` — `idx` is a `(B, T)` LongTensor of token ids
  (`B` = batch size, `T` = sequence length); returns the lookup-table logits,
  shape `(B, T, vocab_size)`.
- `generate(self, idx, max_new_tokens, seed)` — the simplest autoregressive
  sampler. `idx` is a `(B, T0)` LongTensor prompt; `max_new_tokens` is how many
  tokens to append; `seed` is an int passed to `torch.manual_seed` at entry for
  reproducibility. Each step: softmax the last-position logits, draw with
  `torch.multinomial`, append. Returns the full `(B, T0 + max_new_tokens)`
  LongTensor.

**`sample(model, prompt_ids, max_new_tokens, *, temperature=1.0, top_k=None,
top_p=None, seed=None)`** — the full sampler. Parameters:

- `model` — any model with a `.block_size` attribute and a `forward` returning
  logits `(B, T, vocab_size)` (the `GPT` from `references.gpt` qualifies). Passed
  in, so `sample` is model-agnostic.
- `prompt_ids` — a `(B, T0)` LongTensor, the starting sequence (`B = 1` is fine).
- `max_new_tokens` — int, how many tokens to generate.
- `temperature` — float. `1.0` leaves the distribution unchanged; `< 1` sharpens;
  `> 1` flattens; **exactly `0`** takes the greedy `argmax` branch with no
  division.
- `top_k` — int or `None`. If set, keep only the `k` largest logits each step
  (`-inf` the rest) before sampling.
- `top_p` — float in `(0, 1]` or `None`. If set, keep only the nucleus — the
  smallest set of top tokens whose cumulative softmax probability reaches `top_p`
  (always at least one token) — before sampling.
- `seed` — int or `None`. If not `None`, call `torch.manual_seed(seed)` at entry.

Returns the full `(B, T0 + max_new_tokens)` LongTensor (prompt + generated). Each
step composes the controls in order — temperature, top-k, top-p, then
softmax + multinomial — exactly as listed above. Wrap the function in
`@torch.no_grad()`: generation does no training, so there is no need to track
gradients.

Then grade it: the site's Run Grader button or `uv run grade L24-sampling`.

## How the grader checks you

Everything runs on CPU, on a tiny fixed `GPTConfig`, with fixed seeds — so it is
deterministic and seconds-fast. The grader builds its *own* fresh, seeded **but
untrained** `GPT` and tests the sampling math directly: sampling mechanics
(temperature scaling, top-k, top-p, greedy) are model-agnostic, so they need no
trained checkpoint and no import from your other lessons. An untrained model emits
gibberish tokens — fine, because the checks are about the *distribution* you draw
from, not the text quality.

- **Warm-up shape.** `BigramLM.generate` with `max_new_tokens=10` from a length-1
  prompt must return a `(1, 11)` LongTensor of ids all in `[0, vocab_size)` — the
  loop appended the right count of valid ids. The warm-up is an independent root:
  a wrong warm-up does not hide a working `sample`.
- **Temperature lowers entropy.** Over many fixed-seed trials, the empirical
  entropy of the tokens generated at `temperature=0.5` must be **below** that at
  `temperature=1.5`. Lower temperature sharpens the distribution, so its draws
  concentrate on fewer tokens.
- **Top-k confines the draws.** With `top_k=3`, *every* sampled first token (over
  100 trials with varied seeds) must be one of the top-3 first-step logits read
  straight off the model — because the other tokens were `-inf`'d to probability 0.
- **Top-p keeps a `>= p` nucleus.** With `top_p=0.9`, the kept nucleus must carry
  at least 0.9 of the softmax mass at each step (the construction invariant), and
  every `top_p`-sampled first token must fall inside that nucleus over 100 trials —
  this catches the off-by-one that admits one token just past the nucleus boundary.
- **`temperature=0` is deterministic argmax.** Two `temperature=0` calls on
  identical inputs must return *identical* sequences, and the first generated token
  must equal `argmax` of the first-step logits — the greedy branch, not a sampled
  draw.
- **The greedy branch never divides by zero.** A `temperature=0` run must return
  valid finite ids in range — proving there is an explicit argmax branch, not a
  `logits / 0` that would emit `inf`/`NaN`.
- **All four controls compose.** One call with
  `temperature=0.8, top_k=10, top_p=0.95` must return the right shape `(1, 9)` with
  every id in range — proving the controls stack in one pass.

This lesson has **no named trap**: its content is a *composition* of sampling
controls rather than a single-line bug, and the one sharp pitfall — dividing by a
zero temperature — is pinned directly by the greedy-no-divide-by-zero check above.

## The hint ladder

If the grader's message is not enough, open the hints. Each function —
`bigram_generate` and `sample` — has three levels: a conceptual nudge (which
invariant is off), then the key formulas with shapes (the crop, the greedy branch,
the top-k mask, the top-p nucleus), then full pseudocode for the loop. Try level 1
before climbing.

## Done when

`uv run grade L24-sampling` shows all-green: the bigram warm-up returns a valid `(1, 11)`
sequence; `temperature=0.5` yields lower token entropy than `temperature=1.5`;
`top_k=3` confines every sampled first token to the top-3 logits; the `top_p=0.9`
nucleus carries at least 0.9 of the mass and confines the draws; `temperature=0`
is deterministic and equals the first-step argmax with no divide-by-zero; and all
four controls compose in one call. Then run the experiments and record your
predictions.
