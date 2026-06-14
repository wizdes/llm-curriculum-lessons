# Capstone: Write How an LLM Works

This is the final exam, and it has no code. Across the course you built every
piece of a language model from scratch — the tokenizer that turns text into
integers, the attention layer that lets each position read from context, the
transformer block that stacks attention and a small neural network with a
residual connection around each, and the training loop that fits it all to data.
Each of those lived in its own lesson, behind its own grader. The last thing to
prove is different: that you can hold the whole machine in your head at once and
explain it end to end, in your own words, without leaning on the lessons.

**Why a writeup is the capstone.** Code you can copy; a coherent mental model you
cannot. A grader checks one function against one set of inputs — it never asks
whether you see how the tokenizer feeds attention, how attention feeds the block,
how the block stacks into the model, and how the loss pulls all of it toward
predicting the next token. Writing forces those connections. The moment you try
to explain why next-token prediction trains a *general* language model, or why a
causal mask is what makes that training honest, you find out whether the parts are
really joined in your head or just sitting next to each other. That is the skill
this lesson grades.

## The assignment

Write how an LLM works, end to end, in your own words. No code, and no quotes
copied from the lessons — say it as you would explain it to someone. Aim for
**600–1200 words** (long enough to connect the parts, short enough that you have
to choose what matters).

Your writeup must cover the full path from text to a trained model:

1. **Next-token prediction** — why training a model to guess the next token, over
   and over on ordinary text, produces a model that has learned language. (What
   is the model predicting, what is it scored against, and why does "just predict
   the next token" end up teaching grammar, facts, and style?)
2. **Attention** — how each position reads from the context: what the queries,
   keys, and values are doing, and why a *causal* version is required for
   next-token training.
3. **The transformer block** — what one block contains (attention, the
   feed-forward network, the normalization), and why the **residual stream** —
   the running activation that every sublayer adds onto — is the spine the whole
   block is organized around.
4. **BPE tokenization** — how byte-pair encoding builds a vocabulary by merging
   frequent pairs, and how it then encodes new text by replaying those learned
   merges.
5. **The training loop** — how the loop is structured: the AdamW optimizer, the
   learning-rate schedule, gradient clipping, and checkpointing, and what each one
   is there to do.

Write the document a slightly-younger version of you would have wanted on day one
— the one that connects every part to every other part, instead of teaching each
in isolation. You are not summarizing five lessons; you are explaining one
machine.

## The correctness-gap review rubric

When you submit, Claude reviews your writeup against ONE threshold question:

> Could a reader reconstruct the architecture and training loop from this
> document alone?

"Reconstruct" is the bar: not "did you mention attention" but "could someone who
trusts your explanation rebuild the thing." That single question is made concrete
by six checks below. A **correctness gap** is a place where your explanation is
missing or wrong in a way that would stop that reconstruction. Claude hands each
gap back to you **as a question, not an answer** — it points at the hole so you
fill it yourself.

Each check below names a term and then says what the check is actually asking
for. Read these as "here is what a complete answer contains," not as the answer —
the words are yours to write.

1. **The causal masking purpose** — *causal masking* means: when computing
   attention, each position is blocked from looking at positions that come *after*
   it (the "future" tokens). The check wants the *purpose* stated, not just the
   mechanism. Why does it matter for next-token training specifically? Saying it
   "prevents looking ahead" is the mechanism; the purpose is what goes wrong
   without it. A complete answer connects the mask to the fact that the model is
   being asked to predict the next token and so must not be allowed to see it.
2. **QKV projection explained** — *QKV* stands for query, key, and value, the
   three vectors attention computes for each token. *Projection* means each is
   produced by multiplying the token's representation by a learned weight matrix
   (one matrix per role). The check wants these *explained*, not just named: what
   does the query do (it is what a position is looking for), what does a key do
   (what a position advertises about itself), what does a value carry (the content
   that gets passed on when a key is matched)? Naming "Q, K, V" is not enough; the
   reader needs to know what each is *for*.
3. **The additive bus** — describe the residual stream as an **additive bus**, not
   merely as "skip connections." *Bus* is a shared channel that many components
   read from and write to; *additive* means each sublayer writes by **adding** its
   output onto that running channel rather than replacing it. So the residual
   stream is a single running vector that attention and the feed-forward network
   each read from and add back onto. "Skip connection" names the wire; "additive
   bus" names what the wire is *for* — a shared running state every layer
   contributes to. The check wants that framing.
4. **The merge-order rule** — BPE learns its vocabulary by repeatedly merging the
   most frequent adjacent pair, and each merge is recorded with a **rank** (its
   position in the learning order: first merge learned = rank 0, next = rank 1,
   and so on). The *merge-order rule* is what happens at encode time: to tokenize
   new text you do **not** re-count frequencies on that text — you replay the
   learned merges in **rank order**, always applying the lowest-rank applicable
   merge first. The check wants you to state *why* encode is rank-driven and not
   frequency-driven (encode ≠ frequency-at-encode-time): the vocabulary is fixed
   after training, so encoding must reproduce the training-time merges
   deterministically, not invent new ones from the new text's statistics.
5. **Weight decay, motivated** — the check is `weight decay motivated`. AdamW is
   the optimizer; the **W** is its *weight decay*, a small pull on every weight
   toward zero on each step. The check wants it *motivated*, not just named: weight
   decay is regularization — it keeps the
   weights from growing without bound, which discourages the model from
   overfitting (memorizing the training data instead of learning patterns that
   generalize). A complete answer says *why* you'd want that pull, not just that
   AdamW applies it.
6. **A failure mode named** — the check is `one failure mode named`: name one
   specific thing that breaks training or inference if you get it wrong, and say
   what breaks. The
   examples in the lessons are concrete: forget the causal mask and the model can
   see the answer it is supposed to predict, so it "learns" nothing useful and
   collapses at generation time; drop gradient clipping and a single large
   gradient can blow the weights up (the loss spikes to NaN). The check wants one
   such failure called out by name, with its consequence.

The rubric checks for **correctness gaps only**. It does not grade prose quality,
cleverness, or length beyond the 600–1200 word window — a plain, correct
explanation passes; a polished one with a hole does not.

## The one-revision flow

The flow is deliberately short. Here is the procedure:

1. **Submit** — you send your writeup. Claude reads it against the six checks and
   hands back any correctness gaps it finds, phrased as questions. It never dumps
   the answers; the point is for you to close the gaps yourself.
2. **Revise once** — you address the gaps and resubmit. That is your single
   revision.
3. **Done** — the lesson is complete.

There is no further iteration: one review, one revision. The short loop is
deliberate. A capstone is not a tutoring session — the one revision is there to
let you fix a genuine hole you didn't see, not to grind toward a model answer one
hint at a time. If a gap surprises you, that surprise is the lesson; the revision
is where you turn it into understanding.

## Quiz-final mechanics

There is no separate quiz UI for this milestone — the writeup submission **is** the
final. Instead of a grader scoring code, the progress store records two flags:

- `submitted` — set when you send the first draft.
- `revised` — set when you send the one revision.

The lesson is marked complete on `revised`. Submitting alone does **not** complete
it; the revision does. This mirrors the flow above: a first draft proves you wrote
something, but the revision — closing the gaps the review surfaced — is what proves
you understood the machine well enough to fix your own explanation of it.
