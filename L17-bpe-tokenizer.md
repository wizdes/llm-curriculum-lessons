# L15 — BPE Tokenizer

A language model never sees text. It sees integers. Every model so far in this
course took numbers in and gave numbers out — but the things we actually want a
language model to read and write are *strings*: "the cat sat on the mat", an
emoji, a line of French. Something has to stand between the string and the model
and translate. That something is the **tokenizer**.

This is the first lesson where the model's input is *language*, so we start by
defining the words that the rest of the lesson leans on.

- **Tokenization** is the act of splitting a string into a sequence of small
  units the model can look up by number. You cannot feed a model the letter `c`;
  you feed it the integer that *stands for* some chunk of text.
- A **token** is one of those units — a chunk of text (a byte, a character, a
  piece of a word) that the tokenizer treats as a single indivisible symbol.
- A **token id** is the integer assigned to a token. The model only ever sees
  these ids.
- The **vocabulary** is the full set of tokens the tokenizer knows, each paired
  with its id. "Vocab size" is just how many tokens are in that set.

The tokenizer turns a string into a list of token ids and back again, and the
choice it makes about *what counts as a token* shapes everything downstream: how
long your sequences get, how much the model can share between related words
(does it see `running` and `runner` as related, or as two unrelated blobs?), and
whether it can represent an emoji at all. This lesson builds **Byte Pair
Encoding** (BPE) from scratch — the tokenizer behind GPT-2 and most modern LLMs.

Before the new material, a 30-second warm-up re-derives the L4 numerically-stable
`sigmoid` cold (re-implementing it from memory, no peeking). `sigmoid(z) = 1 /
(1 + exp(-z))` squashes any real number into the range `(0, 1)`; the "stable"
version branches on the sign of `z` so the exponential never overflows. Recalling
it now is spaced retrieval — it keeps the earlier muscle warm while you learn
something unrelated.

## Why bytes, not characters

The very first decision a tokenizer makes is: what is the smallest unit I am
willing to break text into? Pick the wrong answer here and everything above it is
built on sand, so we make this one non-negotiable: **BPE here is byte-level**.

To see what that means concretely, here is the rule we will follow before any
merging happens. The input string is encoded to its raw **UTF-8 bytes**, and the
starting vocabulary is exactly the 256 possible byte values (ids `0` through
`255`). A *byte* is one of the 256 values `0..255` that computers store text as;
*UTF-8* is the standard recipe for turning any string into a sequence of those
bytes. In Python that conversion is one call:

```python
ids = list(text.encode("utf-8"))   # e.g. "hi" -> [104, 105]
```

`text.encode("utf-8")` produces the raw bytes; wrapping it in `list(...)` gives
you those byte values as a plain list of integers in `0..255`.

Why not just split on characters instead? Because there is no fixed-size
character vocabulary that covers every accent, script, and emoji you might
encounter — Unicode (the worldwide catalogue of characters) keeps growing and has
over a hundred thousand entries. Bytes are different. There are exactly 256 of
them, full stop, and *every* possible string is some sequence of bytes. So a
byte-level tokenizer can represent **any** input with **zero** unknown tokens —
there is no character it has never seen, because it never works at the level of
characters.

Here is the trap that makes this concrete. A single `é` is two bytes in UTF-8;
the pizza emoji `🍕` is four bytes. The character-level mistake is to split the
string into its **characters** instead — `list(text)` gives one Python string per
character, and the natural fix to turn those into numbers, `[ord(c) for c in
text]`, gives one **codepoint** per character (`ord` returns a character's Unicode
number; a *codepoint* is that abstract "letter" number, before UTF-8 turns it into
bytes). For `🍕` that single integer is `127829`, far outside `0..255`. Your
vocabulary is then not 256 things; it is "however many distinct characters
happened to appear in the training text", and it silently breaks the moment a
character you never trained on shows up. Byte-level by construction sidesteps all
of it. This is the whole foundation, so it is the first thing the grader checks.

## The merge loop: learning a vocabulary

Starting from bytes, every string is a sequence of ids in `0..255`. That works,
but it is wasteful: the word `the` is always three separate tokens `t`, `h`, `e`,
even though it appears constantly. BPE's one idea is to fix this greedily —
**repeatedly find the most frequent adjacent pair of tokens and replace it with a
single new token.**

The two key words:

- A **pair** is two tokens sitting next to each other in the sequence, like `(t,
  h)`.
- A **merge** is the act of minting a brand-new token id for a pair and replacing
  every occurrence of that pair with the new id. The first new id is `256`, the
  next `257`, and so on — they pick up right where the byte ids `0..255` leave
  off.

### A worked example, by hand

Let's trace it on a tiny corpus. Take the string `ababab cdcd ababab`. Its bytes
are (using `a`=97, `b`=98, `c`=99, `d`=100, space=32):

```text
97 98 97 98 97 98 32 99 100 99 100 32 97 98 97 98 97 98
```

**Round 1.** Count every adjacent pair. The pair `(97, 98)` — that is `ab` —
appears 6 times, more than any other. So we mint id `256` for `ab` and replace
all six occurrences. The sequence becomes (writing `256` for each `ab`):

```text
256 256 256 32 99 100 99 100 32 256 256 256
```

**Round 2.** Re-count over the *new* sequence. Now `(256, 256)` — two `ab`s in a
row, i.e. `abab` — appears 4 times. Mint id `257` for it and replace:

```text
257 256 32 99 100 99 100 32 257 256
```

**Round 3.** Re-count again. Now there is a tie: `(99, 100)` — that is `cd` —
appears twice, and so does `(257, 256)` — `abab` followed by `ab`. When two pairs
are equally frequent we need a fixed rule so training is *deterministic* (same
input always gives the same merges); we break the tie toward the **smaller pair
tuple**, which here is `(99, 100)`. Mint id `258` for `cd`:

```text
257 256 32 258 258 32 257 256
```

**Round 4.** The remaining repeated pair is `(257, 256)` — `abab` followed by
`ab`, i.e. `ababab` — appearing twice. Mint id `259`:

```text
259 32 258 258 32 259
```

Four merges, in this exact order: `ab`→256, `abab`→257, `cd`→258, `ababab`→259.
This is the precise sequence the grader's merge-order test pins down. Notice that
each merge re-counts over the result of the previous one, and that a merge can
feed the next: minting `ab` (256) is what *made* the pair `(256, 256)` available
for round 2.

### The loop as steps

Here is the procedure as an explicit recipe. Given a `vocab_size` (the target
number of tokens), do `vocab_size - 256` rounds — one new id per round, starting
at 256:

1. Encode the text to bytes: `ids = list(text.encode("utf-8"))`.
2. For each `new_id` from `256` up to `vocab_size - 1`:
   1. Count every adjacent pair `(a, b)` in the current `ids`.
   2. Pick the **most frequent** pair, breaking ties toward the smaller pair
      tuple so the result is deterministic. (Stop early if no pair repeats — there
      is nothing left to compress.)
   3. Replace every occurrence of that pair in `ids` with `new_id`.
   4. Record the merge: `merges[(a, b)] = new_id`.
3. Return `merges`.

The result, `merges`, is an **insertion-ordered** dict: in Python, a dict
remembers the order you inserted keys, so the first pair you record is "rank 0"
(id 256), the next "rank 1" (id 257), and so on. We call this position the merge's
**rank** — its place in the learning order. This rank is not bookkeeping you can
shuffle; it is the contract that encoding depends on, which is the next section.

## Encoding is not training

Here is the headline trap of the whole lesson. Training learned a set of merge
**rules** from one corpus. Now you hand the tokenizer a *brand-new* string and
ask it to tokenize. The wrong instinct is to run the training loop again on the
new string. **Encoding does not re-count frequencies.** It re-applies the
already-learned rules, and it applies them in **rank order**.

The encode procedure as steps:

1. Start from bytes: `ids = list(text.encode("utf-8"))`.
2. Repeat:
   1. Look at every adjacent pair currently present in `ids`.
   2. Among the pairs that are *learned merge rules*, take the one with the
      **lowest rank** (earliest learned — smallest id).
   3. Apply that merge: replace its occurrences with the rule's new id.
   4. Stop when no present pair is a learned rule.
3. Return `ids`.

Why rank order and not "most frequent right now"? Because the merges were learned
in a sequence where each one could *enable* the next. A low-rank merge often
creates the very token a higher-rank merge consumes. Applying them out of order
can pre-empt a merge that should have happened, producing a *different*
tokenization than the one the model was trained on — and the model only
understands the tokenization it was trained on.

### Seeing rank order and frequency order diverge

This is subtle, so here is a concrete case where the two orders give different
answers. Suppose training learned these two rules (among others), with these
ranks:

- `bc` → id `256` (rank 0, learned first)
- `ab` → id `260` (rank 4, learned later)

Now encode the string `daabaabc`. Its bytes are `d a a b a a b c` =
`[100, 97, 97, 98, 97, 97, 98, 99]`. Two learned pairs are present: `ab` (in two
places) and `bc` (once, at the very end).

- **Rank order (correct).** The lowest-rank present rule is `bc` (id 256), so it
  goes first — even though `ab` is more frequent. Applying it consumes the final
  `b` and `c`:
  `[100, 97, 97, 98, 97, 97, 256]`. Now the only present rule is `ab` (id 260);
  apply it: `[100, 97, 260, 97, 97, 256]`. **Final: `[100, 97, 260, 97, 97,
  256]`.**
- **Frequency order (wrong).** `ab` appears twice, `bc` once, so a
  frequency-greedy encoder merges `ab` first — in *both* spots, including the one
  right before `c`. That destroys the `bc` pair, and the result is
  `[100, 97, 260, 97, 260, 99]`. **Different, and wrong.**

That single end-of-string `bc` is exactly the kind of low-rank, low-frequency
merge that rank order protects and frequency order tramples. Encoding replays the
training history *in order*; it never re-derives it. Mixing these up is the most
common BPE bug, and the grader checks for it head-on.

## Decoding and exact round-trips

**Decoding** turns a list of token ids back into the original string. It works by
rebuilding the byte-meaning of every token. Ids `0..255` each mean their single
byte; each merged id means the concatenation of its two parts' bytes. Because
`merges` is in rank order, both parts of any merge are *always already defined*
before the merge that uses them (you cannot have learned `ababab` before you
learned `abab`).

The decode steps:

1. Build the base vocab: id `i` → the single byte `bytes([i])`, for `i` in
   `0..255`.
2. Walk `merges` in rank order; for each `(a, b) → new_id`, set `vocab[new_id] =
   vocab[a] + vocab[b]` (the bytes of `a` followed by the bytes of `b`).
3. Look up each id in `ids`, join all the bytes, and `.decode("utf-8")` back to a
   string.

```python
vocab = {i: bytes([i]) for i in range(256)}
for (a, b), new_id in merges.items():        # rank order
    vocab[new_id] = vocab[a] + vocab[b]
return b"".join(vocab[i] for i in ids).decode("utf-8")
```

Read aloud: "start the vocab with one entry per byte; then, in the order they
were learned, define each merged token as its left part's bytes followed by its
right part's bytes; finally glue every token's bytes together and read the result
back as UTF-8 text."

Because everything is byte-level, the round-trip is **exact**: `decode(encode(t))
== t` for *any* input — plain ASCII, accented Latin, a string of emoji. There are
no unknown characters to lose, because there were never any characters, only
bytes. That exact round-trip is the correctness bar this lesson holds you to.

## Dataset

There is no dataset to download. The grader trains and encodes over small fixed
strings written directly into the test: a repetitive English sentence for the
compression check, a tiny hand-traceable string (`ababab cdcd ababab`, the one we
traced above) for the merge-order check, and multibyte and emoji strings for the
round-trip. Everything is deterministic and runs in well under a second, so you
can iterate fast.

## Exercise

Implement the warm-up and the three BPE functions in
`my_work/L17-bpe-tokenizer/bpe_tokenizer.py` (run `uv run grade start L17-bpe-tokenizer` to
create that file from the starter). Self-contained: numpy only — no torch, no
cross-lesson imports, no dataset reads.

- `sigmoid(z)` — the **warm-up**, the L4 numerically-stable sigmoid recalled
  cold. Parameter: `z` is a numpy array (or anything array-like) of real numbers.
  It returns an array the same shape as `z`, where each entry is
  `1 / (1 + exp(-z))` — a number in `(0, 1)`. Branch on the sign of each element
  so `exp()` only ever sees a non-positive argument; then `sigmoid(1000) → 1.0`
  and `sigmoid(-1000) → 0.0` with no overflow or NaN. It is graded on its own and
  not used by the BPE functions; re-authoring it from memory is the spaced
  retrieval.
- `train_bpe(text, vocab_size)` — learn the merges. Parameters: `text` is the
  training string; `vocab_size` is the target vocabulary size, an integer that
  must be `>= 256` (raise `ValueError` if it is smaller, since 256 is the byte
  base). It returns the `merges` dict: an insertion-ordered
  `dict[tuple[int, int], int]` mapping each merged byte pair `(a, b)` to its new
  id. Learn `vocab_size - 256` merges (fewer if no pair repeats). Start from
  `list(text.encode("utf-8"))`, **never** `list(text)`.
- `encode(text, merges)` — tokenize a string. Parameters: `text` is the string to
  encode; `merges` is the dict returned by `train_bpe`. It returns a `list[int]`
  of token ids. Start from the UTF-8 bytes, then repeatedly apply the present
  learned pair with the **lowest** merge id (earliest learned), stopping when no
  learned pair is present. Do **not** pick the most-frequent pair — that is
  re-running training.
- `decode(ids, merges)` — invert encoding. Parameters: `ids` is the `list[int]`
  of token ids; `merges` is the same dict. It returns the original string. Build
  the byte vocab from `merges` in rank order, map each id to its bytes, join, and
  `.decode("utf-8")`.

Then grade it: the site's Run Grader button or `uv run grade L17-bpe-tokenizer`.

## How the grader checks you

Six checks, all over fixed inline strings, all in well under a second.

- The **warm-up** `sigmoid` is an independent unit: it is run on
  `[-1000, -2, 0, 2, 1000]` and compared to a stable reference to a relative
  tolerance of `1e-9`. It is graded on its own and never blocks the BPE checks; a
  wrong warm-up does not hide working BPE code.
- **Round-trip** trains on a repetitive English corpus to `vocab_size=400`, then
  checks `decode(encode(t)) == t` on three strings: pure ASCII, multibyte Latin
  (`café — naïve résumé — Größe`), and emoji (`I love 🍕 and 🚀 — 👩‍💻 codes!`). If
  any fails, the message names the exact byte offset where the original and the
  round-tripped string first differ — the signature of a tokenizer that
  initialized from characters instead of bytes.
- **Base-vocab-is-bytes** trains on a repetitive *multibyte* corpus and requires
  the encoding to compress below 70% of the byte length. A character-initialized
  tokenizer learns merges keyed on codepoints that the byte-level encoder can
  never apply, so it fails to compress — that is what this check catches.
- **Rank order** trains on a fixed corpus, encodes a fixed string, and compares
  your output against an independent rank-order reference. A frequency-order
  shortcut produces a different result here and fails.
- **Merge-order trace** trains on `ababab cdcd ababab` to `vocab_size=260` and
  pins the exact learned sequence — `(97,98)→256`, `(256,256)→257`,
  `(99,100)→258`, `(257,256)→259` — naming the first rank where your merges
  diverge. (This is the by-hand trace from earlier in the lesson.)
- **Compression** trains a repetitive corpus to `vocab_size=512` and requires the
  encoding to shrink below 70% of its byte length. An encoder that barely shrinks
  the byte stream is not applying the learned merges.

Each failure message names what it found, what it expected, and the concept to
revisit.

### The named trap: encoding by frequency, not rank

The grader hunts two named mistakes. The first, `bpe_byte_vs_char`, is
initializing the vocab from `list(text)` (str characters) instead of
`list(text.encode("utf-8"))` (bytes); it surfaces as a broken round-trip and a
corpus that will not compress, with a message that says the tokenizer
"initializes vocab from str characters (not bytes)."

The second, `bpe_merge_order`, is the headline one: writing `encode` to apply the
**most frequent** present pair instead of the **lowest-rank** one. The wrong code
looks like picking `pair = max(applicable, key=...frequency...)` inside the encode
loop — it re-runs training instead of replaying the learned rules. As the
`daabaabc` example above showed, that produces a different, wrong tokenization on
strings where a low-rank merge should have fired before a more-frequent one. The
grader catches this exact failure and tells you: "Encoding is NOT re-running
training. Apply merge rules in rank order."

## The hint ladder

If a grader message is not enough, open the hints. Each topic — byte-level base
vocab, encode rank order, and merge insertion order — has three levels: a
conceptual nudge (what the symptom means), then the invariant with shapes (the
rule the code must satisfy), then pseudocode. Try level 1 before climbing.

## Done when

`uv run grade L17-bpe-tokenizer` shows all-green: the `sigmoid` warm-up matches the stable
reference at the ±1000 extremes; every round-trip case (ASCII, multibyte, emoji)
reproduces its input exactly; both multibyte and repetitive corpora compress
below 70% of their byte length; `encode` matches the rank-order reference; and
`train_bpe` reproduces the exact merge sequence on `ababab cdcd ababab`. Then run
the experiments and record your predictions.
