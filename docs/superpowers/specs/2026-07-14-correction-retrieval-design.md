# Relevance-ranked correction retrieval

**Date:** 2026-07-14
**Status:** Approved design (pending spec review), then implementation plan

## Problem

Few-shot learning today uses only the **5 most recent** corrections
([`few_shot_block(n=5)`](../../../app/training.py) → `corrections(n=5)`), injected
into the cleanup prompt as "match these when similar input appears." Two flaws:

- **Old lessons are forgotten** — anything past the last 5 never influences cleanup.
- **Irrelevance** — the 5 most recent may have nothing to do with the current
  utterance, so a small model is fed misleading examples.

## Goal

Inject the most **relevant** corrections drawn from *all* stored history (not the
most recent), and inject **none** when nothing is genuinely similar — so a
relevant old lesson resurfaces while irrelevant ones stop being forced in.

## Design

**Retriever (new `TrainingStore.relevant_corrections`)** ranks every stored
correction by **TF-IDF cosine similarity** of its `raw` text against the current
utterance's `raw` (the query), and returns the top-N that clear a similarity
threshold (possibly zero). TF-IDF weighting makes a shared *distinctive* term (a
name/jargon word like "Ogiop") drive the match while shared filler ("the",
"and") barely counts — well matched to how dictation corrections recur.

**Algorithm (deterministic, pure-Python, offline):**
- Tokenize: `re.findall(r"[a-z0-9']+", text.lower())`.
- Corpus = the `raw` text of every correction (each is one document); `N` = count.
- Document frequency `df(t)` = number of documents containing term `t`.
- `idf(t) = log((1 + N) / (1 + df(t))) + 1` (smoothed; safe for df=0/df=N).
- Term weight in a vector = `tf(t) * idf(t)` where `tf(t)` is the term's count in
  that text.
- Score = cosine similarity between the query vector and each document vector
  (only terms present in both contribute to the dot product).
- Keep documents with `cosine >= SIMILARITY_THRESHOLD` (a documented module
  constant, default `0.1`), ranked by cosine descending, ties broken by recency
  (larger `ts` first) for stable ordering; return the correction dicts, best first,
  capped at `n`.
- Edge cases → `[]`: empty/whitespace query, no corrections, or nothing clears the
  threshold (e.g. an all-novel query shares no terms, so every cosine is 0).
- Corpus is tens–hundreds of entries and texts are short, so the vectors are
  rebuilt per call (sub-millisecond); no caching.

**Wiring:** `few_shot_block(query, n=5)` calls `relevant_corrections(query, n)`
instead of `corrections(n)`, with the same prompt formatting (`''` when empty). In
[`cleanup._ollama_clean(text, …)`](../../../app/cleanup.py) the call becomes
`few_shot_block(text)` — the raw text being cleaned is the query. This flows
through both paths: the batch path passes the whole utterance; streaming passes
each sentence chunk (cheap enough per chunk). `corrections(n)` is unchanged (the
Review dialog still lists all corrections).

**Behavior change:** when nothing clears the threshold, **no** few-shot examples
are injected (today it always injects up to 5). Punctuation differences don't
affect matching — tokenization drops punctuation — so retrieval is consistent
across the `model`/`pauses` punctuation modes.

## Components and interfaces

### `app/training.py`

- `import math` (new).
- `SIMILARITY_THRESHOLD = 0.1` (module constant, documented as tunable).
- `relevant_corrections(self, query: str, n: int = 5) -> list[dict]` — the
  retriever above. Operates over `corrections(n=None)` (all ideal-bearing entries).
- `few_shot_block(self, query: str, n: int = 5) -> str` — now takes a `query`;
  builds from `relevant_corrections(query, n)`; unchanged formatting; `''` if none.
- Optional small helpers kept private (e.g. a tokenizer and a TF-IDF/cosine
  function) so `relevant_corrections` stays readable and the math is unit-testable
  in isolation.

### `app/cleanup.py`

- `_ollama_clean`: `system += self.training.few_shot_block()` →
  `system += self.training.few_shot_block(text)`.

## Testing

- **Ranking**: three stored corrections, two unrelated and one sharing a rare
  term with the query, `relevant_corrections(query)` returns the sharing one first
  (and ranks it above the unrelated ones).
- **Relevance gate**: a query sharing no distinctive terms with any correction
  returns `[]`, and `few_shot_block(query)` returns `""` — even though corrections
  exist (proves it is not the old always-inject-recent behavior).
- **Old lesson resurfaces**: a correction that is NOT among the most recent is
  still retrieved when the query matches it (the core fix).
- **Prompt wiring**: with a matching query, `Cleaner.clean` puts that correction's
  ideal text into the system prompt; with a non-matching query, it does not (mock
  the Ollama call and assert on the `system` payload, mirroring the existing
  `test_few_shot_block_appears_in_prompt`).
- **Edge cases**: empty query, empty corpus, and stable ordering under ties.

## Out of scope

- Embedding-based semantic similarity (chosen against — TF-IDF fits the domain
  and stays dependency-free).
- Persisting audio / acoustic learning (separate roadmap item).
- Making the threshold or `max_examples` user-configurable — the threshold stays a
  code constant; the count stays the existing default of 5.

## Known tradeoffs

- TF-IDF matches on shared surface terms, not meaning — a paraphrase with no shared
  distinctive words won't retrieve its lesson. Acceptable: dictation corrections
  usually recur through shared terms, and a wrong match is worse than a miss.
- The threshold is a fixed heuristic; if it proves too strict/loose in real use it
  is a one-line constant change (or a later config key).
