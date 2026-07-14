# Correction Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject the most *relevant* past corrections (by TF-IDF similarity to the current utterance) into the cleanup prompt instead of the 5 most recent, and inject none when nothing is similar.

**Architecture:** A new pure-Python TF-IDF cosine retriever on `TrainingStore` ranks all stored corrections' `raw` text against the utterance being cleaned; `few_shot_block(query)` uses it; `cleanup._ollama_clean` passes the raw text as the query.

**Tech Stack:** Python 3 stdlib (`math`, `re`); no new dependencies.

## Global Constraints

- No new third-party dependencies — stdlib only.
- Tests run with `.venv/Scripts/python.exe -m pytest` (Windows, Git Bash).
- This IS a git repository on branch `dev`. Commit each task (targeted `git add` of only the files changed). Commit messages: NO `Co-Authored-By` trailer, NO AI-authorship attribution.
- Fully offline and deterministic — no network, no model calls in the retriever.
- Baseline before Task 1: full suite **134 passing**.

**Reference spec:** `docs/superpowers/specs/2026-07-14-correction-retrieval-design.md`

---

## Task 1: TF-IDF retriever (`relevant_corrections`)

**Files:**
- Modify: `app/training.py` (imports; add constant, helpers, `relevant_corrections`)
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `TrainingStore.relevant_corrections(query: str, n: int = 5) -> list[dict]` — corrections whose `raw` is most TF-IDF-similar to `query`, above `SIMILARITY_THRESHOLD`, best first, capped at `n`; `[]` if none qualify or inputs are empty.
- Module: `SIMILARITY_THRESHOLD = 0.1`.

This task is purely additive (a new method + helpers + tests); nothing existing changes, so the suite stays green.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training.py`:

```python
def test_relevant_corrections_ranks_shared_distinctive_term_first(tmp_path):
    store = make_store(tmp_path)
    store.record("schedule the meeting", "Schedule the meeting.", "bad", "Schedule the sync.")
    store.record("deploy the kubernetes cluster", "Deploy the kubernetes cluster.",
                 "bad", "Deploy the Kubernetes cluster.")
    store.record("send the email", "Send the email.", "bad", "Send the mail.")
    got = store.relevant_corrections("redeploy the kubernetes cluster now")
    assert got, "expected a match"
    assert got[0]["ideal"] == "Deploy the Kubernetes cluster."


def test_relevant_corrections_gate_returns_empty_when_nothing_similar(tmp_path):
    store = make_store(tmp_path)
    store.record("deploy the kubernetes cluster", "Deploy the kubernetes cluster.",
                 "bad", "Deploy the Kubernetes cluster.")
    assert store.relevant_corrections("what time is lunch tomorrow") == []


def test_relevant_corrections_surfaces_old_lesson_over_recent_noise(tmp_path):
    store = make_store(tmp_path)
    # one distinctive OLD correction, then 6 unrelated NEWER ones
    store.record("call ogiop about it", "Call Ogiop about it.", "bad", "Call Ogiop about it.")
    for i in range(6):
        store.record(f"unrelated thing number {i}", f"Unrelated thing number {i}.",
                     "bad", f"Totally different {i}.")
    got = store.relevant_corrections("please call ogiop again")
    assert any(e["ideal"] == "Call Ogiop about it." for e in got)


def test_relevant_corrections_empty_inputs(tmp_path):
    store = make_store(tmp_path)
    assert store.relevant_corrections("anything") == []      # no corrections
    store.record("call ogiop", "Call Ogiop.", "bad", "Call Ogiop.")
    assert store.relevant_corrections("") == []               # empty query
    assert store.relevant_corrections("   ") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -k relevant_corrections -v`
Expected: FAIL (`TrainingStore` has no attribute `relevant_corrections`).

- [ ] **Step 3: Implement in `app/training.py`**

Add `import math` to the imports block (with the other stdlib imports).

Add module-level constant + helpers after the existing module constants (near `_MAX_VOCAB_SPAN`):

```python
# Corrections below this TF-IDF cosine similarity to the current utterance are
# not relevant enough to inject as few-shot examples: a shared distinctive term
# (a name/jargon word) easily clears it, shared filler ("the", "and") does not.
# Tunable heuristic — raise to be stricter, lower to inject looser matches.
SIMILARITY_THRESHOLD = 0.1

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _rank_by_tfidf(query: str, docs: list[str]) -> list[tuple[int, float]]:
    """Rank `docs` by TF-IDF cosine similarity to `query`, highest first.
    IDF is computed over `docs`; the query is scored against that same IDF.
    Returns (doc_index, cosine) pairs; [] when query or docs are empty."""
    q_tokens = _tokens(query)
    doc_tokens = [_tokens(d) for d in docs]
    if not q_tokens or not doc_tokens:
        return []
    n = len(doc_tokens)
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        return math.log((1 + n) / (1 + df.get(t, 0))) + 1

    def vec(tokens: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {t: c * idf(t) for t, c in tf.items()}

    def norm(v: dict[str, float]) -> float:
        return math.sqrt(sum(w * w for w in v.values()))

    qv = vec(q_tokens)
    qn = norm(qv)
    scored: list[tuple[int, float]] = []
    for i, toks in enumerate(doc_tokens):
        dv = vec(toks)
        dn = norm(dv)
        if qn == 0 or dn == 0:
            cos = 0.0
        else:
            cos = sum(qv[t] * dv[t] for t in qv if t in dv) / (qn * dn)
        scored.append((i, cos))
    scored.sort(key=lambda x: -x[1])
    return scored
```

Add the method to `TrainingStore` (next to `corrections` / `few_shot_block`):

```python
    def relevant_corrections(self, query: str, n: int = 5) -> list[dict]:
        """Corrections whose raw text is most TF-IDF-similar to `query`, above
        SIMILARITY_THRESHOLD, best first, capped at `n`. Scores against ALL
        stored corrections (not just recent), so a relevant old lesson still
        surfaces; returns [] when nothing clears the bar or inputs are empty."""
        if not query or not query.strip():
            return []
        entries = self.corrections(n=None)
        if not entries:
            return []
        ranked = _rank_by_tfidf(query, [e.get("raw", "") for e in entries])
        chosen: list[tuple[dict, float]] = []
        for i, cos in ranked:
            if cos < SIMILARITY_THRESHOLD:
                break  # ranked is descending — the rest are lower too
            chosen.append((entries[i], cos))
        # tie-break equal-similarity matches toward the most recent
        chosen.sort(key=lambda ec: (-ec[1], -ec[0].get("ts", 0)))
        return [e for e, _ in chosen[:n]]
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py -v`
Expected: PASS (4 new tests plus all existing).

- [ ] **Step 5: Commit**

```bash
git add app/training.py tests/test_training.py
git commit -m "feat(training): TF-IDF relevant_corrections retriever"
```

---

## Task 2: Wire `few_shot_block` to the retriever

**Files:**
- Modify: `app/training.py` (`few_shot_block` signature)
- Modify: `app/cleanup.py` (`_ollama_clean` call)
- Test: `tests/test_training.py` (update `test_few_shot_block_appears_in_prompt`; add a negative)

**Interfaces:**
- Consumes: `relevant_corrections(query, n)`.
- Produces: `few_shot_block(query: str, n: int = 5) -> str` (query now required).

- [ ] **Step 1: Update the prompt tests**

In `tests/test_training.py`, replace `test_few_shot_block_appears_in_prompt` with the version whose query MATCHES the stored correction, and add a negative test:

```python
def test_few_shot_block_appears_in_prompt(tmp_path):
    store = make_store(tmp_path)
    store.record("go by 2pm no sorry 3pm", "Go by 2pm, no sorry, 3pm.",
                 "bad", "Go by 3pm.")
    cfg = CleanupConfig(enabled=True)
    cleaner = Cleaner(cfg, training=store)
    fake = MagicMock()
    fake.json.return_value = {"response": "Whatever."}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        # query shares distinctive terms with the stored correction -> retrieved
        cleaner.clean("go by 4pm no sorry 5pm")
    system = mock_post.call_args.kwargs["json"]["system"]
    assert "Go by 3pm." in system  # relevant correction injected as example


def test_unrelated_utterance_injects_no_correction(tmp_path):
    store = make_store(tmp_path)
    store.record("go by 2pm no sorry 3pm", "Go by 2pm, no sorry, 3pm.",
                 "bad", "Go by 3pm.")
    cfg = CleanupConfig(enabled=True)
    cleaner = Cleaner(cfg, training=store)
    fake = MagicMock()
    fake.json.return_value = {"response": "Whatever."}
    fake.raise_for_status.return_value = None
    with patch("app.cleanup.requests.post", return_value=fake) as mock_post:
        cleaner.clean("the weather is nice today and the sky is clear")
    system = mock_post.call_args.kwargs["json"]["system"]
    assert "Go by 3pm." not in system  # nothing similar -> no example injected
```

- [ ] **Step 2: Run tests to verify the negative fails (and the positive needs the wiring)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_training.py::test_unrelated_utterance_injects_no_correction tests/test_training.py::test_few_shot_block_appears_in_prompt -v`
Expected: FAIL — `few_shot_block()` currently takes no query and injects the recent correction regardless (negative fails; positive may error once the signature changes in Step 3).

- [ ] **Step 3: Change `few_shot_block` and the cleanup call**

In `app/training.py`, replace `few_shot_block`:

```python
    def few_shot_block(self, query: str, n: int = 5) -> str:
        """Correction examples RELEVANT to `query` (TF-IDF), formatted for the
        system prompt. '' when nothing clears the similarity bar."""
        pairs = self.relevant_corrections(query, n)
        if not pairs:
            return ""
        blocks = [
            f"Input: {e['raw']}\nOutput: {e['ideal']}" for e in pairs
        ]
        return (
            "\n\nThis user has corrected past outputs. Match these exactly "
            "when similar input appears:\n" + "\n\n".join(blocks)
        )
```

In `app/cleanup.py`, in `_ollama_clean`, change the few-shot line to pass the transcript being cleaned as the query:

```python
            system += self.training.few_shot_block(text)
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (the updated positive test, the new negative test, and everything else — including the other `Cleaner.clean` tests, whose stored corrections either match their own queries or are absent).

- [ ] **Step 5: Commit**

```bash
git add app/training.py app/cleanup.py tests/test_training.py
git commit -m "feat(cleanup): inject relevance-ranked corrections as few-shot"
```

---

## Task 3: Manual verification

**Files:** none (verification only).

- [ ] **Step 1: Seed a correction and confirm retrieval end-to-end**

Run a short Python check against a temp store to confirm the full path (retriever → prompt) with a realistic pair:

```bash
.venv/Scripts/python.exe - <<'PY'
from app.training import TrainingStore
from pathlib import Path
import tempfile
d = Path(tempfile.mkdtemp())
s = TrainingStore(data_path=d/"data.jsonl", vocab_path=d/"vocab.json")
s.record("meet with mr ogiop tomorrow", "Meet with Mr Ogiop tomorrow.",
         "bad", "Meet with Mr Ogiop tomorrow.")
for i in range(6):
    s.record(f"random note {i}", f"Random note {i}.", "bad", f"Different {i}.")
print("MATCH:", [e["ideal"] for e in s.relevant_corrections("call mr ogiop again")])
print("NO MATCH:", s.relevant_corrections("what is the weather like"))
print("BLOCK:\n", s.few_shot_block("call mr ogiop again"))
PY
```
Expected: `MATCH` contains "Meet with Mr Ogiop tomorrow." (the old, non-recent lesson surfaced past 6 newer ones); `NO MATCH` is `[]`; `BLOCK` contains the Ogiop example only.

- [ ] **Step 2: Full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

---

## Self-Review (completed by plan author)

- **Spec coverage:** TF-IDF cosine retriever with smoothed IDF + threshold (Task 1 `_rank_by_tfidf`/`relevant_corrections`, `SIMILARITY_THRESHOLD`); relevance-gated, returns [] when nothing matches (Task 1 + negative test in Task 2); `few_shot_block(query)` wiring and `cleanup._ollama_clean(text)` query (Task 2); flows through batch and streaming (both reach `_ollama_clean(text)`); `corrections(n)` unchanged (untouched); pure-Python/offline/deterministic (no deps, unit tests); edge cases empty query/corpus (Task 1). All spec sections mapped.
- **Type consistency:** `relevant_corrections(query, n=5) -> list[dict]`, `few_shot_block(query, n=5) -> str`, `_rank_by_tfidf(query, docs) -> list[tuple[int,float]]`, `_tokens(text) -> list[str]`, `SIMILARITY_THRESHOLD` — consistent across Tasks 1–2. `few_shot_block`'s only caller (`cleanup.py`) is updated in the same task the signature changes.
- **Placeholder scan:** none — every code and test step is complete.
