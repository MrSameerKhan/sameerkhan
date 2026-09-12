# Module 24 — Long-Term Vector Memory
Status: `✅ Run (no API key)`

Theory: [../../../8.agents/05_agent_memory.md](../../../8.agents/05_agent_memory.md) §3 (episodic vs semantic) · §8 (retrieval quality) · §10 (the namespace leak)

---

## Use Case

Recall into a genuinely new conversation, and a second user added specifically to prove no leakage.

The claim being tested: **long-term memory IS retrieval, and the namespace is the entire security model.**

---

## The Mechanism

```
MemoryStore.add(user_id, kind, text, when)   kind = episodic | semantic
MemoryStore.search(user_id, query, k)
   -> FILTER BY user_id FIRST, then rank by cosine. Never the other way round.
```

---

## Key Implementation Details

**Runs on `BAAI/bge-small-en-v1.5`**, already in the local HuggingFace cache, so the module is fully offline and free.

**The filter precedes similarity.** Filter after ranking and a neighbouring tenant's row can still displace yours from top-k.

**The leak test is an `assert` in the notebook**, not in a test file, because a missing `user_id` is a B2B incident rather than a bug.

---

## Fixes Applied (during run)

None — ran clean on first execution after switching `get_sentence_embedding_dimension()` to the non-deprecated `get_embedding_dimension()`.

---

## Actual Output

```
encoder ready, dim = 384
[episodic] 0.71  Alice was declined at 95% LTV and asked about a larger deposit.
...
Bob's rows visible to Alice: NONE
assertion passed — namespace isolation holds
```

---

## How to Run

Open `24_memory_long_term_vector.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`25_memory_procedural_and_forgetting.ipynb`
