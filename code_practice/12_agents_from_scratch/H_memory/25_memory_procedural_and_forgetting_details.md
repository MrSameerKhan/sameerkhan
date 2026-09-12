# Module 25 — Procedural Memory and Forgetting
Status: `✅ Run (no API key)`

Theory: [../../../8.agents/05_agent_memory.md](../../../8.agents/05_agent_memory.md) §3 (procedural) · §4 (forget / consolidation / eviction)

---

## Use Case

The two tiers almost nobody builds: remembering *how* to do a task, and deliberately forgetting.

The claim being tested: **a store that only grows degrades, because precision falls as the noise-to-signal ratio rises.**

---

## The Mechanism

```
Skill(trigger, steps, successes, failures, last_used)
  reliability = Laplace-smoothed success rate  (1/1 is NOT 100%)
  score       = reliability * exp(-age * ln2 / half_life)
  evict when score < threshold
```

---

## Key Implementation Details

**Laplace smoothing** stops one lucky success outranking a long track record.

**The decay half-life is why the pandemic payment-holiday skill is evicted despite a perfect 9-0 record.** A pure success-rate ranking keeps it forever and confidently applies a policy that no longer exists.

---

## Fixes Applied (during run)

None — ran clean on first execution.

---

## Actual Output

```
after 4 more runs: 5W/1L  reliability=0.75
evicted 1: ['apply a pandemic payment holiday']
```

---

## How to Run

Open `25_memory_procedural_and_forgetting.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`../I_planning/26_plan_and_execute.ipynb`
