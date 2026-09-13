# Module 11 — Agent Cost and Tokens
Status: `🔧 Code-built` — **re-run required after the correction below**

Theory: [../../../8.agents/01b_agents_end_to_end.md](../../../8.agents/01b_agents_end_to_end.md) §2.2 (tokens per turn) · [../../../8.agents/09_agent_evaluation.md](../../../8.agents/09_agent_evaluation.md) §7 (cost thresholds)

---

## Use Case

Puts a measured number on the claim every agent article repeats without evidence.

The claim being tested: **an agent costs roughly 10x a single call, and the driver is
re-sent input.**

---

## Fixes Applied (during run)

| Found | Fix |
|---|---|
| **The headline claim did not survive its own measurement.** The input multiplier came out at **10.2x**, exactly as predicted. But total cost came out at **0.7x** — the agent loop was *cheaper* than the single call. The module's closing text asserted the ~10x total anyway. | Rewrote the lesson to lead with the **input** multiplier, which is what the module actually measures, and to explain when total cost falls below 1x. The "~10x" figure is now stated as what it is: an assumption about a **long** loop. |

**Why it happened, and why it is worth keeping.** Output is priced around 4x input. The
no-tools baseline answered discursively in **287 output tokens**; the agent answered in
**112** across two turns. A verbose single call can out-cost a terse two-turn agent even
while the agent re-sends ten times the input. The trade only tips the other way as turns
accumulate, because input grows roughly with the square of turn count.

A demo that asserts 10x while printing 0.7x teaches the reader to distrust the demo.

---

## Actual Output (macOS M1, `gpt-4.1-mini`, 2026-09-12)

```
single call: in=36 out=287 cost=$0.000474

turn  msgs      in    out   $ this turn
   1     1     124     54      0.000136
   2     4     242     58      0.000190

                       in     out           $
single call            36     287    0.000474
agent loop            366     112    0.000326
multiplier          10.2x    0.4x        0.7x

first turn input 124 -> last turn input 242
```

**The input story held perfectly.** 36 -> 366 tokens is 10.2x, and within the loop itself
input nearly doubled from turn 1 to turn 2 on a two-turn run. Extrapolate that curve
across ten turns and the familiar multiplier appears — which is the point the corrected
lesson now makes explicitly instead of assuming.

---

## How to Run

Open `11_agent_cost_and_tokens.ipynb`, run all cells. Needs `OPENAI_API_KEY` only.

---

## Next

`../D_reliability/12_failure_modes_reproduced.ipynb`
