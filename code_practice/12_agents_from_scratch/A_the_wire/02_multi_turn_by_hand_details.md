# Module 02 — Multi-turn by Hand
Status: `✅ Run`

Theory: [../../../8.agents/01b_agents_end_to_end.md](../../../8.agents/01b_agents_end_to_end.md) §1.3 (memory IS the context window) · [../../../8.agents/05_agent_memory.md](../../../8.agents/05_agent_memory.md) §2 (short-term memory + compaction)

---

## Use Case

Module 01 sent one message and read one reply. Every agent from module 07 onward is a
loop that appends to a list. This module builds that list by hand, so that when the loop
arrives it contains no magic — just `messages.append(...)` in a `while`.

The claim being tested: **the API is stateless; memory is the list you carry.**

---

## The Mechanism

```
STATELESS — two independent calls
  call 1:  [user "My name is Sameer"]           → asks for clarification
  call 2:  [user "What is my name?"]            → "I'm not able to know your name"
           ↑ brand-new list. No session id exists to send.

CARRIED — one list you own
  turn 1:  [user A]                             → reply A
  turn 2:  [user A, asst A, user B]             → reply B    ← re-sends EVERYTHING
  turn 3:  [user A, asst A, user B, asst B, user C] → reply C

  The model gained nothing. You gained a list.
```

---

## Key Implementation Details

**`ask()` takes no state handle.** No session id, no conversation id. That absence is the
lesson — there is nothing to pass, because the server stores nothing.

**Append the assistant turn too.** A list containing only user turns loses the model's own
prior answers and it starts contradicting itself. Both roles go in.

**Runs on `gpt-4.1-mini`, not Opus 5.** Module 01 already taught both envelopes, so this
module stays on one idea. It also keeps the numbers legible: Opus 5 would add several
hundred invisible thinking tokens per turn and bury the growth curve being measured.

---

## Fixes Applied (during run)

| Found | Fix |
|---|---|
| The lesson text claimed **"INPUT grows while OUTPUT stays flat."** The measured output was 51, 16, 66 tokens — noisy, not flat. | Reworded to **"INPUT grows with conversation length while OUTPUT does not."** That is what the data supports: output tracks what was asked, and shows no growth trend. The original phrasing invited a reader to expect a straight line and then distrust the module when they saw one. |

---

## Actual Output (macOS M1, `gpt-4.1-mini`, 2026-09-12)

**Statelessness proved.** Call 2, on a fresh list:

```
CALL 2 -> I'm not able to know your name or personal details unless you share
          them with me. Also, you haven't asked a question yet.
```

**The carried list answers it.** Same question, turn 2:

```
TURN 2 -> Your name is Sameer, and you asked for a "5-year fix."
```

**The cost of that memory:**

| turn | msgs | in | out |
|---|---|---|---|
| 1 | 2 | 21 | 51 |
| 2 | 4 | 91 | 16 |
| 3 | 6 | 123 | 66 |

Input went **21 → 123, a 5.9x rise across only three turns**, while output moved 51 → 16 →
66 with no trend. The growth is entirely re-sent history, and it is already steep at turn
three. Project that across a ten-step agent loop and module 11's ~10x multiplier stops
being a claim.

---

## How to Run

```bash
conda activate sameerkhan
cd code_practice/12_agents_from_scratch/A_the_wire
```

Open `02_multi_turn_by_hand.ipynb`, select the `sameerkhan` kernel, run all cells. Needs
`OPENAI_API_KEY` only — no Anthropic, no Ollama. Five calls, well under a cent.

---

## Next

`03_structured_output.ipynb` — why parsing free text is a bug, and how constrained output
deletes a whole failure class.
