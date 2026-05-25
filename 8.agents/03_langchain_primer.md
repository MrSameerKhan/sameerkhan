# LangChain Primer

> Why this file exists: LangChain is the most-used agent framework as of 2024-2025, but it's also the most-criticized. This primer gives you the working subset (LCEL, runnables, output parsers) and is honest about what's better done with LangGraph or with no framework at all.

---

## Quick Reference

| Concept | What it is |
|---------|-----------|
| LCEL (LangChain Expression Language) | Compose components with `\|` operator (`prompt \| model \| parser`) |
| Runnable | Anything that has `.invoke()`, `.batch()`, `.stream()`, `.ainvoke()` |
| PromptTemplate / ChatPromptTemplate | Templated prompts with `{variable}` slots |
| Output parser | Coerce raw LLM output into typed Python objects |
| Memory | Stateful conversation history (mostly legacy — use LangGraph state instead) |
| Tools | Functions the LLM can call (function/tool calling) |
| Chains | LCEL-composed Runnables (replaces the old `LLMChain` / `Chain` classes) |
| Agents | LLM + tools + loop. Legacy `AgentExecutor`; modern: LangGraph (see `04_langgraph_deep.md`) |
| LangSmith | Observability product (paid; companion to LangChain) |

---

## 1. LCEL — The One Idea That Matters

LCEL replaced the old class-based chains with a **pipe-composition syntax**:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate '{text}' to {language}."),
])
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser   # LCEL composition

chain.invoke({"text": "hello world", "language": "Spanish"})
# → "Hola mundo"
```

Every component is a Runnable. The `|` operator builds a new Runnable that runs them in sequence. Same chain supports `.invoke()`, `.batch()`, `.stream()`, and async (`.ainvoke()`) for free.

### Why LCEL Exists

The pre-LCEL API (`LLMChain`, `SequentialChain`, etc.) had inconsistent interfaces, no streaming, no batching, hard to debug. LCEL fixes all that with one pipe operator.

---

## 2. Runnables — The Building Blocks

Every LangChain component implements the Runnable protocol:

```python
class Runnable:
    def invoke(self, input):    # single call
    def batch(self, inputs):    # parallel calls
    def stream(self, input):    # token-stream
    async def ainvoke(self, input):  # async
    async def abatch(self, inputs):
    async def astream(self, input):
```

**Useful runnables built into LangChain:**

| Runnable | Purpose |
|---------|---------|
| `RunnablePassthrough` | Pass input through unchanged (for parallel branches) |
| `RunnableLambda(fn)` | Wrap a regular Python function as a Runnable |
| `RunnableParallel({"a": chain_a, "b": chain_b})` | Run multiple chains on the same input; returns dict |
| `RunnableBranch` | if/else routing based on a condition |
| `RunnableWithMessageHistory` | Add chat history persistence to a Runnable |
| `RunnableConfig` | Attach callbacks, tags, run name (for LangSmith tracing) |

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Run two summaries in parallel, then combine
summarize_short = prompt_short | model | parser
summarize_long  = prompt_long  | model | parser

combined = RunnableParallel({
    "short": summarize_short,
    "long": summarize_long,
    "original": RunnablePassthrough(),
}) | RunnableLambda(lambda d: f"Short: {d['short']}\nLong: {d['long']}")

combined.invoke("Article text here...")
```

---

## 3. Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Today is {today}."),
    MessagesPlaceholder("chat_history"),   # injects prior messages
    ("human", "{user_input}"),
])

# Partial fill (locks one variable, leaves others open)
prompt_for_today = prompt.partial(today="2026-05-10")
```

`MessagesPlaceholder` is how you wire chat history into a prompt without hardcoding the turn count.

---

## 4. Output Parsers — Getting Typed Data

```python
from langchain_core.output_parsers import (
    StrOutputParser, JsonOutputParser, PydanticOutputParser
)
from pydantic import BaseModel

class Joke(BaseModel):
    setup: str
    punchline: str

parser = PydanticOutputParser(pydantic_object=Joke)

prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}\n{format_instructions}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
joke = chain.invoke({"topic": "programming"})
# joke is a Joke instance — typed
```

**However:** PydanticOutputParser uses prompt instructions to coerce JSON, which fails ~5-15% of the time on schema violations. For production, prefer:

- OpenAI / Anthropic **native structured outputs** (`response_format={"type": "json_schema", ...}`) or `tool_use`
- **Instructor library** — wraps Pydantic + native structured output with retries
- **outlines** — constrained decoding for local models

See `../5.transformers/models/12_constrained_decoding.md`.

LangChain has caught up partially via `with_structured_output(Pydantic_model)`:

```python
model_structured = model.with_structured_output(Joke)
chain = prompt | model_structured   # no separate parser needed
```

---

## 5. Tools & Tool Calling

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 25°C in {city}"

# Bind tools to a model
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in Tokyo?")
response.tool_calls   # [{"name": "get_weather", "args": {"city": "Tokyo"}, "id": "..."}]

# Execute manually
if response.tool_calls:
    for call in response.tool_calls:
        result = get_weather.invoke(call["args"])
        # feed result back to model
```

**Modern pattern:** don't use `AgentExecutor` (legacy). Use **LangGraph** for the tool-use loop (see next file).

---

## 6. Memory (Mostly Legacy)

LangChain has many memory classes (`ConversationBufferMemory`, `ConversationSummaryMemory`, `ConversationKGMemory`, ...). Most are legacy from the pre-LangGraph era.

**Modern advice:** don't use LangChain memory classes. Use **LangGraph state** for in-graph history (see `04_langgraph_deep.md`) · **Vector store + retrieval** for long-term memory (see `05_agent_memory.md`) · **mem0 / Letta** for managed agent memory backends.

The one exception: `RunnableWithMessageHistory` is useful for simple chat history persistence with LCEL chains.

---

## 7. When to Use LangChain vs Not

**Use LangChain when:**
- Rapid prototyping — you want chains stitched together fast
- Integrations: it has wrappers for ~100+ LLM/vector-DB/tool providers (saves you from each provider's SDK)
- Your team already uses it (familiarity > preference)
- Simple linear chains (`prompt → model → parser`)

**Use LangGraph when:**
- You need explicit state machines (loops, conditionals, retries, HITL)
- Production agents — LangGraph is the modern successor
- You want checkpointing / time travel / persistence
- See `04_langgraph_deep.md`

**Use neither (direct SDK) when:**
- Performance / latency critical — LangChain adds 100-300ms of overhead per chain
- Simple single-call LLM apps — `openai.chat.completions.create()` is just as easy
- Avoiding the dependency tree (LangChain pulls 100+ packages)
- You need fine control over caching, retries, prompts

**Use Pydantic AI / smolagents / etc. when:**
- Type-safety matters (Pydantic AI)
- Code-as-action paradigm (smolagents — LLM emits Python instead of JSON tool calls)
- See `07_multi_agent_orchestration.md`

---

## 8. Honest Critique

LangChain is widely used but widely criticized. The main complaints:

| Complaint | Severity |
|-----------|----------|
| Pre-LCEL legacy classes still in docs → newcomer confusion | High |
| Frequent breaking changes (0.1 → 0.2 → 0.3 within ~18 months) | High |
| Heavy abstraction — hard to debug what prompts the model actually sees | Medium |
| Performance overhead (~100-300ms per chain) | Medium |
| Sprawling dependency tree | Medium |
| Some integrations are thin wrappers — easier to use provider SDK directly | Low |

**Mitigations:** Always use `set_debug(True)` or LangSmith to see actual prompts/responses · Pin versions in `requirements.txt` — don't auto-upgrade · For production, lift critical paths out of LangChain (talk to provider SDK directly) · Use LCEL exclusively; avoid legacy classes.

---

## 9. Migration Path (Old → New)

```
LLMChain(prompt=..., llm=..., output_parser=...)
    ↓
prompt | model | parser    # LCEL

SequentialChain([chain1, chain2])
    ↓
chain1 | chain2

ConversationChain(memory=...)
    ↓
RunnableWithMessageHistory(chain, get_session_history=...)

AgentExecutor(agent=..., tools=...)
    ↓
LangGraph (see 04_langgraph_deep.md)
```

If you inherit a codebase with `LLMChain` / `AgentExecutor` patterns, plan a migration to LCEL + LangGraph. The legacy classes still work but won't get new features.

---

## 10. Code Template — Complete Working Chain

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI

# 1. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise assistant. Always cite sources."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# 2. Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 3. Chain
chain = prompt | model | StrOutputParser()

# 4. Add session-keyed history
store: dict[str, InMemoryChatMessageHistory] = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 5. Use it
config = {"configurable": {"session_id": "user_42"}}
print(chat.invoke({"input": "Hi, I'm Alex."}, config))
print(chat.invoke({"input": "What's my name?"}, config))
# → "Your name is Alex."
```

---

## 11. Gotchas

**Hidden prompts:** LangChain prompt templates can include hidden formatting (e.g., `format_instructions`). Print the actual prompt with `chain.get_prompts()` and `prompt.format_messages(...)` before trusting it.

**Lazy imports:** `from langchain_chat_models import ChatOpenAI` works but is deprecated — use `from langchain_openai import ChatOpenAI`. Different sub-packages now.

**Version pinning:** A `pip install langchain` without a version pin will pull whatever is latest, which often breaks. Pin to a specific minor version (`langchain==0.3.*`).

**`AgentExecutor` is legacy.** Tutorials still reference it. For new agent code, start with LangGraph from day one.

**Memory + tools combo is awkward.** Combining LangChain memory with tool use historically required custom callbacks. LangGraph state replaces both cleanly.

---

## 12. Interview Q&A

**Q: What is LCEL and why does it matter?**

LCEL (LangChain Expression Language) is a composition syntax that lets you build chains by piping Runnables together with `|`. Every component — prompt templates, models, parsers, custom functions — implements the Runnable protocol (`.invoke()` / `.batch()` / `.stream()` / `.ainvoke()`). LCEL replaced the old class-based chains (`LLMChain`, `SequentialChain`) which had inconsistent interfaces, no streaming, and no batching. Practical impact: a chain you write once supports synchronous calls, async, batching, and token-streaming for free; debugging and observability work uniformly across all components.

**Q: When would you NOT use LangChain?**

Several cases: (1) **Latency-critical paths** — LangChain adds 100-300ms overhead per chain, which is unacceptable for sub-second SLAs; talk to the provider SDK directly. (2) **Single-call LLM apps** — calling `openai.chat.completions.create()` directly is no harder than wrapping it in LangChain, with fewer dependencies. (3) **Production agents** — start with LangGraph (which is technically separate but closely related), not LangChain's legacy `AgentExecutor`. (4) **When you need clear visibility** into the exact prompt sent to the model — LangChain's abstractions can hide that; so I always run `set_debug(True)` or use LangSmith during development.

**Q: LangChain vs LangGraph?**

LangChain is for composing linear/branching chains of LLM calls using LCEL. LangGraph (from the same team) is for building stateful agents as explicit state graphs — nodes are functions, edges define transitions, the graph carries persistent state. LangGraph is the right tool for "prompt → model → parser" pipelines. LangGraph is the right tool for agentic loops with tool use, retries, conditional logic, human-in-the-loop, and persistence. The two are complementary: a LangGraph node may itself contain an LCEL chain. Modern production agents use LangGraph as the outer loop, LCEL inside individual nodes.

**Q: What's the difference between `RunnablePassthrough` and `RunnableLambda`?**

`RunnablePassthrough` passes its input through unchanged — useful in `RunnableParallel` to keep the original input alongside computed values. `RunnableLambda(fn)` wraps an arbitrary Python function as a Runnable so it composes with `|`. They serve different purposes: pass-through preserves data flow without modification; lambda lets you inject custom Python logic into a chain.

---

## 13. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| LangGraph deep-dive | `04_langgraph_deep.md` | Short-term memory = LangGraph state |
| Agent fundamentals | `01_agents.md` | What an agent is conceptually |
| Multi-agent frameworks | `07_multi_agent_orchestration.md` | Planner-executor patterns |
| Structured output / constrained decoding | `../5.transformers/models/12_constrained_decoding.md` | Alternative to PydanticOutputParser |
| Pydantic structured extraction | `../4.nlp/04_applications/03_information_extraction.md` | Same pattern for tool-call validation |
| LLM observability (LangSmith etc.) | `../10.mlops/11_llm_observability_tools.md` | |
| Code practice | `code_practice/06_agents/02_langchain_primer/` | |

---

## Key Takeaway

LangChain has one good idea (LCEL) and a lot of legacy baggage. Use it for: rapid prototyping; integrations. Prefer for stateful agents: use Pydantic + Instructor for structured outputs; use `with_structured_output` (Pydantic model); avoid `SequentialChain`, and most of the memory classes — **the migration path is to LangGraph**. Always use `set_debug(True)` to print actual prompts during dev, and don't be afraid to drop down to the provider SDK directly when the abstractions cost more than they save.

---

## Code Practice — Wired by Phase 6

- `code_practice/06_agents/02_langchain_primer/` —
