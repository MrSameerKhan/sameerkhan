# JPMorgan Chase R1 — Morning Cheat Sheet (April 15, 12 PM IST)

> Read this in 20 minutes. Then close it and trace one ReAct loop in your head.

---

## 0. Interview Mindset

```
This is a TECHNICAL deep-dive — not behavioral.
They will ask "how does X work" and then "now implement it" or "now trace through it."

Formula for every answer:
  1. Define the concept in 1 sentence
  2. Give the mechanism ("the way it works is...")
  3. Dry-run with numbers or pseudo-code
  4. State a real trade-off or failure mode

Show working, not just conclusions.
```

---

## 1. RAG — Retrieval Augmented Generation

### Full Pipeline

```
OFFLINE (build index):
  Raw docs → Chunk → Embed → Store in vector DB

ONLINE (query):
  Query → Embed → Retrieve top-k → [Rerank] → Prompt + Context → LLM → Answer
```

### Chunking Decisions

```
Chunk size:      512 tokens = good default. Too small = missing context. Too large = diluted retrieval.
Overlap:         50-100 tokens between chunks — preserve sentence boundaries.
Strategy:        Recursive text splitter (paragraph → sentence → word) beats fixed-size splits.
Semantic split:  Split at embedding similarity drops — best quality, slower.
```

### Retrieval

```
Dense (vector):   embed query → cosine similarity → top-k from FAISS/Chroma/Weaviate
Sparse (BM25):    TF-IDF style keyword match → fast, good for exact terms
Hybrid:           RRF = 1/(rank_dense + k) + 1/(rank_sparse + k), k=60 default

Why hybrid: Dense = semantic understanding. Sparse = exact keyword recall.
Neither alone is best — hybrid always beats single retrieval on mixed queries.
```

### Reranking

```
Cross-encoder reranker:
  Takes (query, chunk) pair → outputs single relevance score
  Much slower (can't precompute) but far more accurate than bi-encoder
  Common: ms-marco-MiniLM-L-6-v2

Flow: retrieve top-50 → rerank → take top-5 for prompt context
```

### Evaluation

```
Context Recall     = % of ground-truth answers findable in retrieved context
Context Precision  = % of retrieved chunks that are relevant
Answer Faithfulness = does LLM answer stick to retrieved context?
Answer Relevance   = does LLM answer actually address the question?

Tool: RAGAS library — computes all four automatically with LLM-as-judge
```

### Common Failure Modes

```
Retrieval failure   → irrelevant chunks → LLM hallucinates to fill gap
Chunking failure    → answer split across chunk boundary → never retrieved
Reranker skipped    → precision low → context cluttered → answer diluted
No eval set         → you don't know if RAG is better than baseline
```

---

## 2. Fine-Tuning

### When to Fine-Tune vs Prompt

```
Use prompting when:
  - Task can be solved with good examples in context
  - You don't have 500+ labeled examples
  - Latency budget is flexible

Fine-tune when:
  - Consistent output format is required (JSON, structured extraction)
  - Task requires domain knowledge not in base model
  - Inference cost at scale matters (smaller fine-tuned < larger prompted)
  - Privacy: can't send data to external API
```

### LoRA — Low-Rank Adaptation

```
Idea: don't update full weight matrix W (d×d parameters)
      instead learn ΔW = A·B   where A is d×r, B is r×d, r << d

Parameters: 7B model = 7B params. LoRA with r=16 = ~0.1% of params trained.

W_new = W_frozen + α/r · A·B

Key hyperparams:
  r     = rank (8, 16, 32, 64) — higher r = more capacity, more memory
  α     = scaling (usually = r or 2r)
  target_modules = ["q_proj", "v_proj"]  ← attention layers most impactful

QLoRA = LoRA + quantize base model to 4-bit → fine-tune 70B on single A100
```

### RLHF Pipeline

```
Step 1: SFT (Supervised Fine-Tuning)
  Fine-tune base LLM on (prompt, good_response) pairs
  Standard cross-entropy loss

Step 2: Reward Model
  Input: (prompt, response) → output: scalar score
  Train on preference pairs (y_w preferred over y_l):
  L = -log σ(r(x, y_w) − r(x, y_l))    ← Bradley-Terry loss

  Dry-run: r_w=2.4, r_l=0.6 → difference=1.8 → σ(1.8)=0.858 → L=-log(0.858)=0.153

Step 3: PPO
  R_penalized(x,y) = R(x,y) − β·KL(π_θ || π_ref)
  Example: R=1.9, KL=0.45, β=0.2 → R_penalized = 1.9 − 0.09 = 1.81
```

### DPO — Direct Preference Optimization

```
Skips reward model entirely. One loss:

L_DPO = -log σ(β · [log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)])

Dry-run:
  log π_θ(y_w) = -0.50,  log π_ref(y_w) = -0.80  → ratio_w = +0.30
  log π_θ(y_l) = -1.50,  log π_ref(y_l) = -1.10  → ratio_l = -0.40
  β=1 → argument = 0.30 - (-0.40) = 0.70
  σ(0.70) = 0.668 → L = -log(0.668) = 0.404

DPO vs RLHF:
  DPO: simpler, no separate RM, fewer hyperparams, slightly lower ceiling
  PPO: more powerful, harder to tune, needs RM, better for complex tasks
```

---

## 3. Agents — ReAct Pattern

### The Loop

```
THOUGHT:  "I need to find the population of Paris. I'll use search."
ACTION:   {"tool": "web_search", "query": "Paris population 2024"}
OBSERVATION: "Paris population is approximately 2.1 million (2024)"
THOUGHT:  "I have the answer. I can respond now."
ACTION:   {"tool": "final_answer", "answer": "Paris has ~2.1 million people"}
```

### Tool Call Structure

```python
# Tool definition (schema)
tools = [{
    "name": "web_search",
    "description": "Search the web for current information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    }
}]

# Agent loop
while turn < max_turns:
    response = client.messages.create(model=model, tools=tools, messages=messages)
    
    if response.stop_reason == "end_turn":
        break  # done
    
    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(block.name, block.input)
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                ]})
```

### LangGraph Pattern

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # ← operator.add = append, not replace
    turn_count: int

graph = StateGraph(AgentState)
graph.add_node("agent", call_llm)
graph.add_node("tools", execute_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue,
    {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()

# Key: operator.add on messages means each node APPENDS to message list
# Without it, nodes would OVERWRITE — you'd lose history
```

### Memory Types

```
In-context:    messages list — fast, limited by context window
Vector store:  FAISS/Chroma — semantic search across past sessions
Key-value:     Redis — store entities {"user_name": "Sameer"}
Episodic:      past conversation summaries — compressed long-term memory

When to use what:
  Short conversation          → in-context only
  Long sessions               → summarize + store in vector
  Factual entity tracking     → key-value (Redis)
  Cross-session memory        → vector store + entity store
```

### Agent Failure Modes

```
Infinite loop       → add max_turns budget (10-15), raise after
Tool hallucination  → validate inputs with Pydantic before execution
Prompt injection    → never interpolate tool outputs directly into system prompt
Cost explosion      → track total tokens, kill if > budget
Context overflow    → sliding window or summarize old messages
```

---

## 4. LLM Core Concepts

### Attention

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V

Q = query  (what am I looking for?)
K = key    (what does each token offer?)
V = value  (what does each token contain?)

√d_k scaling: prevents dot products from growing too large → keeps softmax gradients healthy

Multi-head: run h attention heads in parallel, each with different Q,K,V projections
  → each head learns to attend to different aspects (syntax, semantics, coreference)
```

### KV Cache

```
Problem: at each decoding step, recompute attention over ALL previous tokens = O(n²)
KV Cache: store K and V tensors from all previous steps, reuse them

Result: decoding step goes from O(n) to O(1) per token
Trade-off: memory grows linearly with sequence length
  llama-2-7B, seq=4096, batch=1 → ~2GB KV cache just for this sequence

Flash Attention: reorders computation to avoid materializing full attention matrix
  Memory: O(n²) → O(n).  Speed: 2-4× faster on long sequences.
```

### Tokenization

```
BPE (Byte-Pair Encoding): merge most frequent byte pairs iteratively
  "lower" → ["low", "er"] — vocabulary built from training data

Vocabulary size: GPT-4 = 100K tokens. Llama-3 = 128K tokens.
Rule of thumb: 1 token ≈ 0.75 words (English)

Implication: non-English text = more tokens per word → costs more
```

### Context Window vs RAG

```
When to use long context:       document fits (< 128K tokens), no retrieval latency budget
When to use RAG:                document too large, need freshness, cost control at scale

Lost-in-the-middle problem:     LLMs attend poorly to middle of long contexts
  Fix: put most critical info at start or end of context
```

---

## 5. AWS for ML — Quick Reference

```
Service           Use Case
─────────────────────────────────────────────────────
S3                Store datasets, model artifacts, checkpoints
EC2 GPU           p3.2xlarge (V100, $3.06/hr), g4dn.xlarge (T4, $0.53/hr)
SageMaker         Managed training jobs, HP tuning, endpoints, auto-scaling
Lambda            Serverless inference (small models, spiky traffic)
ECS + ECR         Containerized inference, own infra control
IAM               Least-privilege roles for training jobs
Step Functions    Pipeline orchestration (preprocess→train→evaluate→deploy)
CloudWatch        Metrics, alarms, logs
inf1 instances    AWS Inferentia chip — 70% cheaper than GPU for inference
```

### SageMaker Training Job (Key Pattern)

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    framework_version="2.0",
    hyperparameters={"epochs": 5, "lr": 2e-5}
)

estimator.fit({"train": "s3://bucket/data/train/",
               "val": "s3://bucket/data/val/"})

# Inside train.py: os.environ["SM_CHANNEL_TRAIN"] = local path to data
```

### Cost Optimization

```
1. Spot instances: 70-90% cheaper, can be interrupted
   → Always checkpoint every 500 steps, use SageMaker managed spot
2. inf1 for inference: 70% cheaper than p3 for same throughput
3. S3 lifecycle: move models to Glacier after 90 days → 23× cheaper storage
4. Right-size: start with g4dn.xlarge ($0.53/hr) before jumping to p3 ($3.06/hr)
```

---

## 6. LangChain / LangGraph Key Patterns

```
Chain:           LLM | PromptTemplate | OutputParser — sequential, no branching
Agent:           LLM + tools + loop — can decide to use tools
LangGraph:       Explicit state machine — you define nodes, edges, conditions

Use LangGraph when:
  - Multi-step agent with branching logic
  - Need to persist state across turns
  - Need human-in-the-loop pause points
  - Debugging matters (LangSmith traces every node)

Key LangGraph concepts:
  StateGraph   → define the state schema (TypedDict)
  add_node     → add a computation step
  add_edge     → unconditional flow
  add_conditional_edges → branching based on function output
  operator.add → for lists: means APPEND (not replace)
  checkpointer → enable pause/resume (MemorySaver for dev, PostgresSaver for prod)
```

---

## 7. The Questions They Will Definitely Ask

```
"Walk me through how RAG works end to end."
  → Offline: chunk → embed → store. Online: embed query → retrieve → rerank → prompt → generate.

"What's the difference between LoRA and full fine-tuning?"
  → LoRA: freeze base, learn low-rank ΔW = A·B (0.1% params). Full: update all params.

"How does the ReAct agent loop work?"
  → Thought → Action (tool call) → Observation → Thought → ... → Final Answer

"What's DPO and how does it differ from RLHF?"
  → RLHF: train reward model, then PPO. DPO: direct preference learning, no RM needed.

"How does attention work?"
  → softmax(QKᵀ / √d_k) · V. Query finds relevant keys, values are weighted sum.

"What is KV cache?"
  → Store K,V tensors for all previous tokens, reuse at each decoding step. O(n²) → O(1).

"When would you use agents vs chains?"
  → Chain: fixed, predictable. Agent: dynamic, decides what tools to call and when.

"How do you evaluate a RAG system?"
  → Context Recall, Context Precision, Faithfulness, Answer Relevance (RAGAS).
```

---

## 8. Morning Schedule

```
11:00 AM — Read this cheat sheet (20 min)
11:20 AM — Trace the DPO dry-run in your head (2 min)
11:25 AM — Trace the ReAct Paris example in your head (2 min)
11:30 AM — Quiet room. Water. Camera on.
11:50 AM — Join 10 min early. Test audio/video.
12:00 PM — Interview starts.

Opening: always define before you explain.
          "DPO is... the way it works is... let me trace through a concrete example..."
```
