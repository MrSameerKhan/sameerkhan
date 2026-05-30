# Constrained Decoding — Structured Output Generation

> Force LLM output to satisfy a grammar or schema at every decode step — not via prompt engineering, but by masking invalid tokens in the logits. Guarantees syntax-valid output by construction.

---

## Core Concept

Standard LLM generation: at each step, sample freely from the full vocabulary.

Constrained decoding: at each step, compute the set of tokens that keep the partial output schema-valid, then mask all others to -inf before softmax.

```python
# Standard step
logits = model.forward(emitted)
next_tok = sample(softmax(logits))

# Constrained step
logits = model.forward(emitted)
allowed = grammar_state.next_token_mask(emitted)   # boolean[vocab]
logits  = logits.masked_fill(~allowed, float('-inf'))
next_tok = sample(softmax(logits))
grammar_state.advance(next_tok)
return next_tok
```

The grammar or schema defines which token sequences are valid. At each position, only the subset of tokens that can legally follow the current partial output is allowed.

```mermaid
stateDiagram-v2
    [*] --> expect_open : start JSON generation

    expect_open --> expect_key : emit  {
    expect_key --> in_key : emit  "
    in_key --> in_key : any char except "
    in_key --> expect_colon : emit  "   end key
    expect_colon --> expect_value : emit  :

    expect_value --> in_str_val : emit  "   string
    expect_value --> in_number : emit  0-9   number
    expect_value --> in_bool : emit  t/f   boolean
    expect_value --> expect_open : emit  {   nested

    in_str_val --> in_str_val : any char except "
    in_str_val --> after_value : emit  "   end value

    in_number --> in_number : 0-9
    in_number --> after_value : next non-digit

    in_bool --> after_value : complete true/false

    after_value --> expect_key : emit  ,   more keys
    after_value --> [*] : emit  }   done

    note right of expect_value
        At each state: only valid
        next tokens are ALLOWED
        All others masked to -inf
        before softmax
        Guarantees valid schema
        by construction — not prompt
    end note
```

---

## Approaches

**JSON Schema-guided:** parse the JSON schema into a grammar; at each token step only allow tokens consistent with the partial JSON. Forces `{`, `"key"`, `:`, value tokens, `}` etc. in correct order with correct types.

**CFG-guided (Context-Free Grammar):** define exact grammar in EBNF/GBNF. Python is huge but tractable; SQL is well-defined. For deep nested JSON on long contexts, grammar stack can grow.

**Regex-guided:** simpler — match output to a regex pattern. Used for phone numbers, dates, structured IDs.

**Outlines (popular open-source library):** unified Python API for JSON schema, regex, and CFG-guided decoding.

**lm-format-enforcer:** lightweight alternative to outlines; used in vLLM production serving.

**vLLM guided_decoding:** production-grade integration; supports JSON schema natively via `guided_decoding` sampling params.

---

## 3. When to use

| Use case | Pick |
|----------|------|
| Extract entities from text into JSON | JSON Schema-guided |
| Generate valid SQL queries | CFG-guided (define SQL grammar) |
| Match phone numbers, dates, etc. | Regex-guided |
| Function calling / tool-use | JSON Schema for tool args |
| You're using OpenAI / Anthropic | Native structured output |
| Local / on-prem deployment | vLLM `guided_decoding` or outlines |
| Code generation | Full CFG for Python is huge; usually combine with prompt + retry |

**Don't use** when output is genuinely free-form (creative writing, chat) — the constraint makes generation worse, not better.

---

## 4. Code / formula

### Outlines (the popular open-source library)

```python
from outlines import models, generate
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    is_subscribed: bool

model     = models.transformers("llama-3-8b")
generator = generate.json(model, User)
result    = generator("Extract the user from: 'Alice, 30, subscribed'")
# result is a User instance — guaranteed valid by construction
```

### vLLM guided decoding (production)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/llama-3-8b")
sampling = SamplingParams(
    max_tokens=200,
    guided_decoding=GuidedDecodingParams(json=User.model_json_schema())
)
outputs = llm.generate([prompt], sampling)
# outputs[0].outputs[0].text is guaranteed valid JSON matching User schema
```

### The underlying mechanism (simplified)

```python
def constrained_step(model, emitted, grammar_state):
    logits = model.forward(emitted)
    allowed = grammar_state.next_token_mask(emitted)   # boolean[vocab]
    logits  = logits.masked_fill(~allowed, float('-inf'))
    next_tok = sample(softmax(logits))
    grammar_state.advance(next_tok)
    return next_tok
```

---

## 5. Failure modes

1. **Grammar is over-strict — model can't satisfy it** — if your schema is impossibly narrow ("name must start with 'X' and be exactly 7 chars"), the model is forced into garbage that satisfies. Fix: relax schema; use the model's natural distribution.

2. **Pre-filled tokens collide with grammar** — system prompt or chat template includes `<|assistant|>` followed by space; grammar template might expect `{`. Tokenization mismatch. Fix: tokenizer-aware schema construction.

3. **Performance regression at long contexts** — building the mask is O(vocab × grammar_state_depth). On 100K-token contexts with deep nested JSON, the mask construction can dominate. Cache compiled grammars.

4. **Hallucinated structure** — model outputs valid JSON but with wrong field values (`"age": -50`). Constrained decoding enforces SYNTAX, not SEMANTICS. Add a Pydantic validation layer on top.

5. **Doesn't compose with sampling temperature well** — extreme temperatures + restrictive grammar = the few allowed tokens become near-uniform. Reduce temperature when constraining heavily.

---

## 6. Interview questions (5)

**Q1: How do you guarantee an LLM produces valid JSON?**

Three levels: prompt-based (unreliable), provider JSON mode (syntax-valid), constrained decoding (schema-valid by construction). At each decode step, mask logits to allow only tokens that keep the partial output grammar-valid. Each token is therefore syntactically forced; schema validity is guaranteed by construction.

**Q2: What's the performance cost of constrained decoding?**

Typically 5-15% throughput hit. For simple JSON schemas the cost is negligible; for deep nested CFGs it can be 30%+. Compiled grammars (outlines) amortize the cost.

**Q3: When does constrained decoding hurt output quality?**

When the grammar is restrictive and the model's natural distribution wants to be elsewhere — output is forced through a narrow path. Also for genuinely creative tasks where any constraint is the wrong abstraction.

**Q4: Can you constrain Python or SQL generation?**

Yes via CFG-based constrained decoding (outlines; lm-format-enforcer support CFGs). The grammar must be defined in EBNF/GBNF. Python is huge but tractable. SQL is well-defined. The model can only syntactically valid code, but semantic errors remain (wrong types, wrong logic).

**Q5: How does this differ from JSON mode in OpenAI/Anthropic APIs?**

JSON mode enforces valid JSON SYNTAX (curly braces match, no trailing commas, valid quoting). Structured Outputs (OpenAI) goes further — enforces a specific SCHEMA. Anthropic's tool_use does the same for tool calls. Open-source equivalents are outlines (local, model-agnostic) or vLLM guided_decoding (production serving).

---

## 7. Further reading

- Outlines (Willard & Louf 2023) — arXiv:2307.09702
- Grammar-aligned decoding (Park et al. 2024) — arXiv
- lm-format-enforcer — GitHub: noamgat/lm-format-enforcer
- vLLM Guided Decoding docs
- Outlines documentation
- OpenAI Structured Outputs
