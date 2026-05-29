# 04 — CLIP Fine-Tuning End-to-End: Worked Examples

> Complete dry-run of CLIP: contrastive loss computation, forward pass, zero-shot inference, and fine-tuning. Every step traced with numbers.

---

## What CLIP Actually Does

```
Problem: Learn a joint embedding space for images and text
         such that matching image-text pairs are close,
         non-matching pairs are far apart.

Training data: 400 million (image, text) pairs from the internet
  ("a photo of a dog", [dog image])
  ("Invoice from Acme Corp", [invoice image])

Result: an image encoder and text encoder that map to the SAME vector space.
  Similar concepts → close vectors (high cosine similarity)
  Different concepts → distant vectors (low cosine similarity)

This enables:
  Zero-shot classification: no fine-tuning needed for new classes
  Image-text retrieval: find images matching a text query
  Fine-tuning backbone: strong visual features for downstream tasks
```

---

## Part 1: Architecture

### Image Encoder (ViT-B/32)

```
Input: image, resized to 224×224 RGB

Step 1: Patch extraction
  Patch size: 32×32 pixels
  Number of patches: (224/32) × (224/32) = 7 × 7 = 49 patches
  Each patch: 32×32×3 = 3072 values

Step 2: Linear projection
  Each patch → linear(3072, 768) → 768-dim patch embedding

Step 3: Add [CLS] token
  49 patch embeddings + 1 [CLS] token = 50 tokens total

Step 4: Add position embeddings (learned)
  50 learnable position vectors added to token embeddings

Step 5: 12-layer transformer (ViT-B/32)
  Self-attention: 12 heads × 64 dim/head = 768 total
  FFN: 768 → 3072 → 768
  Output: 50 contextual embeddings

Step 6: Take [CLS] token → project to embedding space
  [CLS] ∈ R^768 → linear(768, 512) → image_embedding ∈ R^512

Final: image_embedding ∈ R^512, L2-normalized (unit vector)
```

### Text Encoder

```
Input: text string (up to 77 tokens with BPE tokenizer)

Step 1: Tokenize
  "Invoice from Acme Corp" → [<SOS>, "Invoice", "from", "Ac", "me", "Corp", <EOS>]
  vocab_size=49408, embed_dim=512

Step 2: Token + position embeddings
  Padded to length 77

Step 3: 12-layer causal transformer (GPT-style)
  Causal mask: each token only attends to previous tokens
  Output: 77 contextual embeddings

Step 4: Take [EOS] token → project to embedding space
  [EOS] ∈ R^512 → linear(512, 512) → text_embedding ∈ R^512

Final: text_embedding ∈ R^512, L2-normalized (unit vector)

Why [EOS] instead of [CLS]?
  Causal attention means [EOS] has seen all previous tokens.
  It aggregates the full sentence meaning.
```

---

## Part 2: Contrastive Loss — Complete Dry Run

### Setup

```
Mini-batch: N=4 image-text pairs (N=32,768 in actual CLIP training)
  Pair 1: (invoice_image, "Invoice from Acme Corp")
  Pair 2: (dog_photo,     "A brown labrador dog running")
  Pair 3: (contract_image,"Legal contract for services")
  Pair 4: (receipt_image, "Restaurant receipt total $45")
```

### Step 1: Compute Embeddings

```
Image embeddings (after L2 normalization, each ∈ R^512):
  I₁ = image_encoder(invoice_image)   = [0.12, -0.30, 0.07, ...]
  I₂ = image_encoder(dog_photo)       = [-0.45, 0.33, 0.18, ...]
  I₃ = image_encoder(contract_image)  = [0.09, -0.29, 0.11, ...]
  I₄ = image_encoder(receipt_image)   = [0.31, -0.15, -0.22, ...]

Text embeddings (after L2 normalization, each ∈ R^512):
  T₁ = text_encoder("Invoice from Acme Corp")        = [0.11, -0.33, 0.08, ...]
  T₂ = text_encoder("A brown labrador dog running")   = [-0.48, 0.22, 0.17, ...]
  T₃ = text_encoder("Legal contract for services")    = [0.09, -0.28, 0.10, ...]
  T₄ = text_encoder("Restaurant receipt total $45")   = [0.30, -0.14, -0.21, ...]
```

### Step 2: Similarity Matrix

```
Compute cosine similarity between ALL pairs: S = I · T^T  (N×N matrix)
Since embeddings are L2-normalized: cosine_sim = dot product

s[i][j] = cosine_similarity(Iᵢ, Tⱼ)

              T₁(invoice) T₂(dog)  T₃(contract) T₄(receipt)
I₁(invoice) [  0.92,      0.03,      0.41,        0.38  ]
I₂(dog)     [  0.04,      0.90,      0.05,        0.05  ]
I₃(contract)[  0.39,      0.05,      0.88,        0.35  ]
I₄(receipt) [  0.36,      0.04,      0.33,        0.91  ]

Diagonal = matching pairs → HIGH similarity (0.88–0.92)
Off-diagonal = non-matching → LOW similarity (0.03–0.41)

Note: (invoice, contract) = 0.41 — somewhat similar (both business documents)
      (invoice, dog) = 0.03 — very different → good separation
```

### Step 3: Scale by Temperature

```
CLIP learns a temperature parameter τ (log scale):
  logit_scale = exp(log_scale)
  Initial log_scale = log(1/0.07) = 2.66  →  scale ≈ 14.3

Scaled logits = S × scale

              T₁         T₂         T₃         T₄
I₁:  [  13.16,   0.43,    5.86,    5.43  ]
I₂:  [   0.57,  12.73,    0.86,    0.71  ]
I₃:  [   5.58,   0.71,   12.58,    5.00  ]
I₄:  [   5.15,   0.57,    4.72,   13.01  ]

Temperature τ=0.07 (low → sharper distribution → harder negatives)
Temperature effect:
  Low τ → softmax concentrates on top prediction → easier to push apart
  High τ → softer distribution → model must work harder
```

### Step 4: Cross-Entropy Loss (InfoNCE)

```
CLIP uses InfoNCE loss = cross-entropy in BOTH directions:
  Image→Text: "given image i, which text matches?"
  Text→Image: "given text j, which image matches?"

Image→Text direction (row-wise softmax):
  For I₁ (invoice image), correct text = T₁ (index 0)
    softmax([13.16, 0.43, 5.86, 5.43]):
    exp(13.16) = 513,340
    exp(0.43)  = 1.537
    exp(5.86)  = 350.7
    exp(5.43)  = 228.0
    sum = 513,920.2

    P(T₁|I₁) = 513,340 / 513,920 = 0.9999
    L_I1 = -log(0.9999) = 0.0001

  For I₂ (dog), correct text = T₂ (index 1)
    softmax([0.57, 12.73, 0.86, 0.71]):
    P(T₂|I₂) = exp(12.73) / sum = 0.9975
    L_I2 = -log(0.9975) = 0.0025

  For I₃ (contract), correct text = T₃:
    P(T₃|I₃) = exp(12.583) / sum = 0.9970
    L_I3 = 0.0030

  For I₄ (receipt), correct text = T₄:
    P(T₄|I₄) = 0.9983
    L_I4 = 0.0017

  L_image→text = (L_I1 + L_I2 + L_I3 + L_I4) / 4
               = (0.0001 + 0.0025 + 0.0030 + 0.0017) / 4
               = 0.0021

Text→Image direction (column-wise softmax):
  Same calculation but transposed: for each text, find the correct image.
  By symmetry of the example: L_text→image = 0.0021

Total CLIP loss = (L_image→text + L_text→image) / 2
               = (0.0021 + 0.0021) / 2
               = 0.0021

Note: this is after ~10K steps of training. Early training has much higher loss.
Random init → P(correct) = 1/N = 0.25 for N=4 → loss = -log(0.25) = 1.386
```

---

## What Makes a Hard Negative

```
The invoice vs contract confusion (S=0.41) is a "hard negative":
  Both are business documents → similar visual features
  Without many such pairs, model wouldn't learn to distinguish them.

CLIP handles this through massive batch size (N=32,768 in paper):
  Each image is compared against 32,767 negative texts
  Many hard negatives naturally included in a batch of 32K
  → Forces very fine-grained discrimination

This is why CLIP uses TPU pods with enormous batch sizes.
Smaller batch → fewer negatives → easier task → weaker representations.
```

---

## Part 3: Zero-Shot Classification — Dry Run

### Task: Classify document type without fine-tuning

```
Classes: ["invoice", "contract", "receipt", "purchase order", "bank statement"]

Step 1: Create text prompts for each class
  "a photo of an invoice"
  "a photo of a contract"
  "a photo of a receipt"
  "a photo of a purchase order"
  "a photo of a bank statement"

Step 2: Encode all class prompts → text embeddings
  T_invoice       = text_encoder("a photo of an invoice")
  T_contract      = text_encoder("a photo of a contract")
  T_receipt       = text_encoder("a photo of a receipt")
  T_purchase_order = text_encoder("a photo of a purchase order")
  T_bank_statement = text_encoder("a photo of a bank statement")

Step 3: Encode the test image
  I_test = image_encoder(unknown_document)

Step 4: Compute similarity to each class
  S = [cosine_sim(I_test, T_class) for T_class in class_embeddings]

Results for an invoice image:
  S("invoice")         = 0.87   ← highest
  S("contract")        = 0.41
  S("receipt")         = 0.29
  S("purchase order")  = 0.52
  S("bank statement")  = 0.21

Step 5: Softmax → class probabilities
  P(invoice)        = 0.71
  P(contract)       = 0.08
  P(receipt)        = 0.05
  P(purchase order) = 0.14
  P(bank statement) = 0.02

Prediction: invoice (71% confidence) → CORRECT

No fine-tuning needed. No labeled examples needed.
Works because CLIP saw many invoice images with their descriptions during pretraining.
```

### Prompt Engineering for CLIP

```
Prompt matters — CLIP is sensitive to exact wording.

Bad prompt: "invoice"
Better:     "a photo of an invoice"
Even better: "a scanned image of a business invoice document"

Ensemble prompting (OpenAI technique):
  Generate multiple prompts per class:
    ["a photo of an invoice", "a scanned invoice", "a business invoice document",
     "an invoice form with line items"]
  Average their text embeddings → single class embedding
  More robust than a single prompt.

Code:
  class_embeds = []
  for prompt in prompts:
      embed = text_encoder(prompt)
      class_embeds.append(embed)
  class_embeds = torch.stack(class_embeds).mean(0)
  class_embed = F.normalize(class_embeds, dim=-1)
```

---

## Part 4: Fine-Tuning Strategies

### When to Fine-Tune vs Zero-Shot

```
Use zero-shot when:
  - Classes well-represented in CLIP training data (common objects, scenes)
  - < 100 labeled examples available
  - Fast prototyping / exploration

Fine-tune when:
  - Specialized domain (medical images, satellite imagery, document types)
  - Need > 90% accuracy on specific classes
  - Large labeled dataset available (10K+ examples)
  - Domain shift from web images (e.g., industrial inspection)
```

### Strategy 1: Linear Probe (Fastest)

```
Freeze CLIP image encoder entirely.
Train only a linear layer on top of CLIP features.

image_features = clip.encode_image(images)  # frozen, no grad
logits = linear_layer(image_features)       # only this trained

Why works: CLIP features are already rich → linear classifier often sufficient
Trade-off: fast (< 1 hour), but can't adapt visual features to domain

Code:
  for param in clip.visual.parameters():
      param.requires_grad = False

  classifier = nn.Linear(512, num_classes)
  optimizer = AdamW(classifier.parameters(), lr=1e-3)
```

### Strategy 2: Fine-Tune Full Image Encoder

```
Unfreeze image encoder, freeze text encoder.
Fine-tune with small learning rate (don't destroy pre-trained features).

Learning rate schedule:
  Image encoder: lr = 1e-6  (very small — preserve pre-trained features)
  Classifier head: lr = 1e-4 to 1e-100+ larger — train from scratch)

Risk: catastrophic forgetting of pre-trained visual knowledge
Mitigation: use layer-wise learning rate decay
  Earlier layers: lr × 0.1 (close to input, general features — change less)
  Later layers:   lr × 1.0 (task-specific — change more)
```

### Strategy 3: Fine-Tune Both Encoders (Full CLIP)

```
Fine-tune both image and text encoders with contrastive loss
on domain-specific (image, text) pairs.

Use case: specialized domain with descriptive text available
  Documents: (invoice_image, "Invoice from Acme Corp, total $1250, date 2026-03-14")
  Medical:   (xray_image, "Chest X-ray showing bilateral pneumonia")

Loss: same contrastive (InfoNCE) loss as pre-training
  but with small learning rate (2e-6) and domain data

Result: adapted embedding space that understands domain-specific language
```

### Strategy 4: LoRA on CLIP

```
Apply LoRA to the attention layers of the image encoder.
Freeze most weights, train only low-rank ΔW = A·B.

Benefits over full fine-tuning:
  - 0.1–0.5% of parameters trained (vs 100%)
  - Less catastrophic forgetting
  - Multiple LoRA adapters for different domains (swap at inference)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # attention matrices
    lora_dropout=0.1,
)
clip_model = get_peft_model(clip.visual, lora_config)
clip_model.print_trainable_parameters()
# trainable params: 1.2M || all params: 86M || trainable%: 1.39%
```

---

## Part 5: CLIP for Document AI

### Document Classification Pipeline

```python
import torch
import clip
from PIL import Image

device = "cuda"
model, preprocess = clip.load("ViT-B/32", device=device)

# Class labels with engineered prompts
doc_classes = {
    "invoice":          ["a scanned invoice document", "a business invoice with line items"],
    "contract":         ["a legal contract document", "a signed agreement document"],
    "receipt":          ["a receipt from a store or restaurant", "a payment receipt"],
    "purchase_order":   ["a purchase order", "a procurement document"],
    "bank_statement":   ["a bank statement showing transactions", "a financial statement"],
}

# Encode class embeddings (do once, cache)
class_embeddings = {}
with torch.no_grad():
    for class_name, prompts in doc_classes.items():
        text_tokens = clip.tokenize(prompts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        class_embeddings[class_name] = text_feats.mean(0)  # average prompts

# Inference
def classify_document(image_path: str) -> dict:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Compute similarity to each class
    similarities = {}
    for class_name, class_emb in class_embeddings.items():
        sim = (image_features @ class_emb.unsqueeze(1)).item()
        similarities[class_name] = sim

    sims = torch.tensor(list(similarities.values()))
    probs = torch.softmax(sims * 100, dim=0)  # scale before softmax

    return {
        class_name: prob.item()
        for class_name, prob in zip(similarities.keys(), probs)
    }

result = classify_document("unknown_doc.jpg")
# {"invoice": 0.71, "contract": 0.09, "receipt": 0.05, ...}
```

### Image-Text Retrieval for Document Search

```python
# Use case: "find all invoices from Acme Corp with amount > $1000"
# CLIP handles the visual + semantic part

def build_document_index(image_paths: list) -> torch.Tensor:
    """Encode all document images into CLIP embedding space."""
    all_embeddings = []
    for path in image_paths:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb)
    return torch.cat(all_embeddings, dim=0)  # (N, 512)

def search_documents(query: str, index: torch.Tensor, top_k=5) -> list:
    """Find documents matching a text query."""
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    similarities = (index @ text_emb.T).squeeze()
    top_indices = similarities.topk(top_k).indices.tolist()
    return top_indices

# Usage
index = build_document_index(image_paths)
results = search_documents("invoice with large total amount", index, top_k=5)
```

---

## Part 6: Interview Questions

**Q: Walk me through the CLIP contrastive loss.**

CLIP creates an N×N similarity matrix where N is batch size. Row i = image i compared to all N texts. Column j = text j compared to all N images.

The diagonal contains matching pairs (correct answers). We apply cross-entropy loss row-wise (image→text direction): "given image i, which text j is the match?" → correct answer = diagonal[i]. And column-wise (text→image direction). Total loss = average of both directions.

The key ingredient: large N (32,768 in original paper). Large batch → many negative pairs → harder task → better representations. Temperature τ sharpens/softens the softmax — CLIP learns τ jointly.

---

**Q: How does CLIP enable zero-shot classification?**

During pre-training on 400M image-text pairs, CLIP learns that "a photo of a dog" is close to dog images, "a receipt" is close to receipt images, etc.

At inference: create one text embedding per class ("a photo of X"), then find which class embedding is closest to the test image embedding. No labeled examples needed — the knowledge comes from pre-training.

Zero-shot works well when the class was well-represented in training data. Fails for rare/specialized domains (satellite imagery, medical scans, documents).

---

**Q: What's the difference between fine-tuning CLIP vs training a classifier on CLIP features?**

Linear probe (classifier on frozen features):
- Train only a linear layer on top of frozen CLIP image features
- Fast, no risk of destroying pre-trained knowledge
- Works well when domain is similar to CLIP training data
- Limited: can't adapt visual representations

Full fine-tuning:
- Unfreeze image encoder, update with small LR
- Can adapt visual features to domain (e.g., learn to focus on text in documents)
- Risk: catastrophic forgetting if LR too high or dataset too small
- Use layer-wise LR decay: early layers change less than later layers

LoRA on CLIP: best of both — adapt with < 2% of parameters, less forgetting.

---

**Q: Why does CLIP use such a large batch size?**

The contrastive loss requires negative examples. Each image is compared against N-1 texts in the batch. More negatives → harder task → model must learn finer distinctions → better representations.

With N=256: each image has 255 negatives — easy to distinguish. With N=32,768: each image has 32,767 negatives — must distinguish very similar concepts.

The "hardness" of negatives determines representation quality. Large batch approximates having all possible negatives in every update.

Practical implication: reproducing CLIP from scratch requires TPU pods. Most practitioners fine-tune the pre-trained CLIP rather than pre-train from scratch.

---

## Key Takeaway

```
CLIP = dual encoder (image ViT + text transformer) trained with contrastive loss.

Forward pass:
  Image → 49 patches → 12-layer ViT → [CLS] → project → 512-dim L2-normalized vector
  Text  → tokenize (77 tokens) → 12-layer causal transformer → [EOS] → project → 512-dim vector

Contrastive loss (InfoNCE):
  N×N similarity matrix = image_embeddings · text_embeddings^T
  Scale by temperature τ (learned, init=1/0.07=14.3)
  Cross-entropy row-wise (image→text) + column-wise (text→image)
  Large N = many negatives = key to representation quality

Zero-shot: encode class names as text → cosine sim to image → softmax → class

Fine-tuning strategies (weakest → strongest):
  Linear probe < full image encoder < full both encoders + LoRA

Document AI use cases:
  Zero-shot doc classification (5 classes, no labels needed)
  Document retrieval by text query
  Backbone features for LayoutLM-style models
```
