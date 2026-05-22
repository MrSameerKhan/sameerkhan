# 01 — Deep Learning Foundations

## Quick Reference (30-sec scan)

- **Neuron:** y = f(Wx + b) — weighted sum → bias → activation
- **Layer:** many neurons in parallel, vectorized as matrix ops
- **Activation:** adds non-linearity — without it, deep net = one linear transform
- **Loss:** measures error per sample; **Cost** = average loss over dataset
- **Forward pass:** data flows left → right, intermediate values saved for backprop
- **Default:** ReLU for hidden layers, Softmax for multi-class output, BCE for binary

---

## 1. The Neuron

```
$$y = f(Wx + b)$$
```

| Component | Role | Gotcha |
|-----------|------|--------|
| W (weight) | How much each input matters | Initialized randomly, not zero |
| b (bias) | Shifts activation threshold | Without bias, neuron can't fire at zero input |
| f (activation) | Adds non-linearity | Without it, all layers collapse to one linear map |

**Perceptron (1958)** → step function, not differentiable → can't use gradient descent. **Modern neuron** → smooth activation → differentiable → backprop works.

---

## 2. From Neuron to Layer

A layer = N neurons operating in parallel on the same input.

```
$$\mathbf{y} = f(W\mathbf{x} + \mathbf{b})$$

- W: matrix [neurons × inputs]
- b: vector [neurons]
- y: vector [neurons]
```

This is a single matrix multiply — fully parallelizable on GPU.

---

## 3. Activation Functions

**Why needed:** Without activations, any deep network collapses to `y = W_total × x` — just one linear transform, regardless of depth.

| Activation | Formula | Range | Problem | Use |
|-----------|---------|-------|---------|-----|
| Sigmoid | 1/(1+e^-x) | (0,1) | Vanishing gradients (max deriv = 0.25) | Binary output only |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | (-1,1) | Vanishing gradients | RNNs, older nets |
| ReLU | max(0,x) | [0,∞) | Dying neurons (grad=0 for x≤0) | Hidden layers (default) |
| Leaky ReLU | x if x>0, else αx | (-∞,∞) | Small negative slope | When dying ReLU is an issue |
| Softmax | e^xi / Σe^xj | (0,1), sum=1 | Saturation | Multi-class output |
| GELU | Smooth ReLU approx | (-∞,∞) | Computationally heavier | Transformers (BERT, GPT-2/3) |
| Swish/SiLU | x · σ(x) | (-∞,∞) | Smooth; no zero-gradient region | Modern CV (EfficientNet); building block of SwiGLU |
| SwiGLU (gated) | Swish(W1x) ⊙ (W2x then W_3) | (-∞,∞) | 3 matmuls instead of 2 | LLaMA, Mistral, Gemma, PaLM FFNs — see `05_modern_components.md` |
| GeGLU (gated) | GELU(W1x) ⊙ (W2x then W_3) | (-∞,∞) | Same shape as SwiGLU | T5 v1.1 and several modern LLMs |

**Softmax toy example:**

```python
Logits: [5, 2, -1, 3]
Exp:    [148.4, 7.4, 0.4, 20.1]  sum=176.3
Output: [0.842, 0.042, 0.002, 0.114]  ← probability distribution
```

**Numerical stability:** always compute `softmax(x - max(x))` to prevent overflow.

---

## 4. Loss Functions

**Loss** = error for one sample. **Cost** = average loss over dataset/batch.

| Task | Loss | Formula | Output Activation |
|------|------|---------|------------------|
| Regression | MSE | (y - ŷ)² | Linear |
| Regression (outliers) | MAE | \|y - ŷ\| | Linear |
| Binary classification | BCE | -[y·log(ŷ) + (1-y)·log(1-ŷ)] | Sigmoid |
| Multi-class | CCE | -Σ y·log(ŷ_c) | Softmax |

**Why log in cross-entropy?** Log heavily punishes confident wrong predictions:

```
y=1, ŷ=0.9 → loss = -log(0.9) = 0.10  (correct, small penalty)
y=1, ŷ=0.1 → loss = -log(0.1) = 2.30  (wrong, large penalty)
```

**Why not MSE for classification?** MSE treats errors as quadratic — not aligned with probability outputs. Cross-entropy + softmax gradient simplifies to `ŷ - y`, which is clean and stable.

---

## 5. Forward Pass

Data flows **left → right** through each layer.

```
Input x
  → z1 = W1·x + b1   (linear transform)
  → a1 = f(z1)        (activation)
  → z2 = W2·a1 + b2
  → a2 = f(z2)
  → ŷ = output
  → L = loss(ŷ, y)
```

**Key:** intermediate values (z, a) are **cached** during forward pass — backprop needs them for gradient computation.

During inference: no caching needed — use `torch.no_grad()` to save memory.

---

## When to Use What

| Situation | Choice |
|-----------|--------|
| Hidden layer activation | ReLU (default) |
| Dying neurons in hidden layers | Leaky ReLU |
| Transformer hidden layers | GELU |
| Binary classification output | Sigmoid + BCE |
| Multi-class output | Softmax + CCE |
| Regression output | Linear + MSE or MAE |
| Outliers in regression data | MAE over MSE |

---

## Gotchas

**1. All-zero weight initialization = symmetry problem.** Every neuron gets identical gradients and learns the same features. Always initialize randomly.

**2. Sigmoid in hidden layers = vanishing gradient.** Max derivative is 0.25. In a 10-layer net: 0.25^10 = 0.000001. Early layers receive near-zero gradient updates. Use ReLU instead.

**3. ReLU neurons can permanently die.** If a neuron's pre-activation is always negative (e.g., large negative bias or LR too high), it outputs 0 forever and receives zero gradients. Monitor dead neuron fraction during training.

**4. Softmax with one class = not sigmoid.** A common mistake: using sigmoid when you have 2 classes. You should use softmax with 2 outputs OR sigmoid with 1 output — they're equivalent but don't mix up the loss function.

**5. Log(0) = -∞ in cross-entropy.** If a model is extremely confident and wrong, loss becomes infinite. Always clip predictions: `ŷ = clip(ŷ, 1e-7, 1-1e-7)`.

---

## Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All neurons output same value | Zero/identical initialization | Use random init (He or Xavier) |
| Loss is NaN from step 1 | Log(0) in cross-entropy | Clip predictions, check for zero outputs |
| Loss never decreases | Dying ReLU in all neurons | Check init, reduce LR, try Leaky ReLU |
| Loss stuck at high value | Wrong activation for task | Check output activation matches loss function |
| Training fast but poor generalization | Too many parameters, no regularization | Add dropout/L2, reduce model size |

---

## Interview Q&A

**Q: Why can't you just stack linear layers without activations?**

No matter how many linear layers you stack, the composition is still a single linear transformation: W[N](W[N-1]...W[1] × x) = W_total × x. The network has no more expressive power than a single layer. Activations introduce non-linearity, making depth meaningful — each layer can learn increasingly abstract representations.

**Q: Why is ReLU preferred over sigmoid for hidden layers?**

Sigmoid's derivative maxes out at 0.25, causing vanishing gradients in deep networks. ReLU's derivative is 1 for all positive inputs, so gradients propagate without shrinkage. ReLU is also computationally cheaper (no exponential) and induces sparsity — on average 50% of neurons output zero, which is computationally efficient.

**Q: When would you use MAE over MSE for a regression task?**

When the target variable has outliers. MSE squares the error, so outliers (e.g., a house price 10× the median) get 100× more influence on the loss. MAE treats all errors linearly, making it robust to outliers. Huber loss is a middle ground — MAE for large errors, MSE for small ones.

**Q: Why does BCE use log?**

Log creates a convex loss landscape for probability-based outputs, making optimization well-behaved. More importantly, it severely penalizes confident wrong predictions — log(0.01) = -4.6 — which pushes the model to be both correct and well-calibrated.

---

## Connections

- Leads to: `02_training_loop.md` — forward pass produces ŷ and loss; backprop propagates gradients back
- Leads to: `03_training_stability.md` — activation choice (ReLU vs sigmoid) directly causes vanishing/exploding gradients
- Relevant in: `05_modern_components.md` — attention uses softmax; transformers use GELU
- **If things break:** check activation choice → then initialization → then loss function match

---

## Key Takeaway

```
Neuron:        y = f(Wx + b)
Layer:         vectorized neurons operating in parallel
Activation:    non-linearity — required for depth to be meaningful
Loss:          signal that tells the network how wrong it is
Forward pass:  data flows right, intermediate values cached for backprop
```
