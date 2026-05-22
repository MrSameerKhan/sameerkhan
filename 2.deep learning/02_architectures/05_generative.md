# 05 — Generative Models (VAE, GAN, Diffusion)

## Quick Reference

| Model | Core Idea | Loss | Strength | Weakness |
|-------|-----------|------|----------|---------|
| VAE | Encode to latent distribution, decode | ELBO = Reconstruction + KL | Stable training, smooth latent space | Blurry outputs |
| GAN | Generator vs Discriminator adversarial game | Minimax / Wasserstein | Sharp, photorealistic outputs | Mode collapse, unstable training |
| Diffusion | Learn to reverse Gaussian noise addition | Noise prediction MSE | Best quality, diverse outputs | Slow inference (many denoising steps) |

**One-line summary:** VAE = encode to distribution → sample → decode. GAN = fool a discriminator. Diffusion = learn to denoise from pure noise step by step.

---

## 1. Variational Autoencoder (VAE)

### Architecture

```
Input x
    ↓
Encoder (CNN/MLP)
    ↓
μ, log_σ² = two separate linear heads
    ↓
z = μ + σ·ε   where ε ~ N(0,1)  ← reparameterization trick
    ↓
Decoder (CNN/MLP)
    ↓
x̂ (reconstruction)
```

### Loss: ELBO (Evidence Lower BOund)

```
L = E[log p(x|z)]  -  KL(q(z|x) || p(z))
  = Reconstruction  -  KL divergence penalty
```

- **Reconstruction term:** how well decoder recovers input (BCE for binary, MSE for continuous)
- **KL term:** forces encoder distribution q(z|x) = N(μ,σ²) to stay close to prior N(0,1)
- **KL in closed form:** KL = -½ Σ(1 + log_σ² - μ² - σ²)

### Reparameterization Trick

Standard sampling z ~ N(μ, σ²) is not differentiable. Fix: **z = μ + σ·ε** where ε ~ N(0,1) — gradient flows through μ and σ; ε is just noise.

### KL Annealing

Start training with KL weight = 0, gradually increase to 1. Without annealing → posterior collapse (encoder ignores x, outputs prior N(0,1), decoder learns mean).

### When to Use VAE

- Smooth, interpolable latent space needed (e.g., morphing between samples)
- Downstream: latent space for downstream classification
- When training stability > output sharpness
- Anomaly detection: score by reconstruction error + KL

---

## 2. Generative Adversarial Network (GAN)

### Core Idea

Two networks trained adversarially:
- **Generator G:** maps noise z ~ N(0,1) → fake samples G(z)
- **Discriminator D:** distinguishes real x from fake G(z), outputs P(real)

### Minimax Objective

```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

- D maximizes: correctly classify real as 1, fake as 0
- G minimizes: make D classify G(z) as 1 (fool D)

In practice, G minimizes **-log D(G(z))** (non-saturating loss) instead of log(1-D(G(z))) — avoids vanishing gradients early when D is confident.

### Training Loop

```python
for each batch:
    # Train D
    real_loss = BCE(D(real_x), 1)
    fake_loss = BCE(D(G(z).detach()), 0)
    d_loss    = (real_loss + fake_loss) / 2
    d_optimizer.step()

    # Train G
    g_loss = BCE(D(G(z)), 1)   # want D to output 1 for fakes
    g_optimizer.step()
```

### Key GAN Variants

| Variant | Problem Solved | Key Change |
|---------|---------------|------------|
| DCGAN | Stable image generation | CNN-based G and D, BatchNorm |
| WGAN | Mode collapse, training instability | Wasserstein distance, gradient penalty (WGAN-GP) |
| StyleGAN | High-quality faces, style control | Style-based generator, AdaIN, progressive growing |
| CycleGAN | Unpaired image translation | Cycle consistency loss (no paired data needed) |
| Conditional GAN (cGAN) | Controlled generation | Condition G and D on class label y |

**Wasserstein GAN (WGAN-GP):** Replace JS divergence with Wasserstein distance — smoother gradients, better training signal. Gradient penalty (WGAN-GP): penalize `||∇D(x̃)||₂ = 1` constraint (Lipschitz condition).

---

## 3. Diffusion Models (DDPM)

### Core Idea

**Forward process:** gradually add Gaussian noise to data over T steps until x_T ~ N(0,I).
**Reverse process:** learn a neural network (UNet) to predict and remove the noise step by step.

### Forward Process (Noise Schedule)

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)

β_t: noise schedule (linear 1e-4 → 0.02, or cosine schedule)

Key property — can sample x_t directly from x_0:
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε   where ε ~ N(0,1)
ᾱ_t = Π(1-β_s) for s=1..t
```

### Reverse Process (What the Network Learns)

At inference, start from x_T ~ N(0,I), iteratively denoise:

```
x_{t-1} = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t)) + σ_t · z

ε_θ(x_t, t): UNet that predicts the noise added at step t
```

### Training Objective (DDPM)

```
L = E_{x,ε,t}[ ||ε - ε_θ(x_t, t)||² ]
```

Just predict the noise — simple MSE. UNet is conditioned on timestep t (via sinusoidal embedding).

### Architecture: UNet with Attention

```
Input (noisy image x_t) + timestep embedding
    ↓
Encoder: [Conv + ResBlock + Downsample] × N
    ↓
Bottleneck: ResBlock + Self-Attention
    ↓
Decoder: [Upsample + ResBlock + Cross-Attention (for text conditioning)] × N
    ↓
Output: predicted noise ε
```

Cross-attention in decoder: keys/values from text encoder (CLIP/T5), queries from UNet features — enables text-conditional generation (Stable Diffusion).

### Sampling Speed

- **DDPM:** T=1000 steps. Slow.
- **DDIM:** deterministic sampler, 50-200 steps, same quality.
- **LCM/Turbo:** distilled to 1-4 steps (knowledge distillation from full model).

### Modern Variants (2023-2025) — Flow Matching, Consistency, Rectified Flow

DDPM defines a stochastic noising chain. **Flow Matching** generalizes this to a **continuous-time vector field**: learn v_θ(x_t, t) such that integrating it from noise to data produces samples. Cleaner formulation, often better quality at the same compute.

| Method | Year | Idea | Where used |
|--------|------|------|------------|
| Score-based generative models | 2019-2020 (Song) | Learn ∇log p(x) directly; SDE/ODE solvers for sampling | Unified view; basis of modern diffusion |
| Flow Matching | 2023 (Lipman et al., Meta) | Regress to a target probability flow ODE | SD3 base, FLUX.1 |
| Rectified Flow | 2023 (Liu et al.) | Straighten the trajectory between data and noise — fewer steps | SD3 base, FLUX.1 |
| Consistency Models | 2023 (Song et al., OpenAI) | Self-consistency loss — 1-4 step sampling from pretrained diffusion | LCM, Latent Consistency Models |
| Diffusion Transformer (DiT) | 2022-23 (Peebles & Xie) | Replace UNet with a transformer backbone — scales like LLMs | SD3, SORA, Stable Video 4D |
| Classifier-Free Guidance (CFG) | 2022 | Train conditional + unconditional at sample time, extrapolate | Universal in modern image diffusion |

### ControlNet and Conditional Diffusion

Beyond text prompts, **ControlNet** (Zhang et al., 2023) adds spatial control to a frozen diffusion model: condition on edge maps, depth, pose, segmentation, scribbles — without retraining the base model.

```
ControlNet =
    copy of the UNet encoder, weights initialized from base model
    zero-initialized zeros in each output to base UNet (training starts as identity)
    trained on (condition, image) pairs while base UNet stays frozen
```

Used everywhere generative models need spatial layout control: poster generation, document synthesis, virtual try-on, layout-conditioned image generation. For document automation: condition on bounding-box layouts to synthesize training documents with controlled field positions.

Related: IP-Adapter (image prompts), LoRA adapters (style/character), T2I-Adapter (lighter than ControlNet) — all instances of the "freeze base diffusion + train small adapter" pattern.

---

## 4. VAE vs GAN vs Diffusion Comparison

| Aspect | VAE | GAN | Diffusion |
|--------|-----|-----|-----------|
| Output quality | Blurry | Sharp | Sharpest |
| Training stability | Stable | Unstable (mode collapse) | Stable |
| Latent space | Smooth, interpretable | Unstructured | No explicit latent (x_T is latent) |
| Inference speed | Fast (single forward pass) | Fast (single forward pass) | Slow (T denoising steps) |
| Diversity | Good | Poor (mode collapse) | Excellent |
| Text conditioning | Add to decoder | cGAN | Cross-attention in UNet |
| Current SOTA | Legacy (mostly) | Legacy (mostly) | Image generation (SD, DALL-E 3) |
| Still used for | Latent space learning, anomaly detection | Video (some), domain adaptation | Image/video/audio generation |

---

## 5. When to Use What

| Task | Model | Why |
|------|-------|-----|
| Image generation (highest quality) | Diffusion (Stable Diffusion) | SOTA quality + diversity |
| Fast real-time generation | GAN (StyleGAN) | Single forward pass |
| Anomaly detection | VAE | Reconstruction error = anomaly score |
| Image-to-image translation (unpaired) | CycleGAN | Cycle consistency, no paired data |
| Smooth interpolation in latent space | VAE | Latent space is structured N(0,1) |
| Text-to-image | Diffusion + CLIP/T5 | Cross-attention conditions on text |
| Feature learning for downstream tasks | VAE encoder | Extract compressed representations |

---

## 6. Gotchas

**1. GAN: Mode collapse.** Generator learns to produce a few high-scoring samples, ignores diversity. D can't distinguish → G keeps repeating. Fix: Wasserstein loss + gradient penalty (WGAN-GP); minibatch discrimination; diverse noise.

**2. GAN: Training instability.** D becomes too strong → G gradients vanish (log(1-D(G(z))) → 0). Fix: non-saturating loss, balance D/G update steps, lower D LR.

**3. VAE: Posterior collapse.** KL term dominates → encoder outputs prior N(0,1) regardless of input → decoder ignores latent z. Fix: KL annealing (β-VAE), free bits, gradient blocking on KL.

**4. VAE: Blurry outputs.** MSE reconstruction loss encourages averaging over modes. Fix: perceptual loss (VGG features), VQ-VAE (discrete latent space, used in DALL-E 1).

**5. Diffusion: Slow inference is a real cost.** 50 DDIM steps × UNet forward pass = significant latency. In production: use quantized UNet (FP16/INT8), SDXL-Turbo (4 steps). Don't benchmark with T=1000.

**6. Diffusion: Guidance scale.** CFG scale controls quality vs diversity. Scale=1: unconditional. Scale=7-10: sharp, prompt-following. Scale>15: oversaturated, artifacts. Tune per use case.

**7. All generative models: Evaluation is hard.** FID (Fréchet Inception Distance) is the standard but imperfect. Low FID ≠ useful outputs. Supplement with: LPIPS (perceptual similarity), CLIP score (text-image alignment), human eval.

---

## 7. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| GAN loss oscillates wildly | LR too high or D/G imbalance | Reduce LR; use TTUR (lower LR for G than D) |
| GAN generates same image for all inputs | Mode collapse | Switch to WGAN-GP; add noise to D inputs |
| VAE reconstructions are blurry | KL too strong relative to reconstruction | Reduce KL weight β; use perceptual loss |
| VAE latent space not smooth | KL weight β too small | Increase β (β-VAE) |
| Diffusion: gray/washed outputs | CFG scale too low | Increase guidance_scale to 7-10 |
| Diffusion: oversaturated/artifacts | CFG scale too high | Reduce to 5-8 |
| Diffusion: training loss not decreasing | Noise schedule mismatch | Use cosine schedule; verify x_t formula |
| GAN discriminator accuracy ≈ 100% | D too strong, G can't learn | Add noise to D inputs; reduce D capacity |

---

## 8. Code Reference

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── VAE (PyTorch) ───────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 400), nn.ReLU())
        self.mu      = nn.Linear(400, latent_dim)
        self.log_var = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400), nn.ReLU(),
            nn.Linear(400, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std   # gradient flows through mu, std

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss

# ─── GAN (WGAN-GP pattern) ───────────────────────────────────────────────────
def gradient_penalty(D, real, fake, device):
    alpha        = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp     = D(interpolated)
    grad         = torch.autograd.grad(d_interp, interpolated,
                                       grad_outputs=torch.ones_like(d_interp),
                                       create_graph=True)[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# D Loss (Wasserstein + gradient penalty)
d_loss = D(fake).mean() - D(real).mean() + 10 * gradient_penalty(D, real, fake, device)
# G Loss (Wasserstein)
g_loss = -D(G(z)).mean()

# ─── Using Stable Diffusion (HuggingFace diffusers) ─────────────────────────
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    prompt="a document scanner photograph of a handwritten invoice",
    negative_prompt="blurry, low quality",
    num_inference_steps=30,    # DDIM steps
    guidance_scale=7.5,        # CFG scale
    height=512, width=512,
).images[0]

# ─── Sampling from VAE latent space ─────────────────────────────────────────
# Generate new samples (no input needed)
z       = torch.randn(10, latent_dim)   # sample from prior N(0,1)
samples = model.decoder(z)              # decode to image space

# Interpolate between two samples
z1, z2 = model.encode(x1)[0], model.encode(x2)[0]
alphas  = torch.linspace(0, 1, 10)
interpolations = [model.decoder(alpha * z1 + (1-alpha) * z2) for alpha in alphas]
```

---

## 9. Interview Q&A (Senior Level)

**Q: Why does GAN training fail and what's the fundamental cause?**

The core issue is that minimizing JS divergence between real and generated distributions leads to vanishing gradients when the distributions don't overlap (common early in training — generator produces garbage, discriminator is 100% confident). The gradient of log(1-D(G(z))) → 0 when D(G(z)) → 0. WGAN fixes this by replacing JS divergence with Wasserstein distance (Earth Mover's distance), which provides useful gradients even when distributions don't overlap, because it measures the "cost" of transporting mass between them.

**Q: Explain the reparameterization trick and why it's necessary.**

To train the VAE encoder to output good μ and σ, we need gradients to flow through the sampling step z ~ N(μ, σ²). Sampling is a stochastic operation — not differentiable. The trick: instead of sampling z directly, write **z = μ + σ·ε** where ε ~ N(0,1) is sampled independently. Now gradients flow through μ and σ (deterministic operations), while ε is just external randomness. This enables backprop through the latent sampling step.

**Q: What is posterior collapse in VAEs and how do you fix it?**

Posterior collapse happens when the KL term overwhelms the reconstruction term, causing the encoder to output the prior q(z|x) = N(0,1) regardless of input x. The decoder then ignores z and learns to generate the dataset mean. Fix: (1) KL annealing — start with KL weight β=0, increase to 1 over training. (2) Free bits: only penalize KL above a threshold per dimension. (3) β-VAE — lower β to reduce KL pressure. (4) VQ-VAE — quantize latent space so encoder must use it.

**Q: Why did diffusion models overtake GANs as SOTA for image generation?**

Three reasons: (1) **Training stability** — no adversarial game; simple MSE on noise prediction is easy to optimize. (2) **Mode coverage** — diffusion explores the full data distribution; GANs have mode collapse. (3) **Composability** — classifier-free guidance lets you condition on any signal (text, class, image) via cross-attention without retraining. GANs needed architecture changes per conditioning type. The tradeoff is inference speed (100s of forward passes vs 1) but distillation (LCM, SDXL-Turbo) narrows this gap.

**Q: What is classifier-free guidance and how does it work?**

CFG is a technique to improve text-image alignment in diffusion without a separate classifier. During training, randomly drop the text conditioning (set to null) ~10-20% of the time. At inference, run the UNet twice — once conditioned on prompt (ε_cond) and once unconditional (ε_uncond). Final prediction: ε = ε_uncond + s·(ε_cond - ε_uncond). Scale s (guidance_scale) controls the trade-off: higher s = more prompt-following but less diversity/realism. This amplifies the "direction" in noise-prediction space that corresponds to the prompt.

**Q: How would you use a generative model for data augmentation in a document understanding pipeline?**

(1) **Diffusion for synthetic data** — generate realistic invoice/receipt images with controlled layout using ControlNet (conditions on edge maps). (2) **CycleGAN for domain adaptation** — convert clean scanned docs to noisy/degraded versions without paired data. (3) **VAE for anomaly detection** — train VAE on normal documents; anomalous documents have high reconstruction error + KL divergence. (4) Caution: always validate synthetic data doesn't introduce distribution shift. Measure with held-out real test set only.

---

## 10. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| Cross-attention in diffusion UNet | `04_transformer.md` | Cross-attention mechanism (Q from image, K/V from text) |
| VAE latent space | `04_transformer.md` | VQVAE-2, DALL-E — discrete latent + transformer |
| Loss functions (ELBO, Wasserstein) | `../01_fundamentals/01_foundations.md` | BCE/MSE building blocks used in VAE loss |
| BatchNorm in DCGAN | `../01_fundamentals/03_training_stability.md` | BN in generator; spectral norm in discriminator |
| KL divergence | `../01_fundamentals/01_foundations.md` | Information-theoretic perspective on loss functions |
| Diffusion for document synthesis | Your domain | ControlNet on layout → synthetic training data for OCR |

---

## Key Takeaway

```
VAE       = stable training, interpretable latent space, blurry outputs
            → anomaly detection, latent feature learning

GAN       = sharp outputs, hard to train
            → largely replaced by diffusion for images, still used in video and domain adaptation

Diffusion = current SOTA for image generation, text-conditional via cross-attention
            → slow inference (mitigated by distillation)

For your domain:
  Diffusion → synthetic document generation + augmentation
  VAE       → anomaly detection on documents
  Know:       reparameterization trick, ELBO loss, CFG guidance scale
```
