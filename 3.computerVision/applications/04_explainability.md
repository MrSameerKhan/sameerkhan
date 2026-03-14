# 04 — Model Explainability (CAM, Grad-CAM, SHAP)

## Quick Reference

| Method | Model Required | Granularity | Cost | Best For |
|--------|---------------|-------------|------|---------|
| CAM | GAP + FC (specific arch) | Coarse heatmap | Low | Simple classifiers |
| Grad-CAM | Any CNN | Coarse heatmap | Low | General CNN explanations |
| Grad-CAM++ | Any CNN | Better multi-instance | Low | Multiple objects, finer maps |
| LIME | Any (black-box) | Superpixel importance | Medium | Model-agnostic |
| SHAP (DeepSHAP) | PyTorch/TF | Pixel-level | High | Consistent, theoretically grounded |
| Integrated Gradients | Differentiable models | Pixel-level | Medium | Fine-grained attribution |

**When to use:** Any time you need to justify a model's decision to a business stakeholder, debug unexpected predictions, detect bias, or validate that the model reasons correctly.

---

## 1. Why Explainability in CV

### Business Reasons
```
Medical AI: "Why did the model flag this X-ray as abnormal?"
  → Radiologist needs to verify the model focused on the right region

Autonomous driving: "Why did the model not brake for this pedestrian?"
  → Safety audit requires knowing what the model attended to

Document AI: "Why did the model extract wrong field value?"
  → Debug whether the model reads the right region on the document

Regulatory: GDPR Article 22 requires explanations for automated decisions
```

### Debugging Reasons
```
Model accuracy 95% but is it learning the right thing?

Classic failure: chest X-ray model achieves 90% accuracy
  → Explainability reveals: model focuses on hospital scanner watermarks
     (certain hospitals have higher disease prevalence → model learned the artifact)

Short-cut learning: model uses spurious correlation instead of actual signal
  Explainability reveals this before deployment
```

---

## 2. Class Activation Mapping (CAM)

**Original paper:** Zhou et al., 2016 — "Learning Deep Features for Discriminative Localization"

### How CAM Works
```
Requires specific architecture:
  CNN → Global Average Pooling (GAP) → FC → Softmax

GAP: each channel c becomes one number: f_c = mean(feature_map_c)
FC:  score for class k = Σ_c w_k_c × f_c

CAM for class k at position (x, y):
  M_k(x, y) = Σ_c w_k_c × A_c(x, y)

  where A_c(x,y) = activation of channel c at position (x,y)
        w_k_c    = weight connecting channel c to class k in FC

Upsampling M_k to input size → heatmap showing "where class k was detected"
```

### Why GAP is Required
```
Without GAP: FC layer receives flattened spatial info → spatial mapping destroyed
With GAP: FC weights w_k_c directly weight the importance of each channel c
          for class k → those weights × spatial feature maps = spatial class importance
```

```python
import torch
import torchvision.models as models
import numpy as np
import cv2

def get_cam(model, image_tensor, target_class):
    """
    model: must end with GAP → FC (e.g., ResNet without the final FC layer replaced)
    image_tensor: [1, 3, H, W]
    target_class: int
    """
    # Register hook to get feature maps before GAP
    feature_maps = []
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())

    # Hook the last conv layer
    model.layer4[-1].register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    # Get FC weights for target class
    fc_weights = model.fc.weight[target_class].cpu().numpy()   # [n_channels]

    # feature_maps[0]: [1, n_channels, H', W']
    feats = feature_maps[0].squeeze(0).cpu().numpy()   # [n_channels, H', W']

    # Weighted sum of feature maps
    cam = np.zeros(feats.shape[1:], dtype=np.float32)
    for i, w in enumerate(fc_weights):
        cam += w * feats[i]

    # ReLU (only positive contributions)
    cam = np.maximum(cam, 0)

    # Normalize and resize to input size
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (image_tensor.shape[-1], image_tensor.shape[-2]))

    return cam

# Overlay heatmap on image
def overlay_cam(image_np, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (1-alpha) * image_np + alpha * heatmap / 255.0
```

**CAM Limitation:** Only works for networks with GAP + single FC before softmax. VGG (with multiple FC layers) can't use CAM directly.

---

## 3. Grad-CAM (Gradient-weighted Class Activation Mapping)

**Selkoet al., 2017 — works with any CNN architecture, any layer.**

### How Grad-CAM Works
```
Instead of using FC weights (CAM), use gradients as importance weights.

For target class c and feature maps A^k from the last conv layer:

1. Forward pass → get class score y^c
2. Backward pass → get gradients ∂y^c/∂A^k_ij at each spatial position (i,j)

3. Importance weight for each channel k:
   α^c_k = (1/Z) ΣᵢΣⱼ (∂y^c / ∂A^k_ij)   ← Global Average Pooling of gradients

4. Grad-CAM heatmap:
   L^c_Grad-CAM = ReLU(Σ_k α^c_k · A^k)
   ReLU: only keep features with positive influence on class c
```

```python
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class=None):
        self.model.eval()

        # Forward pass
        output = self.model(image_tensor)   # [1, num_classes]
        if target_class is None:
            target_class = output.argmax(1).item()

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        # α_k = mean of gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]

        # Weighted sum of activations + ReLU
        cam = (weights * self.activations).sum(dim=1, keepdim=True)   # [1, 1, H', W']
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input
        cam = F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
            size=image_tensor.shape[-2:],
            mode='bilinear', align_corners=False
        ).squeeze().numpy()

        return cam, target_class

# Usage
import torchvision.models as models
model = models.resnet50(weights='IMAGENET1K_V2')
target_layer = model.layer4[-1].conv2   # last conv layer

grad_cam = GradCAM(model, target_layer)
cam, pred_class = grad_cam.generate(image_tensor)
```

### Using pytorch-grad-cam Library (Recommended)
```python
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model = models.resnet50(weights='IMAGENET1K_V2').eval()
target_layers = [model.layer4[-1]]

# Multiple CAM variants via same API
with GradCAM(model=model, target_layers=target_layers) as cam:
    targets = [ClassifierOutputTarget(281)]   # class 281 = tabby cat
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]   # [H, W]

# Overlay on image
visualization = show_cam_on_image(
    img=np.float32(image_np) / 255.0,
    mask=grayscale_cam,
    use_rgb=True
)
```

---

## 4. Grad-CAM++

Addresses Grad-CAM's weakness with multiple instances of the same class.

```
Problem: If 3 cats in image, Grad-CAM highlights only the most prominent one.

Grad-CAM++: uses higher-order derivatives (second derivative of score w.r.t. activations)
  to weight each spatial location within each channel more accurately

α^c_kij = exp(y^c) · (∂²y^c/∂A^kij²) / (2∂²y^c/∂A^kij² + A^k_ij · ∂³y^c/∂A^kij³)

Result: better handles multiple instances of same class
```

```python
from pytorch_grad_cam import GradCAMPlusPlus
with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
```

---

## 5. LIME (Local Interpretable Model-agnostic Explanations)

**Model-agnostic:** works with any model (black box).

```
1. Take the original image, divide into superpixels (e.g., 50 patches)
2. Create many perturbed samples: randomly mask (zero out) different superpixels
3. Run each perturbed sample through the model → get prediction scores
4. Fit a simple linear model (LIME) to explain which superpixels drive the prediction
5. Visualize the top positive (green) and negative (red) superpixels
```

```python
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import numpy as np

def predict_fn(images):
    """Takes numpy [N, H, W, C], returns [N, num_classes]"""
    tensor = torch.FloatTensor(images).permute(0, 3, 1, 2)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    return probs.numpy()

explainer = lime_image.LimeImageExplainer()

# image_np: [H, W, C] numpy array
explanation = explainer.explain_instance(
    image_np,
    predict_fn,
    top_labels=5,
    num_samples=1000,     # number of perturbed samples
    batch_size=32
)

# Get image and mask for the top class
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,   # show only positive superpixels
    num_features=5,       # show top 5 superpixels
    hide_rest=False
)
```

**LIME limitations:**
- Slow (1000 forward passes)
- Non-deterministic (different runs → different explanations)
- Superpixel boundaries ≠ semantically meaningful boundaries

---

## 6. SHAP for Vision (DeepSHAP / GradientSHAP)

**Theoretically grounded:** Shapley values from game theory ensure fair attribution.

### DeepSHAP
```
Extends SHAP to deep neural networks using backpropagation rules.
Attributes each input pixel's contribution to the output score.

Based on DeepLIFT: computes difference-from-reference for each neuron
Reference: typically a black image (all zeros) or blurred image
```

```python
import shap
import numpy as np

model = models.resnet50(weights='IMAGENET1K_V2').eval()

# DeepSHAP requires background dataset
background = torch.zeros(10, 3, 224, 224)   # or use training set samples

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(image_tensor)   # list[num_classes] of [N, C, H, W]

# Plot attribution for top class
shap.image_plot(
    shap_values=shap_values,
    pixel_values=-image_tensor.permute(0, 2, 3, 1).numpy()
)
```

### GradientSHAP (Faster, More General)
```python
# GradientSHAP: averages integrated gradients over random baselines
explainer = shap.GradientExplainer(model, background)
shap_values, indices = explainer.shap_values(image_tensor, ranked_outputs=1)
```

---

## 7. Integrated Gradients

**Axiomatically justified** pixel-level attribution method.

```
Baseline: black image x' (all zeros)
Path: interpolate from baseline to actual image: x̂(α) = x' + α(x - x')  for α ∈ [0,1]

Integrated Gradients:
  IG_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x̂(α)) / ∂x_i) dα

Approximated by computing gradients at n steps along the path (n=50 typical):
  IG_i ≈ (x_i - x'_i) × (1/n) Σ_{k=1}^{n} ∂F(x' + (k/n)(x-x')) / ∂x_i
```

```python
from captum.attr import IntegratedGradients, NoiseTunnel, visualization

model.eval()
ig = IntegratedGradients(model)

# Baseline: black image
baseline = torch.zeros_like(image_tensor)

attributions = ig.attribute(
    image_tensor,
    baseline,
    target=target_class,       # which output neuron to explain
    n_steps=50,                # integration steps
    return_convergence_delta=True
)

# Visualize
from captum.attr import visualization as viz
fig, axes = viz.visualize_image_attr(
    np.transpose(attributions[0].squeeze().cpu().numpy(), (1, 2, 0)),
    np.transpose(image_np, (1, 2, 0)),
    method='heat_map',
    sign='positive',
    title='Integrated Gradients'
)
```

---

## 8. Comparing Methods

| Method | Architecture | Speed | Resolution | Faithful? |
|--------|-------------|-------|-----------|----------|
| CAM | GAP+FC required | Fast | Coarse (7×7 → upsample) | Moderate |
| Grad-CAM | Any CNN | Fast | Coarse (feature map resolution) | Moderate |
| Grad-CAM++ | Any CNN | Fast | Slightly better | Moderate |
| LIME | Any (black-box) | Slow (1000 FWD) | Superpixel | Low (approx) |
| DeepSHAP | Differentiable | Medium | Pixel | High |
| Integrated Gradients | Differentiable | Medium | Pixel | High (axioms) |

**Recommendation:**
- Quick debugging: Grad-CAM
- Pixel-level attribution: Integrated Gradients (Captum) or GradientSHAP
- Black-box model: LIME
- Stakeholder presentation: Grad-CAM (visually intuitive)
- Formal audit / regulatory: SHAP (theoretically grounded)

---

## 9. Practical Applications

### Bias Detection
```python
# Check if model uses spurious features
for image, label in test_loader:
    cam, pred = grad_cam.generate(image)

    # If all explanations focus on background → spurious correlation
    # Example: model predicting "wolf" by looking at snow (not the wolf)
    background_attention = cam[background_mask].mean()
    foreground_attention = cam[foreground_mask].mean()

    if background_attention > foreground_attention:
        print(f"WARNING: Model focusing on background for class {label}")
```

### Document AI Application
```python
# For document field extraction, verify model reads the right region
# e.g., for invoice total: heatmap should focus on the total amount area

def explain_document_prediction(model, document_image, field_class):
    grad_cam = GradCAM(model, model.layer4[-1])
    cam, pred = grad_cam.generate(document_image, target_class=field_class)

    # Overlay heatmap on document
    visualization = show_cam_on_image(document_image_np/255.0, cam)

    return visualization, cam

# Expected: cam should be high around the "Total" label and its value
# Red flag: cam high on company logo or header → model learned wrong shortcut
```

---

## 10. When to Use What

| Situation | Method | Why |
|-----------|--------|-----|
| Quick debug of CNN prediction | Grad-CAM | Fast, easy, works on any CNN |
| Multiple instances of same class | Grad-CAM++ | Better handles all instances |
| Non-differentiable model (sklearn, XGBoost) | LIME | Black-box compatible |
| Production ML audit / regulatory | SHAP | Consistent, theoretically principled |
| Fine-grained pixel attribution | Integrated Gradients | Pixel-level, axiom-satisfying |
| Executive presentation | Grad-CAM overlay | Visually intuitive and clear |
| Detecting spurious correlations | Grad-CAM or IG, compare across many samples | Pattern of wrong focus across dataset |

---

## 11. Gotchas

**Grad-CAM coarseness is a feature, not a bug.**
The heatmap resolution equals the last conv layer feature map size (7×7 for ResNet with 224×224 input). The upsampling to full resolution is bilinear interpolation — it's not pixel-level accuracy. For pixel-accurate attribution, use Integrated Gradients or SHAP.

**Grad-CAM gradient saturation.**
For very confident predictions (score → 1.0 before softmax), gradients saturate → near-zero gradients → meaningless Grad-CAM. Use the logit (pre-softmax) score for backward pass, not the softmax probability.

```python
# Wrong: backward on softmax output (saturates for confident predictions)
output = torch.softmax(model(x), dim=1)
output[0, target_class].backward()

# Right: backward on logit
logits = model(x)
logits[0, target_class].backward()
```

**LIME is non-deterministic — seed it for reproducibility.**
```python
explainer.explain_instance(image, predict_fn, num_samples=1000,
                           random_seed=42)
```

**Not all saliency methods agree.**
Different methods highlight different regions for the same prediction. This reflects their different definitions of "importance." No single method is definitively correct — use multiple methods and look for consensus.

---

## 12. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Grad-CAM all uniform (no focus) | Gradient saturation | Use logits not softmax for backward |
| Grad-CAM activates everywhere | Wrong target layer | Choose last conv layer before GAP |
| SHAP values all zero | Wrong baseline | Use mean image or noise baseline instead of zeros |
| IG attributions very noisy | Few integration steps | Increase n_steps to 100-300 |
| Heatmap doesn't match visual object | Model learned spurious feature | Diagnostic: check many samples; retrain with debiased data |
| Library error "hooks not removed" | Forward hooks accumulate | Use context manager (`with GradCAM(...)`) |

---

## 13. Interview Q&A (Senior Level)

**Q: What is Grad-CAM and how does it differ from CAM?**
A: CAM requires a specific architecture — Global Average Pooling followed by a single FC layer — because it uses the FC weights as channel importance scores. This limits CAM to architectures with this exact design. Grad-CAM generalizes this by using gradients of the class score with respect to the feature maps as importance weights. Specifically, it global-average-pools the gradients over spatial dimensions to get a per-channel weight, then takes a weighted sum of the feature maps. This works with any CNN architecture and any layer — you can even apply Grad-CAM to intermediate layers to understand early vs late feature usage. The tradeoff: Grad-CAM can be noisy due to gradient saturation; CAM is more stable where applicable.

**Q: Why is explainability critical in production ML, beyond regulatory compliance?**
A: (1) Debugging: when a model fails on specific cases, saliency maps reveal whether it's looking at the right features — catching data leakage, spurious correlations, and distribution shift before they become production incidents. (2) Trust calibration: business stakeholders and domain experts (doctors, lawyers) need to see model reasoning before trusting its outputs — "black box accepted" is rarely true in high-stakes decisions. (3) Continuous monitoring: by running Grad-CAM on production samples and checking if the focus region distribution shifts, you can detect when the model starts reasoning differently — early warning of distribution shift before metrics degrade. (4) Dataset curation: saliency maps on wrongly classified samples guide what training data to collect next.

**Q: What are the limitations of saliency-based explanations for neural networks?**
A: (1) Coarseness — Grad-CAM resolution is limited to feature map size (7×7 for ResNet), not pixel-level. (2) Gradient saturation — for high-confidence predictions, gradients near zero → meaningless maps. (3) Faithfulness — there's no guarantee saliency maps reflect what the model actually used. A study showed that saliency maps for randomly initialized networks look similar to trained networks — suggesting they may reflect image statistics, not model computation. (4) Input sensitivity — slight perturbations to input can change saliency maps while predictions remain stable. (5) No counterfactuals — saliency shows where the model looked, not what it would predict if the object were absent (that requires TCAV, LIME, or counterfactual generation). Use multiple methods and validate explanations through intervention experiments.

---

## 14. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| SHAP for tabular models | `../../0.machine learning/algorithms/02_tree_models.md` | Same SHAP concept for XGBoost/RF |
| Feature maps being explained | `../fundamentals/01_cnn_mechanics.md` | CNN feature maps = what Grad-CAM visualizes |
| Model debugging workflow | `../../0.machine learning/fundamentals/04_model_evaluation.md` | Production monitoring |
| Attention visualization | `../../1.deep learning/architectures/04_transformer.md` | Attention weights as built-in explanation |

---

## Key Takeaway

**Explainability is not optional for production CV — it's how you debug and validate.**

**Method selection:**
```
Fast debug → Grad-CAM (any CNN, any layer)
Multiple objects → Grad-CAM++
Pixel-level → Integrated Gradients (Captum)
Black-box → LIME
Regulatory / audit → SHAP (consistent, game-theoretically grounded)
```

**Most important insight:** A model with 95% accuracy can still be wrong for the right reasons or right for the wrong reasons. Explainability reveals which.

For document AI: always verify that field extraction models focus on the correct document region, not on template artifacts, page headers, or neighboring fields.
