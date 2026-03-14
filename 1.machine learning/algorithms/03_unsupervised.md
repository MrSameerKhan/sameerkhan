# 03 — Unsupervised Learning (Clustering, Dimensionality Reduction, Anomaly Detection)

## Quick Reference

| Algorithm | Type | Key Parameter | Best For |
|-----------|------|--------------|---------|
| K-Means | Clustering | k (number of clusters) | Globular clusters, large data |
| DBSCAN | Clustering | eps, min_samples | Arbitrary shapes, noise/outliers |
| Hierarchical | Clustering | linkage, n_clusters | When hierarchy matters, no k needed upfront |
| GMM | Clustering | n_components | Soft assignments, elliptical clusters |
| PCA | Dim reduction | n_components | Linear reduction, visualization, preprocessing |
| t-SNE | Dim reduction | perplexity | 2D/3D visualization only |
| UMAP | Dim reduction | n_neighbors, min_dist | Visualization + preserves global structure |
| Isolation Forest | Anomaly detection | contamination | General-purpose anomaly detection |
| LOF | Anomaly detection | n_neighbors | Local density anomalies |
| One-Class SVM | Anomaly detection | nu | High-dim anomaly detection |

---

## 1. K-Means Clustering

### How It Works
```
1. Initialize k centroids (randomly or k-means++ smart init)
2. Assign each point to nearest centroid (Euclidean distance)
3. Update each centroid = mean of assigned points
4. Repeat until centroids don't move (convergence)

Objective: minimize Within-Cluster Sum of Squares (WCSS / Inertia)
  WCSS = Σ_k Σ_{xᵢ∈Cₖ} ‖xᵢ − μₖ‖²
```

### K-Means++ Initialization
Random init → bad clusters if centroids start near same region.
K-Means++: first centroid random, each subsequent centroid chosen with probability ∝ distance² from nearest existing centroid → spread out initial centroids → faster convergence, better solution.

### Choosing K — Elbow Method + Silhouette
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Elbow plot
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K'); plt.ylabel('Inertia')
plt.title('Elbow Method')

# Silhouette plot (higher = better separation)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('K'); plt.ylabel('Silhouette Score')
```

**Silhouette score:**
```
s(i) = (b(i) − a(i)) / max(a(i), b(i))

a(i) = mean distance to points in same cluster (cohesion)
b(i) = mean distance to points in nearest other cluster (separation)

s=1: well-clustered
s=0: on boundary between clusters
s=-1: likely assigned to wrong cluster

Overall score = mean s(i) over all points
Best k = k with highest mean silhouette score
```

### K-Means Limitations
```
1. Assumes spherical clusters (uses Euclidean distance)
2. Sensitive to scale → always normalize first
3. Sensitive to outliers (outliers distort centroid)
4. k must be specified in advance
5. Can't find non-convex clusters (use DBSCAN instead)
```

```python
# Full K-Means pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

km_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42))
])
labels = km_pipe.fit_predict(X)
```

---

## 2. DBSCAN (Density-Based Spatial Clustering)

### Core Concepts
```
eps (ε): radius of neighborhood
min_samples: minimum points within eps to be a core point

Core point: has ≥ min_samples points within eps (including itself)
Border point: within eps of a core point but not itself a core point
Noise point: neither core nor border → labeled -1 (outlier)

Cluster = all points density-reachable from a core point
```

### Why DBSCAN Over K-Means
- Finds **arbitrary-shaped clusters** (crescent, ring, irregular)
- **Automatically detects outliers** (noise points = -1)
- **No k required** — number of clusters determined by data density
- Handles clusters of **varying density** (if eps/min_samples tuned)

### Parameter Selection
```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Choose eps: k-distance plot
# Plot distance to k-th nearest neighbor, sorted ascending
# Elbow = good eps
k = 5  # min_samples
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1])
plt.plot(distances)
plt.ylabel(f'{k}-th nearest neighbor distance')
plt.xlabel('Points sorted by distance')
# Choose eps at the elbow

# Run DBSCAN
db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters: {n_clusters}, Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
```

### DBSCAN Limitations
```
Struggles with:
  1. Clusters of very different densities → one eps doesn't fit all
  2. High-dimensional data → distances become similar (curse of dimensionality)
  3. Large datasets → O(n²) without spatial index (use HDBSCAN for speed)
```

### HDBSCAN (Modern Alternative)
Hierarchical DBSCAN — handles varying density clusters, more robust parameter selection.
```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
labels = clusterer.fit_predict(X_scaled)
```

---

## 3. PCA (Principal Component Analysis)

### What PCA Does
Find directions (principal components) of maximum variance in the data. Project data onto these directions → low-dimensional representation preserving maximum variance.

### Math
```
1. Center data: X_centered = X − mean(X)
2. Compute covariance matrix: Σ = (1/n) Xᵀ X
3. Eigendecomposition: Σ = VΛVᵀ
   V = eigenvectors (principal components)
   Λ = diagonal matrix of eigenvalues (variance explained)
4. Project: Z = X_centered · V[:, :k]  (keep top k components)

Explained variance ratio: λᵢ / Σλⱼ
```

### How Many Components?

**Scree plot + cumulative explained variance:**
```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_scaled)

# Cumulative explained variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumvar)
plt.axhline(0.95, color='red', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components'); plt.ylabel('Cumulative Explained Variance')

# Choose k where cumulative variance ≥ 0.95
k = np.argmax(cumvar >= 0.95) + 1
print(f"Components for 95% variance: {k}")

# Apply PCA with k components
pca_k = PCA(n_components=k)
X_reduced = pca_k.fit_transform(X_scaled)
```

### PCA for Visualization (2D)
```python
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.5)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)')
```

### When to Use PCA
```
✓ Feature preprocessing before linear models / neural nets (reduces multicollinearity)
✓ Visualization (reduce to 2-3 dims for scatter plot)
✓ Noise reduction (low-variance components often capture noise)
✓ Compression (images, signals)

✗ Not for: tree-based models (don't need — handle high-dim natively, lose interpretability)
✗ Not for: when you need original feature names for business interpretation
✗ Not for: non-linear structure (use UMAP or autoencoders instead)
```

### Limitations of PCA
- Linear only — misses nonlinear structure
- Sensitive to outliers (use RobustPCA or remove outliers first)
- Components are linear combinations — not interpretable as original features
- Must scale features first (otherwise high-variance features dominate)

---

## 4. t-SNE and UMAP (Non-linear Visualization)

### t-SNE

**Use case:** 2D/3D visualization of high-dimensional data. NOT for preprocessing before modeling — use PCA for that.

```
1. Compute pairwise similarities in high-dim space (Gaussian kernel)
2. Map to low-dim, compute similarities (Student t-distribution, heavier tail)
3. Minimize KL divergence between high-dim and low-dim similarities via gradient descent
```

```python
from sklearn.manifold import TSNE

# PCA first to 50 dims (standard practice for speed + numerical stability)
X_pca50 = PCA(n_components=50).fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
            random_state=42, init='pca')
X_tsne = tsne.fit_transform(X_pca50)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.5, s=5)
```

**Perplexity** ≈ expected number of nearest neighbors per point. Typical: 5-50. Smaller datasets → smaller perplexity.

**t-SNE pitfalls:**
- Distances between clusters are meaningless (only local structure preserved)
- Different random seeds → different visualizations
- Slow for n > 10K (use UMAP instead)
- Cannot transform new points (not a transformer)

### UMAP (Uniform Manifold Approximation and Projection)

Better than t-SNE for: faster, preserves global structure, can transform new points.

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric='euclidean', random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# UMAP can transform new data (t-SNE cannot)
X_new_umap = reducer.transform(X_new_scaled)
```

**n_neighbors**: controls local vs global structure balance (15-50 typical).
**min_dist**: how tightly clustered points are in embedding (0.0-1.0).

---

## 5. Anomaly Detection

### Isolation Forest
**Idea:** Anomalies are easier to isolate — they need fewer random splits to be separated.
Build random trees where each split randomly selects a feature and threshold. Anomaly score = average path length to isolate the point (shorter = more anomalous).

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=100,
    contamination=0.05,   # expected fraction of outliers (0.05 = 5%)
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train)

# Predict: -1 = anomaly, 1 = normal
labels = iso.predict(X_test)
scores = iso.score_samples(X_test)   # negative = more anomalous

anomalies = X_test[labels == -1]
print(f"Anomalies detected: {(labels == -1).sum()} ({(labels==-1).mean()*100:.1f}%)")
```

### Local Outlier Factor (LOF)
**Idea:** Compare local density of a point to its neighbors. Points in low-density regions relative to neighbors = anomalies.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=True   # novelty=True to predict on new data (fit on train, predict on test)
)
lof.fit(X_train)
labels = lof.predict(X_test)    # -1 = anomaly
scores = lof.score_samples(X_test)
```

**LOF vs Isolation Forest:**
- LOF: better for local density anomalies (anomaly in a dense region that's less dense than its neighbors)
- Isolation Forest: better for global anomalies and high-dimensional data
- LOF is O(n²) — slow for large data; IsoForest is O(n log n)

### One-Class SVM
Train only on normal data, finds a boundary enclosing it. Useful when anomalies are not available during training.

```python
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
ocsvm.fit(X_train_normal)   # train only on normal samples
labels = ocsvm.predict(X_test)  # -1 = anomaly, 1 = normal
```

**nu**: upper bound on fraction of outliers (0.0-1.0).

### Choosing Anomaly Detection Method

| Method | When | Notes |
|--------|------|-------|
| Isolation Forest | General purpose, large data | Fast; set contamination to expected outlier % |
| LOF | Local density anomalies, small data | Slow (O(n²)); better for clustered normal data |
| One-Class SVM | High-dim, no anomaly labels | Slow; scale data first |
| DBSCAN (noise points) | Known cluster structure | Anomalies = -1 label |
| VAE reconstruction error | Image/text anomalies | Need labeled normal data for training |
| Z-score / IQR | Univariate, quick check | Simple; misses multivariate anomalies |

---

## 6. Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

# Compute linkage matrix
Z = linkage(X_scaled, method='ward')   # 'ward', 'complete', 'average', 'single'

# Dendrogram (choose cut height = number of clusters)
plt.figure(figsize=(15, 5))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.axhline(y=5, color='red', linestyle='--', label='Cut here → 3 clusters')
plt.show()

# Extract cluster labels at given number of clusters
labels = fcluster(Z, t=3, criterion='maxclust')
```

**Linkage methods:**
```
Ward: minimize within-cluster variance (usually best, spherical clusters)
Complete: max distance between clusters (compact, equal-size clusters)
Average: mean distance (between ward and single)
Single: min distance (can chain outliers into clusters)
```

---

## 7. Gaussian Mixture Models (GMM)

Soft clustering — each point has probability of belonging to each cluster (unlike K-Means hard assignment).

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full',
                      n_init=10, random_state=42)
gmm.fit(X_scaled)

# Soft cluster assignments (probabilities)
probs = gmm.predict_proba(X_scaled)   # [n_samples, n_components]
labels = gmm.predict(X_scaled)         # hard assignment = argmax

# Model selection (BIC/AIC)
bics = [GaussianMixture(n_components=k).fit(X_scaled).bic(X_scaled) for k in range(2, 11)]
best_k = np.argmin(bics) + 2
```

**GMM vs K-Means:**
- GMM: elliptical clusters, soft probabilistic assignments, BIC/AIC for k selection
- K-Means: spherical clusters, hard assignments, faster

---

## 8. Clustering Evaluation (No Ground Truth)

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Silhouette: higher is better (range -1 to 1)
sil = silhouette_score(X_scaled, labels)

# Calinski-Harabasz: higher is better (between-cluster / within-cluster variance ratio)
ch = calinski_harabasz_score(X_scaled, labels)

# Davies-Bouldin: lower is better (average cluster similarity)
db = davies_bouldin_score(X_scaled, labels)

print(f"Silhouette: {sil:.3f}, CH: {ch:.1f}, DB: {db:.3f}")
```

---

## 9. Gotchas

**K-Means requires scaled features.**
A feature with range [0, 10000] dominates distance calculation over a feature with range [0, 1]. Always StandardScale or MinMaxScale before K-Means, DBSCAN, LOF.

**t-SNE is for visualization only — never use as preprocessing for a model.**
t-SNE distances are not preserved, new points can't be transformed, and results change with random seed. Use PCA or UMAP for preprocessing.

**DBSCAN eps is not intuitive to set.**
Always use the k-nearest-neighbor distance plot to guide eps selection. A wrong eps → everything is one cluster or all noise.

**Isolation Forest contamination must be set.**
Default contamination=0.1 (10% outliers). If true outlier rate is different, calibrate. Use `contamination='auto'` to let the algorithm decide (less interpretable).

**K-Means local optima.**
Different initializations → different solutions. Always use `n_init=10+` (run 10 times, keep best). K-Means++ reduces but doesn't eliminate this.

**Hierarchical clustering doesn't scale.**
O(n² log n) time and O(n²) memory — can't handle > 10K-100K samples. For large datasets, use K-Means, DBSCAN, or HDBSCAN.

---

## 10. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| K-Means all points in one cluster | eps too large / features not scaled | Scale features; reduce eps for DBSCAN |
| DBSCAN returns all noise | eps too small | Use k-distance plot to calibrate eps |
| PCA first component explains 99% variance | One feature dominates | Scale features before PCA |
| t-SNE different result each run | Stochastic algorithm | Set random_state; run multiple times and compare |
| Isolation Forest flags normal points | contamination too high | Lower contamination; check data quality |
| Silhouette score low for all k | No natural clusters in data | Try different algorithm; consider if clustering is appropriate |
| Anomaly detection misses known outliers | contamination too low | Increase contamination; try ensemble of detectors |

---

## 11. Interview Q&A (Senior Level)

**Q: How do you choose between K-Means and DBSCAN for a new clustering problem?**
A: Start with exploratory analysis. K-Means when: you have a reasonable prior on number of clusters, clusters are roughly spherical/convex in feature space, dataset is large (K-Means scales to millions), need fast results. DBSCAN when: you don't know number of clusters in advance, you expect non-convex shapes (rings, crescents), you need automatic outlier detection, data has varying-density regions. In practice, try both with proper scaling and compare silhouette scores. If clusters look elongated in PCA/UMAP visualization, DBSCAN or GMM will handle them better than K-Means.

**Q: What's the curse of dimensionality and how does it affect unsupervised learning?**
A: In high dimensions, distances between all pairs of points converge to the same value — the ratio of max to min distance → 1 as d → ∞. This makes distance-based algorithms (K-Means, KNN, DBSCAN, LOF) unreliable since they can't distinguish "nearby" from "far away." Effects: K-Means clusters become meaningless, DBSCAN can't find meaningful neighborhoods, KNN neighbors are no more similar to a point than random samples. Solutions: dimensionality reduction (PCA/UMAP) before clustering, feature selection, use cosine similarity (better than Euclidean in high-dim), subspace clustering methods.

**Q: When would you use anomaly detection vs supervised classification for fraud detection?**
A: Anomaly detection when: no labeled fraud examples available (unsupervised), fraudulent patterns are novel and constantly evolving (supervised model overfits to historical fraud), extreme class imbalance makes supervised learning unstable. Supervised classification when: labeled examples exist, patterns are stable enough to learn from, you need calibrated probability estimates per transaction. In practice: hybrid approach — use Isolation Forest as a feature (anomaly score) fed into a supervised model alongside business features. This combines the novelty detection capability of unsupervised with the discrimination power of supervised learning.

---

## 12. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| PCA preprocessing | `../fundamentals/03_feature_engineering.md` | Dimensionality reduction as feature preprocessing |
| K-Means as quantization | `../../1.deep learning/architectures/05_generative.md` | VQ-VAE uses K-Means-like discrete latent codes |
| GMM and EM algorithm | `04_probabilistic.md` | GMM trained with EM (Expectation-Maximization) |
| Anomaly detection for documents | Your domain | Isolation Forest on OCR confidence scores + layout features |
| UMAP vs t-SNE | `../fundamentals/02_eda.md` | Visualization in EDA section |
| LOF and outlier handling | `../fundamentals/02_eda.md` | Outlier detection section |

---

## Key Takeaway

**Clustering:** K-Means (fast, spherical) → DBSCAN (arbitrary shape, finds outliers) → GMM (soft, elliptical).
**Dim reduction:** PCA (linear, preprocessing) → UMAP (nonlinear, visualization) → t-SNE (visualization only, not preprocessing).
**Anomaly detection:** Isolation Forest (general, fast) → LOF (local density) → One-Class SVM (high-dim).

Always scale before any distance-based algorithm. Always visualize clusters with PCA/UMAP. Always check silhouette score to validate cluster quality.

For your domain: Isolation Forest on OCR pipeline outputs (confidence distributions, character-level uncertainty) is excellent for detecting pages/documents where extraction will fail before running expensive downstream processing.
