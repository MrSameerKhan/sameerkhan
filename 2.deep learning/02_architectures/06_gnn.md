# 06 — Graph Neural Networks (GNN)

## Quick Reference

| Variant | Core Idea | Strength | Use Case |
|---------|-----------|----------|---------|
| GCN | Average neighbor features + self (spectral) | Simple, fast | Node classification, citation graphs |
| GraphSAGE | Sample + aggregate neighbors (inductive) | Scales to unseen nodes | Large graphs, social networks |
| GAT | Attention-weighted neighbor aggregation | Learns which neighbors matter | Heterogeneous graphs, relational data |
| GIN | Sum aggregation (most expressive) | Provably as powerful as WL test | Graph-level classification |
| MPNN | General message passing framework | Flexible | Molecular property prediction |

**One-line summary:** GNNs update each node's embedding by aggregating information from its neighbors, iteratively — after K layers, each node sees its K-hop neighborhood.

---

## 1. Why Graphs? When Standard DL Fails

CNNs assume grid structure (images). RNNs assume sequential structure (text). Neither handles **irregular, relational structure**:

- Molecules: atoms = nodes, bonds = edges (no natural ordering)
- Social networks: users = nodes, friendships = edges (variable degree)
- Knowledge graphs: entities = nodes, relations = edges
- Document understanding: words/entities with relational structure (dependency parse, document graph)
- Scene graphs: objects = nodes, spatial relations = edges
- Citation networks: papers = nodes, citations = edges

**Key property:** GNNs are **permutation invariant** — output doesn't change if you relabel nodes (unlike applying MLP to adjacency matrix).

---

## 2. Message Passing Framework (General)

All GNN variants follow this pattern across K layers:

```
For each layer k = 1, ..., K:
  For each node v:
    1. AGGREGATE: collect messages from neighbors
       m_v = AGG({h_u^{k-1}: u ∈ N(v)})

    2. UPDATE: combine aggregated message with own embedding
       h_v^{k} = UPDATE(h_v^{k-1}, m_v)
```

After K layers, `h_v^{(K)}` = embedding of node v incorporating its K-hop neighborhood.

**Task heads on top:**
- Node classification: `MLP(h_v^{(K)})` → class label per node
- Edge prediction: `MLP([h_u || h_v])` → edge exists/type
- Graph classification: `READOUT({h_v^{(K)}})` → MLP → graph label
- READOUT = sum / mean / max over all node embeddings

---

## 3. GCN (Graph Convolutional Network)

**Kipf & Welling, 2017** — spectral graph convolution simplified to:

```
H^{l+1} = σ( D̃^{-½} Ã D̃^{-½} H^{l} W^{l} )
```

Where:
- Ã = A + I (adjacency + self-loops)
- D̃ = degree matrix of Ã
- D̃^{-½} Ã D̃^{-½} = symmetric normalization (prevents scale issues for high-degree nodes)
- W^{l} = learnable weight matrix for layer l
- σ = activation (ReLU)

**Intuition:** each node aggregates its neighbors' features (normalized by degree) + itself, then applies a linear transform.

**Limitation:** transductive — requires all nodes at training time (can't handle new nodes). Fixed by GraphSAGE.

---

## 4. GraphSAGE (Inductive Representation Learning)

```
1. Sample fixed-size neighborhood (K neighbors, not all)
2. h_N(v) = AGG({h_u: u ∈ sampled_N(v)})
   AGG options: Mean, LSTM, Pooling
3. h_v^{(k)} = W · [h_v^{(k-1)} || h_N(v)]
4. h_v^{(k)} = normalize(h_v^{(k)})
```

**Key improvement over GCN:**
- **Inductive:** new unseen nodes can be embedded using their neighbors
- **Scalable:** sample fixed-size neighborhood (not all neighbors) → minibatch training on large graphs

---

## 5. GAT (Graph Attention Network)

Instead of fixed normalization (GCN), **learn attention weights** for each neighbor:

```
e_{uv}   = LeakyReLU( a^T [W·h_u || W·h_v] )        # attention score
α_{uv}   = softmax over N(v): exp(e_{uv}) / Σ exp(e_{uw})
h_v^{(k)} = σ( Σ_{u∈N(v)} α_{uv} · W · h_u^{(k-1)} )
```

**Multi-head:** run H independent attention heads, concatenate (or average) outputs.

**Strength:** different neighbors contribute differently based on learned relevance. **Use case:** heterogeneous graphs where not all neighbors are equally informative.

---

## 6. GIN (Graph Isomorphism Network)

Most expressive GNN (Xu et al., 2019) — as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test:

```
h_v^{(k)} = MLP^{(k)}( (1+ε) · h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)} )
```

- **Sum aggregation** — preserves multiset structure (mean can't distinguish {1,2} from {1,2,2})
- ε: learnable scalar (or fixed to 0)
- MLP per layer (not just linear) — captures nonlinear structural features

**Use case:** graph-level tasks (molecular property prediction, chemical reaction prediction).

---

## 6.5 Graph Transformers (Modern Alternative)

Standard GNNs (GCN/GAT/GIN) pass messages only between adjacent nodes — K layers = K-hop receptive field. **Graph Transformers** apply self-attention across all node pairs, treating the graph structure as a bias on attention scores rather than a hard constraint.

| Model | Year | Key Idea |
|-------|------|---------|
| Graphormer (Microsoft) | 2021 | Add structural encodings (centrality, spatial distance, edge features) as biases to attention. SOTA on OGB-LSC PCQM4M (molecule property prediction) |
| GraphGPS | 2022 | Modular framework: positional encoding + message-passing + global attention |
| GraphTransformer / NodeFormer | 2022-23 | Linear-attention variants for million-node graphs |
| Exphormer | 2023 | Sparse expander-graph attention; scales to graphs that don't fit in dense attention |
| TokenGT | 2022 | Treat each node AND edge as a token; pure transformer on graph |

**When Graph Transformers win:** dense graph-level prediction tasks (molecules, scene graphs), tasks where long-range information matters, modest-size graphs where O(n²) attention is acceptable.

**When classical GNNs still win:** very large graphs (billions of nodes — Pinterest, Facebook), where message-passing's locality is a strength.

---

## 7. Heterogeneous and Temporal Graphs

Real graphs aren't single-type — they have multiple node/edge types and evolve over time.

| Variant | Idea | Used for |
|---------|------|---------|
| R-GCN | Relation-specific weight matrices per edge type | Knowledge graphs (FB15K, WordNet) |
| HAN (Heterogeneous Attention Network) | Hierarchical attention over node types and meta-paths | Recommender systems, citation networks |
| HGT (Heterogeneous Graph Transformer) | Type-specific attention; the "BERT" of heterogeneous graphs | E-commerce graphs |
| TGN / TGAT | Temporal message passing with memory module | Fraud detection, dynamic social networks |
| JODIE / EvolveGCN | Time-evolving node embeddings | User interest tracking |

Modern recommendation systems (Pinterest's PinSage, LinkedIn's GraphSAGE variants) and fraud-detection pipelines at financial institutions are usually heterogeneous and temporal.

For knowledge-graph embeddings specifically (TransE, DistMult, ComplEx, RotatE) — these are **not** GNNs but worth knowing for any KG question; they're scoring functions over (head, relation, tail) triples rather than message-passing networks.

---

## 7. Over-Smoothing Problem

With too many GNN layers, all node embeddings converge to same vector (neighborhood overlap across the entire graph).

```
Layer 1: each node sees 1-hop neighbors
Layer 2: each node sees 2-hop neighbors
Layer K: each node sees K-hop neighbors (often = entire graph for small K)
```

**Fix:** · Use fewer layers (2-3 often optimal) · JK-Net: Jumping Knowledge — concatenate embeddings from all layers · DropEdge: randomly drop edges during training · PairNorm: normalize node embeddings to have fixed sum of pairwise distances

---

## 8. Scalability

Full-graph training (load entire graph + adjacency to GPU) fails for large graphs (Twitter: 500M nodes).

| Technique | Idea | Library |
|-----------|------|---------|
| Mini-batch (GraphSAGE) | Sample K-hop neighborhood per node | PyG, DGL |
| Cluster-GCN | Partition graph into clusters, train on clusters | PyG |
| GraphSAINT | Sample subgraphs, normalize bias | PyG |
| DistDGL | Multi-GPU distributed GNN training | DGL |

---

## 9. When to Use GNNs

| Task | GNN Variant | Why |
|------|------------|-----|
| Node classification (citation/social) | GCN / GAT | Standard transductive setting |
| New node embeddings at inference | GraphSAGE | Inductive — generalizes to new nodes |
| Molecular property prediction | GIN / MPNN | Sum aggregation captures molecular structure |
| Link prediction | GraphSAGE + edge head | Predict if edge exists between two nodes |
| Knowledge graph completion | RotatE, TransE (relation-aware GNN) | Relations have different semantics |
| Scene graph understanding | GAT | Spatial relations between objects |
| Document entity relation extraction | GAT on dependency parse | Not all word relations equal — attention helps |

---

## 10. Gotchas

**1. Over-smoothing kills deep GNNs.** More layers ≠ better. 2-3 layers often optimal. Unlike CNNs (deeper = better receptive field), GNNs don't benefit from depth the same way because graph diameter is small.

**2. GCN is transductive — can't embed new nodes.** At inference, if new nodes appear not in training graph — no embedding. Use GraphSAGE for inductive settings (common in production).

**3. Graphs must be on GPU memory.** Large graphs (millions of nodes) don't fit in GPU memory as a single adjacency matrix. Need mini-batch sampling strategies (GraphSAGE, Cluster-GCN).

**4. Disconnected components are isolated.** Nodes with no edges only see themselves — no neighborhood to aggregate. Check your graph construction.

**5. Edge direction matters.** Directed vs undirected changes the neighborhood definition. Social network follows are directed; molecular bonds are undirected. GCN assumes undirected by default.

**6. Heterogeneous graphs need special handling.** If nodes/edges have different types (user, item; click, purchase), a simple GCN mixes types. Use R-GCN (relation-specific weight matrices) or HAN (heterogeneous attention network).

---

## 11. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All node embeddings identical | Over-smoothing (too many layers) | Reduce to 2-3 layers; add JK connections |
| Training loss fine, test loss high | Graph leakage — test edges seen in training | Use proper train/val/test edge splits |
| OOM on large graph | Full adjacency matrix on GPU | Use mini-batch neighbor sampling (GraphSAGE) |
| Poor performance on new nodes | GCN used in inductive setting | Switch to GraphSAGE |
| Graph classification not converging | READOUT function too simple | Try sum then mean; try hierarchical pooling (DiffPool) |
| Node degree imbalance hurting performance | High-degree nodes dominate aggregation | Use degree normalization (GCN) or attention (GAT) |

---

## 12. Code Reference

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_add_pool
from torch_geometric.data import Data

# Build a simple graph
edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)  # [2, num_edges]
x    = torch.rand(3, 16)   # 3 nodes, 16 features each
data = Data(x=x, edge_index=edge_index)

# 2-Layer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)   # node classification

model = GCN(in_dim=16, hidden_dim=64, out_dim=7)

# GAT (Attention-based)
class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# GraphSAGE (Inductive)
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Mini-batch training on large graph
from torch_geometric.loader import NeighborLoader
loader = NeighborLoader(data,
    num_neighbors=[10, 5],   # sample 10 neighbors at hop 1, 5 at hop 2
    batch_size=64,
    input_nodes=train_mask)

# Graph Classification (Graph-level prediction)
from torch_geometric.loader import DataLoader

class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim))
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1      = GINConv(nn1)
        self.conv2      = GINConv(nn2)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)   # sum pooling = graph embedding
        return self.classifier(x)
```

---

## 13. Interview Q&A (Senior Level)

**Q: Why can't you just flatten the adjacency matrix and use an MLP?**

Two problems: (1) **Permutation variance** — MLP(adjacency_row) depends on the arbitrary ordering of nodes. Relabeling node 3 as node 7 gives a completely different input, but the graph is the same. GNNs are permutation-invariant by design. (2) **Variable graph size** — MLPs have fixed-size inputs; graphs have variable numbers of nodes and edges. GNNs operate on each node's local neighborhood regardless of graph size.

**Q: How many GNN layers should you use and why?**

Typically 2-3. Each layer propagates information one hop. With K layers, each node sees its K-hop neighborhood. For most real graphs (small world property, short diameter), 2-3 hops covers most relevant context. Beyond 3-4 layers, over-smoothing kicks in — all nodes see the same global structure → embeddings collapse to the same vector. Unlike CNNs where more layers = bigger receptive field with new features, GNNs suffer from neighborhood overlap at depth.

**Q: What's the difference between transductive and inductive learning in GNNs?**

**Transductive** (GCN): the model is trained on the full graph including all test nodes (just without test labels). At inference, test node embeddings are computed using their already-seen neighbors. Can't embed nodes not present at training time. **Inductive** (GraphSAGE): trains an aggregation function that samples and aggregates neighbor features. At inference, new nodes can be embedded by running aggregation over their neighbors — even if those neighbors are new too. Production systems almost always need inductive (new users, new documents, new entities).

**Q: Where do GNNs fit in document understanding?**

Several places: (1) **Entity relation extraction**: build a graph over entities in a document (from NER), use GAT to model which entities relate to which. (2) **Document layout understanding**: LayoutLM is Transformer-based, but GNN alternatives model layout as graph (words = nodes, spatial proximity = edges). (3) **Table parsing**: cells = nodes, row/column relationships = edges. (4) **Knowledge graph integration**: link extracted entities to a KG, use GNN to enrich entity representations with KG context. Practically, transformer-based models (LayoutLM, BERT) dominate document understanding, but GNNs are useful for explicit relational reasoning.

---

## 14. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| Attention in GAT | `04_transformer.md` | Same attention mechanism concept |
| Message passing aggregation | `../01_fundamentals/05_modern_components.md` | Aggregation related to pooling/attention |
| Over-smoothing → vanishing gradients | `../01_fundamentals/03_training_stability.md` | Both are depth-related stability problems |
| Graph classification READOUT | `01_mlp.md` | READOUT + MLP = graph-level head |
| Entity relation extraction use case | Your domain | Document graph → GAT → relation type classifier |

---

## Key Takeaway

```
GNNs = iterative neighborhood aggregation
        Each layer expands one more hop
        After K layers, each node embedding captures its K-hop structural context

3 variants to know cold:
  GCN        — simple, transductive
  GraphSAGE  — inductive, scalable
  GAT        — attention-weighted neighbors

Core failure mode: over-smoothing (2-3 layers is usually optimal)

For your domain: GNNs are a niche but powerful tool for relational document understanding
(entity graphs, table structure, layout graphs).
Transformers dominate, but GNNs give explicit control over relational structure.
```
