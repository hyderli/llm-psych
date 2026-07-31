# Plan: Emotion Vector Geometry Analysis

**Branch:** cb-gemma-emotions-scripts  
**Model:** google/gemma-3-4b-it (34 transformer layers)  
**Primary layer:** 22 (floor(34 × 2/3), 0-indexed)  
**Input:** `steering_vectors/gemma3-4b-story/` in HF `llm-psych/llm-psych-activations`

---

## Inputs

Four `.pt` files, each a dict `{layer_idx: tensor[hidden_dim=2560]}` for layers 0..33:

```
steering_vectors/gemma3-4b-story/
├── joy_all_layers.pt
├── sadness_all_layers.pt
├── admiration_all_layers.pt
└── loathing_all_layers.pt
```

Working data: 4 emotions × 34 layers = 136 vectors, each in R^2560.  
**Primary analysis uses layer 22.** Cross-layer plots use all 34 layers.

---

## Analyses

### 1. Norms (magnitude per emotion per layer)

```
norm[e][l] = ||v_emotion[e][l]||_2
```

**Plot:** Line plot, x=layer (0..33), one line per emotion.  
Vertical dashed line at layer 22 (primary layer).  
**What to look for:** Do norms peak near layer 22? Do they rise through middle layers then fall?

---

### 2. Pairwise cosine similarity

**Primary:** Compute the 4×4 cosine similarity matrix at layer 22.  
**Cross-layer:** Compute all 6 unique pair similarities across all layers.

```
cos_sim[e1][e2][l] = (v[e1][l] · v[e2][l]) / (||v[e1][l]|| · ||v[e2][l]||)
```

6 pairs: joy–sadness, joy–admiration, joy–loathing, sadness–admiration,
sadness–loathing, admiration–loathing.

**Plot A:** 4×4 heatmap at layer 22 (primary figure).  
**Plot B:** Line plot, x=layer, one line per pair across all 34 layers.  
**What to look for:** Do (joy, admiration) have high mutual similarity and low similarity to (sadness, loathing)? Is valence the dominant separation axis?

---

### 3. Hierarchical clustering at layer 22

Agglomerative clustering with cosine distance on the 4 emotion vectors at layer 22.  
The dendrogram shows which pairs are most similar (merge first) and most orthogonal (merge last).  
Tests valence hypothesis: expect (joy, admiration) and (sadness, loathing) as the two branches.

---

## Output files

```
figures/geometry/
├── norms_by_layer.png
├── cosine_sim_layer22_heatmap.png     # primary figure
├── cosine_sim_by_layer.png            # 6-pair line plot
└── dendrogram_layer22.png
results/geometry/
└── geometry_stats.json                # all numeric outputs
```

**UMAP deferred** — with only 4 emotion vectors, UMAP is not meaningful. Will be
added once more emotion vectors are extracted (target: ~10+ concepts for k-means k=5–10).

---

## Script

`scripts/analyze_emotion_geometry.py`

```
1. load_vectors(hf_repo, hf_prefix) → {emotion: {layer: np.ndarray}}
2. compute_norms(vectors) → DataFrame[emotion, layer, norm]
3. compute_cosine_similarities(vectors) → DataFrame[e1, e2, layer, cos_sim]
4. hierarchical_clustering(vectors, layer=22) → linkage matrix
5. plot_*() → figures/geometry/
6. save_stats() → results/geometry/geometry_stats.json
```

CPU-only. No model needed. ~2 min total.
