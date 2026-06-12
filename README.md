<div align="center">

# 🎬 Recommendation Systems for Personalized Content Discovery

### *Item-Based Collaborative Filtering · Singular Value Decomposition*
### *Netflix Prize Dataset · K=10 Evaluation · 500 Users*

<br />

| Metric | Item-CF | SVD |
|--------|---------|-----|
| **RMSE ↓** | **0.9422** ✅ | 1.0529 |
| **MAE ↓** | **0.7149** ✅ | 0.8852 |
| **MAP@10 ↑** | 0.005689 | **0.013685** ✅ |
| **Precision@10 ↑** | 0.0170 | **0.0374** ✅ |
| **Recall@10 ↑** | 0.006733 | **0.01642** ✅ |
| **NDCG@10 ↑** | 0.017604 | **0.039113** ✅ |

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Module Reference](#-module-reference)
  - [preprocessing.py](#preprocessingpy)
  - [collaborative_filtering.py](#collaborative_filteringpy)
  - [svd_model.py](#svd_modelpy)
  - [evaluation.py](#evaluationpy)
- [Evaluation Methodology](#-evaluation-methodology)
- [Results & Analysis](#-results--analysis)
- [Recommendation Examples](#-recommendation-examples)
- [Future Improvements](#-future-improvements)
- [Citation](#-citation)

---

## 🎯 Overview

This project implements and compares two recommendation system paradigms on the **Netflix Prize Dataset** — one of the most influential benchmarks in the history of machine learning.

> *"Every day, billions of users interact with recommendation systems that influence the movies they watch, the songs they listen to, and the content they consume online."*

The goal is not merely to predict ratings, but to build a recommendation engine that **effectively captures user preferences** and **delivers meaningful, personalized content recommendations**.

**Two models are designed, implemented, and evaluated from scratch:**

| Model | Paradigm | Key Idea |
|-------|----------|----------|
| **Item-CF** | Memory-based Collaborative Filtering | Find items rated similarly; predict via weighted average of user's known ratings |
| **SVD** | Model-based Matrix Factorisation | Learn compressed user & item latent vectors; predict via dot product |

---

## 🔑 Key Findings

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORE FINDING: Rating accuracy and ranking quality are DECOUPLED.   │
│                                                                     │
│  ► Item-CF predicts individual ratings more precisely (RMSE 0.942)  │
│  ► SVD recommends content users actually want (MAP@10 2.4× higher)  │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Adjusted Cosine beats Raw Cosine** — Mean-centering user ratings removes individual rating bias and significantly improves RMSE accuracy in Item-CF.

2. **Latent factors capture global taste structure** — SVD's 100-dimensional embeddings model implicit genre preferences, actor affinities, and cross-genre taste clusters that local item similarity misses.

3. **SVD is 38× faster at evaluation** — RMSE eval: 8.8s vs 339.9s. At production scale, SVD's O(n_factors) dot-product prediction vastly outperforms Item-CF's neighbour intersection.

4. **Sparsity is the dominant challenge** — Both models achieve MAP@10 < 0.02, reflecting the inherent difficulty: each user has rated < 0.05% of the 17,770-movie corpus.

---

## 📊 Dataset

**Netflix Prize Dataset** — [[Kaggle Link]](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)

| Property | Value |
|----------|-------|
| Total Ratings | 100,480,507 |
| Unique Users | 480,189 |
| Unique Movies | 17,770 |
| Rating Scale | 1–5 (integer) |
| Includes | Timestamps, Movie metadata |
| Sparsity | > 99.95% |

**Raw File Format:**
```
<movie_id>:
<user_id>,<rating>,<date>
<user_id>,<rating>,<date>
...
```

**Expected files in `data/`:**
```
data/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
└── movie_titles.csv
```

> ⚠️ Due to memory constraints, the pipeline defaults to loading `combined_data_1.txt` only (`use_subset=True`). Set `use_subset=False` to train on all four files.

---

## 📁 Project Structure

```
netflix-recsys/
│
├── data/
│   ├── combined_data_1.txt          ← Raw Netflix rating files
│   ├── combined_data_2.txt
│   ├── combined_data_3.txt
│   ├── combined_data_4.txt
│   ├── movie_titles.csv             ← Movie metadata
│   └── processed/                   ← Auto-generated cache files
│       ├── ratings_clean.parquet
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       ├── interaction_matrix.npz
│       ├── train_matrix.npz
│       └── item_cf_similarity.pkl   ← Item-CF similarity cache
│
├── results/
│   ├── models/
│   │   └── svd_model.pkl            ← Serialised SVD model
│   └── evaluation_results.json      ← Full metric results
│
├── src/
│   ├── preprocessing.py             ← Data loading & preprocessing pipeline
│   ├── collaborative_filtering.py   ← Item-Based CF with Adjusted Cosine
│   ├── svd_model.py                 ← SVD matrix factorisation (Surprise)
│   └── evaluation.py                ← Unified evaluation framework
│
├── notebooks/                       ← EDA and analysis notebooks (optional)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- 8 GB RAM minimum (16 GB recommended for full dataset)
- ~5 GB disk space for data + artefacts

### 1. Clone the repository

```bash
git clone https://github.com/your-username/netflix-recsys.git
cd netflix-recsys
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
scikit-surprise>=1.1.3
tqdm>=4.65
pyarrow>=12.0          # Parquet support
fastparquet>=2023.0    # Optional, faster Parquet
```

### 4. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and place all files in the `data/` directory.

---

## 🚀 Quick Start

### Step 1 — Preprocess the data

```python
from src.preprocessing import NetflixDataLoader, build_interaction_matrix, split_data, save_splits

# Load and clean ratings (uses cache after first run)
loader = NetflixDataLoader(use_subset=True)  # set False for full dataset
ratings_df = loader.load(cache=True)
movies_df  = loader.load_movies()

# Build sparse interaction matrix
train_df, val_df, test_df = split_data(ratings_df)
save_splits(train_df, val_df, test_df)

train_matrix = build_interaction_matrix(train_df, loader.user2idx, loader.movie2idx)
```

### Step 2 — Train Item-CF

```python
from src.collaborative_filtering import ItemBasedCF

model_cf = ItemBasedCF(top_k_similar=50, use_adjusted=True, batch_size=500)
model_cf.fit(train_matrix)  # caches similarity matrix to disk automatically

# Get Top-10 recommendations
recs = model_cf.recommend(user_idx=0, n=10, exclude_seen=True)

# Get explainable recommendation
explanation = model_cf.explain(
    user_idx=0,
    target_movie_idx=recs[0][0],
    idx2movie=loader.idx2movie,
    movies_df=movies_df
)
print(explanation)
# → "Because you liked "The Shawshank Redemption" (you rated it 5/5) and 
#    "Forrest Gump" (you rated it 4/5), we recommend "The Green Mile" 
#    (similarity score: 0.847)."
```

### Step 3 — Train SVD

```python
from src.svd_model import SVDModel

model_svd = SVDModel(params={
    "n_factors": 100,
    "n_epochs": 25,
    "lr_all": 0.005,
    "reg_all": 0.1
})
model_svd.fit(train_df)

# Predict a single rating
score = model_svd.predict(user_id=12345, movie_id=6789)

# Get Top-10 recommendations
all_movies = train_df["movie_id"].unique().tolist()
seen_movies = set(train_df[train_df["user_id"] == 12345]["movie_id"])
recs = model_svd.recommend(user_id=12345, n=10, all_movie_ids=all_movies, seen_movie_ids=seen_movies)

# Save model to disk
model_svd.save()
```

### Step 4 — Evaluate both models

```python
from src.evaluation import Evaluator

evaluator = Evaluator(k=10, relevance_threshold=3.5, n_eval_users=500)

# Evaluate Item-CF
cf_results = evaluator.evaluate_model(
    model=model_cf, model_type="item_cf",
    test_df=test_df, train_df=train_df,
    model_name="Item-CF (Adjusted Cosine)",
    idx2movie=loader.idx2movie,
    movie2idx=loader.movie2idx,
    user2idx=loader.user2idx
)

# Evaluate SVD
svd_results = evaluator.evaluate_model(
    model=model_svd, model_type="svd",
    test_df=test_df, train_df=train_df,
    model_name="SVD (Surprise)"
)

# Print comparison table
print(evaluator.comparison_table().to_string(index=False))

# Save results to JSON
evaluator.save_results()
```

---

## 📚 Module Reference

### `preprocessing.py`

The data loading and preprocessing pipeline. Handles parsing, filtering, subsampling, index mapping, and train/val/test splitting.

#### `NetflixDataLoader`

```python
loader = NetflixDataLoader(
    data_dir="data/",
    max_users=50_000,          # cap on unique users (memory safety)
    max_movies=5_000,          # cap on unique movies
    min_user_ratings=20,       # activity filter
    min_movie_ratings=50,      # popularity filter
    use_subset=True,           # load only combined_data_1.txt
)
```

| Method | Description |
|--------|-------------|
| `load(cache=True)` | Parse raw files → clean DataFrame; Parquet cache on first run |
| `load_movies()` | Load `movie_titles.csv` → `[movie_id, year, title]` |

**Index Maps exposed after `load()`:**

```python
loader.user2idx    # { raw_user_id  → 0-based index }
loader.movie2idx   # { raw_movie_id → 0-based index }
loader.idx2user    # reverse of user2idx
loader.idx2movie   # reverse of movie2idx
```

#### `split_data()`

Per-user **temporal split** to prevent leakage:

```
For each user → sort by date → last 20% = test, next 10% = val, rest = train
```

```python
train_df, val_df, test_df = split_data(
    ratings_df,
    test_size=0.20,
    val_size=0.10,
    stratify_users=True    # use temporal split (recommended)
)
```

#### `build_interaction_matrix()`

Builds a sparse CSR matrix of shape `(n_users, n_movies)`:

```python
matrix = build_interaction_matrix(ratings_df, user2idx, movie2idx)
# → csr_matrix, float32, shape (n_users, n_movies)
```

---

### `collaborative_filtering.py`

Item-Based Collaborative Filtering with Adjusted Cosine Similarity.

#### How it works

```
1. Compute per-user mean ratings (excluding unrated movies)
2. Subtract user mean from each rating → adjusted centred matrix
3. Compute item-item cosine similarity in batches (batch_size=500)
4. Retain top-K=50 neighbours per item (prune for memory + speed)
5. Predict: similarity-weighted average of user's neighbour ratings
6. Recommend: score all unseen movies → return top-N
```

#### `ItemBasedCF`

```python
model = ItemBasedCF(
    top_k_similar=50,     # neighbours retained per item
    min_common=5,         # min shared raters for valid similarity
    use_adjusted=True,    # adjusted vs raw cosine
    batch_size=500        # items per similarity batch
)
model.fit(train_matrix, cache_path="data/processed/item_cf_similarity.pkl")
```

| Method | Description |
|--------|-------------|
| `fit(train_matrix)` | Compute and cache item-item similarity |
| `predict(user_idx, movie_idx)` | Predict single rating |
| `predict_batch(user_indices, movie_indices)` | Batch rating prediction |
| `recommend(user_idx, n=10)` | Top-N recommendations for one user |
| `recommend_batch(user_indices, n=10)` | Top-N for multiple users |
| `explain(user_idx, movie_idx, ...)` | Human-readable recommendation explanation |
| `save(path)` / `load(path)` | Model serialisation |

#### Explainability

```python
explanation = model.explain(
    user_idx=42,
    target_movie_idx=17,
    idx2movie=loader.idx2movie,
    movies_df=movies_df,
    top_k_reasons=3
)
# → "Because you liked "Inception" (you rated it 5/5) and 
#    "The Matrix" (you rated it 5/5), we recommend 
#    "Interstellar" (similarity score: 0.912)."
```

#### `get_similar_movies()`

```python
from src.collaborative_filtering import get_similar_movies

similar = get_similar_movies(model, movie_idx=0, idx2movie=loader.idx2movie, movies_df=movies_df, n=5)
# Returns DataFrame: [rank, movie_idx, movie_id, title, similarity]
```

---

### `svd_model.py`

Matrix Factorisation via [scikit-surprise](https://surpriselib.com/) SVD.

#### Objective Function

```
min  Σ (r_ui − μ − b_u − b_i − p_u · q_i)²  +  λ(||p_u||² + ||q_i||²)

where:
  μ   = global mean rating
  b_u = user bias term
  b_i = item bias term
  p_u = user latent factor vector  (n_factors,)
  q_i = item latent factor vector  (n_factors,)
  λ   = L2 regularisation weight
```

#### Default Hyperparameters

```python
DEFAULT_PARAMS = {
    "n_factors":    100,    # latent dimension
    "n_epochs":     25,     # SGD passes
    "lr_all":       0.005,  # learning rate
    "reg_all":      0.1,    # L2 regularisation
    "biased":       True,   # include bias terms
    "random_state": 42,
}
```

#### `SVDModel`

```python
model = SVDModel(params={"n_factors": 100, "n_epochs": 25})
model.fit(train_df)
```

| Method | Description |
|--------|-------------|
| `fit(train_df)` | Build Surprise trainset and fit SVD |
| `tune(train_df, param_grid, cv=3)` | GridSearchCV hyperparameter tuning |
| `predict(user_id, movie_id)` | Single rating prediction |
| `predict_batch(user_ids, movie_ids)` | Vectorised batch prediction |
| `predict_df(ratings_df)` | Add predictions column to DataFrame |
| `recommend(user_id, n=10, ...)` | Top-N recommendations |
| `recommend_batch(user_ids, n=10)` | Top-N for multiple users |
| `cross_validate(ratings_df, cv=5)` | K-fold cross-validation |
| `get_user_factors(user_id)` | Latent vector for user |
| `get_item_factors(movie_id)` | Latent vector for movie |
| `get_user_bias(user_id)` | Learned user bias |
| `get_item_bias(movie_id)` | Learned item bias |
| `summary()` | Model config and training summary dict |
| `save(path)` / `load(path)` | Pickle serialisation |

#### Optional Hyperparameter Tuning

```python
best_params = model.tune(
    train_df,
    param_grid={
        "n_factors": [50, 100, 150],
        "n_epochs":  [20, 30],
        "lr_all":    [0.003, 0.005, 0.007],
        "reg_all":   [0.05, 0.1, 0.2],
    },
    cv=3,
    time_limit_minutes=30
)
```

---

### `evaluation.py`

Model-agnostic evaluation framework supporting both Item-CF and SVD.

#### Metrics

| Metric | Type | Definition |
|--------|------|------------|
| **RMSE** | Accuracy | `sqrt(mean((actual - pred)²))` |
| **MAE** | Accuracy | `mean(\|actual - pred\|)` |
| **MAP@K** | Ranking | Mean Average Precision @ K |
| **Precision@K** | Ranking | Fraction of top-K recs that are relevant |
| **Recall@K** | Ranking | Fraction of all relevant items in top-K |
| **NDCG@K** | Ranking | Position-discounted cumulative gain, normalised |

> **Relevance definition:** A movie is relevant if its actual user rating ≥ **3.5 stars**

#### `Evaluator`

```python
evaluator = Evaluator(
    k=10,                      # ranking cutoff
    relevance_threshold=3.5,   # min rating = relevant
    n_eval_users=500,          # users to subsample for ranking eval
    random_state=42
)

results = evaluator.evaluate_model(
    model, model_type,         # "item_cf" or "svd"
    test_df, train_df,
    model_name="My Model"
)

evaluator.print_report(results)
table = evaluator.comparison_table()  # pd.DataFrame of all models
evaluator.save_results()              # → results/evaluation_results.json
```

#### Standalone utility functions

```python
from src.evaluation import (
    compute_rmse,           # single RMSE computation
    compute_mae,            # single MAE computation
    compute_map_at_k,       # MAP@K from pre-built recommendation dict
    compute_ranking_metrics,# all 4 ranking metrics in one pass
    per_user_metrics,       # per-user metric breakdown → DataFrame
    rating_bias_analysis,   # RMSE / bias breakdown by rating bucket
    evaluate_rmse_only,     # fast RMSE+MAE without generating recs
)
```

---

## 📐 Evaluation Methodology

### Train / Validation / Test Split

```
Per-user temporal sort → last 20% = test | next 10% = val | rest = train
Users with < 5 ratings → placed entirely in train
```

This design **prevents data leakage** by ensuring the model learns from the past and is evaluated on genuinely unseen future interactions.

### Ranking Evaluation Pipeline

```
Step 1: Identify users with ≥ 1 relevant movie in test set (rating ≥ 3.5)
Step 2: Subsample 500 users for tractable evaluation
Step 3: For each user → score ALL unseen movies → return Top-10
Step 4: Compute MAP@10, Precision@10, Recall@10, NDCG@10
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Relevance threshold | 3.5 stars | Per problem specification; filters genuine preference signal |
| Evaluation users | 500 (sampled) | Balances evaluation quality with compute time |
| Unseen movies only | Yes | Ensures recommendations are actually novel |
| Temporal split | Yes | Mimics real recommendation scenario; prevents leakage |

---

## 📈 Results & Analysis

### Full Metric Table

| Metric | Item-CF (Adj. Cosine) | SVD (Surprise) | Winner |
|--------|----------------------|----------------|--------|
| RMSE ↓ | **0.9422** | 1.0529 | Item-CF |
| MAE ↓ | **0.7149** | 0.8852 | Item-CF |
| MAP@10 ↑ | 0.005689 | **0.013685** | SVD (+2.4×) |
| Precision@10 ↑ | 0.0170 | **0.0374** | SVD (+2.2×) |
| Recall@10 ↑ | 0.006733 | **0.01642** | SVD (+2.4×) |
| NDCG@10 ↑ | 0.017604 | **0.039113** | SVD (+2.2×) |
| Users Evaluated | 500 | 500 | — |
| RMSE Eval Time | 339.85 s | **8.80 s** | SVD (38×) |
| Ranking Eval Time | 266.95 s | **16.72 s** | SVD (16×) |

### The Rating Accuracy vs. Ranking Quality Trade-off

```
RMSE measures how close predictions are to actual ratings — uniformly,
including low ratings. A model that accurately predicts 2-star ratings
for bad movies scores well on RMSE but has no practical recommendation value.

MAP@10 only rewards models that surface genuinely liked content at the top.
SVD's latent space captures the global structure of taste that local
item-item similarity misses — hence 2.4× better ranking quality.
```

### Computational Profile

```
Item-CF:  ~607 s total evaluation  (339.85 RMSE + 266.95 ranking)
SVD:      ~25.5 s total evaluation  (8.80 RMSE  + 16.72 ranking)

For production serving:
  Item-CF  → O(top_k × |user_ratings|) per prediction
  SVD      → O(n_factors) per prediction  ≈ microseconds
```

---

## 🎬 Recommendation Examples

### Item-CF — With Explanation

```
User 42 | Top-3 Recommendations

Rank 1: The Green Mile         → Score: 4.91
  💬 Because you liked "The Shawshank Redemption" (5★) and 
     "Forrest Gump" (4★), we recommend "The Green Mile" 
     (similarity score: 0.847).

Rank 2: Schindler's List       → Score: 4.87
  💬 Because you liked "Saving Private Ryan" (5★) and 
     "The Pianist" (4★), we recommend "Schindler's List"
     (similarity score: 0.821).

Rank 3: Goodfellas             → Score: 4.82
  💬 Because you liked "The Godfather" (5★) and 
     "Casino" (4★), we recommend "Goodfellas"
     (similarity score: 0.794).
```

### SVD — Latent Factor Ranking

```
User 42 | Top-5 Recommendations

Rank 1: Movie #3245  → Predicted Score: 4.82
Rank 2: Movie #1897  → Predicted Score: 4.79
Rank 3: Movie #7612  → Predicted Score: 4.75
Rank 4: Movie #2341  → Predicted Score: 4.72
Rank 5: Movie #5098  → Predicted Score: 4.69

Score = μ + b_user + b_item + p_user · q_item
      = 3.61 + 0.42 + 0.51 + (100-dim dot product)
```

---

## 🚀 Future Improvements

### Short-Term

- [ ] **SVD hyperparameter tuning** via `GridSearchCV` over `n_factors`, `n_epochs`, `lr_all`, `reg_all`
- [ ] **Full dataset training** — expand from `combined_data_1.txt` to all four files
- [ ] **Vectorised Item-CF prediction** — replace Python-level loop with NumPy batch ops; expected 10× speedup
- [ ] **Min-common rater tuning** — experiment with higher `min_common` thresholds

### Medium-Term

- [ ] **Alternating Least Squares (ALS)** — parallelisable, better for implicit feedback; try `implicit` library
- [ ] **Neural Collaborative Filtering (NCF)** — MLP over user/item embeddings; typically 2–5% NDCG gain
- [ ] **Temporal dynamics** — decay older ratings; model user taste drift over time
- [ ] **Hybrid ensemble** — combine Item-CF similarity scores + SVD latent factors as LightGBM features

### Long-Term

- [ ] **Graph Neural Networks** — LightGCN propagates preference signals over user-item bipartite graph
- [ ] **LLM-enhanced recommendations** — encode movie plots/reviews via embedding models
- [ ] **Diversity & serendipity metrics** — avoid filter bubbles; reward novel-yet-relevant recommendations
- [ ] **Cold-start content-based fallback** — genre + release year similarity for new users / new movies

---

## 📄 Citation

If you use this codebase in your research or project, please cite:

```bibtex
@misc{netflix-recsys-2026,
  title   = {Recommendation Systems for Personalized Content Discovery},
  author  = {ML Engineering Division},
  year    = {2026},
  note    = {Netflix Prize Dataset. Item-Based CF and SVD.},
  url     = {https://github.com/your-username/netflix-recsys}
}
```

**Dataset citation:**
```
Netflix Prize Dataset. (2006). Netflix, Inc.
https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
```

---

<div align="center">

**Built with ❤️ on the Netflix Prize Dataset**

`python` · `numpy` · `scipy` · `scikit-surprise` · `pandas` · `tqdm`

<img src="https://img.shields.io/badge/RMSE-0.9422-E50914?style=flat-square" />
<img src="https://img.shields.io/badge/MAP@10-0.01369-46D369?style=flat-square" />
<img src="https://img.shields.io/badge/Users%20Evaluated-500-141414?style=flat-square" />

</div>
